"""Real subprocess/lock tests; every cache is a synthetic temporary fixture."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from lrs3_cache_parallel import ChildSupervisor, FINISH_FD_ENV, ShutdownRequested, process_table


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, mode, role, rank=0, world=8):
    """Exercise actual entrypoints with tiny workers instead of neural models."""
    canonical = root / 'manifests/source_inventory.jsonl'
    if role == 'worker':
        inherited = int(os.environ[FINISH_FD_ENV]) if FINISH_FD_ENV in os.environ else None
        dump(root / f'worker{rank}.ready', {'pid': os.getpid(),
            'finish_inode': os.fstat(inherited).st_ino if inherited is not None else None})
        if mode != 'success':
            while True:
                time.sleep(.1)
        selected = [json.loads(line) for line in canonical.read_text().splitlines()][rank::world]
        progress = root / f'reports/video40.{os.getpid()}.jsonl'
        progress.write_text(''.join(json.dumps({'sample_id': row['sample_id'], 'status': 'ok'})+'\n'
                                    for row in selected))
        dump(root / f'reports/video40.{os.getpid()}.summary.json',
             {'manifest_sha256': hashlib.sha256(canonical.read_bytes()).hexdigest(),
              'progress_path': str(progress)})
        return
    if role == 'cache':
        import lrs3_cache_parallel as cache
        original = ChildSupervisor.spawn
        def fake_worker(self, command, **kwargs):
            rank = int(command[command.index('--rank')+1])
            if mode == 'spawn_error' and rank == 1:
                raise RuntimeError('Injected spawn failure after a live worker')
            command = [sys.executable, __file__, 'fixture', str(root), mode, 'worker', str(rank), '8']
            return original(self, command, **kwargs)
        ChildSupervisor.spawn = fake_worker
        sys.argv = ['cache', '--root', str(root), '--manifest', str(canonical), '--stage', 'video40',
                    '--gpus', ','.join(['cpu']*8), '--name', 'cache']
        cache.main()
        return
    import lrs3_finish_production as finish
    def one_stage(self):
        self.command('cache', [sys.executable, __file__, 'fixture', str(root), mode, 'cache'],
            lambda: (root/'reports/cache/summary.json').exists()
                    and json.loads((root/'reports/cache/summary.json').read_text()).get('processing_complete') is True)
        self.state.update(status='complete', finished_unix=time.time())
        self.save()
    finish.Finisher.run_all = one_stage
    sys.argv = ['finish', '--root', str(root), '--name', 'finish', '--poll-seconds', '.05']
    finish.main()


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        # Set LRS3_SHUTDOWN_TEST_ROOT only to an isolated scratch directory to
        # run these same real flock tests on the production filesystem type.
        base = os.environ.get('LRS3_SHUTDOWN_TEST_ROOT')
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix='graceful_fixture_', dir=base))
            # Explicit filesystem probes retain their synthetic evidence.
            dump(self.root/'fixture_identity.json', {'test':self.id(), 'production_cache':False})
        else:
            self.temp = tempfile.TemporaryDirectory(prefix='graceful_fixture_')
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
        canonical = self.root/'manifests/source_inventory.jsonl'
        canonical.parent.mkdir(parents=True)
        canonical.write_text(''.join(json.dumps({'sample_id':str(i), 'relative_stem':f'train/pretrain/{i}',
            'original_clip_id':f'pretrain/v/{i}', 'text_status':'original_pipeline'})+'\n' for i in range(8)))
        dump(self.root/'reports/source_contract.json', {'source_inventory_sha256':hashlib.sha256(canonical.read_bytes()).hexdigest(),
            'source_media_pairs':8,'original_clip_count':8})
        self.process = None
        self.addCleanup(self.emergency_cleanup)

    def emergency_cleanup(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def start(self, mode='wait'):
        environment = dict(os.environ)
        scripts = Path(__file__).resolve().parent
        environment['PYTHONPATH'] = os.pathsep.join([str(scripts.parent/'src'), str(scripts)])
        self.log = (self.root/'fixture.log').open('w')
        self.addCleanup(self.log.close)
        self.process = subprocess.Popen([sys.executable, __file__, 'fixture', str(self.root), mode, 'finish'],
            env=environment, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        return self.process

    def wait_ready(self):
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            if len(list(self.root.glob('worker*.ready'))) == 8:
                return
            if self.process.poll() is not None:
                self.fail((self.root/'fixture.log').read_text())
            time.sleep(.05)
        self.fail('Workers did not start')

    def locks(self, free):
        for name in ['finish', 'video40']:
            with (self.root/f'state/locks/{name}.lock').open('a') as handle:
                if free:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle, fcntl.LOCK_UN)
                else:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def check_stopped(self, signum):
        self.start()
        self.wait_ready()
        self.locks(False)
        inode = (self.root/'state/locks/finish.lock').stat().st_ino
        self.assertTrue(all(json.loads(path.read_text())['finish_inode'] == inode
                            for path in self.root.glob('worker*.ready')))
        self.process.send_signal(signum)  # Parent only; it must stop the whole tree.
        self.assertEqual(self.process.wait(timeout=20), 128+signum)
        self.locks(True)
        parent = json.loads((self.root/'reports/finish/state.json').read_text())
        cache = json.loads((self.root/'reports/cache/summary.json').read_text())
        self.assertEqual(parent['status'], 'interrupted')
        self.assertEqual(cache['status'], 'interrupted')
        self.assertFalse(cache['complete'])
        self.assertFalse(cache['processing_complete'])
        self.assertTrue((self.root/'state/video_40hz/WRITE_ACTIVE.json').exists())
        live = process_table()
        for report in [parent, cache]:
            self.assertTrue(report['shutdown']['all_registered_descendants_exited'])
            self.assertTrue(all(pid not in live or live[pid][3] in {'Z','X'}
                                for pid in report['shutdown']['registered_pids']))

    def test_parent_term_reaps_workers_retains_guard_then_unlocks(self):
        self.check_stopped(signal.SIGTERM)

    def test_parent_int_reaps_workers_retains_guard_then_unlocks(self):
        self.check_stopped(signal.SIGINT)

    def test_normal_completion_explicitly_unlocks_both_locks(self):
        self.start('success')
        self.assertEqual(self.process.wait(timeout=20), 0, (self.root/'fixture.log').read_text())
        self.locks(True)
        for path in ['reports/finish/state.json', 'reports/cache/summary.json']:
            self.assertTrue(json.loads((self.root/path).read_text())['shutdown']['all_registered_descendants_exited'])
        self.assertFalse((self.root/'state/video_40hz/WRITE_ACTIVE.json').exists())

    def test_spawn_exception_stops_previously_started_child(self):
        self.start('spawn_error')
        self.assertNotEqual(self.process.wait(timeout=20), 0)
        self.locks(True)
        cache = json.loads((self.root/'reports/cache/summary.json').read_text())
        self.assertEqual(cache['status'], 'failed')
        self.assertFalse(cache['complete'])
        self.assertTrue(cache['shutdown']['all_registered_descendants_exited'])
        self.assertTrue((self.root/'state/video_40hz/WRITE_ACTIVE.json').exists())

    def test_pending_stop_prevents_spawn(self):
        supervisor = ChildSupervisor()
        supervisor.signum = signal.SIGTERM
        with self.assertRaises(ShutdownRequested):
            supervisor.spawn([sys.executable, '-c', 'raise AssertionError("must not start")'])
        self.assertEqual(supervisor.children, [])

    def test_orphan_group_is_verified_and_killed_before_unlock(self):
        supervisor = ChildSupervisor(isolated_children=True, grace_seconds=.1)
        command = ('import subprocess,sys; subprocess.Popen([sys.executable,"-c",'
                   '"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"]);')
        parent = supervisor.spawn([sys.executable, '-c', command])
        parent.wait(timeout=5)
        self.assertTrue(supervisor.live())
        receipt = supervisor.stop_and_reap()
        self.assertTrue(receipt['all_registered_descendants_exited'])
        self.assertFalse(supervisor.live())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fixture':
        fixture(Path(sys.argv[2]), sys.argv[3], sys.argv[4],
                int(sys.argv[5]) if len(sys.argv)>5 else 0, int(sys.argv[6]) if len(sys.argv)>6 else 8)
    else:
        unittest.main()
