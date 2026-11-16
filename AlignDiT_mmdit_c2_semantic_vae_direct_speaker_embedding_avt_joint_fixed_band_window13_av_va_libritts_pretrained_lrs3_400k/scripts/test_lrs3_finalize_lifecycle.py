"""Real spawn-worker lifecycle through finalizer-owned publication locks.

Only this test driver replaces validation jobs/preflight with synthetic work;
the production helper and finalizer contain no test bypass or slow-job option.
"""
from contextlib import closing, contextmanager
import errno
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


STAGES = ('audio', 'video40', 'media', 'publication')


@contextmanager
def fixture_directory():
    path = tempfile.mkdtemp(prefix='lrs3-finalizer-lifecycle-')
    try:
        yield path
    finally:
        # DPC can briefly report ENOTEMPTY after the last entry was removed.
        # Worker/lock assertions run before cleanup; never suppress other errors.
        for attempt in range(20):
            try:
                shutil.rmtree(path)
                break
            except FileNotFoundError:
                break
            except OSError as error:
                if error.errno != errno.ENOTEMPTY or attempt == 19:
                    raise
                time.sleep(.1)


def runtime_group(pgid):
    live = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in {'Z', 'X'}:
            live.append(int(path.parent.name))
    return live


def synthetic_batch(batch):
    import lrs3_finalize_parallel as parallel
    row = batch[0][0]
    root = Path(row['root'])
    inherited = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if target.endswith('.lock'):
            inherited.append(target)
    receipt = {'pid': os.getpid(), 'lock_fds': inherited}
    (root / f'worker.{os.getpid()}.ready.json').write_text(json.dumps(receipt))
    if inherited:
        raise AssertionError(f'Inherited lock descriptors: {inherited}')
    while not (root / 'release').exists():
        if parallel._PROCESS_STOP.is_set():
            raise RuntimeError('Synthetic job cooperatively cancelled')
        time.sleep(.02)
    if row['scenario'] == 'worker_failure' and row['sample_id'] == '0':
        raise ValueError('Synthetic worker failure')
    return [(r, []) for r, _label, _exclusion, _receipt in batch]


def synthetic_finalize(args, prefix, control):
    import lrs3_finalize_parallel as parallel
    root = args.root
    scenario = os.environ['LRS3_FINALIZER_TEST_SCENARIO']
    rows = [{'sample_id': str(i), 'root': str(root), 'scenario': scenario} for i in range(24)]
    options = dict(root=root, vocabulary={}, recovered={}, strict_provenance=False,
                   text_policy='source_grounded', media_quality_exclusions={},
                   imported_source=None, import_receipts=None, executor='process',
                   workers=2, batch_size=2, control=control)
    with closing(parallel.ordered_validate(rows, **options)) as results:
        for _row, _reasons in results:
            if scenario == 'aggregation_failure':
                raise ValueError('Synthetic aggregation failure')
    control.check()
    marker = root / 'manifests/test/cache_completion.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{"status":"complete"}')


def synthetic_owner(root, scenario):
    import lrs3_finalize_cache as finalizer
    import lrs3_finalize_parallel as parallel
    original_flock = fcntl.flock

    def observed_flock(handle, operation):
        if operation == fcntl.LOCK_UN:
            # These are actual multiprocessing workers, excluding Python's
            # separate resource tracker, which never receives data/cache FDs.
            observation = {'active_workers': [p.pid for p in multiprocessing.active_children()],
                           'lock': os.readlink(f'/proc/self/fd/{handle.fileno()}')}
            with (root / 'unlock.jsonl').open('a') as log:
                log.write(json.dumps(observation) + '\n')
        return original_flock(handle, operation)

    argv = ['finalizer', '--root', str(root), '--source-manifest', str(root / 'unused.jsonl'),
            '--name', 'test', '--mode', 'full', '--executor', 'process', '--workers', '2',
            '--batch-size', '2']
    with patch.object(sys, 'argv', argv), patch.object(finalizer, 'finalize', synthetic_finalize), \
            patch.object(parallel, '_process_batch', synthetic_batch), \
            patch.object(finalizer.fcntl, 'flock', observed_flock):
        if scenario == 'spawn_failure':
            with patch.object(parallel, 'ProcessPoolExecutor', side_effect=OSError('Synthetic spawn failure')):
                finalizer.main()
        elif scenario == 'partial_submit_failure':
            original_submit = parallel.ProcessPoolExecutor.submit
            calls = 0
            def fail_second_submit(pool, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('Synthetic submit failure after worker spawn')
                return original_submit(pool, *args, **kwargs)
            with patch.object(parallel.ProcessPoolExecutor, 'submit', fail_second_submit):
                finalizer.main()
        else:
            finalizer.main()


class FinalizerLifecycleTests(unittest.TestCase):
    def run_owner(self, scenario, signum=None, group=False):
        with fixture_directory() as temporary:
            root = Path(temporary)
            environment = dict(os.environ, LRS3_FINALIZER_TEST_SCENARIO=scenario,
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            script_dir = Path(__file__).resolve().parent
            environment['PYTHONPATH'] = os.pathsep.join((str(script_dir), str(script_dir.parent / 'src')))
            command = [sys.executable, '-c',
                       'from pathlib import Path; import sys; '
                       'from test_lrs3_finalize_lifecycle import synthetic_owner; '
                       'synthetic_owner(Path(sys.argv[1]),sys.argv[2])', str(root), scenario]
            with (root / 'owner.log').open('w') as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           close_fds=True, start_new_session=True)
                try:
                    if scenario not in {'spawn_failure', 'partial_submit_failure'}:
                        deadline = time.monotonic() + 120
                        while len(list(root.glob('worker.*.ready.json'))) < 2:
                            if process.poll() is not None or time.monotonic() >= deadline:
                                self.fail('Workers did not become ready: ' + (root / 'owner.log').read_text())
                            time.sleep(.05)
                        # During active jobs all four lock descriptions belong
                        # to the owner, and separate opens must be blocked.
                        for stage in STAGES:
                            with (root / f'state/locks/{stage}.lock').open('a') as handle:
                                with self.assertRaises(BlockingIOError):
                                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if signum is None:
                            (root / 'release').touch()
                        else:
                            if group:
                                os.killpg(process.pid, signum)
                            else:
                                os.kill(process.pid, signum)
                    returncode = process.wait(timeout=120)
                    # multiprocessing's resource tracker can briefly retain
                    # redirected stderr after its owner exits, especially when
                    # pool construction fails. It owns no validation jobs/locks;
                    # wait for its log FD to close before deleting DPC fixtures.
                    deadline = time.monotonic() + 10
                    while runtime_group(process.pid) and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertEqual(runtime_group(process.pid), [])
                    self.assertEqual(returncode, 0 if scenario == 'normal' else
                                     128 + signum if signum else 1, (root / 'owner.log').read_text())
                    markers = list(root.glob('worker.*.ready.json'))
                    for path in markers:
                        receipt = json.loads(path.read_text())
                        self.assertEqual(receipt['lock_fds'], [])
                        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists(),
                                         f'Worker {receipt["pid"]} was not reaped')
                    unlocks = [json.loads(line) for line in (root / 'unlock.jsonl').read_text().splitlines()]
                    self.assertEqual(len(unlocks), 4)
                    self.assertTrue(all(not item['active_workers'] for item in unlocks), unlocks)
                    for stage in STAGES:
                        with (root / f'state/locks/{stage}.lock').open('a') as handle:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(handle, fcntl.LOCK_UN)
                    self.assertEqual((root / 'manifests/test/cache_completion.json').exists(), scenario == 'normal')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            # Test-only emergency cleanup after failed assertions.
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()

    def test_normal_completion_reaps_before_all_unlocks(self):
        self.run_owner('normal')

    def test_term_cancels_and_reaps_before_all_unlocks(self):
        self.run_owner('term', signal.SIGTERM)

    def test_int_cancels_and_reaps_before_all_unlocks(self):
        self.run_owner('int', signal.SIGINT)

    def test_foreground_group_term_reaps_before_all_unlocks(self):
        self.run_owner('group_term', signal.SIGTERM, group=True)

    def test_foreground_group_int_reaps_before_all_unlocks(self):
        self.run_owner('group_int', signal.SIGINT, group=True)

    def test_worker_failure_reaps_before_all_unlocks(self):
        self.run_owner('worker_failure')

    def test_aggregation_failure_closes_iterator_and_reaps(self):
        self.run_owner('aggregation_failure')

    def test_spawn_failure_releases_all_locks(self):
        self.run_owner('spawn_failure')

    def test_partial_submit_failure_reaps_already_spawned_worker(self):
        self.run_owner('partial_submit_failure')


if __name__ == '__main__':
    unittest.main()
