"""Real-process launcher lock/shutdown tests; no models, GPUs or network calls."""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def wait_until(predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise TimeoutError('Synthetic lifecycle fixture did not reach its expected state')


def lock_descriptors():
    found = []
    for fd in Path('/proc/self/fd').glob('*'):
        try:
            target = str(fd.readlink())
            if target.endswith('.lock'):
                found.append(target)
        except FileNotFoundError:
            pass
    return found


def process_identity():
    return {'pid': os.getpid(), 'ppid': os.getppid(), 'sid': os.getsid(0),
            'lock_fds': lock_descriptors(), 'time_unix': time.time()}


def synthetic_process(mode, directory, argument=None):
    """The child fixture models elastic ranks in separate sessions."""
    directory = Path(directory)
    stopped = []
    if mode != '_worker':
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda number, frame: stopped.append(number))
    if mode == '_tensorboard':
        dump(directory / 'tensorboard_ready.json', process_identity())
        while not stopped:
            time.sleep(.05)
        return 0
    if mode == '_rank':
        dump(directory / f'rank{argument}_ready.json', process_identity())
        while not stopped and not (directory / 'finish_ranks').exists():
            time.sleep(.05)
        dump(directory / f'rank{argument}_exit.json', process_identity())
        return 0
    if mode == '_torchrun':
        dump(directory / 'torchrun_identity.json', process_identity())
        children = []
        try:
            for rank in range(4):
                children.append(subprocess.Popen(
                    [sys.executable, '-u', str(HERE), '_rank', str(directory), str(rank)],
                    start_new_session=True, close_fds=True))
            wait_until(lambda: all((directory / f'rank{i}_ready.json').exists() for i in range(4)))
            dump(directory / 'training_ready.json', {'pid': os.getpid(), 'ranks': [p.pid for p in children]})
            while not stopped and not (directory / 'complete_training').exists():
                time.sleep(.05)
        finally:
            # Elastic's ordinary nonzero and signal paths stop/reap their ranks.
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait(timeout=10)
            dump(directory / 'torchrun_exit.json', {'time_unix': time.time(),
                 'rank_returncodes': [p.returncode for p in children]})
        return 7 if argument == 'nonzero' else 0
    if mode == '_worker':
        # Lifecycle tests never instantiate a dataset or hash a checkpoint.
        # Avoid importing the neural/audio stack in every synthetic owner.
        dataset = types.ModuleType('aligndit.model.lrs3_semantic_vae_dataset')
        dataset.SemanticVaeLRS3Dataset = object
        dataset._iter_jsonl = lambda path: iter(())
        semantic = types.ModuleType('aligndit.model.semantic_vae_dataset')
        semantic.sha256_file = lambda path: 'unused-in-lifecycle-fixture'
        sys.modules[dataset.__name__] = dataset
        sys.modules[semantic.__name__] = semantic
        import launch_lrs3_training as launch
        plan_path = directory / 'plan.json'
        plan = json.loads(plan_path.read_text())
        launch.verify_pins = lambda artifacts: None
        launch.validate_devices = lambda devices, port: None

        def tensorboard(output, event_dir, port):
            with (directory / 'fake_tensorboard.log').open('ab') as log:
                process = subprocess.Popen(
                    [sys.executable, '-u', str(HERE), '_tensorboard', str(directory)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            dump(directory / 'tensorboard_process.json', {'pid': process.pid})
            wait_until(lambda: (directory / 'tensorboard_ready.json').exists())
            return {'pid': process.pid, 'local_url': 'http://synthetic-no-network',
                    'port': port, 'logdir': str(event_dir)}

        class Response:
            def json(self):
                return {}

        class Session:
            trust_env = False

            def get(self, *args, **kwargs):
                if plan['fixture_mode'] == 'monitor_error':
                    wait_until(lambda: (directory / 'training_ready.json').exists())
                    raise RuntimeError('Synthetic monitor failure after ranks started')
                return Response()

        launch.ensure_tensorboard = tensorboard
        launch.requests.Session = Session
        return launch.worker(plan_path)
    raise ValueError(mode)


class TrainingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Adopt fixture TensorBoard after its launcher exits so tests can reap it.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER, Linux only.
            raise OSError(ctypes.get_errno(), 'Cannot adopt synthetic fixture processes')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='lrs3_launcher_lifecycle_')
        self.directory = Path(self.temp.name)
        self.root = self.directory / 'cache'
        self.output = self.directory / 'output'
        self.output.mkdir()
        self.root.mkdir()
        self.owner = None
        self.log = (self.directory / 'owner.log').open('w+')

    def tearDown(self):
        if self.owner is not None and self.owner.poll() is None:
            self.owner.terminate()
            self.owner.wait(timeout=40)
        tb = self.directory / 'tensorboard_process.json'
        if tb.exists():
            pid = json.loads(tb.read_text())['pid']
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        self.log.close()
        self.temp.cleanup()

    def paths(self):
        return [self.root / f'state/locks/{name}.lock'
                for name in ('audio', 'video40', 'media', 'publication')] + [self.output / 'launch.lock']

    def assert_exclusive(self, path, available):
        with path.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.assertFalse(available, str(path))
            else:
                try:
                    self.assertTrue(available, str(path))
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def launch(self, mode):
        command = [sys.executable, '-u', str(HERE), '_torchrun', str(self.directory), mode]
        if mode == 'spawn_failure':
            command = [str(self.directory / 'missing_executable')]
        plan = {'output_dir': str(self.output), 'scope': {'synthetic': True},
                'cache_root': str(self.root), 'artifacts': [], 'gpus': '0,1,2,3',
                'master_port': 1, 'tensorboard_port': 2, 'event_dir': str(self.directory / 'events'),
                'command': command, 'fixture_mode': mode}
        dump(self.directory / 'plan.json', plan)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'PYTHONPATH': str(PROJECT / 'src') + os.pathsep + str(PROJECT / 'scripts')}
        self.owner = subprocess.Popen(
            [sys.executable, '-u', str(HERE), '_worker', str(self.directory)],
            cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL, stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)

    def finish(self, expected_code):
        code = self.owner.wait(timeout=50)
        self.log.flush()
        self.log.seek(0)
        self.assertEqual(code, expected_code, self.log.read())
        state = json.loads((self.output / 'training_process.json').read_text())
        self.assertTrue(state['cleanup']['all_registered_descendants_exited'])
        self.assertTrue(state['locks_release_after_descendants'])
        for pid in state['cleanup']['registered_pids']:
            self.assertFalse(Path(f'/proc/{pid}').exists(), f'Training process {pid} remains')
        for path in self.paths():
            self.assert_exclusive(path, True)
        tb = json.loads((self.directory / 'tensorboard_ready.json').read_text())
        self.assertTrue(Path(f'/proc/{tb["pid"]}').exists(), 'Supervisor stopped separate TensorBoard')
        self.assertNotIn(tb['pid'], state['cleanup']['registered_pids'])
        self.assertEqual(tb['lock_fds'], [])
        if (self.directory / 'torchrun_identity.json').exists():
            parent = json.loads((self.directory / 'torchrun_identity.json').read_text())
            self.assertEqual(parent['lock_fds'], [], 'Torchrun inherited a cache/launch lock')
            self.assertEqual(parent['pid'], parent['sid'])
            for rank in range(4):
                data = json.loads((self.directory / f'rank{rank}_ready.json').read_text())
                self.assertEqual(data['lock_fds'], [])
                self.assertEqual(data['pid'], data['sid'])
                self.assertNotEqual(data['sid'], parent['sid'])
                ended = json.loads((self.directory / f'rank{rank}_exit.json').read_text())
                self.assertLessEqual(ended['time_unix'], state['finished_unix'])
            ended = json.loads((self.directory / 'torchrun_exit.json').read_text())
            self.assertLessEqual(ended['time_unix'], state['finished_unix'])
        return state

    def ready(self):
        wait_until(lambda: (self.directory / 'training_ready.json').exists())
        for path in self.paths():
            self.assert_exclusive(path, False)
        # Let the owner record rank birth identities while all ranks remain live.
        time.sleep(.6)
        return {str(path): (path.stat().st_dev, path.stat().st_ino) for path in self.paths()}

    def run_completion(self, mode, code):
        self.launch(mode)
        identities = self.ready()
        (self.directory / 'complete_training').touch()
        state = self.finish(code)
        self.assertEqual(state['status'], 'completed' if code == 0 else 'failed')
        self.assertEqual(state['returncode'], code)
        self.assertEqual(len(state['cleanup']['registered_pids']), 5)
        self.assertEqual(identities, {str(p): (p.stat().st_dev, p.stat().st_ino) for p in self.paths()})

    def test_normal_completion_reaps_before_owner_unlock_and_keeps_tensorboard(self):
        self.run_completion('normal', 0)

    def test_elastic_nonzero_reaps_ranks_before_owner_unlock(self):
        self.run_completion('nonzero', 7)

    def run_signal(self, number):
        self.launch('wait')
        identities = self.ready()
        self.owner.send_signal(number)
        state = self.finish(1)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['shutdown_signal'], number)
        self.assertEqual(len(state['cleanup']['registered_pids']), 5)
        self.assertEqual(identities, {str(p): (p.stat().st_dev, p.stat().st_ino) for p in self.paths()})

    def test_owner_sigterm_reaps_isolated_ranks_before_unlock(self):
        self.run_signal(signal.SIGTERM)

    def test_owner_sigint_reaps_isolated_ranks_before_unlock(self):
        self.run_signal(signal.SIGINT)

    def test_spawn_failure_releases_locks_without_stopping_tensorboard(self):
        self.launch('spawn_failure')
        state = self.finish(1)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('FileNotFoundError', state['error'])
        self.assertEqual(state['cleanup']['registered_pids'], [])

    def test_monitor_exception_cleans_started_process_tree(self):
        self.launch('monitor_error')
        state = self.finish(1)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('Synthetic monitor failure', state['error'])
        self.assertEqual(len(state['cleanup']['registered_pids']), 5)

    def test_read_audit_guard_failure_explicitly_unlocks_each_owned_fd(self):
        import launch_lrs3_training as launch
        guard = self.root / 'state/video_40hz/WRITE_ACTIVE.json'
        guard.parent.mkdir(parents=True)
        guard.write_text('{}')
        original_flock, unlocks = fcntl.flock, []

        def checked(fd, operation):
            if operation == fcntl.LOCK_UN:
                number = fd.fileno() if hasattr(fd, 'fileno') else fd
                self.assertGreaterEqual(os.fstat(number).st_ino, 1)
                unlocks.append(str(Path(f'/proc/self/fd/{number}').readlink()))
            return original_flock(fd, operation)

        with patch.object(launch.fcntl, 'flock', checked):
            with self.assertRaisesRegex(RuntimeError, 'Unfinished cache writer'):
                with ExitStack() as stack:
                    launch.acquire_read_locks(self.root, stack)
        self.assertEqual(set(unlocks), {str(path) for path in self.paths()[:-1]})
        self.assertEqual(len(unlocks), 4)
        for path in self.paths()[:-1]:
            self.assert_exclusive(path, True)

    def test_failed_lock_acquisition_does_not_unlock_another_owner(self):
        import launch_lrs3_training as launch
        path = self.root / 'negative_control.lock'
        with launch.owned_lock(path, fcntl.LOCK_EX):
            with self.assertRaises(BlockingIOError):
                with launch.owned_lock(path, fcntl.LOCK_SH):
                    self.fail('A conflicting reader acquired the owner lock')
            self.assert_exclusive(path, False)
        self.assert_exclusive(path, True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].startswith('_'):
        raise SystemExit(synthetic_process(*sys.argv[1:]))
    unittest.main()
