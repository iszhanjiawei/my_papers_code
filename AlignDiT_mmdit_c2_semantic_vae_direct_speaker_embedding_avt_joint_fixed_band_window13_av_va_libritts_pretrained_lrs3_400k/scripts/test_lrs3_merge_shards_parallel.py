"""Exercise real isolated merge children, publication, resume and failure cleanup."""
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import lrs3_merge_shards_parallel as parallel
import lrs3_sharded_import as imp
import test_lrs3_imported_source as fixture
from aligndit.lrs3_imported_source import validate_import_completion, validate_imported_source


class ParallelMergeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ImportedSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        imp.STOP.clear()
        with patch('builtins.print'):
            imp.prepare(self.root, self.fixture.origin, 2)
            self.pin = imp.digest(self.root / imp.PLAN)
            for index in range(2):
                imp.worker(self.root, index, self.pin)

    def run_merge(self, success=True):
        command = [sys.executable, '-u', str(Path(parallel.__file__).resolve()),
                   '--root', str(self.root), '--plan-sha256', self.pin, '--workers', '2']
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root / imp.COMPLETION).exists())
        self.check_locks()
        return result

    def check_locks(self):
        for name in ('coordinator.lock', 'merge.lock'):
            with imp.owned_lock(self.root / imp.RUN / name):
                pass
        for index in range(2):
            with imp.owned_lock(imp.shard_paths(self.root, index)[0] / 'worker.lock'):
                pass

    def test_real_children_publish_and_load_exact_valid_rows(self):
        self.run_merge()
        contract = validate_imported_source(self.root)
        result = validate_import_completion(self.root, contract, self.fixture.rows)
        self.assertEqual(set(result['receipts']), {'good', 'bad'})
        state = imp.read_json(self.root / parallel.STATE)
        self.assertEqual(state['status'], 'merged')
        self.assertEqual(len(state['finished']), 2)
        self.assertEqual(state['active'], [])
        for index in range(2):
            directory = imp.shard_paths(self.root, index)[0]
            self.assertFalse(imp.read_json(directory / 'merge_completion.json')['controller_lock_fds_inherited'])
            self.assertFalse((directory / 'MERGE_ACTIVE.json').exists())
        self.fixture.publish()
        data = self.fixture.dataset()
        self.assertEqual(len(data), 1)
        batch = data.collate_fn([data[0]])
        self.assertEqual(tuple(batch['mel'].shape), (1, 64, 40))
        self.assertEqual(tuple(batch['video'].shape), (1, 40, 1024))

    def test_resume_preserves_matching_existing_destination_inode(self):
        row = self.fixture.rows[0]
        directory, data_root, _ = imp.shard_paths(self.root, 0)
        record = imp.read_rows(directory / 'records.jsonl')[0]
        contract = validate_imported_source(self.root)
        imp.merge_one(self.root, data_root, row, record, contract['_sha256'], None)
        path = self.root / row['video_40hz_relative_path']
        before = path.stat().st_ino, imp.digest(path)
        imp.write_json(self.root / 'state/video_40hz/WRITE_ACTIVE.json',
                       {'plan_sha256': self.pin, 'stage': 'import_merge', 'pid': 0})
        self.run_merge()
        self.assertEqual(before, (path.stat().st_ino, imp.digest(path)))

    def test_mismatched_destination_is_never_overwritten_and_children_reaped(self):
        row = self.fixture.rows[0]
        path = self.root / row['video_40hz_relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'corrupt existing destination')
        before = path.read_bytes()
        self.run_merge(success=False)
        self.assertEqual(path.read_bytes(), before)
        state = imp.read_json(self.root / parallel.STATE)
        self.assertEqual(state['status'], 'stopped_or_failed')
        self.assertEqual(state['active'], [])
        for child in state['finished']:
            self.assertFalse(Path('/proc', str(child['pid'])).exists())
        self.assertTrue((self.root / 'state/video_40hz/WRITE_ACTIVE.json').exists())

    def test_modified_shard_manifest_fails_before_standard_layout_writes(self):
        plan = imp.read_json(self.root / imp.PLAN)
        path = self.root / plan['shards'][0]['manifest']
        path.write_bytes(path.read_bytes() + b' ')
        result = self.run_merge(success=False)
        self.assertIn('Pinned shard manifest/proof/exclusions changed', result.stderr)
        self.assertFalse((self.root / self.fixture.rows[0]['audio_relative_path']).exists())

    def test_incompatible_prior_guard_is_retained(self):
        guard = self.root / 'state/video_40hz/WRITE_ACTIVE.json'
        imp.write_json(guard, {'plan_sha256': 'other-plan'})
        before = guard.read_bytes()
        result = self.run_merge(success=False)
        self.assertIn('another plan', result.stderr)
        self.assertEqual(guard.read_bytes(), before)

    def test_unfinished_shard_prevents_any_merge(self):
        directory = imp.shard_paths(self.root, 1)[0]
        imp.write_json(directory / 'WRITE_ACTIVE.json', {'plan_sha256': self.pin})
        result = self.run_merge(success=False)
        self.assertIn('still being written', result.stderr)
        self.assertFalse((self.root / self.fixture.rows[0]['audio_relative_path']).exists())

    def test_sigterm_reaps_every_child_before_releasing_controller_locks(self):
        command = [sys.executable, '-u', str(Path(parallel.__file__).resolve()),
                   '--root', str(self.root), '--plan-sha256', self.pin, '--workers', '2']
        with (self.fixture.base / 'signal.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
            active = []
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and child.poll() is None:
                    state_path = self.root / parallel.STATE
                    if state_path.exists():
                        active = imp.read_json(state_path).get('active', [])
                        if len(active) == 2:
                            break
                    time.sleep(.02)
                self.assertEqual(len(active), 2, 'Could not observe running merge children')
                child.terminate()
                self.assertNotEqual(child.wait(timeout=60), 0)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=60)
        self.check_locks()
        self.assertFalse((self.root / imp.COMPLETION).exists())
        state = imp.read_json(self.root / parallel.STATE)
        self.assertEqual(state['status'], 'stopped_or_failed')
        self.assertEqual(state['active'], [])
        self.assertEqual(len(state['finished']), 2)
        for worker in active:
            self.assertFalse(Path('/proc', str(worker['pid'])).exists())


if __name__ == '__main__':
    unittest.main()
