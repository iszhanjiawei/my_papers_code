import fcntl
from pathlib import Path
import tempfile
import unittest

from lrs3_resume_after_lock_release import original_locks_free, validate_guard


class OriginalLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.locks = self.root / 'state/locks'
        self.locks.mkdir(parents=True)
        for name in ('finish', 'video40'):
            (self.locks / f'{name}.lock').touch()
        self.identities = [((self.locks / f'{name}.lock').stat().st_dev,
                            (self.locks / f'{name}.lock').stat().st_ino)
                           for name in ('finish', 'video40')]

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_conflict_is_not_unlocked_and_partial_probe_is_released(self):
        with (self.locks / 'video40.lock').open('r+') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(original_locks_free(self.root, self.identities))
            with (self.locks / 'video40.lock').open('r+') as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (self.locks / 'finish.lock').open('r+') as contender:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(contender, fcntl.LOCK_UN)
            fcntl.flock(owner, fcntl.LOCK_UN)
        self.assertTrue(original_locks_free(self.root, self.identities))

    def test_normal_probe_preserves_inodes(self):
        before = {p.name: p.stat().st_ino for p in self.locks.iterdir()}
        self.assertTrue(original_locks_free(self.root, self.identities))
        self.assertEqual(before, {p.name: p.stat().st_ino for p in self.locks.iterdir()})

    def test_missing_original_lock_is_not_recreated(self):
        (self.locks / 'video40.lock').unlink()
        with self.assertRaises(FileNotFoundError):
            original_locks_free(self.root, self.identities)
        self.assertFalse((self.locks / 'video40.lock').exists())

    def test_replaced_lock_is_rejected_even_before_waiter_starts(self):
        original = self.locks / 'video40.lock'
        original.rename(self.locks / 'archived_probe.lock')
        original.touch()
        with self.assertRaisesRegex(RuntimeError, 'original inode'):
            original_locks_free(self.root, self.identities)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'expected_cache_summary': '/cache/reports/new/summary.json',
                     'source_manifest_sha256': 'source', 'media_quality_terminal_sha256': 'quality',
                     'new_cache_command': ['python', '-u', 'cache.py', '--gpus', 'cpu,cpu']}
        self.summary = {'pid': 123, 'stage': 'video40', 'manifest_sha256': 'source',
                        'media_quality_terminal_sha256': 'quality', 'requested': 10, 'rounds': []}
        self.guard = {'pid': 123, 'run': '/cache/reports/new', 'stage': 'video40'}

    def test_guard_alone_cannot_start_finisher(self):
        self.assertFalse(validate_guard(self.summary, self.guard, self.plan, 123, 10))
        self.summary['rounds'] = [{'workers': [{'rank': 0}, {'rank': 1}]}]
        self.assertTrue(validate_guard(self.summary, self.guard, self.plan, 123, 10))

    def test_duplicate_worker_rank_is_rejected(self):
        self.summary['rounds'] = [{'workers': [{'rank': 0}, {'rank': 0}]}]
        with self.assertRaises(RuntimeError):
            validate_guard(self.summary, self.guard, self.plan, 123, 10)

    def test_wrong_source_or_foreign_guard_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_guard(self.summary, self.guard, self.plan, 123, 11)
        self.guard['pid'] = 456
        with self.assertRaises(RuntimeError):
            validate_guard(self.summary, self.guard, self.plan, 123, 10)


if __name__ == '__main__':
    unittest.main()
