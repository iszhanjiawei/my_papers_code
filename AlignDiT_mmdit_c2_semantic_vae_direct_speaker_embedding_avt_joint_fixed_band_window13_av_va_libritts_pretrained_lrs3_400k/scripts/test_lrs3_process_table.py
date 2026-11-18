"""Regression checks for disappearing /proc PIDs in the training supervisor."""
from contextlib import contextmanager
import errno
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

import lrs3_cache_parallel as parallel


def stat_row(pid, *, parent=10, group=20, start='12345', state='S'):
    fields = [state, str(parent), str(group)] + ['0'] * 16 + [start]
    return f'{pid} (worker name with ) parentheses) ' + ' '.join(fields)


class ProcessTableTests(unittest.TestCase):
    @contextmanager
    def proc(self, rows, names=None):
        # DirEntry metadata is deliberately inaccessible: enumeration must not
        # inspect an exited process before entering the disappearance handler.
        entries = [SimpleNamespace(name=name,
                   is_dir=lambda: self.fail('Do not probe PID metadata before reading stat'))
                   for name in (names if names is not None else rows)]

        @contextmanager
        def scan(path):
            self.assertEqual(path, '/proc')
            yield iter(entries)

        def read(path, *args, **kwargs):
            self.assertEqual(path.parent.parent, Path('/proc'))
            value = rows[path.parent.name]
            if isinstance(value, BaseException):
                raise value
            return value

        with patch.object(parallel.os, 'scandir', scan), patch.object(Path, 'read_text', read):
            yield

    def test_process_exiting_during_old_glob_probe_does_not_abort_scan(self):
        # Reproduce the actual traceback: ESRCH is raised by the glob iterator,
        # before the old for-loop body can catch it.
        def vanished_glob(*args, **kwargs):
            raise ProcessLookupError(errno.ESRCH, 'PID exited during glob exists()', '/proc/908687/stat')
            yield  # Make the failure occur during iteration, not glob creation.

        with self.proc({'123': stat_row(123)}), patch.object(Path, 'glob', vanished_glob):
            self.assertEqual(parallel.process_table(), {123: (10, 20, '12345', 'S')})

    def test_vanished_stat_skips_only_that_pid_and_retains_later_processes(self):
        for error in [FileNotFoundError(errno.ENOENT, 'gone'),
                      ProcessLookupError(errno.ESRCH, 'gone')]:
            with self.subTest(error=type(error).__name__), self.proc({
                    '101': stat_row(101, state='Z'), '102': error,
                    '103': stat_row(103, parent=101, group=103, start='67890')}):
                self.assertEqual(parallel.process_table(), {
                    101: (10, 20, '12345', 'Z'), 103: (101, 103, '67890', 'S')})

    def test_nonnumeric_proc_entries_are_not_opened(self):
        with self.proc({'123': stat_row(123)}, names=['self', 'thread-self', 'net', '12x', '１２３', '123']):
            self.assertEqual(set(parallel.process_table()), {123})

    def test_recycled_pid_keeps_distinct_birth_identity(self):
        with self.proc({'123': stat_row(123, start='111')}):
            before = parallel.process_table()
        with self.proc({'123': stat_row(123, start='222')}):
            after = parallel.process_table()
        self.assertNotEqual(before[123][2], after[123][2])

    def test_unexpected_io_failure_is_not_silently_hidden(self):
        with self.proc({'123': OSError(errno.EIO, 'unexpected I/O failure')}):
            with self.assertRaises(OSError) as caught:
                parallel.process_table()
        self.assertEqual(caught.exception.errno, errno.EIO)

    def test_supervised_child_survives_unrelated_pid_disappearance(self):
        supervisor = parallel.ChildSupervisor(isolated_children=True)
        child = supervisor.spawn([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            identity = parallel.process_table()[child.pid]
            rows = {
                '999999999': ProcessLookupError(errno.ESRCH, 'unrelated process exited'),
                str(child.pid): stat_row(child.pid, parent=identity[0], group=identity[1],
                                        start=identity[2], state=identity[3]),
            }
            with self.proc(rows):
                for _ in range(100):
                    self.assertIn(child.pid, supervisor.live())
                    self.assertIsNone(child.poll())
            self.assertEqual(supervisor.known[child.pid], identity[2])
        finally:
            receipt = supervisor.stop_and_reap()
        self.assertTrue(receipt['all_registered_descendants_exited'])
        self.assertIsNotNone(child.returncode)

    def test_real_proc_scan_survives_process_churn(self):
        # Exercise real getdents/stat races in addition to deterministic faults.
        completed = 0
        scans = 0
        for _ in range(8):
            children = [subprocess.Popen([sys.executable, '-c', 'pass']) for _ in range(12)]
            try:
                while any(child.poll() is None for child in children):
                    table = parallel.process_table()
                    self.assertIn(parallel.os.getpid(), table)
                    scans += 1
            finally:
                for child in children:
                    child.wait(timeout=10)
                    completed += 1
        self.assertEqual(completed, 96)
        self.assertGreater(scans, 0)


if __name__ == '__main__':
    unittest.main()
