"""Real imported-cache validation/publication through thread and spawn workers."""
from contextlib import ExitStack
import fcntl
import multiprocessing
import os
import sys
import unittest
from unittest.mock import patch

import lrs3_finalize_cache as finalizer
from lrs3_finalize_parallel import ordered_validate
from aligndit.lrs3_imported_source import validate_import_completion
from aligndit.lrs3_media_quality import excluded_map, sha256
import test_lrs3_imported_source as real_fixture


class ParallelFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.preexisting_children = {child.pid for child in multiprocessing.active_children()}
        self.fixture = real_fixture.ImportedSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.contract = self.fixture.import_all(recompute=True)
        self.completion = validate_import_completion(self.root, self.contract, self.fixture.rows)
        self.options = {
            'root': self.root,
            'vocabulary': {' ': 0, 'a': 1},
            'recovered': {},
            'strict_provenance': True,
            'text_policy': 'source_grounded',
            'media_quality_exclusions': excluded_map(self.contract['terminal']),
            'imported_source': self.contract,
            'import_receipts': self.completion['receipts'],
            'workers': 2,
            'batch_size': 1,
        }

    def validate(self, rows=None, executor='process', **overrides):
        options = {**self.options, **overrides, 'executor': executor}
        return list(ordered_validate(self.fixture.rows if rows is None else rows, **options))

    def publish(self, name='full', executor='process', pseudo=False):
        argv = ['finalizer', '--root', str(self.root), '--source-manifest',
                str(self.root / 'manifests/source_inventory.jsonl'), '--name', name,
                '--mode', 'full', '--workers', '2', '--executor', executor, '--batch-size', '1']
        if pseudo:
            argv += ['--text-policy', 'source_plus_asr_pseudo', '--strict-baseline', 'manifests/full']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'ROOT_PREFIX': str(self.fixture.base)}), \
                patch('builtins.print'):
            finalizer.main()

    def assert_stage_locks_released(self):
        self.assertFalse([child.pid for child in multiprocessing.active_children()
                          if child.pid not in self.preexisting_children])
        # Separate open descriptions must acquire every finalizer-owned lock.
        for stage in ('audio', 'video40', 'media', 'publication'):
            path = self.root / f'state/locks/{stage}.lock'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)

    def test_spawn_and_thread_results_are_exact_and_keep_requested_order(self):
        for rows, batch_size in ((self.fixture.rows, 1), (self.fixture.rows[::-1], 3)):
            with self.subTest(order=[r['sample_id'] for r in rows], batch_size=batch_size):
                expected = self.validate(rows, executor='thread', batch_size=batch_size)
                actual = self.validate(rows, batch_size=batch_size)
                self.assertEqual(actual, expected)
                self.assertEqual([r['sample_id'] for r, reasons in actual],
                                 [r['sample_id'] for r in rows])
                by_id = {row['sample_id']: reasons for row, reasons in actual}
                self.assertEqual(by_id['good'], [])
                self.assertTrue(any(reason.startswith('media_quality:') for reason in by_id['bad']))
                self.assertTrue(all(row.get('import_receipt_sha256') for row, reasons in actual))

    def test_spawn_workers_do_not_inherit_parent_publication_locks(self):
        with ExitStack() as stack:
            for stage in ('audio', 'video40', 'media', 'publication'):
                path = self.root / f'state/locks/{stage}.lock'
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open('a'))
                # Even explicitly inheritable descriptors must not cross spawn.
                os.set_inheritable(handle.fileno(), True)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stack.callback(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
            actual = self.validate()
            self.assertEqual([r['sample_id'] for r, reasons in actual], ['good', 'bad'])
        self.assert_stage_locks_released()

    def test_recovered_text_and_exclusions_are_bound_to_each_work_item(self):
        rows = [dict(row, text='') for row in self.fixture.rows]
        recovered = {'good': {
            'sample_id': 'good', 'original_clip_id': rows[0]['original_clip_id'],
            'accepted': True, 'source_kind': 'source_asr_consensus', 'text': 'a a',
            'source': 'fixture consensus', 'quality': 'fixture exact agreement',
            'source_audio_sha256': sha256(rows[0]['audio_path']),
        }}
        # This case covers non-imported validation with row-level recovered text;
        # the separate end-to-end tests exercise immutable imported-row binding.
        options = {'recovered': recovered, 'strict_provenance': False,
                   'imported_source': None, 'import_receipts': None}
        actual = self.validate(rows[::-1], **options)
        self.assertEqual(actual, self.validate(rows[::-1], executor='thread', **options))
        self.assertEqual(actual[0][0]['sample_id'], 'bad')
        self.assertIn('text_unresolved', actual[0][1])
        self.assertEqual(actual[1][0]['text'], 'a a')
        self.assertEqual(actual[1][0]['source_kind'], 'source_asr_consensus')
        self.assertEqual(actual[1][0]['recovery_evidence'], recovered['good'])

    def test_process_publication_matches_thread_and_real_loader_and_pseudo(self):
        self.publish('thread_reference', executor='thread')
        self.assert_stage_locks_released()
        self.publish()
        self.assert_stage_locks_released()
        for filename in ('inventory.jsonl', 'train.jsonl', 'valid.jsonl', 'test.jsonl'):
            self.assertEqual((self.root / f'manifests/full/{filename}').read_bytes(),
                             (self.root / f'manifests/thread_reference/{filename}').read_bytes())
        for report_suffix in ('sample_quality.jsonl', 'original_coverage.jsonl'):
            self.assertEqual((self.root / f'reports/full_{report_suffix}').read_bytes(),
                             (self.root / f'reports/thread_reference_{report_suffix}').read_bytes())
        data = self.fixture.dataset()
        self.assertEqual(len(data), 1)
        batch = data.collate_fn([data[0]])
        self.assertEqual(tuple(batch['mel'].shape), (1, 64, 40))
        self.assertEqual(tuple(batch['video'].shape), (1, 40, 1024))
        self.publish('full_pseudo', pseudo=True)
        self.assert_stage_locks_released()
        pseudo = self.fixture.dataset('full_pseudo', pseudo=True)
        self.assertEqual(len(pseudo), 1)
        self.assertEqual(data.data_contract['imported_cache'], pseudo.data_contract['imported_cache'])
        for split in ('valid', 'test'):
            self.assertEqual((self.root / f'manifests/full/{split}.jsonl').read_bytes(),
                             (self.root / f'manifests/full_pseudo/{split}.jsonl').read_bytes())

    def test_tampered_receipt_aborts_process_publication_and_releases_locks(self):
        path = self.root / f'reports/import_receipts/{self.fixture.rows[0]["relative_stem"]}.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Immutable imported cache provenance failed'):
            self.publish()
        self.assertFalse((self.root / 'manifests/full/cache_completion.json').exists())
        self.assert_stage_locks_released()

    def test_missing_import_ledger_pin_fails_in_spawn_worker(self):
        receipts = {key: value for key, value in self.completion['receipts'].items() if key != 'good'}
        with self.assertRaisesRegex(ValueError, 'Missing canonical import ledger entry'):
            self.validate(import_receipts=receipts)
        self.assert_stage_locks_released()


if __name__ == '__main__':
    unittest.main()
