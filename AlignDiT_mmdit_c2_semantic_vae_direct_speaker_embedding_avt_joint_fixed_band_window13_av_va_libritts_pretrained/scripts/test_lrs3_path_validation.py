"""Bounded path prefetch preserves the LRS3 loader's ordered validation contract."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import unittest

import torch

from aligndit.model.lrs3_path_validation import path_checked_rows
from aligndit.model.lrs3_semantic_vae_dataset import SemanticVaeLRS3Dataset
from aligndit.model.semantic_vae_dataset import sha256_file
import test_lrs3_imported_source as imported_fixture
import test_lrs3_semantic_vae_dataset as contract_fixture


def assert_equal_samples(case, left, right):
    case.assertEqual(left.keys(), right.keys())
    for key, value in left.items():
        if torch.is_tensor(value):
            case.assertTrue(torch.equal(value, right[key]), key)
        else:
            case.assertEqual(value, right[key], key)


class PathCheckedRowsTests(unittest.TestCase):
    def test_serial_is_lazy_and_uses_only_calling_thread(self):
        consumed, validated = [], []
        owner = threading.current_thread()
        def rows():
            for index in range(5):
                consumed.append(index)
                yield index
        def validate(row):
            validated.append((row, threading.current_thread()))
        with path_checked_rows(rows(), validate, workers=0) as stream:
            iterator = iter(stream)
            row, check = next(iterator)
            self.assertEqual((row, consumed, validated), (0, [0], []))
            check()
            self.assertEqual(validated, [(0, owner)])
        self.assertEqual(consumed, [0])

    def test_prefetch_is_bounded_ordered_and_all_threads_join(self):
        consumed, thread_objects = [], set()
        release, started = threading.Event(), threading.Event()
        mutex = threading.Lock()
        def rows():
            for index in range(40):
                consumed.append(index)
                yield index
        def validate(row):
            with mutex:
                thread_objects.add(threading.current_thread())
            started.set()
            if not release.wait(10):
                raise TimeoutError('Test did not release path workers')
        try:
            with path_checked_rows(rows(), validate, workers=2) as stream:
                iterator = iter(stream)
                first, check = next(iterator)
                self.assertTrue(started.wait(10))
                self.assertEqual(first, 0)
                self.assertLessEqual(len(consumed), 4)
                release.set()
                check()
                seen = [first]
                for row, check in iterator:
                    check()
                    seen.append(row)
                self.assertEqual(seen, list(range(40)))
        finally:
            release.set()
        self.assertTrue(thread_objects)
        self.assertTrue(all(not thread.is_alive() for thread in thread_objects))

    def test_parent_failure_waits_for_active_workers_before_exit(self):
        started, release = threading.Event(), threading.Event()
        threads = set()
        mutex = threading.Lock()
        def validate(row):
            with mutex:
                threads.add(threading.current_thread())
            started.set()
            if not release.wait(10):
                raise TimeoutError('Test did not release path worker')
        timer = None
        try:
            with self.assertRaisesRegex(ValueError, 'earlier parent validation'):
                with path_checked_rows(range(20), validate, workers=2) as stream:
                    next(iter(stream))
                    self.assertTrue(started.wait(10))
                    timer = threading.Timer(.05, release.set)
                    timer.start()
                    raise ValueError('earlier parent validation')
        finally:
            release.set()
            if timer is not None:
                timer.join()
        self.assertTrue(threads)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_worker_failure_is_deferred_until_check_and_threads_join(self):
        threads = set()
        mutex = threading.Lock()
        def validate(row):
            with mutex:
                threads.add(threading.current_thread())
            if row == 0:
                raise ValueError('first path failure')
        with self.assertRaisesRegex(ValueError, 'first path failure'):
            with path_checked_rows(range(20), validate, workers=2) as stream:
                row, check = next(iter(stream))
                self.assertEqual(row, 0)
                check()
        self.assertTrue(threads)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_later_parser_failure_cannot_preempt_prior_row_validation(self):
        def rows():
            yield {'id': 0}
            raise json.JSONDecodeError('later parser error', '{', 1)
        for workers in (0, 2):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, 'earlier identity'):
                    with path_checked_rows(rows(), lambda row: None, workers=workers) as stream:
                        row, check = next(iter(stream))
                        self.assertEqual(row, {'id': 0})
                        raise ValueError('earlier identity')
                with path_checked_rows(rows(), lambda row: None, workers=workers) as stream:
                    iterator = iter(stream)
                    row, check = next(iterator)
                    check()
                    with self.assertRaisesRegex(json.JSONDecodeError, 'later parser error'):
                        next(iterator)

    def test_invalid_worker_counts_are_rejected(self):
        for workers in (-1, 33):
            with self.subTest(workers=workers), self.assertRaises((ValueError, TypeError)):
                with path_checked_rows([], lambda row: None, workers=workers) as stream:
                    list(stream)


class LoaderPathValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = contract_fixture.LRS3ContractTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def assert_modes_reject_identically(self, message):
        failures = []
        for workers in (0, 16, 32):
            with self.subTest(workers=workers), self.assertRaisesRegex(Exception, message) as result:
                self.fixture.dataset(path_validation_workers=workers)
            failures.append((type(result.exception), str(result.exception)))
        self.assertEqual(failures, [failures[0]] * 3)

    def outside_latent_link(self):
        q = self.fixture
        path = q.root / q.rows[0]['latent_relative_path']
        path.unlink()
        path.symlink_to(q.root.parent / f'{q.root.name}-outside.npy')

    def test_pilot_strict_pseudo_runtime_and_collation_are_identical(self):
        q = self.fixture
        for mode in ('pilot', 'strict', 'pseudo'):
            if mode == 'strict':
                q.strict_publication()
            elif mode == 'pseudo':
                q.pseudo_publication()
            with self.subTest(mode=mode):
                serial = q.dataset(path_validation_workers=0)
                expected = serial.collate_fn([serial[index] for index in range(len(serial))])
                for workers in (16, 32):
                    threaded = q.dataset(path_validation_workers=workers)
                    self.assertEqual(threaded.records, serial.records)
                    self.assertEqual(threaded.data_contract, serial.data_contract)
                    actual = threaded.collate_fn([threaded[index] for index in range(len(threaded))])
                    assert_equal_samples(self, actual, expected)

    def test_within_row_path_loop_keeps_first_error(self):
        self.outside_latent_link()
        self.fixture.rows[0]['video_40hz_relative_path'] = 'wrong.npy'
        self.fixture.freeze()
        self.assert_modes_reject_identically('latent_relative_path escapes cache root')

    def test_identity_and_text_errors_stay_before_prefetched_path_errors(self):
        q = self.fixture
        original = deepcopy(q.rows[0])
        self.outside_latent_link()
        for changes, message in (({'source_split': 'unknown'}, 'Invalid or duplicate'),
                                 ({'text': 'a?'}, 'out-of-vocabulary')):
            q.rows[0] = {**original, **changes}
            q.freeze()
            self.assert_modes_reject_identically(message)

    def test_path_error_stays_before_ctc_and_later_row_error(self):
        q = self.fixture
        q.rows[0]['latent_relative_path'] = 'wrong.npy'
        q.rows[0]['ctc_min_input_frames'] = -1
        q.rows[1]['text'] = 'a?'
        q.freeze()
        self.assert_modes_reject_identically('Invalid LRS3 latent_relative_path')

    def test_earlier_ctc_error_stays_before_later_path_error(self):
        q = self.fixture
        q.rows[0]['ctc_min_input_frames'] = -1
        q.rows[1]['latent_relative_path'] = 'wrong.npy'
        q.freeze()
        self.assert_modes_reject_identically('Invalid LRS3 40 Hz CTC metadata')

    def test_traversal_path_rejection_is_unchanged(self):
        q = self.fixture
        row = q.rows[0]
        row['relative_stem'] = 'train/pretrain/../outside'
        row['utterance_key'] = 'lrs3/' + row['relative_stem']
        for field, prefix in (('latent_relative_path', 'latents'),
                              ('video_40hz_relative_path', 'video_40hz'),
                              ('speaker_relative_path', 'speaker_embeddings')):
            row[field] = f'{prefix}/{row["relative_stem"]}.npy'
        q.freeze()
        self.assert_modes_reject_identically('Invalid latent_relative_path relative path')

    def test_missing_file_and_within_root_symlink_keep_existing_behavior(self):
        q = self.fixture
        path = q.root / q.rows[0]['latent_relative_path']
        payload = path.read_bytes()
        path.unlink()
        for workers in (0, 16, 32):
            data = q.dataset(path_validation_workers=workers)
            with self.assertRaisesRegex(FileNotFoundError, 'cache is missing'):
                data[0]
        target = q.root / 'retained.npy'
        target.write_bytes(payload)
        path.symlink_to(target)
        serial = q.dataset(path_validation_workers=0)[0]
        for workers in (16, 32):
            assert_equal_samples(self, serial, q.dataset(path_validation_workers=workers)[0])

    def test_later_invalid_json_does_not_mask_earlier_text_error(self):
        q = self.fixture
        q.rows[0]['text'] = 'a?'
        q.freeze()
        manifest = q.manifests / 'train.jsonl'
        with manifest.open('a') as handle:
            handle.write('{invalid json\n')
        meta_path = q.manifests / 'inventory_meta.json'
        meta = json.loads(meta_path.read_text())
        meta['manifests']['train.jsonl'].update(sha256=sha256_file(manifest),
                                               size_bytes=manifest.stat().st_size)
        q.write(meta_path, meta)
        self.assert_modes_reject_identically('out-of-vocabulary')


class ImportedLoaderPathValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = imported_fixture.ImportedSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.import_all()
        self.fixture.publish()

    def dataset(self, workers):
        q = self.fixture
        return SemanticVaeLRS3Dataset(
            manifest_path=q.root / 'manifests/full/train.jsonl', cache_root=q.root,
            normalization_path=q.norm, vocab_path=q.vocab,
            expected_normalization_sha256=sha256_file(q.norm),
            expected_vocab_sha256=sha256_file(q.vocab), expected_selection='full',
            expected_text_policy='source_grounded', path_validation_workers=workers,
            speaker_embedding_model_id='iic/speech_campplus_sv_zh_en_16k-common_advanced',
            speaker_embedding_checkpoint_sha256='92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199')

    def test_actual_import_contract_records_and_tensors_are_identical(self):
        serial = self.dataset(0)
        for workers in (16, 32):
            threaded = self.dataset(workers)
            self.assertEqual(threaded.records, serial.records)
            self.assertEqual(threaded.data_contract, serial.data_contract)
            assert_equal_samples(self, threaded[0], serial[0])
            assert_equal_samples(self, threaded.collate_fn([threaded[0]]),
                                 serial.collate_fn([serial[0]]))

    def test_import_receipt_failure_remains_before_path_validation(self):
        q = self.fixture
        manifest = q.root / 'manifests/full/train.jsonl'
        row = json.loads(manifest.read_text())
        row['import_receipt_sha256'] = '0' * 64
        row['latent_relative_path'] = 'wrong.npy'
        manifest.write_text(json.dumps(row) + '\n')
        meta_path = q.root / 'manifests/full/inventory_meta.json'
        meta = json.loads(meta_path.read_text())
        meta['manifests']['train.jsonl'].update(sha256=sha256_file(manifest),
                                               size_bytes=manifest.stat().st_size)
        q.write(meta_path, meta)
        failures = []
        for workers in (0, 16, 32):
            with self.assertRaisesRegex(ValueError, 'immutable import receipt ledger') as result:
                self.dataset(workers)
            failures.append(str(result.exception))
        self.assertEqual(failures, [failures[0]] * 3)


if __name__ == '__main__':
    unittest.main()
