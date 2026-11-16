"""Regression checks for independent publication scope and approval boundaries."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lrs3_audit_imported_publication import audit, project_eligibility


def source_row(text='', split='train', eligible=True):
    return {'text': text, 'split': split,
            'text_status': 'original_pipeline' if text else 'missing',
            'duration_eligible': eligible, 'audio_seconds': 2.0, 'video_seconds': 2.0}


class EligibilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            'original': source_row('original words'),
            'strict': source_row(), 'p1': source_row(), 'p2': source_row(),
            'excluded': source_row(), 'short': source_row('known words', eligible=False),
        }
        self.labels = {
            sid: {'text': 'Recovered Words', 'source_kind':
                  'source_asr_consensus' if sid == 'strict' else 'asr_pseudo'}
            for sid in ('strict', 'p1', 'p2', 'excluded')}
        self.partitions = {p: {p} for p in ('strict', 'p1', 'p2', 'excluded')}
        self.vocab = set('abcdefghijklmnopqrstuvwxyz ')

    def test_strict_and_train_p1_are_distinct_and_candidates_remain_unapproved(self):
        strict, strict_ids = project_eligibility(
            self.source, self.labels, self.partitions, self.vocab, {}, include_p1=False)
        pseudo, pseudo_ids = project_eligibility(
            self.source, self.labels, self.partitions, self.vocab, {}, include_p1=True)
        self.assertEqual(strict_ids, {'original', 'strict'})
        self.assertEqual(pseudo_ids - strict_ids, {'p1'})
        self.assertEqual(strict['strict'][:3],
                         ('recovered words', 'source_asr_consensus', 'recovered_consensus'))
        self.assertIn('text_unresolved', strict['p1'][3])
        self.assertEqual(pseudo['p1'][1], 'asr_pseudo')
        for sid in ('p2', 'excluded'):
            self.assertFalse(pseudo[sid][4])
            self.assertIn('text_unresolved', pseudo[sid][3])

    def test_media_subtraction_only_removes_eligible_intersection(self):
        media = {s: {'reason': 'mouth_geometry_invalid'} for s in ('strict', 'p1', 'p2')}
        _, strict_ids = project_eligibility(
            self.source, self.labels, self.partitions, self.vocab, media, include_p1=False)
        _, pseudo_ids = project_eligibility(
            self.source, self.labels, self.partitions, self.vocab, media, include_p1=True)
        self.assertEqual(len(media), 3)
        self.assertEqual(len(strict_ids & media.keys()), 1)
        self.assertEqual(len(pseudo_ids & media.keys()), 2)
        self.assertEqual(strict_ids - media.keys(), {'original'})
        self.assertEqual(pseudo_ids - media.keys(), {'original'})

    def test_pseudo_cannot_be_added_to_validation_or_test(self):
        for split in ('valid', 'test'):
            with self.subTest(split=split):
                self.source['p1']['split'] = split
                expected, ids = project_eligibility(
                    self.source, self.labels, self.partitions, self.vocab, {}, include_p1=True)
                self.assertNotIn('p1', ids)
                self.assertIn('pseudo_outside_train', expected['p1'][3])

    def test_preserves_known_text_and_rejects_oov_and_duration_mismatch(self):
        self.partitions['strict'].add('original')
        self.labels['original'] = {'text': 'overwrite', 'source_kind': 'source_asr_consensus'}
        self.source['strict']['audio_seconds'] = 2.2
        self.labels['p1']['text'] = 'not in vocab #'
        expected, ids = project_eligibility(
            self.source, self.labels, self.partitions, self.vocab, {}, include_p1=True)
        self.assertEqual(expected['original'][0], 'original words')
        self.assertFalse(expected['original'][4])
        self.assertIn('audio_video_duration_delta', expected['strict'][3])
        self.assertIn('text_oov', expected['p1'][3])
        self.assertEqual(ids, {'original'})


class PublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.origin, self.other = [base / n for n in ('new', 'old', 'wrong')]
        for p in (self.root, self.origin, self.other):
            p.mkdir()

    def markers(self):
        for mode in ('full', 'full_pseudo'):
            p = self.root/'manifests'/mode/'cache_completion.json'
            p.parent.mkdir(parents=True)
            p.write_text('{}')

    def test_partial_publication_does_not_validate_or_create_reports(self):
        with patch('aligndit.lrs3_imported_source.validate_imported_source') as validator:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit(self.root, self.origin), 3)
            validator.assert_not_called()
        self.assertFalse((self.root/'reports').exists())

    def test_missing_import_contract_cannot_be_treated_as_an_original_root(self):
        self.markers()
        with patch('aligndit.lrs3_imported_source.validate_imported_source', return_value=None):
            with self.assertRaisesRegex(ValueError, 'no imported-source'):
                audit(self.root, self.origin)
        self.assertFalse((self.root/'reports').exists())

    def test_wrong_origin_fails_before_writing_and_passes_import_pin(self):
        self.markers()
        with patch('aligndit.lrs3_imported_source.validate_imported_source',
                   return_value={'origin_root': str(self.other)}) as validator:
            with self.assertRaisesRegex(ValueError, 'Explicit origin'):
                audit(self.root, self.origin, expected_import_sha256='a' * 64)
            validator.assert_called_once_with(self.root, expected_sha256='a' * 64)
        self.assertFalse((self.root/'reports').exists())

    def test_original_and_destination_must_differ(self):
        with self.assertRaisesRegex(ValueError, 'must differ'):
            audit(self.origin, self.origin)


if __name__ == '__main__':
    unittest.main()
