import tempfile
import unittest
import wave
from pathlib import Path

from aligndit.script.eval.grid_s1_common import (
    PROTOCOL, ROLE_MODE, identity, metric_summary, normalize_text, output_record, publish, sha,
)
from aligndit.script.eval.eval_celebvdub_grid_reference import word_edit_counts


class GridS1EvaluationTests(unittest.TestCase):
    def test_formatting_is_normalized_symmetrically(self):
        truth = 'bin white by a two again'
        hypothesis = '  BIN, White by A ２ again!\n'
        self.assertEqual(normalize_text(truth), normalize_text(hypothesis))
        self.assertEqual(word_edit_counts(normalize_text(truth), normalize_text(hypothesis)), (0, 6))

    def test_letter_a_is_not_an_article_to_drop(self):
        self.assertEqual(normalize_text('lay green at a 0 now'), 'lay green at a zero now')
        self.assertEqual(normalize_text('B2'), 'b two')

    def test_recognition_errors_are_preserved(self):
        self.assertEqual(word_edit_counts(normalize_text('lay green at a two now'), normalize_text('late green at a two now')), (1, 6))
        self.assertEqual(word_edit_counts('bin white by a two again', ''), (6, 6))

    def test_corpus_wer_is_not_mean_sentence_wer(self):
        rows = [{'id': 'val/s1/x', 'wer': 1.0, 'word_edit_distance': 1, 'reference_word_count': 1},
                {'id': 'val/s2/x', 'wer': 0.0, 'word_edit_distance': 0, 'reference_word_count': 6}]
        self.assertEqual(metric_summary(rows, 'wer')['wer'], 1/7)
        self.assertEqual(metric_summary(rows, 'wer')['reference_word_count'], 7)

    def test_full_identity_disambiguates_same_basename(self):
        for speaker in ('s1', 's2'):
            row = {'audio_relative_path': f'val/{speaker}/same.wav', 'utterance_key': f'grid/{speaker}/same'}
            self.assertEqual(identity(row), f'val/{speaker}/same')
        with self.assertRaises(ValueError):
            identity({'audio_relative_path': '../bad.wav', 'utterance_key': 'grid/bad'})

    def test_role_receipt_binds_both_roles_to_same_s1_clip(self):
        row = {"audio_relative_path": "val/s1/line.wav", "utterance_key": "grid/s1/line",
               "original_num_samples": 320, "text": "bin white by a two again"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / row["audio_relative_path"]
            path.parent.mkdir(parents=True)
            with wave.open(str(path), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(bytes(640))
            contract = root / "protocol.json"
            publish(contract, {"protocol": PROTOCOL, "text_conditioning": ROLE_MODE,
                               "reference_transcript_used_for_generation": True})
            key = identity(row)
            receipt = {"text_conditioning": ROLE_MODE, "protocol_sha256": sha(contract), "id": key, "sha256": sha(path),
                       "target_text": row["text"], "reference_text": row["text"], "reference_id": key}
            destination = root / "receipts" / (key + ".json")
            publish(destination, receipt)
            self.assertEqual(output_record(root, row), receipt)
            publish(destination, dict(receipt, reference_id="val/s1/another"))
            with self.assertRaisesRegex(RuntimeError, "same utterance"):
                output_record(root, row)

    def test_incomplete_or_duplicate_metric_results_fail(self):
        with self.assertRaises(ValueError):
            metric_summary([], 'sim')
        with self.assertRaises(ValueError):
            metric_summary([{'id': 'x', 'sim': 1}, {'id': 'x', 'sim': 1}], 'sim')


if __name__ == '__main__':
    unittest.main()
