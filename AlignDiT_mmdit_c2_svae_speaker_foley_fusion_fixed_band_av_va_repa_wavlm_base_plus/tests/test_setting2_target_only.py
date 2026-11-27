"""Text ablation and actual generated-entry regression checks."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from aligndit.script.eval import infer_celebvdub_setting2_dubbing as adapter
from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs
from aligndit.script.eval.verify_setting2_target_only_wer import verify, word_edits


class TargetOnlyTests(unittest.TestCase):
    def test_target_only_removes_reference_and_separator(self):
        self.assertEqual(adapter.select_model_text("a reference sentence", "only target", "target-only"),
                         "only target")

    def test_default_combined_text_preserves_historical_separator(self):
        self.assertEqual(adapter.select_model_text("reference", "target", "reference-target"),
                         "reference  target")
        with self.assertRaises(ValueError):
            adapter.select_model_text("reference", "target", "invalid")

    def test_executable_adapter_parser_and_sample_call(self):
        source = adapter.build_source()
        namespace = {"__file__": adapter.__file__, "__name__": "adapter_test",
                     "select_model_text": adapter.select_model_text}
        exec(compile(source, adapter.__file__, "exec"), namespace)
        for flags, mode in (([], "reference-target"), (["--text-input-mode", "target-only"], "target-only")):
            with patch.object(sys, "argv", ["inference", "--checkpoint", "/tmp/model.pt", *flags]):
                args = namespace["parse_args"]()
            self.assertEqual(args.text_input_mode, mode)
        self.assertIn('text = select_model_text(row["reference_text"], row["target_text"], args.text_input_mode)', source)
        self.assertIn('text=[text]', source)

    def test_audio_prompt_and_target_video_geometry_are_preserved(self):
        reference = np.ones((40, 64), np.float32)
        video = np.ones((60, 1024), np.float32)
        mean, std = np.zeros(64, np.float32), np.ones(64, np.float32)
        combined = make_sampling_inputs(reference, video, "reference  target", mean, std)
        target_only = make_sampling_inputs(reference, video, "target", mean, std)
        for before, after in zip(combined, target_only):
            np.testing.assert_array_equal(before, after)

    def test_independent_word_distance_counts_all_error_types(self):
        self.assertEqual(word_edits("one two three", "one four three five"), (2, 3))
        self.assertEqual(word_edits("one two", ""), (2, 2))
        self.assertEqual(word_edits("one", " one "), (0, 1))

    def test_target_only_comparison_requires_matching_checkpoint(self):
        shared = {"source_sha256": {}, "checkpoint_sha256": "foley-150k"}
        baseline = dict(shared, text_input_mode="reference-target", reference_text_conditioned=True,
                        text_source="reference_and_target")
        target_only = dict(shared, text_input_mode="target-only", reference_text_conditioned=False,
                           text_source="supplied_target_GT_dialogue_only", checkpoint_sha256="other-150k")
        with patch("aligndit.script.eval.verify_setting2_target_only_wer.read_json",
                   side_effect=[target_only, baseline]):
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                verify(Path("target_only"), Path("baseline"), Path("evaluator"))


if __name__ == "__main__":
    unittest.main()
