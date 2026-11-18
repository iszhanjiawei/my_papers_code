"""Text ablation and actual generated-entry regression checks."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from aligndit.script.eval import infer_celebvdub_setting2_dubbing as adapter
from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs
from aligndit.script.eval.verify_setting2_target_only_wer import s1_imported_code_hash, word_edits


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

    def test_role_config_requires_reference_and_target_at_the_dubbing_entry(self):
        source = adapter.build_source()
        namespace = {"__file__": adapter.__file__, "__name__": "adapter_test",
                     "select_model_text": adapter.select_model_text}
        exec(compile(source, adapter.__file__, "exec"), namespace)
        config = Path(adapter.__file__).parents[2] / "config/finetune_celebvdub_role_conditioned.yaml"
        with patch.object(sys, "argv", ["inference", "--config", str(config), "--checkpoint", "/tmp/model.pt",
                                       "--text-input-mode", "target-only", "--dry-run"]):
            args = namespace["parse_args"]()
        with self.assertRaisesRegex(ValueError, "requires reference and target transcripts"):
            namespace["run"](args)

    def test_role_geometry_ignores_text_length_and_preserves_target_video(self):
        reference = np.ones((3, 64), np.float32)
        target = np.full((5, 1024), 7, np.float32)
        mean, std = np.zeros(64, np.float32), np.ones(64, np.float32)
        cond, video, p, t, duration = make_sampling_inputs(
            reference, target, "a much longer reference transcript", mean, std, exact_duration=True
        )
        self.assertEqual((p, t, duration), (3, 5, 8))
        np.testing.assert_array_equal(cond, reference)
        np.testing.assert_array_equal(video[:p], np.zeros_like(video[:p]))
        np.testing.assert_array_equal(video[p:], target)

    def test_independent_word_distance_counts_all_error_types(self):
        self.assertEqual(word_edits("one two three", "one four three five"), (2, 3))
        self.assertEqual(word_edits("one two", ""), (2, 2))
        self.assertEqual(word_edits("one", " one "), (0, 1))

    def test_historical_helper_audit_ignores_only_unused_s1_functions(self):
        source = "VALUE = 1\ndef build_model():\n return VALUE\ndef run():\n pass\n"
        fingerprint = s1_imported_code_hash(source)
        self.assertEqual(fingerprint, s1_imported_code_hash(source.replace(" pass", " return 42")))
        self.assertNotEqual(fingerprint, s1_imported_code_hash(source.replace("VALUE = 1", "VALUE = 2")))
        self.assertNotEqual(fingerprint, s1_imported_code_hash(source.replace("return VALUE", "return 42")))


if __name__ == "__main__":
    unittest.main()
