"""Regression checks for independent prompt geometry and text concatenation."""
import unittest
import numpy as np
from aligndit.script.eval.infer_setting2_dubbing import geometry, model_text


class Setting2Inputs(unittest.TestCase):
    def test_historical_separator_and_distinct_transcripts(self):
        self.assertEqual(model_text("reference sentence", "different target"), "reference sentence  different target")
        with self.assertRaises(ValueError):
            model_text("", "target")

    def test_prompt_and_target_keep_independent_lengths(self):
        for p, t in ((3, 9), (9, 3)):
            reference = np.ones((p, 64), np.float32)
            video = np.ones((t, 1024), np.float32)
            cond, combined, rp, rt, total = geometry(reference, video, "ab", np.zeros(64), np.ones(64))
            self.assertEqual((rp, rt, total), (p, t, p + t))
            np.testing.assert_array_equal(cond, reference)
            self.assertTrue((combined[:p] == 0).all())
            np.testing.assert_array_equal(combined[p:p+t], video)

    def test_long_text_adds_tail_without_moving_target(self):
        reference, video = np.ones((2, 64)), np.ones((3, 1024))
        _, combined, p, t, total = geometry(reference, video, "x" * 10, np.zeros(64), np.ones(64))
        self.assertEqual((p, t, total), (2, 3, 11))
        np.testing.assert_array_equal(combined[p:p+t], video)
        self.assertTrue((combined[p+t:] == 0).all())
        with self.assertRaises(ValueError):
            geometry(reference, video, "x" * 4096, np.zeros(64), np.ones(64))


if __name__ == "__main__":
    unittest.main()
