"""Independent prompt geometry and real tiny-model CFG regression checks."""

import unittest

import numpy as np
import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.script.eval.infer_celebvdub_grid_reference import (
    make_sampling_inputs,
    reference_target_text,
)
from aligndit.script.misc.smoke_test_adaptive_temporal_band import BASE_ARCH, BAND_ARCH


class GridInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.mean = np.zeros(64, dtype=np.float32)
        self.std = np.ones(64, dtype=np.float32)

    def inputs(self, p, t, text="a b"):
        ref = np.arange(p * 64, dtype=np.float32).reshape(p, 64) / 100
        video = np.ones((t, 1024), dtype=np.float32)
        result = make_sampling_inputs(ref, video, text, self.mean, self.std)
        return ref, video, result

    def test_independent_transcripts(self):
        text = reference_target_text("bin blue at f two now", "you want that alien broad")
        self.assertEqual(text, "bin blue at f two now  you want that alien broad")
        with self.assertRaises(ValueError):
            reference_target_text(" ", "target")

    def test_reference_target_lengths_and_time_offsets(self):
        for p, t in ((3, 7), (7, 3)):
            ref, vid, (cond, total_vid, rp, rt, duration) = self.inputs(p, t)
            np.testing.assert_array_equal(cond, ref)
            np.testing.assert_array_equal(total_vid[p:p + t], vid)
            self.assertTrue((total_vid[:p] == 0).all())
            self.assertEqual((rp, rt, duration), (p, t, p + t))

    def test_text_extension_preserves_video_and_output_region(self):
        _, vid, (_, total_vid, p, t, duration) = self.inputs(3, 4, "x" * 12)
        self.assertEqual(duration, 13)
        np.testing.assert_array_equal(total_vid[p:p + t], vid)
        self.assertTrue((total_vid[p + t:] == 0).all())

    def test_sampler_limit_is_never_silently_clamped(self):
        with self.assertRaises(ValueError):
            self.inputs(2, 2, "x" * 4096)

    def test_real_cfg_sampler_with_different_prompt_lengths(self):
        arch = {**BASE_ARCH, **BAND_ARCH, "video_dim": 1024}
        model = CFM_VT(
            transformer=DiT_VT_MMDiT(**arch),
            mel_spec_module=PrecomputedAudioRepresentation(64, 16000, 400),
            num_channels=64, vocab_char_map={" ": 0, "a": 1, "b": 2},
            audio_video_ratio=1, odeint_kwargs={"method": "euler"},
        ).eval()
        with torch.inference_mode():
            for p, t, text in ((3, 7, "a b"), (7, 3, "a b"), (3, 4, "a " * 6)):
                _, _, (cond, video, _, _, duration) = self.inputs(p, t, text)
                output, _ = model.sample(
                    cond=torch.from_numpy(cond)[None], text=[text], duration=duration,
                    video=torch.from_numpy(video)[None], lens=torch.tensor([p]),
                    speaker_embedding=torch.ones(1, 12), steps=2,
                    cfg_strength=5.0, cfg_strength_v=2.0,
                    seed=0, use_epss=True,
                )
                self.assertEqual(output.shape, (1, duration, 64))
                torch.testing.assert_close(output[:, :p], torch.from_numpy(cond)[None])
                self.assertEqual(output[:, p:p + t].shape, (1, t, 64))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
