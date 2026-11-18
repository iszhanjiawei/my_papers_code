"""Verify prompt boundaries, target teacher timebase and typed CFG inputs at the public CFM API."""

import unittest

import torch
from torch import nn

from aligndit.model.cfm_vt import CFM_VT


class RecordingBackbone(nn.Module):
    dim = 2
    role_conditioned_text = True

    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.5))
        self.cached = False

    def clear_cache(self):
        self.cached = False

    def forward(self, **kwargs):
        self.seen = kwargs
        self.cached = kwargs.get("cache", False)
        pred = torch.ones_like(kwargs["x"]) * self.velocity
        if kwargs.get("return_repa"):
            # A reversed teacher direction in the prefix must not affect the loss.
            features = torch.where(kwargs["generation_mask"][:, :, None], 1.0, -1.0)
            return pred, {}, features.expand_as(pred)
        if kwargs.get("cfg_infer"):
            pred = pred.repeat(3, 1, 1)
        return pred, {}


class RoleFlowTests(unittest.TestCase):
    def setUp(self):
        self.net = RecordingBackbone()
        self.model = CFM_VT(
            self.net,
            num_channels=2,
            audio_video_ratio=1,
            repa_lambda=1.0,
            ctc_lambda=0,
            vocab_char_map={"a": 1, "b": 2},
            audio_drop_prob=0,
            cond_drop_prob=0,
            video_drop_prob=0,
            text_drop_prob=0,
        )

    def test_variable_prompt_offsets_condition_and_supervise_only_target_window(self):
        lens = torch.tensor([6, 4])
        prompt = torch.tensor([2, 1])
        inp = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
        loss, components, cond, pred = self.model(
            inp,
            ["aa", "b"],
            torch.zeros_like(inp),
            lens=lens,
            video_lens=lens,
            reference_text=["b", "aa"],
            prompt_lens=prompt,
            repa_features=torch.ones(2, 5, 2),
            repa_feature_lens=torch.tensor([5, 4]),
        )
        expected = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(self.net.seen["generation_mask"], expected))
        self.assertEqual(self.net.seen["reference_text"].tolist(), [[2, -1], [1, 1]])
        self.assertEqual(self.net.seen["text"].tolist(), [[1, 1], [2, -1]])
        self.assertTrue(torch.equal(cond[0, :2], inp[0, :2]))
        self.assertTrue((cond[expected] == 0).all())
        self.assertAlmostEqual(components["repa_loss"], 0.0, places=5)
        # Gradient from the actual flow objective appears only in target positions.
        pred.retain_grad()
        loss.backward()
        self.assertTrue((pred.grad[~expected] == 0).all())
        self.assertGreater(pred.grad[expected].abs().sum().item(), 0)

    def test_cfg_sample_keeps_prompt_and_target_shape_despite_long_reference(self):
        cond = torch.arange(4.0).reshape(1, 2, 2)
        out, trajectory = self.model.sample(
            cond,
            ["ba"],
            6,
            torch.zeros(1, 6, 2),
            reference_text=["a" * 12],
            seed=0,
            steps=2,
            cfg_strength=5,
            cfg_strength_v=2,
            use_epss=False,
        )
        self.assertTrue(torch.equal(out[:, :2], cond))
        self.assertEqual(trajectory.shape[-2:], (6, 2))
        self.assertEqual(self.net.seen["reference_text"].shape, (1, 12))
        self.assertTrue(self.net.seen["cfg_infer"])
        self.assertFalse(self.net.cached)
        with self.assertRaisesRegex(ValueError, "both reference_text"):
            self.model.sample(cond, ["ba"], 6, torch.zeros(1, 6, 2))

    def test_combined_inference_parses_before_encoding_and_matches_separate_fields(self):
        def sample(text, reference=None):
            return self.model.sample(
                torch.zeros(1, 2, 2),
                [text],
                6,
                torch.zeros(1, 6, 2),
                reference_text=reference,
                steps=2,
                seed=0,
                use_epss=False,
            )[0]

        expected = sample("ba", ["ab"])
        ref = self.net.seen["reference_text"].clone()
        target = self.net.seen["text"].clone()
        for line in ("[REF] ab [TGT] ba", "ab  ba"):
            self.assertTrue(torch.equal(sample(line), expected))
            self.assertTrue(torch.equal(self.net.seen["reference_text"], ref))
            self.assertTrue(torch.equal(self.net.seen["text"], target))
        for line in ("[REF] a", "a  b  a", "a   b", "[TGT] b [REF] a", "[REF] a [TGT] "):
            with self.subTest(line=line), self.assertRaises(ValueError):
                sample(line)
        with self.assertRaisesRegex(ValueError, "plain transcript"):
            sample("[REF] a [TGT] ba", ["ab"])


if __name__ == "__main__":
    unittest.main()
