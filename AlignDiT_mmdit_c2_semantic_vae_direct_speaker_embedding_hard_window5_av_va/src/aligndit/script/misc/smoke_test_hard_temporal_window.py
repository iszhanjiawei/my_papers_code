"""CPU contracts for strict five-frame AV+VA attention in this snapshot.

Run without data, checkpoints, or GPUs::

    PYTHONPATH=src python src/aligndit/script/misc/smoke_test_hard_temporal_window.py

The independent full-softmax reference checks probabilities as well as outputs.
Helpers are imported only from this copied experiment. No sibling is required.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.script.misc import smoke_test_adaptive_temporal_band as helpers


HARD_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "fixed",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.1,
    "temporal_band_bidirectional": True,
    "temporal_band_fixed_window_radius_frames": 2,
}


def make_model(**overrides):
    return helpers.warm_start(helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **HARD_ARCH, **overrides}))


def make_bias(audio_len, video_len, batch=2, radius=2, **kwargs):
    band = FixedTemporalBand(dim=64, window_radius_frames=radius, **kwargs)
    return band.bias(*band(torch.zeros(batch, video_len, 64), audio_len), video_len)


class HardWindowGeometryTests(unittest.TestCase):
    def test_exact_interior_and_boundary_key_sets(self):
        bias = make_bias(9, 9)
        self.assertEqual(bias.dtype, torch.float32)
        self.assertEqual(bias.shape, (2, 9, 9))
        expected = ([0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8], [5, 6, 7, 8], [6, 7, 8])
        for query, keys in enumerate(expected):
            self.assertEqual(torch.isfinite(bias[0, query]).nonzero().flatten().tolist(), keys)
            self.assertTrue((bias[0, query, keys] == 0).all())
        self.assertTrue(torch.isneginf(bias[~torch.isfinite(bias)]).all())
        torch.testing.assert_close(bias[0], bias[1], atol=0, rtol=0)

    def test_short_unequal_sequences_and_zero_radius(self):
        for n_a, n_v in ((1, 1), (2, 3), (3, 2), (9, 2), (2, 9)):
            for radius in (0, 2):
                with self.subTest(audio_len=n_a, video_len=n_v, radius=radius):
                    bias = make_bias(n_a, n_v, radius=radius)
                    for i in range(n_a):
                        for j in range(n_v):
                            self.assertEqual(bias[0, i, j].item(), 0.0 if abs(i - j) <= radius else -math.inf)

    def test_integer_boundaries_at_large_indices_and_under_autocast(self):
        # Simulate a late aligned sequence slice beyond FP32's exact-integer
        # range without allocating a huge square attention matrix. Translating
        # both index vectors must leave every endpoint exactly unchanged.
        expected = make_bias(19, 17)
        original_arange = torch.arange

        def translated_arange(*args, **kwargs):
            self.assertEqual(kwargs.get("dtype"), torch.long)
            return original_arange(*args, **kwargs) + 2**25

        with patch("aligndit.model.fixed_temporal_band.torch.arange", side_effect=translated_arange):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = make_bias(19, 17)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_sigma_and_video_content_do_not_change_hard_geometry(self):
        first = FixedTemporalBand(dim=64, sigma_seconds=0.025, window_radius_frames=2)
        second = FixedTemporalBand(dim=64, sigma_seconds=4.0, window_radius_frames=2)
        expected = first.bias(*first(torch.randn(2, 9, 64), 11), 9)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            video = torch.full((2, 9, 64), float("nan"), dtype=dtype)
            actual = second.bias(*second(video, 11), 9)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_omitted_radius_keeps_exact_legacy_gaussian_formula(self):
        band = FixedTemporalBand(dim=64, audio_fps=40, video_fps=25, offset_seconds=0.025, sigma_seconds=0.08)
        offset, sigma = band(torch.zeros(2, 7, 64), 11)
        audio_times = torch.arange(11, dtype=torch.float32) / 40
        video_times = torch.arange(7, dtype=torch.float32) / 25
        distance = video_times[None, None, :] - audio_times[None, :, None] - offset.float().unsqueeze(-1)
        expected = -0.5 * (distance / sigma.float().unsqueeze(-1)).square()
        torch.testing.assert_close(band.bias(offset, sigma, 7), expected, atol=0, rtol=0)
        self.assertIsNone(band.window_radius_frames)

    def test_rejects_invalid_radius_rates_and_offset(self):
        for radius in (-1, 2.0, True, False, "2"):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "window_radius_frames"):
                FixedTemporalBand(dim=64, window_radius_frames=radius)
        for kwargs in ({"audio_fps": 40, "video_fps": 25}, {"offset_seconds": 0.025},
                       {"offset_seconds": -0.025}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "equal.*zero offset"):
                FixedTemporalBand(dim=64, window_radius_frames=2, **kwargs)


class HardWindowJointAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(739)
        self.block = helpers.MMDiTBlock_VT(
            dim=64, heads=4, dim_head=16, text_dim=32, dropout=0.0,
            attn_mask_enabled=True, prompt_isolated_ca=False, temporal_band_bidirectional=True,
        ).eval()
        self.audio = torch.randn(2, 9, 64)
        self.video = torch.randn(2, 6, 64)
        self.audio_mask = torch.arange(9)[None] < torch.tensor([9, 5])[:, None]
        self.video_mask = torch.arange(6)[None] < torch.tensor([6, 3])[:, None]
        self.bias = make_bias(9, 6)

    def full_reference(self, audio_mask=None, video_mask=None):
        q_a, k_a, v_a = self.block._qkv(self.block.attn, self.audio)
        q_v, k_v, v_v = self.block._qkv(self.block.v_attn, self.video)
        q, k, v = [torch.cat(pair, dim=2) for pair in ((q_a, q_v), (k_a, k_v), (v_a, v_v))]
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        prior = torch.zeros_like(logits)
        # Use explicit integer loop bounds independent of production geometry.
        for i in range(9):
            for j in range(6):
                if j < i - 2 or j > i + 2:
                    prior[:, :, i, 9 + j] = -math.inf
                    prior[:, :, 9 + j, i] = -math.inf
        if self.block.attn_mask_enabled:
            audio_valid = torch.ones(2, 9, dtype=torch.bool) if audio_mask is None else audio_mask
            video_valid = torch.ones(2, 6, dtype=torch.bool) if video_mask is None else video_mask
            valid = torch.cat((audio_valid, video_valid), dim=-1)
            prior.masked_fill_(~valid[:, None, None, :], -math.inf)
        probabilities = (logits + prior).softmax(dim=-1)
        attended = (probabilities @ v).transpose(1, 2).reshape(2, 15, 64)
        out_a = self.block.attn.to_out[1](self.block.attn.to_out[0](attended[:, :9]))
        out_v = self.block.v_attn.to_out[1](self.block.v_attn.to_out[0](attended[:, 9:]))
        if audio_mask is not None:
            out_a = out_a.masked_fill(~audio_mask[..., None], 0)
        if video_mask is not None:
            out_v = out_v.masked_fill(~video_mask[..., None], 0)
        return (out_a, out_v), prior, probabilities

    def test_independent_full_softmax_matches_with_padding_and_mask_toggle(self):
        for training in (False, True):
            self.block.train(training)
            for mask_enabled in (False, True):
                self.block.attn_mask_enabled = mask_enabled
                for audio_mask, video_mask in ((None, None), (self.audio_mask, self.video_mask),
                                               (self.audio_mask, None), (None, self.video_mask)):
                    with self.subTest(training=training, mask_enabled=mask_enabled,
                                      audio_mask=audio_mask is not None, video_mask=video_mask is not None):
                        expected, prior, probabilities = self.full_reference(audio_mask, video_mask)
                        actual, observed = helpers.captured_attention(
                            self.block, self.audio, self.video, mask=audio_mask, v_mask=video_mask,
                            temporal_band_bias=self.bias,
                        )
                        torch.testing.assert_close(observed, prior, atol=0, rtol=0)
                        self.assertEqual(torch.count_nonzero(probabilities[torch.isneginf(prior)]).item(), 0)
                        for out, target in zip(actual, expected):
                            torch.testing.assert_close(out, target, atol=3e-7, rtol=3e-6)
                            self.assertTrue(torch.isfinite(out).all())

    def test_av_va_transpose_with_unchanged_aa_vv_logits(self):
        _, prior = helpers.captured_attention(self.block, self.audio, self.video, temporal_band_bias=self.bias)
        expected = self.bias[:, None].expand(-1, 4, -1, -1)
        torch.testing.assert_close(prior[:, :, :9, 9:], expected, atol=0, rtol=0)
        torch.testing.assert_close(prior[:, :, 9:, :9], expected.transpose(-1, -2), atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(prior[:, :, :9, :9]).item(), 0)
        self.assertEqual(torch.count_nonzero(prior[:, :, 9:, 9:]).item(), 0)

    def test_forbidden_video_keys_have_exactly_zero_direct_gradient(self):
        audio = self.audio.detach().requires_grad_()
        video = self.video.detach().requires_grad_()
        out_a, _ = self.block.joint_attn(audio, video, temporal_band_bias=self.bias)
        out_a[:, 0].square().sum().backward()
        # Audio query 0 directly attends video keys 0, 1, 2 only.
        self.assertEqual(torch.count_nonzero(video.grad[:, 3:]).item(), 0)
        helpers.assert_nonzero_finite(self, video.grad[:, :3], "permitted video keys")

    def test_finite_backward_with_padding_empty_stream_and_mask_toggle(self):
        mixed_audio = self.audio_mask.clone()
        mixed_video = self.video_mask.clone()
        mixed_audio[1] = False
        mixed_video[1] = False
        cases = ((self.audio_mask, self.video_mask),
                 (self.audio_mask, torch.zeros_like(self.video_mask)),
                 (torch.zeros_like(self.audio_mask), self.video_mask),
                 (torch.zeros_like(self.audio_mask), torch.zeros_like(self.video_mask)),
                 (mixed_audio, mixed_video))
        for mask_enabled in (False, True):
            self.block.attn_mask_enabled = mask_enabled
            for audio_mask, video_mask in cases:
                with self.subTest(mask_enabled=mask_enabled,
                                  audio_valid=audio_mask.any().item(), video_valid=video_mask.any().item()):
                    self.block.zero_grad(set_to_none=True)
                    audio = self.audio.detach().requires_grad_()
                    video = self.video.detach().requires_grad_()
                    outputs = self.block.joint_attn(audio, video, mask=audio_mask, v_mask=video_mask,
                                                   temporal_band_bias=self.bias)
                    for output, valid in zip(outputs, (audio_mask, video_mask)):
                        self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
                    loss = sum(out.square().mean() for out in outputs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for value in (audio.grad, video.grad):
                        self.assertTrue(torch.isfinite(value).all())
                    for parameter in self.block.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_empty_rows_keep_forbidden_cross_keys_masked_and_valid_sample_unchanged(self):
        audio_mask = self.audio_mask.clone()
        video_mask = self.video_mask.clone()
        audio_mask[1] = False
        video_mask[1] = False
        reference = self.block.joint_attn(self.audio, self.video, mask=self.audio_mask,
                                         v_mask=self.video_mask, temporal_band_bias=self.bias)
        outputs, observed = helpers.captured_attention(
            self.block, self.audio, self.video, mask=audio_mask, v_mask=video_mask,
            temporal_band_bias=self.bias,
        )
        self.assertTrue(torch.isneginf(observed[1, :, :9, 9:]).all())
        self.assertTrue(torch.isneginf(observed[1, :, 9:, :9]).all())
        for output, expected in zip(outputs, reference):
            torch.testing.assert_close(output[0], expected[0], atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(output[1]).item(), 0)


class HardWindowBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(615)

    def test_parameter_state_and_rng_unchanged_from_gaussian(self):
        models, rng_states = [], []
        for radius in (None, 2):
            torch.manual_seed(253)
            models.append(helpers.DiT_VT_MMDiT(**{
                **helpers.BASE_ARCH, **HARD_ARCH, "temporal_band_fixed_window_radius_frames": radius,
            }))
            rng_states.append(torch.get_rng_state().clone())
        gaussian, hard = models
        torch.testing.assert_close(rng_states[0], rng_states[1], atol=0, rtol=0)
        self.assertEqual(set(gaussian.state_dict()), set(hard.state_dict()))
        self.assertEqual(sum(p.numel() for p in gaussian.parameters()), sum(p.numel() for p in hard.parameters()))
        self.assertEqual(list(hard.temporal_band.parameters()), [])
        self.assertEqual(dict(hard.temporal_band.state_dict()), {})
        for key, value in hard.state_dict().items():
            torch.testing.assert_close(value, gaussian.state_dict()[key], atol=0, rtol=0, msg=key)
        hard.load_state_dict(gaussian.state_dict(), strict=True)

    def test_rejects_disabled_adaptive_or_invalid_hard_window(self):
        cases = ({"temporal_band_enabled": False}, {"temporal_band_mode": "adaptive"},
                 {"temporal_band_fixed_window_radius_frames": True},
                 {"temporal_band_fixed_window_radius_frames": -1},
                 {"temporal_band_fixed_offset_seconds": 0.025},
                 {"temporal_band_video_fps": 25}, {"audio_video_ratio": 4})
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **HARD_ARCH, **overrides})

    def test_cfg_matches_separate_and_null_video_does_not_leak(self):
        model = make_model().eval()
        inputs = helpers.make_inputs()
        cases = (({}, ({}, {"drop_video": True},
                       {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
                 ({"drop_video": True}, ({"drop_video": True},
                                          {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
                 ({"drop_text": True}, ({"drop_text": True},
                                         {"drop_audio_cond": True, "drop_text": True, "drop_video": True})))
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                packed, ctc = model(**inputs, cfg_infer=True, **packed_flags)
                separate = [model(**inputs, **flags) for flags in separate_flags]
                torch.testing.assert_close(packed, torch.cat([out[0] for out in separate]), atol=3e-6, rtol=3e-5)
                for layer in ctc:
                    for field in ("z_tilde", "z_lens"):
                        torch.testing.assert_close(ctc[layer][field],
                                                   torch.cat([out[1][layer][field] for out in separate]),
                                                   atol=3e-6, rtol=3e-5)
            original, _ = model(**inputs, cfg_infer=True)
            changed_video = inputs["video"].clone()
            changed_video[0] = torch.randn_like(changed_video[0]) * 7
            changed, _ = model(**{**inputs, "video": changed_video}, cfg_infer=True)
        torch.testing.assert_close(original[1:], changed[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - changed[0]).abs().max().item(), 1e-5)

    def test_checkpointed_training_gradients_match_and_reach_both_streams(self):
        inputs = helpers.make_inputs()
        target = torch.randn(2, 12, 64)
        for mask_enabled in (False, True):
            plain = make_model(checkpoint_activations=False, attn_mask_enabled=mask_enabled).train()
            checked = make_model(checkpoint_activations=True, attn_mask_enabled=mask_enabled).train()
            checked.load_state_dict(plain.state_dict(), strict=True)
            outputs = []
            for model in (plain, checked):
                out, ctc = model(**inputs)
                loss = (out - target).square().mean()
                loss = loss + 0.01 * sum(item["z_tilde"].square().mean() for item in ctc.values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                params = dict(model.named_parameters())
                for name in ("transformer_blocks.0.attn.to_q.weight",
                             "transformer_blocks.0.v_attn.to_q.weight",
                             "transformer_blocks.0.v_attn.to_k.weight", "speaker_proj.weight"):
                    helpers.assert_nonzero_finite(self, params[name].grad, name)
                outputs.append(out.detach())
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            expected_params = dict(plain.named_parameters())
            for name, parameter in checked.named_parameters():
                expected = expected_params[name].grad
                self.assertEqual(parameter.grad is None, expected is None, name)
                if expected is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
