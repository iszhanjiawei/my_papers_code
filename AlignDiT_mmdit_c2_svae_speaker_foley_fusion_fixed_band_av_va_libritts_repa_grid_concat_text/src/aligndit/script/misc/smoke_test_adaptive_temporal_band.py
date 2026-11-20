"""CPU regression contracts for the isolated adaptive temporal-band experiment.

Run from this experiment's root, without loading data or using any GPU::

    PYTHONPATH=src python src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py

The optional legacy check reads the neighboring original backbone under an
isolated module name; it never edits the original or changes the import path.
Nonzero warm-start-style gates make output/gradient checks non-vacuous.
"""

from __future__ import annotations

import io
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from aligndit.model.backbone.dit_vt_mm import (
    AdaptiveTemporalBand,
    DiT_VT_MMDiT,
    MMDiTBlock_VT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LEGACY_BACKBONE = (
    PROJECT_ROOT.parent
    / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding"
    / "src/aligndit/model/backbone/dit_vt_mm.py"
)
BASE_ARCH = {
    "dim": 64,
    "depth": 4,
    "heads": 4,
    "dim_head": 16,
    "dropout": 0.0,
    "ff_mult": 2,
    "mel_dim": 64,
    "text_num_embeds": 16,
    "text_dim": 32,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 1,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "checkpoint_activations": False,
    "use_conformer": False,
    "layer_indices_ctc": [0, 1],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 2,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
    "normalize_text_context": True,
    "speaker_dim": 12,
    "speaker_condition_start_layer": 2,
}
BAND_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_hidden_dim": 16,
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_max_offset_seconds": 0.1,
    "temporal_band_min_sigma_seconds": 0.025,
    "temporal_band_max_sigma_seconds": 0.25,
    "temporal_band_init_sigma_seconds": 0.1,
}


def make_inputs():
    audio_mask = torch.arange(12)[None, :] < torch.tensor([12, 9])[:, None]
    video_mask = torch.arange(12)[None, :] < torch.tensor([12, 8])[:, None]
    text_mask = torch.arange(4)[None, :] < torch.tensor([4, 3])[:, None]
    generation_mask = audio_mask.clone()
    generation_mask[:, :3] = False
    return {
        "x": torch.randn(2, 12, 64),
        "cond": torch.randn(2, 12, 64),
        "text": torch.randint(0, 16, (2, 4)).masked_fill(~text_mask, -1),
        "video": torch.randn(2, 12, 16),
        "time": torch.tensor([0.2, 0.8]),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": video_mask,
        "complementary_mask": video_mask & ~generation_mask,
        "generation_mask": generation_mask,
        "speaker_embedding": torch.randn(2, 12),
        "cache": False,
    }


def warm_start(model):
    """Simulate trained parent residuals without modifying band initialization."""
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
            if hasattr(block, "cross_attn_ada"):
                block.cross_attn_ada.weight.normal_(std=0.03)
                block.cross_attn_ada.bias.normal_(std=0.03)
                block.v_attn_norm.linear.weight.normal_(std=0.03)
                block.v_attn_norm.linear.bias.normal_(std=0.03)
        model.proj_out.weight.normal_(std=0.03)
        model.norm_out.linear.weight.normal_(std=0.03)
        model.speaker_proj.weight.normal_(std=0.03)
    return model


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, **overrides}))


def last_linear(module):
    return [child for child in module.modules() if isinstance(child, nn.Linear)][-1]


def make_predictor_content_sensitive(module):
    # Initialization deliberately has a content-independent output head. Use a
    # trained-style head when checking that CFG branches cannot leak content.
    with torch.no_grad():
        last_linear(module).weight.normal_(std=0.04)


def assert_nonzero_finite(test, tensor, label):
    test.assertIsNotNone(tensor, f"{label}: missing gradient/tensor")
    test.assertTrue(torch.isfinite(tensor).all().item(), f"{label}: non-finite values")
    test.assertGreater(torch.count_nonzero(tensor).item(), 0, f"{label}: all zeros")


def captured_attention(block, audio, video, **kwargs):
    """Capture either one full SDPA call or split audio/video query calls."""
    calls = []
    original_sdpa = F.scaled_dot_product_attention

    def record(query, key, value, *args, **sdpa_kwargs):
        raw_mask = sdpa_kwargs.get("attn_mask", args[0] if args else None)
        shape = (*query.shape[:-1], key.shape[-2])
        if raw_mask is None:
            additive_mask = torch.zeros(shape, dtype=torch.float32, device=query.device)
        elif raw_mask.dtype == torch.bool:
            additive_mask = torch.zeros(shape, dtype=torch.float32, device=query.device)
            additive_mask = additive_mask.masked_fill(~raw_mask, -float("inf"))
        else:
            additive_mask = raw_mask.float().expand(shape)
        calls.append((query.shape[-2], additive_mask.detach().clone()))
        return original_sdpa(query, key, value, *args, **sdpa_kwargs)

    with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=record):
        outputs = block.joint_attn(audio, video, **kwargs)
    if len(calls) == 1:
        return outputs, calls[0][1]
    if len(calls) == 2 and [call[0] for call in calls] == [audio.shape[1], video.shape[1]]:
        return outputs, torch.cat([call[1] for call in calls], dim=-2)
    raise AssertionError(f"unexpected joint-attention query partition: {[call[0] for call in calls]}")


class TemporalBandFormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)

    def make_band(self, **kwargs):
        return AdaptiveTemporalBand(dim=64, hidden_dim=16, **kwargs)

    def test_initialization_and_physical_time_formula(self):
        band = self.make_band(audio_fps=40.0, video_fps=25.0)
        offset, sigma = band(torch.randn(2, 7, 64), audio_len=10)
        self.assertEqual(offset.shape, (2, 10))
        self.assertEqual(sigma.shape, (2, 10))
        torch.testing.assert_close(offset, torch.zeros_like(offset), atol=0, rtol=0)
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=1e-7, rtol=0)
        bias = band.bias(offset, sigma, video_len=7)
        audio_time = torch.arange(10).float() / 40.0
        video_time = torch.arange(7).float() / 25.0
        expected = -0.5 * ((video_time[None, None, :] - audio_time[None, :, None]) / 0.1).square()
        self.assertEqual(bias.dtype, torch.float32)
        torch.testing.assert_close(bias, expected.expand(2, -1, -1), atol=1e-6, rtol=1e-6)

    def test_positive_offset_moves_peak_to_later_video(self):
        band = self.make_band()
        offset = torch.full((1, 9), 0.05)
        sigma = torch.full_like(offset, 0.1)
        bias = band.bias(offset, sigma, video_len=12)
        self.assertEqual(bias[0, 4].argmax().item(), 6)
        self.assertAlmostEqual(bias[0, 4, 6].item(), 0.0, places=12)
        self.assertLess(bias[0, 4, 4].item(), bias[0, 4, 6].item())

    def test_offset_and_width_bounds_under_extreme_head_values(self):
        band = self.make_band(
            max_offset_seconds=0.12, min_sigma_seconds=0.03, max_sigma_seconds=0.22, init_sigma_seconds=0.08
        )
        head = last_linear(band)
        self.assertEqual(head.out_features, 2)
        with torch.no_grad():
            head.weight.zero_()
            for value in (-100.0, 100.0):
                head.bias.fill_(value)
                offset, sigma = band(torch.randn(2, 7, 64), audio_len=9)
                self.assertTrue(torch.isfinite(offset).all())
                self.assertTrue(torch.isfinite(sigma).all())
                self.assertTrue((offset.abs() <= 0.12 + 1e-7).all())
                self.assertTrue((sigma >= 0.03 - 1e-7).all())
                self.assertTrue((sigma <= 0.22 + 1e-7).all())

    def test_both_geometry_parameters_receive_gradients(self):
        band = self.make_band()
        offset, sigma = band(torch.randn(2, 7, 64), audio_len=10)
        offset.retain_grad()
        sigma.retain_grad()
        bias = band.bias(offset, sigma, video_len=7)
        (bias * torch.randn_like(bias)).sum().backward()
        assert_nonzero_finite(self, offset.grad, "offset")
        assert_nonzero_finite(self, sigma.grad, "sigma")
        head_gradient = last_linear(band).weight.grad
        for index, label in enumerate(("offset output head", "width output head")):
            assert_nonzero_finite(self, head_gradient[index], label)

    def test_no_cross_sample_predictor_leak_and_single_video_frame(self):
        band = self.make_band(audio_fps=40.0, video_fps=25.0)
        make_predictor_content_sensitive(band)
        first = torch.randn(1, 1, 64)
        joined = torch.cat((first, torch.randn_like(first) * 5), dim=0)
        individual = band(first, audio_len=9)
        batched = band(joined, audio_len=9)
        for expected, actual in zip(individual, batched):
            torch.testing.assert_close(expected, actual[:1], atol=1e-7, rtol=1e-6)
            self.assertTrue(torch.isfinite(actual).all())

    def test_interpolation_uses_physical_grid_not_padded_length_stretch(self):
        band = self.make_band(audio_fps=40.0, video_fps=25.0)
        video = torch.randn(2, 4, 64)
        normalized_inputs = []
        first_layer = next(child for child in band.modules() if isinstance(child, nn.Linear))
        handle = first_layer.register_forward_pre_hook(
            lambda _module, args: normalized_inputs.append(args[0].detach().clone())
        )
        try:
            band(video, audio_len=9)
        finally:
            handle.remove()
        # At audio frame 2 (50 ms), interpolate video frames 1/2 (40/80 ms)
        # with weight 1/4. At 200 ms the grid clamps to video frame 3.
        expected_midpoint = F.layer_norm(video[:, 1] * 0.75 + video[:, 2] * 0.25, (64,), eps=1e-6)
        expected_endpoint = F.layer_norm(video[:, 3], (64,), eps=1e-6)
        torch.testing.assert_close(normalized_inputs[0][:, 2], expected_midpoint)
        torch.testing.assert_close(normalized_inputs[0][:, 8], expected_endpoint)

    def test_cpu_bfloat16_autocast_keeps_geometry_float32_and_differentiable(self):
        band = self.make_band()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            offset, sigma = band(torch.randn(2, 7, 64), audio_len=10)
            bias = band.bias(offset, sigma, video_len=7)
            loss = (bias * torch.randn_like(bias)).mean()
        for value in (offset, sigma, bias):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
        loss.backward()
        gradient = last_linear(band).weight.grad
        assert_nonzero_finite(self, gradient[0], "BF16 offset head")
        assert_nonzero_finite(self, gradient[1], "BF16 width head")

    def test_invalid_geometry_configuration_is_rejected(self):
        for invalid in (
            {"audio_fps": 0},
            {"video_fps": -1},
            {"audio_fps": float("nan")},
            {"max_offset_seconds": -0.1},
            {"min_sigma_seconds": 0},
            {"init_sigma_seconds": 0.025},
            {"max_sigma_seconds": 0.09},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.make_band(**invalid)


class JointAttentionBandTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(394)
        self.block = MMDiTBlock_VT(
            dim=64, heads=4, dim_head=16, text_dim=32, dropout=0.0, attn_mask_enabled=True, prompt_isolated_ca=False
        )
        self.block.eval()
        self.audio = torch.randn(2, 7, 64)
        self.video = torch.randn(2, 5, 64)
        self.bias = torch.randn(2, 7, 5).square().neg()

    def test_only_audio_query_video_key_logits_change_without_masks(self):
        baseline, before = captured_attention(self.block, self.audio, self.video)
        changed, after = captured_attention(self.block, self.audio, self.video, temporal_band_bias=self.bias)
        difference = after - before
        torch.testing.assert_close(difference[:, :, :7, 7:], self.bias[:, None].expand(-1, 4, -1, -1))
        self.assertEqual(torch.count_nonzero(difference[:, :, :7, :7]).item(), 0)
        self.assertEqual(torch.count_nonzero(difference[:, :, 7:, :]).item(), 0)
        torch.testing.assert_close(baseline[1], changed[1], atol=1e-7, rtol=1e-6)
        self.assertGreater((baseline[0] - changed[0]).abs().max().item(), 1e-5)

    def test_mixed_lengths_mask_invalid_keys_and_outputs(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        video_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        (out_a, out_v), logits = captured_attention(
            self.block,
            self.audio,
            self.video,
            mask=mask,
            v_mask=video_mask,
            temporal_band_bias=self.bias,
        )
        valid_keys = torch.cat((mask, video_mask), dim=1)
        self.assertTrue(torch.isneginf(logits.masked_select(~valid_keys[:, None, None, :])).all())
        self.assertTrue(torch.isfinite(logits.masked_select(valid_keys[:, None, None, :])).all())
        self.assertEqual(torch.count_nonzero(out_a[~mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(out_v[~video_mask]).item(), 0)
        self.assertTrue(torch.isfinite(out_a).all() and torch.isfinite(out_v).all())

    def test_all_invalid_video_still_has_finite_audio_path(self):
        mask = torch.ones(2, 7, dtype=torch.bool)
        video_mask = torch.zeros(2, 5, dtype=torch.bool)
        out_a, out_v = self.block.joint_attn(
            self.audio,
            self.video,
            mask=mask,
            v_mask=video_mask,
            temporal_band_bias=self.bias,
        )
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertEqual(torch.count_nonzero(out_v).item(), 0)

    def test_video_only_mask_still_masks_video_keys(self):
        video_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        (out_a, out_v), logits = captured_attention(
            self.block,
            self.audio,
            self.video,
            v_mask=video_mask,
            temporal_band_bias=self.bias,
        )
        self.assertTrue(torch.isneginf(logits[1, :, :, -2:]).all())
        self.assertTrue(torch.isfinite(logits[:, :, :, :7]).all())
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertEqual(torch.count_nonzero(out_v[~video_mask]).item(), 0)

    def test_bias_gradients_survive_absent_training_masks(self):
        self.block.train()
        bias = self.bias.clone().requires_grad_()
        out_a, _ = self.block.joint_attn(self.audio, self.video, temporal_band_bias=bias)
        (out_a * torch.randn_like(out_a)).sum().backward()
        assert_nonzero_finite(self, bias.grad, "training attention bias without masks")


class BackboneBandIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_disabled_has_legacy_keys_and_nontrivial_outputs(self):
        if not LEGACY_BACKBONE.is_file():
            self.skipTest(f"optional original snapshot unavailable: {LEGACY_BACKBONE}")
        legacy_module = types.ModuleType("_adaptive_band_readonly_legacy_backbone")
        legacy_module.__file__ = str(LEGACY_BACKBONE)
        # Trusted neighboring source only; compile avoids writing legacy pycache.
        exec(compile(LEGACY_BACKBONE.read_text(), str(LEGACY_BACKBONE), "exec"), legacy_module.__dict__)  # noqa: S102
        legacy = warm_start(legacy_module.DiT_VT_MMDiT(**BASE_ARCH))
        disabled = DiT_VT_MMDiT(**BASE_ARCH, temporal_band_enabled=False)
        self.assertIsNone(disabled.temporal_band)
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        disabled.load_state_dict(legacy.state_dict(), strict=True)
        inputs = make_inputs()
        with torch.no_grad():
            for training in (False, True):
                legacy.train(training)
                disabled.train(training)
                for cfg_infer in (False, True):
                    expected, expected_ctc = legacy(**inputs, cfg_infer=cfg_infer)
                    actual, actual_ctc = disabled(**inputs, cfg_infer=cfg_infer)
                    assert_nonzero_finite(self, expected, "legacy output")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], expected_ctc[layer][key], atol=0, rtol=0)

    def test_enabled_adds_only_predictor_parameters_and_initialization_survives(self):
        disabled = DiT_VT_MMDiT(**BASE_ARCH)
        enabled = DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH)
        new_keys = set(enabled.state_dict()) - set(disabled.state_dict())
        self.assertTrue(new_keys)
        self.assertTrue(all(key.startswith("temporal_band.") for key in new_keys))
        missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)
        self.assertEqual(set(missing), new_keys)
        self.assertFalse(unexpected)
        offset, sigma = enabled.temporal_band(torch.randn(2, 12, 64), audio_len=12)
        torch.testing.assert_close(offset, torch.zeros_like(offset), atol=0, rtol=0)
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=1e-7, rtol=0)

    def test_cfg_packed_equals_separate_for_distinct_batch_members(self):
        model = make_model().eval()
        make_predictor_content_sensitive(model.temporal_band)
        inputs = make_inputs()
        cases = (
            ({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            (
                {"drop_video": True},
                ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True}),
            ),
            (
                {"drop_text": True},
                ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True}),
            ),
        )
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                with self.subTest(packed_flags=packed_flags):
                    packed, packed_ctc = model(**inputs, cfg_infer=True, **packed_flags)
                    packed_offset = model.last_temporal_band_offset_seconds.clone()
                    packed_sigma = model.last_temporal_band_sigma_seconds.clone()
                    pieces, offsets, widths, ctc_pieces = [], [], [], []
                    for flags in separate_flags:
                        piece, ctc = model(**inputs, **flags)
                        pieces.append(piece)
                        offsets.append(model.last_temporal_band_offset_seconds.clone())
                        widths.append(model.last_temporal_band_sigma_seconds.clone())
                        ctc_pieces.append(ctc)
                    torch.testing.assert_close(packed, torch.cat(pieces), atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(packed_offset, torch.cat(offsets), atol=1e-7, rtol=1e-6)
                    torch.testing.assert_close(packed_sigma, torch.cat(widths), atol=1e-7, rtol=1e-6)
                    for layer in packed_ctc:
                        torch.testing.assert_close(
                            packed_ctc[layer]["z_tilde"],
                            torch.cat([ctc[layer]["z_tilde"] for ctc in ctc_pieces]),
                            atol=3e-6,
                            rtol=3e-5,
                        )

    def test_null_video_branches_do_not_see_real_video_or_batch_neighbors(self):
        model = make_model().eval()
        make_predictor_content_sensitive(model.temporal_band)
        inputs = make_inputs()
        altered_inputs = {**inputs, "video": inputs["video"].clone()}
        altered_inputs["video"][0] = torch.randn_like(inputs["video"][0]) * 7
        with torch.no_grad():
            original, _ = model(**inputs, cfg_infer=True)
            original_offset = model.last_temporal_band_offset_seconds.clone()
            original_sigma = model.last_temporal_band_sigma_seconds.clone()
            altered, _ = model(**altered_inputs, cfg_infer=True)
            altered_offset = model.last_temporal_band_offset_seconds.clone()
            altered_sigma = model.last_temporal_band_sigma_seconds.clone()
        # Branch-major layout: [full sample0, full sample1, TTS0, TTS1, null0, null1].
        for expected, actual in (
            (original, altered),
            (original_offset, altered_offset),
            (original_sigma, altered_sigma),
        ):
            torch.testing.assert_close(expected[1:], actual[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - altered[0]).abs().max().item(), 1e-5)
        self.assertGreater((original_offset[0] - altered_offset[0]).abs().max().item(), 1e-7)

    def test_full_training_backward_checkpointing_parity(self):
        plain = make_model(checkpoint_activations=False).train()
        checked = DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, "checkpoint_activations": True}).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs = make_inputs()
        target = torch.randn(2, 12, 64)
        outputs = []
        for model in (plain, checked):
            output, ctc = model(**inputs)
            self.assertEqual(output.shape, (2, 12, 64))
            torch.testing.assert_close(ctc[0]["z_lens"], torch.tensor([12, 9]))
            loss = (output - target).square().mean()
            loss = loss + 0.01 * sum(value["z_tilde"].square().mean() for value in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradient = last_linear(model.temporal_band).weight.grad
            assert_nonzero_finite(self, gradient[0], "offset head model gradient")
            assert_nonzero_finite(self, gradient[1], "width head model gradient")
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected_gradient = plain_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected_gradient is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, expected_gradient, atol=1e-7, rtol=1e-5, msg=name)

    def test_save_load_and_ema_preserve_band(self):
        from ema_pytorch import EMA

        model = make_model().eval()
        make_predictor_content_sensitive(model.temporal_band)
        ema = EMA(model, beta=0.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = make_inputs()
        with torch.no_grad():
            expected, _ = model(**inputs)
            expected_ema, _ = ema.ema_model(**inputs)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        restored = make_model().eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_ema = EMA(restored, beta=0.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(state["ema"], strict=True)
        with torch.no_grad():
            actual, _ = restored(**inputs)
            actual_ema, _ = restored_ema.ema_model(**inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)
        band_ema_keys = [key for key in state["ema"] if "temporal_band." in key]
        self.assertTrue(band_ema_keys)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
