"""CPU regression contracts for the isolated fixed AV+VA temporal prior.

Run from this snapshot's root; no data, checkpoints, or GPUs are needed::

    PYTHONPATH=src python src/aligndit/script/misc/smoke_test_fixed_temporal_band_bidirectional.py

The reference explicitly forms the full joint-attention matrix and applies the
physical-time formula independently. Production uses split queries and a shared
head dimension instead. Tiny-model helpers come only from this snapshot; the
inherited AV-only and adaptive tests remain unchanged.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import validate_fixed_band_contract
from aligndit.script.misc import smoke_test_adaptive_temporal_band as helpers
from aligndit.script.misc.smoke_test_fixed_temporal_band import FIXED_ARCH


BIDIRECTIONAL_ARCH = {**FIXED_ARCH, "temporal_band_bidirectional": True}


def make_model(**overrides):
    return helpers.warm_start(
        helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **BIDIRECTIONAL_ARCH, **overrides})
    )


class BidirectionalJointAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(394)
        self.block = helpers.MMDiTBlock_VT(
            dim=64, heads=4, dim_head=16, text_dim=32, dropout=0.0,
            attn_mask_enabled=True, prompt_isolated_ca=False,
            temporal_band_bidirectional=True,
        ).eval()
        self.audio = torch.randn(2, 7, 64)
        self.video = torch.randn(2, 5, 64)
        self.audio_mask = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
        self.video_mask = torch.arange(5)[None] < torch.tensor([5, 3])[:, None]
        self.band = FixedTemporalBand(
            dim=64, audio_fps=40, video_fps=25, offset_seconds=0.05, sigma_seconds=0.08
        )
        self.bias = self.band.bias(*self.band(self.video, audio_len=7), video_len=5)

    def full_matrix_reference(self, mask=None, video_mask=None):
        """Independent logits/softmax reference, not the production SDPA path."""
        block = self.block
        q_a, k_a, v_a = block._qkv(block.attn, self.audio)
        q_v, k_v, v_v = block._qkv(block.v_attn, self.video)
        query = torch.cat((q_a, q_v), dim=2)
        key = torch.cat((k_a, k_v), dim=2)
        value = torch.cat((v_a, v_v), dim=2)
        logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        # Positive offset means later video keys for each audio query. The
        # reverse block represents the SAME pair relation, not a new Gaussian.
        a_time = torch.arange(7).float() / 40
        v_time = torch.arange(5).float() / 25
        pair_bias = -0.5 * ((v_time[None, :] - a_time[:, None] - 0.05) / 0.08).square()
        prior = torch.zeros_like(logits)
        prior[:, :, :7, 7:] = pair_bias
        prior[:, :, 7:, :7] = pair_bias.T
        if block.attn_mask_enabled and (mask is not None or video_mask is not None):
            audio_valid = torch.ones(2, 7, dtype=torch.bool) if mask is None else mask
            video_valid = torch.ones(2, 5, dtype=torch.bool) if video_mask is None else video_mask
            valid_keys = torch.cat((audio_valid, video_valid), dim=1)
            prior = prior.masked_fill(~valid_keys[:, None, None, :], -float("inf"))
        attended = (logits + prior).softmax(dim=-1) @ value
        attended = attended.transpose(1, 2).reshape(2, 12, 64)
        out_a = block.attn.to_out[1](block.attn.to_out[0](attended[:, :7]))
        out_v = block.v_attn.to_out[1](block.v_attn.to_out[0](attended[:, 7:]))
        if mask is not None:
            out_a = out_a.masked_fill(~mask[..., None], 0)
        if video_mask is not None:
            out_v = out_v.masked_fill(~video_mask[..., None], 0)
        return (out_a, out_v), prior

    def test_full_softmax_reference_unequal_lengths_masks_train_eval(self):
        cases = ((None, None), (self.audio_mask, self.video_mask),
                 (self.audio_mask, None), (None, self.video_mask))
        for training in (False, True):
            for mask_enabled in (False, True):
                self.block.train(training)
                self.block.attn_mask_enabled = mask_enabled
                for audio_mask, video_mask in cases:
                    with self.subTest(training=training, mask_enabled=mask_enabled,
                                      audio_mask=audio_mask is not None, video_mask=video_mask is not None):
                        expected, prior = self.full_matrix_reference(audio_mask, video_mask)
                        actual, observed_prior = helpers.captured_attention(
                            self.block, self.audio, self.video, mask=audio_mask,
                            v_mask=video_mask, temporal_band_bias=self.bias,
                        )
                        torch.testing.assert_close(observed_prior, prior, atol=1e-7, rtol=1e-6)
                        for output, target in zip(actual, expected):
                            torch.testing.assert_close(output, target, atol=3e-7, rtol=3e-6)
                            self.assertTrue(torch.isfinite(output).all())

    def test_raw_logit_bias_only_changes_av_and_va_and_broadcasts_heads(self):
        _, observed = helpers.captured_attention(
            self.block, self.audio, self.video, temporal_band_bias=self.bias
        )
        torch.testing.assert_close(observed[:, :, :7, 7:], self.bias[:, None].expand(-1, 4, -1, -1))
        torch.testing.assert_close(
            observed[:, :, 7:, :7], self.bias.transpose(-1, -2)[:, None].expand(-1, 4, -1, -1)
        )
        self.assertEqual(torch.count_nonzero(observed[:, :, :7, :7]).item(), 0)
        self.assertEqual(torch.count_nonzero(observed[:, :, 7:, 7:]).item(), 0)

        masks = []
        original_sdpa = F.scaled_dot_product_attention

        def capture(query, key, value, **kwargs):
            masks.append(kwargs["attn_mask"].shape)
            return original_sdpa(query, key, value, **kwargs)

        with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=capture):
            self.block.joint_attn(self.audio, self.video, temporal_band_bias=self.bias)
        self.assertEqual(masks, [torch.Size([2, 1, 7, 12]), torch.Size([2, 1, 5, 12])])

    def test_nonzero_offset_reverse_is_transpose_not_same_sign_recomputation(self):
        band = FixedTemporalBand(dim=64, audio_fps=40, video_fps=40,
                                 offset_seconds=0.05, sigma_seconds=0.04)
        bias = band.bias(*band(self.video, audio_len=7), video_len=5)
        _, observed = helpers.captured_attention(self.block, self.audio, self.video, temporal_band_bias=bias)
        # Audio frame 2 is paired with video frame 4. Reversing the query/key
        # roles must preserve this peak; recomputing with +offset would not.
        reverse = observed[0, 0, 7:, :7]
        self.assertEqual(bias[0, 2].argmax().item(), 4)
        self.assertEqual(reverse[4].argmax().item(), 2)
        self.assertAlmostEqual(reverse[4, 2].item(), 0.0, places=10)
        wrong_reverse = -0.5 * (
            ((torch.arange(7)[None] / 40) - (torch.arange(5)[:, None] / 40) - 0.05) / 0.04
        ).square()
        self.assertGreater((reverse - wrong_reverse).abs().max().item(), 1.0)

    def test_av_only_audio_output_is_unchanged_but_video_output_changes(self):
        av_va = self.block.joint_attn(self.audio, self.video, temporal_band_bias=self.bias)
        self.block.temporal_band_bidirectional = False
        av_only = self.block.joint_attn(self.audio, self.video, temporal_band_bias=self.bias)
        torch.testing.assert_close(av_va[0], av_only[0], atol=0, rtol=0)
        self.assertGreater((av_va[1] - av_only[1]).abs().max().item(), 1e-5)
        # This asserts raw AA/VV logits are untouched, not probabilities: a
        # changed cross-modal normalizer legitimately changes AA/VV weights.

    def test_bidirectional_flag_without_bias_keeps_unbanded_attention_identical(self):
        expected = self.block.joint_attn(self.audio, self.video, mask=self.audio_mask, v_mask=self.video_mask)
        self.block.temporal_band_bidirectional = False
        actual = self.block.joint_attn(self.audio, self.video, mask=self.audio_mask, v_mask=self.video_mask)
        for output, target in zip(actual, expected):
            torch.testing.assert_close(output, target, atol=0, rtol=0)

    def test_all_invalid_video_keys_keep_audio_finite_and_video_queries_zero(self):
        out_a, out_v = self.block.joint_attn(
            self.audio, self.video, mask=self.audio_mask, v_mask=torch.zeros_like(self.video_mask),
            temporal_band_bias=self.bias,
        )
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertEqual(torch.count_nonzero(out_a[~self.audio_mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(out_v).item(), 0)


class BidirectionalBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_direction_changes_no_parameters_state_keys_or_rng(self):
        models = []
        rng_states = []
        for direction in (None, False, True):
            torch.manual_seed(83)
            architecture = {**helpers.BASE_ARCH, **FIXED_ARCH}
            if direction is not None:
                architecture["temporal_band_bidirectional"] = direction
            models.append(helpers.DiT_VT_MMDiT(**architecture))
            rng_states.append(torch.get_rng_state().clone())
        expected = models[0]
        for model, rng in zip(models[1:], rng_states[1:]):
            torch.testing.assert_close(rng, rng_states[0], atol=0, rtol=0)
            self.assertEqual(set(model.state_dict()), set(expected.state_dict()))
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in expected.parameters()))
            self.assertEqual(list(model.temporal_band.parameters()), [])
            self.assertEqual(dict(model.temporal_band.state_dict()), {})
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[key], atol=0, rtol=0, msg=key)
            model.load_state_dict(expected.state_dict(), strict=True)
        self.assertFalse(models[0].temporal_band_bidirectional)
        self.assertFalse(models[1].temporal_band_bidirectional)
        self.assertTrue(models[2].temporal_band_bidirectional)
        for model in models:
            for block in model.transformer_blocks[:2]:
                self.assertEqual(block.temporal_band_bidirectional, model.temporal_band_bidirectional)

    def test_omitted_direction_is_exactly_explicit_av_only(self):
        omitted = helpers.warm_start(helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **FIXED_ARCH}))
        av_only = make_model(temporal_band_bidirectional=False)
        av_only.load_state_dict(omitted.state_dict(), strict=True)
        inputs = helpers.make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for cfg in (False, True):
                    expected, expected_ctc = omitted.train(training)(**inputs, cfg_infer=cfg)
                    actual, actual_ctc = av_only.train(training)(**inputs, cfg_infer=cfg)
                    helpers.assert_nonzero_finite(self, actual, "AV-only output")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for field in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][field], expected_ctc[layer][field], atol=0, rtol=0)

    def test_requires_enabled_fixed_geometry(self):
        for overrides in ({"temporal_band_enabled": False},
                          {"temporal_band_mode": "adaptive"},
                          {"temporal_band_enabled": False, "temporal_band_mode": "adaptive"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **BIDIRECTIONAL_ARCH, **overrides})

    def test_nonboolean_direction_is_rejected(self):
        for value in (0, 1, "false", "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "temporal_band_bidirectional"):
                helpers.DiT_VT_MMDiT(
                    **{**helpers.BASE_ARCH, **BIDIRECTIONAL_ARCH, "temporal_band_bidirectional": value}
                )

    def test_cfg_packed_matches_separate_and_null_video_does_not_leak(self):
        model = make_model().eval()
        inputs = helpers.make_inputs()
        cases = (
            ({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_video": True}, ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_text": True}, ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
        )
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                with self.subTest(packed_flags=packed_flags):
                    packed, ctc = model(**inputs, cfg_infer=True, **packed_flags)
                    offset = model.last_temporal_band_offset_seconds.clone()
                    sigma = model.last_temporal_band_sigma_seconds.clone()
                    torch.testing.assert_close(offset, torch.zeros_like(offset), atol=0, rtol=0)
                    torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=0, rtol=0)
                    separate = [model(**inputs, **flags) for flags in separate_flags]
                    torch.testing.assert_close(packed, torch.cat([item[0] for item in separate]), atol=3e-6, rtol=3e-5)
                    for layer in ctc:
                        for field in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(
                                ctc[layer][field], torch.cat([item[1][layer][field] for item in separate]),
                                atol=3e-6, rtol=3e-5,
                            )
            original, _ = model(**inputs, cfg_infer=True)
            changed_video = inputs["video"].clone()
            changed_video[0] = torch.randn_like(changed_video[0]) * 7
            changed, _ = model(**{**inputs, "video": changed_video}, cfg_infer=True)
        # Branch-major [full 0/1, no-video 0/1, null 0/1].
        torch.testing.assert_close(original[1:], changed[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - changed[0]).abs().max().item(), 1e-5)

    def test_checkpointed_backward_matches_and_reaches_both_streams(self):
        plain = make_model(checkpoint_activations=False).train()
        checked = make_model(checkpoint_activations=True).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs = helpers.make_inputs()
        target = torch.randn(2, 12, 64)
        outputs = []
        for model in (plain, checked):
            output, ctc = model(**inputs)
            loss = (output - target).square().mean()
            loss = loss + 0.01 * sum(item["z_tilde"].square().mean() for item in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            parameters = dict(model.named_parameters())
            for name in ("transformer_blocks.0.attn.to_q.weight",
                         "transformer_blocks.0.v_attn.to_q.weight",
                         "transformer_blocks.0.v_attn.to_k.weight", "speaker_proj.weight"):
                helpers.assert_nonzero_finite(self, parameters[name].grad, name)
            self.assertEqual(list(model.temporal_band.parameters()), [])
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = plain_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)
        av_only = make_model(temporal_band_bidirectional=False).eval()
        av_only.load_state_dict(plain.state_dict(), strict=True)
        with torch.no_grad():
            av_only_output, _ = av_only(**inputs)
            av_va_output, _ = plain.eval()(**inputs)
        self.assertGreater((av_only_output - av_va_output).abs().max().item(), 1e-6)

    def test_model_and_ema_roundtrip_with_explicit_direction_configuration(self):
        from ema_pytorch import EMA

        model = make_model().eval()
        ema = EMA(model, beta=0.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = helpers.make_inputs()
        with torch.no_grad():
            expected, _ = model(**inputs)
            expected_ema, _ = ema.ema_model(**inputs)
        stream = io.BytesIO()
        torch.save({"architecture": BIDIRECTIONAL_ARCH, "model": model.state_dict(), "ema": ema.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        self.assertFalse(any("temporal_band" in key for key in state["model"]))
        restored = make_model(**state["architecture"]).eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_ema = EMA(restored, beta=0.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(state["ema"], strict=True)
        with torch.no_grad():
            actual, _ = restored(**inputs)
            actual_ema, _ = restored_ema.ema_model(**inputs)
        self.assertTrue(restored.temporal_band_bidirectional)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)


class BidirectionalInferenceContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fixed-av-va-contract-test-")
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name) / "model_100.pt"
        self.sidecar = self.checkpoint.parent / "speaker_training_contract.json"

    def config(self, direction):
        architecture = dict(FIXED_ARCH)
        if direction is not None:
            architecture["temporal_band_bidirectional"] = direction
        return types.SimpleNamespace(model=types.SimpleNamespace(arch=architecture))

    def write_contract(self, direction):
        parameters = self.config(direction).model.arch
        self.sidecar.write_text(json.dumps({"temporal_band": {
            "mode": "fixed", "parameter_count": 0, "parameters": parameters,
        }}), encoding="utf-8")

    def test_missing_legacy_direction_is_false_and_matching_contracts_pass(self):
        for recorded, requested in ((None, None), (None, False), (False, None), (False, False), (True, True)):
            with self.subTest(recorded=recorded, requested=requested):
                self.write_contract(recorded)
                self.assertIsNone(validate_fixed_band_contract(self.config(requested), self.checkpoint))

    def test_av_only_and_av_va_contracts_reject_each_other(self):
        for recorded, requested in ((None, True), (False, True), (True, None), (True, False)):
            with self.subTest(recorded=recorded, requested=requested):
                self.write_contract(recorded)
                with self.assertRaisesRegex(RuntimeError, "temporal_band_bidirectional"):
                    validate_fixed_band_contract(self.config(requested), self.checkpoint)

    def test_nonboolean_direction_rejected_in_config_and_contract(self):
        for value in (0, 1, "false", "true", None):
            for malformed_side in ("config", "contract"):
                with self.subTest(value=value, malformed_side=malformed_side):
                    self.write_contract(True)
                    config = self.config(True)
                    if malformed_side == "config":
                        config.model.arch["temporal_band_bidirectional"] = value
                    else:
                        contract = json.loads(self.sidecar.read_text(encoding="utf-8"))
                        contract["temporal_band"]["parameters"]["temporal_band_bidirectional"] = value
                        self.sidecar.write_text(json.dumps(contract), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "temporal_band_bidirectional must be a bool"):
                        validate_fixed_band_contract(config, self.checkpoint)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
