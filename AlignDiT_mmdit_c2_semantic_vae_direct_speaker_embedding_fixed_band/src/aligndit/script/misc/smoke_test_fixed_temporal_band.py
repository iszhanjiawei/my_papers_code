"""CPU contracts for the isolated fixed-center, fixed-width time-band ablation.

Run from this snapshot's root without data, checkpoints, or GPUs::

    PYTHONPATH=src python src/aligndit/script/misc/smoke_test_fixed_temporal_band.py

Tiny models and input helpers are reused from this snapshot's adaptive tests.
No neighboring experiment is imported or read. Nonzero warm-start-style gates
ensure the attention and gradient comparisons are not vacuous.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import torch

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import validate_fixed_band_contract
from aligndit.script.misc import smoke_test_adaptive_temporal_band as adaptive_tests


FIXED_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "fixed",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.1,
}


def make_model(**overrides):
    return adaptive_tests.warm_start(
        adaptive_tests.DiT_VT_MMDiT(**{**adaptive_tests.BASE_ARCH, **FIXED_ARCH, **overrides})
    )


class FixedGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)

    def test_fixed_values_physical_time_and_positive_offset_sign(self):
        band = FixedTemporalBand(dim=64, audio_fps=40.0, video_fps=25.0, offset_seconds=0.02, sigma_seconds=0.08)
        offset, sigma = band(torch.randn(2, 7, 64), audio_len=10)
        self.assertEqual(offset.shape, (2, 10))
        self.assertEqual(sigma.shape, (2, 10))
        torch.testing.assert_close(offset, torch.full_like(offset, 0.02), atol=0, rtol=0)
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.08), atol=0, rtol=0)
        audio_time = torch.arange(10).float() / 40.0
        video_time = torch.arange(7).float() / 25.0
        expected = -0.5 * ((video_time[None, None, :] - audio_time[None, :, None] - 0.02) / 0.08).square()
        actual = band.bias(offset, sigma, video_len=7)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected.expand(2, -1, -1), atol=1e-6, rtol=1e-6)

        positive = FixedTemporalBand(dim=64, offset_seconds=0.05)
        positive_bias = positive.bias(*positive(torch.randn(1, 12, 64), audio_len=9), video_len=12)
        self.assertEqual(positive_bias[0, 4].argmax().item(), 6)
        self.assertAlmostEqual(positive_bias[0, 4, 6].item(), 0.0, places=12)

    def test_zero_parameters_zero_state_and_no_rng_consumption(self):
        before = torch.get_rng_state().clone()
        band = FixedTemporalBand(dim=64)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        self.assertEqual(list(band.parameters()), [])
        self.assertEqual(dict(band.state_dict()), {})
        video = torch.randn(2, 7, 64, requires_grad=True)
        before = torch.get_rng_state().clone()
        offset, sigma = band(video, audio_len=10)
        bias = band.bias(offset, sigma, video_len=7)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        self.assertFalse(offset.requires_grad)
        self.assertFalse(sigma.requires_grad)
        self.assertFalse(bias.requires_grad)

    def test_content_length_batch_train_eval_and_dtype_independence(self):
        band = FixedTemporalBand(dim=64, offset_seconds=-0.025, sigma_seconds=0.125)
        first = torch.randn(1, 1, 64)
        expected = band(first, audio_len=9)
        for training in (False, True):
            band.train(training)
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(training=training, dtype=dtype), torch.autocast("cpu", dtype=torch.bfloat16):
                    # Includes non-finite content to prove geometry never reads
                    # semantic feature values; feature shape/device still matter.
                    video = torch.full((3, 13, 64), float("nan"), dtype=dtype)
                    actual = band(video, audio_len=9)
                    bias = band.bias(*actual, video_len=13)
                    for reference, value in zip(expected, actual):
                        self.assertEqual(value.dtype, torch.float32)
                        torch.testing.assert_close(value, reference.expand(3, -1), atol=0, rtol=0)
                    self.assertEqual(bias.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(bias).all())

    def test_default_matches_adaptive_initial_geometry_in_float32(self):
        fixed = FixedTemporalBand(dim=64, audio_fps=40, video_fps=25)
        adaptive = adaptive_tests.AdaptiveTemporalBand(dim=64, hidden_dim=16, audio_fps=40, video_fps=25)
        for video_len, audio_len in ((1, 9), (7, 10), (12, 12)):
            with self.subTest(video_len=video_len, audio_len=audio_len):
                video = torch.randn(2, video_len, 64)
                fixed_values = fixed(video, audio_len)
                adaptive_values = adaptive(video, audio_len)
                for actual, expected in zip(fixed_values, adaptive_values):
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(
                    fixed.bias(*fixed_values, video_len), adaptive.bias(*adaptive_values, video_len),
                    atol=3e-6, rtol=1e-6,
                )

    def test_invalid_configuration_and_shapes_fail_early(self):
        invalid_configurations = [
            {"dim": 0}, {"dim": -1}, {"audio_fps": 0}, {"video_fps": -1},
            {"sigma_seconds": 0}, {"sigma_seconds": -0.1},
        ]
        for field in ("audio_fps", "video_fps", "offset_seconds", "sigma_seconds"):
            invalid_configurations.extend({field: value} for value in (float("nan"), float("inf"), -float("inf")))
        for invalid in invalid_configurations:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FixedTemporalBand(**{"dim": 64, **invalid})
        band = FixedTemporalBand(dim=64)
        for video, audio_len in ((torch.zeros(2, 64), 9), (torch.zeros(2, 3, 16), 9),
                                 (torch.zeros(2, 0, 64), 9), (torch.zeros(2, 3, 64), 0)):
            with self.subTest(shape=tuple(video.shape), audio_len=audio_len), self.assertRaises(ValueError):
                band(video, audio_len)
        for offset, sigma, video_len in ((torch.zeros(2, 3, 1), torch.ones(2, 3, 1), 4),
                                         (torch.zeros(2, 3), torch.ones(2, 4), 4),
                                         (torch.zeros(2, 3), torch.ones(2, 3), 0)):
            with self.subTest(video_len=video_len), self.assertRaises(ValueError):
                band.bias(offset, sigma, video_len)


class FixedJointAttentionTests(adaptive_tests.JointAttentionBandTests):
    """Reuse AV-only/mixed-mask contracts with the actual fixed Gaussian bias."""

    def setUp(self):
        super().setUp()
        band = FixedTemporalBand(dim=64, audio_fps=40, video_fps=25, sigma_seconds=0.08)
        self.bias = band.bias(*band(self.video, audio_len=7), video_len=5)

    def test_explicit_softmax_reference_train_eval_without_padding_masks(self):
        # The independently constructed full matrix changes only A-query/V-key.
        # The temporal prior must also remain active when padding masks are off.
        for training in (False, True):
            for mask_enabled in (False, True):
                with self.subTest(training=training, mask_enabled=mask_enabled):
                    block = self.block.train(training)
                    block.attn_mask_enabled = mask_enabled
                    q_a, k_a, v_a = block._qkv(block.attn, self.audio)
                    q_v, k_v, v_v = block._qkv(block.v_attn, self.video)
                    query = torch.cat((q_a, q_v), dim=2)
                    key = torch.cat((k_a, k_v), dim=2)
                    value = torch.cat((v_a, v_v), dim=2)
                    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
                    full_bias = torch.zeros_like(logits)
                    full_bias[:, :, :7, 7:] = self.bias[:, None]
                    reference = (logits + full_bias).softmax(dim=-1) @ value
                    reference = reference.transpose(1, 2).reshape(2, 12, 64)
                    expected = (
                        block.attn.to_out[1](block.attn.to_out[0](reference[:, :7])),
                        block.v_attn.to_out[1](block.v_attn.to_out[0](reference[:, 7:])),
                    )
                    actual = block.joint_attn(self.audio, self.video, temporal_band_bias=self.bias)
                    baseline = block.joint_attn(self.audio, self.video)
                    for output, target in zip(actual, expected):
                        torch.testing.assert_close(output, target, atol=3e-7, rtol=3e-6)
                    torch.testing.assert_close(actual[1], baseline[1], atol=1e-7, rtol=1e-6)
                    self.assertGreater((actual[0] - baseline[0]).abs().max().item(), 1e-5)


class FixedInferenceContractTests(unittest.TestCase):
    """Weight keys alone cannot distinguish fixed-band and unbanded models."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fixed-band-contract-test-")
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "model_100.pt"
        self.sidecar = self.checkpoint.parent / "speaker_training_contract.json"

    def config(self, **overrides):
        return types.SimpleNamespace(model=types.SimpleNamespace(arch={**FIXED_ARCH, **overrides}))

    def write_contract(self, *, parameters=None, mode="fixed", parameter_count=0):
        contract = {"temporal_band": {
            "mode": mode,
            "parameter_count": parameter_count,
            "parameters": dict(FIXED_ARCH) if parameters is None else parameters,
        }}
        self.sidecar.write_text(json.dumps(contract), encoding="utf-8")

    def test_matching_fixed_sidecar_is_accepted(self):
        self.write_contract()
        self.assertIsNone(validate_fixed_band_contract(self.config(), self.checkpoint))

    def test_fixed_without_contract_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "matching fixed-band training contract"):
            validate_fixed_band_contract(self.config(), self.checkpoint)

    def test_mismatched_width_offset_and_frame_rates_are_rejected(self):
        self.write_contract()
        for field, value in (("temporal_band_fixed_sigma_seconds", 0.075),
                             ("temporal_band_fixed_offset_seconds", 0.025),
                             ("temporal_band_audio_fps", 25.0),
                             ("temporal_band_video_fps", 25.0)):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                validate_fixed_band_contract(self.config(**{field: value}), self.checkpoint)

    def test_recorded_adaptive_or_requested_adaptive_mode_is_rejected(self):
        self.write_contract(mode="adaptive")
        with self.assertRaisesRegex(RuntimeError, "matching fixed-band training contract"):
            validate_fixed_band_contract(self.config(), self.checkpoint)
        self.write_contract()
        with self.assertRaisesRegex(RuntimeError, "matching fixed-band training contract"):
            validate_fixed_band_contract(self.config(temporal_band_mode="adaptive"), self.checkpoint)
        self.write_contract(parameters={**FIXED_ARCH, "temporal_band_mode": "adaptive"})
        with self.assertRaisesRegex(RuntimeError, "temporal_band_mode"):
            validate_fixed_band_contract(self.config(), self.checkpoint)

    def test_recorded_fixed_requested_baseline_is_rejected(self):
        self.write_contract()
        for architecture in ({}, {**FIXED_ARCH, "temporal_band_enabled": False}):
            config = types.SimpleNamespace(model=types.SimpleNamespace(arch=architecture))
            with self.subTest(architecture=architecture), self.assertRaisesRegex(
                RuntimeError, "matching fixed-band training contract"
            ):
                validate_fixed_band_contract(config, self.checkpoint)

    def test_legacy_baseline_without_contract_is_accepted(self):
        for architecture in ({}, {"temporal_band_enabled": False},
                             {"temporal_band_enabled": True, "temporal_band_mode": "adaptive"}):
            config = types.SimpleNamespace(model=types.SimpleNamespace(arch=architecture))
            with self.subTest(architecture=architecture):
                self.assertIsNone(validate_fixed_band_contract(config, self.checkpoint))

    def test_incomplete_or_parameterized_fixed_contract_is_rejected(self):
        parameters = dict(FIXED_ARCH)
        del parameters["temporal_band_fixed_sigma_seconds"]
        self.write_contract(parameters=parameters)
        with self.assertRaisesRegex(RuntimeError, "temporal_band_fixed_sigma_seconds"):
            validate_fixed_band_contract(self.config(), self.checkpoint)
        self.write_contract(parameter_count=1)
        with self.assertRaisesRegex(RuntimeError, "zero temporal-band parameters"):
            validate_fixed_band_contract(self.config(), self.checkpoint)


class FixedBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_same_keys_parameters_rng_and_existing_weight_initialization(self):
        torch.manual_seed(83)
        disabled = adaptive_tests.DiT_VT_MMDiT(**adaptive_tests.BASE_ARCH)
        disabled_rng = torch.get_rng_state().clone()
        torch.manual_seed(83)
        enabled = adaptive_tests.DiT_VT_MMDiT(**{**adaptive_tests.BASE_ARCH, **FIXED_ARCH})
        torch.testing.assert_close(torch.get_rng_state(), disabled_rng, atol=0, rtol=0)
        self.assertIsInstance(enabled.temporal_band, FixedTemporalBand)
        self.assertIsNone(disabled.temporal_band)
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))
        self.assertEqual(sum(p.numel() for p in enabled.parameters()), sum(p.numel() for p in disabled.parameters()))
        for key, value in enabled.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], atol=0, rtol=0, msg=key)
        enabled.load_state_dict(disabled.state_dict(), strict=True)

    def test_disabled_fixed_is_identical_and_default_mode_stays_adaptive(self):
        baseline = make_model(temporal_band_enabled=False)
        disabled = make_model(temporal_band_enabled=False, temporal_band_fixed_sigma_seconds=0.15)
        disabled.load_state_dict(baseline.state_dict(), strict=True)
        self.assertIsNone(disabled.temporal_band)
        adaptive = adaptive_tests.DiT_VT_MMDiT(**{**adaptive_tests.BASE_ARCH, **adaptive_tests.BAND_ARCH})
        self.assertIsInstance(adaptive.temporal_band, adaptive_tests.AdaptiveTemporalBand)
        inputs = adaptive_tests.make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for cfg_infer in (False, True):
                    expected, expected_ctc = baseline.train(training)(**inputs, cfg_infer=cfg_infer)
                    actual, actual_ctc = disabled.train(training)(**inputs, cfg_infer=cfg_infer)
                    adaptive_tests.assert_nonzero_finite(self, expected, "disabled nontrivial output")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], expected_ctc[layer][key], atol=0, rtol=0)

    def test_fixed_matches_initial_adaptive_outputs_and_changes_unbanded_output(self):
        fixed = make_model()
        adaptive = adaptive_tests.DiT_VT_MMDiT(**{**adaptive_tests.BASE_ARCH, **adaptive_tests.BAND_ARCH})
        missing, unexpected = adaptive.load_state_dict(fixed.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(key.startswith("temporal_band.") for key in missing))
        self.assertFalse(unexpected)
        disabled = make_model(temporal_band_enabled=False)
        disabled.load_state_dict(fixed.state_dict(), strict=True)
        inputs = adaptive_tests.make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for model in (fixed, adaptive, disabled):
                    model.train(training)
                actual, _ = fixed(**inputs)
                expected, _ = adaptive(**inputs)
                unbanded, _ = disabled(**inputs)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                self.assertGreater((actual - unbanded).abs().max().item(), 1e-6)

    def test_cfg_packed_matches_separate_and_geometry_is_constant(self):
        model = make_model().eval()
        inputs = adaptive_tests.make_inputs()
        cases = (
            ({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_video": True}, ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_text": True}, ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
        )
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                with self.subTest(packed_flags=packed_flags):
                    packed, packed_ctc = model(**inputs, cfg_infer=True, **packed_flags)
                    offset = model.last_temporal_band_offset_seconds.clone()
                    sigma = model.last_temporal_band_sigma_seconds.clone()
                    torch.testing.assert_close(offset, torch.zeros_like(offset), atol=0, rtol=0)
                    torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=0, rtol=0)
                    pieces = [model(**inputs, **flags) for flags in separate_flags]
                    torch.testing.assert_close(packed, torch.cat([piece[0] for piece in pieces]), atol=3e-6, rtol=3e-5)
                    for layer in packed_ctc:
                        torch.testing.assert_close(
                            packed_ctc[layer]["z_tilde"], torch.cat([piece[1][layer]["z_tilde"] for piece in pieces]),
                            atol=3e-6, rtol=3e-5,
                        )

    def test_null_video_and_neighbor_outputs_do_not_leak_real_video(self):
        model = make_model().eval()
        inputs = adaptive_tests.make_inputs()
        changed = {**inputs, "video": inputs["video"].clone()}
        changed["video"][0] = torch.randn_like(changed["video"][0]) * 7
        with torch.no_grad():
            original, _ = model(**inputs, cfg_infer=True)
            offset = model.last_temporal_band_offset_seconds.clone()
            sigma = model.last_temporal_band_sigma_seconds.clone()
            altered, _ = model(**changed, cfg_infer=True)
        # Branch-major ordering: [full 0/1, no-video 0/1, null 0/1].
        torch.testing.assert_close(original[1:], altered[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - altered[0]).abs().max().item(), 1e-5)
        torch.testing.assert_close(model.last_temporal_band_offset_seconds, offset, atol=0, rtol=0)
        torch.testing.assert_close(model.last_temporal_band_sigma_seconds, sigma, atol=0, rtol=0)

    def test_checkpointed_backward_reaches_existing_audio_video_speaker_parameters(self):
        plain = make_model(checkpoint_activations=False).train()
        checked = make_model(checkpoint_activations=True).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs = adaptive_tests.make_inputs()
        target = torch.randn(2, 12, 64)
        outputs = []
        required_gradients = (
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.v_attn.to_k.weight",
            "speaker_proj.weight",
        )
        for model in (plain, checked):
            output, ctc = model(**inputs)
            torch.testing.assert_close(ctc[0]["z_lens"], torch.tensor([12, 9]))
            loss = (output - target).square().mean()
            loss = loss + 0.01 * sum(value["z_tilde"].square().mean() for value in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertEqual(list(model.temporal_band.parameters()), [])
            parameters = dict(model.named_parameters())
            for name in required_gradients:
                adaptive_tests.assert_nonzero_finite(self, parameters[name].grad, name)
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = plain_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)

    def test_model_and_ema_save_load_with_explicit_fixed_configuration(self):
        from ema_pytorch import EMA

        # Fixed values are architecture configuration, not trained state. The
        # checkpoint consumer must reconstruct the same config before loading.
        architecture = {**FIXED_ARCH, "temporal_band_fixed_offset_seconds": 0.025,
                        "temporal_band_fixed_sigma_seconds": 0.075}
        model = make_model(**architecture).eval()
        ema = EMA(model, beta=0.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = adaptive_tests.make_inputs()
        with torch.no_grad():
            expected, _ = model(**inputs)
            expected_ema, _ = ema.ema_model(**inputs)
        stream = io.BytesIO()
        torch.save({"architecture": architecture, "model": model.state_dict(), "ema": ema.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        self.assertFalse(any("temporal_band." in key for key in state["model"]))
        self.assertFalse(any("temporal_band." in key for key in state["ema"]))
        restored = make_model(**state["architecture"]).eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_ema = EMA(restored, beta=0.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(state["ema"], strict=True)
        with torch.no_grad():
            actual, _ = restored(**inputs)
            actual_ema, _ = restored_ema.ema_model(**inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)
        offset, sigma = restored.temporal_band(torch.randn(2, 12, 64), audio_len=12)
        torch.testing.assert_close(offset, torch.full_like(offset, 0.025), atol=0, rtol=0)
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.075), atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
