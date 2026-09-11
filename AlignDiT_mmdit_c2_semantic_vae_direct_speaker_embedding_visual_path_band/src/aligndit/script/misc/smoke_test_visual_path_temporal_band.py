"""CPU contracts for the isolated native-visual-change-path alignment experiment.

Run from this snapshot's root (no datasets, checkpoints, or GPUs required)::

    PYTHONPATH=src python src/aligndit/script/misc/smoke_test_visual_path_temporal_band.py

Uses tiny model helpers copied into this snapshot. The optional legacy check
reads the original backbone without importing or modifying its source tree.
All model comparisons use nonzero trained-style residual/output gates.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.semantic_vae_dataset import (
    CELEBVDUB_VIDEO_DIM,
    SemanticVaeCelebVDubDataset,
    load_native_video_path,
)
from aligndit.model.visual_path_temporal_band import (
    VisualPathTemporalBand,
    masked_visual_path,
    native_visual_path,
)
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    setting1_video_path,
    validate_fixed_band_contract,
)
from aligndit.script.misc import smoke_test_adaptive_temporal_band as helpers


PATH_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "visual_path",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.100,
    "temporal_band_path_sigma": 2.0,
}


def make_inputs():
    inputs = helpers.make_inputs()
    increments = torch.tensor([[0., .1, .3, .2, .8, 1.2, .7, .5, 1., .2, .3, .4],
                               [0., .4, .5, .2, 1., .3, .8, .5, 0., 0., 0., 0.]])
    inputs["video_path"] = increments.cumsum(dim=1)
    return inputs


def make_model(**overrides):
    return helpers.warm_start(helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **PATH_ARCH, **overrides}))


class NativePathTests(unittest.TestCase):
    def test_native_normalization_cumulative_path_then_scalar_interpolation(self):
        # A -> B -> A has zero endpoint distance but nonzero traversed distance.
        native = torch.tensor([[2., 0.], [0., 7.], [4., 0.]], requires_grad=True)
        actual = native_visual_path(native, target_length=5)
        expected = math.sqrt(2) * torch.tensor([0., .4, 1., 1.6, 2.])
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
        self.assertGreater(actual[-1].item(), 2.8)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)
        # Resampling descriptors before measuring their normalized changes is
        # a different operation and must not silently replace the native path.
        interpolated = F.interpolate(native.detach().T[None], size=5, mode="linear", align_corners=False)[0].T
        wrong = native_visual_path(interpolated, target_length=5)
        self.assertGreater((actual - wrong).abs().max().item(), .1)

    def test_native_path_is_per_frame_scale_invariant_and_batch_independent(self):
        torch.manual_seed(43)
        video = torch.randn(7, 11)
        expected = native_visual_path(video, 11)
        scale = torch.tensor([.2, 1., 5., .01, 100., 3., 8.])[:, None]
        torch.testing.assert_close(native_visual_path(video * scale, 11), expected, atol=1e-6, rtol=1e-6)
        native_visual_path(torch.randn(4, 11) * 100, 7)
        torch.testing.assert_close(native_visual_path(video, 11), expected, atol=0, rtol=0)

    def test_single_frame_static_zero_features_and_no_rng_use(self):
        for native in (torch.tensor([[1., 2.]]), torch.tensor([[1., 2.]]).repeat(5, 1), torch.zeros(4, 2)):
            state = torch.get_rng_state().clone()
            path = native_visual_path(native, 9)
            torch.testing.assert_close(path, torch.zeros(9), atol=0, rtol=0)
            torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)

    def test_native_validation_rejects_invalid_inputs_without_fallback(self):
        for native, length in ((torch.zeros(0, 3), 5), (torch.zeros(3), 5),
                               (torch.zeros(2, 3), 0), (torch.zeros(2, 3), True),
                               (torch.full((2, 3), float("nan")), 5),
                               (torch.full((2, 3), float("inf")), 5), (torch.ones(2, 3, dtype=torch.int64), 5)):
            with self.subTest(shape=tuple(native.shape), length=length), self.assertRaises((ValueError, TypeError)):
                native_visual_path(native, length)

    def test_native_loader_uses_manifest_frames_and_rejects_missing_wrong_or_unsafe_files(self):
        with tempfile.TemporaryDirectory(prefix="visual-path-native-test-") as directory:
            root = Path(directory)
            native = np.zeros((3, CELEBVDUB_VIDEO_DIM), dtype=np.float32)
            native[0, 0], native[1, 1], native[2, 0] = 2., 7., 4.
            np.save(root / "native.npy", native)
            record = {"utterance_key": "fixture", "video_relative_path": "native.npy", "video_frames_25hz": 3}
            actual = load_native_video_path(record, root, 5)
            expected = native_visual_path(torch.from_numpy(native), 5)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for change in ({"video_frames_25hz": 4}, {"video_relative_path": "missing.npy"},
                           {"video_relative_path": "../outside.npy"}):
                with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                    load_native_video_path({**record, **change}, root, 5)
            np.save(root / "native.npy", native.astype(np.float64))
            with self.assertRaises(ValueError):
                load_native_video_path(record, root, 5)

    def test_padding_and_hidden_endpoints_do_not_leak_edges(self):
        path = torch.tensor([[0., 1., 3., 6., 10.], [0., 2., 4., 100., 1000.]])
        visible = torch.tensor([[False, False, True, True, True], [True, True, True, False, False]])
        rebuilt, increments, valid = masked_visual_path(path, visible)
        torch.testing.assert_close(rebuilt, torch.tensor([[0., 0., 0., 3., 7.], [0., 2., 4., 4., 4.]]))
        torch.testing.assert_close(increments, torch.tensor([[0., 0., 3., 4.], [2., 2., 0., 0.]]))
        torch.testing.assert_close(valid, visible[:, :-1] & visible[:, 1:])
        # A hidden interior frame invalidates both adjacent transitions.
        rebuilt, increments, _ = masked_visual_path(path[:1], torch.tensor([[True, True, False, True, True]]))
        torch.testing.assert_close(increments, torch.tensor([[1., 0., 0., 4.]]))
        torch.testing.assert_close(rebuilt, torch.tensor([[0., 1., 1., 1., 5.]]))

    def test_setting1_prompt_is_zero_and_target_rebased(self):
        target = torch.tensor([.2, .5, 1., 2.])
        actual = setting1_video_path(target)
        expected = torch.tensor([[0., 0., 0., 0., 0., .3, .8, 1.8]])
        torch.testing.assert_close(actual, expected)
        visible = torch.tensor([[False, False, False, False, True, True, True, True]])
        _, increments, _ = masked_visual_path(actual, visible)
        torch.testing.assert_close(increments[:, :4], torch.zeros(1, 4), atol=0, rtol=0)

    def test_collate_repeats_last_path_value_and_requires_all_samples(self):
        def row(path):
            length = len(path)
            return {"mel_spec": torch.zeros(64, length), "video": torch.zeros(length, 16),
                    "text": "a", "ctc_feasible": True, "ctc_target_length": 1,
                    "utterance_key": f"sample-{length}", "video_path": torch.tensor(path)}

        long = row([0., 1., 2., 3., 4.])
        short = row([0., 2., 3.])
        actual = SemanticVaeCelebVDubDataset.collate_fn([long, short])
        torch.testing.assert_close(actual["video_path"], torch.tensor([[0., 1., 2., 3., 4.], [0., 2., 3., 3., 3.]]))
        without = {key: value for key, value in short.items() if key != "video_path"}
        with self.assertRaises((ValueError, RuntimeError)):
            SemanticVaeCelebVDubDataset.collate_fn([long, without])
        for invalid in (torch.tensor([0., 2., 1.]), torch.tensor([0., float("nan"), 3.]),
                        torch.zeros(4), torch.zeros(3, dtype=torch.float64)):
            with self.subTest(path=invalid), self.assertRaises((ValueError, TypeError, RuntimeError)):
                SemanticVaeCelebVDubDataset.collate_fn([long, {**short, "video_path": invalid}])


class VisualPathFormulaTests(unittest.TestCase):
    def test_independent_physical_time_plus_path_formula(self):
        band = VisualPathTemporalBand(dim=64, audio_fps=40., video_fps=25., path_sigma=1.)
        path = torch.tensor([[0., .2, 1., 1.5, 2.5], [0., .5, .6, 1.2, 3.]])
        values = band(torch.zeros(2, 5, 64), audio_len=7)
        actual = band.bias(*values, video_len=5, video_path=path)
        query_index = torch.arange(7).float() * 25 / 40
        left = query_index.floor().long()
        right = (left + 1).clamp(max=4)
        query_path = path[:, left] * (1 - query_index.frac()) + path[:, right] * query_index.frac()
        dt = torch.arange(5).float()[None, None] / 25 - torch.arange(7).float()[None, :, None] / 40
        dc = path[:, None, :] - query_path[:, :, None]
        expected = -.5 * (dt / .100).square() - .5 * dc.square()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.dtype, torch.float32)

    def test_static_path_reduces_exactly_to_fixed_band(self):
        path_band = VisualPathTemporalBand(dim=64, path_sigma=2.)
        fixed = FixedTemporalBand(dim=64)
        video = torch.randn(2, 7, 64)
        offset, sigma = fixed(video, audio_len=10)
        expected = fixed.bias(offset, sigma, video_len=7)
        for constant in (0., 3.5):
            actual = path_band.bias(offset, sigma, video_len=7, video_path=torch.full((2, 7), constant))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_path_scale_controls_only_extra_penalty_and_default_is_calibrated(self):
        default = VisualPathTemporalBand(dim=64)
        self.assertEqual(default.path_sigma, 2.)
        one = VisualPathTemporalBand(dim=64, path_sigma=1.)
        two = VisualPathTemporalBand(dim=64, path_sigma=2.)
        video = torch.zeros(1, 5, 64)
        values = one(video, 5)
        path = torch.arange(5).float()[None]
        fixed = FixedTemporalBand(dim=64).bias(*values, 5)
        bias1 = one.bias(*values, 5, video_path=path)
        bias2 = two.bias(*values, 5, video_path=path)
        torch.testing.assert_close(bias2 - fixed, (bias1 - fixed) / 4, atol=1e-6, rtol=1e-6)
        self.assertTrue((bias1 <= fixed).all())
        torch.testing.assert_close(bias1.diagonal(dim1=1, dim2=2), torch.zeros(1, 5), atol=0, rtol=0)

    def test_zero_parameters_state_rng_and_fp32_under_autocast(self):
        state = torch.get_rng_state().clone()
        band = VisualPathTemporalBand(dim=64)
        self.assertEqual(list(band.parameters()), [])
        self.assertEqual(dict(band.state_dict()), {})
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            values = band(torch.zeros(2, 7, 64), 10)
            bias = band.bias(*values, 7, video_path=torch.arange(7).float()[None].expand(2, -1))
        self.assertEqual(bias.dtype, torch.float32)
        self.assertTrue(torch.isfinite(bias).all())
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(path_sigma=invalid), self.assertRaises(ValueError):
                VisualPathTemporalBand(dim=64, path_sigma=invalid)


class VisualPathJointAttentionTests(helpers.JointAttentionBandTests):
    def setUp(self):
        super().setUp()
        band = VisualPathTemporalBand(dim=64, audio_fps=40., video_fps=25., path_sigma=1.)
        path = torch.tensor([[0., .2, 1., 1.3, 2.], [0., .5, 1., 1.4, 3.]])
        self.bias = band.bias(*band(self.video, 7), 5, video_path=path)

    def test_explicit_joint_softmax_reference_and_av_only_direction(self):
        for training in (False, True):
            for mask_enabled in (False, True):
                with self.subTest(training=training, mask_enabled=mask_enabled):
                    block = self.block.train(training)
                    block.attn_mask_enabled = mask_enabled
                    qa, ka, va = block._qkv(block.attn, self.audio)
                    qv, kv, vv = block._qkv(block.v_attn, self.video)
                    query, key, value = [torch.cat(pair, dim=2) for pair in ((qa, qv), (ka, kv), (va, vv))]
                    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
                    full_bias = torch.zeros_like(logits)
                    full_bias[:, :, :7, 7:] = self.bias[:, None]
                    reference = ((logits + full_bias).softmax(-1) @ value).transpose(1, 2).reshape(2, 12, 64)
                    expected = (block.attn.to_out[1](block.attn.to_out[0](reference[:, :7])),
                                block.v_attn.to_out[1](block.v_attn.to_out[0](reference[:, 7:])))
                    actual = block.joint_attn(self.audio, self.video, temporal_band_bias=self.bias)
                    for output, target in zip(actual, expected):
                        torch.testing.assert_close(output, target, atol=3e-7, rtol=3e-6)


class VisualPathBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_parameterless_schema_and_initialization_rng_unchanged(self):
        torch.manual_seed(83)
        disabled = helpers.DiT_VT_MMDiT(**helpers.BASE_ARCH)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(83)
        enabled = helpers.DiT_VT_MMDiT(**{**helpers.BASE_ARCH, **PATH_ARCH})
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertIsInstance(enabled.temporal_band, VisualPathTemporalBand)
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))
        self.assertEqual(sum(p.numel() for p in enabled.parameters()), sum(p.numel() for p in disabled.parameters()))
        for name, value in enabled.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[name], atol=0, rtol=0, msg=name)
        enabled.load_state_dict(disabled.state_dict(), strict=True)

    def test_disabled_matches_readonly_legacy_with_no_path_and_same_rng(self):
        if not helpers.LEGACY_BACKBONE.is_file():
            self.skipTest("optional original snapshot unavailable")
        module = types.ModuleType("_visual_path_readonly_legacy_backbone")
        module.__file__ = str(helpers.LEGACY_BACKBONE)
        # Trusted neighboring source only; compile avoids writing legacy pycache.
        exec(compile(helpers.LEGACY_BACKBONE.read_text(), str(helpers.LEGACY_BACKBONE), "exec"), module.__dict__)  # noqa: S102
        legacy = helpers.warm_start(module.DiT_VT_MMDiT(**helpers.BASE_ARCH))
        disabled = make_model(temporal_band_enabled=False)
        disabled.load_state_dict(legacy.state_dict(), strict=True)
        inputs = helpers.make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for packed in (False, True):
                    rng = torch.get_rng_state().clone()
                    expected, expected_ctc = legacy.train(training)(**inputs, cfg_infer=packed)
                    after = torch.get_rng_state().clone()
                    torch.set_rng_state(rng)
                    actual, actual_ctc = disabled.train(training)(**inputs, cfg_infer=packed)
                    torch.testing.assert_close(torch.get_rng_state(), after, atol=0, rtol=0)
                    helpers.assert_nonzero_finite(self, expected, "legacy output")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], expected_ctc[layer][key], atol=0, rtol=0)

    def test_enabled_requires_path_disabled_still_accepts_legacy_inputs(self):
        with self.assertRaises((ValueError, RuntimeError)):
            make_model()(**helpers.make_inputs())
        with torch.no_grad():
            output, _ = make_model(temporal_band_enabled=False)(**helpers.make_inputs())
        self.assertTrue(torch.isfinite(output).all())

    def test_cfg_packed_equals_separate_for_two_and_three_branches_with_cache(self):
        model = make_model().eval()
        inputs = make_inputs()
        cases = (({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
                 ({"drop_video": True}, ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
                 ({"drop_text": True}, ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})))
        with torch.no_grad():
            for cache in (False, True):
                inputs["cache"] = cache
                for packed_flags, separate_flags in cases:
                    with self.subTest(cache=cache, flags=packed_flags):
                        model.text_cond = model.text_uncond = None
                        packed, packed_ctc = model(**inputs, cfg_infer=True, **packed_flags)
                        packed_increments = model.last_visual_path_increments.clone()
                        pieces, increments = [], []
                        for flags in separate_flags:
                            piece = model(**inputs, **flags)
                            pieces.append(piece)
                            increments.append(model.last_visual_path_increments.clone())
                        torch.testing.assert_close(packed, torch.cat([piece[0] for piece in pieces]), atol=3e-6, rtol=3e-5)
                        torch.testing.assert_close(packed_increments, torch.cat(increments), atol=0, rtol=0)
                        for layer in packed_ctc:
                            torch.testing.assert_close(packed_ctc[layer]["z_tilde"],
                                                       torch.cat([piece[1][layer]["z_tilde"] for piece in pieces]),
                                                       atol=3e-6, rtol=3e-5)

    def test_null_video_cfg_does_not_leak_content_or_path_or_batch_neighbors(self):
        model = make_model().eval()
        inputs = make_inputs()
        with torch.no_grad():
            original, _ = model(**inputs, cfg_infer=True)
            original_increments = model.last_visual_path_increments.clone()
            self.assertEqual(torch.count_nonzero(original_increments[2:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.last_visual_path_valid_mask[2:]).item(), 0)
            for alter_content in (False, True):
                changed = {**inputs, "video_path": inputs["video_path"].clone(), "video": inputs["video"].clone()}
                changed["video_path"][0] *= 4
                if alter_content:
                    changed["video"][0] = torch.randn_like(changed["video"][0]) * 7
                actual, _ = model(**changed, cfg_infer=True)
                torch.testing.assert_close(actual[1:], original[1:], atol=0, rtol=0)
                torch.testing.assert_close(model.last_visual_path_increments[1:], original_increments[1:], atol=0, rtol=0)
                self.assertGreater((actual[0] - original[0]).abs().max().item(), 1e-6)

    def test_changed_path_is_recomputed_with_cached_text(self):
        model = make_model().eval()
        inputs = make_inputs()
        changed = {**inputs, "video_path": inputs["video_path"] * 3}
        with torch.no_grad():
            original, _ = model(**{**inputs, "cache": True}, cfg_infer=True)
            cached, _ = model(**{**changed, "cache": True}, cfg_infer=True)
            cached_increments = model.last_visual_path_increments.clone()
            fresh, _ = model(**changed, cfg_infer=True)
        torch.testing.assert_close(cached, fresh, atol=0, rtol=0)
        torch.testing.assert_close(cached_increments, model.last_visual_path_increments, atol=0, rtol=0)
        self.assertGreater((cached[:2] - original[:2]).abs().max().item(), 1e-6)
        torch.testing.assert_close(cached[2:], original[2:], atol=0, rtol=0)

    def test_complementary_hidden_prefix_and_padding_cannot_change_geometry(self):
        model = make_model().eval()
        inputs = make_inputs()
        changed = {**inputs, "video_path": inputs["video_path"].clone()}
        # Both paths are monotone. Only hidden prefix edges / prefix-target
        # transition and invalid padded edges differ; visible increments agree.
        changed["video_path"][0, 1:] += 20
        changed["video_path"][0, 2:] += 30
        changed["video_path"][0, 3:] += 40
        changed["video_path"][1, 8:] += torch.tensor([100., 200., 300., 400.])
        with torch.no_grad():
            original, _ = model(**inputs)
            increments = model.last_visual_path_increments.clone()
            actual, _ = model(**changed)
        torch.testing.assert_close(model.last_visual_path_increments, increments, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual, original, atol=3e-6, rtol=3e-5)

    def test_static_path_exact_fixed_reduction_but_real_path_changes_output(self):
        model = make_model().eval()
        fixed = make_model(temporal_band_mode="fixed").eval()
        fixed.load_state_dict(model.state_dict(), strict=True)
        inputs = make_inputs()
        with torch.no_grad():
            expected, _ = fixed(**inputs)
            static, _ = model(**{**inputs, "video_path": torch.zeros_like(inputs["video_path"])})
            dynamic, _ = model(**inputs)
        torch.testing.assert_close(static, expected, atol=0, rtol=0)
        self.assertGreater((dynamic - expected).abs().max().item(), 1e-6)

    def test_activation_checkpointing_preserves_output_and_existing_parameter_gradients(self):
        plain = make_model(checkpoint_activations=False).train()
        checked = make_model(checkpoint_activations=True).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs, target = make_inputs(), torch.randn(2, 12, 64)
        outputs = []
        for model in (plain, checked):
            output, ctc = model(**inputs)
            loss = (output - target).square().mean() + .01 * sum(item["z_tilde"].square().mean() for item in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            parameters = dict(model.named_parameters())
            for name in ("transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.v_attn.to_k.weight", "speaker_proj.weight"):
                helpers.assert_nonzero_finite(self, parameters[name].grad, name)
            outputs.append(output.detach())
        torch.testing.assert_close(*outputs, atol=0, rtol=0)
        reference = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = reference[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)

    def test_model_and_ema_save_load_without_new_state_keys(self):
        from ema_pytorch import EMA

        model = make_model().eval()
        ema = EMA(model, beta=.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = make_inputs()
        with torch.no_grad():
            expected, _ = model(**inputs)
            expected_ema, _ = ema.ema_model(**inputs)
        stream = io.BytesIO()
        torch.save({"arch": PATH_ARCH, "model": model.state_dict(), "ema": ema.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        self.assertFalse(any("temporal_band." in key for key in state["model"]))
        self.assertFalse(any("temporal_band." in key for key in state["ema"]))
        restored = make_model(**state["arch"]).eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_ema = EMA(restored, beta=.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(state["ema"], strict=True)
        with torch.no_grad():
            actual, _ = restored(**inputs)
            actual_ema, _ = restored_ema.ema_model(**inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)


class VisualPathInferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="visual-path-contract-test-")
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "model_100.pt"
        self.sidecar = self.checkpoint.parent / "speaker_training_contract.json"

    def config(self, **overrides):
        return OmegaConf.create({"model": {"arch": {**PATH_ARCH, **overrides}},
                                 "datasets": {"video_path_enabled": True,
                                              "native_video_root": self.directory.name}})

    def write_contract(self, **overrides):
        source = {"native_fps": 25, "feature": "L2-normalized frozen AV-HuBERT native features",
                  "coordinate": "cumulative_l2", "resampling": "linear_align_corners_false",
                  "native_video_root": self.directory.name}
        temporal = {"mode": "visual_path", "parameter_count": 0, "parameters": dict(PATH_ARCH),
                    "path_source": source, **overrides}
        self.sidecar.write_text(json.dumps({"temporal_band": temporal}), encoding="utf-8")

    def test_matching_parameterless_contract_accepted_and_missing_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config(), self.checkpoint)
        self.write_contract()
        self.assertIsNone(validate_fixed_band_contract(self.config(), self.checkpoint))

    def test_wrong_or_missing_native_source_provenance_is_rejected(self):
        self.write_contract(path_source={})
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config(), self.checkpoint)
        self.write_contract()
        config = self.config()
        config.datasets.video_path_enabled = False
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(config, self.checkpoint)
        config = self.config()
        config.datasets.native_video_root = str(Path(self.directory.name) / "other-native-root")
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(config, self.checkpoint)

    def test_mode_sigma_and_parameter_count_mismatch_fail_despite_same_weight_keys(self):
        self.write_contract()
        for override in ({"temporal_band_enabled": False}, {"temporal_band_mode": "fixed"},
                         {"temporal_band_path_sigma": 1.}, {"temporal_band_fixed_sigma_seconds": .075}):
            with self.subTest(override=override), self.assertRaises(RuntimeError):
                validate_fixed_band_contract(self.config(**override), self.checkpoint)
        self.write_contract(parameter_count=1)
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config(), self.checkpoint)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
