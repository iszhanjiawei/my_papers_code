"""CPU contracts for the isolated LocAt-inspired temporal Gaussian experiment.

Run from this experiment root with ``PYTHONPATH=src python
src/aligndit/script/misc/test_locat_temporal.py``. No data, checkpoint, GPU,
network request, or write to another experiment is needed. Nonzero residual
gates make output and gradient comparisons meaningful after warm-starting.
"""

from __future__ import annotations

import io
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.locat_temporal import LocAtTemporalBias


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
LOCAT_ARCH = {
    "locat_enabled": True,
    "locat_av_enabled": True,
    "locat_va_enabled": False,
    "locat_audio_fps": 40.0,
    "locat_video_fps": 40.0,
    "locat_sigma_min_seconds": 0.025,
    "locat_sigma_max_seconds": 0.4,
    "locat_sigma_init_seconds": 0.1,
    "locat_alpha_init": 0.1,
    "locat_bias_mode": "gaussian",
}


def make_inputs():
    audio_mask = torch.arange(12)[None] < torch.tensor([12, 9])[:, None]
    video_mask = torch.arange(12)[None] < torch.tensor([12, 8])[:, None]
    text_mask = torch.arange(4)[None] < torch.tensor([4, 3])[:, None]
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
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
            if hasattr(block, "cross_attn_ada"):
                block.cross_attn_ada.weight.normal_(std=0.03)
                block.cross_attn_ada.bias.normal_(std=0.03)
            if hasattr(block, "v_attn_norm"):
                block.v_attn_norm.linear.weight.normal_(std=0.03)
                block.v_attn_norm.linear.bias.normal_(std=0.03)
        model.proj_out.weight.normal_(std=0.03)
        model.norm_out.linear.weight.normal_(std=0.03)
        model.speaker_proj.weight.normal_(std=0.03)
    return model


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **LOCAT_ARCH, **overrides}))


def predictors(model):
    return [module for module in model.modules() if isinstance(module, LocAtTemporalBias)]


def make_content_sensitive(model):
    with torch.no_grad():
        for module in predictors(model):
            module.log_sigma.weight.normal_(std=0.04)
            module.log_alpha.weight.normal_(std=0.04)


def assert_nonzero_finite(test, tensor, name):
    test.assertIsNotNone(tensor, f"{name}: missing tensor/gradient")
    test.assertTrue(torch.isfinite(tensor).all().item(), f"{name}: non-finite values")
    test.assertGreater(torch.count_nonzero(tensor).item(), 0, f"{name}: all zeros")


def load_legacy_module():
    module = types.ModuleType("_locat_readonly_legacy_backbone")
    module.__file__ = str(LEGACY_BACKBONE)
    # Compile trusted source under a separate name, without legacy pycache.
    exec(compile(LEGACY_BACKBONE.read_text(), str(LEGACY_BACKBONE), "exec"), module.__dict__)  # noqa: S102
    return module


def capture_joint(block, audio, video, **kwargs):
    calls = []
    original_sdpa = F.scaled_dot_product_attention

    def record(query, key, value, *args, **sdpa_kwargs):
        mask = sdpa_kwargs.get("attn_mask", args[0] if args else None)
        shape = (*query.shape[:-1], key.shape[-2])
        additive = torch.zeros(shape, dtype=torch.float32, device=query.device)
        if mask is not None:
            additive = additive.masked_fill(~mask, -float("inf")) if mask.dtype == torch.bool else mask.float().expand(shape)
        calls.append((query.detach().clone(), key.detach().clone(), value.detach().clone(), additive.detach().clone()))
        return original_sdpa(query, key, value, *args, **sdpa_kwargs)

    with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=record):
        outputs = block.joint_attn(audio, video, **kwargs)
    combined_bias = torch.cat([call[3] for call in calls], dim=-2)
    return outputs, combined_bias, calls


class GaussianFormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)

    def make_bias(self, **overrides):
        return LocAtTemporalBias(**{"dim_head": 16, "query_fps": 40.0, "key_fps": 40.0, **overrides})

    def test_initial_sigma_alpha_positive_gaussian_and_far_decay(self):
        module = self.make_bias()
        query = torch.randn(2, 4, 11, 16)
        sigma, alpha = module.parameters_from_query(query)
        self.assertEqual(sigma.shape, (2, 4, 11, 1))
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=1e-7, rtol=0)
        torch.testing.assert_close(alpha, torch.full_like(alpha, 0.1), atol=1e-7, rtol=0)
        bias = module(query, key_len=11)
        distance = (torch.arange(11)[:, None] - torch.arange(11)[None, :]).float() / 40
        expected = 0.1 * torch.exp(-0.5 * (distance / 0.1).square())
        torch.testing.assert_close(bias, expected.expand(2, 4, -1, -1), atol=1e-7, rtol=1e-6)
        self.assertTrue((bias >= 0).all())
        self.assertLess(bias[0, 0, 0, -1].item(), bias[0, 0, 0, 1].item())
        self.assertGreater(bias[0, 0, 0, 1].item(), 0)

    def test_unequal_frame_rates_use_seconds_not_sequence_length(self):
        module = self.make_bias(query_fps=40.0, key_fps=25.0)
        query = torch.randn(2, 4, 13, 16)
        actual = module(query, key_len=8)
        distance = torch.arange(13).float()[:, None] / 40 - torch.arange(8).float()[None, :] / 25
        expected = 0.1 * torch.exp(-0.5 * (distance / 0.1).square())
        torch.testing.assert_close(actual, expected.expand(2, 4, -1, -1), atol=1e-7, rtol=1e-6)
        # 200 ms has audio index 8 and video index 5.
        self.assertEqual(actual[0, 0, 8].argmax().item(), 5)
        longer = module(query, key_len=19)
        torch.testing.assert_close(actual, longer[..., :8], atol=0, rtol=0)

    def test_chunk_offset_matches_full_sequence(self):
        module = self.make_bias(query_fps=40.0, key_fps=25.0)
        make_content_sensitive(module)
        query = torch.randn(2, 4, 13, 16)
        full = module(query, key_len=10)
        chunk = module(query[:, :, 5:9], key_len=10, query_offset=5)
        torch.testing.assert_close(chunk, full[:, :, 5:9], atol=2e-8, rtol=2e-7)

    def test_uniform_control_removes_distance_but_keeps_alpha(self):
        module = self.make_bias(bias_mode="uniform")
        make_content_sensitive(module)
        query = torch.randn(2, 4, 9, 16)
        _, alpha = module.parameters_from_query(query)
        bias = module(query, key_len=6)
        torch.testing.assert_close(bias, alpha.expand(-1, -1, -1, 6), atol=0, rtol=0)
        bias.square().sum().backward()
        assert_nonzero_finite(self, module.log_alpha.weight.grad, "uniform alpha")
        self.assertIsNotNone(module.log_sigma.weight.grad, "uniform mode must keep its DDP graph dependency")
        self.assertEqual(torch.count_nonzero(module.log_sigma.weight.grad).item(), 0)

    def test_zero_alpha_limit_matches_unmodified_logits(self):
        module = self.make_bias()
        with torch.no_grad():
            module.log_alpha.bias.fill_(-100)
        query = torch.randn(2, 4, 9, 16)
        logits = torch.randn(2, 4, 9, 7)
        bias = module(query, key_len=7)
        torch.testing.assert_close((logits + bias).softmax(-1), logits.softmax(-1), atol=0, rtol=0)

    def test_predictors_shared_across_heads_but_outputs_vary(self):
        module = self.make_bias()
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 2 * (16 + 1))
        self.assertEqual(set(module.state_dict()), {"log_sigma.weight", "log_sigma.bias", "log_alpha.weight", "log_alpha.bias"})
        make_content_sensitive(module)
        query = torch.randn(2, 4, 9, 16)
        sigma, alpha = module.parameters_from_query(query)
        self.assertGreater((sigma[:, 0] - sigma[:, 1]).abs().max().item(), 1e-5)
        self.assertGreater((alpha[:, :, 0] - alpha[:, :, 1]).abs().max().item(), 1e-5)
        # Identical head inputs must get identical outputs despite head index.
        repeated = query[:, :1].expand(-1, 4, -1, -1)
        shared_sigma, shared_alpha = module.parameters_from_query(repeated)
        for actual in (shared_sigma, shared_alpha):
            torch.testing.assert_close(actual[:, 0], actual[:, 3], atol=2e-8, rtol=2e-7)

    def test_both_predictor_heads_receive_finite_nonzero_gradients(self):
        module = self.make_bias()
        query = torch.randn(2, 4, 9, 16, requires_grad=True)
        bias = module(query, key_len=7)
        (bias * torch.randn_like(bias)).sum().backward()
        for label in ("log_sigma", "log_alpha"):
            predictor = getattr(module, label)
            assert_nonzero_finite(self, predictor.weight.grad, f"{label} weight")
            assert_nonzero_finite(self, predictor.bias.grad, f"{label} bias")

    def test_query_key_padding_and_disabled_samples_have_zero_extra_bias(self):
        module = self.make_bias()
        make_content_sensitive(module)
        query = torch.randn(2, 4, 9, 16)
        query_mask = torch.arange(9)[None] < torch.tensor([7, 5])[:, None]
        key_mask = torch.arange(6)[None] < torch.tensor([4, 3])[:, None]
        enabled = torch.tensor([True, False])
        bias = module(query, 6, query_mask=query_mask, key_mask=key_mask, enabled=enabled)
        valid = query_mask[:, None, :, None] & key_mask[:, None, None, :] & enabled[:, None, None, None]
        valid = valid.expand_as(bias)
        self.assertEqual(torch.count_nonzero(bias[~valid]).item(), 0)
        self.assertGreater(torch.count_nonzero(bias[valid]).item(), 0)
        self.assertTrue(torch.isfinite(bias).all())

    def test_all_disabled_keeps_zero_not_missing_predictor_gradients(self):
        module = self.make_bias()
        query = torch.randn(2, 4, 9, 16)
        bias = module(query, 6, enabled=torch.zeros(2, dtype=torch.bool))
        self.assertEqual(torch.count_nonzero(bias).item(), 0)
        bias.sum().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_sigma_extreme_inputs_stay_in_physical_bounds(self):
        module = self.make_bias()
        query = torch.randn(2, 4, 9, 16)
        with torch.no_grad():
            for value in (-1000.0, 1000.0):
                module.log_sigma.bias.fill_(value)
                sigma, _ = module.parameters_from_query(query)
                self.assertTrue(torch.isfinite(sigma).all())
                self.assertTrue((sigma >= 0.025 - 1e-7).all())
                self.assertTrue((sigma <= 0.4 + 1e-7).all())

    def test_bfloat16_computation_and_backward_remain_finite(self):
        module = self.make_bias()
        query = torch.randn(2, 4, 9, 16).bfloat16()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            sigma, alpha = module.parameters_from_query(query)
            bias = module(query, key_len=7)
            loss = (bias.float() * torch.randn_like(bias.float())).sum()
        self.assertEqual(sigma.dtype, torch.float32)
        self.assertEqual(alpha.dtype, torch.float32)
        self.assertEqual(bias.dtype, query.dtype)
        self.assertTrue(torch.isfinite(bias).all())
        loss.backward()
        assert_nonzero_finite(self, module.log_sigma.weight.grad, "BF16 sigma")
        assert_nonzero_finite(self, module.log_alpha.weight.grad, "BF16 alpha")

    def test_invalid_constructor_configuration_is_rejected(self):
        invalid_cases = (
            {"dim_head": 0}, {"query_fps": 0}, {"key_fps": -1},
            {"query_fps": float("nan")}, {"sigma_min_seconds": 0},
            {"sigma_init_seconds": 0.025}, {"sigma_max_seconds": 0.08},
            {"alpha_init": 0}, {"alpha_init": float("nan")}, {"bias_mode": "log_gaussian"},
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                self.make_bias(**invalid)


class JointAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(207)
        self.audio = torch.randn(2, 7, 64)
        self.video = torch.randn(2, 5, 64)

    def make_block(self, av, va, *, pe_attn_head=1):
        block = MMDiTBlock_VT(
            dim=64, heads=4, dim_head=16, text_dim=32, dropout=0.0,
            qk_norm="rms_norm", pe_attn_head=pe_attn_head,
            attn_mask_enabled=True, prompt_isolated_ca=False,
        )
        block.locat_av = LocAtTemporalBias(16, 40.0, 25.0) if av else None
        block.locat_va = LocAtTemporalBias(16, 25.0, 40.0) if va else None
        return block.eval()

    def assert_manual_sdpa(self, block, calls, actual, mask=None, v_mask=None):
        pieces = []
        for query, key, value, bias in calls:
            logits = query.float() @ key.float().transpose(-1, -2) / query.shape[-1] ** 0.5 + bias
            pieces.append(logits.softmax(-1) @ value.float())
        joint = torch.cat(pieces, dim=-2).transpose(1, 2).reshape(2, 12, 64)
        expected_audio = block.attn.to_out[1](block.attn.to_out[0](joint[:, :7]))
        expected_video = block.v_attn.to_out[1](block.v_attn.to_out[0](joint[:, 7:]))
        if mask is not None:
            expected_audio = expected_audio.masked_fill(~mask[..., None], 0)
        if v_mask is not None:
            expected_video = expected_video.masked_fill(~v_mask[..., None], 0)
        torch.testing.assert_close(actual[0], expected_audio, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(actual[1], expected_video, atol=2e-6, rtol=1e-5)

    def test_all_four_direction_switches_and_manual_joint_softmax(self):
        for av, va in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(av=av, va=va):
                block = self.make_block(av, va)
                actual, bias, calls = capture_joint(block, self.audio, self.video)
                self.assertEqual(bias.shape, (2, 4, 12, 12))
                self.assertEqual(torch.count_nonzero(bias[:, :, :7, :7]).item(), 0)
                self.assertEqual(torch.count_nonzero(bias[:, :, 7:, 7:]).item(), 0)
                self.assertEqual(torch.count_nonzero(bias[:, :, :7, 7:]).item() > 0, av)
                self.assertEqual(torch.count_nonzero(bias[:, :, 7:, :7]).item() > 0, va)
                self.assert_manual_sdpa(block, calls, actual)

    def test_audio_only_augmentation_leaves_video_output_exactly_unchanged(self):
        plain = self.make_block(False, False)
        changed = self.make_block(True, False)
        changed.load_state_dict(plain.state_dict(), strict=False)
        expected = plain.joint_attn(self.audio, self.video)
        actual = changed.joint_attn(self.audio, self.video)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-7, rtol=1e-6)
        self.assertGreater((actual[0] - expected[0]).abs().max().item(), 1e-5)

    def test_true_padding_masks_apply_zero_extra_bias_without_training_key_masks(self):
        block = self.make_block(True, True).train()
        audio_mask = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
        video_mask = torch.arange(5)[None] < torch.tensor([5, 3])[:, None]
        _, bias, _ = capture_joint(
            block, self.audio, self.video,
            locat_audio_mask=audio_mask, locat_video_mask=video_mask,
            locat_video_enabled=torch.ones(2, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(bias).all())
        for query_mask, key_mask, quadrant in (
            (audio_mask, video_mask, bias[:, :, :7, 7:]),
            (video_mask, audio_mask, bias[:, :, 7:, :7]),
        ):
            valid = (query_mask[:, None, :, None] & key_mask[:, None, None, :]).expand_as(quadrant)
            self.assertEqual(torch.count_nonzero(quadrant[~valid]).item(), 0)
            self.assertGreater(torch.count_nonzero(quadrant[valid]).item(), 0)

    def test_eval_padding_masks_block_keys_and_zero_query_outputs(self):
        block = self.make_block(True, True)
        audio_mask = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
        video_mask = torch.arange(5)[None] < torch.tensor([5, 3])[:, None]
        actual, bias, calls = capture_joint(
            block, self.audio, self.video, mask=audio_mask, v_mask=video_mask,
            locat_audio_mask=audio_mask, locat_video_mask=video_mask,
        )
        valid_keys = torch.cat((audio_mask, video_mask), dim=-1)[:, None, None, :].expand_as(bias)
        self.assertTrue(torch.isneginf(bias[~valid_keys]).all())
        self.assertTrue(torch.isfinite(bias[valid_keys]).all())
        self.assertEqual(torch.count_nonzero(actual[0][~audio_mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[1][~video_mask]).item(), 0)
        self.assert_manual_sdpa(block, calls, actual, mask=audio_mask, v_mask=video_mask)

    def test_null_video_switch_disables_both_direction_biases(self):
        block = self.make_block(True, True)
        _, bias, _ = capture_joint(
            block, self.audio, self.video, locat_video_enabled=torch.tensor([True, False])
        )
        self.assertGreater(torch.count_nonzero(bias[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(bias[1]).item(), 0)

    def test_pre_rope_predictor_input_and_partial_rope_backward(self):
        block = self.make_block(True, True, pe_attn_head=1)
        make_content_sensitive(block)
        rotary = DiT_VT_MMDiT(**BASE_ARCH).rotary_embed
        rope = rotary.forward_from_seq_len(7)
        video_rope = rotary.forward(torch.arange(5).float() * 1.6)
        audio = self.audio.clone().requires_grad_()
        video = self.video.clone().requires_grad_()
        expected_a = block._qkv(block.attn, audio)[0].detach().clone()
        expected_v = block._qkv(block.v_attn, video)[0].detach().clone()
        observed = []
        original_av = block.locat_av.parameters_from_query
        original_va = block.locat_va.parameters_from_query

        def record_av(query, **kwargs):
            observed.append(query.detach().clone())
            return original_av(query, **kwargs)

        def record_va(query, **kwargs):
            observed.append(query.detach().clone())
            return original_va(query, **kwargs)

        with patch.object(block.locat_av, "parameters_from_query", side_effect=record_av), patch.object(
            block.locat_va, "parameters_from_query", side_effect=record_va
        ):
            out_a, out_v = block.joint_attn(audio, video, rope=rope, v_rope=video_rope)
            ((out_a * torch.randn_like(out_a)).sum() + (out_v * torch.randn_like(out_v)).sum()).backward()
        self.assertEqual(len(observed), 2)
        torch.testing.assert_close(observed[0], expected_a, atol=0, rtol=0)
        torch.testing.assert_close(observed[1], expected_v, atol=0, rtol=0)
        assert_nonzero_finite(self, audio.grad, "partial-RoPE audio gradient")
        assert_nonzero_finite(self, video.grad, "partial-RoPE video gradient")
        for module in (block.locat_av, block.locat_va):
            assert_nonzero_finite(self, module.log_sigma.weight.grad, "partial-RoPE sigma")
            assert_nonzero_finite(self, module.log_alpha.weight.grad, "partial-RoPE alpha")


class BackboneIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311)

    def test_disabled_preserves_parent_state_keys_rng_and_outputs(self):
        if not LEGACY_BACKBONE.is_file():
            self.skipTest("Optional original snapshot is not available")
        legacy_class = load_legacy_module().DiT_VT_MMDiT
        torch.manual_seed(41)
        legacy = legacy_class(**BASE_ARCH)
        legacy_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(41)
        disabled = DiT_VT_MMDiT(**BASE_ARCH, locat_enabled=False)
        torch.testing.assert_close(torch.random.get_rng_state(), legacy_rng, atol=0, rtol=0)
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        for name, tensor in legacy.state_dict().items():
            torch.testing.assert_close(disabled.state_dict()[name], tensor, atol=0, rtol=0, msg=name)
        warm_start(legacy)
        disabled.load_state_dict(legacy.state_dict(), strict=True)
        inputs = make_inputs()
        with torch.no_grad():
            for training in (False, True):
                legacy.train(training)
                disabled.train(training)
                for cfg_infer in (False, True):
                    expected, expected_ctc = legacy(**inputs, cfg_infer=cfg_infer)
                    actual, actual_ctc = disabled(**inputs, cfg_infer=cfg_infer)
                    assert_nonzero_finite(self, expected, "parent output")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], expected_ctc[layer][key], atol=0, rtol=0)

    def test_enabled_preserves_common_parent_initialization_and_rng(self):
        torch.manual_seed(53)
        disabled = DiT_VT_MMDiT(**BASE_ARCH)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(53)
        enabled = DiT_VT_MMDiT(**{**BASE_ARCH, **LOCAT_ARCH})
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, atol=0, rtol=0)
        for name, tensor in disabled.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[name], tensor, atol=0, rtol=0, msg=name)
        added = set(enabled.state_dict()) - set(disabled.state_dict())
        self.assertEqual(len(added), 2 * 4)
        self.assertTrue(all(".locat_av." in name for name in added))
        missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)
        self.assertEqual(set(missing), added)
        self.assertFalse(unexpected)

    def test_modules_are_layer_specific_and_last_mm_has_no_unused_va(self):
        model = make_model(locat_va_enabled=True)
        first, last = model.transformer_blocks[:2]
        self.assertIsNot(first.locat_av, last.locat_av)
        self.assertIsNotNone(first.locat_va)
        self.assertIsNone(last.locat_va)
        self.assertEqual(len(predictors(model)), 3)
        self.assertEqual(sum(p.numel() for module in predictors(model) for p in module.parameters()), 3 * 2 * 17)
        for tail in model.transformer_blocks[2:]:
            self.assertFalse(any(isinstance(module, LocAtTemporalBias) for module in tail.modules()))

    def test_both_disabled_flags_are_exact_identity_with_no_added_state(self):
        torch.manual_seed(71)
        baseline = warm_start(DiT_VT_MMDiT(**BASE_ARCH)).eval()
        disabled = DiT_VT_MMDiT(**{**BASE_ARCH, **LOCAT_ARCH, "locat_av_enabled": False, "locat_va_enabled": False}).eval()
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        disabled.load_state_dict(baseline.state_dict(), strict=True)
        inputs = make_inputs()
        with torch.no_grad():
            expected, _ = baseline(**inputs)
            actual, _ = disabled(**inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_cfg_two_and_three_branches_match_separate_batch_members(self):
        model = make_model(locat_va_enabled=True).eval()
        make_content_sensitive(model)
        inputs = make_inputs()
        cases = (
            ({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_video": True}, ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_text": True}, ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
        )
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                with self.subTest(packed_flags=packed_flags):
                    packed, packed_ctc = model(**inputs, cfg_infer=True, **packed_flags)
                    separate = [model(**inputs, **flags) for flags in separate_flags]
                    torch.testing.assert_close(packed, torch.cat([output for output, _ in separate]), atol=3e-6, rtol=3e-5)
                    for layer in packed_ctc:
                        torch.testing.assert_close(
                            packed_ctc[layer]["z_tilde"],
                            torch.cat([ctc[layer]["z_tilde"] for _, ctc in separate]),
                            atol=3e-6, rtol=3e-5,
                        )

    def test_null_cfg_outputs_do_not_leak_real_video_or_other_samples(self):
        model = make_model(locat_va_enabled=True).eval()
        make_content_sensitive(model)
        inputs = make_inputs()
        changed = {**inputs, "video": inputs["video"].clone()}
        changed["video"][0] = torch.randn_like(changed["video"][0]) * 7
        with torch.no_grad():
            original, _ = model(**inputs, cfg_infer=True)
            altered, _ = model(**changed, cfg_infer=True)
        # [full0, full1, TTS0, TTS1, null0, null1]; only full0 may change.
        torch.testing.assert_close(original[1:], altered[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - altered[0]).abs().max().item(), 1e-5)

    def test_conformer_enabled_cfg_batch_two_matches_separate_branches(self):
        model = make_model(use_conformer=True, locat_va_enabled=True).eval()
        make_content_sensitive(model)
        inputs = make_inputs()
        self.assertTrue(model.video_embed.use_conformer)
        cases = (
            ({}, ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_video": True}, ({"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
            ({"drop_text": True}, ({"drop_text": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True})),
        )
        with torch.no_grad():
            for packed_flags, separate_flags in cases:
                with self.subTest(packed_flags=packed_flags):
                    packed, _ = model(**inputs, cfg_infer=True, **packed_flags)
                    separate = [model(**inputs, **flags)[0] for flags in separate_flags]
                    self.assertTrue(torch.isfinite(packed).all())
                    torch.testing.assert_close(packed, torch.cat(separate), atol=3e-6, rtol=3e-5)
            changed = {**inputs, "video": inputs["video"].clone()}
            changed["video"][0] = torch.randn_like(changed["video"][0]) * 7
            expected, _ = model(**inputs, cfg_infer=True)
            actual, _ = model(**changed, cfg_infer=True)
        torch.testing.assert_close(actual[1:], expected[1:], atol=0, rtol=0)
        self.assertGreater((actual[0] - expected[0]).abs().max().item(), 1e-5)

    def test_all_null_model_diagnostics_are_finite_zero_with_zero_parameter_gradients(self):
        model = make_model(locat_va_enabled=True).train()
        make_content_sensitive(model)
        inputs = make_inputs()
        # First observe nonzero summaries, then ensure null forwards overwrite
        # them rather than reporting stale conditional-batch statistics.
        model(**inputs)
        self.assertGreater(model.locat_diagnostics()["av/alpha_mean"].item(), 0)
        cases = (
            ("dropped", inputs, {"drop_video": True}),
            ("no_valid_video_keys", {**inputs, "video_mask": torch.zeros_like(inputs["video_mask"])}, {}),
            ("all_video_hidden", {**inputs, "complementary_mask": inputs["video_mask"].clone()}, {}),
        )
        for label, case_inputs, flags in cases:
            with self.subTest(case=label):
                model.zero_grad(set_to_none=True)
                output, ctc = model(**case_inputs, **flags)
                self.assertTrue(torch.isfinite(output).all())
                diagnostics = model.locat_diagnostics()
                self.assertTrue(diagnostics)
                self.assertTrue(any(name.startswith("av/") for name in diagnostics))
                self.assertTrue(any(name.startswith("va/") for name in diagnostics))
                for name, value in diagnostics.items():
                    self.assertEqual(value.ndim, 0, name)
                    self.assertFalse(value.requires_grad, name)
                    self.assertTrue(torch.isfinite(value).item(), name)
                    self.assertEqual(value.item(), 0.0, name)
                loss = output.square().mean() + 0.01 * sum(value["z_tilde"].square().mean() for value in ctc.values())
                loss.backward()
                for module in predictors(model):
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, f"{label}: {name}")
                        self.assertTrue(torch.isfinite(parameter.grad).all(), f"{label}: {name}")
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, f"{label}: {name}")

    def test_complementary_hidden_video_content_is_not_used(self):
        model = make_model(locat_va_enabled=True).eval()
        make_content_sensitive(model)
        inputs = make_inputs()
        changed = {**inputs, "video": inputs["video"].clone()}
        changed["video"][inputs["complementary_mask"]] = torch.randn_like(changed["video"][inputs["complementary_mask"]]) * 100
        with torch.no_grad():
            expected, _ = model(**inputs, cfg_infer=True)
            actual, _ = model(**changed, cfg_infer=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_training_keeps_bias_active_but_passes_true_validity_separately(self):
        model = make_model().train()
        inputs = make_inputs()
        observed = []

        def hook(_module, args, kwargs):
            observed.append(kwargs)

        handle = model.transformer_blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            output, _ = model(**inputs)
            output.square().sum().backward()
        finally:
            handle.remove()
        self.assertIsNone(observed[0]["mask"])
        self.assertIsNone(observed[0]["v_mask"])
        torch.testing.assert_close(observed[0]["locat_audio_mask"], inputs["mask"])
        torch.testing.assert_close(observed[0]["locat_video_mask"], inputs["video_mask"] & ~inputs["complementary_mask"])
        self.assertTrue(observed[0]["locat_video_enabled"].all())
        for module in predictors(model):
            assert_nonzero_finite(self, module.log_sigma.weight.grad, "training sigma")
            assert_nonzero_finite(self, module.log_alpha.weight.grad, "training alpha")

    def test_activation_checkpointing_output_and_gradient_parity(self):
        plain = make_model(locat_va_enabled=True).train()
        make_content_sensitive(plain)
        checked = DiT_VT_MMDiT(**{**BASE_ARCH, **LOCAT_ARCH, "locat_va_enabled": True, "checkpoint_activations": True}).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs = make_inputs()
        target = torch.randn(2, 12, 64)
        outputs = []
        for model in (plain, checked):
            output, ctc = model(**inputs)
            loss = (output - target).square().mean() + 0.01 * sum(value["z_tilde"].square().mean() for value in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            outputs.append(output.detach())
            for module in predictors(model):
                assert_nonzero_finite(self, module.log_sigma.weight.grad, "checkpoint sigma")
                assert_nonzero_finite(self, module.log_alpha.weight.grad, "checkpoint alpha")
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        expected_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = expected_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)

    def test_model_and_ema_state_roundtrip(self):
        from ema_pytorch import EMA

        model = make_model(locat_va_enabled=True).eval()
        make_content_sensitive(model)
        ema = EMA(model, beta=0.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = make_inputs()
        with torch.no_grad():
            expected, _ = model(**inputs)
            expected_ema, _ = ema.ema_model(**inputs)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = make_model(locat_va_enabled=True).eval()
        restored.load_state_dict(state["model"], strict=True)
        restored_ema = EMA(restored, beta=0.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(state["ema"], strict=True)
        with torch.no_grad():
            actual, _ = restored(**inputs)
            actual_ema, _ = restored_ema.ema_model(**inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)
        self.assertTrue(any(".locat_av." in name for name in state["ema"]))

    def test_cfm_and_ctc_backward_reaches_every_active_locat_module(self):
        from aligndit.model.cfm_vt import CFM_VT

        transformer = make_model(locat_va_enabled=True).train()
        flow_model = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1,
            ctc_lambda=0.03, audio_drop_prob=0, cond_drop_prob=0,
            text_drop_prob=0, video_drop_prob=0,
        )
        inputs = make_inputs()
        lengths = inputs["mask"].sum(-1)
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, output = flow_model(
                inp=inputs["x"], text=inputs["text"], video=inputs["video"],
                lens=lengths, text_lens=inputs["text_mask"].sum(-1), video_lens=lengths,
                speaker_embedding=inputs["speaker_embedding"],
            )
        self.assertEqual(output.shape, (2, 12, 64))
        self.assertIn("ctc_loss", components)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in predictors(transformer):
            assert_nonzero_finite(self, module.log_sigma.weight.grad, "CFM+CTC sigma")
            assert_nonzero_finite(self, module.log_alpha.weight.grad, "CFM+CTC alpha")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
