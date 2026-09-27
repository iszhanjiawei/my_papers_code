"""CPU contracts for soft Visual Path Band AV and blocked VA, with global VV.

The independent reference constructs the complete four-quadrant float bias;
production uses split queries to avoid that square allocation. No data/GPU is
required. Original Visual Path Band tests remain separate with VA enabled.
"""

from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.visual_path_temporal_band import VisualPathTemporalBand
from aligndit.script.misc import smoke_test_adaptive_temporal_band as helpers
from aligndit.script.misc import smoke_test_visual_path_temporal_band as original


BASE_ARCH = helpers.BASE_ARCH
ATTENTION_ARCH = {**original.PATH_ARCH, "block_video_audio_attention": True}
make_inputs = original.make_inputs


def make_model(**overrides):
    return helpers.warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **ATTENTION_ARCH, **overrides}))


def make_block(**overrides):
    return MMDiTBlock_VT(**{
        "dim": 16, "heads": 2, "dim_head": 8, "dropout": 0.0,
        "ff_mult": 2, "text_dim": 16, "prompt_isolated_ca": False,
        "attn_mask_enabled": True, "block_video_audio_attention": True,
        **overrides,
    })


def full_joint_reference(block, audio, video, bias=None, mask=None, v_mask=None):
    """Independent softmax reference; VA is the only structural hard mask."""
    batch, na, _ = audio.shape
    nv = video.shape[1]
    qa, ka, va = block._qkv(block.attn, audio)
    qv, kv, vv = block._qkv(block.v_attn, video)
    query, key, value = [torch.cat(pair, dim=2) for pair in ((qa, qv), (ka, kv), (va, vv))]
    full_bias = torch.zeros(batch, 1, na + nv, na + nv, dtype=query.dtype)
    if bias is not None:
        full_bias[:, :, :na, na:] = bias[:, None]
    full_bias[:, :, na:, :na] = -torch.inf
    if block.attn_mask_enabled and (mask is not None or v_mask is not None):
        am = torch.ones(batch, na, dtype=torch.bool) if mask is None else mask
        vm = torch.ones(batch, nv, dtype=torch.bool) if v_mask is None else v_mask
        full_bias = full_bias.masked_fill(~torch.cat([am, vm], dim=1)[:, None, None], -torch.inf)
    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1]) + full_bias
    # A completely masked video row has zero SDPA output, not NaN. Mask its
    # logits before softmax to retain finite derivatives in this reference.
    all_masked = torch.isneginf(logits).all(-1, keepdim=True)
    probabilities = logits.masked_fill(all_masked, 0).softmax(-1).masked_fill(all_masked, 0)
    out = (probabilities @ value).transpose(1, 2).reshape(batch, na + nv, -1)
    out_a = block.attn.to_out[1](block.attn.to_out[0](out[:, :na]))
    out_v = block.v_attn.to_out[1](block.v_attn.to_out[0](out[:, na:]))
    if mask is not None:
        out_a = out_a.masked_fill(~mask[..., None], 0)
    if v_mask is not None:
        out_v = out_v.masked_fill(~v_mask[..., None], 0)
    return (out_a, out_v), full_bias


class VisualPathNoVaAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(815)
        self.audio = torch.randn(2, 7, 16)
        self.video = torch.randn(2, 5, 16)
        band = VisualPathTemporalBand(dim=16)
        path = torch.tensor([[0., .2, .3, .7, .9], [0., .1, .5, .8, 1.]])
        self.bias = band.bias(*band(self.video, 7), 5, video_path=path)
        self.amask = torch.arange(7)[None] < torch.tensor([7, 5])[:, None]
        self.vmask = torch.arange(5)[None] < torch.tensor([4, 3])[:, None]

    def test_four_quadrants_split_outputs_match_full_float_bias_reference(self):
        for training in (False, True):
            for enabled in (False, True):
                for padding in ("none", "both", "video_only", "all_video_empty"):
                    for use_bias in (False, True):
                        with self.subTest(training=training, enabled=enabled, padding=padding, bias=use_bias):
                            block = make_block(attn_mask_enabled=enabled).train(training)
                            mask = self.amask if padding == "both" else None
                            vm = self.vmask if padding in ("both", "video_only") else None
                            if padding == "all_video_empty":
                                vm = torch.zeros_like(self.vmask)
                            bias = self.bias if use_bias else None
                            expected, full_bias = full_joint_reference(block, self.audio, self.video, bias, mask, vm)
                            actual = block.joint_attn(self.audio, self.video, mask=mask, v_mask=vm, temporal_band_bias=bias)
                            self.assertTrue(torch.isneginf(full_bias[:, :, 7:, :7]).all())
                            if padding == "none":
                                self.assertTrue(torch.isfinite(full_bias[:, :, :7, 7:]).all())
                                self.assertFalse(torch.count_nonzero(full_bias[:, :, :7, :7]))
                                self.assertFalse(torch.count_nonzero(full_bias[:, :, 7:, 7:]))
                            for actual_piece, expected_piece in zip(actual, expected):
                                self.assertTrue(torch.isfinite(actual_piece).all())
                                torch.testing.assert_close(actual_piece, expected_piece, atol=2e-7, rtol=3e-6)

    def test_query_split_gradients_match_full_quadrant_reference(self):
        split = make_block().train()
        reference = copy.deepcopy(split)
        results = []
        for block, use_reference in ((split, False), (reference, True)):
            audio = self.audio.clone().requires_grad_()
            video = self.video.clone().requires_grad_()
            bias = self.bias.clone().requires_grad_()
            if use_reference:
                out, _ = full_joint_reference(block, audio, video, bias, self.amask, self.vmask)
            else:
                out = block.joint_attn(audio, video, mask=self.amask, v_mask=self.vmask, temporal_band_bias=bias)
            (out[0].square().sum() + out[1].square().sum()).backward()
            results.append([audio.grad, video.grad, bias.grad])
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=4e-5)
        for (name, actual), (_, expected) in zip(split.named_parameters(), reference.named_parameters()):
            self.assertEqual(actual.grad is None, expected.grad is None, name)
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=5e-5, msg=name)

    def test_direct_av_is_soft_va_is_blocked_aa_and_vv_stay_global(self):
        block = make_block().eval()
        audio = torch.randn(1, 8, 16, requires_grad=True)
        video = torch.randn(1, 8, 16, requires_grad=True)
        band = VisualPathTemporalBand(dim=16)
        bias = band.bias(*band(video, 8), 8, video_path=torch.arange(8)[None].float() * .1)
        self.assertTrue(torch.isfinite(bias).all())
        out_a, out_v = block.joint_attn(audio, video, temporal_band_bias=bias)
        gx, gv = torch.autograd.grad(out_a[0, 1].square().sum(), (audio, video), retain_graph=True)
        self.assertGreater(gx[0, 7].abs().max().item(), 0, "AA remains global")
        self.assertGreater(gv[0, 7].abs().max().item(), 0, "distant AV remains live under soft penalty")
        gx, gv = torch.autograd.grad(out_v[0, 1].square().sum(), (audio, video), allow_unused=True)
        self.assertTrue(gx is None or not torch.count_nonzero(gx), "VA must be blocked")
        self.assertGreater(gv[0, 7].abs().max().item(), 0, "VV remains global")
        with torch.no_grad():
            changed = block.joint_attn(audio * 7 + 9, video, temporal_band_bias=bias)[1]
            unbanded = block.joint_attn(audio, video)[1]
        torch.testing.assert_close(changed, out_v, atol=0, rtol=0)
        torch.testing.assert_close(unbanded, out_v, atol=0, rtol=0)

    def test_va_flag_changes_no_state_or_rng_and_no_visual_gate_exists(self):
        torch.manual_seed(173)
        open_va = DiT_VT_MMDiT(**BASE_ARCH, **original.PATH_ARCH)
        after_open = torch.get_rng_state()
        torch.manual_seed(173)
        closed_va = DiT_VT_MMDiT(**BASE_ARCH, **ATTENTION_ARCH)
        torch.testing.assert_close(torch.get_rng_state(), after_open, atol=0, rtol=0)
        self.assertEqual(open_va.state_dict().keys(), closed_va.state_dict().keys())
        for key, value in open_va.state_dict().items():
            torch.testing.assert_close(value, closed_va.state_dict()[key], atol=0, rtol=0)
        self.assertTrue(all(block.block_video_audio_attention for block in closed_va.transformer_blocks[:2]))
        self.assertFalse(any("av_visual" in key for key in closed_va.state_dict()))
        self.assertFalse(hasattr(closed_va, "av_local_window_radius"))


class VisualPathNoVaBackboneTests(unittest.TestCase):
    """Re-run original meaningful path contracts with blocked VA."""

    def run_original(self, test_name):
        case = original.VisualPathBackboneTests(test_name)
        case.setUp()
        with patch.object(original, "make_model", side_effect=make_model):
            getattr(case, test_name)()

    def test_cfg_two_three_branch_and_cache_parity(self):
        self.run_original("test_cfg_packed_equals_separate_for_two_and_three_branches_with_cache")

    def test_null_video_and_null_path_do_not_leak(self):
        self.run_original("test_null_video_cfg_does_not_leak_content_or_path_or_batch_neighbors")

    def test_path_changes_are_recomputed_with_cached_text(self):
        self.run_original("test_changed_path_is_recomputed_with_cached_text")

    def test_hidden_prefix_and_padding_do_not_change_geometry(self):
        self.run_original("test_complementary_hidden_prefix_and_padding_cannot_change_geometry")

    def test_static_path_reduces_to_fixed_but_dynamic_path_remains_active(self):
        self.run_original("test_static_path_exact_fixed_reduction_but_real_path_changes_output")

    def test_checkpointed_forward_backward_parity(self):
        self.run_original("test_activation_checkpointing_preserves_output_and_existing_parameter_gradients")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
