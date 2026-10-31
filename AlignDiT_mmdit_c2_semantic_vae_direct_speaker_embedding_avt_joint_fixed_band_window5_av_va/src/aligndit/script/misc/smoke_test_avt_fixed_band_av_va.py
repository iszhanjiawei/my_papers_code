"""Data-free contracts for AVT joint attention with the fixed AV/VA prior.

Run from this snapshot with PYTHONPATH=src; pass --cuda for BF16 backward.
An independent dense softmax reference checks the complete [A,V,T] matrix.
The neighboring AVT source is read only for an optional exact compatibility
check; production imports remain isolated within this snapshot.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F


RUN_CUDA = "--cuda" in sys.argv
if RUN_CUDA:
    sys.argv.remove("--cuda")

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_AVT
from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.script.misc.smoke_test_avt_joint import BASE_ARCH, assert_nonzero_finite, make_inputs, warm_start
from f5_tts.model.modules import DiTBlock


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint"
CONFIG_NAME = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_av_va"
SOURCE_CONFIG = "finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup"
BAND_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "fixed",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.1,
    "temporal_band_bidirectional": True,
}


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, **overrides}))


def captured_attention(block, streams, **kwargs):
    """Collect the actual additive masks without replacing SDPA mathematics."""
    calls, priors = [], []
    original = F.scaled_dot_product_attention

    def capture(query, key, value, *args, **options):
        mask = options.get("attn_mask", args[0] if args else None)
        calls.append((query.shape[-2], key.shape[-2], None if mask is None else tuple(mask.shape)))
        shape = (query.shape[0], query.shape[1], query.shape[-2], key.shape[-2])
        prior = query.new_zeros(shape)
        if mask is not None:
            prior = prior.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else prior + mask
        priors.append(prior.detach().clone())
        return original(query, key, value, *args, **options)

    with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=capture):
        output = block.joint_attn(*streams, **kwargs)
    return output, torch.cat(priors, dim=-2), calls


def dense_reference(block, streams, masks, pair_bias):
    """Project QKV independently and use one dense, complete joint softmax."""
    qkv = [[], [], []]
    attentions = (block.attn, block.v_attn, block.t_attn)
    for stream, attention in zip(streams, attentions):
        batch, length, _ = stream.shape
        values = [projection(stream).reshape(batch, length, attention.heads, -1).transpose(1, 2)
                  for projection in (attention.to_q, attention.to_k, attention.to_v)]
        if attention.q_norm is not None:
            values[0] = attention.q_norm(values[0])
        if attention.k_norm is not None:
            values[1] = attention.k_norm(values[1])
        for pieces, value in zip(qkv, values):
            pieces.append(value)
    query, key, value = [torch.cat(pieces, dim=2) for pieces in qkv]
    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    na, nv, nt = [stream.shape[1] for stream in streams]
    prior = torch.zeros_like(logits)
    prior[:, :, :na, na:na + nv] = pair_bias[:, None]
    prior[:, :, na:na + nv, :na] = pair_bias.transpose(-1, -2)[:, None]
    valid = [mask if mask is not None and (block.attn_mask_enabled or i == 2)
             else torch.ones(stream.shape[:2], dtype=torch.bool, device=stream.device)
             for i, (stream, mask) in enumerate(zip(streams, masks))]
    prior = prior.masked_fill(~torch.cat(valid, dim=1)[:, None, None], -torch.inf)
    attended = ((logits + prior).softmax(dim=-1) @ value).transpose(1, 2).flatten(2)
    outputs = []
    for chunk, attention, mask in zip(attended.split((na, nv, nt), dim=1), attentions, masks):
        output = attention.to_out[1](attention.to_out[0](chunk))
        outputs.append(output if mask is None else output.masked_fill(~mask[..., None], 0.0))
    return outputs, prior


class FixedAVTAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)
        self.block = MMDiTBlock_AVT(dim=32, heads=4, dim_head=8, ff_mult=2, dropout=0.0,
                                   qk_norm="rms_norm", attn_mask_enabled=True).eval()
        self.streams = tuple(torch.randn(2, length, 32) for length in (7, 5, 4))
        self.masks = tuple(torch.arange(s.shape[1])[None] < torch.tensor([s.shape[1], s.shape[1] - 2])[:, None]
                           for s in self.streams)
        self.band = FixedTemporalBand(dim=32, audio_fps=40, video_fps=25,
                                      offset_seconds=0.05, sigma_seconds=0.08)
        self.bias = self.band.bias(*self.band(self.streams[1], audio_len=7), video_len=5)

    def test_dense_softmax_reference_unequal_lengths_padding_train_eval(self):
        # Physical geometry is calculated independently of FixedTemporalBand.
        expected_bias = -0.5 * ((torch.arange(5)[None, :] / 25 - torch.arange(7)[:, None] / 40 - 0.05) / 0.08).square()
        torch.testing.assert_close(self.bias, expected_bias[None].expand(2, -1, -1))
        cases = ((None, None, None), self.masks, (self.masks[0], None, self.masks[2]),
                 (None, self.masks[1], self.masks[2]), (None, None, self.masks[2]))
        for training in (False, True):
            for enabled in (False, True):
                self.block.train(training)
                self.block.attn_mask_enabled = enabled
                for masks in cases:
                    with self.subTest(training=training, mask_enabled=enabled,
                                      masks=tuple(m is not None for m in masks)):
                        expected, prior = dense_reference(self.block, self.streams, masks, self.bias)
                        actual, observed, calls = captured_attention(
                            self.block, self.streams, mask=masks[0], v_mask=masks[1],
                            text_mask=masks[2], temporal_band_bias=self.bias)
                        torch.testing.assert_close(observed, prior, atol=0, rtol=0)
                        self.assertEqual([(q, k) for q, k, _ in calls], [(7, 16), (5, 16), (4, 16)])
                        self.assertEqual([shape[1] for _, _, shape in calls if shape is not None],
                                         [1] * sum(shape is not None for _, _, shape in calls))
                        for output, reference in zip(actual, expected):
                            torch.testing.assert_close(output, reference, atol=3e-7, rtol=4e-6)

    def test_bias_changes_only_av_va_and_offset_reverse_is_exact_transpose(self):
        _, observed, _ = captured_attention(self.block, self.streams, temporal_band_bias=self.bias)
        torch.testing.assert_close(observed[:, :, :7, 7:12], self.bias[:, None].expand(-1, 4, -1, -1))
        torch.testing.assert_close(observed[:, :, 7:12, :7], self.bias.transpose(-1, -2)[:, None].expand(-1, 4, -1, -1))
        for chunk in (observed[:, :, :7, :7], observed[:, :, 7:12, 7:12],
                      observed[:, :, 12:, :], observed[:, :, :, 12:]):
            self.assertEqual(torch.count_nonzero(chunk).item(), 0)
        band = FixedTemporalBand(dim=32, offset_seconds=0.05, sigma_seconds=0.04)
        bias = band.bias(*band(self.streams[1], audio_len=7), video_len=5)
        _, observed, _ = captured_attention(self.block, self.streams, temporal_band_bias=bias)
        self.assertEqual(bias[0, 2].argmax().item(), 4)
        self.assertEqual(observed[0, 0, 7 + 4, :7].argmax().item(), 2)

    def test_block_text_output_unchanged_but_audio_and_video_are_affected(self):
        with torch.no_grad():
            for name in ("attn_norm", "v_attn_norm", "t_attn_norm"):
                getattr(self.block, name).linear.weight.normal_(std=0.1)
                getattr(self.block, name).linear.bias.normal_(std=0.1)
            time = torch.randn(2, 32)
            args = {"text": self.streams[2], "mask": self.masks[0], "v_mask": self.masks[1], "text_mask": self.masks[2]}
            plain = self.block(*self.streams[:2], time, **args)
            banded = self.block(*self.streams[:2], time, **args, temporal_band_bias=self.bias)
        torch.testing.assert_close(banded[2], plain[2], atol=2e-7, rtol=2e-6)
        for index in (0, 1):
            self.assertGreater((banded[index] - plain[index]).abs().max().item(), 1e-5)

    def test_text_padding_always_hidden_and_empty_video_or_text_is_finite(self):
        altered = list(self.streams)
        altered[2] = altered[2].clone()
        altered[2][~self.masks[2]] = torch.randn_like(altered[2][~self.masks[2]]) * 100
        for enabled in (False, True):
            self.block.attn_mask_enabled = enabled
            expected = self.block.joint_attn(*self.streams, text_mask=self.masks[2], temporal_band_bias=self.bias)
            actual = self.block.joint_attn(*altered, text_mask=self.masks[2], temporal_band_bias=self.bias)
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
            for vm, tm in ((torch.zeros_like(self.masks[1]), self.masks[2]),
                           (self.masks[1], torch.zeros_like(self.masks[2]))):
                output = self.block.joint_attn(*self.streams, v_mask=vm, text_mask=tm, temporal_band_bias=self.bias)
                self.assertTrue(all(torch.isfinite(value).all() for value in output))
                self.assertEqual(torch.count_nonzero(output[1][~vm]).item(), 0)
                self.assertEqual(torch.count_nonzero(output[2][~tm]).item(), 0)

    def test_invalid_bias_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "temporal_band_bias"):
            self.block.joint_attn(*self.streams, temporal_band_bias=torch.zeros(2, 5, 7))


class FixedAVTBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_enabled_geometry_adds_no_parameters_state_or_rng(self):
        models, rng = [], []
        for config in ({}, {"temporal_band_enabled": False}, BAND_ARCH):
            torch.manual_seed(83)
            models.append(DiT_VT_MMDiT(**{**BASE_ARCH, **config}))
            rng.append(torch.get_rng_state().clone())
        self.assertIsNone(models[0].temporal_band)
        self.assertEqual(list(models[2].temporal_band.parameters()), [])
        self.assertEqual(dict(models[2].temporal_band.state_dict()), {})
        for model, state in zip(models[1:], rng[1:]):
            torch.testing.assert_close(state, rng[0], atol=0, rtol=0)
            self.assertEqual(set(model.state_dict()), set(models[0].state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, models[0].state_dict()[name], atol=0, rtol=0, msg=name)
            model.load_state_dict(models[0].state_dict(), strict=True)

    @unittest.skipUnless(SOURCE_ROOT.is_dir(), "original AVT sibling not available")
    def test_disabled_matches_original_source_initialization_rng_and_outputs_exactly(self):
        spec = importlib.util.spec_from_file_location(
            "_original_avt_backbone_for_regression", SOURCE_ROOT / "src/aligndit/model/backbone/dit_vt_mm.py")
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        models, rng = [], []
        for cls in (source.DiT_VT_MMDiT, DiT_VT_MMDiT):
            torch.manual_seed(83)
            models.append(warm_start(cls(**BASE_ARCH)))
            rng.append(torch.get_rng_state().clone())
        torch.testing.assert_close(rng[0], rng[1], atol=0, rtol=0)
        self.assertEqual(set(models[0].state_dict()), set(models[1].state_dict()))
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], atol=0, rtol=0, msg=name)
        inputs = make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for cfg in (False, True):
                    expected, reference_ctc = models[0].train(training)(**inputs, cfg_infer=cfg)
                    actual, actual_ctc = models[1].train(training)(**inputs, cfg_infer=cfg)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in reference_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], reference_ctc[layer][key], atol=0, rtol=0)

    def test_geometry_content_dropout_and_cfg_independent_with_diagnostics(self):
        model = make_model().eval()
        inputs = make_inputs()
        with torch.no_grad():
            for flags in ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True},
                          {"cfg_infer": True}, {"cfg_infer": True, "drop_video": True}):
                altered = {**inputs, "video": torch.randn_like(inputs["video"]) * 7}
                model(**altered, **flags)
                offset, sigma = model.last_temporal_band_offset_seconds, model.last_temporal_band_sigma_seconds
                repetitions = 3 if flags == {"cfg_infer": True} else 2 if flags.get("cfg_infer") else 1
                self.assertEqual(offset.shape, (2 * repetitions, 12))
                torch.testing.assert_close(offset, torch.zeros_like(offset), atol=0, rtol=0)
                torch.testing.assert_close(sigma, torch.full_like(sigma, 0.1), atol=0, rtol=0)
                torch.testing.assert_close(model.last_temporal_band_valid_mask, inputs["mask"].repeat(repetitions, 1))
                self.assertFalse(offset.requires_grad or sigma.requires_grad)
        # Even non-finite feature content cannot change a content-free prior.
        offset, sigma = model.temporal_band(torch.full((2, 9, 32), torch.nan), 12)
        self.assertTrue(torch.isfinite(model.temporal_band.bias(offset, sigma, 9)).all())

    def test_band_reaches_every_mm_layer_in_training_without_av_padding_masks(self):
        model = make_model().train()
        inputs, observed = make_inputs(), []
        handles = [block.register_forward_pre_hook(
            lambda _module, _args, kwargs: observed.append(kwargs), with_kwargs=True)
            for block in model.transformer_blocks[:2]]
        try:
            output, _ = model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(observed), 2)
        for kwargs in observed:
            self.assertIsNone(kwargs["mask"])
            self.assertIsNone(kwargs["v_mask"])
            torch.testing.assert_close(kwargs["text_mask"], inputs["text_mask"])
            self.assertEqual(kwargs["temporal_band_bias"].shape, (2, 12, 12))
            self.assertGreater(kwargs["temporal_band_bias"].abs().max().item(), 0)
        plain = warm_start(DiT_VT_MMDiT(**BASE_ARCH)).train()
        plain.load_state_dict(model.state_dict(), strict=True)
        plain_output, _ = plain(**inputs)
        self.assertGreater((output - plain_output).abs().max().item(), 1e-6)

    def test_cfg_packed_matches_separate_branches_and_null_video_is_isolated(self):
        model = make_model().eval()
        inputs = make_inputs()
        unconditional = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
        cases = (({}, ({}, {"drop_video": True}, unconditional)),
                 ({"drop_video": True}, ({"drop_video": True}, unconditional)),
                 ({"drop_text": True}, ({"drop_text": True}, unconditional)))
        with torch.no_grad():
            for flags, branches in cases:
                packed, packed_ctc = model(**inputs, cfg_infer=True, **flags)
                separate = [model(**inputs, **branch) for branch in branches]
                torch.testing.assert_close(packed, torch.cat([item[0] for item in separate]), atol=3e-6, rtol=3e-5)
                for layer in packed_ctc:
                    for key in ("z_tilde", "z_lens"):
                        torch.testing.assert_close(packed_ctc[layer][key], torch.cat([item[1][layer][key] for item in separate]),
                                                   atol=3e-6, rtol=3e-5)
            original, _ = model(**inputs, cfg_infer=True)
            changed_video = inputs["video"].clone()
            changed_video[0] = torch.randn_like(changed_video[0]) * 7
            changed, _ = model(**{**inputs, "video": changed_video}, cfg_infer=True)
        torch.testing.assert_close(original[1:], changed[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - changed[0]).abs().max().item(), 1e-6)

    def test_checkpoint_backward_matches_all_gradients(self):
        plain = make_model().train()
        checked = make_model(checkpoint_activations=True).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs, outputs = make_inputs(), []
        target = torch.randn_like(inputs["x"])
        for model in (plain, checked):
            output, ctc = model(**inputs)
            loss = (output - target).square().mean() + 0.01 * sum(item["z_tilde"].square().mean() for item in ctc.values())
            loss.backward()
            outputs.append(output.detach())
            for name in ("text_joint_proj.weight", "speaker_proj.weight", "transformer_blocks.0.attn.to_q.weight",
                         "transformer_blocks.0.v_attn.to_q.weight", "transformer_blocks.0.t_attn.to_q.weight"):
                assert_nonzero_finite(self, dict(model.named_parameters())[name].grad, name)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        reference = dict(plain.named_parameters())
        for name, param in checked.named_parameters():
            self.assertEqual(param.grad is None, reference[name].grad is None, name)
            if param.grad is not None:
                torch.testing.assert_close(param.grad, reference[name].grad, atol=1e-7, rtol=1e-5, msg=name)

    def test_invalid_or_unsupported_band_configuration_fails(self):
        for overrides in ({"text_joint_attention": False}, {"temporal_band_bidirectional": False},
                          {"temporal_band_enabled": False}, {"temporal_band_mode": "adaptive"},
                          {"temporal_band_fixed_sigma_seconds": 0.0}, {"temporal_band_video_fps": 25.0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, **overrides})
        for key in ("temporal_band_enabled", "temporal_band_bidirectional"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, key: "false"})

    def test_production_config_only_adds_band_and_keeps_twelve_plus_six(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        def config_at(root, name):
            with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base=None):
                return OmegaConf.to_container(compose(config_name=name), resolve=False)

        target = config_at(PROJECT_ROOT, CONFIG_NAME)
        base = config_at(PROJECT_ROOT, SOURCE_CONFIG)
        expected = copy.deepcopy(base)
        expected["model"]["name"] = target["model"]["name"]
        expected["ckpts"]["save_dir"] = target["ckpts"]["save_dir"]
        expected["model"]["arch"].update(BAND_ARCH)
        self.assertEqual(target, expected)
        if SOURCE_ROOT.is_dir():
            self.assertEqual(base, config_at(SOURCE_ROOT, SOURCE_CONFIG))
        arch = target["model"]["arch"]
        self.assertEqual((arch["depth"], arch["n_mm_layers"], arch["n_text_layers"]), (18, 12, 12))
        self.assertEqual(arch["speaker_condition_start_layer"], 12)
        self.assertEqual(arch["layer_indices_ctc"], [6, 12])
        self.assertEqual(target["optim"]["run_until_update"], 200000)
        model = make_model(depth=18, n_mm_layers=12, n_text_layers=12,
                           layer_indices_ctc=[6, 12], speaker_condition_start_layer=12)
        self.assertEqual(sum(isinstance(block, MMDiTBlock_AVT) for block in model.transformer_blocks), 12)
        self.assertTrue(all(type(block) is DiTBlock for block in model.transformer_blocks[12:]))
        self.assertFalse(any("temporal_band" in name or "cross_attn" in name for name, _ in model.named_parameters()))

    @unittest.skipUnless(RUN_CUDA, "pass --cuda for optional CUDA BF16 regression")
    def test_cuda_bfloat16_checkpoint_backward(self):
        self.assertTrue(torch.cuda.is_available(), "--cuda requested but CUDA is unavailable")
        model = make_model(checkpoint_activations=True).cuda().train()
        inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in make_inputs().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, ctc = model(**inputs)
            loss = output.float().square().mean() + sum(item["z_tilde"].float().square().mean() for item in ctc.values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.last_temporal_band_offset_seconds.dtype, torch.float32)
        self.assertEqual(model.last_temporal_band_sigma_seconds.dtype, torch.float32)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        assert_nonzero_finite(self, model.text_joint_proj.weight.grad, "CUDA text projection")
        assert_nonzero_finite(self, model.transformer_blocks[0].v_attn.to_q.weight.grad, "CUDA video query")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
