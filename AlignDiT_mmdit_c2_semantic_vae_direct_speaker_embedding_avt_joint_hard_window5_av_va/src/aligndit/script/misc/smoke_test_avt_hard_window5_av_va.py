"""Independent checks for strict five-frame AV/VA windows in AVT attention.

Run with PYTHONPATH=src. Pass --cuda for GPU 0 BF16/backward checks after
resource availability has been confirmed. Tests use synthetic inputs only.
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
SOURCE_ROOT = PROJECT_ROOT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va"
GAUSSIAN_SOURCE_ROOT = PROJECT_ROOT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_av_va"
CONFIG_NAME = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window5_av_va"
SOURCE_CONFIG = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va"
GAUSSIAN_ARCH = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "fixed",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.1,
    "temporal_band_bidirectional": True,
}
HARD_ARCH = {**GAUSSIAN_ARCH, "temporal_band_fixed_window_radius_frames": 2}


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **HARD_ARCH, **overrides}))


def load_source_module(name, relative, root=SOURCE_ROOT):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_at(root, name):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base=None):
        return OmegaConf.to_container(compose(config_name=name), resolve=False)


def explicit_joint_attention(block, streams, masks, radius=2):
    """Independently form full logits; no production bias helper or SDPA used."""
    qkv = [[], [], []]
    attentions = (block.attn, block.v_attn, block.t_attn)
    for stream, attention in zip(streams, attentions):
        batch, length, _ = stream.shape
        projected = [projection(stream).reshape(batch, length, attention.heads, -1).transpose(1, 2)
                     for projection in (attention.to_q, attention.to_k, attention.to_v)]
        if attention.q_norm is not None:
            projected[0] = attention.q_norm(projected[0])
        if attention.k_norm is not None:
            projected[1] = attention.k_norm(projected[1])
        for pieces, value in zip(qkv, projected):
            pieces.append(value)
    query, key, value = [torch.cat(pieces, dim=2) for pieces in qkv]
    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    na, nv, nt = [stream.shape[1] for stream in streams]
    allowed = (torch.arange(na)[:, None] - torch.arange(nv)[None, :]).abs() <= radius
    prior = torch.zeros_like(logits)
    prior[:, :, :na, na:na + nv].masked_fill_(~allowed, -torch.inf)
    prior[:, :, na:na + nv, :na].masked_fill_(~allowed.T, -torch.inf)
    valid = [mask if mask is not None and (block.attn_mask_enabled or i == 2)
             else torch.ones(stream.shape[:2], dtype=torch.bool, device=stream.device)
             for i, (stream, mask) in enumerate(zip(streams, masks))]
    prior = prior.masked_fill(~torch.cat(valid, dim=1)[:, None, None], -torch.inf)
    probabilities = (logits + prior).softmax(dim=-1)
    attended = (probabilities @ value).transpose(1, 2).flatten(2)
    outputs = []
    for chunk, attention, mask in zip(attended.split((na, nv, nt), dim=1), attentions, masks):
        output = attention.to_out[1](attention.to_out[0](chunk))
        outputs.append(output if mask is None else output.masked_fill(~mask[..., None], 0.0))
    return outputs, prior, probabilities


def capture_attention(block, streams, **kwargs):
    original, priors, probabilities, calls = F.scaled_dot_product_attention, [], [], []

    def capture(query, key, value, *args, **options):
        mask = options.get("attn_mask", args[0] if args else None)
        shape = (query.shape[0], query.shape[1], query.shape[-2], key.shape[-2])
        prior = query.new_zeros(shape)
        if mask is not None:
            prior = prior.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else prior + mask
        logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        priors.append(prior.detach().clone())
        probabilities.append((logits + prior).softmax(-1).detach())
        calls.append((query.shape[-2], key.shape[-2], None if mask is None else tuple(mask.shape)))
        return original(query, key, value, *args, **options)

    with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=capture):
        outputs = block.joint_attn(*streams, **kwargs)
    return outputs, torch.cat(priors, dim=-2), torch.cat(probabilities, dim=-2), calls


class HardWindowGeometryTests(unittest.TestCase):
    def test_exact_five_positions_interior_clipped_edges_and_inclusive_endpoints(self):
        band = FixedTemporalBand(dim=32, window_radius_frames=2)
        bias = band.bias(*band(torch.randn(2, 23, 32), audio_len=23), video_len=23)
        expected = (torch.arange(23)[:, None] - torch.arange(23)[None, :]).abs() <= 2
        self.assertEqual(bias.dtype, torch.float32)
        torch.testing.assert_close(torch.isfinite(bias), expected[None].expand(2, -1, -1))
        self.assertEqual(torch.count_nonzero(bias[torch.isfinite(bias)]).item(), 0)
        self.assertTrue(torch.isneginf(bias[~expected[None].expand_as(bias)]).all())
        self.assertEqual(expected.sum(-1).tolist(), [3, 4] + [5] * 19 + [4, 3])
        for center in range(5, 18):
            for delta in (-2, 2):
                self.assertEqual(bias[0, center, center + delta].item(), 0)
            for delta in (-3, 3):
                self.assertTrue(torch.isneginf(bias[0, center, center + delta]))

    def test_long_integer_geometry_sigma_and_feature_content_cannot_change_support(self):
        length = 3073
        band = FixedTemporalBand(dim=8, window_radius_frames=2)
        bias = band.bias(*band(torch.full((1, length, 8), torch.nan), audio_len=length), video_len=length)
        self.assertEqual(torch.isfinite(bias[0]).sum(-1).tolist(), [3, 4] + [5] * (length - 4) + [4, 3])
        for center in (40, 511, 1024, 2047, length - 6):
            self.assertTrue(torch.isfinite(bias[0, center, center - 2:center + 3]).all())
            self.assertTrue(torch.isneginf(bias[0, center, center - 3]))
            self.assertTrue(torch.isneginf(bias[0, center, center + 3]))
        alternate = FixedTemporalBand(dim=8, sigma_seconds=17, window_radius_frames=2)
        other = alternate.bias(*alternate(torch.randn(3, 19, 8) * 100, audio_len=21), video_len=19)
        expected = band.bias(*band(torch.zeros(3, 19, 8), audio_len=21), video_len=19)
        torch.testing.assert_close(other, expected, atol=0, rtol=0)
        self.assertFalse(bias.requires_grad)

    def test_radius_none_reproduces_original_gaussian_module_exactly(self):
        source = load_source_module("_source_fixed_band_for_hard5", "src/aligndit/model/fixed_temporal_band.py", GAUSSIAN_SOURCE_ROOT)
        settings = {"dim": 8, "audio_fps": 40, "video_fps": 25, "offset_seconds": 0.05, "sigma_seconds": 0.08}
        old, new = source.FixedTemporalBand(**settings), FixedTemporalBand(**settings, window_radius_frames=None)
        video = torch.randn(2, 17, 8)
        torch.testing.assert_close(new.bias(*new(video, 29), 17), old.bias(*old(video, 29), 17), atol=0, rtol=0)

    def test_invalid_radius_and_hard_window_geometry_rejected(self):
        for value in (-1, True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                FixedTemporalBand(dim=8, window_radius_frames=value)
        for options in ({"video_fps": 25}, {"offset_seconds": 0.025}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FixedTemporalBand(dim=8, window_radius_frames=2, **options)


class HardAVTAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)
        self.block = MMDiTBlock_AVT(dim=32, heads=4, dim_head=8, ff_mult=2, dropout=0.0,
                                   qk_norm="rms_norm", attn_mask_enabled=True).eval()
        self.streams = tuple(torch.randn(2, length, 32) for length in (15, 13, 4))
        self.masks = tuple(torch.arange(s.shape[1])[None] < torch.tensor([s.shape[1], s.shape[1] - 2])[:, None]
                           for s in self.streams)
        band = FixedTemporalBand(dim=32, window_radius_frames=2)
        self.bias = band.bias(*band(self.streams[1], audio_len=15), video_len=13)

    def test_explicit_joint_softmax_unequal_lengths_masks_train_and_eval(self):
        cases = ((None, None, None), self.masks, (self.masks[0], None, self.masks[2]),
                 (None, self.masks[1], self.masks[2]), (None, None, self.masks[2]))
        for training in (False, True):
            for enabled in (False, True):
                self.block.train(training)
                self.block.attn_mask_enabled = enabled
                for masks in cases:
                    with self.subTest(training=training, mask_enabled=enabled,
                                      masks=tuple(m is not None for m in masks)):
                        expected, prior, probabilities = explicit_joint_attention(self.block, self.streams, masks)
                        actual, observed, actual_probabilities, calls = capture_attention(
                            self.block, self.streams, mask=masks[0], v_mask=masks[1],
                            text_mask=masks[2], temporal_band_bias=self.bias)
                        torch.testing.assert_close(observed, prior, atol=0, rtol=0)
                        torch.testing.assert_close(actual_probabilities, probabilities, atol=0, rtol=0)
                        self.assertEqual([(q, k) for q, k, _ in calls], [(15, 32), (13, 32), (4, 32)])
                        self.assertTrue(all(shape[1] == 1 for _, _, shape in calls if shape is not None))
                        for output, reference in zip(actual, expected):
                            torch.testing.assert_close(output, reference, atol=3e-7, rtol=4e-6)

    def test_actual_probabilities_exactly_zero_outside_both_directions_text_and_self_global(self):
        _, prior, probability, _ = capture_attention(self.block, self.streams, temporal_band_bias=self.bias)
        av, va = probability[:, :, :15, 15:28], probability[:, :, 15:28, :15]
        allowed = (torch.arange(15)[:, None] - torch.arange(13)[None, :]).abs() <= 2
        self.assertEqual(torch.count_nonzero(av[..., ~allowed]).item(), 0)
        self.assertEqual(torch.count_nonzero(va[..., ~allowed.T]).item(), 0)
        self.assertTrue((av[..., allowed] > 0).all())
        self.assertTrue((va[..., allowed.T] > 0).all())
        for index in (3, 6, 8):
            for delta in (-2, 2):
                if 0 <= index + delta < 13:
                    self.assertTrue((av[:, :, index, index + delta] > 0).all())
                    self.assertTrue((va[:, :, index + delta, index] > 0).all())
        for chunk in (prior[:, :, :15, :15], prior[:, :, 15:28, 15:28],
                      prior[:, :, 28:, :], prior[:, :, :, 28:]):
            self.assertEqual(torch.count_nonzero(chunk).item(), 0)
        self.assertTrue((probability[:, :, 28:, :] > 0).all())
        self.assertTrue((probability[:, :, :, 28:] > 0).all())
        torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability.sum(-1)))

    def test_outside_cross_modal_features_cannot_change_direct_query_output(self):
        baseline = self.block.joint_attn(*self.streams, temporal_band_bias=self.bias)
        for receiving, sending, query, outside, inside in ((0, 1, 5, 8, 7), (1, 0, 5, 8, 7)):
            changed = [stream.clone() for stream in self.streams]
            changed[sending][:, outside] = torch.randn_like(changed[sending][:, outside]) * 30
            result = self.block.joint_attn(*changed, temporal_band_bias=self.bias)
            torch.testing.assert_close(result[receiving][:, query], baseline[receiving][:, query], atol=0, rtol=0)
            changed[sending][:, inside] = torch.randn_like(changed[sending][:, inside]) * 30
            result = self.block.joint_attn(*changed, temporal_band_bias=self.bias)
            self.assertGreater((result[receiving][:, query] - baseline[receiving][:, query]).abs().max().item(), 1e-6)

    def test_text_padding_hidden_with_or_without_audio_video_padding_policy(self):
        altered = [stream.clone() for stream in self.streams]
        altered[2][~self.masks[2]] = torch.randn_like(altered[2][~self.masks[2]]) * 50
        for enabled in (False, True):
            self.block.attn_mask_enabled = enabled
            expected = self.block.joint_attn(*self.streams, text_mask=self.masks[2], temporal_band_bias=self.bias)
            actual = self.block.joint_attn(*altered, text_mask=self.masks[2], temporal_band_bias=self.bias)
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, atol=0, rtol=0)

    def _assert_pathological_backward(self, device="cpu", autocast=False):
        # Hard masks create rows with no eligible cross-modal key. Even if those
        # queries are padding, SDPA backward must not propagate NaNs.
        for kind in ("all_hidden", "video_only_far_end", "audio_only_far_end", "text_only"):
            with self.subTest(kind=kind, device=device):
                block = copy.deepcopy(self.block).to(device).train()
                streams = [stream.detach().to(device).requires_grad_() for stream in self.streams]
                masks = [torch.zeros(stream.shape[:2], dtype=torch.bool, device=device) for stream in streams]
                if kind == "video_only_far_end":
                    masks[1][:, -1] = True
                elif kind == "audio_only_far_end":
                    masks[0][:, -1] = True
                elif kind == "text_only":
                    masks[2][:, 0] = True
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                    outputs = block.joint_attn(*streams, mask=masks[0], v_mask=masks[1], text_mask=masks[2],
                                               temporal_band_bias=self.bias.to(device))
                    loss = sum(output.float().square().sum() for output in outputs)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for output, mask in zip(outputs, masks):
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(torch.count_nonzero(output[~mask]).item(), 0)
                for index, stream in enumerate(streams):
                    self.assertTrue(torch.isfinite(stream.grad).all(), f"input {index}")
                for name, parameter in block.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_all_hidden_and_no_allowed_cross_key_rows_have_finite_backward(self):
        self._assert_pathological_backward()

    @unittest.skipUnless(RUN_CUDA, "pass --cuda for CUDA BF16 pathological backward")
    def test_cuda_bfloat16_all_hidden_and_no_allowed_cross_key_rows_finite_backward(self):
        self._assert_pathological_backward(device="cuda", autocast=True)


class HardAVTBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_hard_radius_adds_no_parameters_state_or_rng(self):
        models, states = [], []
        for config in ({}, GAUSSIAN_ARCH, {**GAUSSIAN_ARCH, "temporal_band_fixed_window_radius_frames": 4}, HARD_ARCH):
            torch.manual_seed(83)
            models.append(DiT_VT_MMDiT(**{**BASE_ARCH, **config}))
            states.append(torch.get_rng_state().clone())
        self.assertEqual(list(models[-1].temporal_band.parameters()), [])
        self.assertEqual(dict(models[-1].temporal_band.state_dict()), {})
        for model, state in zip(models[1:], states[1:]):
            torch.testing.assert_close(state, states[0], atol=0, rtol=0)
            self.assertEqual(set(model.state_dict()), set(models[0].state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, models[0].state_dict()[name], atol=0, rtol=0, msg=name)
            model.load_state_dict(models[0].state_dict(), strict=True)

    def test_radius_none_matches_original_gaussian_source_outputs_and_rng_exactly(self):
        source = load_source_module("_gaussian_avt_backbone_for_hard5", "src/aligndit/model/backbone/dit_vt_mm.py", GAUSSIAN_SOURCE_ROOT)
        source.FixedTemporalBand = load_source_module("_gaussian_band_for_hard5", "src/aligndit/model/fixed_temporal_band.py", GAUSSIAN_SOURCE_ROOT).FixedTemporalBand
        models, states = [], []
        for cls in (source.DiT_VT_MMDiT, DiT_VT_MMDiT):
            torch.manual_seed(83)
            models.append(warm_start(cls(**{**BASE_ARCH, **GAUSSIAN_ARCH})))
            states.append(torch.get_rng_state().clone())
        torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
        inputs = make_inputs()
        with torch.no_grad():
            for training in (False, True):
                for cfg in (False, True):
                    expected, expected_ctc = models[0].train(training)(**inputs, cfg_infer=cfg)
                    actual, actual_ctc = models[1].train(training)(**inputs, cfg_infer=cfg)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for layer in expected_ctc:
                        for key in ("z_tilde", "z_lens"):
                            torch.testing.assert_close(actual_ctc[layer][key], expected_ctc[layer][key], atol=0, rtol=0)

    def test_geometry_identical_across_cfg_and_dropout_branches_and_all_mm_layers(self):
        model, inputs = make_model(), make_inputs()
        for training in (False, True):
            for flags in ({}, {"drop_video": True}, {"drop_audio_cond": True, "drop_text": True, "drop_video": True},
                          {"cfg_infer": True}, {"cfg_infer": True, "drop_video": True}):
                observed = []
                handles = [block.register_forward_pre_hook(
                    lambda _module, _args, kwargs, observed=observed: observed.append(kwargs), with_kwargs=True)
                    for block in model.transformer_blocks[:2]]
                try:
                    with torch.no_grad():
                        model.train(training)(**inputs, **flags)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(observed), 2)
                expected_allowed = (torch.arange(12)[:, None] - torch.arange(12)[None, :]).abs() <= 2
                for kwargs in observed:
                    bias = kwargs["temporal_band_bias"]
                    torch.testing.assert_close(torch.isfinite(bias), expected_allowed[None].expand_as(bias))
                    self.assertEqual(torch.count_nonzero(bias[torch.isfinite(bias)]).item(), 0)
                    if training:
                        self.assertIsNone(kwargs["mask"])
                        self.assertIsNone(kwargs["v_mask"])
                self.assertIs(observed[0]["temporal_band_bias"], observed[1]["temporal_band_bias"])

    def test_cfg_packed_matches_separate_branches_and_null_video_isolated(self):
        model, inputs = make_model().eval(), make_inputs()
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
            video = inputs["video"].clone()
            video[0] = torch.randn_like(video[0]) * 10
            altered, _ = model(**{**inputs, "video": video}, cfg_infer=True)
        torch.testing.assert_close(original[1:], altered[1:], atol=0, rtol=0)
        self.assertGreater((original[0] - altered[0]).abs().max().item(), 1e-6)

    def test_checkpoint_backward_matches_all_gradients(self):
        plain, checked = make_model().train(), make_model(checkpoint_activations=True).train()
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
        for name, parameter in checked.named_parameters():
            self.assertEqual(parameter.grad is None, reference[name].grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, reference[name].grad, atol=1e-7, rtol=1e-5, msg=name)

    def test_production_config_changes_only_radius_name_path_full_meta_state_and_twelve_plus_six(self):
        target, base = config_at(PROJECT_ROOT, CONFIG_NAME), config_at(SOURCE_ROOT, SOURCE_CONFIG)
        expected = copy.deepcopy(base)
        expected["model"]["name"] = target["model"]["name"]
        expected["ckpts"]["save_dir"] = target["ckpts"]["save_dir"]
        expected["model"]["arch"]["temporal_band_fixed_window_radius_frames"] = 2
        self.assertEqual(target, expected)
        arch = target["model"]["arch"]
        self.assertEqual((arch["depth"], arch["n_mm_layers"], arch["n_text_layers"]), (18, 12, 12))
        self.assertEqual(arch["speaker_condition_start_layer"], 12)
        self.assertEqual(arch["layer_indices_ctc"], [6, 12])
        self.assertEqual(target["optim"]["run_until_update"], 200000)
        # Full-sized architecture on meta: inspect every state shape without
        # allocating model weights or touching a GPU.
        original = load_source_module("_hard9_avt_for_meta_hard5", "src/aligndit/model/backbone/dit_vt_mm.py")
        with torch.device("meta"):
            models = [DiT_VT_MMDiT(**{**arch, "text_num_embeds": 256, "mel_dim": 64}),
                      original.DiT_VT_MMDiT(**{**base["model"]["arch"], "text_num_embeds": 256, "mel_dim": 64})]
        states = [{name: (tuple(value.shape), value.dtype) for name, value in model.state_dict().items()}
                  for model in models]
        self.assertEqual(states[0], states[1])
        self.assertEqual(sum(p.numel() for p in models[0].parameters()), sum(p.numel() for p in models[1].parameters()))
        self.assertEqual(sum(isinstance(block, MMDiTBlock_AVT) for block in models[0].transformer_blocks), 12)
        self.assertTrue(all(type(block) is DiTBlock for block in models[0].transformer_blocks[12:]))
        self.assertFalse(any("temporal_band" in name or "cross_attn" in name for name in states[0]))

    @unittest.skipUnless(RUN_CUDA, "pass --cuda for CUDA BF16 checkpoint backward")
    def test_cuda_bfloat16_checkpoint_backward(self):
        model = make_model(checkpoint_activations=True).cuda().train()
        inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in make_inputs().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, ctc = model(**inputs)
            loss = output.float().square().mean() + sum(item["z_tilde"].float().square().mean() for item in ctc.values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.last_temporal_band_offset_seconds.dtype, torch.float32)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        assert_nonzero_finite(self, model.text_joint_proj.weight.grad, "CUDA text projection")
        assert_nonzero_finite(self, model.transformer_blocks[0].v_attn.to_q.weight.grad, "CUDA video query")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
