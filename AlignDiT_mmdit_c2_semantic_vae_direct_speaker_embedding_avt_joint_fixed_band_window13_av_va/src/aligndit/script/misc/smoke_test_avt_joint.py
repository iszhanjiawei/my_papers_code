"""Data-free regression checks for the speaker experiment's AVT joint attention.

Run from this snapshot with PYTHONPATH=src. Pass --cuda to additionally run a
tiny CUDA BF16 forward/backward. The CPU tests compare attention to explicit
softmax mathematics and use nonzero gates to avoid vacuous residual checks.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_AVT
from f5_tts.model.modules import DiTBlock


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding"
CONFIG_NAME = "finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup"
RUN_CUDA = "--cuda" in sys.argv
if RUN_CUDA:
    sys.argv.remove("--cuda")

BASE_ARCH = {
    "dim": 32,
    "depth": 4,
    "heads": 4,
    "dim_head": 8,
    "dropout": 0.0,
    "ff_mult": 2,
    "mel_dim": 64,
    "text_num_embeds": 16,
    "text_dim": 16,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 1,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "checkpoint_activations": False,
    "use_conformer": False,
    "layer_indices_ctc": [1, 2],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 2,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
    "normalize_text_context": True,
    "speaker_dim": 192,
    "speaker_condition_start_layer": 2,
    "text_joint_attention": True,
}


def make_inputs():
    audio_mask = torch.arange(12)[None, :] < torch.tensor([12, 10])[:, None]
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
        "video_mask": audio_mask.clone(),
        "complementary_mask": audio_mask & ~generation_mask,
        "generation_mask": generation_mask,
        "speaker_embedding": torch.randn(2, 192),
        "cache": False,
    }


def warm_start(model):
    with torch.no_grad():
        for block in model.transformer_blocks:
            for name in ("attn_norm", "v_attn_norm", "t_attn_norm"):
                if hasattr(block, name):
                    getattr(block, name).linear.weight.normal_(std=0.04)
                    getattr(block, name).linear.bias.normal_(std=0.04)
        model.proj_out.weight.normal_(std=0.04)
        model.norm_out.linear.weight.normal_(std=0.04)
        model.speaker_proj.weight.normal_(std=0.04)
    return model


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **overrides}))


def assert_nonzero_finite(test, value, label):
    test.assertIsNotNone(value, f"{label}: missing tensor/gradient")
    test.assertTrue(torch.isfinite(value).all().item(), f"{label}: non-finite values")
    test.assertGreater(torch.count_nonzero(value).item(), 0, f"{label}: all zeros")


def manual_joint_attention(block, streams, masks):
    """Independent Q/K/V projection, one shared softmax, and per-stream output."""
    queries, keys, values = [], [], []
    attentions = (block.attn, block.v_attn, block.t_attn)
    for attention, stream in zip(attentions, streams):
        batch, length, _ = stream.shape
        head_dim = attention.inner_dim // attention.heads
        q, k, v = [
            projection(stream).reshape(batch, length, attention.heads, head_dim).transpose(1, 2)
            for projection in (attention.to_q, attention.to_k, attention.to_v)
        ]
        queries.append(q if attention.q_norm is None else attention.q_norm(q))
        keys.append(k if attention.k_norm is None else attention.k_norm(k))
        values.append(v)
    query, key, value = [torch.cat(pieces, dim=2) for pieces in (queries, keys, values)]
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    key_masks = []
    for index, (stream, mask) in enumerate(zip(streams, masks)):
        enabled = block.attn_mask_enabled or index == 2
        key_masks.append(
            mask if mask is not None and enabled
            else torch.ones(stream.shape[:2], dtype=torch.bool, device=stream.device)
        )
    scores = scores.masked_fill(~torch.cat(key_masks, dim=1)[:, None, None, :], -torch.inf)
    attended = (scores.softmax(dim=-1) @ value).transpose(1, 2).flatten(2)
    results = []
    for attention, mask, chunk in zip(attentions, masks, attended.split([s.shape[1] for s in streams], dim=1)):
        result = attention.to_out[1](attention.to_out[0](chunk))
        if mask is not None:
            result = result.masked_fill(~mask[..., None], 0.0)
        results.append(result)
    return results


class JointAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)
        self.block = MMDiTBlock_AVT(
            dim=32, heads=4, dim_head=8, ff_mult=2, dropout=0.0,
            qk_norm="rms_norm", attn_mask_enabled=True,
        ).eval()
        self.streams = (torch.randn(2, 7, 32), torch.randn(2, 5, 32), torch.randn(2, 4, 32))
        self.masks = tuple(
            torch.arange(stream.shape[1])[None, :] < torch.tensor([stream.shape[1], stream.shape[1] - 2])[:, None]
            for stream in self.streams
        )

    def test_all_three_streams_share_one_softmax(self):
        for enabled in (True, False):
            self.block.attn_mask_enabled = enabled
            expected = manual_joint_attention(self.block, self.streams, self.masks)
            calls = []
            original = F.scaled_dot_product_attention

            def record(query, key, value, *args, calls=calls, original=original, **kwargs):
                calls.append((query.shape[-2], key.shape[-2]))
                return original(query, key, value, *args, **kwargs)

            with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=record):
                actual = self.block.joint_attn(
                    *self.streams, mask=self.masks[0], v_mask=self.masks[1], text_mask=self.masks[2]
                )
            self.assertEqual(calls, [(16, 16)], "AV and text must participate in the same attention")
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=2e-7, rtol=3e-6)

    def test_every_modality_changes_both_other_modalities(self):
        with torch.no_grad():
            original = self.block.joint_attn(*self.streams)
            for changed_stream in range(3):
                altered = list(self.streams)
                altered[changed_stream] = torch.randn_like(altered[changed_stream]) * 3
                actual = self.block.joint_attn(*altered)
                for receiver in range(3):
                    if receiver != changed_stream:
                        self.assertGreater((original[receiver] - actual[receiver]).abs().max().item(), 1e-5)

    def test_text_padding_is_always_hidden_even_without_av_masks(self):
        altered = list(self.streams)
        altered[2] = altered[2].clone()
        altered[2][~self.masks[2]] = torch.randn_like(altered[2][~self.masks[2]]) * 100
        for enabled in (True, False):
            self.block.attn_mask_enabled = enabled
            expected = self.block.joint_attn(*self.streams, text_mask=self.masks[2])
            actual = self.block.joint_attn(*altered, text_mask=self.masks[2])
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(actual[2][~self.masks[2]]).item(), 0)

    def test_empty_text_keys_do_not_create_nan(self):
        text_mask = torch.zeros_like(self.masks[2])
        for output in self.block.joint_attn(*self.streams, text_mask=text_mask):
            self.assertTrue(torch.isfinite(output).all())


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)

    def test_text_is_updated_and_forwarded_between_mm_layers(self):
        model = make_model().eval()
        inputs = make_inputs()
        before, after = [], []

        def pre_hook(_module, _args, kwargs):
            before.append(kwargs["text"].detach().clone())

        def post_hook(_module, _args, output):
            self.assertEqual(len(output), 3)
            after.append(output[2].detach().clone())

        handles = []
        for block in model.transformer_blocks[:2]:
            handles.extend([
                block.register_forward_pre_hook(pre_hook, with_kwargs=True),
                block.register_forward_hook(post_hook),
            ])
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(after), 2)
        torch.testing.assert_close(before[1], after[0], atol=0, rtol=0)
        for initial, updated in zip(before, after):
            self.assertGreater((initial[inputs["text_mask"]] - updated[inputs["text_mask"]]).abs().max().item(), 1e-5)
        self.assertEqual(model.text_joint_proj.in_features, BASE_ARCH["text_dim"])
        self.assertEqual(model.text_joint_proj.out_features, BASE_ARCH["dim"])

    def test_cfg_packing_equals_separate_branches_with_two_samples(self):
        model = make_model().eval()
        inputs = make_inputs()
        unconditional = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
        cases = (
            ({}, ({}, {"drop_video": True}, unconditional)),
            ({"drop_video": True}, ({"drop_video": True}, unconditional)),
            ({"drop_text": True}, ({"drop_text": True}, unconditional)),
        )
        with torch.no_grad():
            for flags, branches in cases:
                packed, packed_ctc = model(**inputs, cfg_infer=True, **flags)
                outputs = [model(**inputs, **branch) for branch in branches]
                torch.testing.assert_close(packed, torch.cat([out[0] for out in outputs]), atol=3e-6, rtol=3e-5)
                for layer in packed_ctc:
                    for key in ("z_tilde", "z_lens"):
                        torch.testing.assert_close(
                            packed_ctc[layer][key], torch.cat([out[1][layer][key] for out in outputs]),
                            atol=3e-6, rtol=3e-5,
                        )

    def test_speaker_changes_only_audio_tail_and_ctc_lengths_are_preserved(self):
        model = make_model().eval()
        inputs = make_inputs()
        observed = []
        handles = [
            block.register_forward_pre_hook(
                lambda _module, args, index=i: observed.append((index, args[2 if index < 2 else 1].detach().clone()))
            )
            for i, block in enumerate(model.transformer_blocks)
        ]
        try:
            with torch.no_grad():
                output, ctc = model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        time = model.time_embed(inputs["time"])
        delta = model.get_speaker_delta(inputs["speaker_embedding"], time)
        assert_nonzero_finite(self, delta, "speaker delta")
        for index, actual in observed:
            torch.testing.assert_close(actual, time if index < 2 else time + delta)
        self.assertEqual(output.shape, (2, 12, 64))
        self.assertEqual(set(ctc), {1, 2})
        for value in ctc.values():
            torch.testing.assert_close(value["z_lens"], torch.tensor([12, 10]))
        torch.testing.assert_close(model.get_speaker_delta(inputs["speaker_embedding"], time, drop_speaker=True), torch.zeros_like(delta))

    def test_checkpointing_matches_forward_and_all_gradients(self):
        plain = make_model().train()
        checked = DiT_VT_MMDiT(**{**BASE_ARCH, "checkpoint_activations": True}).train()
        checked.load_state_dict(plain.state_dict(), strict=True)
        inputs = make_inputs()
        target = torch.randn_like(inputs["x"])
        outputs = []
        for model in (plain, checked):
            output, ctc = model(**inputs)
            loss = (output - target).square().mean()
            loss = loss + 0.01 * sum(item["z_tilde"].square().mean() for item in ctc.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            outputs.append(output.detach())
            for name in (
                "speaker_proj.weight", "text_joint_proj.weight", "transformer_blocks.0.t_attn.to_q.weight",
                "transformer_blocks.0.t_attn.to_k.weight", "transformer_blocks.0.t_attn.to_v.weight",
                "transformer_blocks.0.t_ff.ff.0.0.weight",
            ):
                parameter = dict(model.named_parameters()).get(name)
                self.assertIsNotNone(parameter, name)
                assert_nonzero_finite(self, parameter.grad, name)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = plain_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)

    def test_production_config_keeps_source_settings_and_twelve_plus_six_blocks(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        def config_at(root):
            with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base=None):
                return OmegaConf.to_container(compose(config_name=CONFIG_NAME), resolve=False)

        target = config_at(PROJECT_ROOT)
        if SOURCE_ROOT.is_dir():
            source = config_at(SOURCE_ROOT)
            expected = source.copy()
            # These identify the new isolated run; all training/data/model
            # settings apart from the explicit topology switch remain equal.
            expected["model"]["name"] = target["model"]["name"]
            expected["ckpts"]["save_dir"] = target["ckpts"]["save_dir"]
            expected["model"]["arch"]["text_joint_attention"] = True
            self.assertEqual(target, expected)
        arch = target["model"]["arch"]
        self.assertEqual((arch["depth"], arch["n_mm_layers"], arch["n_text_layers"]), (18, 12, 12))
        self.assertEqual(arch["speaker_condition_start_layer"], 12)
        self.assertEqual(arch["layer_indices_ctc"], [6, 12])
        self.assertTrue(arch["text_joint_attention"])
        model = DiT_VT_MMDiT(**{
            **BASE_ARCH, "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
            "layer_indices_ctc": [6, 12], "speaker_condition_start_layer": 12,
        })
        self.assertEqual(sum(isinstance(block, MMDiTBlock_AVT) for block in model.transformer_blocks), 12)
        self.assertTrue(all(type(block) is DiTBlock for block in model.transformer_blocks[12:]))
        self.assertFalse(any("cross_attn" in name for name, _ in model.named_parameters()))
        self.assertFalse(any("temporal_band" in name for name, _ in model.named_parameters()))
        for block in model.transformer_blocks[:12]:
            for norm, attention in ((block.v_attn_norm, block.v_attn), (block.t_attn_norm, block.t_attn)):
                self.assertEqual(torch.count_nonzero(norm.linear.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(norm.linear.bias).item(), 0)
                assert_nonzero_finite(self, attention.to_out[0].weight, "normally initialized gated branch")

    @unittest.skipUnless(RUN_CUDA, "pass --cuda for optional CUDA BF16 regression")
    def test_cuda_bfloat16_training_backward(self):
        self.assertTrue(torch.cuda.is_available(), "--cuda requested but CUDA is unavailable")
        model = make_model(checkpoint_activations=True).cuda().train()
        inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in make_inputs().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, ctc = model(**inputs)
            loss = output.float().square().mean() + sum(item["z_tilde"].float().square().mean() for item in ctc.values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        assert_nonzero_finite(self, model.text_joint_proj.weight.grad, "CUDA text projection")
        assert_nonzero_finite(self, model.transformer_blocks[0].t_attn.to_q.weight.grad, "CUDA text query")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
