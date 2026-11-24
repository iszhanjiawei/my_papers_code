"""Data-free topology and backbone regressions for this experiment's paired attention.

Run with PYTHONPATH=src python -m aligndit.script.misc.smoke_test_paired_joint.
"""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.paired_attention_contract import checkpoint_architecture
from aligndit.script.misc import smoke_test_avt_joint as baseline
from f5_tts.model.modules import DiTBlock


CONFIG = (
    Path(__file__).resolve().parents[2]
    / "config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml"
)
ORDER = OmegaConf.load(CONFIG).model.arch.paired_joint_order
BAND_CONFIG = {
    "temporal_band_enabled": True,
    "temporal_band_mode": "fixed",
    "temporal_band_audio_fps": 40.0,
    "temporal_band_video_fps": 40.0,
    "temporal_band_fixed_offset_seconds": 0.0,
    "temporal_band_fixed_sigma_seconds": 0.150,
    "temporal_band_bidirectional": True,
}


def dense_pair(block, audio, condition, name, masks, ropes, rotated_heads, band=None):
    """Form the full dense attention matrix independently of the grouped implementation."""
    attn = block.attn
    cond_attn = getattr(block, f"{name}_attn")
    n = audio.shape[1]
    half = attn.heads // 2
    start = 0 if name == "v" else half
    stop = start + half
    d = attn.inner_dim // attn.heads

    def project(stream, module, first, last, rotary, rotate_count):
        def heads(linear):
            return linear(stream).unflatten(-1, (module.heads, d)).transpose(1, 2)[:, first:last]

        q, k, v = heads(module.to_q), heads(module.to_k), heads(module.to_v)
        q, k = module.q_norm(q), module.k_norm(k)
        if rotate_count:
            freqs, scale = rotary
            q_scale, k_scale = (1.0, 1.0) if scale is None else (scale, scale**-1)
            q = torch.cat((apply_rotary_pos_emb(q[:, :rotate_count], freqs, q_scale), q[:, rotate_count:]), dim=1)
            k = torch.cat((apply_rotary_pos_emb(k[:, :rotate_count], freqs, k_scale), k[:, rotate_count:]), dim=1)
        return q, k, v

    rotated_audio = min(half, max(0, rotated_heads - start))
    a = project(audio, attn, start, stop, ropes[0], rotated_audio)
    c = project(condition, cond_attn, 0, half, ropes[1], min(half, rotated_heads) if name == "v" else 0)
    q, k, v = (torch.cat(parts, dim=2) for parts in zip(a, c))
    audio_mask, condition_mask = masks
    valid_a = audio_mask if block.attn_mask_enabled else torch.ones_like(audio_mask)
    valid_c = condition_mask if name == "t" or block.attn_mask_enabled else torch.ones_like(condition_mask)
    valid = torch.cat((valid_a, valid_c), dim=-1)
    score = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    if name == "v" and band is not None:
        score[:, :, :n, n:] += band[:, None]
        score[:, :, n:, :n] += band.transpose(-1, -2)[:, None]
    score = score.masked_fill(~valid[:, None, None, :], -torch.inf)
    ctx = (score.softmax(-1) @ v).transpose(1, 2).flatten(2)
    condition_delta = cond_attn.to_out[0](ctx[:, n:]).masked_fill(~condition_mask[..., None], 0)
    audio_delta = F.linear(ctx[:, :n], attn.to_out[0].weight[:, start * d : stop * d])
    return audio_delta.masked_fill(~audio_mask[..., None], 0), condition_delta


class TopologyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(729)
        self.model = baseline.make_model(paired_joint_order=ORDER).eval()
        self.block = self.model.transformer_blocks[0]
        self.a, self.v, self.t = (torch.randn(2, n, 32) for n in (7, 5, 3))
        self.emb = torch.randn(2, 32)
        self.masks = [torch.arange(n)[None, :] < torch.tensor([n, n - 1])[:, None] for n in (7, 5, 3)]
        self.ropes = (self.model.rotary_embed(torch.arange(7)), self.model.rotary_embed(torch.arange(5) * 4))
        with torch.no_grad():
            # A visible bias catches double-counting and accidental placement on the intermediate update.
            self.block.attn.to_out[0].bias.normal_(std=0.3)

    def forward_block(self, a, v, text, **kwargs):
        return self.block(
            a,
            v,
            self.emb,
            mask=self.masks[0],
            v_mask=self.masks[1],
            text=text,
            text_mask=self.masks[2],
            rope=self.ropes[0],
            v_rope=self.ropes[1],
            **kwargs,
        )

    def dense_reference(self, band=None):
        b = self.block
        na, ga, sa, za, fa = b.attn_norm(self.a, self.emb)
        nv, gv, sv, zv, fv = b.v_attn_norm(self.v, self.emb)
        nt, gt, st, zt, ft = b.t_attn_norm(self.t, self.emb)
        order = ("t", "v") if ORDER == "at_then_av" else ("v", "t")
        rotated = b.attn.heads if b.pe_attn_head is None else b.pe_attn_head
        audio_delta = torch.zeros_like(self.a)
        cond_delta = {}
        for i, name in enumerate(order):
            is_video = name == "v"
            delta, cond_delta[name] = dense_pair(
                b,
                na,
                nv if is_video else nt,
                name,
                (self.masks[0], self.masks[1 if is_video else 2]),
                self.ropes,
                rotated,
                band,
            )
            audio_delta = audio_delta + delta
            if i == 0 and ORDER != "av_at_parallel":
                na = b.attn_norm(self.a + ga[:, None] * delta, self.emb)[0]
        bias = b.attn.to_out[0].bias * self.masks[0][..., None]
        a = self.a + ga[:, None] * (audio_delta + bias)
        v = self.v + gv[:, None] * cond_delta["v"]
        text = self.t + gt[:, None] * cond_delta["t"]

        def ffn(x, norm, proj, scale, shift, gate):
            return x + gate[:, None] * proj(norm(x) * (1 + scale[:, None]) + shift[:, None])

        return (
            ffn(a, b.ff_norm, b.ff, za, sa, fa),
            ffn(v, b.v_ff_norm, b.v_ff, zv, sv, fv),
            ffn(text, b.t_ff_norm, b.t_ff, zt, st, ft).masked_fill(~self.masks[2][..., None], 0),
        )

    @torch.no_grad()
    def test_dense_reference_and_two_disjoint_head_groups(self):
        b = self.block
        self.assertEqual((b.attn.heads, b.v_attn.heads, b.t_attn.heads), (4, 2, 2))
        for masked in (True, False):
            with self.subTest(attn_mask_enabled=masked):
                b.attn_mask_enabled = masked
                reference = self.dense_reference()
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as calls:
                    actual = self.forward_block(self.a, self.v, self.t)
                self.assertCountEqual(
                    [tuple(call.args[0].shape[1:3]) for call in calls.call_args_list], [(2, 12), (2, 10)]
                )
                for out, expected in zip(actual, reference):
                    torch.testing.assert_close(out, expected, atol=1.5e-6, rtol=2e-5)

    @torch.no_grad()
    def test_within_block_condition_to_condition_relay_direction(self):
        a, v, t = self.forward_block(self.a, self.v, self.t)
        from_video = self.forward_block(self.a, torch.randn_like(self.v), self.t)
        from_text = self.forward_block(self.a, self.v, torch.randn_like(self.t))
        for audio in (from_video[0], from_text[0]):
            self.assertGreater((a - audio).abs().max().item(), 1e-4)
        for out, expected, connected in (
            (from_video[2], t, ORDER == "av_then_at"),
            (from_text[1], v, ORDER == "at_then_av"),
        ):
            if connected:
                self.assertGreater((out - expected).abs().max().item(), 1e-5)
            else:
                torch.testing.assert_close(out, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_fixed_av_va_gaussian_matches_joint_softmax(self):
        # Unequal lengths and a nonzero offset expose an incorrect reverse direction.
        self.a = torch.randn(2, 19, 32)
        self.v = torch.randn(2, 16, 32)
        self.masks[:2] = [torch.arange(n)[None, :] < torch.tensor([n, n - 4])[:, None] for n in (19, 16)]
        self.ropes = tuple(self.model.rotary_embed(torch.arange(n)) for n in (19, 16))
        for offset_frames in (0, 2):
            band = FixedTemporalBand(32, 40, 40, offset_frames / 40, 0.150)
            actual_bias = band.bias(*band(self.v, 19), video_len=16)
            distance = torch.arange(16)[None, :] - torch.arange(19)[:, None] - offset_frames
            expected_bias = (-0.5 * (distance / 6.0).square())[None].expand(2, -1, -1)
            torch.testing.assert_close(actual_bias, expected_bias)
            center = 7 + offset_frames
            for key in (center - 6, center + 6):
                self.assertAlmostEqual(actual_bias[0, 7, key].item(), -0.5, places=6)
            self.assertLess(actual_bias[0, 18, 0].item(), -0.5)
            self.assertTrue(actual_bias.isfinite().all())
            for use_padding in (True, False):
                self.block.attn_mask_enabled = use_padding
                expected = self.dense_reference(expected_bias)
                actual = self.forward_block(self.a, self.v, self.t, temporal_band_bias=actual_bias)
                for out, ref in zip(actual, expected):
                    torch.testing.assert_close(out, ref, atol=2e-6, rtol=2e-5)

    @torch.no_grad()
    def test_empty_condition_keys_and_padding(self):
        self.masks[2][0] = False
        a, v, text = self.forward_block(self.a, self.v, self.t)
        changed_text = self.t.masked_fill(~self.masks[2][..., None], 1000.0)
        changed_video = self.v.masked_fill(~self.masks[1][..., None], -1000.0)
        other = self.forward_block(self.a, changed_video, changed_text)
        for expected, out, mask in zip((a, v, text), other, self.masks):
            self.assertTrue(out.isfinite().all().item())
            torch.testing.assert_close(out[mask], expected[mask], atol=0, rtol=0)
        self.assertTrue(torch.all(text[~self.masks[2]] == 0))

    def test_repeatable_front_and_audio_only_tail(self):
        for mm in (1, 3):
            model = baseline.make_model(
                paired_joint_order=ORDER,
                depth=mm + 6,
                n_mm_layers=mm,
                n_text_layers=mm,
                speaker_condition_start_layer=mm,
                layer_indices_ctc=[mm - 1, mm],
            )
            front, tail = model.transformer_blocks[:mm], model.transformer_blocks[mm:]
            self.assertEqual(len(front), mm)
            self.assertTrue(all(type(b) is type(self.block) for b in front))
            self.assertEqual(len(tail), 6)
            self.assertTrue(all(type(b) is DiTBlock for b in tail))


class BackboneTests(unittest.TestCase):
    # Reuse the snapshot's existing input fixtures and equivalence tests on its paired model.
    def setUp(self):
        torch.manual_seed(527)
        config_patch = patch.dict(baseline.BASE_ARCH, paired_joint_order=ORDER, **BAND_CONFIG)
        config_patch.start()
        self.addCleanup(config_patch.stop)

    test_cfg_packing_equals_separate_branches = (
        baseline.BackboneTests.test_cfg_packing_equals_separate_branches_with_two_samples
    )
    test_checkpointed_backward_matches_all_gradients = (
        baseline.BackboneTests.test_checkpointing_matches_forward_and_all_gradients
    )
    test_text_propagates_between_blocks = baseline.BackboneTests.test_text_is_updated_and_forwarded_between_mm_layers
    test_speaker_and_ctc = baseline.BackboneTests.test_speaker_changes_only_audio_tail_and_ctc_lengths_are_preserved

    @torch.no_grad()
    def test_band_changes_backbone_output_without_changing_state_dict(self):
        with_band = baseline.make_model().eval()
        plain = baseline.make_model(temporal_band_enabled=False, temporal_band_bidirectional=False).eval()
        plain.load_state_dict(with_band.state_dict(), strict=True)
        inputs = baseline.make_inputs()
        n_audio, n_video = inputs["x"].size(1), inputs["video"].size(1)
        distance = torch.arange(n_video)[None, :] - torch.arange(n_audio)[:, None]
        expected_bias = (-0.5 * (distance / 6.0).square())[None].expand(inputs["x"].size(0), -1, -1)
        observed = []

        def record_band(_module, _args, kwargs):
            observed.append(kwargs["temporal_band_bias"])

        for block in with_band.transformer_blocks[:with_band.n_mm_layers]:
            self.addCleanup(block.register_forward_pre_hook(record_band, with_kwargs=True).remove)
        for training in (True, False):
            observed.clear()
            expected, _ = plain.train(training)(**inputs)
            actual, _ = with_band.train(training)(**inputs)
            self.assertEqual(len(observed), with_band.n_mm_layers)
            for bias in observed:
                torch.testing.assert_close(bias, expected_bias)
            self.assertGreater((expected - actual).abs().max().item(), 1e-6)


class CheckpointIdentityTests(unittest.TestCase):
    def test_grid_rejects_paired_checkpoint_from_celebvdub(self):
        from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import build_model, load_composed_config

        config_path = CONFIG.with_name("finetune_grid.yaml")
        cfg = load_composed_config(config_path)
        celeb_identity = checkpoint_architecture(cfg.model.arch, cfg.ckpts.initialization_mode)
        grid_identity = checkpoint_architecture(
            cfg.model.arch, cfg.ckpts.initialization_mode, dataset_name=cfg.datasets.name,
        )
        self.assertEqual(grid_identity, {**celeb_identity, "dataset": "GRID"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1.pt"
            torch.save({"architecture": celeb_identity}, path)
            with self.assertRaisesRegex(RuntimeError, "Checkpoint architecture mismatch"):
                build_model(config_path, path, 1, torch.device("cpu"))

    def test_geometry_is_required_for_windowed_checkpoint(self):
        from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import build_model

        legacy = {"paired_joint_order": "at_then_av"}
        self.assertIsNone(checkpoint_architecture({}))
        self.assertEqual(checkpoint_architecture(legacy), legacy)
        expected = {**legacy, **BAND_CONFIG}
        cfg = OmegaConf.load(CONFIG)
        self.assertEqual(checkpoint_architecture(cfg.model.arch), expected)
        self.assertEqual(checkpoint_architecture(cfg.model.arch, cfg.ckpts.initialization_mode),
                         {**expected, "parent_stage": "paired_libritts_text_audio"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1.pt"
            for identity in (legacy, expected, {**expected, "temporal_band_fixed_sigma_seconds": 0.100}):
                torch.save({"architecture": identity}, path)
                with self.assertRaisesRegex(RuntimeError, "Checkpoint architecture mismatch"):
                    build_model(CONFIG, path, 1, torch.device("cpu"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
