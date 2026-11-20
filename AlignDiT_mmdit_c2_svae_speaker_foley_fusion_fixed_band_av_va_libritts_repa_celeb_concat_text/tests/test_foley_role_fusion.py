"""Numerical checks for the production two-stream text and AV band paths."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.fixed_temporal_band import FixedTemporalBand


class FoleyConcatFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117)
        self.block = MMDiTBlock_VT(
            dim=16,
            heads=2,
            dim_head=8,
            text_dim=12,
            ff_mult=2,
            dropout=0,
            prompt_isolated_ca=False,
            joint_av_text_ca=True,
            temporal_band_bidirectional=True,
            attn_mask_enabled=True,
        )

    def test_gaussian_both_directions_equals_single_dense_softmax(self):
        a, v = torch.randn(2, 7, 16), torch.randn(2, 5, 16)
        a_valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
        v_valid = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
        band = FixedTemporalBand(16)
        bias = band.bias(*band(v, a.shape[1]), v.shape[1])
        self.assertEqual(len(band.state_dict()), 0)
        torch.testing.assert_close(bias[0, 0, 4], torch.tensor(-0.5))  # 4 frames at 40 Hz = one sigma
        with torch.no_grad():
            actual = self.block.joint_attn(a, v, mask=a_valid, v_mask=v_valid, temporal_band_bias=bias)
            aqkv = self.block._qkv(self.block.attn, a)
            vqkv = self.block._qkv(self.block.v_attn, v)
            q, k, value = [torch.cat(pair, 2) for pair in zip(aqkv, vqkv)]
            logits = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
            logits[:, :, :7, 7:] += bias[:, None]
            logits[:, :, 7:, :7] += bias.transpose(1, 2)[:, None]
            keys = torch.cat((a_valid, v_valid), 1)
            logits.masked_fill_(~keys[:, None, None], float("-inf"))
            mixed = (logits.softmax(-1) @ value).transpose(1, 2).reshape(2, 12, 16)
            for got, ref, valid, attn in zip(
                actual, mixed.split((7, 5), dim=1), (a_valid, v_valid), (self.block.attn, self.block.v_attn)
            ):
                ref = attn.to_out[0](ref).masked_fill(~valid.unsqueeze(-1), 0)
                torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-5)

    def test_both_streams_read_text_in_prompt_and_target_regions(self):
        block = self.block
        a, v, t = torch.randn(2, 7, 16), torch.randn(2, 5, 16), torch.randn(2, 16)
        t1, t2 = torch.randn(2, 4, 12), torch.randn(2, 4, 12)
        a_region = torch.tensor([[False] * 2 + [True] * 5, [False] * 3 + [True] * 2 + [False] * 2])
        v_region = torch.tensor([[False] * 2 + [True] * 3, [False, True, True, False, False]])
        kwargs = {
            "generation_mask": a_region,
            "video_generation_mask": v_region,
            "text_mask": torch.tensor([[True] * 4, [True, True, False, False]]),
        }
        with torch.no_grad():
            # Leave only the two CA residuals enabled.
            for layer in (
                block.attn_norm.linear,
                block.v_attn_norm.linear,
                block.cross_attn_ada,
                block.v_cross_attn_ada,
            ):
                layer.weight.zero_()
                layer.bias.zero_()
            block.cross_attn_ada.bias[-16:].fill_(1)
            block.v_cross_attn_ada.bias[-16:].fill_(1)
            x1, v1 = block(a, v, t, text=t1, **kwargs)
            x2, v2 = block(a, v, t, text=t2, **kwargs)
            for d, region in ((x1 - x2, a_region), (v1 - v2, v_region)):
                self.assertTrue(torch.all(d[region].abs().sum(dim=-1) > 0))
                self.assertTrue(torch.all(d[~region].abs().sum(dim=-1) > 0))
            block.v_cross_attn_ada.bias.zero_()
            x3, v3 = block(a, v, t, text=t1, **kwargs)
            torch.testing.assert_close(x1, x3)
            torch.testing.assert_close(v, v3)

    def test_checkpointed_two_block_flow_and_cfg_use_audio_only_output(self):
        model = DiT_VT_MMDiT(
            dim=32,
            depth=3,
            heads=2,
            dim_head=16,
            ff_mult=2,
            dropout=0,
            mel_dim=8,
            text_num_embeds=32,
            text_dim=16,
            conv_layers=1,
            use_conformer=False,
            layer_indices_ctc=[2],
            ctc_sampling_ratios=(1,),
            n_mm_layers=2,
            n_text_layers=2,
            role_conditioned_text=False,
            prompt_isolated_ca=False,
            audio_video_ratio=1,
            video_dim=12,
            speaker_dim=4,
            attn_mask_enabled=True,
            temporal_band_enabled=True,
            temporal_band_mode="fixed",
            temporal_band_bidirectional=True,
            joint_av_text_ca=True,
            checkpoint_activations=True,
            repa_layer=1,
            repa_target_dim=12,
            repa_projector_dim=32,
        )
        # Simulate learned nonzero gates so the first video CA can influence the next audio block.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not name.endswith(".bias") and param.count_nonzero() == 0:
                    param.normal_(std=0.1)
        valid = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
        generate = torch.tensor([[False] * 2 + [True] * 6, [False] * 3 + [True] * 3 + [False] * 2])
        inputs = {
            "x": torch.randn(2, 8, 8),
            "cond": torch.randn(2, 8, 8),
            "video": torch.randn(2, 8, 12),
            "time": torch.rand(2),
            "text": torch.tensor([[1, 2, 3], [4, 5, -1]]),
            "text_mask": torch.tensor([[True, True, True], [True, True, False]]),
            "video_mask": valid,
            "mask": valid,
            "generation_mask": generate,
            "complementary_mask": valid & ~generate,
            "speaker_embedding": F.normalize(torch.randn(2, 4)),
            "cache": False,
        }
        out, ctc, teacher_projection = model(**inputs, return_repa=True)
        self.assertEqual(out.shape, (2, 8, 8))
        self.assertIsNone(model.role_conditioner)
        self.assertEqual(ctc[2]["z_lens"].tolist(), [8, 6])
        self.assertEqual(teacher_projection.shape, (2, 8, 12))
        out[generate].square().mean().backward()
        grad = model.transformer_blocks[0].cross_attn.v_cond_cross_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.norm() > 0)
        model.eval()
        with torch.no_grad():
            regular, _ = model(**inputs)
            packed, _ = model(**inputs, cfg_infer=True)
            torch.testing.assert_close(packed[:2], regular, rtol=2e-5, atol=2e-5)
            self.assertEqual(packed.shape, (6, 8, 8))

    def test_one_text_sequence_and_raw_prompt_in_sampling_and_training(self):
        net = DiT_VT_MMDiT(
            dim=32, depth=2, heads=2, dim_head=16, ff_mult=2, dropout=0,
            mel_dim=8, text_num_embeds=32, text_dim=16, conv_layers=0,
            use_conformer=False, layer_indices_ctc=[], ctc_sampling_ratios=(1,),
            n_mm_layers=1, n_text_layers=1, role_conditioned_text=False,
            prompt_isolated_ca=False, audio_video_ratio=1, video_dim=12,
            joint_av_text_ca=True, temporal_band_enabled=True, temporal_band_mode="fixed",
            temporal_band_bidirectional=True,
        )
        flow = CFM_VT(
            net, num_channels=8, audio_video_ratio=1, audio_drop_prob=0,
            video_drop_prob=0, text_drop_prob=0, cond_drop_prob=0,
            ctc_lambda=0, vocab_char_map={" ": 0, "a": 1, "b": 2},
        )
        tokens, stem = [], []
        net.text_embed.register_forward_pre_hook(lambda _, args: tokens.append(args[0].detach().clone()))
        net.input_embed.register_forward_pre_hook(
            lambda _, args: stem.append((args[0].detach().clone(), args[1].detach().clone()))
        )
        clip = torch.randn(1, 7, 8)
        video = torch.randn(1, 7, 12)
        prompt = clip[:, :2]
        sampled, _ = flow.sample(
            prompt, ["aa  bb"], 4, video, steps=2, use_epss=False, seed=8,
            cfg_strength=5.0, cfg_strength_v=2.0,
        )
        # Length can expand to the combined character count plus one.
        self.assertEqual(sampled.shape, clip.shape)
        self.assertTrue(torch.equal(sampled[:, :2], prompt))
        self.assertTrue(all(t.tolist() == [[1, 1, 0, 0, 2, 2]] for t in tokens))
        x, cond = stem[0]
        self.assertGreater(x[:, :2].abs().sum().item(), 0)
        torch.testing.assert_close(cond[:, :2], prompt)
        self.assertEqual(cond[:, 2:].count_nonzero().item(), 0)

        tokens.clear()
        stem.clear()
        middle_span = torch.tensor([[False, False, True, True, True, False, False]])
        flow.train()
        with patch("aligndit.model.cfm_vt.mask_from_frac_lengths", return_value=middle_span) as span_sampler:
            loss, _, cond, _ = flow(clip, ["bb"], video)
        span_sampler.assert_called_once()
        torch.testing.assert_close(cond, clip.masked_fill(middle_span.unsqueeze(-1), 0))
        torch.testing.assert_close(stem[0][1], cond)
        self.assertEqual(tokens[0].tolist(), [[2, 2]])
        self.assertTrue(loss.isfinite())
        self.assertFalse(any("role_conditioner" in key for key in flow.state_dict()))


if __name__ == "__main__":
    unittest.main()
