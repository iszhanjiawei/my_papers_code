"""CFM integration checks that do not load datasets, checkpoints or a vocoder."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.script.eval.utils import format_prompt_and_target_text


class FakeMel(nn.Module):
    n_mel_channels = 2


class RecordingTransformer(nn.Module):
    dim = 4
    tpca_enabled = True

    def __init__(self, with_audio_ctc=False):
        super().__init__()
        self.flow_weight = nn.Parameter(torch.tensor(0.2))
        self.visual_weight = nn.Parameter(torch.tensor(2.0))
        self.path_weight = nn.Parameter(torch.tensor(3.0))
        self.audio_logits = nn.Parameter(torch.zeros(5))
        self.text_embed = SimpleNamespace(text_embed=SimpleNamespace(num_embeddings=4))
        self.with_audio_ctc = with_audio_ctc
        self.calls = []
        self.cache_clears = 0
        self.fail_forward = False

    def clear_cache(self):
        self.cache_clears += 1

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_forward:
            raise RuntimeError("intentional ODE failure")
        x = kwargs["x"]
        output = torch.ones_like(x) * self.flow_weight
        if kwargs.get("cfg_infer", False):
            branches = 2 if kwargs.get("drop_text") or kwargs.get("drop_video") else 3
            output = output.repeat(branches, 1, 1)
        aux = {
            "__tpca__": {
                "ctc_loss": self.visual_weight.square(),
                "path_loss": self.path_weight.square(),
                "path_scale": 0.5,
                "feasible_fraction": torch.tensor(0.75),
                "active_fraction": torch.tensor(1.0),
            }
        }
        if self.with_audio_ctc:
            aux[5] = {
                "z_tilde": self.audio_logits.expand(x.shape[0], x.shape[1], -1),
                "z_lens": kwargs["mask"].sum(dim=1),
            }
        return output, aux


def make_model(with_audio_ctc=False, **kwargs):
    transformer = RecordingTransformer(with_audio_ctc=with_audio_ctc)
    defaults = {
        "transformer": transformer,
        "mel_spec_module": FakeMel(),
        "vocab_char_map": {"a": 0, " ": 1, "b": 2},
        "ctc_lambda": 0.1 if with_audio_ctc else 0.0,
        "cond_drop_prob": 0.0,
        "text_drop_prob": 0.0,
        "video_drop_prob": 0.0,
        "audio_drop_prob": 0.0,
        "frac_lengths_mask": (1.0, 1.0),
    }
    defaults.update(kwargs)
    return CFM_VT(**defaults), transformer


class TpcaCfmTests(unittest.TestCase):
    def test_auxiliary_losses_keep_audio_ctc_and_receive_gradients(self):
        model, transformer = make_model(with_audio_ctc=True)
        loss, logs, _, _ = model(
            torch.randn(1, 12, 2),
            text=["ab"],
            video=torch.randn(1, 3, 4),
            lens=torch.tensor([12]),
            text_lens=torch.tensor([2]),
            video_lens=torch.tensor([3]),
        )
        expected = logs["diff_loss"] + 0.1 * logs["ctc_loss"] + 0.03 * 4 + 0.01 * 0.5 * 9
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(logs["tpca_feasible_fraction"], 0.75)
        loss.backward()
        self.assertAlmostEqual(transformer.visual_weight.grad.item(), 0.12, places=6)
        self.assertAlmostEqual(transformer.path_weight.grad.item(), 0.03, places=6)
        self.assertTrue(torch.isfinite(transformer.audio_logits.grad).all())
        self.assertGreater(transformer.audio_logits.grad.abs().sum().item(), 0)

    def test_dropped_text_cannot_train_the_conditioned_auxiliary_path(self):
        model, transformer = make_model(text_drop_prob=0.2)
        with patch("aligndit.model.cfm_vt.random", side_effect=[0.9, 0.1]):
            loss, logs, _, _ = model(
                torch.randn(1, 12, 2),
                text=["ab"],
                video=torch.randn(1, 3, 4),
                text_lens=torch.tensor([2]),
            )
        self.assertTrue(transformer.calls[-1]["drop_text"])
        self.assertEqual(logs["tpca_active"], 0)
        self.assertEqual(logs["tpca_visual_ctc_weighted_loss"], 0)
        self.assertEqual(logs["tpca_path_weighted_loss"], 0)
        self.assertAlmostEqual(loss.item(), logs["diff_loss"], places=6)
        loss.backward()
        self.assertEqual(transformer.visual_weight.grad.item(), 0)
        self.assertEqual(transformer.path_weight.grad.item(), 0)

    def test_batch_sampling_uses_video_rate_masks_and_exact_prefixes(self):
        model, transformer = make_model()
        model.sample(
            cond=torch.randn(2, 8, 2),
            text=["a bb", "aba bb"],
            duration=torch.tensor([12, 20]),
            video=torch.randn(2, 7, 4),
            lens=torch.tensor([4, 8]),
            prompt_text_lens=torch.tensor([2, 4]),
            steps=2,
            cfg_strength=0,
            cfg_strength_v=0,
            use_epss=False,
        )
        call = transformer.calls[0]
        self.assertEqual(call["video"].shape, (2, 5, 4))
        self.assertEqual(call["video_mask"].tolist(), [[True, True, True, False, False], [True] * 5])
        self.assertEqual(
            call["complementary_mask"].tolist(),
            [[True, False, False, False, False], [True, True, False, False, False]],
        )
        self.assertEqual(call["tpca_text_start"].tolist(), [2, 4])
        self.assertEqual(call["tpca_video_start"].tolist(), [1, 2])
        self.assertEqual(transformer.cache_clears, 2)

    def test_partial_last_video_frame_survives_duration_clipping(self):
        model, transformer = make_model()
        model.sample(
            cond=torch.randn(1, 4, 2), text=["a bb"], duration=20,
            video=torch.randn(1, 5, 4), lens=torch.tensor([4]),
            prompt_text_lens=torch.tensor([2]), max_duration=17, steps=2,
            cfg_strength=0, cfg_strength_v=0, use_epss=False,
        )
        self.assertEqual(transformer.calls[0]["video"].shape[1], 5)
        self.assertEqual(transformer.calls[0]["video_mask"].shape[1], 5)

    def test_sampling_requires_unambiguous_prompt_text_boundary(self):
        model, _ = make_model()
        with self.assertRaisesRegex(ValueError, "requires explicit prompt_text_lens"):
            model.sample(cond=torch.randn(1, 4, 2), text=["a bb"], duration=12, video=torch.randn(1, 3, 4))

    def test_failed_ode_clears_the_prior_cache(self):
        model, transformer = make_model()
        transformer.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "intentional ODE failure"):
            model.sample(
                cond=torch.randn(1, 4, 2), text=["a bb"], duration=12,
                video=torch.randn(1, 3, 4), prompt_text_lens=torch.tensor([2]),
                steps=2, cfg_strength=0, cfg_strength_v=0, use_epss=False,
            )
        self.assertEqual(transformer.cache_clears, 2)

    def test_legacy_text_is_unchanged_and_all_separators_are_in_prefix(self):
        joined, start = format_prompt_and_target_text("ab", " ab")
        self.assertEqual(joined, "ab  ab")
        self.assertEqual(start, 4)
        self.assertEqual(joined[start:], "ab")
        joined, start = format_prompt_and_target_text("你好", " 世界")
        self.assertEqual(joined, "你好 世界")
        self.assertEqual(joined[start:], "世界")

    def test_inference_mode_real_backbone_computes_posterior_once_per_sample(self):
        transformer = DiT_VT_MMDiT(
            dim=32, depth=2, heads=4, dim_head=8, dropout=0.0, ff_mult=2,
            mel_dim=2, text_num_embeds=3, text_dim=16, conv_layers=0,
            text_mask_padding=False, qk_norm="rms_norm", pe_attn_head=None,
            attn_mask_enabled=True, use_conformer=False, layer_indices_ctc=[0],
            n_mm_layers=1, n_text_layers=1, prompt_isolated_ca=False,
            audio_video_ratio=4, video_dim=8, text_attention_mode="hunyuan_dual",
            tpca_enabled=True, tpca_layers=[0], tpca_local_heads=2,
            tpca_visual_hidden_dim=8, tpca_warmup_steps=0, tpca_ramp_steps=0,
            tpca_query_chunk_size=4,
        )
        model = CFM_VT(transformer=transformer, mel_spec_module=FakeMel(), ctc_lambda=0.0)
        calls = []

        def record_alignment(_module, args, kwargs):
            fixed_conditions = (*args, kwargs["video_start"], kwargs["text_start"])
            for tensor in fixed_conditions:
                self.assertFalse(torch.is_inference(tensor))
                self.assertFalse(tensor.requires_grad)
            calls.append(tuple(tensor.clone() for tensor in fixed_conditions))

        handle = transformer.tpca_aligner.register_forward_pre_hook(record_alignment, with_kwargs=True)
        try:
            with torch.inference_mode():
                video = torch.randn(1, 5, 8)
                text = torch.tensor([[0, 1, 2, 2]])
                cond = torch.randn(1, 4, 2)
                original_video = video.clone()
                original_text = text.clone()
                arguments = {
                    "cond": cond, "text": text, "duration": 20, "video": video,
                    "prompt_text_lens": torch.tensor([2]), "steps": 4,
                    "cfg_strength": 1.0, "cfg_strength_v": 1.0, "use_epss": False,
                }
                model.sample(**arguments)
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(calls[0][0], original_video, rtol=0, atol=0)
                torch.testing.assert_close(calls[0][1], original_text, rtol=0, atol=0)
                torch.testing.assert_close(video, original_video, rtol=0, atol=0)
                torch.testing.assert_close(text, original_text, rtol=0, atol=0)
                self.assertEqual(calls[0][4].tolist(), [1])
                self.assertEqual(calls[0][5].tolist(), [2])
                self.assertIsNone(transformer._tpca_cached_alignment)
                video.add_(1.0)
                model.sample(**arguments)
                self.assertEqual(len(calls), 2)
                torch.testing.assert_close(calls[1][0], video, rtol=0, atol=0)
                self.assertIsNone(transformer._tpca_cached_alignment)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
