"""C2 + TPCA integration checks at the 40 Hz, one-to-one latent/video grid.

Run from this independent project with ``PYTHONPATH=src python -m unittest
discover -s tests -p test_c2_tpca_integration.py -v``. No data, checkpoint,
audio decoder or background training process is needed.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.trainer_semantic_vae_direct_speaker_tpca import SemanticVaeDirectC2SpeakerTPCATrainer


def tiny_model(**overrides):
    options = dict(
        dim=64, depth=4, heads=4, dim_head=16, dropout=0.0, ff_mult=2,
        mel_dim=8, text_num_embeds=16, text_dim=32, conv_layers=0,
        text_mask_padding=False, qk_norm="rms_norm", pe_attn_head=None,
        attn_mask_enabled=True, use_conformer=False, layer_indices_ctc=[1, 2],
        ctc_sampling_ratios=[1, 1], n_mm_layers=2, n_text_layers=2,
        prompt_isolated_ca=False, audio_video_ratio=1, video_dim=16,
        video_rope_scaled=False, speaker_dim=8, speaker_condition_start_layer=2,
        tpca_enabled=True, tpca_layers=[0, 1], tpca_local_heads=2,
        tpca_visual_hidden_dim=16, tpca_ctc_upsample_factor=2,
        tpca_warmup_steps=2, tpca_ramp_steps=2, tpca_query_chunk_size=5,
    )
    options.update(overrides)
    return DiT_VT_MMDiT(**options)


def inputs():
    mask = torch.arange(24)[None] < torch.tensor([24, 19])[:, None]
    text = torch.tensor([[1, 1, 2, 3, 2], [2, 3, 2, 1, -1]])
    generation = mask.clone()
    generation[:, :4] = False
    return {
        "x": torch.randn(2, 24, 8), "cond": torch.randn(2, 24, 8),
        "video": torch.randn(2, 24, 16), "text": text,
        "time": torch.tensor([0.2, 0.7]), "mask": mask,
        "video_mask": mask.clone(), "text_mask": text != -1,
        "generation_mask": generation,
        "complementary_mask": mask & ~generation,
        "speaker_embedding": torch.randn(2, 8),
    }


def open_gates(model):
    """Emulate nonzero pretrained paths, so speaker/attention tests are useful."""
    with torch.no_grad():
        nn.init.normal_(model.proj_out.weight, std=0.1)
        for block in model.transformer_blocks:
            for name in ("attn_norm", "v_attn_norm"):
                norm = getattr(block, name, None)
                if norm is None:
                    continue
                nn.init.normal_(norm.linear.weight, std=0.015)
                dim = model.dim
                norm.linear.bias[2 * dim:3 * dim].fill_(0.2)
                norm.linear.bias[5 * dim:6 * dim].fill_(0.2)
            if hasattr(block, "cross_attn_ada"):
                nn.init.normal_(block.cross_attn_ada.weight, std=0.015)
                block.cross_attn_ada.bias[2 * model.dim:].fill_(0.3)


class FakeLatentAdapter(nn.Module):
    n_mel_channels = 8


class RecordingTransformer(nn.Module):
    dim = 4
    tpca_enabled = True

    def __init__(self):
        super().__init__()
        self.flow_weight = nn.Parameter(torch.tensor(0.2))
        self.visual_weight = nn.Parameter(torch.tensor(2.0))
        self.path_weight = nn.Parameter(torch.tensor(3.0))
        self.audio_logits = nn.Parameter(torch.zeros(5))
        self.text_embed = SimpleNamespace(text_embed=SimpleNamespace(num_embeddings=4))
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
        return output, {
            "__tpca__": {
                "ctc_loss": self.visual_weight.square(),
                "path_loss": self.path_weight.square(), "path_scale": 0.5,
                "feasible_fraction": torch.tensor(0.75),
                "active_fraction": torch.tensor(1.0),
            },
            1: {"z_tilde": self.audio_logits.expand(x.shape[0], x.shape[1], -1),
                "z_lens": kwargs["mask"].sum(dim=1)},
        }


def recording_cfm(**overrides):
    transformer = RecordingTransformer()
    options = dict(
        transformer=transformer, mel_spec_module=FakeLatentAdapter(),
        vocab_char_map={"a": 0, " ": 1, "b": 2}, ctc_lambda=0.03,
        audio_video_ratio=1, cond_drop_prob=0.0, text_drop_prob=0.0,
        video_drop_prob=0.0, audio_drop_prob=0.0, frac_lengths_mask=(1.0, 1.0),
    )
    options.update(overrides)
    return CFM_VT(**options), transformer


class C2TpcaBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(46)

    def test_new_aligner_does_not_change_seeded_c2_base_initialization(self):
        torch.manual_seed(666)
        baseline = tiny_model(tpca_enabled=False)
        torch.manual_seed(666)
        tpca = tiny_model()
        baseline_state, tpca_state = baseline.state_dict(), tpca.state_dict()
        added = set(tpca_state) - set(baseline_state)
        self.assertEqual(added, {"tpca_step"} | {
            "tpca_aligner." + key for key in tpca.tpca_aligner.state_dict()})
        self.assertEqual(len(added), 9)
        for name, tensor in baseline_state.items():
            torch.testing.assert_close(tensor, tpca_state[name], atol=0, rtol=0, msg=name)

    def test_warmup_and_dropped_branches_preserve_existing_c2_attention(self):
        model = tiny_model().eval()
        open_gates(model)
        disabled = copy.deepcopy(model)
        disabled.tpca_enabled = False
        args = inputs()
        self.assertIsInstance(model.transformer_blocks[0].cross_attn, nn.MultiheadAttention)
        with torch.no_grad():
            torch.testing.assert_close(model(**args)[0], disabled(**args)[0], atol=0, rtol=0)
            model.set_tpca_step(4)
            for drops in ({"drop_text": True}, {"drop_video": True},
                          {"drop_text": True, "drop_video": True, "drop_audio_cond": True}):
                out, aux = model(**args, **drops)
                torch.testing.assert_close(out, disabled(**args, **drops)[0], atol=0, rtol=0)
                self.assertEqual(aux["__tpca__"]["path_loss"].item(), 0)

    def test_padded_batch_two_packed_cfg_matches_individual_speaker_branches(self):
        model = tiny_model().eval()
        open_gates(model)
        model.set_tpca_step(4)
        with torch.no_grad():
            nn.init.normal_(model.speaker_proj.weight, std=0.1)
            args = inputs()
            full = model(**args)[0]
            tts = model(**args, drop_video=True)[0]
            null = model(**args, drop_video=True, drop_text=True, drop_audio_cond=True)[0]
            packed = model(**args, cfg_infer=True)[0]
            torch.testing.assert_close(packed, torch.cat((full, tts, null)), atol=3e-6, rtol=3e-5)
            for ignored in ("drop_video", "drop_text"):
                conditioned = model(**args, **{ignored: True})[0]
                packed_ignored = model(**args, cfg_infer=True, **{ignored: True})[0]
                torch.testing.assert_close(packed_ignored, torch.cat((conditioned, null)), atol=3e-6, rtol=3e-5)

    def test_active_tpca_and_speaker_gradients_and_checkpointing(self):
        plain = tiny_model().train()
        plain.set_tpca_step(4)
        open_gates(plain)
        checkpointed = copy.deepcopy(plain)
        checkpointed.checkpoint_activations = True
        args = inputs()
        values = []
        for model in (plain, checkpointed):
            out, auxiliary = model(**args)
            aux = auxiliary["__tpca__"]
            loss = out[args["generation_mask"]].square().mean()
            loss = loss + 0.03 * aux["ctc_loss"] + 0.01 * aux["path_loss"]
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(aux["path_loss"].item(), 0)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for name, parameter in (
                ("speaker", model.speaker_proj.weight),
                ("aligner", model.tpca_aligner.output_proj.weight),
                ("text query", model.transformer_blocks[0].cross_attn.q_proj_weight),
            ):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            values.append((out, loss))
        for left, right in zip(values[0], values[1]):
            torch.testing.assert_close(left, right)
        for (name, left), (_, right) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, msg=name)

    def test_40hz_prompt_occurrences_are_excluded_from_alignment(self):
        model = tiny_model().eval()
        model.set_tpca_step(4)
        args = inputs()
        args["text"] = torch.tensor([[1, 2, 1, 2, 3], [2, 3, 2, 3, -1]])
        args["text_mask"] = args["text"] != -1
        with torch.inference_mode():
            alignment = model._get_tpca_alignment(
                args["video"], args["text"], args["video_mask"], args["text_mask"],
                torch.tensor([4, 4]), torch.tensor([2, 2]), False,
            )
            self.assertEqual(alignment["prior"][..., :2].abs().sum().item(), 0)
            out, aux = model(**args, tpca_text_start=torch.tensor([2, 2]),
                             tpca_video_start=torch.tensor([4, 4]))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(aux["__tpca__"]["path_loss"]))


class C2TpcaCfmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(46)

    def test_weighted_losses_retain_audio_ctc_and_speaker_condition(self):
        model, transformer = recording_cfm()
        speaker = torch.randn(1, 8)
        loss, logs, _, _ = model(
            torch.randn(1, 12, 8), text=["ab"], video=torch.randn(1, 12, 16),
            speaker_embedding=speaker, lens=torch.tensor([12]),
            text_lens=torch.tensor([2]), video_lens=torch.tensor([12]),
        )
        expected = logs["diff_loss"] + 0.03 * logs["ctc_loss"] + 0.03 * 4 + 0.01 * 0.5 * 9
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertIs(transformer.calls[0]["speaker_embedding"], speaker)
        self.assertFalse(transformer.calls[0]["drop_speaker"])
        loss.backward()
        self.assertAlmostEqual(transformer.visual_weight.grad.item(), 0.12, places=6)
        self.assertAlmostEqual(transformer.path_weight.grad.item(), 0.03, places=6)
        self.assertGreater(transformer.audio_logits.grad.abs().sum().item(), 0)

    def test_dropped_modalities_disable_tpca_and_audio_dropout_drops_speaker(self):
        for condition_value, expected in ((0.9, (False, False, False)),
                                          (0.1, (True, True, True)),
                                          (0.3, (True, False, False)),
                                          (0.5, (False, True, False))):
            model, transformer = recording_cfm(
                cond_drop_prob=0.2, text_drop_prob=0.2, video_drop_prob=0.2)
            with patch("aligndit.model.cfm_vt.random", side_effect=[0.99, condition_value]):
                loss, logs, _, _ = model(
                    torch.randn(1, 12, 8), text=["ab"], video=torch.randn(1, 12, 16),
                    speaker_embedding=torch.randn(1, 8), text_lens=torch.tensor([2]),
                )
            call = transformer.calls[-1]
            self.assertEqual(tuple(call[key] for key in ("drop_text", "drop_video", "drop_speaker")), expected)
            if expected[0] or expected[1]:
                self.assertEqual(logs["tpca_active"], 0)
                self.assertEqual(logs["tpca_visual_ctc_weighted_loss"], 0)
                self.assertEqual(logs["tpca_path_weighted_loss"], 0)
                loss.backward()
                self.assertEqual(transformer.visual_weight.grad.item(), 0)
                self.assertEqual(transformer.path_weight.grad.item(), 0)

    def test_batch_sampling_uses_40hz_masks_and_exact_prefixes(self):
        model, transformer = recording_cfm()
        speaker = torch.randn(2, 8)
        model.sample(
            cond=torch.randn(2, 8, 8), text=["a bb", "aba bb"],
            duration=torch.tensor([12, 20]), video=torch.randn(2, 24, 16),
            lens=torch.tensor([4, 8]), prompt_text_lens=torch.tensor([2, 4]),
            speaker_embedding=speaker, steps=2, cfg_strength=0, cfg_strength_v=0,
            use_epss=False,
        )
        call = transformer.calls[0]
        self.assertEqual(call["video"].shape, (2, 20, 16))
        self.assertEqual(call["video_mask"].sum(1).tolist(), [12, 20])
        self.assertEqual(call["complementary_mask"].sum(1).tolist(), [4, 8])
        self.assertEqual(call["generation_mask"].sum(1).tolist(), [8, 12])
        self.assertEqual(call["tpca_text_start"].tolist(), [2, 4])
        self.assertEqual(call["tpca_video_start"].tolist(), [4, 8])
        self.assertIs(call["speaker_embedding"], speaker)
        self.assertEqual(transformer.cache_clears, 2)

    def test_missing_prefix_rejected_and_ode_failure_clears_cache(self):
        model, transformer = recording_cfm()
        arguments = dict(cond=torch.randn(1, 4, 8), text=["a bb"], duration=12,
                         video=torch.randn(1, 12, 16), speaker_embedding=torch.randn(1, 8))
        with self.assertRaisesRegex(ValueError, "requires explicit prompt_text_lens"):
            model.sample(**arguments)
        transformer.fail_forward = True
        previous_clears = transformer.cache_clears
        with self.assertRaisesRegex(RuntimeError, "intentional ODE failure"):
            model.sample(**arguments, prompt_text_lens=torch.tensor([2]), steps=2,
                         cfg_strength=0, cfg_strength_v=0, use_epss=False)
        self.assertEqual(transformer.cache_clears - previous_clears, 2)

    def test_real_backbone_inference_cache_once_per_ode_and_speaker_retained(self):
        backbone = tiny_model().eval()
        backbone.set_tpca_step(4)
        model = CFM_VT(transformer=backbone, mel_spec_module=FakeLatentAdapter(),
                       audio_video_ratio=1, ctc_lambda=0.0)
        calls = []

        def record_alignment(_module, args, kwargs):
            fixed = (*args, kwargs["video_start"], kwargs["text_start"])
            self.assertTrue(all(not torch.is_inference(tensor) for tensor in fixed))
            calls.append(tuple(tensor.clone() for tensor in fixed))

        handle = backbone.tpca_aligner.register_forward_pre_hook(record_alignment, with_kwargs=True)
        try:
            with torch.inference_mode():
                video = torch.randn(2, 20, 16)
                text = torch.tensor([[0, 1, 2, 2], [1, 2, 1, 2]])
                arguments = dict(
                    cond=torch.randn(2, 4, 8), text=text, duration=torch.tensor([20, 16]),
                    video=video, speaker_embedding=torch.randn(2, 8),
                    prompt_text_lens=torch.tensor([2, 2]), steps=3,
                    cfg_strength=1.0, cfg_strength_v=1.0, use_epss=False,
                )
                model.sample(**arguments)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][4].tolist(), [4, 4])
                self.assertEqual(calls[0][5].tolist(), [2, 2])
                self.assertIsNone(backbone._tpca_cached_alignment)
                video.add_(0.1)
                model.sample(**arguments)
                self.assertEqual(len(calls), 2)
                torch.testing.assert_close(calls[1][0], video, atol=0, rtol=0)
                self.assertIsNone(backbone._tpca_cached_alignment)
        finally:
            handle.remove()


class C2TpcaScheduleTests(unittest.TestCase):
    def test_completed_update_and_ema_schedule_buffers_are_exact(self):
        model = CFM_VT(
            transformer=tiny_model(tpca_warmup_steps=2000, tpca_ramp_steps=8000),
            mel_spec_module=FakeLatentAdapter(), audio_video_ratio=1)
        trainer = object.__new__(SemanticVaeDirectC2SpeakerTPCATrainer)
        trainer.model = model
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value, is_main_process=True)
        trainer.ema_model = SimpleNamespace(ema_model=copy.deepcopy(model))
        trainer.ctc_target_lambda = 0.03
        trainer.ctc_warmup_start = 10000
        trainer.ctc_warmup_end = 30000
        for completed, tpca_scale, next_ctc in (
            (0, 0.0, 0.0), (2000, 0.0, 0.0), (6000, 0.5, 0.0),
            (10000, 1.0, 0.03 / 20000), (29999, 1.0, 0.03),
        ):
            trainer._before_update(completed)
            self.assertEqual(int(model.transformer.tpca_step), completed)
            self.assertEqual(model.transformer.tpca_scale(), tpca_scale)
            self.assertAlmostEqual(model.ctc_lambda, next_ctc, places=12)
            trainer._after_update(completed + 1)
            self.assertEqual(int(model.transformer.tpca_step), completed + 1)
            self.assertEqual(int(trainer.ema_model.ema_model.transformer.tpca_step), completed + 1)


if __name__ == "__main__":
    unittest.main()
