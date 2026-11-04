"""CPU contracts for the isolated LibriTTS text/audio path.

Run with PYTHONPATH=src python -m unittest
aligndit.script.misc.test_text_audio_model -v. No dataset or GPU is needed.
"""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_text_audio import CFMTextAudio
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.script.misc.smoke_test_avt_joint import BASE_ARCH, make_inputs, warm_start
from aligndit.script.misc.smoke_test_avt_fixed_band_av_va import BAND_ARCH


def make_model(**overrides):
    return warm_start(DiT_VT_MMDiT(**{
        **BASE_ARCH, **BAND_ARCH, "text_audio_only": True, **overrides,
    }))


def inputs():
    result = make_inputs()
    result["video"] = None
    for key in ("video_mask", "complementary_mask"):
        result.pop(key)
    result["compute_ctc"] = False
    return result


def cfm(model=None, **kwargs):
    return CFMTextAudio(
        transformer=model or make_model(), num_channels=64,
        mel_spec_module=PrecomputedAudioRepresentation(64, 16000, 400),
        audio_drop_prob=0, cond_drop_prob=0, text_drop_prob=0,
        vocab_char_map={" ": 0, "a": 1, "b": 2}, **kwargs,
    )


class TextAudioModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_default_avt_output_is_unchanged_by_explicit_false_flag(self):
        implicit = warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH})).eval()
        explicit = warm_start(DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH, "text_audio_only": False})).eval()
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        arguments = make_inputs()
        expected, expected_ctc = implicit(**arguments)
        actual, actual_ctc = explicit(**arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(expected_ctc.keys(), actual_ctc.keys())

    def test_all_avt_keys_and_shapes_survive_and_visual_parameters_are_frozen(self):
        avt = DiT_VT_MMDiT(**{**BASE_ARCH, **BAND_ARCH})
        at = make_model()
        self.assertEqual({k: v.shape for k, v in avt.state_dict().items()},
                         {k: v.shape for k, v in at.state_dict().items()})
        avt.load_state_dict(at.state_dict(), strict=True)
        frozen = [name for name, p in at.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen)
        self.assertTrue(all(name.startswith("video_embed.") or ".v_" in name for name in frozen))
        self.assertTrue(all(p.requires_grad for p in avt.parameters()))

    def test_true_12_plus_6_configuration_retains_802_keys(self):
        options = {**BASE_ARCH, **BAND_ARCH, "dim": 768, "depth": 18,
                   "heads": 12, "dim_head": 64, "text_dim": 512,
                   "text_num_embeds": 159, "conv_layers": 4, "use_conformer": True,
                   "n_mm_layers": 12, "n_text_layers": 12, "speaker_condition_start_layer": 12,
                   "layer_indices_ctc": [6, 12], "video_dim": 1024}
        with torch.device("meta"):
            avt = DiT_VT_MMDiT(**options)
            at = DiT_VT_MMDiT(**options, text_audio_only=True)
        self.assertEqual(len(at.state_dict()), 802)
        self.assertEqual({k: v.shape for k, v in avt.state_dict().items()},
                         {k: v.shape for k, v in at.state_dict().items()})

    def test_no_visual_operations_or_band_and_real_text_and_speaker_gradients(self):
        model = make_model().train()
        hooks = []

        def forbid(module, arguments):
            raise AssertionError(f"AT execution touched visual module {type(module).__name__}")

        for name, module in model.named_modules():
            if name.startswith("video_embed") or ".v_" in name or name == "temporal_band":
                hooks.append(module.register_forward_pre_hook(forbid))
        try:
            arguments = inputs()
            output, heads = model(**arguments)
            self.assertEqual(heads, {})
            output.square().mean().backward()
            for parameter in (model.text_embed.text_embed.weight,
                              model.transformer_blocks[0].t_attn.to_v.weight,
                              model.transformer_blocks[1].t_attn.to_v.weight,
                              model.speaker_proj.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                                if name.startswith("video_embed.") or ".v_" in name))
            changed = {**arguments, "text": (arguments["text"] + 1).clamp(max=15)}
            changed["text"] = changed["text"].masked_fill(~arguments["text_mask"], -1)
            altered, _ = model(**changed)
            self.assertGreater((altered - output).abs().max().item(), 1e-6)
        finally:
            for hook in hooks:
                hook.remove()

    def test_audio_and_text_padding_values_cannot_affect_valid_predictions(self):
        model = make_model()
        arguments = inputs()
        altered = copy.deepcopy(arguments)
        altered["x"][~arguments["mask"]] = 1e4
        altered["cond"][~arguments["mask"]] = -1e4
        altered["text"][~arguments["text_mask"]] = 13
        for training in (True, False):
            model.train(training)
            expected, _ = model(**arguments)
            actual, _ = model(**altered)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual[~arguments["mask"]].count_nonzero().item(), 0)

    def test_activation_checkpointing_preserves_output_and_gradients(self):
        regular = make_model().train()
        checkpointed = copy.deepcopy(regular)
        checkpointed.checkpoint_activations = True
        arguments = inputs()
        first, _ = regular(**arguments)
        second, _ = checkpointed(**arguments)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.square().mean().backward()
        second.square().mean().backward()
        for (name, parameter), (other_name, other) in zip(
            regular.named_parameters(), checkpointed.named_parameters(),
        ):
            self.assertEqual(name, other_name)
            if parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, atol=2e-7, rtol=2e-5, msg=name)


class TextAudioFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(8)

    def test_latent_forward_ctc_warmup_and_backprop(self):
        model = cfm(ctc_lambda=0)
        latent = torch.randn(2, 12, 64)
        speaker = torch.randn(2, 192)
        loss, components, cond, prediction = model(
            latent, ["ab ab", "bab"], lens=torch.tensor([12, 10]), speaker_embedding=speaker,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(components["ctc_loss"], 0)
        self.assertEqual(cond.shape, prediction.shape)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.transformer.projectors_ctc.parameters()))
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = 0.03
        loss, components, _, _ = model(latent, ["ab ab", "bab"], speaker_embedding=speaker)
        self.assertGreater(components["ctc_loss"], 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.transformer.projectors_ctc.parameters()))

    def test_full_condition_dropout_also_drops_prompt_and_speaker(self):
        model = cfm(ctc_lambda=0)
        model.cond_drop_prob = 1
        original = model.transformer.forward
        observed = []

        def capture(*args, **kwargs):
            observed.append((kwargs["drop_audio_cond"], kwargs["drop_speaker"], kwargs["drop_text"]))
            return original(*args, **kwargs)

        with patch.object(model.transformer, "forward", side_effect=capture):
            model(torch.randn(1, 8, 64), ["ab"], speaker_embedding=torch.randn(1, 192))
        self.assertEqual(observed, [(True, True, True)])

    def test_sampling_keeps_prompt_padding_and_mode_and_is_seeded(self):
        model = cfm(ctc_lambda=0).train()
        cond = torch.randn(2, 3, 64)
        kwargs = dict(cond=cond, text=["ab ab", "ba"], duration=torch.tensor([9, 7]),
                      lens=torch.tensor([3, 2]), speaker_embedding=torch.randn(2, 192),
                      steps=2, cfg_strength=1.0, seed=42)
        output, trajectory = model.sample(**kwargs)
        repeat, _ = model.sample(**kwargs)
        self.assertTrue(model.training)
        self.assertEqual(output.shape, (2, 9, 64))
        self.assertEqual(trajectory.shape, (3, 2, 9, 64))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, repeat, rtol=0, atol=0)
        torch.testing.assert_close(output[0, :3], cond[0])
        torch.testing.assert_close(output[1, :2], cond[1, :2])
        self.assertEqual(output[1, 7:].count_nonzero().item(), 0)
        empty, _ = model.sample(torch.empty(1, 0, 64), ["ab"], 6,
                                speaker_embedding=kwargs["speaker_embedding"][:1], steps=1, seed=3)
        self.assertEqual(empty.shape, (1, 6, 64))
        self.assertTrue(torch.isfinite(empty).all())

    def test_invalid_text_and_duration_fail_explicitly(self):
        model = cfm(ctc_lambda=0)
        with self.assertRaisesRegex(ValueError, "absent"):
            model(torch.randn(1, 5, 64), ["unknown"], speaker_embedding=torch.randn(1, 192))
        with self.assertRaisesRegex(ValueError, "exceed"):
            model.sample(torch.randn(1, 5, 64), ["ab"], 5, speaker_embedding=torch.randn(1, 192))


if __name__ == "__main__":
    unittest.main()
