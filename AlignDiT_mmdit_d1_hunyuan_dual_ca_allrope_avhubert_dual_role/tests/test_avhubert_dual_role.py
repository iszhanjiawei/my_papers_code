"""CPU regression checks; no AV-HuBERT weights, datasets, or GPU are needed.

Run from the experiment root:
    PYTHONPATH=src python -m unittest discover -s tests -p test_avhubert_dual_role.py -v
"""

import unittest

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT, _masked_audio_representation_loss


def tiny_config():
    return dict(
        dim=64, depth=2, heads=4, dim_head=16, ff_mult=2, mel_dim=8,
        text_num_embeds=12, text_dim=32, qk_norm="rms_norm", dropout=0.0,
        conv_layers=0, use_conformer=False, layer_indices_ctc=[0], n_mm_layers=1,
        n_text_layers=1, prompt_isolated_ca=False, audio_video_ratio=4,
        video_dim=16, pe_attn_head=None, text_attention_mode="hunyuan_dual",
    )


def tiny_model(enabled=True):
    options = dict(avhubert_rep_layer=0, avhubert_rep_dim=16) if enabled else {}
    return DiT_VT_MMDiT(**tiny_config(), **options)


def tiny_inputs():
    return dict(
        x=torch.randn(2, 32, 8), cond=torch.randn(2, 32, 8),
        text=torch.randint(0, 12, (2, 3)), video=torch.randn(2, 8, 16),
        time=torch.tensor([0.3, 0.7]),
        mask=torch.arange(32)[None, :] < torch.tensor([32, 24])[:, None],
        text_mask=torch.ones(2, 3, dtype=torch.bool),
        video_mask=torch.arange(8)[None, :] < torch.tensor([8, 6])[:, None],
        generation_mask=torch.ones(2, 32, dtype=torch.bool),
    )


def tiny_cfm(model=None, weight=0.25):
    return CFM_VT(
        transformer=tiny_model() if model is None else model,
        num_channels=8, ctc_lambda=0.1, avhubert_rep_lambda=weight,
        audio_drop_prob=0.0, cond_drop_prob=0.0,
        text_drop_prob=0.0, video_drop_prob=0.0, frac_lengths_mask=(0.7, 0.9),
    )


class AVHubertRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_cosine_oracle_mask_and_frozen_teacher(self):
        student = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
        # Teacher's first row is prompt, fourth is padding, and fifth lies
        # outside the student's timeline. None may enter the cosine loss.
        teacher = torch.tensor([
            [[float("nan"), float("nan")], [0., 1.], [0., 1.], [float("nan"), 0.], [5., 6.]]
        ], requires_grad=True)
        mask = torch.tensor([[False, True, True, True]])
        loss, frames = _masked_audio_representation_loss(student, teacher, torch.tensor([3]), mask)
        self.assertEqual(frames, 2)
        # Two valid pairs have cosine distances 1 and 0 respectively.
        self.assertAlmostEqual(loss.item(), 0.5, places=7)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(student.grad[0, 3].abs().sum().item(), 0)
        self.assertGreater(student.grad[0, 1].abs().sum().item(), 0)

    def test_empty_mask_retains_projector_graph(self):
        projector = torch.nn.Linear(3, 2)
        student = projector(torch.randn(1, 4, 3))
        loss, frames = _masked_audio_representation_loss(
            student, torch.randn(1, 5, 2), torch.tensor([0]), torch.ones(1, 4, dtype=torch.bool)
        )
        self.assertEqual(frames, 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for parameter in projector.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_disabled_model_and_seeded_original_parameters_unchanged(self):
        base = tiny_model(enabled=False)
        torch.manual_seed(42)
        supervised = tiny_model()
        self.assertFalse(any("avhubert_rep_projector" in key for key in base.state_dict()))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, supervised.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(len(base(**tiny_inputs())), 2)

    def test_tap_fixed_pooling_and_default_inference_signature(self):
        model = tiny_model().eval()
        inputs = tiny_inputs()
        self.assertEqual(len(model(**inputs)), 2)
        captured = {}

        def capture_block(module, args, output):
            captured["audio"] = output[0]

        hook = model.transformer_blocks[0].register_forward_hook(capture_block)
        try:
            _, ctc, features = model(**inputs, return_audio_features=True)
        finally:
            hook.remove()
        # Explicit slices are an independent expression of fixed 4:1 pooling.
        audio = captured["audio"]
        pooled = torch.stack([audio[:, start:start + 4].mean(1) for start in range(0, 32, 4)], dim=1)
        torch.testing.assert_close(features, model.avhubert_rep_projector(pooled))
        self.assertEqual(features.shape, (2, 8, 16))
        self.assertEqual(set(ctc), {0})

    def test_combined_cfm_loss_and_gradients(self):
        model = tiny_model()
        cfm = tiny_cfm(model).train()
        inputs = tiny_inputs()
        teacher = torch.randn(2, 9, 16, requires_grad=True)
        loss, components, _, _ = cfm(
            inputs["x"], inputs["text"], inputs["video"],
            lens=torch.tensor([32, 24]), text_lens=torch.tensor([3, 3]), video_lens=torch.tensor([8, 6]),
            audio_teacher_features=teacher, audio_teacher_lengths=torch.tensor([8, 5]),
        )
        self.assertTrue(torch.isfinite(loss))
        expected = components["diff_loss"] + 0.1 * components["ctc_loss"] + components["avhubert_rep_weighted_loss"]
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertAlmostEqual(components["avhubert_rep_weighted_loss"], 0.25 * components["avhubert_rep_loss"])
        self.assertGreater(components["avhubert_rep_valid_frames"], 0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        for parameter in model.avhubert_rep_projector.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().max().item(), 0)
        self.assertGreater(model.input_embed.proj.weight.grad.abs().max().item(), 0)

    def test_sampling_needs_no_teacher_and_does_not_run_projector(self):
        cfm = tiny_cfm().eval()

        def forbidden_projection(module, args):
            raise AssertionError("training-only projector was called during inference")

        hook = cfm.transformer.avhubert_rep_projector.register_forward_pre_hook(forbidden_projection)
        prompt = torch.randn(1, 8, 8)
        try:
            generated, _ = cfm.sample(
                cond=prompt, text=torch.tensor([[1, 2, 3]]), duration=16,
                video=torch.randn(1, 4, 16), steps=2, use_epss=False, seed=7,
                cfg_strength=1.0, cfg_strength_v=1.0,
            )
        finally:
            hook.remove()
        self.assertEqual(generated.shape, (1, 16, 8))
        self.assertTrue(torch.isfinite(generated).all())
        torch.testing.assert_close(generated[:, :8], prompt)

    def test_invalid_arguments_fail_explicitly(self):
        for layer in (-1, 2, True):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                DiT_VT_MMDiT(**tiny_config(), avhubert_rep_layer=layer)
        with self.assertRaisesRegex(ValueError, "audio_teacher_features"):
            inputs = tiny_inputs()
            tiny_cfm()(inputs["x"], inputs["text"], inputs["video"])
        with self.assertRaisesRegex(ValueError, "avhubert_rep_layer"):
            tiny_cfm(tiny_model(enabled=False))
        for weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                tiny_cfm(weight=weight)
        with self.assertRaisesRegex(ValueError, "padded teacher length"):
            _masked_audio_representation_loss(
                torch.ones(1, 3, 2), torch.ones(1, 3, 2), torch.tensor([4]),
                torch.ones(1, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
