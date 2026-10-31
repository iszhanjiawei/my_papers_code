"""TRA mathematics and AVT hard-window integration, without external assets."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_AVT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.repa import masked_repa_cosine_loss, repa_timestep_weights
from aligndit.script.misc.smoke_test_avt_hard_window9_av_va import HARD_ARCH
from aligndit.script.misc.smoke_test_avt_joint import BASE_ARCH, make_inputs, warm_start
from f5_tts.model.modules import DiTBlock


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va"


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backbone(**kwargs):
    return DiT_VT_MMDiT(**{**BASE_ARCH, **HARD_ARCH, **kwargs})


def repa_backbone(**kwargs):
    return backbone(repa_layer=1, repa_target_dim=24, repa_projector_dim=48, **kwargs)


def cfm_kwargs():
    return {
        "num_channels": 64,
        "audio_video_ratio": 1,
        "ctc_lambda": 0.0,
        "text_drop_prob": 0.0,
        "video_drop_prob": 0.0,
        "cond_drop_prob": 0.0,
        "audio_drop_prob": 0.0,
    }


def cfm_inputs():
    inputs = make_inputs()
    return {
        "inp": inputs["x"],
        "text": inputs["text"],
        "video": inputs["video"],
        "speaker_embedding": inputs["speaker_embedding"],
        "lens": torch.tensor([12, 10]),
        "text_lens": torch.tensor([4, 3]),
        "video_lens": torch.tensor([12, 10]),
    }


class ScheduleTests(unittest.TestCase):
    def test_equation_orientation_and_endpoints(self):
        t = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], requires_grad=True)
        w = repa_timestep_weights(t, schedule="taro_clean_fraction", eps=0.01)
        torch.testing.assert_close(w, t.detach() / 1.01)
        self.assertTrue(torch.isfinite(w).all())
        self.assertFalse(w.requires_grad)
        self.assertEqual(w[0].item(), 0.0)
        self.assertTrue((w[1:] > w[:-1]).all())
        interior = torch.sigmoid(torch.log(t.detach()[1:-1] / (1 - t.detach()[1:-1] + 0.01)))
        torch.testing.assert_close(w[1:-1], interior)

    def test_none_and_bfloat16(self):
        t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.bfloat16)
        self.assertEqual(repa_timestep_weights(t).dtype, torch.float32)
        torch.testing.assert_close(repa_timestep_weights(t), torch.ones(3))
        w = repa_timestep_weights(t, schedule="taro_clean_fraction")
        torch.testing.assert_close(w, t.float())

    def test_invalid_time_and_schedule(self):
        for t in (
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([-0.1]),
            torch.tensor([1.1]),
            torch.tensor([[0.5]]),
            torch.tensor([1]),
        ):
            with self.subTest(time=t), self.assertRaises(ValueError):
                repa_timestep_weights(t, schedule="taro_clean_fraction")
        for eps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                repa_timestep_weights(torch.tensor([0.5]), eps=eps)
        with self.assertRaises(ValueError):
            repa_timestep_weights(torch.tensor([0.5]), schedule="inverse_time")


class LossTests(unittest.TestCase):
    def test_unequal_lengths_weights_and_no_weight_normalization(self):
        # Sample 0: one orthogonal frame (loss=1), w=.2.
        # Sample 1: three opposite frames (loss=2 each), w=.8.
        teacher = torch.tensor([[[1.0, 0.0]] * 3] * 2)
        student = torch.tensor([[[0.0, 1.0]] * 3, [[-1.0, 0.0]] * 3], requires_grad=True)
        lens = torch.tensor([1, 3])
        mask = torch.ones(2, 3, dtype=torch.bool)
        loss, stats = masked_repa_cosine_loss(
            student, teacher, lens, lens, mask, sample_weights=torch.tensor([0.2, 0.8]), return_diagnostics=True
        )
        self.assertAlmostEqual(loss.item(), (0.2 * 1 + 0.8 * 6) / 4, places=6)
        self.assertAlmostEqual(stats["repa_unweighted_loss"].item(), 7 / 4, places=6)
        self.assertAlmostEqual(stats["repa_time_weight_frame_mean"].item(), (0.2 + 0.8 * 3) / 4, places=6)
        self.assertAlmostEqual(stats["repa_time_weight_mean"].item(), 0.5, places=6)
        self.assertNotAlmostEqual(loss.item(), 0.5 * 7 / 4, places=6)
        zero = masked_repa_cosine_loss(student, teacher, lens, lens, mask, sample_weights=torch.zeros(2))
        zero.backward()
        self.assertEqual(zero.item(), 0.0)
        self.assertEqual(student.grad.abs().sum().item(), 0.0)

    def test_teacher_padding_prompt_and_student_padding_excluded(self):
        torch.manual_seed(12)
        student = torch.randn(2, 7, 5, requires_grad=True)
        teacher = torch.randn(2, 9, 5, requires_grad=True)
        slens, tlens = torch.tensor([7, 4]), torch.tensor([9, 5])
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[:, :2] = False
        weights = torch.tensor([0.25, 0.75])
        baseline = masked_repa_cosine_loss(student, teacher, tlens, slens, mask, sample_weights=weights)
        changed_s, changed_t = student.detach().clone(), teacher.detach().clone()
        changed_s[:, :2] = float("nan")
        changed_s[1, 4:] = float("nan")
        changed_t[1, 5:] = float("nan")
        actual = masked_repa_cosine_loss(changed_s, changed_t, tlens, slens, mask, sample_weights=weights)
        torch.testing.assert_close(actual, baseline)
        baseline.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[:, :2].count_nonzero().item(), 0)
        self.assertEqual(student.grad[1, 4:].count_nonzero().item(), 0)
        self.assertGreater(student.grad[0, 2:].abs().sum().item(), 0)

    def test_equal_weight_scales_loss_and_gradients(self):
        torch.manual_seed(21)
        s = torch.randn(2, 5, 3, requires_grad=True)
        t = torch.randn(2, 7, 3)
        sl, tl = torch.tensor([5, 4]), torch.tensor([7, 6])
        m = torch.ones(2, 5, dtype=torch.bool)
        raw = masked_repa_cosine_loss(s, t, tl, sl, m)
        raw_grad = torch.autograd.grad(raw, s, retain_graph=True)[0]
        weighted = masked_repa_cosine_loss(s, t, tl, sl, m, sample_weights=torch.full((2,), 0.25))
        grad = torch.autograd.grad(weighted, s)[0]
        torch.testing.assert_close(weighted, raw * 0.25)
        torch.testing.assert_close(grad, raw_grad * 0.25)

    def test_empty_mask_and_invalid_weights_fail(self):
        s, t = torch.randn(2, 3, 4), torch.randn(2, 4, 4)
        sl, tl = torch.tensor([3, 3]), torch.tensor([4, 4])
        with self.assertRaises(RuntimeError):
            masked_repa_cosine_loss(s, t, tl, sl, torch.zeros(2, 3, dtype=torch.bool))
        for w in (torch.tensor([0.2]), torch.tensor([-0.1, 1.0]), torch.tensor([float("nan"), 1.0])):
            with self.subTest(weights=w), self.assertRaises(ValueError):
                masked_repa_cosine_loss(s, t, tl, sl, torch.ones(2, 3, dtype=torch.bool), sample_weights=w)


class ArchitectureTests(unittest.TestCase):
    def test_disabled_source_state_rng_forward_unchanged(self):
        old = source_module("original_hard9_backbone", "src/aligndit/model/backbone/dit_vt_mm.py")
        torch.manual_seed(666)
        source = old.DiT_VT_MMDiT(**{**BASE_ARCH, **HARD_ARCH})
        source_rng = torch.random.get_rng_state()
        torch.manual_seed(666)
        target = backbone()
        torch.testing.assert_close(torch.random.get_rng_state(), source_rng, rtol=0, atol=0)
        self.assertEqual(set(source.state_dict()), set(target.state_dict()))
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
        source = warm_start(source).eval()
        target.load_state_dict(source.state_dict())
        target.eval()
        inputs = make_inputs()
        torch.testing.assert_close(source(**inputs)[0], target(**inputs)[0], rtol=0, atol=0)

    def test_projector_built_last_and_inference_never_calls_it(self):
        torch.manual_seed(666)
        plain = backbone()
        torch.manual_seed(666)
        enabled = repa_backbone()
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(len(set(enabled.state_dict()) - set(plain.state_dict())), 6)
        plain = warm_start(plain).eval()
        enabled.load_state_dict(plain.state_dict(), strict=False)
        enabled.eval()
        inputs = make_inputs()
        with patch.object(enabled.repa_projector, "forward", side_effect=AssertionError("inference called REPA")):
            torch.testing.assert_close(enabled(**inputs)[0], plain(**inputs)[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            enabled(**{**inputs, "cfg_infer": True, "return_repa": True})

    def test_production_layer_topology_and_hard_window_unchanged(self):
        model = backbone(
            depth=18,
            n_mm_layers=12,
            n_text_layers=12,
            speaker_condition_start_layer=12,
            repa_layer=9,
            repa_target_dim=768,
            repa_projector_dim=2048,
        )
        self.assertEqual(sum(isinstance(b, MMDiTBlock_AVT) for b in model.transformer_blocks), 12)
        self.assertEqual(len(model.transformer_blocks), 18)
        self.assertTrue(all(type(b) is DiTBlock for b in model.transformer_blocks[12:]))
        self.assertEqual(model.temporal_band.window_radius_frames, 4)
        output = model(**make_inputs(), return_repa=True)
        self.assertEqual(output[2].shape, (2, 12, 768))

    def test_cfm_schedule_uses_actual_sample_times_and_generation_mask(self):
        model = CFM_VT(
            transformer=warm_start(repa_backbone()),
            **cfm_kwargs(),
            repa_lambda=0.1,
            repa_schedule="taro_clean_fraction",
        )
        inputs = cfm_inputs()
        inputs.update(
            repa_features=torch.randn(2, 15, 24, requires_grad=True), repa_feature_lens=torch.tensor([15, 12])
        )
        requested_span = torch.zeros(2, 12, dtype=torch.bool)
        requested_span[:, 3:] = True
        observed = {}

        def capture(_module, _args, kwargs):
            observed.update(kwargs)

        handle = model.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with (
                patch("aligndit.model.cfm_vt.mask_from_frac_lengths", return_value=requested_span),
                patch("aligndit.model.cfm_vt.masked_repa_cosine_loss", wraps=masked_repa_cosine_loss) as loss_fn,
            ):
                loss, components, _, _ = model(**inputs)
        finally:
            handle.remove()
        expected_mask = requested_span & (torch.arange(12)[None] < inputs["lens"][:, None])
        torch.testing.assert_close(loss_fn.call_args.args[4], expected_mask)
        torch.testing.assert_close(loss_fn.call_args.kwargs["sample_weights"], observed["time"].float())
        self.assertAlmostEqual(loss.item(), components["diff_loss"] + 0.1 * components["repa_loss"], places=6)
        self.assertIn("repa_unweighted_loss", components)
        self.assertIn("repa_time_weight_frame_mean", components)
        loss.backward()
        self.assertIsNone(inputs["repa_features"].grad)
        for layer in model.transformer.repa_projector:
            if isinstance(layer, nn.Linear):
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0.0)

    def test_disabled_cfm_forward_api_and_rng_are_unchanged(self):
        old = source_module("original_hard9_cfm", "src/aligndit/model/cfm_vt.py")
        transformer = warm_start(backbone())
        source = old.CFM_VT(transformer=copy.deepcopy(transformer), **cfm_kwargs())
        target = CFM_VT(transformer=copy.deepcopy(transformer), **cfm_kwargs())
        inputs = cfm_inputs()
        results, states = [], []
        for model in (source, target):
            torch.manual_seed(42)
            with patch.object(model.transformer, "forward", wraps=model.transformer.forward) as forward:
                results.append(model(**inputs))
            self.assertNotIn("return_repa", forward.call_args.kwargs)
            states.append(torch.random.get_rng_state())
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        self.assertEqual(results[0][1], results[1][1])
        for i in (0, 2, 3):
            torch.testing.assert_close(results[0][i], results[1][i], rtol=0, atol=0)

    def test_complete_cfg_sampling_does_not_require_teacher_or_projector(self):
        plain = warm_start(backbone()).eval()
        enabled = repa_backbone().eval()
        enabled.load_state_dict(plain.state_dict(), strict=False)
        base_model = CFM_VT(transformer=plain, **cfm_kwargs())
        tra_model = CFM_VT(transformer=enabled, **cfm_kwargs(), repa_lambda=0.1, repa_schedule="taro_clean_fraction")
        inputs = make_inputs()
        sampling = {
            "cond": inputs["cond"][:1, :3],
            "text": inputs["text"][:1],
            "duration": 12,
            "video": inputs["video"][:1],
            "speaker_embedding": inputs["speaker_embedding"][:1],
            "lens": torch.tensor([3]),
            "steps": 3,
            "use_epss": False,
            "cfg_strength": 5.0,
            "cfg_strength_v": 2.0,
            "seed": 0,
        }
        expected = base_model.sample(**sampling)
        with patch.object(enabled.repa_projector, "forward", side_effect=AssertionError("sampling called REPA")):
            actual = tra_model.sample(**sampling)
        for target, source in zip(actual, expected):
            torch.testing.assert_close(target, source, rtol=0, atol=0)

    def test_checkpoint_activations_matches_backward(self):
        plain = warm_start(repa_backbone()).train()
        checked = copy.deepcopy(plain)
        checked.checkpoint_activations = True
        inputs = make_inputs()
        outputs = []
        for model in (plain, checked):
            out, _, projection = model(**inputs, return_repa=True)
            loss = out.square().mean() + projection.square().mean()
            loss.backward()
            outputs.append((out, projection))
        for x, y in zip(*outputs):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for name, p in plain.named_parameters():
            q = dict(checked.named_parameters())[name]
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7, msg=name)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
