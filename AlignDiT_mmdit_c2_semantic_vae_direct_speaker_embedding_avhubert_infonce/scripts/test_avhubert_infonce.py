"""CPU mechanism checks. Run from this snapshot with PYTHONPATH=src."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from aligndit.model.avhubert_infonce import sample_context_on_teacher_grid, temporal_context_infonce
from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT


def make_model(*, alignment=True, checkpoint=False):
    return DiT_VT_MMDiT(
        dim=32, depth=13, heads=4, dim_head=8, ff_mult=2, mel_dim=64,
        text_num_embeds=16, text_dim=16, text_mask_padding=False,
        qk_norm="rms_norm", conv_layers=1, pe_attn_head=1,
        attn_mask_enabled=True, checkpoint_activations=checkpoint,
        use_conformer=False, layer_indices_ctc=[], ctc_sampling_ratios=[1, 1],
        n_mm_layers=12, n_text_layers=12, prompt_isolated_ca=False,
        audio_video_ratio=1, video_dim=16, video_rope_scaled=False, dropout=0.0,
        context_alignment_layer=11 if alignment else None, context_alignment_dim=32,
    )


def inputs():
    return {
        "x": torch.randn(1, 40, 64), "cond": torch.randn(1, 40, 64),
        "text": torch.tensor([[1, 2, 3, 4]]), "video": torch.randn(1, 40, 16),
        "time": torch.tensor([0.4]), "mask": torch.ones(1, 40, dtype=torch.bool),
        "text_mask": torch.ones(1, 4, dtype=torch.bool), "video_mask": torch.ones(1, 40, dtype=torch.bool),
        "generation_mask": torch.ones(1, 40, dtype=torch.bool), "cache": False,
    }


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.student = torch.randn(1, 40, 32, requires_grad=True)
        self.mask = torch.ones(1, 40, dtype=torch.bool)
        self.lengths = torch.tensor([40])
        self.teacher, _ = sample_context_on_teacher_grid(self.student.detach(), self.lengths, self.mask, 25)

    def loss(self, student=None, teacher=None, *, lengths=None, teacher_lengths=None, valid=None, mask=None, **kw):
        teacher = self.teacher if teacher is None else teacher
        teacher_lengths = torch.tensor([teacher.shape[1]]) if teacher_lengths is None else teacher_lengths
        return temporal_context_infonce(
            self.student if student is None else student, teacher,
            self.lengths if lengths is None else lengths, teacher_lengths,
            teacher_lengths if valid is None else valid, self.mask if mask is None else mask, **kw,
        )

    def test_grid_uses_fixed_centres_and_both_generation_positions(self):
        student = torch.arange(40).float()[None, :, None]
        generation = self.mask.clone()
        generation[:, 3] = False
        sampled, valid = sample_context_on_teacher_grid(student, self.lengths, generation, 30)
        torch.testing.assert_close(sampled[0, :24, 0], (torch.arange(24) + 0.5) * 1.6 - 0.5)
        self.assertFalse(valid[0, 2])  # Frame 2 interpolates positions 3 and 4.
        self.assertFalse(valid[0, 25])  # No extrapolation past the 40 Hz sequence.

    def test_padding_and_invalid_teacher_tail_cannot_change_loss(self):
        reference, reference_stats = self.loss(teacher=self.teacher[:, :18], valid=torch.tensor([18]))
        student = F.pad(self.student.detach(), (0, 0, 0, 24), value=float("nan")).requires_grad_()
        teacher = F.pad(self.teacher[:, :18], (0, 0, 0, 31), value=float("nan"))
        mask = F.pad(self.mask, (0, 24), value=False)
        actual, stats = self.loss(student, teacher, valid=torch.tensor([18]), mask=mask)
        torch.testing.assert_close(reference, actual)
        self.assertEqual(reference_stats, stats)
        # Adding another, longer (all-prompt) clip changes no anchors or keys.
        batched, batched_stats = self.loss(
            torch.cat((student, torch.randn_like(student))),
            torch.cat((teacher, torch.randn_like(teacher))),
            lengths=torch.tensor([40, 64]), teacher_lengths=torch.tensor([49, 49]),
            valid=torch.tensor([18, 39]), mask=torch.cat((mask, torch.zeros_like(mask))),
        )
        torch.testing.assert_close(reference, batched)
        self.assertEqual(reference_stats, batched_stats)

    def test_shifted_teacher_is_harder_and_teacher_has_no_gradient(self):
        teacher = self.teacher.detach().requires_grad_()
        aligned, _ = self.loss(teacher=teacher)
        shifted, _ = self.loss(teacher=teacher.roll(6, dims=1))
        self.assertGreater(shifted.item(), aligned.item() + 1.0)
        shifted.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(self.student.grad.norm().item(), 0)

    def test_exclusion_band_includes_distance_five(self):
        student = torch.zeros(1, 16, 7, requires_grad=True)
        with torch.no_grad():
            student[:, :, 0] = 1
        teacher = torch.eye(7)[None]
        teacher[:, 1:6] = teacher[:, :1]  # Offsets 1..4 ignored; offset 5 is a hard negative.
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[:, :2] = True  # Exactly one eligible anchor, at teacher index 0.
        value, stats = temporal_context_infonce(
            student, teacher, torch.tensor([16]), torch.tensor([7]), torch.tensor([7]), mask,
        )
        expected = F.cross_entropy(torch.tensor([[1.0, 1.0, 0.0]]) / 0.07, torch.tensor([0]))
        torch.testing.assert_close(value, expected)
        self.assertEqual(stats["infonce_valid_anchors"], 1)

    def test_disabled_empty_and_no_negative_have_connected_zero(self):
        for kwargs in ({"enabled": False}, {"valid": torch.tensor([0])}, {"valid": torch.tensor([5])},
                       {"mask": torch.zeros_like(self.mask)}):
            self.student.grad = None
            loss, stats = self.loss(**kwargs)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats["infonce_valid_anchors"], 0)
            loss.backward()
            self.assertIsNotNone(self.student.grad)
            self.assertEqual(self.student.grad.abs().sum().item(), 0)

    def test_autocast_does_not_reduce_loss_precision(self):
        reference, _ = self.loss()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, _ = self.loss()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


class ModelTests(unittest.TestCase):
    def test_initialization_preserved_and_raw_tap_precedes_zero_gate(self):
        torch.manual_seed(3)
        baseline = make_model(alignment=False).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(3)
        aligned = make_model().eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, aligned.state_dict()[key]), key)
        data = inputs()
        captured = {}
        handle = aligned.transformer_blocks[11].cross_attn.register_forward_hook(
            lambda _module, _args, output: captured.update(raw=output[0])
        )
        try:
            result = aligned(**data, return_context_alignment=True)
        finally:
            handle.remove()
        self.assertEqual(len(result), 3)
        torch.testing.assert_close(result[2], aligned.context_alignment_projector(captured["raw"]))
        self.assertGreater(result[2].abs().sum().item(), 0)
        self.assertEqual(aligned.transformer_blocks[11].cross_attn_ada.weight.abs().sum().item(), 0)
        self.assertEqual(len(aligned(**data)), 2)
        torch.testing.assert_close(result[0], baseline(**data)[0])
        cfg_result = aligned(**data, cfg_infer=True)
        self.assertEqual(cfg_result[0].shape[0], 3)

    def test_checkpoint_projection_and_gradient_match(self):
        torch.manual_seed(8)
        normal = make_model().train()
        checkpointed = copy.deepcopy(normal)
        checkpointed.checkpoint_activations = True
        data = inputs()
        teacher = torch.randn(1, 25, 32, requires_grad=True)
        outputs = []
        for model in (normal, checkpointed):
            _, _, projected = model(**data, return_context_alignment=True)
            loss, stats = temporal_context_infonce(
                projected, teacher, torch.tensor([40]), torch.tensor([25]), torch.tensor([25]),
                data["generation_mask"],
            )
            self.assertGreater(stats["infonce_valid_anchors"], 0)
            loss.backward()
            outputs.append(projected.detach())
            self.assertGreater(model.context_alignment_projector.weight.grad.norm().item(), 0)
            self.assertGreater(model.transformer_blocks[11].cross_attn.out_proj.weight.grad.norm().item(), 0)
        torch.testing.assert_close(*outputs)
        torch.testing.assert_close(normal.context_alignment_projector.weight.grad,
                                   checkpointed.context_alignment_projector.weight.grad)
        self.assertIsNone(teacher.grad)

    def test_cfm_dropout_and_warmup_keep_projector_in_graph(self):
        cfm = CFM_VT(transformer=make_model(), num_channels=64, audio_video_ratio=1,
                     ctc_lambda=0, infonce_lambda=0.05, frac_lengths_mask=(1, 1))
        data = inputs()
        # random() calls are prompt dropout then modality dropout.
        for weight, modality_draw in ((0.05, 0.9), (0.05, 0.0), (0.05, 0.25), (0.05, 0.45), (0.0, 0.9)):
            cfm.zero_grad(set_to_none=True)
            cfm.infonce_lambda = weight
            with patch("aligndit.model.cfm_vt.random", side_effect=[0.9, modality_draw]):
                loss, components, _, _ = cfm(
                    data["x"], data["text"], data["video"], lens=torch.tensor([40]),
                    text_lens=torch.tensor([4]), video_lens=torch.tensor([40]),
                    audio_teacher=torch.randn(1, 25, 32), audio_teacher_lengths=torch.tensor([25]),
                    audio_teacher_valid_lengths=torch.tensor([25]),
                )
            loss.backward()
            grad = cfm.transformer.context_alignment_projector.weight.grad
            self.assertIsNotNone(grad)
            if weight == 0 or modality_draw < 0.6:
                self.assertEqual(components["infonce_loss"], 0)
                self.assertEqual(grad.norm().item(), 0)
            else:
                self.assertGreater(components["infonce_valid_anchors"], 0)
                self.assertGreater(grad.norm().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
