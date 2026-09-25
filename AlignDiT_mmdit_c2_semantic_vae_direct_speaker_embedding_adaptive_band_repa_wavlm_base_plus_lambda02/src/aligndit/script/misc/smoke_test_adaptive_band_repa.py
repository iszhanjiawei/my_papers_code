"""CPU contracts for jointly enabled adaptive temporal bands and WavLM REPA.

Run in this experiment with PYTHONPATH=src. Uses small synthetic inputs only;
neither pretrained weights, cached WavLM targets, nor a GPU are required.
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.repa import masked_repa_cosine_loss
from aligndit.script.misc.smoke_test_adaptive_temporal_band import (
    BAND_ARCH,
    BASE_ARCH,
    assert_nonzero_finite,
    last_linear,
    make_inputs,
    make_predictor_content_sensitive,
    warm_start,
)
from aligndit.script.misc.smoke_test_semantic_vae_c2_repa import REPA_KEYS


REPA_ARCH = {"repa_layer": 1, "repa_target_dim": 24, "repa_projector_dim": 48}
CFM_ARCH = {
    "num_channels": 64,
    "audio_video_ratio": 1,
    "ctc_lambda": 0.03,
    "repa_lambda": 0.2,
    "audio_drop_prob": 0.0,
    "cond_drop_prob": 0.0,
    "text_drop_prob": 0.0,
    "video_drop_prob": 0.0,
}


def make_pair():
    band_only = warm_start(DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH))
    make_predictor_content_sensitive(band_only.temporal_band)
    combined = DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH, **REPA_ARCH)
    missing, unexpected = combined.load_state_dict(band_only.state_dict(), strict=False)
    assert set(missing) == REPA_KEYS and not unexpected
    return band_only, combined


def cfm_inputs():
    inputs = make_inputs()
    return {
        "inp": inputs["x"],
        "text": inputs["text"],
        "video": inputs["video"],
        "lens": inputs["mask"].sum(-1),
        "text_lens": inputs["text_mask"].sum(-1),
        # Both training grids are 40 Hz; use equal valid audio/video lengths.
        "video_lens": inputs["mask"].sum(-1),
        "speaker_embedding": inputs["speaker_embedding"],
        "repa_features": torch.randn(2, 15, REPA_ARCH["repa_target_dim"]),
        "repa_feature_lens": torch.tensor([15, 11]),
    }


class AdaptiveBandRepaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(611)

    def assert_ctc_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for layer in expected:
            for key in ("z_tilde", "z_lens"):
                torch.testing.assert_close(actual[layer][key], expected[layer][key], atol=0, rtol=0)

    def test_seed_preservation_and_warm_start_key_union(self):
        torch.manual_seed(17)
        band_only = DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH)
        torch.manual_seed(17)
        combined = DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH, **REPA_ARCH)
        for key, tensor in band_only.state_dict().items():
            torch.testing.assert_close(tensor, combined.state_dict()[key], atol=0, rtol=0, msg=key)
        self.assertEqual(set(combined.state_dict()) - set(band_only.state_dict()), REPA_KEYS)

        speaker_only = DiT_VT_MMDiT(**BASE_ARCH)
        repa_only = DiT_VT_MMDiT(**BASE_ARCH, **REPA_ARCH)
        band_keys = {key for key in combined.state_dict() if key.startswith("temporal_band.")}
        self.assertTrue(band_keys)
        for source, allowed_missing in (
            (speaker_only, band_keys | REPA_KEYS),
            (band_only, REPA_KEYS),
            (repa_only, band_keys),
        ):
            with self.subTest(source_keys=len(source.state_dict())):
                missing, unexpected = combined.load_state_dict(source.state_dict(), strict=False)
                self.assertEqual(set(missing), allowed_missing)
                self.assertFalse(unexpected)

    def test_auxiliary_head_preserves_band_inference_and_all_cfg_branches(self):
        band_only, combined = make_pair()
        band_only.eval()
        combined.eval()
        inputs = make_inputs()
        cases = (
            (False, {}),
            (False, {"drop_video": True}),
            (False, {"drop_text": True}),
            (False, {"drop_audio_cond": True, "drop_text": True, "drop_video": True}),
            (True, {}),
            (True, {"drop_video": True}),
            (True, {"drop_text": True}),
        )
        with torch.inference_mode():
            for cfg_infer, flags in cases:
                with self.subTest(cfg_infer=cfg_infer, flags=flags):
                    expected, expected_ctc = band_only(**inputs, cfg_infer=cfg_infer, **flags)
                    actual, actual_ctc = combined(**inputs, cfg_infer=cfg_infer, **flags)
                    assert_nonzero_finite(self, expected, "band inference")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assert_ctc_equal(expected_ctc, actual_ctc)
                    if not cfg_infer:
                        projected, projected_ctc, projection = combined(**inputs, return_repa=True, **flags)
                        torch.testing.assert_close(projected, expected, atol=0, rtol=0)
                        self.assert_ctc_equal(expected_ctc, projected_ctc)
                        self.assertEqual(projection.shape, (*expected.shape[:2], REPA_ARCH["repa_target_dim"]))
                    for name in ("last_temporal_band_offset_seconds", "last_temporal_band_sigma_seconds"):
                        torch.testing.assert_close(
                            getattr(combined, name), getattr(band_only, name), atol=0, rtol=0
                        )

    def test_repa_remains_training_only_and_packed_cfg_matches_separate_branches(self):
        _, model = make_pair()
        model.eval()
        inputs = make_inputs()
        with torch.inference_mode():
            for flags in ({"cfg_infer": True}, {"cache": True}):
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "training-only"):
                    model(**{**inputs, **flags}, return_repa=True)
            packed, _ = model(**inputs, cfg_infer=True)
            separate = [
                model(**inputs, **flags)[0]
                for flags in (
                    {},
                    {"drop_video": True},
                    {"drop_audio_cond": True, "drop_text": True, "drop_video": True},
                )
            ]
        torch.testing.assert_close(packed, torch.cat(separate), atol=3e-6, rtol=3e-5)

    def test_cfm_uses_same_generation_mask_for_flow_and_repa(self):
        _, transformer = make_pair()
        model = CFM_VT(transformer=transformer, **CFM_ARCH)
        kwargs = cfm_inputs()
        requested_span = torch.zeros(2, 12, dtype=torch.bool)
        requested_span[:, 4:] = True
        expected_mask = requested_span & (torch.arange(12)[None] < kwargs["lens"][:, None])
        observed = {}

        def capture_transformer(_module, _args, forward_kwargs):
            observed["generation_mask"] = forward_kwargs["generation_mask"].detach().clone()
            observed["cond"] = forward_kwargs["cond"].detach().clone()

        handle = transformer.register_forward_pre_hook(capture_transformer, with_kwargs=True)
        try:
            with (
                patch("aligndit.model.cfm_vt.random", return_value=0.5),
                patch("aligndit.model.cfm_vt.mask_from_frac_lengths", return_value=requested_span),
                patch("aligndit.model.cfm_vt.masked_repa_cosine_loss", wraps=masked_repa_cosine_loss) as repa,
                patch("aligndit.model.cfm_vt.F.mse_loss", wraps=F.mse_loss) as mse,
            ):
                loss, components, _, _ = model(**kwargs)
        finally:
            handle.remove()
        torch.testing.assert_close(observed["generation_mask"], expected_mask)
        torch.testing.assert_close(repa.call_args.args[4], expected_mask)
        torch.testing.assert_close(repa.call_args.args[3], kwargs["lens"])
        torch.testing.assert_close(observed["cond"][~expected_mask], kwargs["inp"][~expected_mask])
        self.assertEqual(torch.count_nonzero(observed["cond"][expected_mask]).item(), 0)
        predicted_flow, target_flow = mse.call_args.args
        expected_flow_loss = (predicted_flow - target_flow).square()[expected_mask].mean()
        self.assertAlmostEqual(components["diff_loss"], expected_flow_loss.item(), places=6)
        expected_total = (
            components["diff_loss"] + 0.2 * components["repa_loss"] + 0.03 * components["ctc_loss"]
        )
        self.assertAlmostEqual(loss.item(), expected_total, places=5)

    def test_combined_backward_and_checkpointing_parity(self):
        _, plain = make_pair()
        checked = DiT_VT_MMDiT(
            **{**BASE_ARCH, "checkpoint_activations": True}, **BAND_ARCH, **REPA_ARCH
        )
        checked.load_state_dict(plain.state_dict(), strict=True)
        kwargs = cfm_inputs()
        # Frozen cached targets must remain detached even if a caller accidentally
        # supplies requires_grad=True tensors.
        kwargs["repa_features"].requires_grad_()
        outputs = []
        for transformer in (plain, checked):
            model = CFM_VT(transformer=transformer, **CFM_ARCH).train()
            torch.manual_seed(19)
            with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                loss, components, _, prediction = model(**kwargs)
            self.assertEqual(set(components), {"diff_loss", "ctc_loss", "repa_loss"})
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            outputs.append((loss.detach(), prediction.detach()))
            band_gradient = last_linear(transformer.temporal_band).weight.grad
            assert_nonzero_finite(self, band_gradient[0], "combined offset gradient")
            assert_nonzero_finite(self, band_gradient[1], "combined width gradient")
            for index, layer in enumerate(transformer.repa_projector):
                if isinstance(layer, nn.Linear):
                    assert_nonzero_finite(self, layer.weight.grad, f"REPA projector layer {index}")
            self.assertIsNone(kwargs["repa_features"].grad)
        for expected, actual in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            expected = plain_parameters[name].grad
            self.assertEqual(parameter.grad is None, expected is None, name)
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5, msg=name)

    def test_repa_target_padding_and_prompt_are_excluded(self):
        student = torch.randn(2, 12, 24, requires_grad=True)
        teacher = torch.randn(2, 15, 24, requires_grad=True)
        student_lens = torch.tensor([12, 9])
        teacher_lens = torch.tensor([15, 11])
        mask = (torch.arange(12)[None] >= 4) & (torch.arange(12)[None] < student_lens[:, None])
        original = masked_repa_cosine_loss(student, teacher, teacher_lens, student_lens, mask)
        altered_student = student.detach().clone()
        altered_teacher = teacher.detach().clone()
        altered_student[~mask] = 10000 * torch.randn_like(altered_student[~mask])
        altered_teacher[1, 11:] = 10000 * torch.randn_like(altered_teacher[1, 11:])
        altered = masked_repa_cosine_loss(
            altered_student, altered_teacher, teacher_lens, student_lens, mask
        )
        torch.testing.assert_close(altered, original, atol=0, rtol=0)
        original.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(torch.count_nonzero(student.grad[~mask]).item(), 0)
        assert_nonzero_finite(self, student.grad[mask], "generated-frame REPA gradient")

    def test_repa_alone_backpropagates_through_the_adaptive_band(self):
        _, model = make_pair()
        inputs = make_inputs()
        _, _, projection = model(**inputs, return_repa=True)
        repa_loss = masked_repa_cosine_loss(
            projection,
            torch.randn(2, 15, REPA_ARCH["repa_target_dim"]),
            torch.tensor([15, 11]),
            inputs["mask"].sum(-1),
            inputs["generation_mask"],
        )
        repa_loss.backward()
        gradient = last_linear(model.temporal_band).weight.grad
        assert_nonzero_finite(self, gradient[0], "REPA-only offset gradient")
        assert_nonzero_finite(self, gradient[1], "REPA-only width gradient")
        assert_nonzero_finite(self, model.repa_projector[0].weight.grad, "REPA-only projector gradient")
        # The configured tap precedes tail-only speaker conditioning. REPA must
        # train the early multimodal path without creating a spurious tail link.
        self.assertIsNone(model.speaker_proj.weight.grad)

    def test_checkpoint_and_ema_keep_both_auxiliary_parameter_sets(self):
        from ema_pytorch import EMA

        _, model = make_pair()
        model.eval()
        ema = EMA(model, beta=0.9, update_after_step=0, update_every=1)
        ema.update()
        inputs = make_inputs()
        with torch.no_grad():
            expected = ema.ema_model(**inputs, return_repa=True)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        for prefix in ("temporal_band.", "repa_projector."):
            self.assertTrue(any(key.startswith(prefix) for key in saved["model"]))
            self.assertTrue(any(prefix in key for key in saved["ema"]))
        restored = DiT_VT_MMDiT(**BASE_ARCH, **BAND_ARCH, **REPA_ARCH).eval()
        restored.load_state_dict(saved["model"], strict=True)
        restored_ema = EMA(restored, beta=0.9, update_after_step=0, update_every=1)
        restored_ema.load_state_dict(saved["ema"], strict=True)
        with torch.no_grad():
            actual = restored_ema.ema_model(**inputs, return_repa=True)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        self.assert_ctc_equal(expected[1], actual[1])
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
