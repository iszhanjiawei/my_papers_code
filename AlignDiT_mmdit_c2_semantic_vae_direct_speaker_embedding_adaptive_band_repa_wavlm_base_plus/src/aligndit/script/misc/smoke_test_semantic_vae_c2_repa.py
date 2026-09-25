"""CPU smoke contracts for single-teacher WavLM REPA integration."""

from __future__ import annotations

from unittest.mock import patch

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.repa import masked_repa_cosine_loss
from aligndit.script.misc.smoke_test_semantic_vae_c2_speaker import (
    BASE_ARCH,
    SPEAKER_DIM,
    make_inputs,
    make_warmstarted_pair,
)


REPA_DIM = 24
REPA_HIDDEN_DIM = 48
REPA_LAYER = 1
REPA_KEYS = {
    "repa_projector.0.weight",
    "repa_projector.0.bias",
    "repa_projector.2.weight",
    "repa_projector.2.bias",
    "repa_projector.4.weight",
    "repa_projector.4.bias",
}


def make_repa_model(*, checkpoint_activations=False):
    _, speaker = make_warmstarted_pair(checkpoint_activations=checkpoint_activations)
    repa = DiT_VT_MMDiT(
        **{**BASE_ARCH, "checkpoint_activations": checkpoint_activations},
        speaker_dim=SPEAKER_DIM,
        speaker_condition_start_layer=2,
        repa_layer=REPA_LAYER,
        repa_target_dim=REPA_DIM,
        repa_projector_dim=REPA_HIDDEN_DIM,
    )
    missing, unexpected = repa.load_state_dict(speaker.state_dict(), strict=False)
    assert set(missing) == REPA_KEYS and not unexpected
    return speaker, repa


def test_projector_is_auxiliary_and_layer_tap_is_exact():
    speaker, repa = make_repa_model()
    speaker.eval()
    repa.eval()
    inputs = make_inputs()
    embedding = torch.randn(2, SPEAKER_DIM)
    captured = {}

    def capture_tap(_module, _args, output):
        captured["tap"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = repa.transformer_blocks[REPA_LAYER].register_forward_hook(capture_tap)
    try:
        with torch.inference_mode():
            expected, expected_ctc = speaker(**inputs, speaker_embedding=embedding)
            actual, actual_ctc = repa(**inputs, speaker_embedding=embedding)
            projected_output, projected_ctc, projection = repa(
                **inputs, speaker_embedding=embedding, return_repa=True
            )
    finally:
        handle.remove()
    assert torch.equal(expected, actual) and torch.equal(actual, projected_output)
    assert expected_ctc.keys() == actual_ctc.keys() == projected_ctc.keys()
    assert projection.shape == (2, 12, REPA_DIM)
    assert torch.allclose(projection, repa.repa_projector(captured["tap"]))
    print("[OK] REPA head is inference-neutral and taps exactly the configured MM-DiT block")


def test_generation_only_cosine_and_rate_alignment():
    student = torch.randn(2, 8, REPA_DIM)
    teacher = torch.randn(2, 10, REPA_DIM)
    student_lens = torch.tensor([8, 6])
    teacher_lens = torch.tensor([10, 7])
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, 3:8] = True
    mask[1, 2:6] = True
    loss = masked_repa_cosine_loss(student, teacher, teacher_lens, student_lens, mask)
    assert torch.isfinite(loss) and 0 <= loss <= 2

    same_rate_teacher = torch.randn(1, 8, REPA_DIM)
    same_rate_student = torch.randn(1, 8, REPA_DIM)
    same_rate_mask = torch.zeros(1, 8, dtype=torch.bool)
    same_rate_mask[:, 4:] = True
    before = masked_repa_cosine_loss(
        same_rate_student, same_rate_teacher, torch.tensor([8]), torch.tensor([8]), same_rate_mask
    )
    same_rate_teacher[:, :4] = torch.randn_like(same_rate_teacher[:, :4]) * 1000
    after = masked_repa_cosine_loss(
        same_rate_student, same_rate_teacher, torch.tensor([8]), torch.tensor([8]), same_rate_mask
    )
    assert torch.equal(before, after), "prompt-region teacher targets must not affect generation-region REPA"
    print("[OK] 50->40 Hz interpolation is finite and cosine loss uses generated frames only")


def test_cfm_repa_backward():
    for checkpoint_activations in (False, True):
        _, transformer = make_repa_model(checkpoint_activations=checkpoint_activations)
        model = CFM_VT(
            transformer=transformer,
            num_channels=64,
            audio_video_ratio=1,
            ctc_lambda=0.03,
            repa_lambda=0.1,
            audio_drop_prob=0.0,
            cond_drop_prob=0.0,
            text_drop_prob=0.0,
            video_drop_prob=0.0,
        )
        inputs = make_inputs()
        kwargs = {
            "inp": inputs["x"],
            "text": inputs["text"],
            "video": inputs["video"],
            "lens": torch.tensor([12, 10]),
            "text_lens": torch.tensor([4, 3]),
            "video_lens": torch.tensor([12, 10]),
            "speaker_embedding": torch.randn(2, SPEAKER_DIM),
            "repa_features": torch.randn(2, 15, REPA_DIM),
            "repa_feature_lens": torch.tensor([15, 12]),
        }
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = model(**kwargs)
        assert torch.isfinite(loss) and set(components) == {"diff_loss", "ctc_loss", "repa_loss"}
        loss.backward()
        gradients = [
            layer.weight.grad
            for layer in transformer.repa_projector
            if isinstance(layer, torch.nn.Linear)
        ]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
    print("[OK] CFM combines diffusion, CTC and REPA losses; projector receives gradients with checkpointing")


def test_invalid_repa_config_is_rejected():
    try:
        DiT_VT_MMDiT(**BASE_ARCH, repa_layer=2, repa_target_dim=REPA_DIM)
    except ValueError as error:
        assert "MM-DiT" in str(error)
    else:
        raise AssertionError("REPA tap outside the MM-DiT prefix was accepted")
    print("[OK] REPA taps are restricted to the double-stream MM-DiT prefix")


def main():
    torch.set_num_threads(1)
    test_projector_is_auxiliary_and_layer_tap_is_exact()
    test_generation_only_cosine_and_rate_alignment()
    test_cfm_repa_backward()
    test_invalid_repa_config_is_rejected()
    print("All Semantic-VAE Direct-C2 WavLM REPA contracts passed.")


if __name__ == "__main__":
    main()
