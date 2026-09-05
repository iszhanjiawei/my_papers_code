"""CPU model contracts for isolated D1 64-D/40-Hz Semantic-VAE + CAM++.

No data, checkpoints, GPU or training updates are required. Tiny hidden widths
retain the full D1 18-block/6-MM structure and CTC locations [5, 11]. Nonzero
parent AdaLN gates simulate a trained parent, avoiding vacuous zero-output tests.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT


BASE_ARCH = {
    "dim": 32,
    "depth": 18,
    "heads": 4,
    "dim_head": 8,
    "ff_mult": 2,
    "mel_dim": 64,
    "text_num_embeds": 16,
    "text_dim": 16,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 1,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "checkpoint_activations": False,
    "use_conformer": False,
    "layer_indices_ctc": [5, 11],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 6,
    "n_text_layers": 6,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
}
SPEAKER_DIM = 192
SPEAKER_START = 6


def make_inputs():
    audio_mask = torch.arange(12)[None, :] < torch.tensor([12, 10])[:, None]
    text_mask = torch.arange(4)[None, :] < torch.tensor([4, 3])[:, None]
    generation_mask = audio_mask.clone()
    generation_mask[:, :3] = False
    text = torch.randint(0, 16, (2, 4)).masked_fill(~text_mask, -1)
    return {
        "x": torch.randn(2, 12, 64),
        "cond": torch.randn(2, 12, 64),
        "text": text,
        "video": torch.randn(2, 12, 16),
        "time": torch.tensor([0.2, 0.8]),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": audio_mask.clone(),
        "complementary_mask": audio_mask & ~generation_mask,
        "generation_mask": generation_mask,
        "cache": False,
    }


def make_warmstarted_pair(*, checkpoint_activations=False):
    arch = {**BASE_ARCH, "checkpoint_activations": checkpoint_activations}
    torch.manual_seed(7)
    parent = DiT_VT_MMDiT(**arch)
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
    speaker = DiT_VT_MMDiT(**arch, speaker_dim=SPEAKER_DIM, speaker_condition_start_layer=SPEAKER_START)
    missing, unexpected = speaker.load_state_dict(parent.state_dict(), strict=False)
    assert missing == ["speaker_proj.weight"] and not unexpected
    return parent, speaker


def test_zero_initialization_and_ctc_isolation():
    torch.manual_seed(666)
    seeded_parent = DiT_VT_MMDiT(**BASE_ARCH)
    torch.manual_seed(666)
    seeded_speaker = DiT_VT_MMDiT(
        **BASE_ARCH, speaker_dim=SPEAKER_DIM, speaker_condition_start_layer=SPEAKER_START
    )
    for key, value in seeded_parent.state_dict().items():
        assert torch.equal(value, seeded_speaker.state_dict()[key]), f"Speaker module perturbed initialization: {key}"
    del seeded_parent, seeded_speaker
    parent, speaker = make_warmstarted_pair()
    parent.eval()
    speaker.eval()
    assert set(speaker.state_dict()) - set(parent.state_dict()) == {"speaker_proj.weight"}
    assert speaker.speaker_proj.bias is None
    assert not torch.count_nonzero(speaker.speaker_proj.weight)
    inputs = make_inputs()
    first_embedding = torch.randn(2, SPEAKER_DIM)
    with torch.inference_mode():
        expected, parent_ctc = parent(**inputs)
        actual, speaker_ctc = speaker(**inputs, speaker_embedding=first_embedding)
    assert torch.count_nonzero(expected), "warm-started parent must not have trivial zero output"
    assert torch.equal(expected, actual)
    assert set(parent_ctc) == {5, 11}
    for layer_i in parent_ctc:
        assert torch.equal(parent_ctc[layer_i]["z_tilde"], speaker_ctc[layer_i]["z_tilde"])
        assert torch.equal(speaker_ctc[layer_i]["z_lens"], torch.tensor([12, 10]))
    assert actual.shape == (2, 12, 64)
    with torch.no_grad():
        speaker.speaker_proj.weight.normal_(std=0.03)
    with torch.inference_mode():
        first, first_ctc = speaker(**inputs, speaker_embedding=first_embedding)
        second, second_ctc = speaker(**inputs, speaker_embedding=torch.randn(2, SPEAKER_DIM))
    assert not torch.equal(first, second), "active speaker conditioning must affect the final prediction"
    assert torch.equal(first_ctc[5]["z_tilde"], second_ctc[5]["z_tilde"])
    assert not torch.equal(first_ctc[11]["z_tilde"], second_ctc[11]["z_tilde"])
    print("[OK] identical seeded initialization and zero-projection outputs; CTC5 isolated, CTC11 speaker-conditioned")


def test_tail_cfg_and_dropout():
    _, model = make_warmstarted_pair()
    model.eval()
    with torch.no_grad():
        model.speaker_proj.weight.normal_(std=0.03)
    inputs = make_inputs()
    embedding = torch.randn(2, SPEAKER_DIM)
    captured = {}

    def front_hook(_module, args, kwargs):
        captured["front_t"] = args[2].detach().clone()
        captured["front_mask"] = kwargs["mask"].clone()
        captured["text_mask"] = kwargs["text_mask"].clone()

    def pre_speaker_hook(_module, args, _kwargs):
        captured["pre_speaker_t"] = args[2].detach().clone()

    def tail_hook(_module, args, _kwargs):
        captured["tail_t"] = args[1].detach().clone()

    handles = [
        model.transformer_blocks[0].register_forward_pre_hook(front_hook, with_kwargs=True),
        model.transformer_blocks[5].register_forward_pre_hook(pre_speaker_hook, with_kwargs=True),
        model.transformer_blocks[SPEAKER_START].register_forward_pre_hook(tail_hook, with_kwargs=True),
    ]
    t = model.time_embed(inputs["time"])
    delta = model.get_speaker_delta(embedding, t)
    try:
        for branch_flags, branch_count in (({}, 3), ({"drop_video": True}, 2), ({"drop_text": True}, 2)):
            with torch.inference_mode():
                model(**inputs, speaker_embedding=embedding, cfg_infer=True, **branch_flags)
            assert torch.allclose(captured["front_t"], t.repeat(branch_count, 1))
            assert torch.allclose(captured["pre_speaker_t"], t.repeat(branch_count, 1))
            assert torch.allclose(captured["tail_t"], torch.cat([t + delta] * (branch_count - 1) + [t]))
            assert torch.equal(captured["front_mask"], inputs["mask"].repeat(branch_count, 1))
            assert torch.equal(captured["text_mask"], inputs["text_mask"].repeat(branch_count, 1))
            with torch.inference_mode():
                dropped, _ = model(
                    **inputs, speaker_embedding=embedding, cfg_infer=True, drop_audio_cond=True, **branch_flags
                )
                assert torch.allclose(captured["tail_t"], t.repeat(branch_count, 1))
                changed, _ = model(
                    **inputs, speaker_embedding=torch.randn_like(embedding), cfg_infer=True,
                    drop_audio_cond=True, **branch_flags,
                )
                explicit_zero, _ = model(
                    **{**inputs, "cond": torch.zeros_like(inputs["cond"])}, speaker_embedding=embedding,
                    cfg_infer=True, drop_speaker=True, **branch_flags,
                )
            assert torch.equal(dropped, changed), "packed prompt dropout must suppress speaker identity"
            assert torch.equal(dropped, explicit_zero), "packed prompt dropout must suppress the audio prompt"
        with torch.inference_mode():
            model(**inputs, speaker_embedding=embedding, drop_audio_cond=True)
        assert torch.equal(captured["tail_t"], t)
    finally:
        for handle in handles:
            handle.remove()
    assert not torch.count_nonzero(model.get_speaker_delta(embedding, t, drop_speaker=True))
    print("[OK] speaker throughout audio-only [6,18); B=2 packed full/TTS/null CFG and joint prompt/speaker dropout")


def test_first_backward_and_cfm_dropout():
    for checkpoint_activations in (False, True):
        _, transformer = make_warmstarted_pair(checkpoint_activations=checkpoint_activations)
        model = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03,
            audio_drop_prob=0.0, cond_drop_prob=0.0, text_drop_prob=0.0, video_drop_prob=0.0,
        )
        inputs = make_inputs()
        kwargs = {
            "inp": inputs["x"], "text": inputs["text"], "video": inputs["video"],
            "lens": torch.tensor([12, 10]), "text_lens": torch.tensor([4, 3]),
            "video_lens": torch.tensor([12, 10]), "speaker_embedding": torch.randn(2, SPEAKER_DIM),
        }
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = model(**kwargs)
        assert torch.isfinite(loss) and "ctc_loss" in components
        loss.backward()
        gradient = transformer.speaker_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        for drop_name in ("audio_drop_prob", "cond_drop_prob"):
            model.zero_grad(set_to_none=True)
            setattr(model, drop_name, 1.0)
            with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                dropped_loss, _, _, _ = model(**kwargs)
            dropped_loss.backward()
            assert not torch.count_nonzero(transformer.speaker_proj.weight.grad)
            setattr(model, drop_name, 0.0)
    print("[OK] nonzero first speaker gradient, activation checkpointing, zero identity gradient on prompt dropout")


def test_cfm_inference():
    _, transformer = make_warmstarted_pair()
    with torch.no_grad():
        transformer.speaker_proj.weight.normal_(std=0.03)
    model = CFM_VT(transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    kwargs = {
        "cond": torch.randn(2, 3, 64), "text": torch.randint(0, 16, (2, 4)),
        "duration": torch.tensor([12, 10]), "video": torch.randn(2, 12, 16),
        "lens": torch.tensor([3, 3]), "steps": 1, "use_epss": False, "seed": 0,
    }
    first_embedding = torch.randn(2, SPEAKER_DIM)
    second_embedding = torch.randn(2, SPEAKER_DIM)
    for guidance in (0.0, 1.0):
        output, _ = model.sample(
            **kwargs, speaker_embedding=first_embedding, cfg_strength=guidance, cfg_strength_v=guidance
        )
        assert output.shape == (2, 12, 64) and torch.isfinite(output).all()
        assert torch.equal(output[:, :3], kwargs["cond"])
        no_ref_a, _ = model.sample(
            **kwargs, speaker_embedding=first_embedding, no_ref_audio=True,
            cfg_strength=guidance, cfg_strength_v=guidance,
        )
        no_ref_b, _ = model.sample(
            **kwargs, speaker_embedding=second_embedding, no_ref_audio=True,
            cfg_strength=guidance, cfg_strength_v=guidance,
        )
        assert torch.equal(no_ref_a, no_ref_b)
    print("[OK] 40-Hz B=2 sampling, exact reference-prefix preservation, CFG/no-CFG and no_ref_audio")


def main():
    torch.set_num_threads(1)
    test_zero_initialization_and_ctc_isolation()
    test_tail_cfg_and_dropout()
    test_first_backward_and_cfm_dropout()
    test_cfm_inference()
    print("All Semantic-VAE Direct-D1 speaker model contracts passed; training updates: 0.")


if __name__ == "__main__":
    main()
