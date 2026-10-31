"""CPU contracts for Direct-C2 40 Hz / 64-D latents with speaker conditioning.

Run from this experiment root with PYTHONPATH=src. No checkpoints or data are
needed: nonzero parent modulations simulate the warm-started audio backbone.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT


BASE_ARCH = {
    "dim": 32,
    "depth": 4,
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
    "layer_indices_ctc": [1, 2],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 2,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
}
SPEAKER_DIM = 192


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


def make_warmstarted_pair(*, checkpoint_activations=False, normalize_text_context=False):
    arch = {
        **BASE_ARCH,
        "checkpoint_activations": checkpoint_activations,
        "normalize_text_context": normalize_text_context,
    }
    torch.manual_seed(7)
    parent = DiT_VT_MMDiT(**arch)
    # A real warm start has trained AdaLN and output weights. Keeping the
    # scratch initialization's zero gates would make gradient testing vacuous.
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
    speaker = DiT_VT_MMDiT(**arch, speaker_dim=SPEAKER_DIM, speaker_condition_start_layer=2)
    missing, unexpected = speaker.load_state_dict(parent.state_dict(), strict=False)
    assert missing == ["speaker_proj.weight"] and not unexpected
    return parent, speaker


def test_zero_initialization_and_latent_contract():
    for normalize_context in (False, True):
        parent, speaker = make_warmstarted_pair(normalize_text_context=normalize_context)
        parent.eval()
        speaker.eval()
        assert set(speaker.state_dict()) - set(parent.state_dict()) == {"speaker_proj.weight"}
        assert speaker.speaker_proj.bias is None
        assert not torch.count_nonzero(speaker.speaker_proj.weight)
        inputs = make_inputs()
        with torch.inference_mode():
            expected, parent_ctc = parent(**inputs)
            actual, speaker_ctc = speaker(**inputs, speaker_embedding=torch.randn(2, SPEAKER_DIM))
        assert torch.count_nonzero(expected), "parent output must not be the trivial all-zero scratch output"
        assert torch.equal(expected, actual)
        for layer_i in parent_ctc:
            assert torch.equal(parent_ctc[layer_i]["z_tilde"], speaker_ctc[layer_i]["z_tilde"])
            assert torch.equal(speaker_ctc[layer_i]["z_lens"], torch.tensor([12, 10]))
        assert actual.shape == (2, 12, 64)
    print("[OK] zero speaker projection preserves warm-started outputs, 64-D latents and full-rate CTC")


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

    def tail_hook(_module, args, kwargs):
        captured["tail_t"] = args[1].detach().clone()

    handles = [
        model.transformer_blocks[0].register_forward_pre_hook(front_hook, with_kwargs=True),
        model.transformer_blocks[2].register_forward_pre_hook(tail_hook, with_kwargs=True),
    ]
    t = model.time_embed(inputs["time"])
    delta = model.get_speaker_delta(embedding, t)
    try:
        for branch_flags, branch_count in (({}, 3), ({"drop_video": True}, 2), ({"drop_text": True}, 2)):
            with torch.inference_mode():
                model(**inputs, speaker_embedding=embedding, cfg_infer=True, **branch_flags)
            expected_tail = torch.cat([t + delta] * (branch_count - 1) + [t])
            assert torch.allclose(captured["front_t"], t.repeat(branch_count, 1))
            assert torch.allclose(captured["tail_t"], expected_tail)
            assert torch.equal(captured["front_mask"], inputs["mask"].repeat(branch_count, 1))
            assert torch.equal(captured["text_mask"], inputs["text_mask"].repeat(branch_count, 1))
        with torch.inference_mode():
            model(**inputs, speaker_embedding=embedding, drop_audio_cond=True)
        assert torch.equal(captured["tail_t"], t)
    finally:
        for handle in handles:
            handle.remove()
    assert not torch.count_nonzero(model.get_speaker_delta(embedding, t, drop_speaker=True))
    print("[OK] tail-only speaker conditioning, B=2 full/TTS/null CFG and coupled audio dropout")


def test_first_backward_and_cfm_dropout():
    for checkpoint_activations in (False, True):
        _, transformer = make_warmstarted_pair(checkpoint_activations=checkpoint_activations)
        model = CFM_VT(
            transformer=transformer,
            num_channels=64,
            audio_video_ratio=1,
            ctc_lambda=0.03,
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
        }
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = model(**kwargs)
        assert torch.isfinite(loss) and "ctc_loss" in components
        loss.backward()
        gradient = transformer.speaker_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient), "zero speaker projection must receive the first warm-started gradient"
        for drop_name in ("audio_drop_prob", "cond_drop_prob"):
            model.zero_grad(set_to_none=True)
            setattr(model, drop_name, 1.0)
            with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                dropped_loss, _, _, _ = model(**kwargs)
            dropped_loss.backward()
            assert not torch.count_nonzero(transformer.speaker_proj.weight.grad)
            setattr(model, drop_name, 0.0)
    print("[OK] first speaker gradient is finite/nonzero, checkpointing works, CFG dropout removes identity gradient")


def test_cfm_inference():
    _, transformer = make_warmstarted_pair()
    with torch.no_grad():
        transformer.speaker_proj.weight.normal_(std=0.03)
    model = CFM_VT(transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    kwargs = {
        "cond": torch.randn(2, 3, 64),
        "text": torch.randint(0, 16, (2, 4)),
        "duration": torch.tensor([12, 10]),
        "video": torch.randn(2, 12, 16),
        "lens": torch.tensor([3, 3]),
        "steps": 1,
        "use_epss": False,
        "seed": 0,
    }
    first_embedding = torch.randn(2, SPEAKER_DIM)
    second_embedding = torch.randn(2, SPEAKER_DIM)
    for guidance in (0.0, 1.0):
        output, _ = model.sample(
            **kwargs, speaker_embedding=first_embedding, cfg_strength=guidance, cfg_strength_v=guidance
        )
        assert output.shape == (2, 12, 64) and torch.isfinite(output).all()
        no_ref_a, _ = model.sample(
            **kwargs, speaker_embedding=first_embedding, no_ref_audio=True,
            cfg_strength=guidance, cfg_strength_v=guidance,
        )
        no_ref_b, _ = model.sample(
            **kwargs, speaker_embedding=second_embedding, no_ref_audio=True,
            cfg_strength=guidance, cfg_strength_v=guidance,
        )
        assert torch.equal(no_ref_a, no_ref_b)
    print("[OK] 40-Hz B=2 CFM inference, CFG/no-CFG, and no_ref_audio suppresses speaker identity")


def main():
    torch.set_num_threads(1)
    test_zero_initialization_and_latent_contract()
    test_tail_cfg_and_dropout()
    test_first_backward_and_cfm_dropout()
    test_cfm_inference()
    print("All Semantic-VAE Direct-C2 speaker model contracts passed.")


if __name__ == "__main__":
    main()
