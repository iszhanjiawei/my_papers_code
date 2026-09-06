"""CPU contracts for Original D1 40 Hz / 64-D latents with speaker conditioning.

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
    arch = {
        **BASE_ARCH,
        "checkpoint_activations": checkpoint_activations,
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
    speaker = DiT_VT_MMDiT(**arch, speaker_dim=SPEAKER_DIM, speaker_condition_start_layer=6)
    missing, unexpected = speaker.load_state_dict(parent.state_dict(), strict=False)
    assert missing == ["speaker_proj.weight"] and not unexpected
    return parent, speaker


def test_zero_initialization_and_latent_contract():
    parent, speaker = make_warmstarted_pair()
    parent.eval()
    speaker.eval()
    assert set(speaker.state_dict()) - set(parent.state_dict()) == {"speaker_proj.weight"}
    assert speaker.speaker_condition_start_layer == speaker.n_mm_layers == 6
    assert len(speaker.transformer_blocks) - speaker.n_mm_layers == 12
    assert speaker.speaker_proj.bias is None
    assert not torch.count_nonzero(speaker.speaker_proj.weight)
    inputs = make_inputs()
    with torch.inference_mode():
        expected, parent_ctc = parent(**inputs)
        actual, speaker_ctc = speaker(**inputs, speaker_embedding=torch.randn(2, SPEAKER_DIM))
    assert torch.count_nonzero(expected), "parent output must not be the trivial all-zero scratch output"
    assert torch.equal(expected, actual)
    assert set(parent_ctc) == {5, 11}
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
        model.transformer_blocks[6].register_forward_pre_hook(tail_hook, with_kwargs=True),
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


def test_no_front_leak_and_packed_cfg():
    _, model = make_warmstarted_pair()
    model.eval()
    with torch.no_grad():
        model.speaker_proj.weight.normal_(std=0.03)
    inputs = make_inputs()
    first_embedding = torch.randn(2, SPEAKER_DIM)
    second_embedding = torch.randn(2, SPEAKER_DIM)
    with torch.inference_mode():
        first, first_ctc = model(**inputs, speaker_embedding=first_embedding)
        second, second_ctc = model(**inputs, speaker_embedding=second_embedding)
    assert torch.equal(first_ctc[5]["z_tilde"], second_ctc[5]["z_tilde"])
    assert not torch.equal(first_ctc[11]["z_tilde"], second_ctc[11]["z_tilde"])
    assert not torch.equal(first, second), "speaker must affect the audio tail after its projection trains"
    t = model.time_embed(inputs["time"])
    assert torch.allclose(
        model.get_speaker_delta(first_embedding, t), model.get_speaker_delta(3 * first_embedding, t), atol=1e-7
    ), "speaker input must be L2-normalized"

    # B=2 has different time values and masks. Check branch-major packing
    # against independent full / TTS / null forwards with and without cache.
    for cache in (False, True):
        inputs["cache"] = cache
        for branch_flags in ({}, {"drop_video": True}, {"drop_text": True}):
            branches = [branch_flags] if branch_flags else [{}, {"drop_video": True}]
            branches.append({"drop_audio_cond": True, "drop_text": True, "drop_video": True})
            with torch.inference_mode():
                model.clear_cache()
                packed, _ = model(**inputs, speaker_embedding=first_embedding, cfg_infer=True, **branch_flags)
                independent = []
                for flags in branches:
                    model.clear_cache()
                    result, _ = model(**inputs, speaker_embedding=first_embedding, **flags)
                    independent.append(result)
            assert torch.allclose(packed, torch.cat(independent), rtol=1e-5, atol=1e-6)
    model.clear_cache()
    print("[OK] speaker cannot alter first 6 MM layers; B=2 packed CFG matches independent branches and cache")


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
    test_no_front_leak_and_packed_cfg()
    test_cfm_inference()
    print("All Original-D1 Semantic-VAE speaker model contracts passed (6 MM + 12 audio, fixed CTC 0.03).")


if __name__ == "__main__":
    main()
