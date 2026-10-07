"""CPU regression tests for Flowley-window / OmniShow-gate visual injection.

Run from this experiment root with::

    PYTHONPATH=src python scripts/test_audio_local_visual_attention.py

No datasets, pretrained weights or GPU are needed. The backbone tests simulate
nonzero pretrained audio modulation/output weights so gradient checks exercise
the new adapter instead of a trivially zero scratch-model output.
"""

from __future__ import annotations

import math

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.local_visual_attention import GatedLocalVisualAttention
from f5_tts.model.modules import DiTBlock


BASE_ARCH = {
    "dim": 32,
    "depth": 4,
    "heads": 4,
    "dim_head": 8,
    "dropout": 0.0,
    "ff_mult": 2,
    "mel_dim": 64,
    "text_num_embeds": 16,
    "text_dim": 16,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 1,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "use_conformer": False,
    "layer_indices_ctc": [1, 2],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 1,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
    "speaker_dim": 192,
    "speaker_condition_start_layer": 2,
}


def production_arch(n_mm_layers):
    """Use production layer placement with small tensors for CPU regression."""
    return {
        **BASE_ARCH,
        "depth": 18,
        "n_mm_layers": n_mm_layers,
        "n_text_layers": n_mm_layers,
        "speaker_condition_start_layer": 12,
        "layer_indices_ctc": [6, 12],
    }


def make_adapter(**kwargs):
    return GatedLocalVisualAttention(dim=32, visual_dim=16, heads=4, dim_head=8, **kwargs)


def make_inputs():
    audio_mask = torch.arange(24)[None, :] < torch.tensor([24, 20])[:, None]
    text_mask = torch.arange(4)[None, :] < torch.tensor([4, 3])[:, None]
    generation_mask = audio_mask.clone()
    generation_mask[:, :3] = False
    return {
        "x": torch.randn(2, 24, 64),
        "cond": torch.randn(2, 24, 64),
        "text": torch.randint(0, 16, (2, 4)).masked_fill(~text_mask, -1),
        "video": torch.randn(2, 24, 16),
        "time": torch.tensor([0.2, 0.8]),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": audio_mask.clone(),
        "complementary_mask": audio_mask & ~generation_mask,
        "generation_mask": generation_mask,
        "speaker_embedding": torch.randn(2, 192),
        "cache": False,
    }


def make_warmstarted_pair(*, checkpoint_activations=False, architecture=None):
    torch.manual_seed(17)
    arch = {**(BASE_ARCH if architecture is None else architecture),
            "checkpoint_activations": checkpoint_activations}
    parent = DiT_VT_MMDiT(**arch)
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
        parent.speaker_proj.weight.normal_(std=0.03)
    adapted = DiT_VT_MMDiT(**arch, audio_local_visual_attention=True)
    missing, unexpected = adapted.load_state_dict(parent.state_dict(), strict=False)
    assert missing and all(".local_visual_attn." in key for key in missing), missing
    assert not unexpected, unexpected
    return parent, adapted


def test_flowley_window_in_seconds():
    adapter = make_adapter()
    bias = adapter.temporal_bias(42, 42, torch.device("cpu"))
    assert bias.shape == (42, 42) and bias.dtype == torch.float32
    offsets = torch.tensor([0, 5, 10, 15, 20, 21])
    # Flowley's raised-cosine radius is four 8-Hz tokens = 0.5 seconds.
    # On this model's 40-Hz timeline it therefore spans twenty tokens.
    expected = torch.tensor([1.0, (2 + math.sqrt(2)) / 4, 0.5, (2 - math.sqrt(2)) / 4, 0.0, 0.0])
    torch.testing.assert_close(bias[0, offsets].exp(), expected + 1e-6, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(bias, bias.T)
    assert torch.isfinite(bias).all(), "Flowley uses log(M + 1e-6), not a hard -inf window"

    # A video token at 10 Hz represents four 40-Hz audio frames.
    ratio_four = make_adapter(audio_video_ratio=4)
    native_video_bias = ratio_four.temporal_bias(42, 11, torch.device("cpu"))
    # Flowley's implementation rounds the query center to a video token.
    rounded_audio_centers = (torch.arange(42) / 4).round().clamp(max=10).long() * 4
    torch.testing.assert_close(native_video_bias, bias[rounded_audio_centers, ::4])
    torch.testing.assert_close(native_video_bias[20, 5].exp(), torch.tensor(1.0 + 1e-6))

    # Frame-rate changes preserve the physical-time extent.
    hz_eighty = make_adapter(audio_frame_rate=80.0)
    fast_bias = hz_eighty.temporal_bias(84, 84, torch.device("cpu"))
    torch.testing.assert_close(fast_bias[0, offsets * 2], bias[0, offsets])
    print("[OK] Flowley cosine weights, epsilon floor and physical-time alignment")


def test_omnishow_initialization_and_gate():
    torch.manual_seed(19)
    adapter = make_adapter()
    assert adapter.gate.shape == (32,) and adapter.gate.requires_grad
    torch.testing.assert_close(adapter.gate, torch.full((32,), 1e-5), rtol=0, atol=0)
    for name in ("to_q", "to_k", "to_v", "to_out"):
        projection = getattr(adapter, name)
        assert isinstance(projection, torch.nn.Linear), name
        assert torch.count_nonzero(projection.weight), f"{name} must not be zero-initialized"
    for name in ("q_norm", "k_norm"):
        norm = getattr(adapter, name)
        assert any(p.requires_grad for p in norm.parameters()), name

    x, video = torch.randn(2, 8, 32), torch.randn(2, 8, 16)
    with torch.no_grad():
        adapter.gate.fill_(0.3)
        output = adapter(x, video)
        adapter.gate.mul_(-2)
        scaled = adapter(x, video)
    assert output.abs().max() > 0
    torch.testing.assert_close(scaled, -2 * output, rtol=1e-6, atol=1e-7)
    print("[OK] OmniShow channel gate initialized to 1e-5; direct signed residual scaling")


def test_masks_and_empty_visual_condition():
    torch.manual_seed(23)
    adapter = make_adapter()
    with torch.no_grad():
        adapter.gate.fill_(0.3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    video = torch.randn(2, 6, 16, requires_grad=True)
    audio_mask = torch.arange(8)[None, :] < torch.tensor([7, 5])[:, None]
    video_mask = torch.arange(6)[None, :] < torch.tensor([4, 0])[:, None]
    generation_mask = torch.ones(2, 8, dtype=torch.bool)
    generation_mask[:, :2] = False
    kwargs = {"audio_mask": audio_mask, "video_mask": video_mask, "generation_mask": generation_mask}
    output = adapter(x, video, **kwargs)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert torch.count_nonzero(output[0, 2:7])
    assert not torch.count_nonzero(output[~(audio_mask & generation_mask)])
    assert not torch.count_nonzero(output[1]), "no visual keys must mean zero residual, including projection bias"

    altered_padding = video.detach().clone()
    altered_padding[~video_mask] = 1000 * torch.randn_like(altered_padding[~video_mask])
    torch.testing.assert_close(adapter(x, altered_padding, **kwargs), output, rtol=0, atol=0)
    output.square().sum().backward()
    for name, tensor in (("audio", x), ("visual", video)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
    assert not torch.count_nonzero(video.grad[~video_mask])
    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(adapter.gate.grad)
    assert torch.count_nonzero(adapter.to_v.weight.grad)

    no_keys = adapter(x.detach(), video.detach(), video_mask=torch.zeros_like(video_mask))
    assert torch.isfinite(no_keys).all() and not torch.count_nonzero(no_keys)
    print("[OK] padding, prompt-only queries, absent visual keys and finite backward")


def test_pretrained_keys_and_tail_placement():
    parent, adapted = make_warmstarted_pair()
    extra = set(adapted.state_dict()) - set(parent.state_dict())
    assert extra and all(".local_visual_attn." in key for key in extra)
    assert {int(key.split(".")[1]) for key in extra} == {2, 3}
    assert set(parent.state_dict()).issubset(adapted.state_dict())
    for i in (2, 3):
        block = adapted.transformer_blocks[i]
        assert isinstance(block, DiTBlock)
        old_keys = set(parent.transformer_blocks[i].state_dict())
        retained_keys = {key for key in block.state_dict() if not key.startswith("local_visual_attn.")}
        assert retained_keys == old_keys
    assert all(not hasattr(block, "local_visual_attn") for block in adapted.transformer_blocks[:2])

    # Zeroing the diagnostic gate must exactly recover the loaded parent.
    # Default 1e-5 initialization is near-identity rather than strict identity.
    with torch.no_grad():
        for block in adapted.transformer_blocks[2:]:
            block.local_visual_attn.gate.zero_()
    parent.eval()
    adapted.eval()
    inputs = make_inputs()
    with torch.inference_mode():
        expected, parent_ctc = parent(**inputs)
        actual, adapted_ctc = adapted(**inputs)
    assert torch.count_nonzero(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index in parent_ctc:
        torch.testing.assert_close(adapted_ctc[index]["z_tilde"], parent_ctc[index]["z_tilde"], rtol=0, atol=0)
    print("[OK] pretrained audio keys preserved, new keys only in text-free audio tail")


def test_production_layout_and_unchanged_conditioning(n_mm_layers):
    _, model = make_warmstarted_pair(architecture=production_arch(n_mm_layers))
    model.eval()
    blocks = model.transformer_blocks
    assert len(blocks) == 18
    assert [i for i, block in enumerate(blocks) if isinstance(block, MMDiTBlock_VT)] == list(range(n_mm_layers))
    local_layers = [i for i, block in enumerate(blocks) if hasattr(block, "local_visual_attn")]
    assert local_layers == list(range(n_mm_layers, 18)), local_layers
    assert all(isinstance(blocks[i], DiTBlock) for i in local_layers)
    assert model.speaker_condition_start_layer == 12
    assert tuple(model.layer_indices_ctc) == (6, 12)

    # Each audio layer must own an independent adapter, including its gate.
    adapter_parameters = [parameter for i in local_layers for parameter in blocks[i].local_visual_attn.parameters()]
    assert len({parameter.data_ptr() for parameter in adapter_parameters}) == len(adapter_parameters)
    reference = blocks[local_layers[0]].local_visual_attn
    reference_bias = reference.temporal_bias(24, 24, torch.device("cpu"))
    for i in local_layers:
        adapter = blocks[i].local_visual_attn
        torch.testing.assert_close(adapter.gate, torch.full_like(adapter.gate, 1e-5), rtol=0, atol=0)
        torch.testing.assert_close(adapter.temporal_bias(24, 24, torch.device("cpu")), reference_bias, rtol=0, atol=0)

    # Observe the actual modulation inputs: extending the audio tail must not
    # also extend speaker conditioning from six to twelve layers.
    observed_times = {}
    handles = []
    for index, block in enumerate(blocks):
        time_position = 2 if isinstance(block, MMDiTBlock_VT) else 1

        def record_time(_module, args, *, layer=index, position=time_position):
            observed_times[layer] = args[position].detach().clone()

        handles.append(block.register_forward_pre_hook(record_time))
    inputs = make_inputs()
    try:
        with torch.inference_mode():
            _, ctc = model(**inputs)
            time_embedding = model.time_embed(inputs["time"])
            speaker_delta = model.get_speaker_delta(inputs["speaker_embedding"], time_embedding)
        assert torch.count_nonzero(speaker_delta)
        for index in range(18):
            expected = time_embedding + speaker_delta if index >= 12 else time_embedding
            torch.testing.assert_close(observed_times[index], expected, rtol=0, atol=0)
        assert set(ctc) == {6, 12}
        for tap in ctc.values():
            assert torch.equal(tap["z_lens"], torch.tensor([24, 20]))
    finally:
        for handle in handles:
            handle.remove()
    print(f"[OK] {n_mm_layers}+{18 - n_mm_layers}: independent local adapters; common windows; speaker 12..17; CTC [6,12]")


def test_cfg_packed_matches_independent_branches(architecture=None):
    _, model = make_warmstarted_pair(architecture=architecture)
    model.eval()
    with torch.no_grad():
        for block in model.transformer_blocks[model.n_text_layers:]:
            block.local_visual_attn.gate.fill_(0.2)
    inputs = make_inputs()
    null_flags = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
    branches = (
        ({}, [{}, {"drop_video": True}, null_flags]),
        ({"drop_video": True}, [{"drop_video": True}, null_flags]),
        ({"drop_text": True}, [{"drop_text": True}, null_flags]),
    )
    with torch.inference_mode():
        for packed_flags, independent_flags in branches:
            actual, _ = model(**inputs, cfg_infer=True, **packed_flags)
            expected = torch.cat([model(**inputs, **flags)[0] for flags in independent_flags])
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        # The TTS branch must not leak actual raw visual features.
        changed = {**inputs, "video": torch.randn_like(inputs["video"]) * 100}
        first, _ = model(**inputs, drop_video=True)
        second, _ = model(**changed, drop_video=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    print("[OK] B=2 three-/two-branch CFG matches independent execution; dropped video cannot leak")


def test_training_checkpoint_and_adapter_gradients(architecture=None):
    _, plain = make_warmstarted_pair(checkpoint_activations=False, architecture=architecture)
    _, checkpointed = make_warmstarted_pair(checkpoint_activations=True, architecture=architecture)
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    plain.train()
    checkpointed.train()
    inputs = make_inputs()
    target = torch.randn_like(inputs["x"])
    results = []
    for model in (plain, checkpointed):
        prediction, ctc = model(**inputs)
        assert prediction.shape == inputs["x"].shape and torch.isfinite(prediction).all()
        loss = (prediction - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if ".local_visual_attn." not in name:
                continue
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad), f"first adapter gradient is zero: {name}"
            gradients[name] = parameter.grad.detach().clone()
        assert gradients
        assert set(ctc) == set(model.layer_indices_ctc)
        for tap in ctc.values():
            assert torch.equal(tap["z_lens"], torch.tensor([24, 20]))
        results.append((prediction.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-5, atol=1e-9)
    print("[OK] warm-started training and activation checkpointing have matching finite adapter gradients")


def main():
    torch.set_num_threads(1)
    test_flowley_window_in_seconds()
    test_omnishow_initialization_and_gate()
    test_masks_and_empty_visual_condition()
    test_pretrained_keys_and_tail_placement()
    test_cfg_packed_matches_independent_branches()
    test_training_checkpoint_and_adapter_gradients()
    for n_mm_layers in (12, 6):
        test_production_layout_and_unchanged_conditioning(n_mm_layers)
        test_cfg_packed_matches_independent_branches(production_arch(n_mm_layers))
        test_training_checkpoint_and_adapter_gradients(production_arch(n_mm_layers))
    print("All audio-local visual attention regression tests passed.")


if __name__ == "__main__":
    main()
