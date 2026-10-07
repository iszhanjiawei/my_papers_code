"""CPU regressions for gated text injection into the 6MM + 12audio tail.

Run with ``PYTHONPATH=src python scripts/test_audio_tail_text_attention.py``.
Small widths retain the production layer placement. Warm-started nonzero audio
modulation/output weights make first-backward checks meaningful without the
large parent checkpoint, real datasets, or a GPU.
"""

from __future__ import annotations

import torch
from test_audio_local_visual_attention import make_inputs, production_arch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from f5_tts.model.modules import DiTBlock


ARCH = {**production_arch(6), "normalize_text_context": True}
TAIL_LAYERS = list(range(6, 18))
ADAPTER_SUFFIXES = {
    "gate", "to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias",
    "to_v.weight", "to_v.bias", "to_out.weight", "to_out.bias",
    "q_norm.weight", "k_norm.weight",
}


def warmstarted_pair(*, checkpoint_activations=False, prompt_isolated_ca=False):
    torch.manual_seed(17)
    arch = {
        **ARCH, "checkpoint_activations": checkpoint_activations,
        "prompt_isolated_ca": prompt_isolated_ca,
        "audio_local_visual_attention": True,
    }
    parent = DiT_VT_MMDiT(**arch)
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
        parent.speaker_proj.weight.normal_(std=0.03)
    adapted = DiT_VT_MMDiT(**arch, audio_tail_text_attention=True)
    missing, unexpected = adapted.load_state_dict(parent.state_dict(), strict=False)
    assert len(missing) == 132 and all(".tail_text_attn." in key for key in missing), missing
    assert not unexpected, unexpected
    return parent, adapted


def text_modules(model):
    return {
        layer: block.tail_text_attn
        for layer, block in enumerate(model.transformer_blocks)
        if getattr(block, "tail_text_attn", None) is not None
    }


def test_default_disabled_and_pretrained_keys():
    for visual in (False, True):
        torch.manual_seed(31)
        implicit = DiT_VT_MMDiT(**ARCH, audio_local_visual_attention=visual)
        torch.manual_seed(31)
        explicit = DiT_VT_MMDiT(
            **ARCH, audio_local_visual_attention=visual, audio_tail_text_attention=False,
        )
        assert not text_modules(implicit) and not text_modules(explicit)
        assert set(implicit.state_dict()) == set(explicit.state_dict())
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)

    parent, model = warmstarted_pair()
    expected_keys = {
        f"transformer_blocks.{layer}.tail_text_attn.{suffix}"
        for layer in TAIL_LAYERS for suffix in ADAPTER_SUFFIXES
    }
    assert set(model.state_dict()) - set(parent.state_dict()) == expected_keys
    assert len(expected_keys) == 132
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert list(text_modules(model)) == TAIL_LAYERS
    assert all(isinstance(block, MMDiTBlock_VT) for block in model.transformer_blocks[:6])
    assert all(isinstance(block, DiTBlock) for block in model.transformer_blocks[6:])
    assert model.speaker_condition_start_layer == 12
    assert tuple(model.layer_indices_ctc) == (6, 12)
    parameters = []
    for layer, module in text_modules(model).items():
        assert tuple(module.to_k.weight.shape) == (32, 16)
        assert tuple(module.to_v.weight.shape) == (32, 16)
        assert set(module.state_dict()) == ADAPTER_SUFFIXES
        torch.testing.assert_close(module.gate, torch.full((32,), 1e-5), rtol=0, atol=0)
        for name in ("to_q", "to_k", "to_v", "to_out"):
            assert torch.count_nonzero(getattr(module, name).weight), name
        parameters.extend(module.parameters())
        parameters.extend(model.transformer_blocks[layer].local_visual_attn.parameters())
    assert len({p.data_ptr() for p in parameters}) == len(parameters)

    with torch.no_grad():
        for module in text_modules(model).values():
            module.gate.zero_()
    parent.eval()
    model.eval()
    inputs = make_inputs()
    with torch.inference_mode():
        expected, expected_ctc = parent(**inputs)
        actual, actual_ctc = model(**inputs)
    assert torch.count_nonzero(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer, expected_tap in expected_ctc.items():
        for name, tensor in expected_tap.items():
            torch.testing.assert_close(actual_ctc[layer][name], tensor, rtol=0, atol=0)
    print("[OK] default-off compatibility; 132 independent text tensors; zero gates exactly recover visual parent and CTC")


def test_adapter_masks_empty_context_and_gate_scaling():
    _, model = warmstarted_pair()
    adapter = model.transformer_blocks[6].tail_text_attn
    with torch.no_grad():
        adapter.gate.fill_(0.3)
    audio = torch.randn(2, 8, 32, requires_grad=True)
    text = torch.randn(2, 5, 16, requires_grad=True)
    audio_mask = torch.arange(8)[None, :] < torch.tensor([7, 5])[:, None]
    text_mask = torch.arange(5)[None, :] < torch.tensor([3, 0])[:, None]
    generation_mask = audio_mask.clone()
    generation_mask[:, :2] = False
    kwargs = {"audio_mask": audio_mask, "text_mask": text_mask, "generation_mask": generation_mask}
    output = adapter(audio, text, **kwargs)
    assert output.shape == audio.shape and torch.isfinite(output).all()
    assert torch.count_nonzero(output[0, 2:7])
    assert not torch.count_nonzero(output[~generation_mask])
    assert not torch.count_nonzero(output[1]), "all-masked text must suppress output projection bias"

    changed_text = text.detach().clone()
    changed_text[~text_mask] = float("nan")
    changed_audio = audio.detach().clone()
    changed_audio[~generation_mask] = float("nan")
    torch.testing.assert_close(adapter(changed_audio, changed_text, **kwargs), output, rtol=0, atol=0)
    output.square().sum().backward()
    for name, tensor in (("audio", audio), ("text", text)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
    assert not torch.count_nonzero(text.grad[~text_mask])
    assert not torch.count_nonzero(audio.grad[~generation_mask])
    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name

    # Text keys are semantic tokens, so this branch must have no video time window.
    permutation = torch.tensor([2, 4, 0, 3, 1])
    with torch.inference_mode():
        permuted = adapter(
            audio, text[:, permutation], audio_mask=audio_mask,
            text_mask=text_mask[:, permutation], generation_mask=generation_mask,
        )
        torch.testing.assert_close(permuted, output, rtol=1e-5, atol=1e-7)
        adapter.gate.mul_(-2)
        torch.testing.assert_close(adapter(audio, text, **kwargs), -2 * output, rtol=1e-6, atol=1e-7)
        no_text = adapter(audio, text[:, :0], text_mask=text_mask[:, :0])
        assert torch.isfinite(no_text).all() and not torch.count_nonzero(no_text)
    # Modality dropout must leave zero gradients, rather than unused parameters,
    # so a rank that receives only unconditional examples remains DDP-safe.
    for context, context_mask in (
        (text.detach(), torch.zeros_like(text_mask)),
        (text.detach()[:, :0], text_mask[:, :0]),
    ):
        adapter.zero_grad(set_to_none=True)
        absent = adapter(audio.detach(), context, text_mask=context_mask)
        assert not torch.count_nonzero(absent)
        absent.sum().backward()
        for name, parameter in adapter.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert not torch.count_nonzero(parameter.grad), name
    print("[OK] text padding/NaNs, empty contexts, query masks, key permutation and signed gate scaling")


def test_prompt_isolation_and_modality_dropout():
    inputs = make_inputs()
    for isolated in (False, True):
        _, model = warmstarted_pair(prompt_isolated_ca=isolated)
        model.eval()
        module = model.transformer_blocks[6].tail_text_attn
        with torch.no_grad():
            module.gate.fill_(0.2)
        observed = []
        handle = module.register_forward_hook(
            lambda _module, _args, result, observed=observed: observed.append(result.detach().clone())
        )
        try:
            with torch.inference_mode():
                model(**inputs, drop_video=True)
                changed = {**inputs, "text": (inputs["text"] + 5).remainder(16)}
                changed["text"] = changed["text"].masked_fill(~inputs["text_mask"], -1)
                model(**changed, drop_video=True)
                original_drop, _ = model(**inputs, drop_text=True)
                changed_drop, _ = model(**changed, drop_text=True)
            first, second = observed[:2]
            assert torch.count_nonzero(first), "dropping video must retain text injection"
            assert not torch.equal(first, second), "text-only branch must depend on actual text"
            assert not torch.count_nonzero(first[~inputs["mask"]])
            prompt = inputs["mask"] & ~inputs["generation_mask"]
            if isolated:
                assert not torch.count_nonzero(first[prompt])
            else:
                assert torch.count_nonzero(first[prompt]), "nonisolated text must reach all valid queries"
            torch.testing.assert_close(original_drop, changed_drop, rtol=0, atol=0)
            torch.testing.assert_close(observed[2], observed[3], rtol=0, atol=0)
            assert not torch.count_nonzero(observed[2]), "dropped text must disable the new text residual"
        finally:
            handle.remove()
    print("[OK] prompt isolation toggle; dropped video retains text; dropped text cannot leak raw tokens")


def test_parallel_text_visual_residuals():
    _, model = warmstarted_pair()
    model.eval()
    block = model.transformer_blocks[6]
    with torch.no_grad():
        block.tail_text_attn.gate.fill_(0.4)
        block.local_visual_attn.gate.fill_(0.4)
    observed = {}
    handles = []
    for name in ("tail_text_attn", "local_visual_attn"):
        def capture(_module, args, *, branch=name):
            observed[branch] = args[0].detach().clone()
        handles.append(getattr(block, name).register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            model(**make_inputs())
        assert set(observed) == {"tail_text_attn", "local_visual_attn"}
        torch.testing.assert_close(observed["tail_text_attn"], observed["local_visual_attn"], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
    print("[OK] parallel text and visual attention read the same post-self-attention audio state")


def test_cfg_packed_matches_independent_branches():
    _, model = warmstarted_pair()
    model.eval()
    with torch.no_grad():
        for layer, module in text_modules(model).items():
            module.gate.fill_(0.2)
            model.transformer_blocks[layer].local_visual_attn.gate.fill_(0.2)
    null_flags = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
    cases = (
        ({}, [{}, {"drop_video": True}, null_flags]),
        ({"drop_video": True}, [{"drop_video": True}, null_flags]),
        ({"drop_text": True}, [{"drop_text": True}, null_flags]),
    )
    with torch.inference_mode():
        for cache in (False, True):
            inputs = {**make_inputs(), "cache": cache}
            for packed_flags, independent_flags in cases:
                model.clear_cache()
                # Two steps exercise populated conditional/unconditional caches.
                for step in (0, 1):
                    step_inputs = {**inputs, "time": inputs["time"] + step * 0.05,
                                   "x": inputs["x"] + step * 0.1}
                    actual, _ = model(**step_inputs, cfg_infer=True, **packed_flags)
                    expected = torch.cat([model(**step_inputs, **flags)[0] for flags in independent_flags])
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
                model.clear_cache()
    print("[OK] B=2 packed three-/two-branch CFG matches independent execution with cold/warm cache")


def test_all_tail_gradients_and_activation_checkpointing():
    _, plain = warmstarted_pair(checkpoint_activations=False)
    _, checkpointed = warmstarted_pair(checkpoint_activations=True)
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    inputs = make_inputs()
    target = torch.randn_like(inputs["x"])
    results = []
    for model in (plain, checkpointed):
        model.train()
        prediction, ctc = model(**inputs)
        assert torch.isfinite(prediction).all() and set(ctc) == {6, 12}
        loss = (prediction - target).square().mean()
        # Both retained CTC taps must backpropagate alongside the flow head.
        loss = loss + 0.03 * sum(tap["z_tilde"].square().mean() for tap in ctc.values())
        loss.backward()
        gradients = {}
        for layer, module in text_modules(model).items():
            for name, parameter in module.named_parameters():
                label = f"layer_{layer}/{name}"
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), label
                assert torch.count_nonzero(parameter.grad), f"first-step gradient is zero: {label}"
                gradients[label] = parameter.grad.detach().clone()
        assert len(gradients) == 132
        for layer in TAIL_LAYERS:
            gate = model.transformer_blocks[layer].local_visual_attn.gate
            assert gate.grad is not None and torch.isfinite(gate.grad).all() and torch.count_nonzero(gate.grad)
        assert model.speaker_proj.weight.grad is not None and torch.count_nonzero(model.speaker_proj.weight.grad)
        for tap in ctc.values():
            assert torch.equal(tap["z_lens"], torch.tensor([24, 20]))
        results.append((prediction.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-5, atol=1e-9)
    print("[OK] all 12 text gates and projections learn on first backward; checkpoint gradients agree; visual/speaker/CTC paths retained")


def main():
    torch.set_num_threads(1)
    test_default_disabled_and_pretrained_keys()
    test_adapter_masks_empty_context_and_gate_scaling()
    test_prompt_isolation_and_modality_dropout()
    test_parallel_text_visual_residuals()
    test_cfg_packed_matches_independent_branches()
    test_all_tail_gradients_and_activation_checkpointing()
    print("All audio-tail text attention regression tests passed.")


if __name__ == "__main__":
    main()
