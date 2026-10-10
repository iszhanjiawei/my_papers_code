"""CPU regressions for six-layer Flowley progressive visual fade weights.

Run with PYTHONPATH=src python scripts/test_progressive_visual_window.py.
The oracle is extracted from the local Flowley checkout, without importing its
optional dependencies. Pass --flowley-root for a different checkout location.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from pathlib import Path

import torch
from test_audio_local_visual_attention import BASE_ARCH, make_inputs

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT


ARCH = {
    **BASE_ARCH,
    "depth": 18,
    "n_mm_layers": 12,
    "n_text_layers": 12,
    "speaker_condition_start_layer": 12,
    "layer_indices_ctc": [6, 12],
    "normalize_text_context": True,
    "audio_local_visual_attention": True,
}
TAIL_LAYERS = list(range(12, 18))


def load_flowley_oracle(root):
    source = root / "flowley/model/modules/layers/attention.py"
    code = source.read_text()
    function = next(
        node for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_audio_visual_cross_attn_mask"
    )
    namespace = {"math": math, "torch": torch, "Tensor": torch.Tensor}
    # Execute only this function from the user-provided local source checkout.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    print(f"Flowley oracle: {source}; sha256={hashlib.sha256(code.encode()).hexdigest()}")
    return namespace[function.name]


def local_modules(model):
    return {
        layer: block.local_visual_attn
        for layer, block in enumerate(model.transformer_blocks)
        if getattr(block, "local_visual_attn", None) is not None
    }


def make_model(schedule="flowley_progressive", *, checkpoint_activations=False, **overrides):
    torch.manual_seed(17)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint_activations,
           "local_visual_window_schedule": schedule, **overrides}
    )
    # Exercise adapters behind pretrained nonzero modulation/output weights.
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        model.proj_out.weight.normal_(std=0.03)
        model.norm_out.linear.weight.normal_(std=0.03)
        model.speaker_proj.weight.normal_(std=0.03)
    return model


def test_schedule_and_checkpoint_compatibility():
    fixed, model = make_model("fixed"), make_model()
    assert list(local_modules(model)) == TAIL_LAYERS
    assert all(isinstance(block, MMDiTBlock_VT) for block in model.transformer_blocks[:12])
    assert not any(hasattr(block, "tail_text_attn") for block in model.transformer_blocks)
    assert model.speaker_condition_start_layer == 12 and model.layer_indices_ctc == (6, 12)
    expected = [1 - i / 5 for i in range(6)]
    assert [m.window_fade_scale for m in local_modules(model).values()] == expected
    assert [m.window_fade_scale for m in local_modules(fixed).values()] == [1.0] * 6
    scaled = make_model(local_visual_window_fade_scale=0.4)
    assert [m.window_fade_scale for m in local_modules(scaled).values()] == [0.4 * b for b in expected]
    assert set(fixed.state_dict()) == set(model.state_dict())
    assert sum(p.numel() for p in fixed.parameters()) == sum(p.numel() for p in model.parameters())
    for name, value in fixed.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    model.load_state_dict(fixed.state_dict(), strict=True)
    for module in local_modules(model).values():
        torch.testing.assert_close(module.gate, torch.full_like(module.gate, 1e-5), rtol=0, atol=0)
    torch.testing.assert_close(
        model.transformer_blocks[12].local_visual_attn.temporal_bias(44, 44, torch.device("cpu")),
        fixed.transformer_blocks[12].local_visual_attn.temporal_bias(44, 44, torch.device("cpu")),
        rtol=0, atol=0,
    )
    last_bias = model.transformer_blocks[17].local_visual_attn.temporal_bias(44, 44, torch.device("cpu"))
    torch.testing.assert_close(last_bias.exp(), torch.eye(44) + 1e-6, rtol=1e-6, atol=1e-12)
    assert torch.isfinite(last_bias).all(), "beta=0 retains Flowley's epsilon floor"
    print("[OK] MM12/audio6 placement, beta schedule, no tail text gates, unchanged state/parameter schema")


def test_flowley_oracle_and_time_conversion(reference):
    model = make_model()
    for module in local_modules(model).values():
        actual = module.temporal_bias(61, 61, torch.device("cpu"))
        expected = reference(61, 61, video_fps=40, audio_fps=40, window_size=0,
                             fade=True, fade_range=20, fade_type="cosine",
                             fade_scale=module.window_fade_scale)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-6)
        assert module.window_radius_seconds == 0.5
        assert module.window_reference_fps == 8.0 and module.window_core_radius == 0.0
        weights = actual.exp()
        if module.window_fade_scale > 0:
            assert weights[30, 49] > weights[30, 50]
        torch.testing.assert_close(weights[30, 50:52], torch.full((2,), 1e-6), rtol=1e-6, atol=1e-12)
        if module.window_fade_scale == 0:
            torch.testing.assert_close(weights[30, 49], weights[30, 50], rtol=0, atol=0)
    native = make_model(audio_video_ratio=5)
    for layer in (12, 14, 17):
        module = native.transformer_blocks[layer].local_visual_attn
        expected = reference(67, 14, video_fps=8, audio_fps=40, window_size=0,
                             fade=True, fade_range=4, fade_type="cosine",
                             fade_scale=module.window_fade_scale)
        torch.testing.assert_close(module.temporal_bias(67, 14, torch.device("cpu")), expected, rtol=0, atol=0)
        lengths = torch.tensor([14, 9])
        batched = module.temporal_bias(67, 14, torch.device("cpu"), lengths)
        for index, length in enumerate(lengths.tolist()):
            expected_valid = reference(67, length, video_fps=8, audio_fps=40, window_size=0,
                                       fade=True, fade_range=4, fade_type="cosine",
                                       fade_scale=module.window_fade_scale)
            torch.testing.assert_close(batched[index, :, :length], expected_valid, rtol=0, atol=0)
    print("[OK] upstream Flowley oracle at all six depths, physical-time mapping and valid-length clamping")


def test_single_layer_default_and_invalid_schedule():
    arch = {**BASE_ARCH, "depth": 3, "audio_local_visual_attention": True}
    single = DiT_VT_MMDiT(**arch, local_visual_window_schedule="flowley_progressive")
    assert [m.window_fade_scale for m in local_modules(single).values()] == [1.0]
    uneven = DiT_VT_MMDiT(**{**arch, "depth": 5}, local_visual_window_schedule="flowley_progressive")
    assert list(local_modules(uneven)) == [2, 3, 4]
    assert [m.window_fade_scale for m in local_modules(uneven).values()] == [1, 0.5, 0]
    torch.manual_seed(53)
    implicit = DiT_VT_MMDiT(**arch)
    torch.manual_seed(53)
    explicit = DiT_VT_MMDiT(**arch, local_visual_window_schedule="fixed")
    for name, tensor in implicit.state_dict().items():
        torch.testing.assert_close(tensor, explicit.state_dict()[name], rtol=0, atol=0)
    inputs = make_inputs()
    with torch.inference_mode():
        torch.testing.assert_close(implicit.eval()(**inputs)[0], explicit.eval()(**inputs)[0], rtol=0, atol=0)
    for invalid in ("unknown", "", "Flowley"):
        try:
            DiT_VT_MMDiT(**arch, local_visual_window_schedule=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid schedule accepted: {invalid!r}")
    try:
        DiT_VT_MMDiT(**{**arch, "audio_local_visual_attention": False},
                     local_visual_window_schedule="flowley_progressive")
    except ValueError:
        pass
    else:
        raise AssertionError("Progressive schedule accepted without visual adapters")
    print("[OK] single/uneven tail, backward-compatible fixed default, invalid schedule rejection")


def test_zero_visual_gates_restore_fixed_outputs():
    fixed, progressive = make_model("fixed").eval(), make_model().eval()
    for model in (fixed, progressive):
        with torch.no_grad():
            for module in local_modules(model).values():
                module.gate.zero_()
    inputs = make_inputs()
    with torch.inference_mode():
        expected, expected_ctc = fixed(**inputs)
        actual, actual_ctc = progressive(**inputs)
    assert torch.count_nonzero(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer, expected_tap in expected_ctc.items():
        for name, tensor in expected_tap.items():
            torch.testing.assert_close(actual_ctc[layer][name], tensor, rtol=0, atol=0)
    print("[OK] zero visual gates exactly recover fixed-window audio and CTC outputs")


def test_padding_and_absent_visual_conditions():
    model = make_model()
    for layer in (12, 14, 17):
        module = model.transformer_blocks[layer].local_visual_attn
        with torch.no_grad():
            module.gate.fill_(0.2)
        x = torch.randn(2, 9, 32, requires_grad=True)
        video = torch.randn(2, 7, 16, requires_grad=True)
        audio_mask = torch.arange(9)[None] < torch.tensor([9, 6])[:, None]
        video_mask = torch.arange(7)[None] < torch.tensor([5, 0])[:, None]
        generation_mask = audio_mask.clone()
        generation_mask[:, :2] = False
        kwargs = {"audio_mask": audio_mask, "video_mask": video_mask, "generation_mask": generation_mask}
        output = module(x, video, **kwargs)
        changed_x = x.detach().masked_fill(~generation_mask[..., None], float("nan"))
        changed_video = video.detach().masked_fill(~video_mask[..., None], float("nan"))
        torch.testing.assert_close(module(changed_x, changed_video, **kwargs), output, rtol=0, atol=0)
        assert torch.isfinite(output).all() and not torch.count_nonzero(output[1])
        assert not torch.count_nonzero(output[~generation_mask])
        output.square().sum().backward()
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (layer, name)
        assert video.grad is not None and not torch.count_nonzero(video.grad[~video_mask])
    print("[OK] first/middle/final-layer masks, NaN isolation, empty video context and finite gradients")


def test_cfg_and_modality_dropout():
    model = make_model().eval()
    with torch.no_grad():
        for module in local_modules(model).values():
            module.gate.fill_(0.2)
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
                for step in (0, 1):
                    sample = {**inputs, "time": inputs["time"] + 0.05 * step, "x": inputs["x"] + 0.1 * step}
                    actual, _ = model(**sample, cfg_infer=True, **packed_flags)
                    expected = torch.cat([model(**sample, **flags)[0] for flags in independent_flags])
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        model.clear_cache()
        inputs = make_inputs()
        changed_video = {**inputs, "video": torch.randn_like(inputs["video"]) * 100}
        torch.testing.assert_close(model(**inputs, drop_video=True)[0], model(**changed_video, drop_video=True)[0], rtol=0, atol=0)
        changed_text = {**inputs, "text": (inputs["text"] + 5).remainder(16).masked_fill(~inputs["text_mask"], -1)}
        torch.testing.assert_close(model(**inputs, drop_text=True)[0], model(**changed_text, drop_text=True)[0], rtol=0, atol=0)
    print("[OK] cached/uncached three-/two-branch CFG, dropped text/video isolation")


def test_checkpointed_backward_all_layers():
    plain, checkpointed = make_model(), make_model(checkpoint_activations=True)
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    inputs = make_inputs()
    target = torch.randn_like(inputs["x"])
    results = []
    for model in (plain, checkpointed):
        model.train()
        prediction, ctc = model(**inputs)
        assert torch.isfinite(prediction).all() and set(ctc) == {6, 12}
        loss = (prediction - target).square().mean() + 0.03 * sum(tap["z_tilde"].square().mean() for tap in ctc.values())
        loss.backward()
        gradients = {}
        for layer, module in local_modules(model).items():
            for name, parameter in module.named_parameters():
                label = f"layer_{layer}/{name}"
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), label
                # Final beta=0 Q/K gradients can vanish in reduced precision.
                if name in ("gate", "to_v.weight", "to_out.weight"):
                    assert torch.count_nonzero(parameter.grad), label
                gradients[label] = parameter.grad.detach().clone()
        assert len(gradients) == 66
        results.append((prediction.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-5, atol=1e-9)
    print("[OK] all 66 visual adapter gradients finite, final zero-beta layer, activation checkpoint parity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowley-root", type=Path, default=Path(__file__).resolve().parents[3] / "papers_codes/Flowley")
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = load_flowley_oracle(args.flowley_root)
    test_schedule_and_checkpoint_compatibility()
    test_flowley_oracle_and_time_conversion(reference)
    test_single_layer_default_and_invalid_schedule()
    test_zero_visual_gates_restore_fixed_outputs()
    test_padding_and_absent_visual_conditions()
    test_cfg_and_modality_dropout()
    test_checkpointed_backward_all_layers()
    print("All progressive visual window regression tests passed.")


if __name__ == "__main__":
    main()
