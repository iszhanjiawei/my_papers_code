"""CPU contracts for the step-2 visual difference gate.

Run from this experiment root with PYTHONPATH=src. No data, pretrained weights,
GPU, or training job is needed. The attention oracle uses explicit softmax rather
than the implementation's SDPA, and locality assertions apply to one layer only.
The unchanged VA/VV paths can carry distant information across multiple layers.
"""

from __future__ import annotations

import io
import math
from unittest.mock import patch

import torch
import torch.nn.functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.script.misc.smoke_test_av_local_step1 import (
    ARCH, capture_joint_masks, expected_mask, inputs, make_model as make_step1_model,
)


def make_block(gate=1e-5, radius=2, *, attn_mask_enabled=True):
    return MMDiTBlock_VT(
        dim=16, heads=2, dim_head=8, dropout=0.0, ff_mult=2,
        text_dim=16, prompt_isolated_ca=False,
        attn_mask_enabled=attn_mask_enabled, av_local_window_radius=radius,
        av_visual_delta_gate_init=gate,
    ).eval()


def make_model(gate=1e-5, *, checkpoint_activations=False):
    # Shared, nonzero warm-start weights avoid vacuous all-zero output tests.
    parent = make_step1_model(checkpoint_activations=checkpoint_activations)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint_activations},
        av_local_window_radius=2, av_visual_delta_gate_init=gate,
    )
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    assert missing == [f"transformer_blocks.{i}.av_visual_delta_gate" for i in range(2)]
    assert not unexpected
    return model


def audio_only_oracle(block, x, mask=None):
    """Independent mathematical AA oracle (no RoPE/QK norm in these blocks)."""
    assert block.attn.q_norm is None and block.attn.k_norm is None
    batch, length, _ = x.shape
    heads = block.attn.heads
    head_dim = block.attn.inner_dim // heads
    q, k, value = [
        F.linear(x, layer.weight, layer.bias).reshape(batch, length, heads, head_dim).transpose(1, 2)
        for layer in (block.attn.to_q, block.attn.to_k, block.attn.to_v)
    ]
    logits = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    if block.attn_mask_enabled and mask is not None:
        logits = logits.masked_fill(~mask[:, None, None], -torch.inf)
    attended = (logits.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, length, -1)
    projection = block.attn.to_out[0]
    out = F.linear(attended, projection.weight, projection.bias)
    return out.masked_fill(~mask[..., None], 0.0) if mask is not None else out


def test_gate_endpoints_oracle_and_shared_gradients():
    torch.manual_seed(71)
    x, video = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    audio_valid = torch.arange(8)[None] < torch.tensor([8, 6])[:, None]
    video_valid = torch.arange(8)[None] < torch.tensor([7, 0])[:, None]
    for masking in (False, True):
        for radius in (0, 2, 20):
            original = make_block(None, radius, attn_mask_enabled=masking)
            gated = make_block(1.0, radius, attn_mask_enabled=masking)
            missing, extra = gated.load_state_dict(original.state_dict(), strict=False)
            assert missing == ["av_visual_delta_gate"] and not extra
            for with_padding in (False, True):
                kwargs = dict(mask=audio_valid, v_mask=video_valid) if with_padding else {}
                actual_a, actual_v = gated.joint_attn(x, video, **kwargs)
                reference_a, reference_v = original.joint_attn(x, video, **kwargs)
                torch.testing.assert_close(actual_a, reference_a, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual_v, reference_v, rtol=0, atol=0)
                original.zero_grad(set_to_none=True)
                gated.zero_grad(set_to_none=True)
                (reference_a.square().mean() + reference_v.square().mean()).backward()
                (actual_a.square().mean() + actual_v.square().mean()).backward()
                for name, parameter in original.named_parameters():
                    actual_grad = dict(gated.named_parameters())[name].grad
                    if parameter.grad is None:
                        assert actual_grad is None, name
                    else:
                        torch.testing.assert_close(actual_grad, parameter.grad, rtol=5e-5, atol=1e-7, msg=name)
                with torch.no_grad():
                    gated.av_visual_delta_gate.zero_()
                    zero_a, zero_v = gated.joint_attn(x, video, **kwargs)
                    oracle = audio_only_oracle(gated, x, kwargs.get("mask"))
                    torch.testing.assert_close(zero_a, oracle, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(zero_v, reference_v, rtol=0, atol=0)
                    altered_a, _ = gated.joint_attn(x, video * 40 + 19, **kwargs)
                    torch.testing.assert_close(altered_a, zero_a, rtol=0, atol=0)
                    gated.av_visual_delta_gate.fill_(0.37)
                    middle_a, middle_v = gated.joint_attn(x, video, **kwargs)
                    torch.testing.assert_close(middle_a, oracle * 0.63 + reference_a * 0.37, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(middle_v, reference_v, rtol=0, atol=0)
                    gated.av_visual_delta_gate.fill_(1.0)
    print("[OK] g=0 explicit-softmax AA oracle, g=1 step-1 outputs/shared gradients, intermediate g, unchanged video")


def test_live_gate_gradients_and_locality():
    torch.manual_seed(73)
    block = make_block(radius=1)
    x = torch.randn(1, 8, 16, requires_grad=True)
    video = torch.randn(1, 8, 16, requires_grad=True)
    probe = torch.randn(16)
    gradients = {}
    for gate in (0.0, 1e-5, 1.0):
        with torch.no_grad():
            block.av_visual_delta_gate.fill_(gate)
        audio_out, video_out = block.joint_attn(x, video)
        dx, dv, dg = torch.autograd.grad(
            (audio_out[0, 2] * probe).sum(), (x, video, block.av_visual_delta_gate), retain_graph=True,
        )
        assert torch.isfinite(dg) and dg.abs() > 1e-8, "gate must receive a live gradient even at zero"
        assert dx[0, 7].abs().sum() > 0, "AA remains global"
        assert not torch.count_nonzero(dv[0, [0, 4, 5, 6, 7]]), "direct AV remains local"
        gradients[gate] = dv
        feedback_dx, feedback_dv = torch.autograd.grad(video_out[0, 2].square().sum(), (x, video))
        assert feedback_dx[0, 7].abs().sum() > 0, "VA must remain global in step 2"
        assert feedback_dv[0, 7].abs().sum() > 0, "VV must remain global in step 2"
    assert not torch.count_nonzero(gradients[0.0])
    assert gradients[1e-5][0, 1:4].abs().sum() > 0
    torch.testing.assert_close(gradients[1e-5], gradients[1.0] * 1e-5, rtol=3e-5, atol=1e-11)
    changed = video.detach().clone()
    changed[:, 7] += 100
    with torch.no_grad():
        original, _ = block.joint_attn(x, video)
        altered, _ = block.joint_attn(x, changed)
    torch.testing.assert_close(original[:, 2], altered[:, 2], rtol=0, atol=0)
    print("[OK] nonzero gate gradients at 0/1e-5/1; scaled visual gradients; local AV and global AA/VA/VV")


def test_cpu_bfloat16_backward():
    block = make_block()
    x, video = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        audio, _ = block.joint_attn(x, video)
        loss = (audio.float() * torch.randn_like(audio, dtype=torch.float32)).sum()
    loss.backward()
    gate = block.av_visual_delta_gate
    assert gate.dtype == torch.float32 and gate.grad.dtype == torch.float32
    assert torch.isfinite(gate.grad) and gate.grad.abs() > 0
    visual_gradient = block.v_attn.to_v.weight.grad
    assert torch.isfinite(visual_gradient).all() and visual_gradient.abs().sum() > 0
    print("[OK] CPU bf16 autocast retains float32 gate and nonzero gate/video gradients")


def test_gate_registration_and_roundtrip():
    full_depth = DiT_VT_MMDiT(
        **{**ARCH, "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
           "speaker_condition_start_layer": 12, "layer_indices_ctc": [5, 11]},
        av_local_window_radius=2, av_visual_delta_gate_init=1e-5,
    )
    gates = {name: p for name, p in full_depth.named_parameters() if name.endswith("av_visual_delta_gate")}
    assert list(gates) == [f"transformer_blocks.{i}.av_visual_delta_gate" for i in range(12)]
    assert all(p.shape == torch.Size([]) and p.dtype == torch.float32 and p.requires_grad for p in gates.values())
    for parameter in gates.values():
        torch.testing.assert_close(parameter, torch.tensor(1e-5), rtol=0, atol=0)

    parent = make_step1_model().eval()
    model = make_model(1.0).eval()
    kwargs = inputs()
    with torch.inference_mode():
        reference, reference_ctc = parent(**kwargs)
        actual, actual_ctc = model(**kwargs)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    for layer in reference_ctc:
        torch.testing.assert_close(actual_ctc[layer]["z_tilde"], reference_ctc[layer]["z_tilde"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(**kwargs)[0].square().mean()
    loss.backward()
    optimizer.step()
    expected_gates = {name: p.detach().clone() for name, p in model.named_parameters() if name.endswith("av_visual_delta_gate")}
    assert all(optimizer.state[dict(model.named_parameters())[name]]["step"] == 1 for name in expected_gates)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = make_model().eval()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for name, expected in expected_gates.items():
        parameter = dict(restored.named_parameters())[name]
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizer.state[parameter]["exp_avg"], optimizer.state[dict(model.named_parameters())[name]]["exp_avg"])
    with torch.inference_mode():
        torch.testing.assert_close(restored(**kwargs)[0], model(**kwargs)[0], rtol=0, atol=0)
    print("[OK] exactly 12 scalar gates; step-1 checkpoint warm start; model/CTC parity; optimizer/checkpoint roundtrip")


def test_training_checkpoint_and_padding_masks():
    records = []
    for checkpoint in (False, True):
        transformer = make_model(checkpoint_activations=checkpoint)
        cfm = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1,
            ctc_lambda=0.03, audio_drop_prob=0.0, cond_drop_prob=0.0,
            text_drop_prob=0.0, video_drop_prob=0.0,
        ).train()
        torch.manual_seed(79)
        kwargs = inputs()
        with capture_joint_masks(24) as captured, patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = cfm(
                inp=kwargs["x"], text=kwargs["text"], video=kwargs["video"],
                lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 4]),
                video_lens=torch.tensor([12, 10]), speaker_embedding=kwargs["speaker_embedding"],
            )
            assert torch.isfinite(loss) and "ctc_loss" in components
            loss.backward()
        assert len(captured) >= 2
        for mask in captured:
            assert torch.equal(mask, expected_mask(12, 2, batch=2).expand_as(mask))
        gate_gradients = []
        for block in transformer.transformer_blocks[:2]:
            gradient = block.av_visual_delta_gate.grad
            assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0
            gate_gradients.append(gradient.detach().clone())
        video_gradient = transformer.transformer_blocks[0].v_attn.to_v.weight.grad
        assert torch.isfinite(video_gradient).all() and video_gradient.abs().sum() > 0
        records.append((loss.detach(), torch.stack(gate_gradients), video_gradient.detach().clone()))
    for actual, expected in zip(records[1], records[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-9)
    print("[OK] CFM/CTC training backward with/without activation checkpoint; gate gradients; training structural mask")


def test_cfg_packed_matches_sequential_and_null_video():
    torch.manual_seed(83)
    model = make_model(0.4).eval()
    # Make all conditioning routes observable, including speaker branch order.
    with torch.no_grad():
        model.speaker_proj.weight.normal_(std=0.03)
        for block in model.transformer_blocks[:2]:
            block.cross_attn_ada.weight.normal_(std=0.03)
            block.v_attn_norm.linear.weight.normal_(std=0.03)
    kwargs = inputs()
    null = dict(drop_audio_cond=True, drop_text=True, drop_video=True)
    for flags, branches in (({}, [{}, {"drop_video": True}, null]),
                            ({"drop_video": True}, [{"drop_video": True}, null]),
                            ({"drop_text": True}, [{"drop_text": True}, null])):
        with torch.inference_mode():
            packed = model(**kwargs, cfg_infer=True, **flags)[0]
            sequential = torch.cat([model(**kwargs, **branch)[0] for branch in branches], dim=0)
        torch.testing.assert_close(packed, sequential, rtol=3e-5, atol=3e-6)
    with torch.inference_mode():
        dropped = model(**kwargs, drop_video=True)[0]
        altered = model(**{**kwargs, "video": kwargs["video"] * 100 + 15}, drop_video=True)[0]
    torch.testing.assert_close(dropped, altered, rtol=0, atol=0)
    cfm = CFM_VT(transformer=model, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    output, _ = cfm.sample(
        cond=kwargs["cond"][:, :3], text=kwargs["text"], duration=torch.tensor([12, 10]),
        video=kwargs["video"], lens=torch.tensor([3, 3]),
        speaker_embedding=kwargs["speaker_embedding"], steps=1, use_epss=False,
        cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
    )
    assert output.shape == (2, 12, 64) and torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :3], kwargs["cond"][:, :3], rtol=0, atol=0)
    print("[OK] B=2 packed CFG matches sequential 2/3 branches; null-video independence; CFM sample preserves prefix")


def main():
    torch.set_num_threads(1)
    test_gate_endpoints_oracle_and_shared_gradients()
    test_live_gate_gradients_and_locality()
    test_cpu_bfloat16_backward()
    test_gate_registration_and_roundtrip()
    test_training_checkpoint_and_padding_masks()
    test_cfg_packed_matches_sequential_and_null_video()
    print("All AV visual-gate step-2 contracts passed.")


if __name__ == "__main__":
    main()
