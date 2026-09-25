"""CPU contracts for step 4: local video self-attention on the video grid.

Run from this isolated experiment root with PYTHONPATH=src. No dataset,
checkpoint, decoder, or GPU is required. Locality is a single-layer attention
property; upstream video features and stacked blocks retain broader context.
The inherited step-1/2/3 scripts exercise the new option's default None.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import torch
import torch.nn.functional as F
from ema_pytorch import EMA

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.script.misc.smoke_test_av_local_step1 import ARCH, capture_joint_masks, inputs
from aligndit.script.misc.smoke_test_av_visual_gate_step2 import audio_only_oracle
from aligndit.script.misc.smoke_test_va_block_step3 import (
    assert_no_gradient,
    make_block as make_step3_block,
    make_model as make_step3_model,
)


def make_block(radius=2, *, gate=0.37, masking=True, av_radius=1, blocked=True):
    return MMDiTBlock_VT(
        dim=16, heads=2, dim_head=8, dropout=0.0, ff_mult=2,
        text_dim=16, prompt_isolated_ca=False, attn_mask_enabled=masking,
        av_local_window_radius=av_radius, av_visual_delta_gate_init=gate,
        block_video_audio_attention=blocked, vv_local_window_radius=radius,
    ).eval()


def make_model(radius=2, *, checkpoint=False):
    # Reuse nonzero audio/video residuals so full-model comparisons are useful.
    parent = make_step3_model(checkpoint=checkpoint)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint},
        av_local_window_radius=2, av_visual_delta_gate_init=0.37,
        block_video_audio_attention=True, vv_local_window_radius=radius,
    )
    model.load_state_dict(parent.state_dict(), strict=True)
    return model


def local_video_oracle(block, video, video_mask=None):
    """Independent explicit softmax, including each empty attention row."""
    assert block.v_attn.q_norm is None and block.v_attn.k_norm is None
    batch, length, _ = video.shape
    heads = block.v_attn.heads
    head_dim = block.v_attn.inner_dim // heads
    q, k, value = [
        F.linear(video, layer.weight, layer.bias).reshape(batch, length, heads, head_dim).transpose(1, 2)
        for layer in (block.v_attn.to_q, block.v_attn.to_k, block.v_attn.to_v)
    ]
    allowed = torch.ones(batch, 1, length, length, dtype=torch.bool)
    if block.vv_local_window_radius is not None:
        for query in range(length):
            for key in range(length):
                allowed[:, :, query, key] = abs(query - key) <= block.vv_local_window_radius
    if block.attn_mask_enabled and video_mask is not None:
        allowed &= video_mask[:, None, None, :]
    logits = (q @ k.transpose(-1, -2) / math.sqrt(head_dim)).masked_fill(~allowed, -torch.inf)
    nonempty = allowed.any(dim=-1, keepdim=True)
    # Avoid softmax(-inf,...,-inf); the SDPA contract returns zero for that row.
    logits = torch.where(nonempty, logits, 0.0)
    weights = torch.where(nonempty, logits.softmax(dim=-1), 0.0)
    attended = (weights @ value).transpose(1, 2).reshape(batch, length, -1)
    projection = block.v_attn.to_out[0]
    out = F.linear(attended, projection.weight, projection.bias)
    return out.masked_fill(~video_mask[..., None], 0.0) if video_mask is not None else out


def test_oracle_same_layer_audio_and_padding():
    torch.manual_seed(201)
    x, video = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    audio_valid = torch.arange(8)[None] < torch.tensor([8, 6])[:, None]
    video_valid = torch.arange(8)[None] < torch.tensor([6, 0])[:, None]
    for masking in (False, True):
        reference = make_step3_block(radius=1, masking=masking)
        for radius in (None, 0, 1, 2, 20):
            block = make_block(radius, masking=masking)
            block.load_state_dict(reference.state_dict(), strict=True)
            for padding in (False, True):
                kwargs = dict(mask=audio_valid, v_mask=video_valid) if padding else {}
                audio, visual = block.joint_attn(x, video, **kwargs)
                step3_audio, step3_visual = reference.joint_attn(x, video, **kwargs)
                expected = local_video_oracle(block, video, video_valid if padding else None)
                torch.testing.assert_close(audio, step3_audio, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(visual, expected, rtol=2e-5, atol=2e-6)
                assert torch.isfinite(audio).all() and torch.isfinite(visual).all()
                if radius is None or radius >= 7:
                    torch.testing.assert_close(visual, step3_visual, rtol=2e-5, atol=2e-6)
                if padding:
                    assert not torch.count_nonzero(audio[~audio_valid])
                    assert not torch.count_nonzero(visual[~video_valid])
    # VV locality is independent of the optional AV and VA rules, including
    # unequal audio/video lengths at block level. It is in video-token units.
    independent = make_block(1, av_radius=None, blocked=False)
    with capture_joint_masks(13) as captured:
        independent.joint_attn(x[:, :5], video)
    actual = captured[0]
    assert actual is not None and actual[..., :5, :].all() and actual[..., 5:, :5].all()
    for query in range(8):
        for key in range(8):
            assert (actual[..., 5 + query, 5 + key] == (abs(query - key) <= 1)).all()
    print("[OK] independent VV softmax oracle; unchanged same-layer audio; padding/empty video; None/wide fallback")


def test_information_paths_and_gate_endpoints():
    torch.manual_seed(203)
    block = make_block(1, gate=1e-5)
    x = torch.randn(1, 8, 16, requires_grad=True)
    video = torch.randn(1, 8, 16, requires_grad=True)
    audio, visual = block.joint_attn(x, video)
    probe = torch.randn(16)
    dx, dv, dg, dw = torch.autograd.grad(
        (audio[0, 2] * probe).sum(),
        (x, video, block.av_visual_delta_gate, block.v_attn.to_v.weight), retain_graph=True,
    )
    assert dx[0, 7].abs().sum() > 0, "AA must stay global"
    assert dv[0, 1:4].abs().sum() > 0 and not torch.count_nonzero(dv[0, [0, 4, 5, 6, 7]])
    assert torch.isfinite(dg) and dg.abs() > 0
    assert torch.isfinite(dw).all() and dw.abs().sum() > 0, "video remains trainable from the audio loss"
    video_dx, video_dv = torch.autograd.grad((visual[0, 2] * probe).sum(), (x, video), allow_unused=True)
    assert_no_gradient(video_dx)
    assert video_dv[0, 1:4].abs().sum() > 0
    assert not torch.count_nonzero(video_dv[0, [0, 4, 5, 6, 7]]), "direct distant VV must be blocked"
    changed_video = video.detach().clone()
    changed_video[:, 7] += 50
    _, changed = block.joint_attn(x.detach() * 20 + 3, changed_video)
    torch.testing.assert_close(changed[:, 2], visual[:, 2], rtol=0, atol=0)
    ungated = make_block(1, gate=None)
    state = {name: value for name, value in block.state_dict().items() if name != "av_visual_delta_gate"}
    ungated.load_state_dict(state, strict=True)
    with torch.no_grad():
        block.av_visual_delta_gate.zero_()
        actual, _ = block.joint_attn(x, video)
        torch.testing.assert_close(actual, audio_only_oracle(block, x), rtol=2e-5, atol=2e-6)
        block.av_visual_delta_gate.fill_(1)
        actual, _ = block.joint_attn(x, video)
        expected, _ = ungated.joint_attn(x, video)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    print("[OK] local VV/AV gradients; blocked VA; global AA; trainable visual/gate paths; gate endpoints")


def test_full_model_cfg_prefix_and_compatibility():
    legacy = make_step3_model().eval()
    local, wide, disabled = (make_model(radius).eval() for radius in (2, 20, None))
    assert local.state_dict().keys() == legacy.state_dict().keys()
    for model in (local, wide, disabled):
        model.load_state_dict(legacy.state_dict(), strict=True)
    assert all(block.vv_local_window_radius == 2 for block in local.transformer_blocks[:2])
    assert all(not hasattr(block, "vv_local_window_radius") for block in local.transformer_blocks[2:])
    ema = EMA(local, include_online_model=False)
    assert all(block.vv_local_window_radius == 2 for block in ema.ema_model.transformer_blocks[:2])
    assert all(not parameter.requires_grad for parameter in ema.ema_model.parameters())
    kwargs = inputs()
    with torch.inference_mode():
        expected = legacy(**kwargs)[0]
        assert torch.count_nonzero(expected)
        for model in (wide, disabled):
            torch.testing.assert_close(model(**kwargs)[0], expected, rtol=3e-5, atol=3e-6)
    null = dict(drop_audio_cond=True, drop_text=True, drop_video=True)
    for flags, branches in (({}, [{}, {"drop_video": True}, null]),
                            ({"drop_video": True}, [{"drop_video": True}, null]),
                            ({"drop_text": True}, [{"drop_text": True}, null])):
        with torch.inference_mode():
            packed = local(**kwargs, cfg_infer=True, **flags)[0]
            sequential = torch.cat([local(**kwargs, **branch)[0] for branch in branches])
        torch.testing.assert_close(packed, sequential, rtol=3e-5, atol=3e-6)
    with torch.inference_mode():
        dropped = local(**kwargs, drop_video=True)[0]
        altered = local(**{**kwargs, "video": kwargs["video"] * 30 + 20}, drop_video=True)[0]
    torch.testing.assert_close(dropped, altered, rtol=0, atol=0)
    cfm = CFM_VT(transformer=local, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    output, _ = cfm.sample(
        cond=kwargs["cond"][:, :3], text=kwargs["text"], duration=torch.tensor([12, 10]),
        video=kwargs["video"], lens=torch.tensor([3, 3]), speaker_embedding=kwargs["speaker_embedding"],
        steps=1, use_epss=False, cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :3], kwargs["cond"][:, :3], rtol=0, atol=0)
    # Sampling resolves duration against text length.  Long text therefore
    # requires zero-padding a shorter real-video clip before local AV/VV masks
    # are constructed (the formal S1 list contains this corner case).
    long_text = torch.zeros((1, 15), dtype=torch.long)
    with torch.inference_mode():
        long_output, _ = cfm.sample(
            cond=kwargs["cond"][:1, :3], text=long_text, duration=torch.tensor([12]),
            video=kwargs["video"][:1], lens=torch.tensor([3]),
            speaker_embedding=kwargs["speaker_embedding"][:1], steps=1,
            use_epss=False, cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
        )
    assert long_output.shape == (1, 16, 64) and torch.isfinite(long_output).all()
    torch.testing.assert_close(long_output[:, :3], kwargs["cond"][:1, :3], rtol=0, atol=0)
    print("[OK] state-key/EMA compatibility; full-model None/wide fallback; packed CFG/null video; reference prefix; long-text video padding")


def test_training_recomputation_and_bfloat16():
    results = []
    for checkpoint in (False, True):
        transformer = make_model(checkpoint=checkpoint)
        cfm = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03,
            audio_drop_prob=0.0, cond_drop_prob=0.0, text_drop_prob=0.0, video_drop_prob=0.0,
        ).train()
        torch.manual_seed(207)
        kwargs = inputs()
        with capture_joint_masks(24) as captured, patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = cfm(
                inp=kwargs["x"], text=kwargs["text"], video=kwargs["video"],
                lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 4]),
                video_lens=torch.tensor([12, 10]), speaker_embedding=kwargs["speaker_embedding"],
            )
            assert torch.isfinite(loss) and math.isfinite(components["ctc_loss"])
            loss.backward()
        assert len(captured) >= (4 if checkpoint else 2), "capture must cover recomputation"
        positions = torch.arange(12)
        local_window = (positions[:, None] - positions[None, :]).abs() <= 2
        for mask in captured:
            assert mask is not None and mask[..., :12, :12].all()
            assert not mask[..., 12:, :12].any()
            assert torch.equal(mask[..., 12:, 12:], local_window.expand_as(mask[..., 12:, 12:]))
            assert torch.equal(mask[..., :12, 12:], local_window.expand_as(mask[..., :12, 12:]))
        parameters = [block.av_visual_delta_gate for block in transformer.transformer_blocks[:2]]
        parameters += [transformer.transformer_blocks[0].v_attn.to_v.weight, transformer.speaker_proj.weight]
        for parameter in parameters:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        results.append((loss.detach(), *(parameter.grad.detach().clone() for parameter in parameters)))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-8)
    block = make_block(2, gate=1e-5)
    x, video = torch.randn(2, 8, 16, requires_grad=True), torch.randn(2, 8, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        audio, visual = block.joint_attn(x, video)
        loss = audio.float().square().sum()
    assert_no_gradient(torch.autograd.grad(visual.float().sum(), x, retain_graph=True, allow_unused=True)[0])
    loss.backward()
    assert block.av_visual_delta_gate.grad.dtype == torch.float32
    assert torch.isfinite(block.av_visual_delta_gate.grad) and block.av_visual_delta_gate.grad.abs() > 0
    assert torch.isfinite(video.grad).all() and video.grad.abs().sum() > 0
    print("[OK] train mask=None topology; CFM/CTC backward; activation recomputation parity; bf16 finite gradients")


def test_validation():
    for value, error_type in ((True, TypeError), (1.5, TypeError), (-1, ValueError)):
        for factory in (make_block, make_model):
            try:
                factory(value)
            except error_type as error:
                assert "vv_local_window_radius" in str(error)
            else:
                raise AssertionError(f"Expected {error_type.__name__} for radius={value!r}")
    print("[OK] invalid VV radii rejected in block and backbone")


def main():
    torch.set_num_threads(1)
    test_oracle_same_layer_audio_and_padding()
    test_information_paths_and_gate_endpoints()
    test_full_model_cfg_prefix_and_compatibility()
    test_training_recomputation_and_bfloat16()
    test_validation()
    print("All VV-local step-4 contracts passed.")


if __name__ == "__main__":
    main()
