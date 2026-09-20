"""CPU contracts for step 3: video queries cannot read audio keys/values.

Run from this experiment root with PYTHONPATH=src. No dataset, checkpoint, VAE
decoder, or GPU is required. The inherited step-1/2 smoke scripts separately
exercise the backward-compatible default (block_video_audio_attention=False).
Direct AV locality is asserted within a layer; VV and AA still propagate global
context across layers. The video stream is trainable, not frozen or detached.
"""

from __future__ import annotations

import io
import math
from unittest.mock import patch

import torch
import torch.nn.functional as F
from ema_pytorch import EMA

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.script.misc.smoke_test_av_local_step1 import ARCH, capture_joint_masks, inputs
from aligndit.script.misc.smoke_test_av_visual_gate_step2 import make_model as make_step2_model


def make_block(*, blocked=True, gate=0.37, radius=1, masking=True):
    return MMDiTBlock_VT(
        dim=16, heads=2, dim_head=8, dropout=0.0, ff_mult=2,
        text_dim=16, prompt_isolated_ca=False, attn_mask_enabled=masking,
        av_local_window_radius=radius, av_visual_delta_gate_init=gate,
        block_video_audio_attention=blocked,
    ).eval()


def make_model(*, blocked=True, checkpoint=False):
    parent = make_step2_model(0.37, checkpoint_activations=checkpoint)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint},
        av_local_window_radius=2, av_visual_delta_gate_init=0.37,
        block_video_audio_attention=blocked,
    )
    model.load_state_dict(parent.state_dict(), strict=True)
    # Nonzero video residuals make multilayer isolation checks meaningful.
    with torch.no_grad():
        for block in model.transformer_blocks[:2]:
            block.v_attn_norm.linear.weight.normal_(std=0.03)
            block.v_attn_norm.linear.bias.normal_(std=0.03)
            block.cross_attn_ada.weight.normal_(std=0.03)
        model.speaker_proj.weight.normal_(std=0.03)
    return model


def video_only_oracle(block, video, mask=None):
    """Independent explicit-softmax VV oracle, including all-masked rows."""
    batch, length, _ = video.shape
    heads = block.v_attn.heads
    head_dim = block.v_attn.inner_dim // heads
    q, k, value = [
        F.linear(video, layer.weight, layer.bias).reshape(batch, length, heads, head_dim).transpose(1, 2)
        for layer in (block.v_attn.to_q, block.v_attn.to_k, block.v_attn.to_v)
    ]
    logits = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    if block.attn_mask_enabled and mask is not None:
        logits = logits.masked_fill(~mask[:, None, None], -torch.inf)
    weights = logits.softmax(dim=-1)
    if block.attn_mask_enabled and mask is not None:
        weights = torch.where(mask.any(dim=-1)[:, None, None, None], weights, 0.0)
    attended = (weights @ value).transpose(1, 2).reshape(batch, length, -1)
    projection = block.v_attn.to_out[0]
    out = F.linear(attended, projection.weight, projection.bias)
    return out.masked_fill(~mask[..., None], 0.0) if mask is not None else out


def assert_no_gradient(gradient):
    assert gradient is None or not torch.count_nonzero(gradient)


def test_video_oracle_audio_parity_and_padding():
    torch.manual_seed(101)
    x, video = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    audio_valid = torch.arange(8)[None] < torch.tensor([8, 6])[:, None]
    video_valid = torch.arange(8)[None] < torch.tensor([7, 0])[:, None]
    for masking in (False, True):
        # Blocking must also work without the local-window option.
        for radius in (None, 0, 2):
            reference = make_block(blocked=False, radius=radius, masking=masking)
            blocked = make_block(radius=radius, masking=masking)
            blocked.load_state_dict(reference.state_dict(), strict=True)
            for padding in (False, True):
                kwargs = dict(mask=audio_valid, v_mask=video_valid) if padding else {}
                actual_a, actual_v = blocked.joint_attn(x, video, **kwargs)
                reference_a, _ = reference.joint_attn(x, video, **kwargs)
                oracle_v = video_only_oracle(blocked, video, video_valid if padding else None)
                torch.testing.assert_close(actual_a, reference_a, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual_v, oracle_v, rtol=2e-5, atol=2e-6)
                assert torch.isfinite(actual_a).all() and torch.isfinite(actual_v).all()
                altered_a, altered_v = blocked.joint_attn(x * 20 + 13, video, **kwargs)
                torch.testing.assert_close(altered_v, actual_v, rtol=0, atol=0)
                assert not torch.allclose(altered_a, actual_a)
                if padding:
                    assert not torch.count_nonzero(actual_a[~audio_valid])
                    assert not torch.count_nonzero(actual_v[~video_valid])
    print("[OK] VV explicit-softmax oracle; same-input audio parity with step 2; all-video padding; radius=None")


def test_forward_and_backward_information_paths():
    torch.manual_seed(103)
    block = make_block(gate=1e-5)
    x = torch.randn(1, 8, 16, requires_grad=True)
    video = torch.randn(1, 8, 16, requires_grad=True)
    audio, visual = block.joint_attn(x, video)
    probe = torch.randn(16)
    dx, dv, dg, dv_weight = torch.autograd.grad(
        (audio[0, 2] * probe).sum(),
        (x, video, block.av_visual_delta_gate, block.v_attn.to_v.weight), retain_graph=True,
    )
    assert dx[0, 7].abs().sum() > 0, "AA remains global"
    assert dv[0, 1:4].abs().sum() > 0, "nearby AV must remain live at g=1e-5"
    assert not torch.count_nonzero(dv[0, [0, 4, 5, 6, 7]]), "single-layer AV remains local"
    assert torch.isfinite(dg) and dg.abs() > 0, "visual gate remains trainable"
    assert torch.isfinite(dv_weight).all() and dv_weight.abs().sum() > 0, "video parameters must not be frozen/detached"
    feedback_dx, feedback_dv = torch.autograd.grad(
        (visual[0, 2] * probe).sum(), (x, video), allow_unused=True,
    )
    assert_no_gradient(feedback_dx)
    assert feedback_dv[0, 7].abs().sum() > 0, "VV remains global"
    with torch.no_grad():
        block.av_visual_delta_gate.zero_()
    audio, _ = block.joint_attn(x, video)
    zero_dv, zero_dg = torch.autograd.grad(
        (audio[0, 2] * probe).sum(), (video, block.av_visual_delta_gate), allow_unused=True,
    )
    assert_no_gradient(zero_dv)
    assert torch.isfinite(zero_dg) and zero_dg.abs() > 0
    print("[OK] zero VA gradient; global AA/VV; local live AV; trainable video parameters and gate endpoints")


def test_multilayer_video_isolation_and_cfg():
    model = make_model().eval()
    kwargs = inputs()

    def capture_video(call_kwargs):
        states = []
        handles = [block.register_forward_hook(lambda _module, _args, output: states.append(output[1].detach().clone()))
                   for block in model.transformer_blocks[:2]]
        try:
            with torch.inference_mode():
                model(**call_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return states

    original = capture_video(kwargs)
    changed = capture_video({**kwargs, "x": kwargs["x"] * 20 + 7,
                             "cond": kwargs["cond"] * 30 - 2,
                             "text": kwargs["text"].roll(1, dims=1)})
    changed_video = capture_video({**kwargs, "video": kwargs["video"] * 2 + 3})
    assert len(original) == 2
    for reference, actual, other_video in zip(original, changed, changed_video):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert not torch.allclose(other_video, reference), "video path must be active"

    null = dict(drop_audio_cond=True, drop_text=True, drop_video=True)
    for flags, branches in (({}, [{}, {"drop_video": True}, null]),
                            ({"drop_video": True}, [{"drop_video": True}, null]),
                            ({"drop_text": True}, [{"drop_text": True}, null])):
        with torch.inference_mode():
            packed = model(**kwargs, cfg_infer=True, **flags)[0]
            sequential = torch.cat([model(**kwargs, **branch)[0] for branch in branches])
        torch.testing.assert_close(packed, sequential, rtol=3e-5, atol=3e-6)
    with torch.inference_mode():
        dropped = model(**kwargs, drop_video=True)[0]
        altered = model(**{**kwargs, "video": kwargs["video"] * 100 + 15}, drop_video=True)[0]
    torch.testing.assert_close(dropped, altered, rtol=0, atol=0)
    cfm = CFM_VT(transformer=model, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    output, _ = cfm.sample(
        cond=kwargs["cond"][:, :3], text=kwargs["text"], duration=torch.tensor([12, 10]),
        video=kwargs["video"], lens=torch.tensor([3, 3]), speaker_embedding=kwargs["speaker_embedding"],
        steps=1, use_epss=False, cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :3], kwargs["cond"][:, :3], rtol=0, atol=0)
    print("[OK] multilayer video independence from audio/text at fixed flow time; packed CFG; null video; prefix sample")


def test_training_checkpoint_and_bfloat16():
    records = []
    for checkpoint in (False, True):
        transformer = make_model(checkpoint=checkpoint)
        cfm = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03,
            audio_drop_prob=0.0, cond_drop_prob=0.0, text_drop_prob=0.0, video_drop_prob=0.0,
        ).train()
        torch.manual_seed(107)
        kwargs = inputs()
        with capture_joint_masks(24) as captured, patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = cfm(
                inp=kwargs["x"], text=kwargs["text"], video=kwargs["video"],
                lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 4]),
                video_lens=torch.tensor([12, 10]), speaker_embedding=kwargs["speaker_embedding"],
            )
            assert torch.isfinite(loss) and math.isfinite(components["ctc_loss"])
            loss.backward()
        assert len(captured) >= 2
        for mask in captured:
            assert mask is not None
            assert mask[..., :12, :12].all(), "training AA stays global when padding mask=None"
            assert not mask[..., 12:, :12].any(), "VA block must survive training and recomputation"
            assert mask[..., 12:, 12:].all(), "training VV stays global"
            assert mask[..., 6, 12 + 6].all(), "reference prefix keeps common sequence coordinates"
            assert not mask[..., 6, 12 + 3].any()
        parameters = [block.av_visual_delta_gate for block in transformer.transformer_blocks[:2]]
        parameters.append(transformer.transformer_blocks[0].v_attn.to_v.weight)
        parameters.append(transformer.speaker_proj.weight)
        for parameter in parameters:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        records.append((loss.detach(), *(parameter.grad.detach().clone() for parameter in parameters)))
    for actual, expected in zip(records[1], records[0]):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-8)
    block = make_block(gate=1e-5)
    x = torch.randn(2, 8, 16, requires_grad=True)
    video = torch.randn(2, 8, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        audio, visual = block.joint_attn(x, video)
        loss = audio.float().square().sum()
    feedback_dx = torch.autograd.grad(visual.float().sum(), x, retain_graph=True, allow_unused=True)[0]
    assert_no_gradient(feedback_dx)
    loss.backward()
    assert block.av_visual_delta_gate.grad.dtype == torch.float32
    assert torch.isfinite(block.av_visual_delta_gate.grad) and block.av_visual_delta_gate.grad.abs() > 0
    assert torch.isfinite(video.grad).all() and video.grad.abs().sum() > 0
    print("[OK] CFM/CTC backward; checkpoint recomputation parity; masks active with training mask=None; bf16")


def test_state_optimizer_and_ema_roundtrip():
    torch.manual_seed(109)
    model = make_model().eval()
    off = make_model(blocked=False).eval()
    assert model.state_dict().keys() == off.state_dict().keys()
    off.load_state_dict(model.state_dict(), strict=True)
    assert all(block.block_video_audio_attention for block in model.transformer_blocks[:2])
    assert all(not block.block_video_audio_attention for block in off.transformer_blocks[:2])
    assert all(not hasattr(block, "block_video_audio_attention") for block in model.transformer_blocks[2:])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, include_online_model=False, update_after_step=0, update_every=1)
    kwargs = inputs()
    model(**kwargs)[0].square().mean().backward()
    optimizer.step()
    ema.update()
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = make_model().eval()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_ema = EMA(restored, include_online_model=False, update_after_step=0, update_every=1)
    restored_ema.load_state_dict(checkpoint["ema"], strict=True)
    assert all(block.block_video_audio_attention for block in restored_ema.ema_model.transformer_blocks[:2])
    assert all(not parameter.requires_grad for parameter in restored_ema.ema_model.parameters())
    for name, parameter in restored.named_parameters():
        source = dict(model.named_parameters())[name]
        if source in optimizer.state:
            for key, value in optimizer.state[source].items():
                torch.testing.assert_close(restored_optimizer.state[parameter][key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(**kwargs)[0], model(**kwargs)[0], rtol=0, atol=0)
        torch.testing.assert_close(restored_ema.ema_model(**kwargs)[0], ema.ema_model(**kwargs)[0], rtol=0, atol=0)
    print("[OK] no state keys added; strict step-2 state loading; model/optimizer/EMA roundtrip; flag in EMA topology")


def main():
    torch.set_num_threads(1)
    test_video_oracle_audio_parity_and_padding()
    test_forward_and_backward_information_paths()
    test_multilayer_video_isolation_and_cfg()
    test_training_checkpoint_and_bfloat16()
    test_state_optimizer_and_ema_roundtrip()
    print("All VA-block step-3 contracts passed.")


if __name__ == "__main__":
    main()
