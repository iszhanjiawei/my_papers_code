"""CPU regressions for progressive Audio-Query -> Video-Key joint bias.

Run with PYTHONPATH=src python scripts/test_joint_av_progressive_attention.py.
The attention oracle explicitly forms a full joint softmax and computes cosine
weights with scalar math; it does not call the production bias or SDPA helpers.
"""

from __future__ import annotations

import math

import torch
from test_audio_local_visual_attention import make_inputs
from test_progressive_visual_window import make_model as make_tail_model

from aligndit.model.backbone.dit_vt_mm import MMDiTBlock_VT


def make_model(*, enabled=True, checkpoint_activations=False, **overrides):
    model = make_tail_model(
        checkpoint_activations=checkpoint_activations,
        joint_av_local_attention=enabled,
        joint_av_window_schedule="flowley_progressive" if enabled else "fixed",
        **overrides,
    )
    # Exercise all streams/gates as in a trained model, rather than rely on
    # zero video/text residuals hiding a wrong joint-attention implementation.
    with torch.no_grad():
        for block in model.transformer_blocks[:model.n_mm_layers]:
            block.v_attn_norm.linear.weight.normal_(std=0.03)
            block.v_attn_norm.linear.bias.normal_(std=0.03)
            block.cross_attn_ada.weight.normal_(std=0.03)
            block.cross_attn_ada.bias.normal_(std=0.03)
        for block in model.transformer_blocks[model.n_text_layers:]:
            block.local_visual_attn.gate.fill_(0.15)
    return model


def scalar_window(window, generation_mask, window_video_mask):
    batch, audio_len = generation_mask.shape
    video_len = window_video_mask.shape[1]
    bias = torch.zeros(batch, audio_len, video_len)
    for b in range(batch):
        positions = torch.where(window_video_mask[b])[0].tolist()
        last_valid = positions[-1] if positions else 0
        for i in range(audio_len):
            if not generation_mask[b, i]:
                continue
            center = min(round(i / window.audio_video_ratio), last_valid)
            for j in positions:
                distance = abs(j - center) * window.window_reference_fps / window.video_frame_rate
                radius = window.window_radius_seconds * window.window_reference_fps
                if distance <= window.window_core_radius:
                    weight = 1.0
                else:
                    phase = min(1.0, max(0.0, (distance - window.window_core_radius) /
                                             (radius - window.window_core_radius)))
                    weight = window.window_fade_scale * (1 + math.cos(math.pi * phase)) / 2
                bias[b, i, j] = math.log(weight + 1e-6)
    return bias


def full_joint_oracle(block, x, video, *, generation_mask, window_video_mask,
                      mask=None, v_mask=None, apply_window=True):
    batch, n_audio, dim = x.shape
    n_video = video.shape[1]
    heads, head_dim = block.attn.heads, block.attn.inner_dim // block.attn.heads

    def qkv(attention, tokens):
        tensors = [projection(tokens).reshape(batch, -1, heads, head_dim).transpose(1, 2)
                   for projection in (attention.to_q, attention.to_k, attention.to_v)]
        if attention.q_norm is not None:
            tensors[0] = attention.q_norm(tensors[0])
        if attention.k_norm is not None:
            tensors[1] = attention.k_norm(tensors[1])
        return tensors

    audio_qkv, video_qkv = qkv(block.attn, x), qkv(block.v_attn, video)
    query, key, value = [torch.cat(pair, dim=2).double() for pair in zip(audio_qkv, video_qkv)]
    logits = query @ key.transpose(-1, -2) / math.sqrt(head_dim)
    expected_bias = torch.zeros(batch, 1, n_audio, n_audio + n_video)
    if apply_window:
        expected_bias[:, 0, :, n_audio:] = scalar_window(
            block.joint_av_window, generation_mask, window_video_mask,
        )
        logits[:, :, :n_audio, n_audio:] += expected_bias[:, :, :, n_audio:].double()
    if block.attn_mask_enabled and mask is not None:
        keys = torch.cat((mask, torch.ones(batch, n_video, dtype=torch.bool)
                          if v_mask is None else v_mask), dim=1)
        logits = logits.masked_fill(~keys[:, None, None], -math.inf)
        expected_bias = expected_bias.masked_fill(~keys[:, None, None], -math.inf)
    probabilities = logits.softmax(dim=-1)
    attended = (probabilities @ value).transpose(1, 2).reshape(batch, n_audio + n_video, -1).to(x.dtype)
    out_audio = block.attn.to_out[1](block.attn.to_out[0](attended[:, :n_audio]))
    out_video = block.v_attn.to_out[1](block.v_attn.to_out[0](attended[:, n_audio:]))
    if mask is not None:
        out_audio = out_audio.masked_fill(~mask[..., None], 0)
    if v_mask is not None:
        out_video = out_video.masked_fill(~v_mask[..., None], 0)
    assert out_audio.shape == (batch, n_audio, dim)
    return out_audio, out_video, expected_bias, probabilities


def block_pair(*, ratio=1, fade=0.45):
    kwargs = {
        "dim": 32, "heads": 4, "dim_head": 8, "ff_mult": 2, "dropout": 0.0,
        "qk_norm": "rms_norm", "text_dim": 16, "prompt_isolated_ca": False,
        "attn_mask_enabled": True,
    }
    torch.manual_seed(229)
    plain = MMDiTBlock_VT(**kwargs).eval()
    windowed = MMDiTBlock_VT(
        **kwargs,
        joint_av_window_kwargs={"audio_video_ratio": ratio, "audio_frame_rate": 40.0,
                                "window_radius_seconds": 0.5, "window_reference_fps": 8.0,
                                "window_core_radius": 0.0, "window_fade_scale": fade},
    ).eval()
    windowed.load_state_dict(plain.state_dict(), strict=True)
    return plain, windowed


def test_manual_joint_oracle_and_direction():
    for ratio, n_audio, n_video in ((1, 31, 31), (3, 31, 12)):
        for fade in (1.0, 0.45, 0.0):
            plain, block = block_pair(ratio=ratio, fade=fade)
            x, video = torch.randn(2, n_audio, 32), torch.randn(2, n_video, 32)
            mask = torch.arange(n_audio)[None] < torch.tensor([n_audio, n_audio - 5])[:, None]
            v_mask = torch.arange(n_video)[None] < torch.tensor([n_video, n_video - 3])[:, None]
            generation_mask = mask.clone()
            generation_mask[:, :6] = False
            window_video_mask = v_mask.clone()
            window_video_mask[:, :2] = False  # prompt prefix must retain its timeline
            window_video_mask[0, 4] = False  # arbitrary holes must not shift the clamp endpoint
            for use_padding in (False, True):
                kwargs = {"mask": mask if use_padding else None,
                          "v_mask": v_mask if use_padding else None,
                          "generation_mask": generation_mask,
                          "window_video_mask": window_video_mask}
                expected_a, expected_v, bias, probabilities = full_joint_oracle(block, x, video, **kwargs)
                actual_a, actual_v = block.joint_attn(x, video, **kwargs)
                torch.testing.assert_close(actual_a, expected_a, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(actual_v, expected_v, rtol=3e-5, atol=3e-6)
                actual_bias = block.joint_audio_attn_bias(
                    n_audio, n_video, dtype=x.dtype, **kwargs,
                )
                torch.testing.assert_close(actual_bias, bias, rtol=3e-5, atol=3e-6)
                plain_a, plain_v = plain.joint_attn(x, video, mask=kwargs["mask"], v_mask=kwargs["v_mask"])
                torch.testing.assert_close(actual_v, plain_v, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(actual_a[~generation_mask], plain_a[~generation_mask],
                                           rtol=3e-5, atol=3e-6)
                assert (actual_a[generation_mask] - plain_a[generation_mask]).abs().max() > 1e-4
                _, _, _, global_probs = full_joint_oracle(block, x, video, apply_window=False, **kwargs)
                # The shared denominator also changes Audio->Audio probabilities;
                # this is still one joint softmax, not a separate visual residual.
                assert (probabilities[:, :, :n_audio, :n_audio] -
                        global_probs[:, :, :n_audio, :n_audio]).abs().max() > 1e-4
            # Verify the entire block, including AdaLN, text CA and tokenwise FFN.
            t, text = torch.randn(2, 32), torch.randn(2, 5, 16)
            kwargs = {
                "mask": mask, "v_mask": v_mask, "text": text,
                "text_mask": torch.ones(2, 5, dtype=torch.bool), "generation_mask": generation_mask,
            }
            with torch.no_grad():
                global_a, global_v = plain(x, video, t, **kwargs)
                local_a, local_v = block(x, video, t, window_video_mask=window_video_mask, **kwargs)
            torch.testing.assert_close(local_v, global_v, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(local_a[~generation_mask], global_a[~generation_mask],
                                       rtol=3e-5, atol=3e-6)
    print("[OK] independent full-softmax oracle, only AQ->VK bias, shared denominator, prompt/video parity")


def test_empty_condition_and_masks():
    plain, block = block_pair()
    x, video = torch.randn(2, 12, 32), torch.randn(2, 12, 32)
    valid = torch.ones(2, 12, dtype=torch.bool)
    for generation, real_video in ((valid, torch.zeros_like(valid)), (torch.zeros_like(valid), valid)):
        local = block.joint_attn(x, video, generation_mask=generation, window_video_mask=real_video)
        global_output = plain.joint_attn(x, video)
        for actual, expected in zip(local, global_output):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    # One dropped example must not borrow another example's temporal mask.
    mixed = valid.clone()
    mixed[1] = False
    actual_a, actual_v = block.joint_attn(x, video, generation_mask=valid, window_video_mask=mixed)
    expected_a, expected_v = plain.joint_attn(x, video)
    torch.testing.assert_close(actual_a[1], expected_a[1], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_v, expected_v, rtol=3e-5, atol=3e-6)
    assert (actual_a[0] - expected_a[0]).abs().max() > 1e-4
    for bad in ({"generation_mask": None, "window_video_mask": valid},
                {"generation_mask": valid, "window_video_mask": None},
                {"generation_mask": valid.float(), "window_video_mask": valid},
                {"generation_mask": valid[:, :-1], "window_video_mask": valid}):
        try:
            block.joint_attn(x, video, **bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"Invalid joint window masks accepted: {bad}")
    print("[OK] empty/drop-video and prompt-only bias is zero; batched masks and input validation")


def test_schedules_and_state_compatibility():
    baseline, joint = make_model(enabled=False), make_model()
    assert list(baseline.state_dict()) == list(joint.state_dict())
    assert [name for name, _ in baseline.named_parameters()] == [name for name, _ in joint.named_parameters()]
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)
    joint.load_state_dict(baseline.state_dict(), strict=True)
    assert [block.joint_av_window.window_fade_scale for block in joint.transformer_blocks[:12]] == [
        1 - layer / 11 for layer in range(12)
    ]
    assert [block.local_visual_attn.window_fade_scale for block in joint.transformer_blocks[12:]] == [
        1 - layer / 5 for layer in range(6)
    ]
    assert all(block.joint_av_window is None for block in baseline.transformer_blocks[:12])
    assert all(not block.joint_av_window.state_dict() for block in joint.transformer_blocks[:12])
    assert all(block.joint_av_window.window_radius_seconds == 0.5 for block in joint.transformer_blocks[:12])
    single = make_model(n_mm_layers=1)
    assert single.transformer_blocks[0].joint_av_window.window_fade_scale == 1.0
    scaled = make_model(local_visual_window_fade_scale=0.4)
    assert [block.joint_av_window.window_fade_scale for block in scaled.transformer_blocks[:12]] == [
        0.4 * (1 - layer / 11) for layer in range(12)
    ]
    # No visual conditioning yields the old complete model even with active
    # pretrained audio/video/text gates, and cached branch state is irrelevant.
    baseline.eval()
    joint.eval()
    inputs = make_inputs()
    with torch.no_grad():
        expected = baseline(**inputs, drop_video=True)[0]
        actual = joint(**inputs, drop_video=True)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-6)
    print("[OK] MM12/audio6 independent beta schedules, identical parameter/state ordering, strict checkpoint load")


def test_full_model_cfg_and_cache():
    for ratio in (1, 3):
        model = make_model(audio_video_ratio=ratio, video_rope_scaled=True).eval()
        inputs = make_inputs()
        if ratio != 1:
            inputs["video"] = inputs["video"][:, ::ratio].contiguous()
            inputs["video_mask"] = inputs["video_mask"][:, ::ratio].contiguous()
            inputs["complementary_mask"] = inputs["complementary_mask"][:, ::ratio].contiguous()
        null = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
        cases = (({}, ({}, {"drop_video": True}, null)),
                 ({"drop_video": True}, ({"drop_video": True}, null)),
                 ({"drop_text": True}, ({"drop_text": True}, null)),
                 ({"drop_text": True, "drop_video": True},
                  ({"drop_text": True, "drop_video": True}, null)))
        with torch.no_grad():
            for cache in (False, True):
                for flags, independent_flags in cases:
                    model.clear_cache()
                    for step in (0, 1):
                        sample = {**inputs, "cache": cache, "time": inputs["time"] + step * 0.03,
                                  "x": inputs["x"] + step * 0.02}
                        actual = model(**sample, cfg_infer=True, **flags)[0]
                        expected = torch.cat([model(**sample, **branch)[0] for branch in independent_flags])
                        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=5e-6)
            model.clear_cache()
            changed = {**inputs, "video": torch.randn_like(inputs["video"]) * 100}
            torch.testing.assert_close(model(**inputs, drop_video=True)[0], model(**changed, drop_video=True)[0],
                                       rtol=0, atol=0)
    print("[OK] B=2 two/three-branch CFG, cached/uncached flows, all modality drops, ratios 1 and 3")


def test_checkpointed_backward():
    plain, checkpointed = make_model(), make_model(checkpoint_activations=True)
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    inputs = make_inputs()
    target = torch.randn_like(inputs["x"])
    results = []
    for model in (plain, checkpointed):
        model.train()
        output, ctc = model(**inputs)
        loss = (output - target).square().mean() + 0.03 * sum(
            tap["z_tilde"].square().mean() for tap in ctc.values()
        )
        assert torch.isfinite(loss) and set(ctc) == {6, 12}
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                gradients[name] = parameter.grad.detach().clone()
        for layer in range(12):
            for suffix in ("attn.to_q.weight", "v_attn.to_k.weight", "v_attn.to_v.weight"):
                key = f"transformer_blocks.{layer}.{suffix}"
                assert key in gradients and torch.count_nonzero(gradients[key]), key
        for layer in range(12, 18):
            key = f"transformer_blocks.{layer}.local_visual_attn.gate"
            assert key in gradients and torch.count_nonzero(gradients[key]), key
        results.append((output.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert set(results[0][1]) == set(results[1][1])
    for key, grad in results[0][1].items():
        torch.testing.assert_close(grad, results[1][1][key], rtol=2e-5, atol=2e-8)
    print("[OK] full-model CTC backward, all 12 joint Q/K/V gradients, tail6 gates, checkpoint parity")


def main():
    torch.set_num_threads(1)
    test_manual_joint_oracle_and_direction()
    test_empty_condition_and_masks()
    test_schedules_and_state_compatibility()
    test_full_model_cfg_and_cache()
    test_checkpointed_backward()
    print("All progressive joint Audio-Query -> Video-Key attention tests passed.")


if __name__ == "__main__":
    main()
