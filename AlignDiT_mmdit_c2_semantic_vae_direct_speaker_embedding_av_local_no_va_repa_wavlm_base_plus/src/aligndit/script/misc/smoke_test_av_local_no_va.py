"""CPU contracts for local AV + blocked VA (without a visual delta gate) on the 40-Hz grid.

Run from this experiment root with PYTHONPATH=src. The test groups require no
datasets, checkpoints, VAE decoder, or GPU. Direct attention tests isolate one layer:
other global paths can legitimately carry distant video information across
multiple layers, so whole-model locality is deliberately not asserted.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, MMDiTBlock_VT
from aligndit.model.cfm_vt import CFM_VT


ARCH = dict(
    dim=32, depth=4, heads=4, dim_head=8, dropout=0.0, ff_mult=2,
    mel_dim=64, text_num_embeds=16, text_dim=16, text_mask_padding=False,
    qk_norm="rms_norm", conv_layers=1, pe_attn_head=1,
    attn_mask_enabled=True, checkpoint_activations=False, use_conformer=False,
    layer_indices_ctc=[1, 2], ctc_sampling_ratios=[1, 1], n_mm_layers=2,
    n_text_layers=2, prompt_isolated_ca=False, audio_video_ratio=1,
    video_dim=16, video_rope_scaled=False, speaker_dim=192,
    speaker_condition_start_layer=2,
)


def make_block(radius, *, attn_mask_enabled=True):
    return MMDiTBlock_VT(
        dim=16, heads=2, dim_head=8, dropout=0.0, ff_mult=2,
        text_dim=16, prompt_isolated_ca=False,
        attn_mask_enabled=attn_mask_enabled, av_local_window_radius=radius, block_video_audio_attention=True,
    ).eval()


def make_model(radius=2, *, checkpoint_activations=False):
    torch.manual_seed(41)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint_activations},
        av_local_window_radius=radius, block_video_audio_attention=True,
    )
    # Scratch AdaLN/output zeros would hide broken gradient paths. Simulate
    # the nonzero modulation and output weights of the audio warm start.
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        model.proj_out.weight.normal_(std=0.03)
        model.norm_out.linear.weight.normal_(std=0.03)
    return model


def inputs():
    valid = torch.arange(12)[None] < torch.tensor([12, 10])[:, None]
    generated = valid.clone()
    generated[:, :3] = False
    return dict(
        x=torch.randn(2, 12, 64), cond=torch.randn(2, 12, 64),
        text=torch.randint(0, 16, (2, 4)), video=torch.randn(2, 12, 16),
        time=torch.tensor([0.2, 0.8]), mask=valid,
        text_mask=torch.ones(2, 4, dtype=torch.bool), video_mask=valid.clone(),
        generation_mask=generated, complementary_mask=valid & ~generated,
        speaker_embedding=torch.randn(2, 192), cache=False,
    )


@contextmanager
def capture_joint_masks(total_tokens):
    """Capture the masks actually passed to SDPA, keeping real computation."""
    original = F.scaled_dot_product_attention
    captured = []

    def wrapped(query, key, value, *args, **kwargs):
        if key.shape[-2] == total_tokens:
            mask = kwargs.get("attn_mask", args[0] if args else None)
            expanded = None if mask is None else mask.expand(
                query.shape[0], query.shape[1], query.shape[-2], total_tokens
            ).detach().clone()
            captured.append((query.shape[-2], expanded))
        return original(query, key, value, *args, **kwargs)

    with patch("aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention", side_effect=wrapped):
        yield captured


def expected_mask(length, radius, batch=1, *, audio_valid=None, video_valid=None):
    expected = torch.ones(batch, 1, 2 * length, 2 * length, dtype=torch.bool)
    for query in range(length):
        for video_key in range(length):
            expected[:, :, query, length + video_key] = abs(query - video_key) <= radius
    expected[:, :, length:, :length] = False
    if audio_valid is not None:
        valid = torch.cat([audio_valid, video_valid], dim=1)
        expected &= valid[:, None, None, :]
    return expected


def expected_rectangular_mask(audio_length, video_length, radius, batch=1):
    total = audio_length + video_length
    expected = torch.ones(batch, 1, total, total, dtype=torch.bool)
    for audio_query in range(audio_length):
        for video_key in range(video_length):
            expected[:, :, audio_query, audio_length + video_key] = (
                abs(audio_query - video_key) <= radius
            )
    expected[:, :, audio_length:, :audio_length] = False
    return expected


def test_four_quadrants_and_padding():
    torch.manual_seed(11)
    n = 8
    x, v = torch.randn(2, n, 16), torch.randn(2, n, 16)
    audio_valid = torch.arange(n)[None] < torch.tensor([8, 6])[:, None]
    video_valid = torch.arange(n)[None] < torch.tensor([7, 5])[:, None]
    for enabled in (True, False):
        block = make_block(2, attn_mask_enabled=enabled)
        for use_padding in (False, True):
            kwargs = dict(mask=audio_valid, v_mask=video_valid) if use_padding else {}
            with capture_joint_masks(2 * n) as captured:
                out_a, out_v = block.joint_attn(x, v, **kwargs)
            expected = expected_mask(n, 2, batch=2, **(
                dict(audio_valid=audio_valid, video_valid=video_valid)
                if use_padding and enabled else {}
            ))
            assert len(captured) == 1 and captured[0][1] is not None
            query_count, actual = captured[0]
            assert torch.equal(actual, expected[:, :, :query_count].expand_as(actual))
            assert torch.isfinite(out_a).all() and torch.isfinite(out_v).all()
            if use_padding:
                assert not torch.count_nonzero(out_a[~audio_valid])
                assert not torch.count_nonzero(out_v[~video_valid])
    print("[OK] AV local and VA blocked; AA/VV global; padding policy preserved")


def test_direct_information_and_gradient_paths():
    torch.manual_seed(19)
    local, global_block = make_block(1), make_block(None)
    global_block.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 8, 16, requires_grad=True)
    v = torch.randn(1, 8, 16, requires_grad=True)
    out_a, out_v = local.joint_attn(x, v)
    global_a, global_v = global_block.joint_attn(x, v)
    torch.testing.assert_close(out_v, global_v, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(out_a, global_a)
    grad_x, grad_v = torch.autograd.grad(out_a[0, 2].square().sum(), (x, v), retain_graph=True)
    assert torch.count_nonzero(grad_x[0, 7]), "AA must retain distant audio access"
    assert torch.count_nonzero(grad_v[0, 1:4]), "local AV must remain trainable"
    assert not torch.count_nonzero(grad_v[0, [0, 4, 5, 6, 7]]), "distant direct AV path must be blocked"
    grad_x_v, grad_v_v = torch.autograd.grad(out_v[0, 2].square().sum(), (x, v), allow_unused=True)
    assert grad_x_v is None or not torch.count_nonzero(grad_x_v), "VA must be blocked"
    assert torch.count_nonzero(grad_v_v[0, 7]), "VV must retain distant video access"
    changed_v = v.detach().clone()
    changed_v[:, 7] += 20.0
    changed_a, _ = local.joint_attn(x.detach(), changed_v)
    torch.testing.assert_close(changed_a[:, 2], out_a[:, 2], rtol=0, atol=0)
    print("[OK] distant AV values/gradients blocked; nearby AV and distant AA/VV remain live; VA gradients are zero")


def test_checkpoint_compatibility_and_wide_window():
    torch.manual_seed(23)
    legacy = make_model(None).eval()
    wide = make_model(100).eval()
    local = make_model(2).eval()
    assert legacy.state_dict().keys() == local.state_dict().keys() == wide.state_dict().keys()
    wide.load_state_dict(legacy.state_dict(), strict=True)
    local.load_state_dict(legacy.state_dict(), strict=True)
    kwargs = inputs()
    with torch.inference_mode():
        reference, reference_ctc = legacy(**kwargs)
        actual, actual_ctc = wide(**kwargs)
    assert torch.count_nonzero(reference), "comparison must not use zero scratch outputs"
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    for layer in reference_ctc:
        torch.testing.assert_close(actual_ctc[layer]["z_tilde"], reference_ctc[layer]["z_tilde"])
    assert all(block.av_local_window_radius == 2 for block in local.transformer_blocks[:2])
    print("[OK] unchanged checkpoint keys; wide local window recovers global outputs and CTC")


def test_training_mask_none_and_checkpoint_backward():
    for checkpoint in (False, True):
        transformer = make_model(checkpoint_activations=checkpoint)
        model = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1,
            ctc_lambda=0.03, audio_drop_prob=0.0, cond_drop_prob=0.0,
            text_drop_prob=0.0, video_drop_prob=0.0,
        ).train()
        kwargs = inputs()
        with capture_joint_masks(24) as captured:
            with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                loss, components, _, _ = model(
                    inp=kwargs["x"], text=kwargs["text"], video=kwargs["video"],
                    lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 4]),
                    video_lens=torch.tensor([12, 10]), speaker_embedding=kwargs["speaker_embedding"],
                )
            assert torch.isfinite(loss) and "ctc_loss" in components
            loss.backward()
        assert len(captured) >= 2
        for query_count, mask in captured:
            assert mask is not None
            # Baseline intentionally drops padding masks during training.
            # The structural AV window must nevertheless stay active.
            assert torch.equal(mask, expected_mask(12, 2, batch=2)[:, :, :query_count].expand_as(mask))
        for name, parameter in {
            "audio query": transformer.transformer_blocks[0].attn.to_q.weight,
            "video value": transformer.transformer_blocks[0].v_attn.to_v.weight,
            "speaker": transformer.speaker_proj.weight,
        }.items():
            gradient = parameter.grad
            assert gradient is not None and torch.isfinite(gradient).all(), name
            assert torch.count_nonzero(gradient), f"{name} must receive a nonzero gradient"
    print("[OK] CFM/CTC backward and recomputation preserve locality with training mask=None")


def test_cfg_prefix_and_null_video():
    model = make_model().eval()
    kwargs = inputs()
    for flags, branch_count in (({}, 3), ({"drop_video": True}, 2), ({"drop_text": True}, 2)):
        with torch.inference_mode(), capture_joint_masks(24) as captured:
            actual, _ = model(**kwargs, cfg_infer=True, **flags)
        assert actual.shape == (2 * branch_count, 12, 64) and torch.isfinite(actual).all()
        valid = kwargs["mask"].repeat(branch_count, 1)
        expected = expected_mask(12, 2, batch=2 * branch_count, audio_valid=valid, video_valid=valid)
        assert len(captured) == 2
        for query_count, mask in captured:
            assert torch.equal(mask, expected[:, :, :query_count].expand_as(mask))
            # Three prefix positions are retained in both grids; query 6
            # reads video 4..8, not target-relative video 1..5.
            assert mask[0, 0, 6, 12 + 6]
            assert not mask[0, 0, 6, 12 + 3]
    with torch.inference_mode():
        dropped_a, _ = model(**kwargs, drop_video=True)
        dropped_b, _ = model(**{**kwargs, "video": kwargs["video"] * 100 + 17}, drop_video=True)
    torch.testing.assert_close(dropped_a, dropped_b, rtol=0, atol=0)

    cfm = CFM_VT(transformer=model, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    for batch in (1, 2):
        with capture_joint_masks(24) as captured:
            output, _ = cfm.sample(
                cond=kwargs["cond"][:batch, :3], text=kwargs["text"][:batch],
                duration=torch.tensor([12, 10])[:batch], video=kwargs["video"][:batch],
                lens=torch.full((batch,), 3), speaker_embedding=kwargs["speaker_embedding"][:batch],
                steps=1, use_epss=False, cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
            )
        assert output.shape == (batch, 12, 64) and torch.isfinite(output).all()
        torch.testing.assert_close(output[:, :3], kwargs["cond"][:batch, :3], rtol=0, atol=0)
        assert captured and all(mask is not None for _, mask in captured)
    print("[OK] 2/3-branch CFG, prefix coordinates, dropped video and B=1/2 sampling")


def test_text_clamped_audio_can_exceed_video():
    """Setting 1 may extend audio duration when token count exceeds 2x frames."""

    model = make_model().eval()
    kwargs = inputs()
    audio_length, video_length = 15, kwargs["video"].shape[1]
    kwargs["x"] = F.pad(kwargs["x"][:1], (0, 0, 0, audio_length - 12))
    kwargs["cond"] = F.pad(kwargs["cond"][:1], (0, 0, 0, audio_length - 12))
    kwargs["text"] = kwargs["text"][:1]
    kwargs["video"] = kwargs["video"][:1]
    kwargs["time"] = kwargs["time"][:1]
    kwargs["mask"] = None
    kwargs["text_mask"] = kwargs["text_mask"][:1]
    kwargs["video_mask"] = None
    kwargs["complementary_mask"] = kwargs["complementary_mask"][:1]
    kwargs["generation_mask"] = torch.arange(audio_length)[None] >= 3
    kwargs["speaker_embedding"] = kwargs["speaker_embedding"][:1]

    with torch.inference_mode(), capture_joint_masks(audio_length + video_length) as captured:
        actual, _ = model(**kwargs, cfg_infer=True)
    assert actual.shape == (3, audio_length, 64) and torch.isfinite(actual).all()
    expected = expected_rectangular_mask(audio_length, video_length, 2, batch=3)
    assert len(captured) == 2
    for query_count, mask in captured:
        assert torch.equal(mask, expected[:, :, :query_count].expand_as(mask))
    cfm = CFM_VT(transformer=model, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
    original_inputs = inputs()
    for batch in (1, 2):
        observed_shapes = []

        def record_shape(_module, _args, forward_kwargs):
            observed_shapes.append((forward_kwargs["x"].shape[1], forward_kwargs["video"].shape[1]))

        handle = model.register_forward_pre_hook(record_shape, with_kwargs=True)
        try:
            with capture_joint_masks(audio_length + video_length) as captured:
                output, _ = cfm.sample(
                    cond=original_inputs["cond"][:batch, :3],
                    text=torch.randint(0, 16, (batch, audio_length - 1)),
                    duration=torch.tensor([12, 10])[:batch], video=original_inputs["video"][:batch],
                    lens=torch.full((batch,), 3),
                    speaker_embedding=original_inputs["speaker_embedding"][:batch],
                    steps=1, use_epss=False, cfg_strength=1.0, cfg_strength_v=1.0, seed=0,
                )
        finally:
            handle.remove()
        assert observed_shapes and set(observed_shapes) == {(audio_length, video_length)}
        assert output.shape == (batch, audio_length, 64) and torch.isfinite(output).all()
        torch.testing.assert_close(output[:, :3], original_inputs["cond"][:batch, :3], atol=0, rtol=0)
        assert captured and all(mask is not None for _, mask in captured)
    print("[OK] text-clamped backbone and B=1/2 CFM sampling preserve rectangular AV without padding video")



def test_explicit_full_mask_reference_and_gradients():
    """A split SDPA implementation must equal one shared four-quadrant SDPA."""
    torch.manual_seed(73)
    for n_a, n_v, padding in ((8, 8, False), (11, 8, False), (11, 8, True)):
        block = make_block(2)
        x = torch.randn(2, n_a, 16, requires_grad=True)
        v = torch.randn(2, n_v, 16, requires_grad=True)
        audio_mask = torch.arange(n_a)[None] < torch.tensor([n_a, n_a - 2])[:, None]
        video_mask = torch.arange(n_v)[None] < torch.tensor([n_v, n_v - 3])[:, None]
        actual = block.joint_attn(x, v, **(dict(mask=audio_mask, v_mask=video_mask) if padding else {}))
        qa, ka, va = block._qkv(block.attn, x)
        qv, kv, vv = block._qkv(block.v_attn, v)
        allowed = expected_rectangular_mask(n_a, n_v, 2, batch=2)
        if padding:
            allowed &= torch.cat((audio_mask, video_mask), dim=1)[:, None, None]
        output = F.scaled_dot_product_attention(
            torch.cat((qa, qv), dim=2), torch.cat((ka, kv), dim=2),
            torch.cat((va, vv), dim=2), attn_mask=allowed,
        ).transpose(1, 2).reshape(2, n_a + n_v, 16)
        expected_a = block.attn.to_out[1](block.attn.to_out[0](output[:, :n_a]))
        expected_v = block.v_attn.to_out[1](block.v_attn.to_out[0](output[:, n_a:]))
        if padding:
            expected_a = expected_a.masked_fill(~audio_mask[..., None], 0)
            expected_v = expected_v.masked_fill(~video_mask[..., None], 0)
        for computed, reference in zip(actual, (expected_a, expected_v)):
            torch.testing.assert_close(computed, reference, atol=2e-6, rtol=2e-5)
        parameters = (x, v, *block.parameters())
        actual_grads = torch.autograd.grad(sum(t.square().sum() for t in actual), parameters, retain_graph=True, allow_unused=True)
        expected_grads = torch.autograd.grad(expected_a.square().sum() + expected_v.square().sum(), parameters, allow_unused=True)
        for computed, reference in zip(actual_grads, expected_grads):
            assert (computed is None) == (reference is None)
            if computed is not None:
                torch.testing.assert_close(computed, reference, atol=4e-6, rtol=3e-5)
    assert not any("av_visual" in key or "temporal_band" in key for key in block.state_dict())
    print("[OK] split attention matches explicit full-mask reference outputs and gradients")


def test_video_hidden_states_do_not_read_audio():
    torch.manual_seed(79)
    block = make_block(2)
    # Nonzero modulations make this a real updated-video test, not an identity.
    with torch.no_grad():
        block.v_attn_norm.linear.weight.normal_(std=0.2)
        block.v_attn_norm.linear.bias.normal_(std=0.2)
    x, v, t, text = torch.randn(1, 8, 16), torch.randn(1, 8, 16), torch.randn(1, 16), torch.randn(1, 4, 16)
    _, first = block(x, v, t, text=text)
    _, second = block(x * 10 + 9, v, t, text=text)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, v), "test must exercise nontrivial video updates"
    print("[OK] updated video states are independent of audio at fixed video/time")


def main():
    torch.set_num_threads(1)
    test_explicit_full_mask_reference_and_gradients()
    test_video_hidden_states_do_not_read_audio()
    test_four_quadrants_and_padding()
    test_direct_information_and_gradient_paths()
    test_checkpoint_compatibility_and_wide_window()
    test_training_mask_none_and_checkpoint_backward()
    test_cfg_prefix_and_null_video()
    test_text_clamped_audio_can_exceed_video()
    print("All AV-local / no-VA contracts passed.")


if __name__ == "__main__":
    main()
