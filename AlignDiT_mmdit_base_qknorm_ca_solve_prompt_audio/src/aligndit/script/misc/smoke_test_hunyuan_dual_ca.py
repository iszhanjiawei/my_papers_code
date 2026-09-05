"""CPU-first regression tests for the Hunyuan-style dual-query text CA.

Run from this experiment's root with ``PYTHONPATH=src python -u
src/aligndit/script/misc/smoke_test_hunyuan_dual_ca.py``. Pass ``--device
cuda:0`` for an optional GPU run. No datasets, checkpoints, or outputs are
created. The original smoke_test_mmdit.py remains the legacy regression suite.

The CA oracle uses explicit RMS normalization, complex-number RoPE, and
softmax/matmul instead of the implementation's real-valued RoPE and SDPA.
It follows Hunyuan TwoStreamCABlock's *CA* ordinal positions independently
for audio/video/text, not the AV joint attention's frame-rate-scaled RoPE.
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aligndit.model.backbone.dit_vt_mm import (
    DiT_VT_MMDiT,
    HunyuanDualTextCrossAttention,
    MMDiTBlock_VT,
)
from f5_tts.model.modules import DiTBlock


DIM, HEADS, HEAD_DIM, TEXT_DIM = 64, 4, 16, 32


def assert_grad(parameter, label, *, nonzero=True):
    grad = parameter.grad
    assert grad is not None, f"{label}: missing gradient"
    assert torch.isfinite(grad).all(), f"{label}: non-finite gradient"
    if nonzero:
        assert grad.abs().max() > 0, f"{label}: zero gradient"


def reference_rope(x):
    """Hunyuan theta=10000, ordinal positions, adjacent real/imag pairs."""
    sequence_length, head_dim = x.shape[-2:]
    positions = torch.arange(sequence_length, device=x.device, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    phase = torch.outer(positions, inv_freq)
    rotation = torch.polar(torch.ones_like(phase), phase)
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], head_dim // 2, 2))
    return torch.view_as_real(pairs * rotation).flatten(-2).to(x.dtype)


def heads(x):
    return x.reshape(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)


def reference_norm(x, norm):
    assert isinstance(norm, nn.RMSNorm)
    assert norm.normalized_shape == (HEAD_DIM,)
    assert norm.eps == 1e-6
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)).to(x.dtype) * norm.weight


def reference_ca(ca, audio, video, text, text_mask=None, use_rope=True):
    qa = reference_norm(heads(ca.audio_cross_q(audio)), ca.audio_cross_q_norm)
    qv = reference_norm(heads(ca.v_cond_cross_q(video)), ca.v_cond_cross_q_norm)
    kt, vt = ca.text_cross_kv(text).chunk(2, -1)
    kt = reference_norm(heads(kt), ca.text_cross_k_norm)
    vt = heads(vt)
    if use_rope:
        qa, qv, kt = map(reference_rope, (qa, qv, kt))

    def attend(query, projection):
        logits = query @ kt.transpose(-2, -1) / math.sqrt(HEAD_DIM)
        if text_mask is not None:
            logits = logits.masked_fill(~text_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        out = (weights @ vt).transpose(1, 2).reshape(audio.shape[0], query.shape[2], DIM)
        return projection(out)

    return attend(qa, ca.audio_cross_proj), attend(qv, ca.v_cond_cross_proj)


def test_ca_reference_and_text_mask(device):
    ca = HunyuanDualTextCrossAttention(DIM, HEADS, TEXT_DIM).to(device).eval()
    audio = torch.randn(2, 12, DIM, device=device)
    video = torch.randn(2, 3, DIM, device=device)
    text = torch.randn(2, 7, TEXT_DIM, device=device)
    text_mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3], device=device)

    assert ca.audio_cross_q is not ca.v_cond_cross_q
    assert ca.text_cross_kv.out_features == 2 * DIM
    assert not any("in_proj" in name for name, _ in ca.named_parameters())
    for mask in (None, text_mask):
        actual = ca(audio, video, text, text_mask=mask)
        expected = reference_ca(ca, audio, video, text, text_mask=mask)
        for stream, left, right in zip(("audio", "video"), actual, expected):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5, msg=f"{stream} CA reference")

    actual = ca(audio, video, text, text_mask=text_mask)
    perturbed = text.clone()
    perturbed[~text_mask] = 100 * torch.randn_like(perturbed[~text_mask])
    padded_changed = ca(audio, video, perturbed, text_mask=text_mask)
    for left, right in zip(actual, padded_changed):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
    perturbed = text.clone()
    perturbed[:, 0] += torch.randn_like(perturbed[:, 0])
    valid_changed = ca(audio, video, perturbed, text_mask=text_mask)
    for left, right in zip(actual, valid_changed):
        assert not torch.allclose(left, right), "Both streams must read valid text tokens"

    # An entirely absent condition must neither produce NaNs nor leak the
    # output projection bias into either stream.
    empty_mask = text_mask.clone()
    empty_mask[1] = False
    for out in ca(audio, video, text, text_mask=empty_mask):
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[1]) == 0

    changed_query = copy.deepcopy(ca)
    with torch.no_grad():
        changed_query.audio_cross_q.weight.add_(0.1 * torch.randn_like(ca.audio_cross_q.weight))
    changed = changed_query(audio, video, text, text_mask=text_mask)
    assert not torch.allclose(actual[0], changed[0]), "Audio query projection must affect audio CA"
    torch.testing.assert_close(actual[1], changed[1])

    # Position zero is unchanged, and *every* head rotates nonzero positions.
    q = torch.randn(2, HEADS, 9, HEAD_DIM, device=device)
    rotated = ca.apply_rope(q)
    torch.testing.assert_close(rotated, reference_rope(q), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rotated[:, :, 0], q[:, :, 0])
    for head_index in range(HEADS):
        assert not torch.allclose(rotated[:, head_index, 1:], q[:, head_index, 1:])
    no_rope = reference_ca(ca, audio, video, text, text_mask, use_rope=False)
    assert all(not torch.allclose(left, right) for left, right in zip(actual, no_rope))

    loss = sum((out * torch.randn_like(out)).mean() for out in actual)
    loss.backward()
    for name, parameter in ca.named_parameters():
        assert_grad(parameter, f"standalone CA {name}")
    print("[OK] dual CA matches independent Hunyuan formula; all-head RoPE, separate Q/shared KV, text mask")


def tiny_model(device, n_mm_layers=6, *, mode="hunyuan_dual", checkpoint=False):
    return DiT_VT_MMDiT(
        dim=DIM,
        depth=18,
        heads=HEADS,
        dim_head=HEAD_DIM,
        ff_mult=2,
        dropout=0.0,
        mel_dim=8,
        text_num_embeds=32,
        text_dim=TEXT_DIM,
        text_mask_padding=False,
        qk_norm="rms_norm",
        conv_layers=0,
        pe_attn_head=None if mode == "hunyuan_dual" else 1,
        attn_mask_enabled=True,
        checkpoint_activations=checkpoint,
        use_conformer=False,
        layer_indices_ctc=[5, 11] if n_mm_layers == 6 else [6, 12],
        n_mm_layers=n_mm_layers,
        n_text_layers=n_mm_layers,
        prompt_isolated_ca=False,
        audio_video_ratio=4,
        video_dim=16,
        video_rope_scaled=True,
        text_attention_mode=mode,
    ).to(device)


def model_inputs(device):
    audio_mask = torch.ones(2, 16, dtype=torch.bool, device=device)
    audio_mask[1, 12:] = False
    video_mask = audio_mask[:, ::4]
    text_mask = torch.ones(2, 7, dtype=torch.bool, device=device)
    text_mask[1, 5:] = False
    text = torch.randint(1, 32, (2, 7), device=device)
    text[~text_mask] = -1
    return {
        "x": torch.randn(2, 16, 8, device=device),
        "cond": torch.randn(2, 16, 8, device=device),
        "text": text,
        "video": torch.randn(2, 4, 16, device=device),
        "time": torch.rand(2, device=device),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": video_mask,
        "complementary_mask": torch.zeros_like(video_mask),
        "generation_mask": audio_mask.clone(),
        "cache": False,
    }


def open_gates(model):
    # Probe trained/nonzero gates, not an all-zero AdaLN/proj_out initialization.
    with torch.no_grad():
        nn.init.normal_(model.proj_out.weight, std=0.05)
        for block in model.transformer_blocks:
            for name in ("attn_norm", "v_attn_norm"):
                if hasattr(block, name):
                    bias = getattr(block, name).linear.bias
                    bias[2 * DIM : 3 * DIM].fill_(0.25)
                    bias[5 * DIM : 6 * DIM].fill_(0.25)
            for name in ("cross_attn_ada", "v_cross_attn_ada"):
                if hasattr(block, name):
                    getattr(block, name).bias[2 * DIM :].fill_(0.25)


def test_gate_liveness_and_all_head_joint_rope(device):
    model = tiny_model(device)
    block = model.transformer_blocks[0]
    audio = torch.randn(2, 12, DIM, device=device)
    video = torch.randn(2, 3, DIM, device=device)
    text = torch.randn(2, 7, TEXT_DIM, device=device)
    time = torch.randn(2, DIM, device=device)

    def forward_probe():
        return block(audio, video, time, text=text, text_mask=torch.ones(2, 7, dtype=torch.bool, device=device))

    for name in ("cross_attn_ada", "v_cross_attn_ada"):
        module = getattr(block, name)
        assert torch.count_nonzero(module.weight) == 0
        assert torch.count_nonzero(module.bias) == 0
    for name in ("audio_cross_proj", "v_cond_cross_proj"):
        assert torch.count_nonzero(getattr(block.cross_attn, name).weight) > 0

    outputs = forward_probe()
    sum((out * torch.randn_like(out)).mean() for out in outputs).backward()
    for name in ("cross_attn_ada", "v_cross_attn_ada"):
        grad = getattr(block, name).bias.grad
        assert grad is not None and grad[2 * DIM :].abs().max() > 0, f"Dead zero-init {name} gate"

    model.zero_grad(set_to_none=True)
    open_gates(model)
    outputs = forward_probe()
    sum((out * torch.randn_like(out)).mean() for out in outputs).backward()
    for name, parameter in block.cross_attn.named_parameters():
        assert_grad(parameter, f"opened gate CA {name}")

    q = torch.randn(2, HEADS, 9, HEAD_DIM, device=device)
    k = torch.randn_like(q)
    rope = model.rotary_embed.forward_from_seq_len(q.shape[2])
    qr, kr = block._apply_rope(q.clone(), k.clone(), rope)
    assert block.pe_attn_head is None
    for head_index in range(HEADS):
        assert not torch.allclose(qr[:, head_index, 1:], q[:, head_index, 1:])
        assert not torch.allclose(kr[:, head_index, 1:], k[:, head_index, 1:])
    assert all(block.attn.processor.pe_attn_head is None for block in model.transformer_blocks)
    print("[OK] both CA gates learn on first step; projections learn after gates open; AV/tail RoPE all-head")


def expected_unused(model):
    # The final video state is deliberately discarded before the audio-only
    # tail, just as in the original architecture. Preserve and document this
    # topology rather than silently attach an artificial training objective.
    prefix = f"transformer_blocks.{model.n_mm_layers - 1}."
    # Concatenated A/V CA queries give the final V query/modulation path
    # zero-valued (rather than missing) gradients. Only these output branches
    # are entirely disconnected from the final audio/CTC objectives.
    video_output_prefixes = ("v_ff.", "cross_attn.v_cond_cross_proj.")
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(name.startswith(prefix + suffix) for suffix in video_output_prefixes)
    }


def test_full_model(device, n_mm_layers, checkpoint):
    model = tiny_model(device, n_mm_layers, checkpoint=checkpoint).train()
    assert all(isinstance(block, MMDiTBlock_VT) for block in model.transformer_blocks[:n_mm_layers])
    assert all(type(block) is DiTBlock for block in model.transformer_blocks[n_mm_layers:])
    assert all(not hasattr(block, "cross_attn") for block in model.transformer_blocks[n_mm_layers:])
    open_gates(model)
    inputs = model_inputs(device)
    pred, taps = model(**inputs)
    assert pred.shape == (2, 16, 8)
    assert list(taps) == ([5, 11] if n_mm_layers == 6 else [6, 12])
    loss = F.mse_loss(pred, torch.randn_like(pred))
    loss = loss + sum(tap["z_tilde"].square().mean() for tap in taps.values())
    assert torch.isfinite(loss)
    loss.backward()
    unused = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    }
    assert unused == expected_unused(model), (
        f"Unexpected unused parameters: {sorted(unused - expected_unused(model))}; "
        f"expected-but-used: {sorted(expected_unused(model) - unused)}"
    )
    for name, parameter in model.named_parameters():
        if name not in unused:
            assert_grad(parameter, name, nonzero=False)
    # Earlier video CA must influence the final AUDIO prediction, rather than
    # only producing an unused side output in its own block.
    for layer in (0, n_mm_layers - 2):
        ca = model.transformer_blocks[layer].cross_attn
        assert_grad(ca.v_cond_cross_q.weight, f"layer {layer} video-to-text path")
        assert_grad(ca.audio_cross_q.weight, f"layer {layer} audio-to-text path")
    model.eval()
    with torch.no_grad():
        eval_pred, _ = model(**inputs)
    assert torch.isfinite(eval_pred).all()
    name = "D1" if n_mm_layers == 6 else "C2"
    print(
        f"[OK] {name}: {n_mm_layers} MM + {18 - n_mm_layers} native audio layers; checkpoint={checkpoint}; finite train/eval/backward"
    )
    print(
        f"     {len(unused)} expected final-video-only parameter tensors unused: DDP requires find_unused_parameters=True"
    )


def test_legacy_mode(device):
    legacy = tiny_model(device, mode="audio_only")
    clone = tiny_model(device, mode="audio_only")
    clone.load_state_dict(legacy.state_dict(), strict=True)
    assert all(isinstance(block.cross_attn, nn.MultiheadAttention) for block in legacy.transformer_blocks[:6])
    assert all(not hasattr(block, "v_cross_attn_ada") for block in legacy.transformer_blocks[:6])
    assert legacy.transformer_blocks[0].pe_attn_head == 1
    new_model = tiny_model(device)
    old_audio = {
        key: value.shape for key, value in legacy.state_dict().items() if key.startswith("transformer_blocks.6.")
    }
    new_audio = {
        key: value.shape for key, value in new_model.state_dict().items() if key.startswith("transformer_blocks.6.")
    }
    assert old_audio == new_audio, "Native audio-only parameter names/shapes must remain checkpoint-compatible"
    print("[OK] legacy audio-only CA mode strict-loads; native audio-only parameter structure unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Defaults to CPU; optional example: cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(20260905)
    torch.set_num_threads(2)
    test_ca_reference_and_text_mask(device)
    test_gate_liveness_and_all_head_joint_rope(device)
    test_legacy_mode(device)
    for n_mm_layers in (6, 12):
        for checkpoint in (False, True):
            test_full_model(device, n_mm_layers, checkpoint)
    print(f"All Hunyuan dual-CA regressions passed on {device}.")


if __name__ == "__main__":
    main()
