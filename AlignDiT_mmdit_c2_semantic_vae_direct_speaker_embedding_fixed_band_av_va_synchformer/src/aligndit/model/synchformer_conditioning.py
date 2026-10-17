"""Temporal Synchformer conditioning, following HunyuanVideo-Foley.

Eight native temporal tokens per visual window receive a shared learned
position embedding. Kernel-one gated convolutions are expressed as linears.
Only valid clip frames are resampled; batch padding never changes clip time.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from f5_tts.model.modules import DiTBlock


def sequence_modulation(value):
    return value[:, None] if value.ndim == 2 else value


def adaptive_norm(module, x, t):
    if t.ndim == 2:
        norm, *parts = module(x, emb=t)
        return norm, *[sequence_modulation(part) for part in parts]
    shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = module.linear(module.silu(t)).chunk(6, dim=-1)
    return module.norm(x) * (1 + scale) + shift, gate, mlp_shift, mlp_scale, mlp_gate


class TemporalDiTBlock(DiTBlock):
    """Same parameter names as the pretrained block; accepts [B,T,D] time."""

    def forward(self, x, t, mask=None, rope=None):
        norm, gate, shift, scale, mlp_gate = adaptive_norm(self.attn_norm, x, t)
        x = x + gate * self.attn(x=norm, mask=mask, rope=rope)
        return x + mlp_gate * self.ff(self.ff_norm(x) * (1 + scale) + shift)


class SyncMLP(nn.Module):
    """Foley ConvMLP(kernel_size=1), with a single zero output projection."""

    def __init__(self, dim):
        super().__init__()
        hidden = 256 * ((int(2 * (dim * 4) / 3) + 255) // 256)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def resample_sync(sync, source_lens, target_lens, target_length, start=None):
    """Per-example nearest-exact interpolation onto an optional shifted axis.

    target_lens describes the original clip duration, even if the requested
    output is truncated. start is its audio-timeline offset (e.g. prompt size).
    """
    if start is None:
        start = torch.zeros_like(target_lens)
    positions = torch.arange(target_length, device=sync.device)[None] - start[:, None]
    indices = ((positions.to(torch.float64) + 0.5) * source_lens[:, None] / target_lens[:, None].clamp_min(1)).floor().long()
    indices = torch.minimum(indices, source_lens[:, None] - 1).clamp_min(0)
    result = sync.gather(1, indices[..., None].expand(-1, -1, sync.shape[-1]))
    valid = (positions >= 0) & (positions < target_lens[:, None])
    return result.masked_fill(~valid[..., None], 0.0)
