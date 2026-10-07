"""Text cross-attention with an independent, near-zero per-channel gate.

The audio tail reads the existing encoded text context. As in the local visual
branch, Q/K are RMS-normalized over their projected width, and a small residual
gate protects the pretrained audio path without blocking branch gradients.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from f5_tts.model.modules import RMSNorm


class GatedTailTextAttention(nn.Module):
    def __init__(self, dim, text_dim, heads, dim_head, gate_init=1e-5):
        super().__init__()
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite")
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(text_dim, inner_dim)
        self.to_v = nn.Linear(text_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.q_norm = RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = RMSNorm(inner_dim, eps=1e-6)
        # Keep output projection normally initialized: zeroing both it and the
        # gate would prevent the branch from learning on the first update.
        self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(self, x, text, audio_mask=None, text_mask=None, generation_mask=None):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, audio_time, dim]")
        batch, audio_len, _ = x.shape
        if text.ndim != 3 or text.shape[0] != batch or text.shape[2] != self.to_k.in_features:
            raise ValueError("text must have shape [batch, tokens, text_dim]")
        if text.device != x.device:
            raise ValueError(f"text must be on {x.device}, got {text.device}")
        text_len = text.shape[1]
        for name, mask, shape in (
            ("audio_mask", audio_mask, (batch, audio_len)),
            ("text_mask", text_mask, (batch, text_len)),
            ("generation_mask", generation_mask, (batch, audio_len)),
        ):
            if mask is not None and (mask.dtype != torch.bool or mask.shape != shape or mask.device != x.device):
                raise ValueError(f"{name} must be a bool tensor of shape {shape} on {x.device}")
        if text_mask is None:
            text_mask = torch.ones((batch, text_len), dtype=torch.bool, device=x.device)
        query_mask = torch.ones((batch, audio_len), dtype=torch.bool, device=x.device)
        if audio_mask is not None:
            query_mask = query_mask & audio_mask
        if generation_mask is not None:
            query_mask = query_mask & generation_mask
        # A masked dummy token keeps empty contexts finite and leaves all branch
        # parameters in the autograd graph (with zero gradients).
        if text_len == 0:
            text = F.pad(text, (0, 0, 0, 1))
            text_mask = F.pad(text_mask, (0, 1), value=False)
        clean_text = text.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        clean_x = x.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        query = self.q_norm(self.to_q(clean_x))
        key = self.k_norm(self.to_k(clean_text))
        value = self.to_v(clean_text)
        query, key, value = [
            tensor.reshape(batch, -1, self.heads, self.dim_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        has_text = text_mask.any(dim=-1)
        # Avoid all-masked softmax on older SDPA kernels. Zero dropped examples
        # after output projection as its bias can otherwise leak a residual.
        attention_mask = text_mask | ~has_text[:, None]
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask[:, None, None, :], dropout_p=0.0,
        )
        output = output.transpose(1, 2).reshape(batch, audio_len, self.heads * self.dim_head)
        output = self.to_out(output) * self.gate
        return output.masked_fill(~(query_mask & has_text[:, None]).unsqueeze(-1), 0.0)
