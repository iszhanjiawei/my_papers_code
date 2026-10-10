"""Flowley temporal weighting with OmniShow's per-channel residual gate.

The default window is omega=0, delta=4 at Flowley's 8 FPS: a cosine
fade with a 0.5 s radius. At 40 Hz this spans 20 tokens on either side.
Like Flowley's implementation, log(weight + 1e-6) is a soft attention
bias, not a hard cutoff. The gated residual excludes padding with -inf;
joint attention combines the temporal prior with its own masking policy.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from f5_tts.model.modules import RMSNorm


class FlowleyTemporalWindow(nn.Module):
    """Parameter-free temporal prior shared by joint and visual-only attention.

    The module deliberately registers no parameters or buffers, so adding the
    prior to an existing attention preserves pretrained checkpoint keys.
    """

    def __init__(
        self,
        audio_frame_rate=40.0,
        audio_video_ratio=1,
        window_radius_seconds=0.5,
        window_reference_fps=8.0,
        window_core_radius=0.0,
        window_fade_scale=1.0,
    ):
        super().__init__()
        for name, value in (
            ("audio_frame_rate", audio_frame_rate),
            ("audio_video_ratio", audio_video_ratio),
            ("window_radius_seconds", window_radius_seconds),
            ("window_reference_fps", window_reference_fps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not math.isfinite(window_core_radius) or not 0 <= window_core_radius < window_radius_seconds * window_reference_fps:
            raise ValueError("window_core_radius must be nonnegative and smaller than the total window radius")
        if not math.isfinite(window_fade_scale) or not 0 <= window_fade_scale <= 1:
            raise ValueError("window_fade_scale must be in [0, 1]")
        self.audio_frame_rate = float(audio_frame_rate)
        self.video_frame_rate = float(audio_frame_rate / audio_video_ratio)
        self.audio_video_ratio = float(audio_video_ratio)
        self.window_radius_seconds = float(window_radius_seconds)
        self.window_reference_fps = float(window_reference_fps)
        self.window_core_radius = float(window_core_radius)
        self.window_fade_scale = float(window_fade_scale)

    def temporal_bias(self, audio_len, video_len, device, video_lengths=None):
        """Return [Ta,Tv], or [B,Ta,Tv] for per-example valid lengths.

        Coordinates use the configured frame rates, never the padded batch
        length ratio. Rounding follows Flowley's code (the paper uses floor).
        """
        centers = torch.round(torch.arange(audio_len, device=device, dtype=torch.float32) / self.audio_video_ratio)
        if video_lengths is None:
            centers = centers.clamp(max=max(video_len - 1, 0))
        else:
            centers = torch.minimum(centers[None, :], (video_lengths[:, None] - 1).clamp_min(0))
        positions = torch.arange(video_len, device=device, dtype=torch.float32)
        distance = (positions - centers.unsqueeze(-1)).abs() * (self.window_reference_fps / self.video_frame_rate)
        radius = self.window_radius_seconds * self.window_reference_fps
        phase = ((distance - self.window_core_radius) / (radius - self.window_core_radius)).clamp(0, 1)
        weight = self.window_fade_scale * 0.5 * (1 + torch.cos(math.pi * phase))
        weight = torch.where(distance <= self.window_core_radius, 1.0, weight)
        return (weight + 1e-6).log()


class GatedLocalVisualAttention(FlowleyTemporalWindow):
    def __init__(
        self,
        dim,
        visual_dim,
        heads,
        dim_head,
        audio_frame_rate=40.0,
        audio_video_ratio=1,
        window_radius_seconds=0.5,
        window_reference_fps=8.0,
        window_core_radius=0.0,
        window_fade_scale=1.0,
        gate_init=1e-5,
    ):
        super().__init__(
            audio_frame_rate=audio_frame_rate,
            audio_video_ratio=audio_video_ratio,
            window_radius_seconds=window_radius_seconds,
            window_reference_fps=window_reference_fps,
            window_core_radius=window_core_radius,
            window_fade_scale=window_fade_scale,
        )
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite")
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(visual_dim, inner_dim)
        self.to_v = nn.Linear(visual_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        # OmniShow normalizes the full projected Q/K before splitting heads.
        self.q_norm = RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = RMSNorm(inner_dim, eps=1e-6)
        self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(self, x, video, audio_mask=None, video_mask=None, generation_mask=None):
        batch, audio_len, _ = x.shape
        video_len = video.shape[1]
        if video.ndim != 3 or video.shape[0] != batch or video_len == 0:
            raise ValueError("video must have shape [batch, nonempty time, visual_dim]")
        for name, mask, shape in (
            ("audio_mask", audio_mask, (batch, audio_len)),
            ("video_mask", video_mask, (batch, video_len)),
            ("generation_mask", generation_mask, (batch, audio_len)),
        ):
            if mask is not None and (mask.dtype != torch.bool or mask.shape != shape or mask.device != x.device):
                raise ValueError(f"{name} must be a bool tensor of shape {shape} on {x.device}")
        if video_mask is None:
            video_mask = torch.ones((batch, video_len), dtype=torch.bool, device=x.device)
        query_mask = torch.ones((batch, audio_len), dtype=torch.bool, device=x.device)
        if audio_mask is not None:
            query_mask = query_mask & audio_mask
        if generation_mask is not None:
            query_mask = query_mask & generation_mask
        # Exclude invalid values before projections, including fully dropped CFG
        # examples. Masking only attention scores would leave NaN/Inf K/V unsafe.
        clean_video = video.masked_fill(~video_mask.unsqueeze(-1), 0.0)
        clean_x = x.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        query = self.q_norm(self.to_q(clean_x))
        key = self.k_norm(self.to_k(clean_video))
        value = self.to_v(clean_video)
        query, key, value = [
            tensor.reshape(batch, -1, self.heads, self.dim_head).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        # The final valid index (rather than sum) handles prompt-prefix holes.
        positions = torch.arange(1, video_len + 1, device=x.device)
        video_lengths = (video_mask * positions).amax(dim=-1)
        bias = self.temporal_bias(audio_len, video_len, x.device, video_lengths)
        bias = bias.masked_fill(~video_mask[:, None, :], float("-inf"))
        has_video = video_mask.any(dim=-1)
        # Avoid undefined all-masked softmax on older SDPA kernels. These
        # examples are zeroed AFTER output projection (including its bias).
        bias = torch.where(has_video[:, None, None], bias, torch.zeros_like(bias))
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias[:, None].to(query.dtype), dropout_p=0.0,
        )
        output = output.transpose(1, 2).reshape(batch, audio_len, self.heads * self.dim_head)
        output = self.to_out(output) * self.gate
        return output.masked_fill(~(query_mask & has_video[:, None]).unsqueeze(-1), 0.0)
