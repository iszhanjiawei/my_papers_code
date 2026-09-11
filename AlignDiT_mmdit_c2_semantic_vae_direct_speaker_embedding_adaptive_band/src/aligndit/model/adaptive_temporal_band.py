"""Video-conditioned Gaussian time prior, shared by the MM attention blocks.

Time quantities are in seconds. Positive offsets move an audio query's band
towards *later* video frames. This is a soft logit prior, not a hard mask or a
normalization that preserves the total attention allocated to video.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveTemporalBand(nn.Module):
    """Predict one center offset and width per audio query from video features.

    ``video`` must be the branch-specific embedded video, after classifier-free
    conditioning dropout. Interpolation uses physical frame times rather than
    stretching each padded sequence to the audio length. Queries beyond the
    available video grid use its last feature, with the time grid unchanged.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        audio_fps: float = 40.0,
        video_fps: float = 40.0,
        max_offset_seconds: float = 0.100,
        min_sigma_seconds: float = 0.025,
        max_sigma_seconds: float = 0.250,
        init_sigma_seconds: float = 0.100,
    ):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        quantities = (
            audio_fps, video_fps, max_offset_seconds,
            min_sigma_seconds, max_sigma_seconds, init_sigma_seconds,
        )
        if not all(math.isfinite(value) for value in quantities):
            raise ValueError("temporal-band frame rates and bounds must be finite")
        if audio_fps <= 0 or video_fps <= 0:
            raise ValueError("audio_fps and video_fps must be positive")
        if max_offset_seconds < 0:
            raise ValueError("max_offset_seconds must be non-negative")
        if not 0 < min_sigma_seconds < init_sigma_seconds < max_sigma_seconds:
            raise ValueError("expected 0 < min_sigma_seconds < init_sigma_seconds < max_sigma_seconds")
        self.dim = dim
        self.audio_fps = float(audio_fps)
        self.video_fps = float(video_fps)
        self.max_offset_seconds = float(max_offset_seconds)
        self.min_sigma_seconds = float(min_sigma_seconds)
        self.max_sigma_seconds = float(max_sigma_seconds)
        self.init_sigma_seconds = float(init_sigma_seconds)
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        initial_fraction = (init_sigma_seconds - min_sigma_seconds) / (max_sigma_seconds - min_sigma_seconds)
        with torch.no_grad():
            self.net[-1].bias[1] = math.log(initial_fraction / (1.0 - initial_fraction))

    def forward(self, video: torch.Tensor, audio_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 3 or video.shape[-1] != self.dim:
            raise ValueError(f"video must have shape [batch, frames, {self.dim}], got {tuple(video.shape)}")
        if video.shape[1] == 0 or audio_len <= 0:
            raise ValueError("audio and video sequences must be non-empty")

        if self.audio_fps == self.video_fps and audio_len == video.shape[1]:
            aligned_video = video
        else:
            positions = torch.arange(audio_len, device=video.device, dtype=torch.float32)
            positions = (positions * (self.video_fps / self.audio_fps)).clamp(max=video.shape[1] - 1)
            left = positions.floor().long()
            right = (left + 1).clamp(max=video.shape[1] - 1)
            fraction = (positions - left).to(video.dtype)[None, :, None]
            aligned_video = video[:, left] * (1.0 - fraction) + video[:, right] * fraction

        normalized = F.layer_norm(aligned_video, (self.dim,), weight=None, bias=None, eps=1e-6)
        # Keep the physical time calculations in FP32 under mixed precision.
        raw_offset, raw_sigma = self.net(normalized).float().unbind(dim=-1)
        offset = self.max_offset_seconds * raw_offset.tanh()
        sigma = self.min_sigma_seconds + (self.max_sigma_seconds - self.min_sigma_seconds) * raw_sigma.sigmoid()
        return offset, sigma

    def bias(self, offset_seconds: torch.Tensor, sigma_seconds: torch.Tensor, video_len: int) -> torch.Tensor:
        """Return FP32 [batch, audio queries, video keys] Gaussian logit bias.

        Compute squared distances directly: expanding the square into dot-
        product features is numerically unstable in BF16 on long sequences.
        """
        if offset_seconds.ndim != 2 or sigma_seconds.shape != offset_seconds.shape:
            raise ValueError("offset_seconds and sigma_seconds must have matching [batch, audio] shapes")
        if video_len <= 0:
            raise ValueError("video_len must be positive")
        audio_times = torch.arange(offset_seconds.shape[1], device=offset_seconds.device, dtype=torch.float32)
        video_times = torch.arange(video_len, device=offset_seconds.device, dtype=torch.float32)
        audio_times = audio_times / self.audio_fps
        video_times = video_times / self.video_fps
        distance = video_times[None, None, :] - audio_times[None, :, None] - offset_seconds.float().unsqueeze(-1)
        return -0.5 * (distance / sigma_seconds.float().unsqueeze(-1)).square()
