"""Parameter-free, content-independent Gaussian prior on physical frame times.

The center follows each audio query's timestamp plus a fixed offset; "fixed"
does not mean that every query attends to one global video timestamp. Positive
offsets point towards later video frames. Width is the Gaussian standard
deviation in seconds, not a hard cutoff or a number of frames.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FixedTemporalBand(nn.Module):
    """Match the adaptive band's interface without learning or reading content.

    Constants are plain Python attributes, not parameters or buffers. Creating
    this module consumes no random numbers and adds no checkpoint state keys.
    """

    def __init__(
        self,
        dim: int,
        audio_fps: float = 40.0,
        video_fps: float = 40.0,
        offset_seconds: float = 0.0,
        sigma_seconds: float = 0.100,
    ):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not all(math.isfinite(value) for value in (audio_fps, video_fps, offset_seconds, sigma_seconds)):
            raise ValueError("temporal-band frame rates, offset, and sigma must be finite")
        if audio_fps <= 0 or video_fps <= 0:
            raise ValueError("audio_fps and video_fps must be positive")
        if sigma_seconds <= 0:
            raise ValueError("sigma_seconds must be positive")
        self.dim = dim
        self.audio_fps = float(audio_fps)
        self.video_fps = float(video_fps)
        self.offset_seconds = float(offset_seconds)
        self.sigma_seconds = float(sigma_seconds)

    def forward(self, video: torch.Tensor, audio_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 [batch, audio queries] constants on the video's device.

        Only video shape and device are used, so conditioning dropout, feature
        changes, and packed CFG branches cannot move or widen the band.
        """
        if video.ndim != 3 or video.shape[-1] != self.dim:
            raise ValueError(f"video must have shape [batch, frames, {self.dim}], got {tuple(video.shape)}")
        if video.shape[0] == 0 or video.shape[1] == 0:
            raise ValueError("batch, audio, and video sequences must be non-empty")
        if not isinstance(audio_len, int) or isinstance(audio_len, bool) or audio_len <= 0:
            raise ValueError("audio_len must be a positive integer")
        shape = (video.shape[0], audio_len)
        offset = torch.full(shape, self.offset_seconds, device=video.device, dtype=torch.float32)
        sigma = torch.full(shape, self.sigma_seconds, device=video.device, dtype=torch.float32)
        return offset, sigma

    def bias(self, offset_seconds: torch.Tensor, sigma_seconds: torch.Tensor, video_len: int) -> torch.Tensor:
        """Return FP32 [batch, audio queries, video keys] soft logit bias.

        Offset and sigma tensors come from ``forward``, whose constructor-
        validated constants are finite with positive sigma. Compute physical
        distances directly in FP32, including under BF16 autocast; expanded
        dot-product forms suffer cancellation on long sequences.
        """
        if offset_seconds.ndim != 2 or sigma_seconds.shape != offset_seconds.shape:
            raise ValueError("offset_seconds and sigma_seconds must have matching [batch, audio] shapes")
        if offset_seconds.shape[0] == 0 or offset_seconds.shape[1] == 0:
            raise ValueError("batch and audio sequences must be non-empty")
        if offset_seconds.device != sigma_seconds.device:
            raise ValueError("offset_seconds and sigma_seconds must be on the same device")
        if not isinstance(video_len, int) or isinstance(video_len, bool) or video_len <= 0:
            raise ValueError("video_len must be a positive integer")
        audio_times = torch.arange(offset_seconds.shape[1], device=offset_seconds.device, dtype=torch.float32)
        video_times = torch.arange(video_len, device=offset_seconds.device, dtype=torch.float32)
        audio_times = audio_times / self.audio_fps
        video_times = video_times / self.video_fps
        distance = video_times[None, None, :] - audio_times[None, :, None] - offset_seconds.float().unsqueeze(-1)
        return -0.5 * (distance / sigma_seconds.float().unsqueeze(-1)).square()
