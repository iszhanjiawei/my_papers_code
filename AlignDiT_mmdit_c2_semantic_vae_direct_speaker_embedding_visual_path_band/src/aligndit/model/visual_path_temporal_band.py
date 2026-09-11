"""Parameter-free AV alignment in physical time and frozen visual path distance.

The path is measured on native-rate AV-HuBERT features, before their cached
40 Hz interpolation. It is a cumulative Euclidean path on unit-normalized
features, not an endpoint similarity or a per-utterance normalized position.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aligndit.model.fixed_temporal_band import FixedTemporalBand


@torch.no_grad()
def native_visual_path(video: torch.Tensor, target_length: int) -> torch.Tensor:
    """Return detached FP32 ``[target_length]`` cumulative native-feature path.

    Call at CPU cache loading time: validation deliberately checks finiteness.
    Scalar coordinates use exactly ``linear, align_corners=False`` resampling,
    matching the existing native-feature-to-40 Hz video cache coordinates.
    Computing increments *after* feature interpolation would define a different
    path, and is intentionally not offered as an implicit fallback.
    """
    if video.ndim != 2 or video.shape[0] == 0 or video.shape[1] == 0:
        raise ValueError("native video must have non-empty shape [frames, features]")
    if not video.is_floating_point():
        raise TypeError("native video features must be floating point")
    if not isinstance(target_length, int) or isinstance(target_length, bool) or target_length <= 0:
        raise ValueError("target_length must be a positive integer")
    video_fp32 = video.detach().float()
    if not torch.isfinite(video_fp32).all().item():
        raise ValueError("native video features must be finite")
    normalized = F.normalize(video_fp32, p=2, dim=-1, eps=1e-12)
    increments = torch.linalg.vector_norm(normalized[1:] - normalized[:-1], ord=2, dim=-1)
    cumulative = F.pad(increments.cumsum(dim=0), (1, 0))
    return F.interpolate(
        cumulative[None, None], size=target_length, mode="linear", align_corners=False
    )[0, 0].contiguous()


@torch.no_grad()
def masked_visual_path(
    video_path: torch.Tensor, visible_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove hidden/invalid adjacent edges, then rebuild a cumulative path.

    Returns ``(path[B,N], increments[B,N-1], valid_pairs[B,N-1])``. Rebuilding
    ensures hidden-edge motion cannot survive as a cumulative-coordinate shift
    after a prompt span. Null-video branches pass an all-false visibility mask
    and obtain a constant-zero path independent of the original content.
    Sources are finite/monotone-validated at loading; no GPU synchronization is
    introduced here. Clamping differences at zero handles FP32 roundoff.
    """
    if video_path.ndim != 2 or video_path.shape[0] == 0 or video_path.shape[1] == 0:
        raise ValueError("video_path must have non-empty shape [batch, video frames]")
    if not video_path.is_floating_point():
        raise TypeError("video_path must be floating point")
    if visible_mask is None:
        visible_mask = torch.ones_like(video_path, dtype=torch.bool)
    if visible_mask.shape != video_path.shape:
        raise ValueError("visible_mask must match video_path shape")
    if visible_mask.dtype != torch.bool:
        raise TypeError("visible_mask must be bool")
    if visible_mask.device != video_path.device:
        raise ValueError("visible_mask and video_path must be on the same device")
    valid_pairs = visible_mask[:, 1:] & visible_mask[:, :-1]
    increments = (video_path.detach().float()[:, 1:] - video_path.detach().float()[:, :-1]).clamp_min(0)
    increments = increments.masked_fill(~valid_pairs, 0.0)
    cumulative = F.pad(increments.cumsum(dim=1), (1, 0))
    return cumulative, increments, valid_pairs


class VisualPathTemporalBand(FixedTemporalBand):
    """A fixed time prior plus content-derived, non-learned visual path distance.

    ``B(i,j) = -.5 * ((t_i-t_j)/sigma_t)^2
                -.5 * ((c(t_i)-c(t_j))/sigma_c)^2``.

    All constants are plain Python attributes; no parameters, buffers, random
    draws, mass-preserving renormalization, or checkpoint state are added.
    """

    def __init__(
        self,
        dim: int,
        audio_fps: float = 40.0,
        video_fps: float = 40.0,
        offset_seconds: float = 0.0,
        sigma_seconds: float = 0.100,
        path_sigma: float = 2.0,
    ):
        super().__init__(dim, audio_fps, video_fps, offset_seconds, sigma_seconds)
        if offset_seconds != 0.0:
            raise ValueError("visual_path mode requires zero fixed temporal offset")
        if not math.isfinite(path_sigma) or path_sigma <= 0:
            raise ValueError("path_sigma must be finite and positive")
        self.path_sigma = float(path_sigma)

    def bias(
        self,
        offset_seconds: torch.Tensor,
        sigma_seconds: torch.Tensor,
        video_len: int,
        video_path: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return detached FP32 ``[batch,audio,video]`` time-and-path bias.

        ``video_path`` must already be masking/CFG-safe. Audio-query path
        coordinates are linearly sampled at physical timestamps, with boundary
        clamping; at the experiment's 40/40 Hz rates this is exact indexing.
        """
        temporal_bias = super().bias(offset_seconds, sigma_seconds, video_len)
        if video_path is None:
            raise ValueError("visual_path mode requires native-derived video_path")
        if video_path.shape != (offset_seconds.shape[0], video_len):
            raise ValueError("video_path must have shape [batch, video_len]")
        if not video_path.is_floating_point():
            raise TypeError("video_path must be floating point")
        if video_path.device != offset_seconds.device:
            raise ValueError("video_path and temporal-band tensors must be on the same device")
        path = video_path.detach().float()
        query_positions = torch.arange(offset_seconds.shape[1], device=path.device, dtype=torch.float32)
        query_positions = (query_positions * (self.video_fps / self.audio_fps)).clamp(max=video_len - 1)
        left = query_positions.floor().long()
        right = (left + 1).clamp(max=video_len - 1)
        fraction = query_positions - left.float()
        query_path = path[:, left] + fraction[None] * (path[:, right] - path[:, left])
        distance = query_path[:, :, None] - path[:, None, :]
        return temporal_bias - 0.5 * (distance / self.path_sigma).square()
