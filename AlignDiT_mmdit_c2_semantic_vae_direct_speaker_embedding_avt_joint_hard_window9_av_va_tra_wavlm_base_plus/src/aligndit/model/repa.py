"""Contracts and loss helpers for frozen frame-level WavLM REPA targets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


WAVLM_BASE_PLUS_MODEL_ID = "microsoft/wavlm-base-plus"
WAVLM_BASE_PLUS_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WAVLM_BASE_PLUS_CHECKPOINT_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
WAVLM_BASE_PLUS_DIM = 768
WAVLM_BASE_PLUS_LAYER = 12
WAVLM_FRAME_STRIDE_SAMPLES = 320
REPA_CACHE_DTYPE = np.dtype("float16")


class RepaFeatureError(RuntimeError):
    """Raised when a cached REPA target violates the experiment contract."""


def validate_repa_feature_array(
    feature: np.ndarray,
    *,
    expected_dim: int = WAVLM_BASE_PLUS_DIM,
    expected_dtype: np.dtype = REPA_CACHE_DTYPE,
    source: str | Path = "REPA feature",
) -> None:
    if feature.ndim != 2 or feature.shape[0] <= 0 or feature.shape[1] != expected_dim:
        raise RepaFeatureError(f"{source}: expected shape [frames, {expected_dim}], got {feature.shape}")
    if feature.dtype != expected_dtype:
        raise RepaFeatureError(f"{source}: expected {expected_dtype}, got {feature.dtype}")
    if not np.isfinite(feature).all():
        raise RepaFeatureError(f"{source}: contains NaN or Inf")


def validate_repa_cache_metadata(
    cache_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_count: int,
    expected_dim: int = WAVLM_BASE_PLUS_DIM,
    model_id: str = WAVLM_BASE_PLUS_MODEL_ID,
    model_revision: str = WAVLM_BASE_PLUS_REVISION,
    checkpoint_sha256: str = WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    teacher_layer: int = WAVLM_BASE_PLUS_LAYER,
) -> dict:
    """Validate the immutable extraction and complete-coverage markers."""

    root = Path(cache_dir).expanduser().absolute()
    metadata_path = root / "metadata.json"
    coverage_path = root / "coverage_report.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RepaFeatureError(f"cannot read complete REPA cache metadata under {root}") from error
    expected = {
        "schema_version": 1,
        "status": "complete",
        "feature": "wavlm_hidden_state",
        "model_id": model_id,
        "model_revision": model_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "teacher_layer": teacher_layer,
        "sample_rate": 16_000,
        "frame_stride_samples": WAVLM_FRAME_STRIDE_SAMPLES,
        "frame_rate_hz": 50.0,
        "output_dim": expected_dim,
        "output_dtype": "float16",
        "source_audio": "complete_unmasked_waveform",
        "manifest_sha256": expected_manifest_sha256,
        "expected_count": expected_count,
        "coverage_report": "coverage_report.json",
    }
    if not isinstance(metadata, dict):
        raise RepaFeatureError(f"expected a JSON object in {metadata_path}")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RepaFeatureError(
                f"REPA cache contract mismatch in {metadata_path}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    if (
        not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or coverage.get("expected") != expected_count
        or coverage.get("valid") != expected_count
        or coverage.get("split_counts") != {"train": expected_count}
        or any(coverage.get(key) != [] for key in ("missing", "invalid", "extra"))
    ):
        raise RepaFeatureError(f"REPA cache failed coverage contract: {coverage_path}")
    return metadata


def validate_repa_schedule(schedule: str, eps: float) -> None:
    if schedule not in ("none", "taro_clean_fraction"):
        raise ValueError(f"unsupported repa_schedule: {schedule!r}")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError(f"repa_schedule_eps must be finite and positive, got {eps!r}")


def repa_timestep_weights(
    time: torch.Tensor,
    *,
    schedule: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """TARO Eq. 7 in this model's noise-to-clean flow-time convention.

    Here x_t = (1-t)*noise + t*clean, hence alpha=t and sigma=1-t.
    sigmoid(log(alpha/(sigma+eps))) = alpha/(alpha+sigma+eps).
    The rational form is finite at both endpoints without evaluating log(0).
    This is amplitude-ratio weighting, not squared SNR or log-sigmoid.
    """
    validate_repa_schedule(schedule, eps)
    if time.ndim != 1 or not time.is_floating_point():
        raise ValueError("REPA time must be a floating-point tensor with shape [batch]")
    time = time.detach().float()
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("REPA time must contain finite values in [0, 1]")
    if schedule == "none":
        return torch.ones_like(time)
    alpha, sigma = time, 1.0 - time
    return alpha / (alpha + sigma + float(eps))


def masked_repa_cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    teacher_lens: torch.Tensor,
    student_lens: torch.Tensor,
    generation_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align frozen 50-Hz targets to valid 40-Hz generated audio frames.

    Each sample's weight multiplies its frame losses BEFORE batch reduction.
    The denominator stays the number of generated, valid frames (not the sum
    of weights), so low-clean-fraction batches really receive weaker guidance.
    Interpolation is per utterance, excluding padded teacher frames. Prompt
    frames and student padding contribute neither loss nor gradients.
    """
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError("student and teacher REPA tensors must have shape [batch, frames, dim]")
    if student.shape[0] != teacher.shape[0] or student.shape[2] != teacher.shape[2]:
        raise ValueError(f"student/teacher REPA shape mismatch: {student.shape} vs {teacher.shape}")
    if generation_mask.dtype != torch.bool or generation_mask.shape != student.shape[:2]:
        raise ValueError(
            f"generation_mask must be bool with shape {tuple(student.shape[:2])}, got {generation_mask.shape}"
        )
    batch = student.shape[0]
    if teacher_lens.shape != (batch,) or student_lens.shape != (batch,):
        raise ValueError("teacher_lens and student_lens must each have shape [batch]")
    if sample_weights is None:
        weights = torch.ones(batch, device=student.device, dtype=torch.float32)
    else:
        if sample_weights.shape != (batch,) or not sample_weights.is_floating_point():
            raise ValueError("REPA sample_weights must be floating-point with shape [batch]")
        weights = sample_weights.detach().to(device=student.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("REPA sample_weights must be finite and non-negative")

    cosine_sum = student.new_zeros((), dtype=torch.float32)
    weighted_loss_sum = student.new_zeros((), dtype=torch.float32)
    frame_weight_sum = student.new_zeros((), dtype=torch.float32)
    frame_count = 0
    for index in range(batch):
        teacher_len = int(teacher_lens[index].item())
        student_len = int(student_lens[index].item())
        if not 0 < teacher_len <= teacher.shape[1] or not 0 < student_len <= student.shape[1]:
            raise ValueError(
                f"invalid REPA lengths at batch index {index}: teacher={teacher_len}, student={student_len}"
            )
        active = generation_mask[index, :student_len]
        active_count = int(active.sum().item())
        if active_count == 0:
            continue
        target = teacher[index, :teacher_len].detach().float().transpose(0, 1).unsqueeze(0)
        target = F.interpolate(target, size=student_len, mode="linear", align_corners=False)
        target = target.squeeze(0).transpose(0, 1)[active]
        prediction = student[index, :student_len][active].float()
        cosine = F.cosine_similarity(prediction, target, dim=-1)
        cosine_sum = cosine_sum + cosine.sum()
        weighted_loss_sum = weighted_loss_sum + weights[index] * (1.0 - cosine).sum()
        frame_weight_sum = frame_weight_sum + weights[index] * active_count
        frame_count += active_count
    if frame_count == 0:
        raise RuntimeError("REPA generation mask selected no valid frames")
    unweighted_loss = 1.0 - cosine_sum / frame_count
    # Retain the donor's unweighted computation when no weights are supplied.
    loss = unweighted_loss if sample_weights is None else weighted_loss_sum / frame_count
    if not return_diagnostics:
        return loss
    return loss, {
        "repa_unweighted_loss": unweighted_loss.detach(),
        "repa_time_weight_mean": weights.mean(),
        "repa_time_weight_min": weights.min(),
        "repa_time_weight_max": weights.max(),
        "repa_time_weight_frame_mean": frame_weight_sum / frame_count,
    }
