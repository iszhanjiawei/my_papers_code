"""Contracts and loss helpers for frozen frame-level WavLM REPA targets."""

from __future__ import annotations

import json
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
                f"REPA cache contract mismatch in {metadata_path}: "
                f"{key}={metadata.get(key)!r}, expected {value!r}"
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


def masked_repa_cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    teacher_lens: torch.Tensor,
    student_lens: torch.Tensor,
    generation_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``1-cos`` after aligning 50-Hz targets to valid 40-Hz frames.

    Only randomly masked (generated) frames contribute. Interpolation is done
    per utterance so padded teacher frames can never leak into a valid target.
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

    cosine_sum = student.new_zeros((), dtype=torch.float32)
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
        cosine_sum = cosine_sum + F.cosine_similarity(prediction, target, dim=-1).sum()
        frame_count += active_count
    if frame_count == 0:
        raise RuntimeError("REPA generation mask selected no valid frames")
    return 1.0 - cosine_sum / frame_count
