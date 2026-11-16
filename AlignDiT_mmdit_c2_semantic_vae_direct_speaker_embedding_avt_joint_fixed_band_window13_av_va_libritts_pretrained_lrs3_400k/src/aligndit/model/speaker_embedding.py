from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_SPEAKER_EMBEDDING_DIM = 192


class SpeakerEmbeddingError(RuntimeError):
    """Raised when a cached speaker embedding violates the experiment contract."""


def validate_speaker_cache_metadata(
    cache_dir: str | Path,
    *,
    expected_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM,
    model_id: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict:
    """Validate the frozen CAM++ extraction contract before training/inference.

    The coverage report records extraction-time validation. Every vector is
    still checked by ``load_speaker_embedding`` when it is consumed.
    """
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SpeakerEmbeddingError(f"cannot read speaker cache metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise SpeakerEmbeddingError(f"expected a JSON object in {metadata_path}")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "output_shape": [expected_dim],
        "output_dtype": "float32",
        "source_audio": "complete_unmasked_waveform",
        "sample_rate": 16000,
        "fbank_dim": 80,
        "fbank_mean_normalization": "per_chunk_per_mel_over_time",
        "chunk_seconds": 10,
        "max_seconds": 90,
        "padding": "repeat_complete_utterance_to_chunk_multiple",
        "chunk_aggregation": "arithmetic_mean_then_l2_normalize",
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, value in expected.items():
        if value is not None and metadata.get(key) != value:
            raise SpeakerEmbeddingError(
                f"speaker cache contract mismatch in {metadata_path}: "
                f"{key}={metadata.get(key)!r}, expected {value!r}"
            )
    coverage_name = metadata.get("coverage_report")
    if coverage_name != "coverage_report.json":
        raise SpeakerEmbeddingError(f"unexpected speaker coverage report: {coverage_name!r}")
    coverage_path = cache_dir / coverage_name
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SpeakerEmbeddingError(f"cannot read speaker coverage report: {coverage_path}") from error
    expected_count = metadata.get("expected_count")
    if (
        not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or coverage.get("expected") != expected_count
        or coverage.get("valid") != expected_count
        or any(coverage.get(key) != [] for key in ("missing", "invalid", "extra"))
    ):
        raise SpeakerEmbeddingError(f"speaker cache failed coverage contract: {coverage_path}")
    return metadata


def speaker_embedding_path(
    audio_path: str | Path,
    cache_dir: str | Path,
    *,
    audio_root: str | Path | None = None,
) -> Path:
    """Map ``audio/<split>/...wav`` to the mirrored speaker-cache path."""
    audio_path = Path(audio_path)
    if audio_root is not None:
        try:
            relative_path = audio_path.relative_to(Path(audio_root))
        except ValueError as error:
            raise SpeakerEmbeddingError(
                f"audio path {audio_path} is not under the configured audio root {audio_root}"
            ) from error
    else:
        try:
            audio_component_i = len(audio_path.parts) - 1 - audio_path.parts[::-1].index("audio")
        except ValueError as error:
            raise SpeakerEmbeddingError(f"audio path does not contain an 'audio' component: {audio_path}") from error
        relative_path = Path(*audio_path.parts[audio_component_i + 1 :])

    if relative_path.suffix.lower() != ".wav":
        raise SpeakerEmbeddingError(f"expected a .wav audio path, got {audio_path}")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SpeakerEmbeddingError(f"unsafe relative speaker-cache path: {relative_path}")
    return Path(cache_dir) / relative_path.with_suffix(".npy")


def validate_speaker_embedding_array(
    embedding: np.ndarray,
    *,
    expected_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM,
    source: str | Path = "speaker embedding",
    norm_tolerance: float = 1e-4,
) -> None:
    if embedding.shape != (expected_dim,):
        raise SpeakerEmbeddingError(f"{source}: expected shape {(expected_dim,)}, got {embedding.shape}")
    if embedding.dtype != np.float32:
        raise SpeakerEmbeddingError(f"{source}: expected float32, got {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise SpeakerEmbeddingError(f"{source}: contains NaN or Inf")
    norm = float(np.linalg.norm(embedding))
    if abs(norm - 1.0) > norm_tolerance:
        raise SpeakerEmbeddingError(
            f"{source}: expected an L2-normalized vector, got norm={norm:.8f} (tolerance={norm_tolerance})"
        )


def load_speaker_embedding(
    audio_path: str | Path,
    cache_dir: str | Path,
    *,
    expected_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM,
    audio_root: str | Path | None = None,
) -> torch.Tensor:
    cache_path = speaker_embedding_path(audio_path, cache_dir, audio_root=audio_root)
    if not cache_path.is_file():
        raise SpeakerEmbeddingError(f"missing speaker embedding cache: {cache_path} (audio: {audio_path})")
    try:
        embedding = np.load(cache_path, allow_pickle=False)
    except Exception as error:
        raise SpeakerEmbeddingError(f"failed to read speaker embedding cache {cache_path}: {error}") from error
    validate_speaker_embedding_array(embedding, expected_dim=expected_dim, source=cache_path)
    return torch.from_numpy(embedding)
