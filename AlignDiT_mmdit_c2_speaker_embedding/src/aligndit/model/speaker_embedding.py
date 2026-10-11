from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DEFAULT_SPEAKER_EMBEDDING_DIM = 192


class SpeakerEmbeddingError(RuntimeError):
    """Raised when a cached speaker embedding violates the experiment contract."""


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
