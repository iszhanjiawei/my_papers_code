"""Read the existing audio-only AV-HuBERT targets without an online encoder.

The cache identity and entry format match the original AV-HuBERT dual-role
experiment. Missing or stale entries are errors: this module never creates
directories, writes targets, imports fairseq, or loads a teacher checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_AUDIO_TEACHER_IDENTITY = "aa9876bf51d0af280b87c575d53b8c0e408f1e3e677e74d341d45673f6b2a770"
AUDIO_TEACHER_DIM = 1024
AUDIO_TEACHER_SAMPLE_RATE = 16_000


def identity_key(identity: dict) -> str:
    """Keep the donor cache's JSON serialization, including its whitespace."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def waveform_identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio teacher source is not a file: {resolved}")
    stat = resolved.stat()
    return {"path": str(resolved), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def teacher_frame_lengths(num_samples: int) -> tuple[int, int]:
    """Return stored frame count and the prefix without preprocessing padding.

    python_speech_features uses 400-sample windows, a 160-sample hop and
    ceil-based tail padding. Four consecutive filterbank frames form one
    teacher token. Token j therefore has nominal source support
    [640*j, 640*j + 880); this describes preprocessing, not the contextual
    Transformer's receptive field. No teacher output is temporally stretched.
    """
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError(f"Expected a positive integer waveform length, got {num_samples!r}")
    filterbank_frames = 1 + max(0, (num_samples - 400 + 159) // 160)
    stored_frames = (filterbank_frames + 3) // 4
    valid_frames = max(0, (num_samples - 880) // 640 + 1)
    return stored_frames, min(stored_frames, valid_frames)


class AudioTeacherCache:
    """Strict cache-only reader for original-GT-waveform teacher features."""

    def __init__(
        self,
        cache_dir: str | Path,
        audio_root: str | Path,
        *,
        expected_identity: str = DEFAULT_AUDIO_TEACHER_IDENTITY,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=True)
        self.audio_root = Path(audio_root).expanduser().resolve(strict=True)
        if not self.cache_dir.is_dir() or not self.audio_root.is_dir():
            raise NotADirectoryError("Audio teacher cache and source audio roots must be directories")
        metadata_path = self.cache_dir / "teacher_metadata.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing regular audio teacher metadata: {metadata_path}")
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise TypeError(f"Audio teacher metadata must be an object: {metadata_path}")
        actual_identity = identity_key(metadata)
        if actual_identity != expected_identity or self.cache_dir.name != expected_identity:
            raise RuntimeError(
                "Audio teacher namespace/metadata identity mismatch: "
                f"expected={expected_identity}, actual={actual_identity}, directory={self.cache_dir.name}"
            )
        required = {
            "format_version": 1,
            "preprocessing": "avhubert_audio_only_pcm_logfbank26_stack4_v1",
            "sample_rate_hz": AUDIO_TEACHER_SAMPLE_RATE,
            "filterbank_bins": 26,
            "stack_order_audio": 4,
            "normalize_per_frame": True,
            "source_modality": "audio_only",
            "output_layer": "final_contextual",
            "feature_dim": AUDIO_TEACHER_DIM,
            "cache_dtype": "float16",
        }
        for name, value in required.items():
            if metadata.get(name) != value:
                raise ValueError(f"Unexpected audio teacher {name}: {metadata.get(name)!r} != {value!r}")
        self.teacher_identity = actual_identity
        self.contract = {
            "cache_dir": str(self.cache_dir),
            "audio_root": str(self.audio_root),
            "teacher_identity": actual_identity,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "metadata": metadata,
            "cache_only": True,
            "source_audio": "original_ground_truth_waveform",
            "frame_rate_hz": 25,
            "output_dtype": "float32",
            "valid_frame_rule": "stack4_support_640j_to_640j_plus_880_within_original_samples",
        }

    def entry_for_record(self, record: dict) -> tuple[Path, dict, int, int]:
        """Resolve and authenticate the source identity without reading its PCM."""
        key = record.get("utterance_key", "<unknown>")
        relative_value = record.get("audio_relative_path")
        if not isinstance(relative_value, str):
            raise TypeError(f"Missing audio_teacher source path for {key}")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix.lower() != ".wav"
        ):
            raise ValueError(f"Invalid audio teacher source path for {key}: {relative_value!r}")
        audio_path = (self.audio_root / relative).resolve(strict=True)
        try:
            audio_path.relative_to(self.audio_root)
        except ValueError as error:
            raise ValueError(f"Audio teacher source escapes audio root for {key}: {audio_path}") from error
        if (
            record.get("sample_rate") != AUDIO_TEACHER_SAMPLE_RATE
            or record.get("source_sample_rate", AUDIO_TEACHER_SAMPLE_RATE) != AUDIO_TEACHER_SAMPLE_RATE
        ):
            raise ValueError(f"Audio teacher manifest must describe original 16 kHz audio for {key}")
        num_samples = record.get("original_num_samples")
        expected_frames, valid_frames = teacher_frame_lengths(num_samples)
        if record.get("source_num_samples", num_samples) != num_samples:
            raise ValueError(f"Audio teacher original/source waveform lengths differ for {key}")
        audio_identity = waveform_identity(audio_path)
        cache_key = identity_key(audio_identity)
        path = self.cache_dir / cache_key[:2] / f"{cache_key}.npz"
        return path, audio_identity, expected_frames, valid_frames

    def load(self, record: dict) -> tuple[np.ndarray, int]:
        """Return float32 [T25,1024] features and the valid (unpadded) prefix."""
        key = record.get("utterance_key", "<unknown>")
        path, audio_identity, expected_frames, valid_frames = self.entry_for_record(record)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Missing or stale audio teacher cache for {key}: {path}; "
                "cache-only training never re-extracts teacher features"
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data["audio_identity"].item())) != audio_identity:
                    raise ValueError("original waveform identity mismatch")
                if str(data["teacher_identity"].item()) != self.teacher_identity:
                    raise ValueError("teacher identity mismatch")
                features = data["features"]
                if features.shape != (expected_frames, AUDIO_TEACHER_DIM):
                    raise ValueError(f"feature shape {features.shape} != {(expected_frames, AUDIO_TEACHER_DIM)}")
                if features.dtype != np.float16 or not np.isfinite(features).all():
                    raise ValueError("teacher features must contain finite float16 values")
                features = features.astype(np.float32)
        except (ValueError, KeyError, OSError, EOFError, TypeError, zipfile.BadZipFile) as error:
            raise RuntimeError(
                f"Invalid audio teacher cache for {key}: {path}: {error}; "
                "cache-only training never re-extracts teacher features"
            ) from error
        return features, valid_frames
