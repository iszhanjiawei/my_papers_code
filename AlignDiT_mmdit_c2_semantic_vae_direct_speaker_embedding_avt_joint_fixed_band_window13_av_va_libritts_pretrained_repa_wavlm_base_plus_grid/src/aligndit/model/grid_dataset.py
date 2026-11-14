"""GRID caches with the original Direct-C2 feature and tokenizer contracts.

Only inventory validation changes: batch tensors and normalization follow the
copied Semantic-VAE dataset, without its CelebV-Dub-specific sample counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
    validate_repa_feature_array,
)
from aligndit.model.semantic_vae_dataset import (
    SEMANTIC_VAE_FEATURE,
    VIDEO_40HZ_FEATURE,
    SemanticVaeCelebVDubDataset,
    _ctc_lengths,
    _load_vocab,
    _read_json,
    _read_jsonl,
    _regular_file,
    _safe_join,
    sha256_file,
)
from aligndit.model.speaker_embedding import validate_speaker_embedding_array


NORMALIZATION_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
VOCAB_SHA256 = "225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d"
VAE_CHECKPOINT_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
SPEAKER_CHECKPOINT_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
SPEAKER_MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"


def _expect_fields(actual: dict, expected: dict, label: str) -> None:
    if not isinstance(actual, dict):
        raise TypeError(f"{label} must be a JSON object")
    for name, value in expected.items():
        if actual.get(name) != value:
            raise RuntimeError(f"{label} mismatch: {name}={actual.get(name)!r}, expected {value!r}")


class GridDataset(SemanticVaeCelebVDubDataset):
    """A full GRID train/val partition, never silently filtered for CTC.

    A small extraction marked ``selection.mode=smoke`` can only be consumed by
    diagnostics that explicitly pass ``allow_partial=True``. The training entry
    never passes that flag. Vocabulary and normalization stay pinned to the
    pretrained parent; they are not fitted to GRID.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        normalization_path: str | Path,
        vocab_path: str | Path,
        *,
        expected_normalization_sha256: str = NORMALIZATION_SHA256,
        expected_vocab_sha256: str = VOCAB_SHA256,
        expected_vae_checkpoint_sha256: str = VAE_CHECKPOINT_SHA256,
        expected_video_checkpoint_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_inventory_sha256: str | None = None,
        expected_record_count: int | None = None,
        speaker_embedding_cache_dir: str | Path | None = None,
        speaker_embedding_dim: int = 192,
        speaker_embedding_model_id: str = SPEAKER_MODEL_ID,
        speaker_embedding_checkpoint_sha256: str = SPEAKER_CHECKPOINT_SHA256,
        repa_feature_cache_dir: str | Path | None = None,
        repa_feature_dim: int = 768,
        repa_model_id: str = WAVLM_BASE_PLUS_MODEL_ID,
        repa_model_revision: str = WAVLM_BASE_PLUS_REVISION,
        repa_checkpoint_sha256: str = WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
        repa_teacher_layer: int = 12,
        allow_partial: bool = False,
    ):
        root = Path(cache_root).expanduser().absolute()
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"GRID cache root must be a regular directory: {root}")
        self.cache_root = root.resolve(strict=True)
        write_guard = root / "WRITE_ACTIVE.json"
        if write_guard.exists() or write_guard.is_symlink():
            raise RuntimeError(f"Refusing to read GRID cache while writer guard exists: {write_guard}")
        self.manifest_path = _regular_file(manifest_path, label="GRID manifest")
        if self.manifest_path.parent != self.cache_root / "manifests" or self.manifest_path.name not in {
            "train.jsonl",
            "val.jsonl",
        }:
            raise ValueError("GRID manifest must be cache_root/manifests/{train,val}.jsonl")
        self.split = self.manifest_path.stem
        meta_path = _regular_file(root / "manifests/inventory_meta.json", label="GRID inventory metadata")
        meta = _read_json(meta_path, label="GRID inventory metadata")
        _expect_fields(meta, {"schema_version": 1, "dataset": "GRID"}, "GRID inventory")
        selection = meta.get("selection")
        if selection != {"mode": "full"} and not (
            allow_partial
            and isinstance(selection, dict)
            and selection.get("mode") == "smoke"
            and type(selection.get("limit")) is int
            and selection["limit"] > 0
        ):
            raise RuntimeError("GRID training requires a full cache; smoke caches are diagnostic-only")
        completion_path = _regular_file(root / "complete.json", label="GRID completion marker")
        completion = _read_json(completion_path, label="GRID completion marker")
        _expect_fields(
            completion,
            {
                "schema_version": 1,
                "dataset": "GRID",
                "complete": True,
                "inventory_meta_sha256": sha256_file(meta_path),
                "missing": [],
                "invalid": [],
                "feature_contract": meta.get("feature_contract"),
            },
            "GRID completion marker",
        )

        manifests = meta.get("manifests")
        if not isinstance(manifests, dict):
            raise TypeError("GRID inventory metadata has no manifests mapping")
        all_records = {}
        for filename in ("inventory.jsonl", "train.jsonl", "val.jsonl"):
            path = _regular_file(root / "manifests" / filename, label=f"GRID {filename}")
            rows = _read_jsonl(path)
            _expect_fields(
                manifests.get(filename),
                {
                    "sha256": sha256_file(path),
                    "count": len(rows),
                    "size_bytes": path.stat().st_size,
                },
                f"GRID {filename}",
            )
            all_records[filename] = rows
        inventory = all_records["inventory.jsonl"]
        if not inventory or not all_records["train.jsonl"]:
            raise ValueError("GRID inventory and training split must be nonempty")
        inventory_sha = manifests["inventory.jsonl"]["sha256"]
        _expect_fields(
            completion,
            {
                "inventory_sha256": inventory_sha,
                "count": len(inventory),
                "split_counts": {split: len(all_records[f"{split}.jsonl"]) for split in ("train", "val")},
            },
            "GRID completion coverage",
        )
        selected_entry = manifests[self.manifest_path.name]
        for actual, expected, label in (
            (selected_entry["sha256"], expected_manifest_sha256, "manifest SHA256"),
            (inventory_sha, expected_inventory_sha256, "inventory SHA256"),
            (selected_entry["count"], expected_record_count, "record count"),
        ):
            if expected is not None and actual != expected:
                raise RuntimeError(f"GRID expected {label} mismatch: {actual!r} != {expected!r}")

        normalization_file = _regular_file(normalization_path, label="LibriSpeech normalization")
        vocab_file = _regular_file(vocab_path, label="pretrained vocabulary")
        for path, expected, field in (
            (normalization_file, expected_normalization_sha256, "normalization_sha256"),
            (vocab_file, expected_vocab_sha256, "vocab_sha256"),
        ):
            if sha256_file(path) != expected or meta.get(field) != expected:
                raise RuntimeError(f"GRID {field} differs from the pinned pretrained artifact")
        normalization = _read_json(normalization_file, label="LibriSpeech normalization")
        _expect_fields(
            normalization,
            {
                "channel_count": 64,
                "feature": SEMANTIC_VAE_FEATURE,
                "method": "per_channel_population_mean_std_float64_welford_v1",
                "scope": "train",
            },
            "LibriSpeech normalization",
        )
        mean = np.asarray(normalization.get("mean"), dtype=np.float32)
        std = np.asarray(normalization.get("std"), dtype=np.float32)
        if (
            mean.shape != (64,)
            or std.shape != (64,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError("Invalid per-channel Semantic-VAE normalization statistics")
        self.latent_mean, self.latent_std = mean, std
        vocabulary, _ = _load_vocab(vocab_file)

        inventory_by_key = {}
        for record in inventory:
            key = record.get("utterance_key")
            text, frames, split = record.get("text"), record.get("latent_frames"), record.get("split")
            if not isinstance(key, str) or key in inventory_by_key or split not in {"train", "val"}:
                raise ValueError(f"Invalid or duplicate GRID record: {key!r}")
            parts = key.split("/")
            if len(parts) != 3 or parts[0] != "grid" or any(part in {"", ".", ".."} for part in parts):
                raise ValueError(f"Invalid GRID utterance key: {key!r}")
            if not isinstance(text, str) or not text or any(char not in vocabulary for char in text):
                raise ValueError(f"GRID text must use the pretrained vocabulary: {key!r}")
            if type(frames) is not int or frames <= 0:
                raise ValueError(f"Invalid GRID latent length: {key!r}")
            target_length, repeats, min_frames = _ctc_lengths(text, vocabulary)
            suffix = f"{split}/{parts[1]}/{parts[2]}"
            _expect_fields(
                record,
                {
                    "latent_dim": 64,
                    "video_dim": 1024,
                    "audio_relative_path": suffix + ".wav",
                    "latent_relative_path": "latents/" + suffix + ".npy",
                    "video_40hz_relative_path": "video_40hz/" + suffix + ".npy",
                    "ctc_target_length": target_length,
                    "ctc_adjacent_repeats": repeats,
                    "ctc_min_input_frames": min_frames,
                    "ctc_feasible_40hz": frames >= min_frames,
                },
                f"GRID record {key}",
            )
            for field in ("audio_relative_path", "latent_relative_path", "video_40hz_relative_path"):
                _safe_join(self.cache_root, record[field], label=field)
            inventory_by_key[key] = record
        partition_keys = set()
        for split in ("train", "val"):
            for record in all_records[f"{split}.jsonl"]:
                key = record.get("utterance_key")
                if key in partition_keys or record.get("split") != split or inventory_by_key.get(key) != record:
                    raise ValueError(f"GRID split overlap or inventory mismatch: {key!r}")
                partition_keys.add(key)
        if partition_keys != set(inventory_by_key):
            raise ValueError("GRID train/val manifests do not partition the full inventory")

        features = meta.get("feature_contract")
        if not isinstance(features, dict):
            raise TypeError("GRID metadata has no feature contract")
        _expect_fields(
            features.get("latent"),
            {
                "dimension": 64,
                "dtype": "float32",
                "frame_rate_hz": 40.0,
                "feature": SEMANTIC_VAE_FEATURE,
                "checkpoint_sha256": expected_vae_checkpoint_sha256,
            },
            "GRID latent feature",
        )
        video_expected = {
            "dimension": 1024,
            "dtype": "float32",
            "frame_rate_hz": 40.0,
            "feature": VIDEO_40HZ_FEATURE,
        }
        if expected_video_checkpoint_sha256 is not None:
            video_expected["checkpoint_sha256"] = expected_video_checkpoint_sha256
        _expect_fields(features.get("video"), video_expected, "GRID video feature")
        video_sha = features["video"].get("checkpoint_sha256")
        if (
            not isinstance(video_sha, str)
            or len(video_sha) != 64
            or any(c not in "0123456789abcdef" for c in video_sha)
        ):
            raise ValueError("GRID video contract must pin its AV-HuBERT checkpoint SHA256")
        self.records = all_records[self.manifest_path.name]
        self.ctc_feasible_count = sum(row["ctc_feasible_40hz"] for row in self.records)
        self.ctc_infeasible_count = len(self.records) - self.ctc_feasible_count
        self.speaker_embedding_dim, self.repa_feature_dim = speaker_embedding_dim, repa_feature_dim
        self.speaker_embedding_cache_dir = None
        self.repa_feature_cache_dir = None
        self.speaker_embedding_contract = None
        self.repa_feature_contract = None
        for directory, feature_name, expected in (
            (
                speaker_embedding_cache_dir,
                "speaker",
                {
                    "dimension": speaker_embedding_dim,
                    "dtype": "float32",
                    "model_id": speaker_embedding_model_id,
                    "checkpoint_sha256": speaker_embedding_checkpoint_sha256,
                    "source_audio": "complete_unmasked_waveform",
                },
            ),
            (
                repa_feature_cache_dir,
                "repa",
                {
                    "dimension": repa_feature_dim,
                    "dtype": "float16",
                    "model_id": repa_model_id,
                    "model_revision": repa_model_revision,
                    "checkpoint_sha256": repa_checkpoint_sha256,
                    "teacher_layer": repa_teacher_layer,
                    "source_audio": "complete_unmasked_waveform",
                },
            ),
        ):
            if directory is None:
                continue
            feature_root = Path(directory).expanduser().absolute()
            expected_root = self.cache_root / ("speakers" if feature_name == "speaker" else "repa")
            if feature_root.is_symlink() or feature_root.resolve() != expected_root or not feature_root.is_dir():
                raise ValueError(f"GRID {feature_name} cache must be {expected_root}")
            _expect_fields(features.get(feature_name), expected, f"GRID {feature_name} feature")
            if feature_name == "speaker":
                if speaker_embedding_dim != 192:
                    raise ValueError("The CAM++ speaker condition requires 192 dimensions")
                self.speaker_embedding_cache_dir = expected_root
                self.speaker_embedding_contract = features[feature_name]
            else:
                if repa_feature_dim != 768:
                    raise ValueError("WavLM-Base+ REPA requires 768 dimensions")
                self.repa_feature_cache_dir = expected_root
                self.repa_feature_contract = features[feature_name]
        self.inventory_metadata = meta
        self.cache_contract = {
            "cache_root": str(self.cache_root),
            "split": self.split,
            "inventory_meta_sha256": sha256_file(meta_path),
            "complete_sha256": sha256_file(completion_path),
            "manifest_sha256": selected_entry["sha256"],
            "inventory_sha256": inventory_sha,
            "record_count": len(self.records),
            "feature_contract": features,
            "selection": selection,
        }

    def _feature_path(self, root: Path | None, record: dict, label: str) -> Path:
        if root is None:
            raise RuntimeError(f"GRID {label} cache is not configured")
        relative = Path(record["audio_relative_path"]).with_suffix(".npy")
        return _regular_file(_safe_join(root, str(relative), label=label), label=label)

    def _load_speaker_embedding_array(self, record: dict) -> np.ndarray:
        path = self._feature_path(self.speaker_embedding_cache_dir, record, "speaker embedding")
        embedding = np.load(path, allow_pickle=False)
        validate_speaker_embedding_array(embedding, expected_dim=self.speaker_embedding_dim, source=path)
        return embedding

    def _load_repa_feature_array(self, record: dict) -> np.ndarray:
        path = self._feature_path(self.repa_feature_cache_dir, record, "REPA feature")
        feature = np.load(path, allow_pickle=False)
        validate_repa_feature_array(feature, expected_dim=self.repa_feature_dim, source=path)
        return feature
