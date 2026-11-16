"""LRS3 input with frozen, dynamically sized split and cache receipts.

Pilot and full manifests use the same representation contract. A pilot receipt
only certifies its own inventory; it can never certify the full LRS3 inventory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from aligndit.model.semantic_vae_dataset import (
    CELEBVDUB_VIDEO_DIM,
    SEMANTIC_VAE_FEATURE,
    SEMANTIC_VAE_LATENT_DIM,
    VIDEO_40HZ_FEATURE,
    SemanticVaeCelebVDubDataset,
    _ctc_lengths,
    _load_vocab,
    _read_json,
    _regular_file,
    _safe_join,
    sha256_file,
)
from aligndit.model.speaker_embedding import (
    validate_speaker_cache_metadata,
    validate_speaker_embedding_array,
)


LATENT_SPEC = {
    "dimension": 64, "dtype": "float32", "frame_rate_hz": 40.0,
    "hop_length_samples": 400, "mode": "fixed_posterior_sample", "sample_rate": 16000,
}


def _validate_entry(path: Path, entry: dict) -> str:
    _regular_file(path, label="LRS3 frozen manifest")
    digest = sha256_file(path)
    if (
        not isinstance(entry, dict)
        or type(entry.get("count")) is not int
        or entry["count"] < 0
        or entry.get("sha256") != digest
        or entry.get("size_bytes") != path.stat().st_size
    ):
        raise RuntimeError(f"LRS3 manifest differs from inventory metadata: {path}")
    return digest


# Preserve only runtime fields in RAM. Frozen provenance remains in the hashed
# source JSONL; shared field-name strings prevent per-record key duplication.
RUNTIME_FIELDS = (
    "split", "source_split", "original_clip_id", "relative_stem", "utterance_key",
    "text", "latent_frames", "latent_dim", "video_dim", "latent_relative_path",
    "video_40hz_relative_path", "speaker_relative_path", "audio_relative_path",
    "ctc_target_length", "ctc_adjacent_repeats", "ctc_min_input_frames", "ctc_feasible_40hz",
)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"LRS3 record {number} is not an object: {path}")
            yield row


def _record_digest(row: dict) -> bytes:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).digest()


class SemanticVaeLRS3Dataset(SemanticVaeCelebVDubDataset):
    """Reuse the established tensor loading/collation, with LRS3 provenance.

    All split manifests and inventory membership are verified at construction.
    Actual FP32 tensor shape, finiteness and CAM++ norm are verified on every
    item read. The producer must write the completion receipt only after its
    full cache validation. CTC-infeasible examples retain diffusion supervision.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        normalization_path: str | Path,
        vocab_path: str | Path,
        *,
        expected_normalization_sha256: str,
        expected_vocab_sha256: str,
        speaker_embedding_model_id: str,
        speaker_embedding_checkpoint_sha256: str,
        expected_manifest_sha256: str | None = None,
        expected_inventory_sha256: str | None = None,
        expected_inventory_meta_sha256: str | None = None,
        expected_record_count: int | None = None,
        expected_selection: str = "full",
        speaker_embedding_dim: int = 192,
        speaker_embedding_cache_dir: str | Path | None = None,
    ):
        self.manifest_path = _regular_file(manifest_path, label="LRS3 split manifest")
        self.split = self.manifest_path.stem
        if self.manifest_path.name not in {"train.jsonl", "valid.jsonl", "test.jsonl"}:
            raise ValueError("LRS3 manifest must be train.jsonl, valid.jsonl, or test.jsonl")
        root = Path(cache_root).expanduser().absolute()
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"LRS3 cache root must be a regular directory: {root}")
        self.cache_root = root.resolve(strict=True)
        manifest_dir = self.manifest_path.parent
        meta_path = _regular_file(manifest_dir / "inventory_meta.json", label="LRS3 inventory metadata")
        meta_digest = sha256_file(meta_path)
        if expected_inventory_meta_sha256 and meta_digest != expected_inventory_meta_sha256:
            raise RuntimeError("LRS3 inventory metadata SHA256 mismatch")
        meta = _read_json(meta_path, label="LRS3 inventory metadata")
        if (
            meta.get("schema_version") != 1 or meta.get("dataset") != "LRS3"
            or meta.get("status") != "complete" or expected_selection not in {"pilot", "full"}
            or meta.get("selection") != {"mode": expected_selection}
            or meta.get("latent_spec") != LATENT_SPEC
        ):
            raise RuntimeError("LRS3 metadata selection, completion or latent contract mismatch")
        entries = meta.get("manifests")
        if not isinstance(entries, dict) or set(entries) != {
            "train.jsonl", "valid.jsonl", "test.jsonl", "inventory.jsonl"
        }:
            raise ValueError("LRS3 metadata requires train, valid, test and inventory manifests")
        digests = {name: _validate_entry(manifest_dir / name, entry) for name, entry in entries.items()}
        for expected, actual, label in (
            (expected_manifest_sha256, digests[self.manifest_path.name], "selected split"),
            (expected_inventory_sha256, digests["inventory.jsonl"], "inventory"),
        ):
            if expected and expected != actual:
                raise RuntimeError(f"LRS3 {label} SHA256 mismatch")

        norm_path = _regular_file(normalization_path, label="LibriSpeech latent normalization")
        if sha256_file(norm_path) != expected_normalization_sha256:
            raise RuntimeError("LibriSpeech latent normalization SHA256 mismatch")
        norm = _read_json(norm_path, label="LibriSpeech latent normalization")
        if (
            norm.get("channel_count") != 64 or norm.get("feature") != SEMANTIC_VAE_FEATURE
            or norm.get("method") != "per_channel_population_mean_std_float64_welford_v1"
            or norm.get("scope") != "train"
        ):
            raise RuntimeError("Invalid LibriSpeech normalization contract")
        mean, std = np.asarray(norm.get("mean"), dtype=np.float64), np.asarray(norm.get("std"), dtype=np.float64)
        if (mean.shape != (64,) or std.shape != (64,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Invalid Semantic-VAE channel statistics")
        self.latent_mean, self.latent_std = mean.astype(np.float32), std.astype(np.float32)
        vocab_file = _regular_file(vocab_path, label="LRS3 parent-compatible vocabulary")
        vocabulary, vocab_digest = _load_vocab(vocab_file)
        if vocab_digest != expected_vocab_sha256:
            raise RuntimeError("LRS3 vocabulary SHA256 mismatch")

        original_splits, inventory_rows = {}, {}
        split_counts, total_frames = {}, 0
        self.records = []
        for split in ("train", "valid", "test"):
            entry = entries[f"{split}.jsonl"]
            feasible_count, record_count = 0, 0
            for row in _iter_jsonl(manifest_dir / f"{split}.jsonl"):
                record_count += 1
                key, text, frames = row.get("utterance_key"), row.get("text"), row.get("latent_frames")
                original, source = row.get("original_clip_id"), row.get("source_split")
                if (row.get("split") != split or not isinstance(key, str) or key in inventory_rows
                        or not isinstance(original, str) or not original or source not in {"pretrain", "trainval", "test"}):
                    raise ValueError(f"Invalid or duplicate LRS3 record identity: {key!r}")
                if (source == "test") != (split == "test"):
                    raise ValueError(f"Official LRS3 test membership changed: {key}")
                original_key = (source, original)
                if original_splits.setdefault(original_key, split) != split:
                    raise ValueError(f"LRS3 original clip leaks across splits: {original_key}")
                if (not isinstance(text, str) or not text.strip() or type(frames) is not int or frames <= 0
                        or row.get("latent_dim") != 64 or row.get("video_dim") != 1024):
                    raise ValueError(f"Invalid LRS3 text/feature dimensions: {key}")
                unknown = sorted(set(text) - vocabulary.keys())
                if unknown:
                    raise ValueError(f"LRS3 text has out-of-vocabulary characters for {key}: {unknown!r}")
                stem = row.get("relative_stem")
                if not isinstance(stem, str) or not stem.startswith(f"{split}/{source}/") or key != f"lrs3/{stem}":
                    raise ValueError(f"LRS3 utterance key/path identity mismatch: {key}")
                for field, prefix in (("latent_relative_path", "latents"),
                                      ("video_40hz_relative_path", "video_40hz"),
                                      ("speaker_relative_path", "speaker_embeddings")):
                    expected_path = f"{prefix}/{stem}.npy"
                    if row.get(field) != expected_path:
                        raise ValueError(f"Invalid LRS3 {field}: {key}")
                    _safe_join(self.cache_root, expected_path, label=field)
                target, repeats, minimum = _ctc_lengths(text, vocabulary)
                feasible = frames >= minimum
                if (row.get("ctc_target_length") != target or row.get("ctc_adjacent_repeats") != repeats
                        or row.get("ctc_min_input_frames") != minimum or row.get("ctc_feasible_40hz") is not feasible):
                    raise ValueError(f"Invalid LRS3 40 Hz CTC metadata: {key}")
                feasible_count += int(feasible)
                total_frames += frames
                inventory_rows[key] = _record_digest(row)
                if split == self.split:
                    self.records.append({field: row[field] for field in RUNTIME_FIELDS})
            if record_count != entry["count"]:
                raise RuntimeError(f"LRS3 {split} record count differs from metadata")
            split_counts[split] = record_count
            if (entry.get("ctc_feasible_count") != feasible_count
                    or entry.get("ctc_infeasible_count") != record_count - feasible_count):
                raise RuntimeError(f"LRS3 {split} CTC counts differ from metadata")
            if split == self.split:
                self.ctc_feasible_count = feasible_count
                self.ctc_infeasible_count = record_count - feasible_count
        if not self.records or (expected_record_count is not None and len(self.records) != expected_record_count):
            raise RuntimeError("LRS3 selected split is empty or differs from expected count")
        inventory_count = 0
        for row in _iter_jsonl(manifest_dir / "inventory.jsonl"):
            inventory_count += 1
            key = row.get("utterance_key")
            if key not in inventory_rows or _record_digest(row) != inventory_rows.pop(key):
                raise ValueError(f"LRS3 inventory row differs from its split: {key}")
        if inventory_rows or inventory_count != entries["inventory.jsonl"]["count"]:
            raise RuntimeError("LRS3 inventory does not cover exactly the three splits")
        del original_splits, inventory_rows

        receipt_path = _regular_file(manifest_dir / "cache_completion.json", label="LRS3 cache completion receipt")
        receipt = _read_json(receipt_path, label="LRS3 cache completion receipt")
        if (receipt.get("schema_version") != 1 or receipt.get("status") != "complete"
                or receipt.get("selection") != meta["selection"]
                or receipt.get("inventory_sha256") != digests["inventory.jsonl"]
                or receipt.get("count") != inventory_count):
            raise RuntimeError("LRS3 cache receipt does not certify this exact inventory")
        features = receipt.get("features", {})
        for name, feature in (("latents", SEMANTIC_VAE_FEATURE), ("video_40hz", VIDEO_40HZ_FEATURE)):
            part = features.get(name, {})
            if (part.get("feature") != feature or part.get("count") != inventory_count
                    or part.get("total_frames") != total_frames):
                raise RuntimeError(f"LRS3 {name} cache completion mismatch")
        speaker_receipt = features.get("speaker_embeddings", {})
        if speaker_receipt.get("count") != inventory_count or speaker_receipt.get("dimension") != 192:
            raise RuntimeError("LRS3 speaker cache completion mismatch")
        for name in ("latents", "video_25hz", "video_40hz", "speaker_embeddings"):
            guard = self.cache_root / f"state/{name}/WRITE_ACTIVE.json"
            if guard.exists() or guard.is_symlink():
                raise RuntimeError(f"Cannot train while cache writer is active: {guard}")
        speaker_root = self.cache_root / "speaker_embeddings"
        if speaker_embedding_cache_dir is not None and Path(speaker_embedding_cache_dir).resolve() != speaker_root:
            raise ValueError("LRS3 speaker cache must be cache_root/speaker_embeddings")
        if speaker_embedding_dim != 192:
            raise ValueError("LRS3 requires the frozen 192-D CAM++ representation")
        speaker_meta_dir = Path(receipt.get("speaker_metadata_dir", manifest_dir / "speaker_contract")).resolve()
        speaker_meta_path = _regular_file(speaker_meta_dir / "metadata.json", label="LRS3 CAM++ metadata")
        speaker_coverage_path = _regular_file(speaker_meta_dir / "coverage_report.json", label="LRS3 CAM++ coverage")
        if (sha256_file(speaker_meta_path) != receipt.get("speaker_metadata_sha256")
                or sha256_file(speaker_coverage_path) != receipt.get("speaker_coverage_sha256")):
            raise RuntimeError("LRS3 speaker metadata differs from completion receipt")
        speaker_metadata = validate_speaker_cache_metadata(
            speaker_meta_dir, expected_dim=192, model_id=speaker_embedding_model_id,
            checkpoint_sha256=speaker_embedding_checkpoint_sha256,
        )
        speaker_coverage = _read_json(speaker_coverage_path, label="LRS3 CAM++ coverage")
        if (speaker_metadata.get("expected_count") != inventory_count
                or speaker_coverage.get("split_counts") != split_counts):
            raise RuntimeError("LRS3 speaker coverage differs from frozen inventory")
        self.speaker_embedding_cache_dir, self.speaker_embedding_dim = speaker_root, 192
        self.speaker_embedding_contract = {
            "cache_dir": str(speaker_root), "metadata_sha256": sha256_file(speaker_meta_path),
            "coverage_report_sha256": sha256_file(speaker_coverage_path),
            "model_id": speaker_embedding_model_id, "checkpoint_sha256": speaker_embedding_checkpoint_sha256,
            "dimension": 192, "source_audio": speaker_metadata["source_audio"], "split_counts": split_counts,
        }
        self.data_contract = {
            "dataset": "LRS3", "selection": meta["selection"], "split": self.split,
            "inventory_meta_sha256": meta_digest, "manifest_sha256": digests[self.manifest_path.name],
            "inventory_sha256": digests["inventory.jsonl"], "cache_completion_sha256": sha256_file(receipt_path),
            "normalization_sha256": expected_normalization_sha256, "vocab_sha256": vocab_digest,
            "split_counts": split_counts, "total_latent_frames": total_frames,
            "ctc_feasible_count": self.ctc_feasible_count, "ctc_infeasible_count": self.ctc_infeasible_count,
            "speaker": self.speaker_embedding_contract,
        }

    def _load_speaker_embedding_array(self, record: dict) -> np.ndarray:
        path = _safe_join(self.cache_root, record["speaker_relative_path"], label="LRS3 speaker embedding")
        _regular_file(path, label="LRS3 speaker embedding")
        embedding = np.load(path, allow_pickle=False)
        validate_speaker_embedding_array(embedding, expected_dim=192, source=path)
        return embedding

    def audit_cache(self) -> dict:
        """Read every tensor in this split without changing any cached data."""
        total_frames = 0
        for index in range(len(self)):
            sample = self[index]
            total_frames += sample["mel_spec"].shape[1]
        return {"complete": True, "split": self.split, "count": len(self),
                "total_frames": total_frames, "data_contract": self.data_contract}
