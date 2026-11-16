"""LRS3 input with frozen, dynamically sized split and cache receipts.

Pilot and full manifests use the same representation contract. A pilot receipt
only certifies its own inventory; it can never certify the full LRS3 inventory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from aligndit.lrs3_media_quality import evidence_paths, excluded_map, load_terminal, scope as media_quality_scope
from aligndit.lrs3_imported_source import (
    imported_source_scope, imported_source_artifacts, validate_import_completion,
)

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
from aligndit.model.lrs3_path_validation import path_checked_rows
from aligndit.model.speaker_embedding import (
    validate_speaker_cache_metadata,
    validate_speaker_embedding_array,
)


LATENT_SPEC = {
    "dimension": 64, "dtype": "float32", "frame_rate_hz": 40.0,
    "hop_length_samples": 400, "mode": "fixed_posterior_sample", "sample_rate": 16000,
}
SOURCE_KINDS = {
    "existing_transcript", "source_asr_consensus", "source_asr_mfa_consensus",
    "source_asr_ctc_consensus", "asr_pseudo",
}
TEXT_POLICIES = {"source_grounded", "source_plus_asr_pseudo"}


def _source_contract(root, meta):
    """Bind full-source scope to the canonical inventory, not accepted labels."""
    if meta["selection"] != {"mode": "full"}:
        return {}, []
    contract_path = _regular_file(root / "reports/source_contract.json", label="LRS3 source contract")
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    contract_digest = hashlib.sha256(contract_bytes).hexdigest()
    inventory_path = _regular_file(root / "manifests/source_inventory.jsonl", label="canonical LRS3 source inventory")
    inventory_digest = sha256_file(inventory_path)
    if (contract.get("schema_version") != 1
            or type(contract.get("source_media_pairs")) is not int or contract["source_media_pairs"] <= 0
            or type(contract.get("original_clip_count")) is not int
            or not 0 < contract["original_clip_count"] <= contract["source_media_pairs"]):
        raise RuntimeError("Invalid full LRS3 source contract")
    if (contract.get("source_inventory_sha256") != inventory_digest
            or meta.get("source_manifest_sha256") != inventory_digest
            or meta.get("source_contract_sha256") != contract_digest):
        raise RuntimeError("Full LRS3 source contract or canonical inventory hash mismatch")
    if (type(meta.get("source_pairs")) is not int
            or meta["source_pairs"] != contract["source_media_pairs"]
            or type(meta.get("source_original_ids")) is not int
            or meta["source_original_ids"] != contract["original_clip_count"]):
        raise RuntimeError("Full LRS3 source scope differs from its canonical source contract")
    return {
        "source_contract_sha256": contract_digest, "source_manifest_sha256": inventory_digest,
        "source_pairs": contract["source_media_pairs"], "source_original_ids": contract["original_clip_count"],
    }, [contract_path, inventory_path]


def _text_policy(root, manifest_dir, meta, expected):
    """Validate the label policy independently of the source-coverage scope."""
    if expected not in TEXT_POLICIES:
        raise ValueError(f"Unknown LRS3 text policy: {expected}")
    if "text_policy" not in meta:
        if meta["selection"] == {"mode": "pilot"} and expected == "source_grounded":
            return None, None, None, []
        raise RuntimeError("Full LRS3 publication requires an explicit text policy")
    path = _regular_file(manifest_dir / "text_policy.json", label="LRS3 text policy")
    fingerprint = sha256_file(path)
    policy = _read_json(path, label="LRS3 text policy")
    json.dumps(policy, allow_nan=False)
    if (meta["text_policy"] != expected or meta.get("text_policy_sha256") != fingerprint
            or policy.get("schema_version") != 1 or policy.get("policy") != expected
            or policy.get("version") != "lrs3_text_policy_v1"):
        raise RuntimeError("LRS3 text policy identity/hash mismatch")
    artifacts = [path]
    baseline = policy.get("strict_baseline")
    if expected == "source_grounded":
        if policy.get("pseudo_scope") != "none" or baseline is not None:
            raise RuntimeError("Strict LRS3 policy cannot authorize pseudo labels")
        return policy, fingerprint, None, artifacts
    if (policy.get("pseudo_scope") != "train_only" or not isinstance(baseline, dict)
            or baseline.get("manifest_dir") != "manifests/full"):
        raise RuntimeError("Pseudo LRS3 policy requires a pinned strict baseline and train-only scope")
    baseline_dir = _safe_join(root, baseline["manifest_dir"], label="strict LRS3 baseline")
    if baseline_dir.resolve() == manifest_dir.resolve():
        raise RuntimeError("Pseudo LRS3 publication must be separate from its strict baseline")
    for filename, field in (("inventory_meta.json", "inventory_meta_sha256"),
                            ("inventory.jsonl", "inventory_sha256"), ("train.jsonl", "train_sha256"),
                            ("valid.jsonl", "valid_sha256"), ("test.jsonl", "test_sha256"),
                            ("text_policy.json", "text_policy_sha256")):
        artifact = _regular_file(baseline_dir / filename, label="pinned strict LRS3 baseline")
        if sha256_file(artifact) != baseline.get(field):
            raise RuntimeError(f"Strict LRS3 baseline hash mismatch: {filename}")
        artifacts.append(artifact)
    strict_meta = _read_json(baseline_dir / "inventory_meta.json", label="strict baseline metadata")
    strict_policy = _read_json(baseline_dir / "text_policy.json", label="strict baseline text policy")
    if (strict_meta.get("status") != "complete" or strict_meta.get("selection") != {"mode": "full"}
            or strict_meta.get("text_policy") != "source_grounded"
            or strict_meta.get("text_policy_sha256") != baseline["text_policy_sha256"]
            or strict_policy.get("policy") != "source_grounded"
            or strict_policy.get("pseudo_scope") != "none" or strict_policy.get("strict_baseline") is not None):
        raise RuntimeError("Pseudo publication baseline is not a complete strict publication")
    for split in ("valid", "test"):
        if meta["manifests"][f"{split}.jsonl"]["sha256"] != baseline[f"{split}_sha256"]:
            raise RuntimeError(f"Pseudo publication changed the strict {split} split")
    return policy, fingerprint, baseline_dir, artifacts


def _validate_pseudo_label(row, root, evidence_hashes):
    if (row["split"] != "train" or row.get("independent_content_verified") is not False
            or not isinstance(row.get("acceptance_policy_version"), str)
            or not row["acceptance_policy_version"].strip()):
        raise ValueError("Pseudo labels require train-only scope and explicit unverified provenance")
    confidence = row.get("confidence")
    if (not isinstance(confidence, dict) or not confidence
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in confidence.values())):
        raise ValueError("Pseudo label confidence must contain finite numeric measurements")
    references = row.get("evidence_refs")
    if not isinstance(references, list) or not references:
        raise ValueError("Pseudo labels require hashed evidence references")
    for reference in references:
        if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                or not reference["path"] or not isinstance(reference.get("sha256"), str)):
            raise ValueError("Malformed pseudo-label evidence reference")
        path = Path(reference["path"])
        if not path.is_absolute():
            path = _safe_join(root, reference["path"], label="pseudo-label evidence")
        path = _regular_file(path, label="pseudo-label evidence")
        actual = evidence_hashes.get(path)
        if actual is None:
            actual = evidence_hashes[path] = sha256_file(path)
        if actual != reference["sha256"]:
            raise RuntimeError(f"Pseudo-label evidence hash mismatch: {path}")


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


def _validate_record_paths(root, row):
    # Preserve the original per-field identity and _safe_join order exactly.
    stem, key = row.get("relative_stem"), row.get("utterance_key")
    for field, prefix in (("latent_relative_path", "latents"),
                          ("video_40hz_relative_path", "video_40hz"),
                          ("speaker_relative_path", "speaker_embeddings")):
        expected_path = f"{prefix}/{stem}.npy"
        if row.get(field) != expected_path:
            raise ValueError(f"Invalid LRS3 {field}: {key}")
        _safe_join(root, expected_path, label=field)


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
        expected_text_policy: str = "source_grounded",
        speaker_embedding_dim: int = 192,
        speaker_embedding_cache_dir: str | Path | None = None,
        path_validation_workers: int = 16,
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
        policy, policy_digest, baseline_dir, self.text_policy_artifact_paths = _text_policy(
            self.cache_root, manifest_dir, meta, expected_text_policy,
        )
        source_contract, self.source_artifact_paths = _source_contract(self.cache_root, meta)
        terminal = load_terminal(self.cache_root, source_sha256=meta.get('source_manifest_sha256'))
        quality_scope = media_quality_scope(terminal)
        imported_source = terminal.get('_imported_source') if terminal else None
        import_scope = imported_source_scope(imported_source)
        imported_cache = (validate_import_completion(self.cache_root, {**imported_source, 'terminal': terminal})
                          if imported_source is not None else None)
        cache_scope = imported_cache['scope'] if imported_cache else None
        if meta.get('imported_source') != import_scope or meta.get('imported_cache') != cache_scope:
            raise RuntimeError('LRS3 publication does not bind its exact imported source/cache closure')
        self.imported_source_artifact_paths = imported_source_artifacts(imported_source)
        if imported_cache:self.imported_source_artifact_paths.extend(imported_cache['artifacts'])
        if meta.get('media_quality') != quality_scope:
            raise RuntimeError('LRS3 publication does not bind the fixed media quality closure')
        self.media_quality_exclusions = excluded_map(terminal)
        excluded_stems = {row['relative_stem'] for row in self.media_quality_exclusions.values()}
        self.media_quality_artifact_paths = evidence_paths(terminal)
        if baseline_dir is not None:
            baseline_meta = _read_json(baseline_dir / 'inventory_meta.json', label='Strict baseline metadata')
            if baseline_meta.get('media_quality') != quality_scope:
                raise RuntimeError('Strict/pseudo media quality closures differ')
            if (baseline_meta.get('imported_source') != import_scope
                    or baseline_meta.get('imported_cache') != cache_scope):
                raise RuntimeError('Strict/pseudo imported source/cache closures differ')
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
        kind_counts, pseudo_keys, evidence_hashes = {}, set(), {}
        split_counts, total_frames = {}, 0
        self.records = []
        for split in ("train", "valid", "test"):
            entry = entries[f"{split}.jsonl"]
            feasible_count, record_count = 0, 0
            source_kinds = Counter()
            with path_checked_rows(
                _iter_jsonl(manifest_dir / f"{split}.jsonl"),
                partial(_validate_record_paths, self.cache_root), workers=path_validation_workers,
            ) as checked_rows:
                for row, check_paths in checked_rows:
                    record_count += 1
                    if imported_cache is not None:
                        import_entry = imported_cache['receipts'].get(row.get('sample_id'))
                        if (import_entry is None or import_entry['relative_stem'] != row.get('relative_stem')
                                or import_entry['import_receipt_sha256'] != row.get('import_receipt_sha256')):
                            raise ValueError('Published LRS3 row differs from its immutable import receipt ledger')
                    if row.get('sample_id') in self.media_quality_exclusions or row.get('relative_stem') in excluded_stems:
                        raise ValueError('Published inventory contains a sealed media quality exclusion')
                    key, text, frames = row.get("utterance_key"), row.get("text"), row.get("latent_frames")
                    original, source = row.get("original_clip_id"), row.get("source_split")
                    if (row.get("split") != split or not isinstance(key, str) or key in inventory_rows
                            or not isinstance(original, str) or not original or source not in {"pretrain", "trainval", "test"}):
                        raise ValueError(f"Invalid or duplicate LRS3 record identity: {key!r}")
                    if (source == "test") != (split == "test"):
                        raise ValueError(f"Official LRS3 test membership changed: {key}")
                    kind = row.get("source_kind")
                    if policy is not None:
                        if kind not in SOURCE_KINDS:
                            raise ValueError(f"Unknown or missing LRS3 label source_kind: {key}")
                        if kind == "asr_pseudo":
                            if expected_text_policy != "source_plus_asr_pseudo":
                                raise ValueError("Strict LRS3 publication cannot contain pseudo labels")
                            _validate_pseudo_label(row, self.cache_root, evidence_hashes)
                            pseudo_keys.add(key)
                        source_kinds[kind] += 1
                    elif kind == "asr_pseudo":
                        raise ValueError("Legacy pilot cannot authorize pseudo labels")
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
                    check_paths()
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
            kind_counts[split] = dict(source_kinds)
            if (entry.get("ctc_feasible_count") != feasible_count
                    or entry.get("ctc_infeasible_count") != record_count - feasible_count):
                raise RuntimeError(f"LRS3 {split} CTC counts differ from metadata")
            if split == self.split:
                self.ctc_feasible_count = feasible_count
                self.ctc_infeasible_count = record_count - feasible_count
        if not self.records or (expected_record_count is not None and len(self.records) != expected_record_count):
            raise RuntimeError("LRS3 selected split is empty or differs from expected count")
        if policy is not None and meta.get("source_kind_counts") != kind_counts:
            raise RuntimeError("LRS3 label source-kind counts differ from metadata")
        if baseline_dir is not None:
            strict_count, strict_keys = 0, set()
            for row in _iter_jsonl(baseline_dir / "train.jsonl"):
                key = row.get("utterance_key")
                if (key in strict_keys or row.get("split") != "train"
                        or row.get("source_kind") not in SOURCE_KINDS - {"asr_pseudo"}
                        or inventory_rows.get(key) != _record_digest(row)):
                    raise RuntimeError(f"Pseudo publication removed or changed a strict training record: {key}")
                strict_keys.add(key)
                strict_count += 1
            if (strict_keys & pseudo_keys or split_counts["train"] != strict_count + len(pseudo_keys)):
                raise RuntimeError("Pseudo publication may only add explicit pseudo labels to strict training")
            del strict_keys
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
        if receipt.get('imported_source') != import_scope or receipt.get('imported_cache') != cache_scope:
            raise RuntimeError('LRS3 completion receipt differs from imported source/cache provenance')
        if receipt.get('media_quality') != quality_scope:
            raise RuntimeError('LRS3 completion receipt differs from the media quality closure')
        if (receipt.get("schema_version") != 1 or receipt.get("status") != "complete"
                or receipt.get("selection") != meta["selection"]
                or receipt.get("inventory_sha256") != digests["inventory.jsonl"]
                or receipt.get("count") != inventory_count):
            raise RuntimeError("LRS3 cache receipt does not certify this exact inventory")
        if policy is not None and (receipt.get("text_policy") != expected_text_policy
                or receipt.get("text_policy_sha256") != policy_digest):
            raise RuntimeError("LRS3 cache receipt does not certify this text policy")
        if source_contract and (
                receipt.get("source_contract_sha256") != source_contract["source_contract_sha256"]
                or type(receipt.get("source_original_ids")) is not int
                or receipt["source_original_ids"] != source_contract["source_original_ids"]):
            raise RuntimeError("LRS3 cache receipt does not certify the canonical source contract")
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
            **source_contract,
            **({"text_policy": expected_text_policy, "text_policy_sha256": policy_digest,
                "source_kind_counts": kind_counts, "strict_baseline": policy.get("strict_baseline")}
               if policy is not None else {}),
            "inventory_meta_sha256": meta_digest, "manifest_sha256": digests[self.manifest_path.name],
            "inventory_sha256": digests["inventory.jsonl"], "cache_completion_sha256": sha256_file(receipt_path),
            "normalization_sha256": expected_normalization_sha256, "vocab_sha256": vocab_digest,
            "split_counts": split_counts, "total_latent_frames": total_frames,
            "ctc_feasible_count": self.ctc_feasible_count, "ctc_infeasible_count": self.ctc_infeasible_count,
            "speaker": self.speaker_embedding_contract,
        }
        if quality_scope is not None:self.data_contract['media_quality'] = quality_scope
        if import_scope is not None:
            self.data_contract.update(imported_source=import_scope, imported_cache=cache_scope)
        # The launcher pins every unique evidence file; training constructors
        # verify hashes without retaining full transcript/probability payloads.
        self.text_evidence_artifacts = [
            {"path": str(path.resolve()), "sha256": fingerprint, "size_bytes": path.stat().st_size}
            for path, fingerprint in sorted(evidence_hashes.items())
        ]

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
