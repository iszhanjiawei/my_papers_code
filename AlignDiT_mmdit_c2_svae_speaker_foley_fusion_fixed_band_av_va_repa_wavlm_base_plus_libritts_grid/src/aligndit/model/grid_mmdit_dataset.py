"""GRID data with immutable, complete acoustic/visual/teacher cache contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aligndit.model.repa import validate_repa_feature_array
from aligndit.model.semantic_vae_dataset import (
    SEMANTIC_VAE_FEATURE,
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


class GridMmditDataset(SemanticVaeCelebVDubDataset):
    """Keep the original sample/collation math with a separate GRID inventory.

    A subset is accepted only for explicit diagnostics. A published full cache
    must contain the existing baseline's exact 29,557 / 3,281 train/val split.
    """

    def __init__(
        self,
        cache_root,
        manifest_path,
        *,
        allow_partial=False,
        expected_manifest_sha256=None,
        expected_inventory_sha256=None,
        expected_record_count=None,
        expected_vae_checkpoint_sha256="7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516",
        expected_video_checkpoint_sha256=None,
        speaker_embedding_dim=192,
        speaker_embedding_cache_dir=None,
        repa_feature_dim=768,
        repa_feature_cache_dir=None,
        normalization_path=None,
        vocab_path=None,
        expected_normalization_sha256=None,
        expected_vocab_sha256=None,
        speaker_embedding_model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced",
        speaker_embedding_checkpoint_sha256="92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
        repa_model_id="microsoft/wavlm-base-plus",
        repa_model_revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
        repa_checkpoint_sha256="3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
        repa_teacher_layer=12,
    ):
        manifest_path = _regular_file(manifest_path, label="GRID manifest")
        split = manifest_path.stem
        if split not in {"train", "val"} or manifest_path.name != f"{split}.jsonl":
            raise ValueError(f"Unsupported GRID split: {split}")
        root = Path(cache_root).expanduser().absolute()
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"GRID cache must be a regular directory: {root}")
        self.cache_root = root.resolve(strict=True)
        if manifest_path.parent != self.cache_root / "manifests":
            raise ValueError("GRID manifest must belong to the selected cache")
        self.split = split
        contract_file = _regular_file(root / "data_contract.json", label="GRID data contract")
        contract = _read_json(contract_file, label="GRID data contract")
        complete = _read_json(_regular_file(root / "complete.json", label="GRID completion"), label="GRID completion")
        self.contract_sha256 = sha256_file(contract_file)
        self.data_contract = contract
        if (
            contract.get("schema_version") != 1
            or contract.get("dataset") != "GRID"
            or contract.get("complete") is not True
            or complete.get("schema_version") != 1
            or complete.get("complete") is not True
            or complete.get("contract_sha256") != self.contract_sha256
            or complete.get("source_and_feature_sha256_verified") is not True
            or complete.get("train_val_overlap") != 0
            or complete.get("split_counts") != contract.get("split_counts")
            or contract.get("source_audio") != "complete_unmasked_waveform"
        ):
            raise RuntimeError("GRID cache is incomplete or its completion/contract binding is invalid")
        if (root / "WRITE_ACTIVE.json").exists():
            raise RuntimeError("GRID cache still has an active writer")
        selection = contract.get("selection", {}).get("mode")
        if selection == "subset":
            if not allow_partial:
                raise RuntimeError("A diagnostic subset cannot be used for full GRID training")
        elif selection != "full" or contract.get("split_counts") != {"train": 29557, "val": 3281}:
            raise RuntimeError("Full GRID cache must preserve the baseline 29,557 / 3,281 split")
        self.is_subset = selection == "subset"
        expected_specs = {
            "latent_spec": {
                "dimension": 64,
                "dtype": "float32",
                "frame_rate_hz": 40.0,
                "hop_length_samples": 400,
                "mode": "fixed_posterior_sample",
                "sample_rate": 16000,
            },
            "video_spec": {
                "dimension": 1024,
                "dtype": "float32",
                "frame_rate_hz": 40.0,
                "interpolation": "linear_align_corners_false_exact_latent_length",
                "source": "AV-HuBERT_Large_video_only",
            },
            "speaker_spec": {
                "dimension": speaker_embedding_dim,
                "dtype": "float32",
                "model_id": speaker_embedding_model_id,
                "checkpoint_sha256": speaker_embedding_checkpoint_sha256,
            },
            "repa_spec": {
                "dimension": repa_feature_dim,
                "dtype": "float16",
                "model_id": repa_model_id,
                "model_revision": repa_model_revision,
                "checkpoint_sha256": repa_checkpoint_sha256,
                "teacher_layer": int(repa_teacher_layer),
            },
        }
        if expected_video_checkpoint_sha256 is not None:
            expected_specs["video_spec"]["checkpoint_sha256"] = expected_video_checkpoint_sha256
        if speaker_embedding_dim != 192 or repa_feature_dim != 768:
            raise ValueError("Expected 192-D CAM++ and 768-D WavLM-Base+ features")
        for name, expected in expected_specs.items():
            actual = contract.get(name, {})
            if any(actual.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"GRID cache violates {name}: expected {expected}, got {actual}")
        vae_contract = contract.get("semantic_vae_contract", {})
        if (
            vae_contract.get("checkpoint", {}).get("ema_sha256")
            != expected_vae_checkpoint_sha256
            or vae_contract.get("extraction", {}).get("protocol") != SEMANTIC_VAE_FEATURE
        ):
            raise RuntimeError("GRID latents must use the original Semantic-VAE 1000k EMA posterior-sample protocol")

        def artifact(name, configured_path, expected_sha):
            entry = contract.get(name, {})
            path = _regular_file(configured_path or entry.get("path", ""), label=f"GRID {name}")
            actual_sha = sha256_file(path)
            if actual_sha != entry.get("sha256") or (expected_sha is not None and actual_sha != expected_sha):
                raise RuntimeError(f"GRID {name} differs from its pinned artifact")
            return path

        normalization_file = artifact("normalization", normalization_path, expected_normalization_sha256)
        normalization = _read_json(normalization_file, label="LibriSpeech normalization")
        if (
            normalization.get("channel_count") != 64
            or normalization.get("feature") != SEMANTIC_VAE_FEATURE
            or normalization.get("method") != "per_channel_population_mean_std_float64_welford_v1"
            or normalization.get("scope") != "train"
        ):
            raise RuntimeError("GRID must reuse the pretrained LibriSpeech latent normalization")
        self.latent_mean = np.asarray(normalization.get("mean"), dtype=np.float32)
        self.latent_std = np.asarray(normalization.get("std"), dtype=np.float32)
        if (
            self.latent_mean.shape != (64,)
            or self.latent_std.shape != (64,)
            or not np.isfinite(self.latent_mean).all()
            or not np.isfinite(self.latent_std).all()
            or np.any(self.latent_std <= 0)
        ):
            raise ValueError("Invalid 64-channel normalization")
        vocab_file = artifact("vocab", vocab_path, expected_vocab_sha256)
        vocabulary, _ = _load_vocab(vocab_file)

        manifests = {}
        for name in ("train", "val", "inventory"):
            path = _regular_file(root / "manifests" / f"{name}.jsonl", label=f"GRID {name} manifest")
            entry = contract.get("manifests", {}).get(f"{name}.jsonl", {})
            rows = _read_jsonl(path)
            if sha256_file(path) != entry.get("sha256") or len(rows) != entry.get("count"):
                raise RuntimeError(f"GRID {name} manifest differs from its contract")
            expected_hash = {split: expected_manifest_sha256, "inventory": expected_inventory_sha256}.get(name)
            if expected_hash is not None and entry["sha256"] != expected_hash:
                raise ValueError(f"Configured GRID {name} manifest digest differs")
            if name == split and expected_record_count is not None and len(rows) != expected_record_count:
                raise ValueError(f"Configured GRID {name} record count differs")
            if name != "inventory" and len(rows) != contract["split_counts"].get(name):
                raise RuntimeError(f"GRID {name} split count mismatch")
            manifests[name] = rows
        all_rows = manifests["train"] + manifests["val"]
        keys = [row.get("utterance_key") for row in all_rows]
        # Use audio suffixes because split prefixes deliberately differ.
        ids = [tuple(Path(row.get("audio_relative_path", "")).parts[1:]) for row in all_rows]
        if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
            raise ValueError("GRID train/val overlap or duplicate utterance IDs")
        inventory = {row.get("utterance_key"): row for row in manifests["inventory"]}
        if len(inventory) != len(all_rows) or any(inventory.get(row["utterance_key"]) != row for row in all_rows):
            raise RuntimeError("GRID full inventory differs from train + val")
        for name in ("train", "val"):
            for row in manifests[name]:
                key, text, frames = row.get("utterance_key"), row.get("text"), row.get("latent_frames")
                if (
                    row.get("split") != name
                    or not isinstance(key, str)
                    or not key.startswith("grid/" + name + "/")
                    or not isinstance(text, str)
                    or not text
                    or type(frames) is not int
                    or frames <= 0
                    or row.get("latent_dim") != 64
                    or row.get("video_dim") != 1024
                ):
                    raise ValueError(f"Invalid GRID record: {key!r}")
                if any(character not in vocabulary for character in text):
                    raise ValueError(f"GRID text contains out-of-vocabulary characters: {key}")
                target, repeats, minimum = _ctc_lengths(text, vocabulary)
                if any(
                    row.get(k) != v
                    for k, v in {
                        "ctc_target_length": target,
                        "ctc_adjacent_repeats": repeats,
                        "ctc_min_input_frames": minimum,
                        "ctc_feasible_40hz": frames >= minimum,
                    }.items()
                ):
                    raise ValueError(f"Invalid GRID CTC lengths for {key}")
                parts = key.split("/")
                if len(parts) != 4 or any(part in {"", ".", ".."} for part in parts):
                    raise ValueError(f"Invalid GRID utterance key: {key!r}")
                suffix = "/".join(parts[1:])
                paths = {
                    "audio_relative_path": suffix + ".wav",
                    "latent_relative_path": f"latents/{suffix}.npy",
                    "video_40hz_relative_path": f"video_40hz/{suffix}.npy",
                    "speaker_relative_path": f"speaker_embeddings/{suffix}.npy",
                    "repa_relative_path": f"repa/{suffix}.npy",
                }
                if any(row.get(field) != value for field, value in paths.items()):
                    raise ValueError(f"GRID features do not all belong to {key}")
        self.records = manifests[split]
        self.manifest_path = root / "manifests" / f"{split}.jsonl"
        self.ctc_feasible_count = sum(row["ctc_feasible_40hz"] for row in self.records)
        self.ctc_infeasible_count = len(self.records) - self.ctc_feasible_count
        self.speaker_embedding_dim = 192
        self.repa_feature_dim = 768
        self.speaker_embedding_cache_dir = Path(speaker_embedding_cache_dir).resolve() if speaker_embedding_cache_dir else None
        self.repa_feature_cache_dir = Path(repa_feature_cache_dir).resolve() if repa_feature_cache_dir else None
        for path, subdir in ((self.speaker_embedding_cache_dir, "speaker_embeddings"), (self.repa_feature_cache_dir, "repa")):
            if path is not None and (path != self.cache_root / subdir or not path.is_dir()):
                raise ValueError(f"Expected GRID {subdir} directory under {self.cache_root}")
        self.speaker_embedding_contract = contract["speaker_spec"]
        self.repa_feature_contract = contract["repa_spec"]
        self.cache_contract = {
            "cache_format": "grid_mmdit_svae",
            "cache_root": str(self.cache_root),
            "contract_sha256": self.contract_sha256,
            "split": self.split,
            "manifest_sha256": contract["manifests"][f"{split}.jsonl"]["sha256"],
            "inventory_sha256": contract["manifests"]["inventory.jsonl"]["sha256"],
            "record_count": len(self.records),
            "selection": contract["selection"],
        }

    def _load_speaker_embedding_array(self, record):
        path = _regular_file(
            _safe_join(self.cache_root, record["speaker_relative_path"], label="speaker"),
            label="GRID speaker embedding",
        )
        embedding = np.load(path, allow_pickle=False)
        validate_speaker_embedding_array(embedding, source=path)
        return embedding

    def _load_repa_feature_array(self, record):
        path = _regular_file(
            _safe_join(self.cache_root, record["repa_relative_path"], label="REPA"), label="GRID REPA target"
        )
        feature = np.load(path, allow_pickle=False)
        validate_repa_feature_array(feature, source=path)
        return feature
