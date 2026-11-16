"""CPU contract tests: reject split leakage, stale receipts and bad tensors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aligndit.model.lrs3_semantic_vae_dataset import LATENT_SPEC, SemanticVaeLRS3Dataset
from aligndit.model.semantic_vae_dataset import SEMANTIC_VAE_FEATURE, VIDEO_40HZ_FEATURE, sha256_file


class LRS3ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifests = self.root / "manifests/pilot"
        self.manifests.mkdir(parents=True)
        self.vocab = self.root / "vocab.txt"
        self.vocab.write_text(" \na\nb\n")
        self.norm = self.root / "norm.json"
        self.write(self.norm, {
            "channel_count": 64, "feature": SEMANTIC_VAE_FEATURE,
            "method": "per_channel_population_mean_std_float64_welford_v1", "scope": "train",
            "mean": [1.0] * 64, "std": [2.0] * 64,
        })
        self.rows = []
        for split, source, sample, frames in (("train", "pretrain", "one", 4),
                ("train", "pretrain", "two", 2), ("valid", "trainval", "three", 3),
                ("test", "test", "four", 5)):
            stem = f"{split}/{source}/{sample}"
            row = {
                "split": split, "source_split": source, "original_clip_id": sample,
                "relative_stem": stem, "utterance_key": f"lrs3/{stem}",
                "text": "aa", "latent_frames": frames, "latent_dim": 64, "video_dim": 1024,
                "latent_relative_path": f"latents/{stem}.npy",
                "video_40hz_relative_path": f"video_40hz/{stem}.npy",
                "speaker_relative_path": f"speaker_embeddings/{stem}.npy",
                "audio_relative_path": f"audio/{stem}.wav",
                "ctc_target_length": 2, "ctc_adjacent_repeats": 1,
                "ctc_min_input_frames": 3, "ctc_feasible_40hz": frames >= 3,
            }
            self.rows.append(row)
            for field, value in (
                ("latent_relative_path", np.full((frames, 64), 3.0, dtype=np.float32)),
                ("video_40hz_relative_path", np.zeros((frames, 1024), dtype=np.float32)),
                ("speaker_relative_path", np.eye(1, 192, dtype=np.float32).reshape(192)),
            ):
                path = self.root / row[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, value)
        self.freeze()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def freeze(self):
        entries, counts = {}, {}
        for name in ("train", "valid", "test", "inventory"):
            rows = self.rows if name == "inventory" else [row for row in self.rows if row["split"] == name]
            path = self.manifests / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            entry = {"count": len(rows), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if name != "inventory":
                entry["ctc_feasible_count"] = sum(row["ctc_feasible_40hz"] for row in rows)
                entry["ctc_infeasible_count"] = len(rows) - entry["ctc_feasible_count"]
                counts[name] = len(rows)
            entries[path.name] = entry
        self.write(self.manifests / "inventory_meta.json", {
            "schema_version": 1, "dataset": "LRS3", "status": "complete",
            "selection": {"mode": "pilot"}, "latent_spec": LATENT_SPEC, "manifests": entries,
        })
        speaker_dir = self.manifests / "speaker_contract"
        self.write(speaker_dir / "metadata.json", {
            "schema_version": 1, "status": "complete", "output_shape": [192], "output_dtype": "float32",
            "source_audio": "complete_unmasked_waveform", "sample_rate": 16000, "fbank_dim": 80,
            "fbank_mean_normalization": "per_chunk_per_mel_over_time", "chunk_seconds": 10,
            "max_seconds": 90, "padding": "repeat_complete_utterance_to_chunk_multiple",
            "chunk_aggregation": "arithmetic_mean_then_l2_normalize", "model_id": "fixture",
            "checkpoint_sha256": "fixture", "coverage_report": "coverage_report.json", "expected_count": len(self.rows),
        })
        self.write(speaker_dir / "coverage_report.json", {
            "complete": True, "expected": len(self.rows), "valid": len(self.rows),
            "missing": [], "invalid": [], "extra": [], "split_counts": counts,
        })
        total = sum(row["latent_frames"] for row in self.rows)
        self.write(self.manifests / "cache_completion.json", {
            "schema_version": 1, "status": "complete", "selection": {"mode": "pilot"},
            "inventory_sha256": entries["inventory.jsonl"]["sha256"], "count": len(self.rows),
            "features": {
                "latents": {"feature": SEMANTIC_VAE_FEATURE, "count": len(self.rows), "total_frames": total},
                "video_40hz": {"feature": VIDEO_40HZ_FEATURE, "count": len(self.rows), "total_frames": total},
                "speaker_embeddings": {"count": len(self.rows), "dimension": 192},
            },
            "speaker_metadata_sha256": sha256_file(speaker_dir / "metadata.json"),
            "speaker_coverage_sha256": sha256_file(speaker_dir / "coverage_report.json"),
        })

    def dataset(self, **overrides):
        kwargs = dict(
            manifest_path=self.manifests / "train.jsonl", cache_root=self.root,
            normalization_path=self.norm, vocab_path=self.vocab,
            expected_normalization_sha256=sha256_file(self.norm), expected_vocab_sha256=sha256_file(self.vocab),
            expected_selection="pilot", speaker_embedding_model_id="fixture",
            speaker_embedding_checkpoint_sha256="fixture",
        )
        return SemanticVaeLRS3Dataset(**(kwargs | overrides))

    def test_load_normalize_pad_and_keep_ctc_infeasible(self):
        data = self.dataset()
        self.assertEqual((len(data), data.ctc_feasible_count, data.ctc_infeasible_count), (2, 1, 1))
        batch = data.collate_fn([data[0], data[1]])
        self.assertEqual(tuple(batch["mel"].shape), (2, 64, 4))
        self.assertEqual(tuple(batch["video"].shape), (2, 4, 1024))
        self.assertTrue(bool((batch["mel"][0] == 1).all()))
        self.assertEqual(batch["ctc_feasible"].tolist(), [True, False])
        self.assertEqual(data.audit_cache()["total_frames"], 6)

    def test_pilot_cannot_certify_full(self):
        with self.assertRaisesRegex(RuntimeError, "selection"):
            self.dataset(expected_selection="full")

    def test_original_id_cannot_cross_splits(self):
        self.rows[2]["source_split"] = "pretrain"
        self.rows[2]["original_clip_id"] = self.rows[0]["original_clip_id"]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            self.dataset()

    def test_modified_manifest_rejected(self):
        with (self.manifests / "train.jsonl").open("a") as file:
            file.write("\n")
        with self.assertRaisesRegex(RuntimeError, "differs from inventory metadata"):
            self.dataset()

    def test_stale_completion_rejected(self):
        path = self.manifests / "cache_completion.json"
        value = json.loads(path.read_text())
        value["inventory_sha256"] = "bad"
        self.write(path, value)
        with self.assertRaisesRegex(RuntimeError, "exact inventory"):
            self.dataset()

    def test_unknown_text_rejected(self):
        self.rows[0]["text"] = "a?"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            self.dataset()

    def test_bad_ctc_metadata_rejected(self):
        self.rows[0]["ctc_min_input_frames"] = 2
        self.freeze()
        with self.assertRaisesRegex(ValueError, "CTC metadata"):
            self.dataset()

    def test_nonfinite_features_rejected_on_read(self):
        data = self.dataset()
        np.save(self.root / self.rows[0]["latent_relative_path"], np.full((4, 64), np.nan, dtype=np.float32))
        with self.assertRaises(FloatingPointError):
            data[0]

    def test_misaligned_feature_shape_rejected(self):
        data = self.dataset()
        np.save(self.root / self.rows[0]["video_40hz_relative_path"], np.zeros((3, 1024), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Invalid video"):
            data[0]

    def test_bad_speaker_norm_rejected(self):
        data = self.dataset()
        np.save(self.root / self.rows[0]["speaker_relative_path"], np.zeros(192, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "L2-normalized"):
            data[0]

    def test_active_writer_rejected(self):
        self.write(self.root / "state/latents/WRITE_ACTIVE.json", {})
        with self.assertRaisesRegex(RuntimeError, "writer is active"):
            self.dataset()

    def test_resolved_lrs3_config_retains_fixed_band_and_parent_pins(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from aligndit.model.fixed_temporal_band_contract import validate_fixed_band_training_config
        project = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base="1.3", config_dir=str(project / "src/aligndit/config")):
            config = compose(config_name="finetune_lrs3_svae_speaker_window13_libritts")
            baseline = compose(config_name="finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained")
        self.assertTrue(validate_fixed_band_training_config(config))
        self.assertEqual(OmegaConf.to_container(config.model.arch), OmegaConf.to_container(baseline.model.arch))
        for name in ("pretrained_path", "expected_parent_sha256", "expected_parent_size", "expected_parent_update"):
            self.assertEqual(config.ckpts[name], baseline.ckpts[name])
        for name in ("expected_normalization_sha256", "expected_vocab_sha256", "speaker_embedding_checkpoint_sha256"):
            self.assertEqual(config.datasets[name], baseline.datasets[name])

    def test_root_prefix_and_explicit_cache_override(self):
        from hydra import compose, initialize_config_dir
        project = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base="1.3", config_dir=str(project / "src/aligndit/config")):
            config = compose(config_name="finetune_lrs3_svae_speaker_window13_libritts")
        with patch.dict(os.environ, {"ROOT_PREFIX": "/mounted"}, clear=True):
            self.assertEqual(config.datasets.cache_root, "/mounted/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1")
            self.assertTrue(config.datasets.manifest_path.endswith("/manifests/full/train.jsonl"))
        with patch.dict(os.environ, {"ROOT_PREFIX": "/mounted", "LRS3_CACHE_ROOT": "/custom/lrs3"}, clear=True):
            self.assertEqual(config.datasets.cache_root, "/custom/lrs3")

    def test_heavy_provenance_is_verified_but_not_retained_in_runtime_rows(self):
        self.rows[0]["large_provenance"] = "evidence " * 1000
        self.freeze()
        data = self.dataset()
        self.assertNotIn("large_provenance", data.records[0])
        inventory = self.manifests / "inventory.jsonl"
        changed = [dict(row) for row in self.rows]
        changed[0]["large_provenance"] = "different evidence"
        inventory.write_text("".join(json.dumps(row) + "\n" for row in changed))
        meta_path = self.manifests / "inventory_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["manifests"]["inventory.jsonl"].update(sha256=sha256_file(inventory), size_bytes=inventory.stat().st_size)
        self.write(meta_path, meta)
        with self.assertRaisesRegex(ValueError, "inventory row differs"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
