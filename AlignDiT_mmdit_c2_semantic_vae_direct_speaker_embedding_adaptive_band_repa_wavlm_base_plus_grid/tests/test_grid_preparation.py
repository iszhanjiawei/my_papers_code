"""Independent CPU checks for incomplete extraction, provenance and safe resume."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from aligndit.script.misc import prepare_grid as preparation


class GridPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grid-preparation-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "cache"
        self.root.mkdir()
        self.row = {
            "utterance_key": "grid/s1/bbaf2n",
            "split": "train",
            "source_sha256": "source-a",
            "original_num_samples": 1600,
            "latent_frames": 4,
            "audio_relative_path": "train/s1/bbaf2n.wav",
            "latent_relative_path": "latents/train/s1/bbaf2n.npy",
            "video_40hz_relative_path": "video_40hz/train/s1/bbaf2n.npy",
        }
        self.fingerprint = "extraction-a"
        self.metadata = {
            "schema_version": 1,
            "dataset": "GRID",
            "selection": {"mode": "smoke", "limit": 1},
            "manifests": {},
            "split_counts": {"train": 1},
            "feature_contract": {},
            "extraction_sha256": self.fingerprint,
        }
        for name, rows in (("inventory", [self.row]), ("train", [self.row]), ("val", [])):
            path = self.root / f"manifests/{name}.jsonl"
            preparation.atomic_bytes(path, preparation.jsonl_bytes(rows))
            self.metadata["manifests"][name + ".jsonl"] = {
                "sha256": preparation.sha256_file(path),
                "count": len(rows),
                "size_bytes": path.stat().st_size,
            }
        self._write_metadata()
        self.args = SimpleNamespace(output_dir=self.root, device="cpu")

    def _write_metadata(self):
        preparation.write_json(self.root / "manifests/inventory_meta.json", self.metadata, replace=True)

    def _make_stage(self, stage):
        path = preparation.artifact_path(self.root, self.row, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        if stage == "audio":
            sf.write(path, np.sin(np.arange(1600) * 0.1) * 0.2, 16000, subtype="PCM_16")
        else:
            values = {
                "latent": np.ones((4, 64), dtype=np.float32),
                "video": np.ones((4, 1024), dtype=np.float32),
                "speaker": np.eye(1, 192, dtype=np.float32).reshape(192),
                "repa": np.ones((4, 768), dtype=np.float16),
            }
            np.save(path, values[stage])
        preparation.publish_receipt(self.root, self.row, stage, self.fingerprint)
        return path

    def _make_complete(self):
        for stage in preparation.STAGES:
            self._make_stage(stage)
        self.assertTrue(preparation.verify([self.row], self.metadata, self.args, required=True))

    def test_incomplete_extraction_never_publishes_completion(self):
        self._make_stage("audio")
        self.assertFalse(preparation.verify([self.row], self.metadata, self.args, required=False))
        self.assertFalse((self.root / "complete.json").exists())
        report = json.loads((self.root / "coverage_report.json").read_text())
        self.assertEqual(len(report["missing"]), 4)
        self.assertEqual(report["invalid"], [])
        with self.assertRaisesRegex(RuntimeError, "missing=4"):
            preparation.verify([self.row], self.metadata, self.args, required=True)

    def test_missing_receipt_invalidates_even_an_existing_valid_artifact(self):
        self._make_complete()
        preparation.receipt_path(self.root, self.row, "repa").unlink()
        self.assertFalse(preparation.verify([self.row], self.metadata, self.args, required=False))
        self.assertFalse((self.root / "complete.json").exists())
        self.assertTrue(preparation.artifact_path(self.root, self.row, "repa").is_file())

    def test_shape_preserving_corruption_is_rejected_and_revokes_completion(self):
        self._make_complete()
        path = preparation.artifact_path(self.root, self.row, "latent")
        array = np.load(path)
        array[0, 0] += 0.5
        np.save(path, array)
        preparation.validate_artifact(path, self.row, "latent")
        with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
            preparation.validate_receipt(self.root, self.row, "latent", self.fingerprint)
        with self.assertRaisesRegex(RuntimeError, "invalid=1"):
            preparation.verify([self.row], self.metadata, self.args, required=True)
        self.assertFalse((self.root / "complete.json").exists())

    def test_features_cannot_follow_a_different_canonical_audio_receipt(self):
        self._make_complete()
        audio = preparation.artifact_path(self.root, self.row, "audio")
        sf.write(audio, np.zeros(1600), 16000, subtype="PCM_16")
        receipt_path = preparation.receipt_path(self.root, self.row, "audio")
        receipt = preparation.read_json_object(receipt_path)
        receipt["artifact_sha256"] = preparation.sha256_file(audio)
        preparation.write_json(receipt_path, receipt, replace=True)
        self.assertTrue(preparation.validate_receipt(self.root, self.row, "audio", self.fingerprint))
        for stage in ("latent", "speaker", "repa"):
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, "different canonical audio"):
                preparation.validate_receipt(self.root, self.row, stage, self.fingerprint)
        self.assertTrue(preparation.validate_receipt(self.root, self.row, "video", self.fingerprint))

    def test_changed_audio_bytes_are_detected_even_if_features_remain_unchanged(self):
        self._make_complete()
        audio = preparation.artifact_path(self.root, self.row, "audio")
        sf.write(audio, np.zeros(1600), 16000, subtype="PCM_16")
        self.assertFalse(preparation.verify([self.row], self.metadata, self.args, required=False))
        report = preparation.read_json_object(self.root / "coverage_report.json")
        self.assertEqual([record["stage"] for record in report["invalid"]], ["audio"])
        self.assertFalse((self.root / "complete.json").exists())

    def test_completed_receipts_reject_new_source_or_extraction_protocol(self):
        self._make_stage("audio")
        changed_source = dict(self.row, source_sha256="source-b")
        with self.assertRaisesRegex(RuntimeError, "Stale GRID cache receipt"):
            preparation.validate_receipt(self.root, changed_source, "audio", self.fingerprint)
        with self.assertRaisesRegex(RuntimeError, "Stale GRID cache receipt"):
            preparation.validate_receipt(self.root, self.row, "audio", "extraction-b")

    def test_completed_stage_resumes_without_loading_the_model_or_rewriting(self):
        self._make_stage("audio")
        latent = self._make_stage("latent")
        receipt = preparation.receipt_path(self.root, self.row, "latent")
        before = (latent.read_bytes(), latent.stat().st_mtime_ns, receipt.read_bytes())
        with patch.object(preparation, "load_semantic_vae", side_effect=AssertionError("model must not load")):
            preparation.run_stage("latent", [self.row], self.metadata, {}, self.args)
        self.assertEqual((latent.read_bytes(), latent.stat().st_mtime_ns, receipt.read_bytes()), before)

    def _configure_frozen_plan(self):
        from aligndit.script.misc.grid_sources import SPLIT_SUBDIR

        self.args.grid_root = self.root.parent / "raw_grid"
        self.args.limit = 1
        self.args.source_audit_dir = None
        self.args.semantic_vae_spec = self.root.parent / "vae_spec.json"
        self.args.semantic_vae_spec.write_text("{}\n")
        splits = {}
        for split in ("train", "val"):
            path = self.args.grid_root / SPLIT_SUBDIR / f"{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{split}/frozen/path\n")
            splits[split] = {"sha256": preparation.sha256_file(path)}
        assets = {"feature_contract": {}, "wavlm_files": {"pytorch_model.bin": "pinned-teacher"}}
        protocol = {
            "version": "grid_full_mpeg_audio_gray96_training_v1",
            "device_type": "cpu",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "semantic_vae_source_spec_sha256": preparation.sha256_file(self.args.semantic_vae_spec),
            "implementation_sha256": preparation.implementation_hashes(),
            "wavlm_files": assets["wavlm_files"],
        }
        self.metadata.update(
            {
                "normalization_sha256": preparation.NORMALIZATION_SHA256,
                "vocab_sha256": preparation.VOCAB_SHA256,
                "source_summary": {"dataset_root": str(self.args.grid_root.resolve()), "splits": splits},
                "extraction_protocol": protocol,
                "extraction_sha256": preparation.digest_object(protocol),
            }
        )
        self._write_metadata()
        return assets

    def test_frozen_plan_resume_validates_manifest_and_existing_source_split(self):
        from aligndit.script.misc.grid_sources import SPLIT_SUBDIR

        assets = self._configure_frozen_plan()
        with patch.object(preparation, "pinned_assets", return_value=({}, assets)):
            rows, metadata, _ = preparation.reuse_plan(self.args)
            self.assertEqual(rows, [self.row])
            self.assertEqual(metadata, self.metadata)
            manifest = self.root / "manifests/train.jsonl"
            original = manifest.read_bytes()
            manifest.write_bytes(original + b"\n")
            with self.assertRaisesRegex(RuntimeError, "manifest was modified"):
                preparation.reuse_plan(self.args)
            manifest.write_bytes(original)
            (self.args.grid_root / SPLIT_SUBDIR / "val.txt").write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "val source list changed"):
                preparation.reuse_plan(self.args)

    def test_frozen_plan_cannot_be_promoted_from_smoke_to_full(self):
        assets = self._configure_frozen_plan()
        self.args.limit = None
        with (
            patch.object(preparation, "pinned_assets", return_value=({}, assets)),
            self.assertRaisesRegex(RuntimeError, "dataset/selection/assets"),
        ):
            preparation.reuse_plan(self.args)

    def test_frozen_plan_rejects_changed_extraction_implementation(self):
        assets = self._configure_frozen_plan()
        changed_hashes = preparation.implementation_hashes()
        changed_hashes["script/misc/grid_sources.py"] = "0" * 64
        with (
            patch.object(preparation, "pinned_assets", return_value=({}, assets)),
            patch.object(preparation, "implementation_hashes", return_value=changed_hashes),
            self.assertRaisesRegex(RuntimeError, "extraction protocol/runtime differs"),
        ):
            preparation.reuse_plan(self.args)


if __name__ == "__main__":
    unittest.main()
