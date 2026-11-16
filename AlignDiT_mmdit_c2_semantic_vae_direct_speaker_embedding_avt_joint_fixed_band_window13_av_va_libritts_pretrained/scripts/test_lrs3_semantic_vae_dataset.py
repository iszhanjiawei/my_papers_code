"""CPU contract tests: reject split leakage, stale receipts and bad tensors."""

from __future__ import annotations

import json
import os
from collections import Counter
from copy import deepcopy
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
        selection = {"mode": getattr(self, "selection", "pilot")}
        self.manifests.mkdir(parents=True, exist_ok=True)
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
        policy_fields = {}
        if getattr(self, "policy", None) is not None:
            self.write(self.manifests / "text_policy.json", self.policy)
            policy_fields = {"text_policy": self.policy["policy"],
                             "text_policy_sha256": sha256_file(self.manifests / "text_policy.json")}
        self.write(self.manifests / "inventory_meta.json", {
            "schema_version": 1, "dataset": "LRS3", "status": "complete",
            "selection": selection, "latent_spec": LATENT_SPEC, "manifests": entries,
            **policy_fields,
            **({"source_kind_counts": {split: dict(Counter(row["source_kind"] for row in self.rows if row["split"] == split))
                                     for split in ("train", "valid", "test")}} if policy_fields else {}),
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
            "schema_version": 1, "status": "complete", "selection": selection,
            **policy_fields,
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
            expected_selection=getattr(self, "selection", "pilot"), speaker_embedding_model_id="fixture",
            speaker_embedding_checkpoint_sha256="fixture",
            expected_text_policy=self.policy["policy"] if getattr(self, "policy", None) else "source_grounded",
        )
        return SemanticVaeLRS3Dataset(**(kwargs | overrides))

    def strict_publication(self):
        self.selection = "full"
        self.manifests = self.root / "manifests/full"
        self.policy = {"schema_version": 1, "policy": "source_grounded", "version": "lrs3_text_policy_v1",
                       "pseudo_scope": "none", "strict_baseline": None}
        for row in self.rows:
            row["source_kind"] = "existing_transcript"
        self.freeze()

    def pseudo_publication(self):
        self.strict_publication()
        baseline = {"manifest_dir": "manifests/full"}
        for filename, field in (("inventory_meta.json", "inventory_meta_sha256"),
                                ("inventory.jsonl", "inventory_sha256"), ("train.jsonl", "train_sha256"),
                                ("valid.jsonl", "valid_sha256"), ("test.jsonl", "test_sha256"),
                                ("text_policy.json", "text_policy_sha256")):
            baseline[field] = sha256_file(self.manifests / filename)
        self.manifests = self.root / "manifests/full_pseudo"
        self.policy = {"schema_version": 1, "policy": "source_plus_asr_pseudo", "version": "lrs3_text_policy_v1",
                       "pseudo_scope": "train_only", "strict_baseline": baseline}
        row = deepcopy(self.rows[0])
        row.update(original_clip_id="five", relative_stem="train/pretrain/five", utterance_key="lrs3/train/pretrain/five",
                   source_kind="asr_pseudo", independent_content_verified=False, acceptance_policy_version="asr_policy_v1",
                   confidence={"min_word_probability": .8, "mean_word_probability": .9, "avg_logprob": -.1})
        evidence = self.root / "evidence/pseudo.json"
        self.write(evidence, {"independent_clip_decode": "aa"})
        row["evidence_refs"] = [{"path": str(evidence), "sha256": sha256_file(evidence)}]
        for field in ("latent_relative_path", "video_40hz_relative_path", "speaker_relative_path", "audio_relative_path"):
            old = row[field]
            row[field] = old.replace("/one.", "/five.")
            source = self.root / old
            if source.exists():
                (self.root / row[field]).write_bytes(source.read_bytes())
        self.rows.append(row)
        self.freeze()

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

    def test_full_requires_explicit_text_policy(self):
        self.selection = "full"
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, "explicit text policy"):
            self.dataset()

    def test_strict_full_policy_and_counts(self):
        self.strict_publication()
        data = self.dataset()
        self.assertEqual(data.data_contract["text_policy"], "source_grounded")
        self.assertEqual(data.data_contract["source_kind_counts"]["train"], {"existing_transcript": 2})
        self.assertEqual(data.audit_cache()["count"], 2)

    def test_strict_publication_cannot_relabel_pseudo_as_accepted(self):
        self.strict_publication()
        self.rows[0]["source_kind"] = "asr_pseudo"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "cannot contain pseudo"):
            self.dataset()

    def test_pseudo_publication_preserves_strict_rows_and_uses_all_existing_cache_checks(self):
        self.pseudo_publication()
        data = self.dataset()
        self.assertEqual(len(data), 3)
        self.assertEqual(data.data_contract["source_kind_counts"]["train"], {"existing_transcript": 2, "asr_pseudo": 1})
        self.assertEqual(len(data.text_evidence_artifacts), 1)
        self.assertNotIn("confidence", data.records[-1])
        self.assertEqual(data.audit_cache()["total_frames"], 10)
        with self.assertRaisesRegex(RuntimeError, "text policy identity"):
            self.dataset(expected_text_policy="source_grounded")
        np.save(self.root / self.rows[-1]["latent_relative_path"], np.full((4, 64), np.nan, dtype=np.float32))
        with self.assertRaises(FloatingPointError):
            data[-1]

    def test_pseudo_cannot_change_strict_training_text(self):
        self.pseudo_publication()
        self.rows[0]["text"] = "bb"
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, "changed a strict training record"):
            self.dataset()

    def test_pseudo_cannot_change_strict_evaluation(self):
        self.pseudo_publication()
        self.rows[2]["text"] = "bb"
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, "changed the strict valid"):
            self.dataset()

    def test_pseudo_requires_unchanged_evidence_and_finite_confidence(self):
        self.pseudo_publication()
        evidence = Path(self.rows[-1]["evidence_refs"][0]["path"])
        evidence.write_text("changed evidence\n")
        with self.assertRaisesRegex(RuntimeError, "evidence hash mismatch"):
            self.dataset()
        self.rows[-1]["evidence_refs"][0]["sha256"] = sha256_file(evidence)
        self.rows[-1]["confidence"]["mean_word_probability"] = float("nan")
        self.freeze()
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.dataset()

    def test_pseudo_cannot_be_claimed_independently_verified(self):
        self.pseudo_publication()
        self.rows[-1]["independent_content_verified"] = True
        self.freeze()
        with self.assertRaisesRegex(ValueError, "unverified provenance"):
            self.dataset()

    def test_pseudo_receipt_cannot_reuse_strict_policy_hash(self):
        self.pseudo_publication()
        path = self.manifests / "cache_completion.json"
        receipt = json.loads(path.read_text())
        receipt["text_policy_sha256"] = self.policy["strict_baseline"]["text_policy_sha256"]
        self.write(path, receipt)
        with self.assertRaisesRegex(RuntimeError, "does not certify this text policy"):
            self.dataset()

    def test_label_source_counts_are_recomputed(self):
        self.pseudo_publication()
        path = self.manifests / "inventory_meta.json"
        meta = json.loads(path.read_text())
        meta["source_kind_counts"]["train"]["asr_pseudo"] = 2
        self.write(path, meta)
        with self.assertRaisesRegex(RuntimeError, "source-kind counts"):
            self.dataset()

    def test_explicit_pseudo_config_keeps_training_semantics_and_separate_outputs(self):
        from argparse import Namespace
        from omegaconf import OmegaConf
        from launch_lrs3_training import load_config
        args = Namespace(cache_root=None, checkpoint_dir=None, run_until_update=None, epochs=None, variant="strict")
        strict = load_config(args)
        args.variant = "pseudo"
        pseudo = load_config(args)
        self.assertEqual(strict.datasets.expected_text_policy, "source_grounded")
        self.assertEqual(pseudo.datasets.expected_text_policy, "source_plus_asr_pseudo")
        self.assertTrue(pseudo.datasets.manifest_path.endswith("/manifests/full_pseudo/train.jsonl"))
        self.assertNotEqual(strict.ckpts.save_dir, pseudo.ckpts.save_dir)
        self.assertNotEqual(strict.model.name, pseudo.model.name)
        self.assertEqual(OmegaConf.to_container(strict.model.arch), OmegaConf.to_container(pseudo.model.arch))
        self.assertEqual(OmegaConf.to_container(strict.optim), OmegaConf.to_container(pseudo.optim))
        self.assertEqual(strict.ckpts.expected_parent_sha256, pseudo.ckpts.expected_parent_sha256)


if __name__ == "__main__":
    unittest.main()
