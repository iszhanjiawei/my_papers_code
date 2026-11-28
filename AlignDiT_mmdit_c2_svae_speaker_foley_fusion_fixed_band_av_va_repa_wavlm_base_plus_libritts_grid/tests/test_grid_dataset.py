"""GRID feature contracts, held-out split integrity and DDP schedule regressions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aligndit.model.grid_dataset import (
    SPEAKER_CHECKPOINT_SHA256,
    SPEAKER_MODEL_ID,
    VAE_CHECKPOINT_SHA256,
    GridDataset,
)
from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
)
from aligndit.model.semantic_vae_dataset import SEMANTIC_VAE_FEATURE, VIDEO_40HZ_FEATURE, _ctc_lengths, sha256_file


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")


class GridFixture:
    def __init__(self, root):
        self.root = root
        self.normalization = root / "normalization.json"
        write_json(
            self.normalization,
            {
                "channel_count": 64,
                "feature": SEMANTIC_VAE_FEATURE,
                "method": "per_channel_population_mean_std_float64_welford_v1",
                "scope": "train",
                "mean": [1.0] * 64,
                "std": [2.0] * 64,
            },
        )
        self.vocab = root / "vocab.txt"
        self.vocab.write_text(" \n" + "\n".join("abcdefghijklmnopqrstuvwxyz") + "\n", encoding="utf-8")
        vocabulary = {char: i for i, char in enumerate(self.vocab.read_text().splitlines())}
        self.rows = []
        for index, (split, frames) in enumerate((("train", 120), ("train", 12), ("val", 120))):
            suffix = f"{split}/s1/clip{index}"
            text = "bin blue at f two now"
            target, repeats, minimum = _ctc_lengths(text, vocabulary)
            row = {
                "utterance_key": f"grid/s1/clip{index}",
                "split": split,
                "text": text,
                "latent_frames": frames,
                "latent_dim": 64,
                "video_dim": 1024,
                "audio_relative_path": suffix + ".wav",
                "latent_relative_path": "latents/" + suffix + ".npy",
                "video_40hz_relative_path": "video_40hz/" + suffix + ".npy",
                "ctc_target_length": target,
                "ctc_adjacent_repeats": repeats,
                "ctc_min_input_frames": minimum,
                "ctc_feasible_40hz": frames >= minimum,
            }
            self.rows.append(row)
            for prefix, array in (
                ("latents", np.full((frames, 64), 3.0, dtype=np.float32)),
                ("video_40hz", np.ones((frames, 1024), dtype=np.float32)),
                ("speakers", np.eye(1, 192, dtype=np.float32).reshape(192)),
                ("repa", np.ones((frames + 5, 768), dtype=np.float16)),
            ):
                path = root / prefix / (suffix + ".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, array)
        self.metadata = {
            "schema_version": 1,
            "dataset": "GRID",
            "selection": {"mode": "full"},
            "normalization_sha256": sha256_file(self.normalization),
            "vocab_sha256": sha256_file(self.vocab),
            "feature_contract": {
                "latent": {
                    "dimension": 64,
                    "dtype": "float32",
                    "frame_rate_hz": 40.0,
                    "feature": SEMANTIC_VAE_FEATURE,
                    "checkpoint_sha256": VAE_CHECKPOINT_SHA256,
                },
                "video": {
                    "dimension": 1024,
                    "dtype": "float32",
                    "frame_rate_hz": 40.0,
                    "feature": VIDEO_40HZ_FEATURE,
                    "checkpoint_sha256": "a" * 64,
                },
                "speaker": {
                    "dimension": 192,
                    "dtype": "float32",
                    "model_id": SPEAKER_MODEL_ID,
                    "checkpoint_sha256": SPEAKER_CHECKPOINT_SHA256,
                    "source_audio": "complete_unmasked_waveform",
                },
                "repa": {
                    "dimension": 768,
                    "dtype": "float16",
                    "model_id": WAVLM_BASE_PLUS_MODEL_ID,
                    "model_revision": WAVLM_BASE_PLUS_REVISION,
                    "checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
                    "teacher_layer": 12,
                    "source_audio": "complete_unmasked_waveform",
                },
            },
        }
        self.publish()

    def publish(self, train=None, val=None):
        manifests = {}
        for name, rows in (
            ("inventory.jsonl", self.rows),
            ("train.jsonl", train if train is not None else [r for r in self.rows if r["split"] == "train"]),
            ("val.jsonl", val if val is not None else [r for r in self.rows if r["split"] == "val"]),
        ):
            path = self.root / "manifests" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifests[name] = {"sha256": sha256_file(path), "count": len(rows), "size_bytes": path.stat().st_size}
        self.metadata["manifests"] = manifests
        meta_path = self.root / "manifests/inventory_meta.json"
        write_json(meta_path, self.metadata)
        write_json(
            self.root / "complete.json",
            {
                "schema_version": 1,
                "dataset": "GRID",
                "complete": True,
                "inventory_meta_sha256": sha256_file(meta_path),
                "inventory_sha256": manifests["inventory.jsonl"]["sha256"],
                "count": len(self.rows),
                "split_counts": {split: manifests[f"{split}.jsonl"]["count"] for split in ("train", "val")},
                "feature_contract": self.metadata["feature_contract"],
                "missing": [],
                "invalid": [],
            },
        )

    def load(self, split="train", **kwargs):
        return GridDataset(
            self.root / f"manifests/{split}.jsonl",
            self.root,
            self.normalization,
            self.vocab,
            expected_normalization_sha256=sha256_file(self.normalization),
            expected_vocab_sha256=sha256_file(self.vocab),
            expected_video_checkpoint_sha256="a" * 64,
            speaker_embedding_cache_dir=self.root / "speakers",
            repa_feature_cache_dir=self.root / "repa",
            **kwargs,
        )


class GridDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grid-dataset-test-")
        self.addCleanup(temporary.cleanup)
        self.fixture = GridFixture(Path(temporary.name))

    def test_full_batch_contract_normalization_and_padding(self):
        dataset = self.fixture.load()
        self.assertEqual(len(dataset), 2)
        self.assertEqual((dataset.ctc_feasible_count, dataset.ctc_infeasible_count), (1, 1))
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["mel"].shape), (2, 64, 120))
        self.assertEqual(tuple(batch["video"].shape), (2, 120, 1024))
        self.assertEqual(tuple(batch["speaker_embedding"].shape), (2, 192))
        self.assertEqual(tuple(batch["repa_features"].shape), (2, 125, 768))
        self.assertEqual(batch["repa_features"].dtype, torch.float16)
        self.assertTrue(torch.equal(batch["mel_lengths"], batch["video_lengths"]))
        self.assertTrue(torch.all(batch["mel"][1, :, :12] == 1))
        self.assertTrue(torch.all(batch["mel"][1, :, 12:] == 0))
        self.assertEqual(batch["ctc_feasible"].tolist(), [True, False])

    def test_val_uses_held_out_conditions(self):
        dataset = self.fixture.load("val")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["utterance_key"], "grid/s1/clip2")
        self.assertEqual(tuple(dataset[0]["speaker_embedding"].shape), (192,))

    def test_smoke_cache_requires_explicit_diagnostic_opt_in(self):
        self.fixture.metadata["selection"] = {"mode": "smoke", "limit": 3}
        self.fixture.publish()
        with self.assertRaisesRegex(RuntimeError, "full cache"):
            self.fixture.load()
        self.assertEqual(len(self.fixture.load(allow_partial=True)), 2)

    def test_missing_completion_and_active_writer_fail_closed(self):
        marker = self.fixture.root / "WRITE_ACTIVE.json"
        marker.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "writer guard"):
            self.fixture.load()
        marker.unlink()
        (self.fixture.root / "complete.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "completion marker"):
            self.fixture.load()

    def test_manifest_and_metadata_tampering_detected(self):
        manifest = self.fixture.root / "manifests/train.jsonl"
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "train.jsonl mismatch"):
            self.fixture.load()
        self.fixture.publish()
        self.fixture.metadata["selection"] = {"mode": "smoke", "limit": 3}
        write_json(self.fixture.root / "manifests/inventory_meta.json", self.fixture.metadata)
        with self.assertRaisesRegex(RuntimeError, "inventory_meta_sha256"):
            self.fixture.load(allow_partial=True)

    def test_split_overlap_rejected_even_with_consistent_hashes(self):
        train = [self.fixture.rows[0], self.fixture.rows[0]]
        self.fixture.publish(train=train)
        with self.assertRaisesRegex(ValueError, "split overlap"):
            self.fixture.load()

    def test_checkpoint_and_ctc_metadata_are_verified(self):
        self.fixture.metadata["feature_contract"]["latent"]["checkpoint_sha256"] = "b" * 64
        self.fixture.publish()
        with self.assertRaisesRegex(RuntimeError, "latent feature mismatch"):
            self.fixture.load()
        self.fixture.metadata["feature_contract"]["latent"]["checkpoint_sha256"] = VAE_CHECKPOINT_SHA256
        self.fixture.rows[0]["ctc_min_input_frames"] += 1
        self.fixture.publish()
        with self.assertRaisesRegex(RuntimeError, "ctc_min_input_frames"):
            self.fixture.load()

    def test_cache_paths_cannot_escape_or_cross_splits(self):
        self.fixture.rows[0]["latent_relative_path"] = "../outside.npy"
        self.fixture.publish()
        with self.assertRaisesRegex(RuntimeError, "latent_relative_path"):
            self.fixture.load()

    def test_array_corruption_detected_on_access(self):
        dataset = self.fixture.load()
        path = self.fixture.root / self.fixture.rows[0]["video_40hz_relative_path"]
        np.save(path, np.full((120, 1024), float("nan"), dtype=np.float32))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            dataset[0]

    def test_wrong_speaker_norm_and_repa_dtype_rejected(self):
        dataset = self.fixture.load()
        path = self.fixture.root / "speakers/train/s1/clip0.npy"
        np.save(path, np.ones((192,), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "L2-normalized"):
            dataset[0]
        np.save(path, np.eye(1, 192, dtype=np.float32).reshape(192))
        np.save(self.fixture.root / "repa/train/s1/clip0.npy", np.ones((125, 768), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "float16"):
            dataset[0]

    def test_pretrained_normalization_and_vocab_cannot_be_replaced_silently(self):
        self.fixture.metadata["vocab_sha256"] = "b" * 64
        self.fixture.publish()
        with self.assertRaisesRegex(RuntimeError, "vocab_sha256"):
            self.fixture.load()


class GridScheduleTests(unittest.TestCase):
    def test_ddp_tail_has_equal_rank_lengths_and_no_silent_short_training(self):
        from accelerate.data_loader import BatchSamplerShard
        from omegaconf import OmegaConf
        from torch.utils.data import SequentialSampler

        from aligndit.script.train.finetune_grid import training_schedule_from_config
        from f5_tts.model.dataset import DynamicBatchSampler

        class TinyDataset:
            def __init__(self):
                self.records = [{"latent_frames": 4} for _ in range(9)]

            def __len__(self):
                return len(self.records)

            def get_frame_len(self, index):
                return self.records[index]["latent_frames"]

        dataset = TinyDataset()
        cfg = OmegaConf.create(
            {
                "seed": 666,
                "datasets": {"batch_size_type": "frame", "batch_size_per_gpu": 4, "max_samples": 32},
                "optim": {"grad_accumulation_steps": 1, "epochs": 10, "run_until_update": 20, "num_warmup_updates": 1},
            }
        )
        report = training_schedule_from_config(cfg, dataset, world_size=4)
        self.assertEqual(report["per_rank_batches"], 2)
        self.assertEqual(report["dropped_tail_batches_per_epoch"], 1)
        sampler = DynamicBatchSampler(SequentialSampler(dataset), 4, random_seed=666)
        lengths = [
            len(list(BatchSamplerShard(sampler, num_processes=4, process_index=rank, even_batches=False)))
            for rank in range(4)
        ]
        self.assertEqual(lengths, [2, 2, 2, 2])
        cfg.optim.run_until_update = 21
        with self.assertRaisesRegex(ValueError, "supports 20 updates"):
            training_schedule_from_config(cfg, dataset, world_size=4)
        cfg.datasets.batch_size_per_gpu = 3
        with self.assertRaisesRegex(ValueError, "silently discard"):
            training_schedule_from_config(cfg, dataset, world_size=4)


if __name__ == "__main__":
    unittest.main()
