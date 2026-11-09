"""CPU integration checks for the GRID transfer; no optimizer updates or GPU use.

Run with ``PYTHONPATH=src python scripts/test_grid_transfer.py --real-parent``.
The optional parent check reads the pinned 5 GB LibriTTS artifact and verifies
all 802 online/EMA tensors, including the five vocabulary row mappings.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

import test_libritts_parent as parent_tests
from aligndit.model import libritts_pretrained_migration as migration
from aligndit.script.train.finetune_grid import (
    dataset_from_config,
    training_schedule_from_config,
    validate_grid_training_target,
)


PROJECT = Path(__file__).resolve().parents[1]
BASELINE_PROJECT = PROJECT.name.removesuffix("_grid")
BASELINE_REF = "5da4270"
REAL_PARENT = False


def config():
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        return compose(config_name="finetune_grid")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class GridIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.train = dataset_from_config(cls.config)
        val_cfg = copy.deepcopy(cls.config)
        val_cfg.datasets.manifest_path = str(Path(val_cfg.datasets.cache_root) / "manifests/val.jsonl")
        val_cfg.datasets.expected_manifest_sha256 = None
        val_cfg.datasets.expected_record_count = None
        cls.val = dataset_from_config(val_cfg)
        cls.schedule = training_schedule_from_config(cls.config, cls.train, world_size=4)

    def test_only_grid_adaptation_changes_resolved_configuration(self):
        parent_tests.BASELINE_PROJECT = BASELINE_PROJECT
        parent_tests.BASELINE_REF = BASELINE_REF
        baseline = parent_tests.baseline_config(parent_tests.CONFIG_NAME)
        baseline.pop("hydra", None)
        expected = OmegaConf.to_container(baseline, resolve=True)
        actual = OmegaConf.to_container(self.config, resolve=True)
        allowed = {
            "datasets.name", "datasets.cache_root", "datasets.manifest_path",
            "datasets.expected_manifest_sha256", "datasets.expected_inventory_sha256",
            "datasets.expected_record_count", "datasets.expected_vae_checkpoint_sha256",
            "datasets.expected_video_checkpoint_sha256", "datasets.speaker_embedding_cache_dir",
            "model.name", "ckpts.save_dir", "optim.epochs",
        }
        changes = set(parent_tests.differences(expected, actual))
        self.assertFalse(changes - allowed, f"Unexpected changes: {changes - allowed}")
        self.assertEqual(actual["model"]["arch"], expected["model"]["arch"])
        self.assertEqual(actual["ckpts"]["pretrained_path"], expected["ckpts"]["pretrained_path"])
        self.assertEqual(actual["optim"]["epochs"], 1000)
        self.assertEqual(actual["optim"]["run_until_update"], 200000)
        self.assertFalse(any("repa" in key.lower() for key in actual["model"]))

    def test_all_inherited_model_and_trainer_code_is_unchanged(self):
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASELINE_REF, "--",
             f"{BASELINE_PROJECT}/src/aligndit/model", f"{BASELINE_PROJECT}/src/f5_tts/model"],
            cwd=PROJECT.parent, text=True,
        ).splitlines()
        checked = []
        for original in paths:
            relative = original.removeprefix(BASELINE_PROJECT + "/")
            if not relative.endswith(".py") or relative.endswith("/libritts_pretrained_migration.py"):
                continue
            with self.subTest(file=relative):
                expected = subprocess.check_output(["git", "show", f"{BASELINE_REF}:{original}"],
                                                   cwd=PROJECT.parent)
                self.assertEqual((PROJECT / relative).read_bytes(), expected)
            checked.append(relative)
        self.assertGreater(len(checked), 20)
        self.assertIn("src/aligndit/model/trainer_semantic_vae_direct.py", checked)

    def test_real_partitions_and_normalized_collation_have_no_repa(self):
        self.assertEqual((len(self.train), len(self.val)), (29531, 3274))
        train_keys = {row["utterance_key"] for row in self.train.records}
        val_keys = {row["utterance_key"] for row in self.val.records}
        self.assertFalse(train_keys & val_keys)
        self.assertEqual(len(train_keys | val_keys), 32805)
        for dataset in (self.train, self.val):
            indices = (0, len(dataset) // 2, len(dataset) - 1)
            samples = [dataset[i] for i in indices]
            for index, sample in zip(indices, samples):
                record = dataset.records[index]
                raw = np.load(dataset.cache_root / record["latent_relative_path"], allow_pickle=False)
                expected = ((raw - dataset.latent_mean) / dataset.latent_std).T
                np.testing.assert_array_equal(sample["mel_spec"].numpy(), expected)
            batch = dataset.collate_fn(samples)
            self.assertEqual(set(batch), {"mel", "mel_lengths", "text", "text_lengths", "video",
                                         "video_lengths", "utterance_keys", "ctc_feasible", "speaker_embedding"})
            self.assertEqual(batch["mel"].shape[:2], (3, 64))
            self.assertEqual(batch["video"].shape[-1], 1024)
            self.assertEqual(batch["speaker_embedding"].shape, (3, 192))
            self.assertTrue(torch.equal(batch["mel_lengths"], batch["video_lengths"]))
            for key in ("mel", "video", "speaker_embedding"):
                self.assertTrue(torch.isfinite(batch[key]).all(), key)

    def test_schedule_reaches_200k_and_rejects_200_epochs_or_validation_training(self):
        self.assertEqual(self.schedule["frame_batches"], 985)
        self.assertEqual(self.schedule["per_rank_batches"], 246)
        self.assertEqual(self.schedule["effective_max_updates"], 246000)
        self.assertEqual(self.schedule["scheduler_total_steps"], 985000)
        self.assertEqual(self.schedule["scheduler_warmup_steps"], 80000)
        self.assertEqual(self.schedule["dropped_tail_batches_per_epoch"], 1)
        target = copy.deepcopy(self.config)
        target.datasets.expected_manifest_sha256 = self.train.cache_contract["manifest_sha256"]
        target.datasets.expected_inventory_sha256 = self.train.cache_contract["inventory_sha256"]
        target.datasets.expected_record_count = len(self.train)
        with tempfile.TemporaryDirectory(prefix="grid_test_") as directory:
            target.ckpts.save_dir = directory
            validate_grid_training_target(target, self.train, self.schedule)
            with self.assertRaisesRegex(ValueError, "train manifest"):
                validate_grid_training_target(target, self.val, self.schedule)
            (Path(directory) / "speaker_training_contract.json").write_text('{"config": {}}')
            with self.assertRaisesRegex(RuntimeError, "different experiment"):
                validate_grid_training_target(target, self.train, self.schedule)
        insufficient = copy.deepcopy(self.config)
        insufficient.optim.epochs = 200
        with self.assertRaisesRegex(ValueError, "49200 updates"):
            training_schedule_from_config(insufficient, self.train, world_size=4)

    def test_grid_posterior_and_completion_mismatches_are_rejected(self):
        data = OmegaConf.to_container(self.config.datasets, resolve=True)
        root = Path(data["cache_root"])
        metadata = json.loads((root / "manifests/inventory_meta.json").read_text())
        completion = json.loads((root / "complete.json").read_text())
        protocol = "semantic_vae_posterior_sample_v1"
        migration._validate_grid_representation(data, protocol)
        for label in ("posterior_logvar_clamp", "latent_normalized", "complete"):
            with self.subTest(field=label), tempfile.TemporaryDirectory() as directory:
                altered_meta, altered_complete = copy.deepcopy(metadata), copy.deepcopy(completion)
                if label == "complete":
                    altered_complete["complete"] = False
                else:
                    altered_meta["extraction_protocol"][label] = [-10, 10] if label == "posterior_logvar_clamp" else True
                    altered_meta["extraction_sha256"] = canonical_hash(altered_meta["extraction_protocol"])
                    altered_complete["extraction_sha256"] = altered_meta["extraction_sha256"]
                meta_path = Path(directory) / "manifests/inventory_meta.json"
                meta_path.parent.mkdir()
                meta_path.write_text(json.dumps(altered_meta))
                altered_complete["inventory_meta_sha256"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
                (Path(directory) / "complete.json").write_text(json.dumps(altered_complete))
                with self.assertRaises((ValueError, RuntimeError)):
                    migration._validate_grid_representation({**data, "cache_root": directory}, protocol)


class RealGridParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_PARENT:
            raise unittest.SkipTest("Pass --real-parent to verify the actual LibriTTS checkpoint on CPU")
        torch.set_num_threads(4)
        cls.config = config()
        cls.path = Path(cls.config.ckpts.pretrained_path)
        cls.checkpoint = torch.load(cls.path, map_location="cpu", weights_only=True, mmap=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "checkpoint"):
            del cls.checkpoint
            gc.collect()

    def test_grid_loads_all_802_online_and_ema_tensors_with_semantic_vocab_mapping(self):
        parent_tests.RealParentMigrationTests.test_all_online_and_ema_tensors_load_exactly_without_optimizer_resume(self)

    def test_wrong_parent_metadata_is_rejected_before_loading_weights(self):
        for field, value in (("update", 49999), ("training_contract_sha256", "0" * 64)):
            with self.subTest(field=field):
                corrupt = {**self.checkpoint, field: value}
                with patch.object(migration.torch, "load", return_value=corrupt):
                    with self.assertRaises(ValueError):
                        migration.load_libritts_ema_into_model(torch.nn.Module(), self.path, self.config,
                                                             verify_checkpoint_hash=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parent", action="store_true")
    args, remaining = parser.parse_known_args()
    REAL_PARENT = args.real_parent
    unittest.main(argv=[__file__, *remaining], verbosity=2)
