"""GRID split/corruption guards and checkpoint state restoration on CPU."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from aligndit.model.grid_semantic_vae_dataset import GridSemanticVaeDataset
from aligndit.model.semantic_vae_dataset import SEMANTIC_VAE_FEATURE, sha256_file
from aligndit.model.trainer_grid_semantic_vae import GridEvenBatchSampler, GridSemanticVaeTrainer
from aligndit.model.trainer_vt import Trainer_VT


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def fixture(root):
    vocab = root / "vocab.txt"
    vocab.write_text(" \na\n")
    norm = root / "normalization.json"
    write_json(
        norm,
        {
            "channel_count": 64,
            "feature": SEMANTIC_VAE_FEATURE,
            "method": "per_channel_population_mean_std_float64_welford_v1",
            "scope": "train",
            "mean": [2.0] * 64,
            "std": [2.0] * 64,
        },
    )
    rows = []
    for split, clip, frames in (("train", "clip74", 119), ("train", "clip75", 120), ("val", "heldout", 120)):
        relative = f"{split}/s1/{clip}.npy"
        row = {
            "utterance_key": f"grid/{split}/s1/{clip}",
            "split": split,
            "text": "a a",
            "latent_frames": frames,
            "latent_dim": 64,
            "video_dim": 1024,
            "audio_relative_path": f"{split}/s1/{clip}.wav",
            "latent_relative_path": f"latents/{relative}",
            "video_40hz_relative_path": f"video_40hz/{relative}",
            "speaker_relative_path": f"speaker_embeddings/{relative}",
            "repa_relative_path": f"repa/{relative}",
            "ctc_target_length": 3,
            "ctc_adjacent_repeats": 0,
            "ctc_min_input_frames": 3,
            "ctc_feasible_40hz": True,
        }
        arrays = {
            "latent_relative_path": np.full((frames, 64), 4, dtype=np.float32),
            "video_40hz_relative_path": np.ones((frames, 1024), dtype=np.float32),
            "speaker_relative_path": np.eye(1, 192, dtype=np.float32)[0],
            "repa_relative_path": np.ones((frames + 29, 768), dtype=np.float16),
        }
        for key, array in arrays.items():
            path = root / row[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
        rows.append(row)
    contract = {
        "schema_version": 1,
        "dataset": "GRID",
        "complete": True,
        "source_audio": "complete_unmasked_waveform",
        "selection": {"mode": "subset"},
        "split_counts": {"train": 2, "val": 1},
        "manifests": {},
        "normalization": {"path": str(norm), "sha256": sha256_file(norm)},
        "vocab": {"path": str(vocab), "sha256": sha256_file(vocab)},
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
        "semantic_vae_contract": {
            "checkpoint": {"ema_sha256": "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"},
            "extraction": {"protocol": SEMANTIC_VAE_FEATURE},
        },
        "speaker_spec": {
            "dimension": 192,
            "dtype": "float32",
            "model_id": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
            "checkpoint_sha256": "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
        },
        "repa_spec": {
            "dimension": 768,
            "dtype": "float16",
            "model_id": "microsoft/wavlm-base-plus",
            "model_revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
            "checkpoint_sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
            "teacher_layer": 12,
        },
    }
    (root / "manifests").mkdir()
    for name in ("train", "val", "inventory"):
        selected = rows if name == "inventory" else [row for row in rows if row["split"] == name]
        path = root / "manifests" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in selected))
        contract["manifests"][path.name] = {"sha256": sha256_file(path), "count": len(selected)}
    publish(root, contract)
    return contract, rows


def publish(root, contract):
    path = root / "data_contract.json"
    write_json(path, contract)
    write_json(
        root / "complete.json",
        {
            "schema_version": 1,
            "complete": True,
            "contract_sha256": sha256_file(path),
            "split_counts": contract["split_counts"],
        },
    )


class GridDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract, self.rows = fixture(self.root)

    def dataset(self, **kwargs):
        return GridSemanticVaeDataset(self.root, allow_subset=True, **kwargs)

    def test_subset_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "diagnostic subset"):
            GridSemanticVaeDataset(self.root)

    def test_real_time_boundaries_and_teacher_padding(self):
        dataset = self.dataset()
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(batch["mel"].shape, (2, 64, 120))
        self.assertEqual(batch["mel_lengths"].tolist(), [119, 120])
        self.assertEqual(batch["video_lengths"].tolist(), [119, 120])
        self.assertEqual(batch["repa_feature_lengths"].tolist(), [148, 149])
        self.assertTrue(torch.equal(batch["mel"][0, :, :119], torch.ones(64, 119)))
        self.assertEqual(torch.count_nonzero(batch["mel"][0, :, 119]).item(), 0)
        self.assertEqual(len(self.dataset(split="val")), 1)

    def test_manifest_tampering_is_rejected(self):
        with (self.root / "manifests/train.jsonl").open("a") as file:
            file.write("{}\n")
        with self.assertRaisesRegex(RuntimeError, "manifest differs"):
            self.dataset()

    def test_teacher_identity_is_not_silently_changed(self):
        self.contract["repa_spec"]["teacher_layer"] = 6
        publish(self.root, self.contract)
        with self.assertRaisesRegex(RuntimeError, "repa_spec"):
            self.dataset()

    def test_speaker_corruption_is_checked_on_consumption(self):
        np.save(self.root / self.rows[0]["speaker_relative_path"], np.zeros(192, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "L2-normalized"):
            self.dataset()[0]

    def test_claiming_full_cache_cannot_bypass_inventory_count(self):
        self.contract["selection"] = {"mode": "full"}
        publish(self.root, self.contract)
        with self.assertRaisesRegex(RuntimeError, "29,557"):
            self.dataset()


class GridCheckpointTests(unittest.TestCase):
    def test_grid_loader_epoch_is_preserved_after_skip_and_resume(self):
        from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, skip_first_batches

        class SeededSampler:
            def __init__(self):
                self.epoch = 0

            def __len__(self):
                return 16

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                generator = torch.Generator().manual_seed(666 + self.epoch)
                return iter([[index] for index in torch.randperm(16, generator=generator).tolist()])

        for world_size in (1, 4):
            for start_epoch, skipped_batches in ((0, 0), (7, 1)):
                with self.subTest(world_size=world_size, start_epoch=start_epoch):
                    source = SeededSampler()
                    sampler = GridEvenBatchSampler(source, world_size)
                    if world_size > 1:
                        sampler = BatchSamplerShard(
                            sampler, num_processes=world_size, process_index=0, even_batches=False
                        )
                    loader = DataLoaderShard(
                        list(range(16)), batch_sampler=sampler, generator=torch.Generator().manual_seed(666)
                    )
                    skipped_loader = skip_first_batches(loader, num_batches=skipped_batches)
                    trainer = SimpleNamespace(total_optimizer_updates=100000)
                    for epoch in range(start_epoch, start_epoch + 3):
                        current = skipped_loader if epoch == start_epoch else loader
                        Trainer_VT._set_dataloader_epoch(trainer, loader, current, epoch)
                        actual = [int(batch[0]) for batch in current]
                        generator = torch.Generator().manual_seed(666 + epoch)
                        expected = torch.randperm(16, generator=generator).tolist()[::world_size]
                        if epoch == start_epoch:
                            expected = expected[skipped_batches:]
                        self.assertEqual(actual, expected)
                        self.assertEqual(source.epoch, epoch)

        # Historical snapshots retain their original sampler-only behavior.
        source = SeededSampler()
        loader = DataLoaderShard(
            list(range(16)),
            batch_sampler=GridEvenBatchSampler(source, 1),
            generator=torch.Generator().manual_seed(666),
        )
        skipped_loader = skip_first_batches(loader, num_batches=0)
        observed_epochs = []
        for epoch in range(3):
            current = skipped_loader if epoch == 0 else loader
            Trainer_VT._set_dataloader_epoch(SimpleNamespace(), loader, current, epoch)
            list(current)
            observed_epochs.append(source.epoch)
        self.assertEqual(observed_epochs, [0, 0, 1])

    def test_distributed_epoch_preserves_samples_and_equal_rank_steps(self):
        from accelerate.data_loader import BatchSamplerShard

        class TinySampler:
            def __init__(self):
                self.epoch = 0

            def __len__(self):
                return 5

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                return iter([[index] for index in (range(5) if self.epoch == 0 else reversed(range(5)))])

        sampler = GridEvenBatchSampler(TinySampler(), 4)
        self.assertEqual(len(sampler), 8)
        shards = [
            BatchSamplerShard(sampler, num_processes=4, process_index=rank, even_batches=False) for rank in range(4)
        ]
        for epoch in range(2):
            sampler.set_epoch(epoch)
            rank_batches = [list(shard) for shard in shards]
            self.assertEqual([len(batches) for batches in rank_batches], [2, 2, 2, 2])
            self.assertEqual({batch[0] for batches in rank_batches for batch in batches}, set(range(5)))

    def trainer(self, root):
        trainer = object.__new__(GridSemanticVaeTrainer)
        trainer.accelerator = SimpleNamespace(
            num_processes=1,
            process_index=0,
            is_main_process=True,
            wait_for_everyone=lambda: None,
            unwrap_model=lambda model: model,
        )
        trainer.model = torch.nn.Linear(2, 1)
        trainer.ema_model = copy.deepcopy(trainer.model)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.scheduler = torch.optim.lr_scheduler.LinearLR(trainer.optimizer, total_iters=10)
        trainer.checkpoint_path = str(root)
        trainer.training_contract_sha256 = "fixture-contract"
        trainer.keep_last_n_checkpoints = -1
        return trainer

    def test_checkpoint_roundtrip_restores_states_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self.trainer(root)
            trainer.model(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            expected = copy.deepcopy(trainer.model.state_dict())
            trainer.save_checkpoint(1, last=True)
            expected_rng = (random.random(), np.random.rand(), torch.rand(3))
            trainer.model.weight.data.zero_()
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            self.assertEqual(trainer.load_checkpoint(), 1)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, expected[key])
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.rand(), expected_rng[1])
            torch.testing.assert_close(torch.rand(3), expected_rng[2])
            self.assertEqual(trainer.scheduler.last_epoch, 1)
            self.assertTrue(trainer.optimizer.state_dict()["state"])
            self.assertFalse(list(root.glob("*.tmp")))
            trainer.training_contract_sha256 = "another-dataset"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                trainer.load_checkpoint()


if __name__ == "__main__":
    unittest.main()
