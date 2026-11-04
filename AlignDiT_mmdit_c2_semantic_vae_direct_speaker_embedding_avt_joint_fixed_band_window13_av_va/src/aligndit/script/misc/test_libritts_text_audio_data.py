"""Cache completeness, immutable inventory, NFS locks, and no-drop batching tests."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from aligndit.model.libritts_text_audio_dataset import (
    LengthBudgetBatchSampler, LibriTTSTextAudioDataset, load_manifest, safe_path, sha256_file,
)
from aligndit.script.misc.prepare_libritts_text_audio import (
    POSTERIOR_PROTOCOL, audit_cache, build_inventory, cache_lock, json_bytes,
    normalize_text, save_record,
)


class CacheTests(unittest.TestCase):
    def setUp(self):
        # Point this at the real shared filesystem to exercise Linux NFS locking.
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("LIBRITTS_TEST_TMPDIR"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_cache(self):
        source = self.root / "source"
        for subset, speakers in (("train-clean-100", (1, 2)), ("train-clean-360", (3, 4))):
            for speaker in speakers:
                path = source / subset / str(speaker) / "1" / f"{speaker}_1_1.wav"
                path.parent.mkdir(parents=True)
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(24_000)
                    stream.writeframes(np.zeros(24_000, dtype=np.int16).tobytes())
                path.with_suffix(".normalized.txt").write_text("Hello: ‘you’!", encoding="utf-8")
        vocabulary = self.root / "base.txt"
        vocabulary.write_text(" \na\nb\n", encoding="utf-8")
        normalization = self.root / "norm.json"
        normalization.write_bytes(json_bytes({"channel_count": 64, "feature": POSTERIOR_PROTOCOL, "mean": [0.] * 64, "std": [1.] * 64}))
        cache = self.root / "cache"
        args = argparse.Namespace(cache_root=cache, dataset_root=source, base_vocab=vocabulary,
                                  normalization=normalization, expected_normalization_sha256=sha256_file(normalization),
                                  expected_count=4, validation_speakers=1, seed=666, limit_per_split=None)
        build_inventory(args)
        return cache, args

    def complete_cache(self, cache):
        _, rows, _ = load_manifest(cache)
        spec = cache / "extraction_spec.json"
        spec.write_bytes(json_bytes({"metadata_sha256": sha256_file(cache / "metadata.json"),
                                     "semantic_vae_checkpoint_sha256": "fixture-codec", "speaker_checkpoint_sha256": "fixture-speaker",
                                     "posterior_protocol": POSTERIOR_PROTOCOL}))
        speaker = np.zeros(192, dtype=np.float32)
        speaker[0] = 1
        for row in rows:
            save_record(cache, row, sha256_file(spec), np.zeros((row["latent_frames"], 64), dtype=np.float32), speaker)
        audit_cache(argparse.Namespace(cache_root=cache))
        return rows

    def test_shared_locks_allow_readers_exclude_writer(self):
        with cache_lock(self.root, exclusive=False):
            with cache_lock(self.root, exclusive=False):
                pass
            with self.assertRaises(BlockingIOError):
                with cache_lock(self.root, exclusive=True):
                    pass
        with cache_lock(self.root, exclusive=True):
            with self.assertRaises(BlockingIOError):
                with cache_lock(self.root, exclusive=False):
                    pass

    def test_runtime_paths_reject_parent_traversal_and_symlink_escape(self):
        for relative in ("../outside", "/outside", "a/../../outside", "a\0b"):
            with self.assertRaises(ValueError):
                safe_path(self.root, relative)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_path(self.root, "escape/outside")

    def test_inventory_retains_vocab_and_speaker_disjointness(self):
        cache, args = self.make_cache()
        metadata, rows, vocab = load_manifest(cache)
        self.assertEqual(list(vocab)[:3], [" ", "a", "b"])
        self.assertEqual(metadata["counts"], {"accepted": 4, "discovered": 4, "rejected": 0})
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(metadata["split_counts"].values()), 4)
        before = (cache / "metadata.json").read_bytes()
        build_inventory(args)
        self.assertEqual(before, (cache / "metadata.json").read_bytes())
        (cache / "train.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "inventory file changed"):
            load_manifest(cache)

    def test_missing_record_never_creates_complete_marker(self):
        cache, _ = self.make_cache()
        rows = self.complete_cache(cache)
        (cache / rows[0]["latent_relative_path"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "audit failed"):
            audit_cache(argparse.Namespace(cache_root=cache))
        self.assertFalse((cache / "complete.json").exists())
        report = json.loads((cache / "audit_report.json").read_text())
        self.assertEqual((report["expected"], report["valid"], len(report["failures"])), (4, 3, 1))
        with self.assertRaises(FileNotFoundError):
            LibriTTSTextAudioDataset(cache)

    def test_runtime_hash_check_and_padded_collation(self):
        cache, _ = self.make_cache()
        self.complete_cache(cache)
        dataset = LibriTTSTextAudioDataset(cache)
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["audio"].shape), (2, 40, 64))
        self.assertEqual(tuple(batch["speaker_embedding"].shape), (2, 192))
        path = cache / dataset.records[0]["latent_relative_path"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupt latent"):
            dataset[0]

    def test_missing_transcript_rejects_whole_inventory(self):
        cache, args = self.make_cache()
        next(args.dataset_root.rglob("*.normalized.txt")).unlink()
        with self.assertRaisesRegex(RuntimeError, "Inventory rejected"):
            build_inventory(args)
        self.assertTrue((cache / "inventory_failures.json").exists())

    def test_normalization_preserves_text_instead_of_unknown_token_loss(self):
        self.assertEqual(normalize_text("  HELLO: ‘there’—yes…\n"), "hello: 'there'-yes...")


class BatchTests(unittest.TestCase):
    def test_every_record_retained_including_oversize(self):
        class Dataset:
            frames = [40, 2000, 50, 10, 250, 100]

            def __len__(self):
                return len(self.frames)

            def get_frame_len(self, index):
                return self.frames[index]

        data = Dataset()
        sampler = LengthBudgetBatchSampler(data, max_frames=300, max_samples=2)
        batches = list(sampler)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(data))))
        self.assertIn([1], batches)
        for batch in batches:
            self.assertTrue(len(batch) == 1 or len(batch) * max(data.frames[index] for index in batch) <= 300)
        self.assertEqual(batches, list(sampler))
        with self.assertRaisesRegex(ValueError, "nothing was dropped"):
            LengthBudgetBatchSampler(data, max_frames=300, overflow="error")


if __name__ == "__main__":
    unittest.main()
