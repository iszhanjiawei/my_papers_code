"""CPU contract tests for cache-only audio targets and batch masks.

Run from this experiment with PYTHONPATH=src. No real cache is modified.
Use audit_audio_teacher_cache.py --sample-count 32 --check-features for the
separate real-data check.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aligndit.model.audio_teacher_cache import (
    AudioTeacherCache,
    identity_key,
    teacher_frame_lengths,
    waveform_identity,
)
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset


class AudioTeacherCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.audio_root = root / "audio"
        audio_path = self.audio_root / "train/clip/example.wav"
        audio_path.parent.mkdir(parents=True)
        # The reader intentionally checks file identity without decoding PCM.
        audio_path.write_bytes(b"source waveform identity fixture")
        self.audio_path = audio_path
        self.record = {
            "utterance_key": "celebvdub/train/clip/example",
            "audio_relative_path": "train/clip/example.wav",
            "sample_rate": 16000,
            "source_sample_rate": 16000,
            "original_num_samples": 1600,
            "source_num_samples": 1600,
        }
        self.metadata = {
            "format_version": 1,
            "checkpoint": {"path": "/fixture/checkpoint", "size_bytes": 123, "mtime_ns": 456},
            "preprocessing": "avhubert_audio_only_pcm_logfbank26_stack4_v1",
            "sample_rate_hz": 16000,
            "filterbank_bins": 26,
            "stack_order_audio": 4,
            "normalize_per_frame": True,
            "source_modality": "audio_only",
            "output_layer": "final_contextual",
            "feature_dim": 1024,
            "cache_dtype": "float16",
            "hubert_source_sha256": "fixture",
        }
        self.teacher_identity = identity_key(self.metadata)
        self.cache_dir = root / "cache" / self.teacher_identity
        self.cache_dir.mkdir(parents=True)
        (self.cache_dir / "teacher_metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.reader = AudioTeacherCache(self.cache_dir, self.audio_root, expected_identity=self.teacher_identity)
        self.audio_identity = waveform_identity(audio_path)
        cache_key = identity_key(self.audio_identity)
        self.entry = self.cache_dir / cache_key[:2] / f"{cache_key}.npz"
        self.entry.parent.mkdir()
        self.features = np.arange(3 * 1024, dtype=np.float16).reshape(3, 1024) / 100
        self.write_entry()

    def write_entry(self, **overrides):
        values = {
            "features": self.features,
            "audio_identity": json.dumps(self.audio_identity, sort_keys=True),
            "teacher_identity": self.teacher_identity,
        }
        values.update(overrides)
        np.savez(self.entry, **values)

    def test_native_length_and_tail_validity_are_distinct(self):
        features, valid = self.reader.load(self.record)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (3, 1024))
        self.assertEqual(valid, 2)
        np.testing.assert_array_equal(features, self.features.astype(np.float32))
        self.assertEqual(teacher_frame_lengths(880), (1, 1))
        self.assertEqual(teacher_frame_lengths(879), (1, 0))
        self.assertEqual(teacher_frame_lengths(881), (2, 1))
        self.assertEqual(teacher_frame_lengths(1520), (2, 2))

    def test_missing_cache_fails_without_rewriting(self):
        self.entry.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "never re-extracts"):
            self.reader.load(self.record)
        self.assertFalse(self.entry.exists())

    def test_changed_waveform_uses_new_identity_and_fails(self):
        self.audio_path.write_bytes(b"different current waveform")
        with self.assertRaisesRegex(FileNotFoundError, "Missing or stale"):
            self.reader.load(self.record)
        self.assertEqual(len(list(self.cache_dir.glob("*/*.npz"))), 1)

    def test_identity_corruption_is_rejected(self):
        self.write_entry(audio_identity=json.dumps({**self.audio_identity, "size_bytes": 999}))
        with self.assertRaisesRegex(RuntimeError, "waveform identity mismatch"):
            self.reader.load(self.record)
        self.write_entry(teacher_identity="unrelated teacher")
        with self.assertRaisesRegex(RuntimeError, "teacher identity mismatch"):
            self.reader.load(self.record)

    def test_invalid_dtype_shape_and_values_are_rejected(self):
        for values in (
            self.features.astype(np.float32),
            self.features[:2],
            np.full_like(self.features, np.nan),
            np.array([{"unsafe": "object dtype"}], dtype=object),
        ):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                self.write_entry(features=values)
                with self.assertRaises(RuntimeError):
                    self.reader.load(self.record)

    def test_metadata_identity_mismatch_is_rejected(self):
        changed = {**self.metadata, "source_modality": "audio_video"}
        (self.cache_dir / "teacher_metadata.json").write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            AudioTeacherCache(self.cache_dir, self.audio_root, expected_identity=self.teacher_identity)

    def test_manifest_path_and_length_contracts_are_rejected(self):
        for change in (
            {"audio_relative_path": "../outside.wav"},
            {"source_sample_rate": 44100},
            {"source_num_samples": 1500},
            {"original_num_samples": 0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.reader.load({**self.record, **change})

    @staticmethod
    def batch_item(audio_frames, teacher_frames, teacher_valid):
        return {
            "mel_spec": torch.ones(64, audio_frames),
            "video": torch.ones(audio_frames, 1024),
            "text": "a",
            "ctc_feasible": True,
            "ctc_target_length": 1,
            "utterance_key": f"clip-{audio_frames}",
            "audio_teacher": torch.ones(teacher_frames, 1024),
            "audio_teacher_valid_length": teacher_valid,
        }

    def test_collate_preserves_native_and_valid_lengths(self):
        batch = [self.batch_item(8, 5, 4), self.batch_item(4, 3, 2)]
        result = SemanticVaeCelebVDubDataset.collate_fn(batch)
        self.assertEqual(result["audio_teacher"].shape, (2, 5, 1024))
        self.assertEqual(result["audio_teacher_lengths"].tolist(), [5, 3])
        self.assertEqual(result["audio_teacher_valid_lengths"].tolist(), [4, 2])
        self.assertEqual(torch.count_nonzero(result["audio_teacher"][1, 3:]).item(), 0)
        self.assertEqual(result["mel_lengths"].tolist(), [8, 4])

    def test_collate_rejects_partially_present_teacher(self):
        batch = [self.batch_item(8, 5, 4), self.batch_item(4, 3, 2)]
        del batch[1]["audio_teacher"]
        with self.assertRaisesRegex(RuntimeError, "every sample"):
            SemanticVaeCelebVDubDataset.collate_fn(batch)

    def test_collate_without_teacher_keeps_baseline_interface(self):
        batch = [self.batch_item(8, 5, 4)]
        del batch[0]["audio_teacher"]
        del batch[0]["audio_teacher_valid_length"]
        result = SemanticVaeCelebVDubDataset.collate_fn(batch)
        self.assertNotIn("audio_teacher", result)


if __name__ == "__main__":
    unittest.main()
