"""Regression tests for isolated WavLM data and strict AVT S2c migration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from aligndit.model.repa import RepaFeatureError
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.semantic_vae_direct_migration import (
    REPA_PROJECTOR_SHAPES,
    S2C_IGNORED_PROJECTOR_KEYS,
    _expected_new_target_keys,
    migrate_s2c_ema_into_model,
)


class StateFixture:
    """Small synthetic tensors exercise migration's actual public boundary."""

    def __init__(self, with_repa=True):
        self.transformer = SimpleNamespace(
            speaker_proj=object(), repa_projector=object() if with_repa else None,
            text_joint_attention=True,
        )
        added = _expected_new_target_keys(text_joint_attention=True, has_speaker=True, has_repa=with_repa)
        self.target = {key: torch.zeros(1) for key in added}
        self.source = {}
        for key in added:
            if key.startswith("transformer.transformer_blocks."):
                _, _, layer, stream_name = key.split(".", 3)
                if stream_name.startswith(("v_", "t_")):
                    self.source[f"transformer.transformer_blocks.{layer}.{stream_name[2:]}"] = torch.ones(1)
        for index in range(303 - len(self.source)):
            self.source[f"inherited_tensor_{index}"] = torch.ones(1)
        self.target.update({key: torch.zeros_like(value) for key, value in self.source.items()})
        self.source.update({key: torch.ones(1) for key in S2C_IGNORED_PROJECTOR_KEYS})
        self.target["transformer.speaker_proj.weight"] = torch.zeros(768, 192)
        self.target["transformer.text_joint_proj.weight"] = torch.zeros(1).expand(768, 512)
        self.target["transformer.text_joint_proj.bias"] = torch.zeros(768)
        if with_repa:
            for key, shape in REPA_PROJECTOR_SHAPES.items():
                self.target[key] = torch.zeros(1).expand(shape)

    def state_dict(self):
        return self.target

    def load_state_dict(self, state, strict):
        if not strict or set(state) != set(self.target):
            raise AssertionError("Migration must load strictly with all target keys")
        self.target = state


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.parent = Path(self.tempdir.name) / "parent.pt"
        self.parent.touch()

    def tearDown(self):
        self.tempdir.cleanup()

    def migrate(self, fixture):
        return migrate_s2c_ema_into_model(
            fixture, fixture.source, parent_path=self.parent,
            parent_sha256="test", parent_size=0, parent_contract_sha256="test", parent_ema_step=70000,
        )

    def test_hard9_tra_and_disabled_counts_preserve_all_imported_tensors(self):
        for enabled, expected_target, expected_new in ((True, 808, 505), (False, 802, 499)):
            with self.subTest(enabled=enabled):
                fixture = StateFixture(enabled)
                report = self.migrate(fixture)
                self.assertEqual((report.source_key_count, report.target_key_count, report.loaded_key_count,
                                  len(report.ignored_source_keys), len(report.new_target_keys)),
                                 (313, expected_target, 303, 10, expected_new))
                for key in fixture.source.keys() & fixture.target.keys():
                    torch.testing.assert_close(fixture.target[key], fixture.source[key], rtol=0, atol=0)

    def test_missing_extra_wrong_shape_nonfinite_projectors_fail(self):
        for corruption in ("missing", "extra", "wrong_shape", "nonfinite"):
            with self.subTest(corruption=corruption):
                fixture = StateFixture()
                key = "transformer.repa_projector.4.bias"
                if corruption == "missing":
                    del fixture.target[key]
                elif corruption == "extra":
                    fixture.target["transformer.repa_projector.unexpected"] = torch.zeros(1)
                elif corruption == "wrong_shape":
                    fixture.target[key] = torch.zeros(767)
                else:
                    fixture.target[key] = torch.full((768,), float("nan"))
                with self.assertRaisesRegex(RuntimeError, "exactly six new finite tensors"):
                    self.migrate(fixture)

    def test_repa_keys_cannot_enter_disabled_model(self):
        fixture = StateFixture()
        fixture.transformer.repa_projector = None
        with self.assertRaisesRegex(RuntimeError, "Unexpected S2c-to-C2 new tensor keys"):
            self.migrate(fixture)

    def test_avt_shape_validation_remains_active(self):
        fixture = StateFixture()
        fixture.target["transformer.transformer_blocks.0.t_attn.to_q.weight"] = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "New modality tensor does not match"):
            self.migrate(fixture)

    def test_speaker_zero_initialization_remains_required(self):
        fixture = StateFixture()
        fixture.target["transformer.speaker_proj.weight"][0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "zero-initialized"):
            self.migrate(fixture)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = SemanticVaeCelebVDubDataset.__new__(SemanticVaeCelebVDubDataset)
        self.dataset.repa_feature_cache_dir = self.root
        self.dataset.repa_feature_dim = 768

    def tearDown(self):
        self.tempdir.cleanup()

    def record(self, path):
        return {"audio_relative_path": path, "utterance_key": "test-id"}

    def test_full_path_disambiguates_same_clip_names(self):
        for video, value in (("video_a", 1), ("video_b", 2)):
            path = self.root / "train" / video / "clip.npy"
            path.parent.mkdir(parents=True)
            np.save(path, np.full((3, 768), value, dtype=np.float16))
            feature = self.dataset._load_repa_feature_array(self.record(f"train/{video}/clip.wav"))
            self.assertTrue((feature == value).all())

    def test_traversal_missing_invalid_dtype_and_nonfinite_fail(self):
        for path in ("train/../../escape.wav", "/train/absolute.wav", "test/other.wav"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.dataset._load_repa_feature_array(self.record(path))
        with self.assertRaises(FileNotFoundError):
            self.dataset._load_repa_feature_array(self.record("train/missing.wav"))
        path = self.root / "train" / "invalid.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        for invalid in (np.zeros((3, 768), np.float32), np.zeros((3, 767), np.float16),
                        np.full((3, 768), np.nan, np.float16)):
            np.save(path, invalid)
            with self.assertRaises(RepaFeatureError):
                self.dataset._load_repa_feature_array(self.record("train/invalid.wav"))

    @staticmethod
    def item(audio_len, teacher_len):
        return {
            "mel_spec": torch.ones(64, audio_len), "video": torch.ones(audio_len, 1024),
            "ctc_feasible": True, "text": "a", "ctc_target_length": 1,
            "utterance_key": f"clip{audio_len}", "speaker_embedding": torch.ones(192),
            "repa_feature": torch.ones(teacher_len, 768, dtype=torch.float16),
        }

    def test_collation_keeps_teacher_lengths_separate_and_half_precision(self):
        batch = SemanticVaeCelebVDubDataset.collate_fn([self.item(4, 5), self.item(8, 9)])
        self.assertEqual(batch["mel_lengths"].tolist(), [4, 8])
        self.assertEqual(batch["repa_feature_lengths"].tolist(), [5, 9])
        self.assertEqual(tuple(batch["repa_features"].shape), (2, 9, 768))
        self.assertEqual(batch["repa_features"].dtype, torch.float16)
        self.assertEqual(torch.count_nonzero(batch["repa_features"][0, 5:]).item(), 0)
        self.assertEqual(tuple(batch["speaker_embedding"].shape), (2, 192))

    def test_collation_rejects_partially_missing_teacher_targets(self):
        first, second = self.item(4, 5), self.item(8, 9)
        del second["repa_feature"]
        with self.assertRaisesRegex(RuntimeError, "every sample"):
            SemanticVaeCelebVDubDataset.collate_fn([first, second])


if __name__ == "__main__":
    unittest.main()
