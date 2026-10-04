"""Protocol regressions: run with ``python -m unittest discover -s evaluation/lse``.

The optional upstream parity test uses SYNCNET_REPO (or the installed default).
No model weights, GPU, or test-set media are needed for these unit tests.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate
import numpy as np
import torch
from metrics import extract_embeddings, score_embeddings


class MetricTests(unittest.TestCase):
    def test_mean_over_windows_precedes_shift_minimum(self):
        # Each window could match perfectly at a different shift. One global
        # shift instead leaves mean distance 5, preventing optimistic scores.
        result = score_embeddings(torch.tensor([[10.0], [20.0]]), torch.tensor([[20.0], [10.0]]), vshift=1)
        self.assertAlmostEqual(result["lse_d"], 5.0, places=5)
        self.assertAlmostEqual(result["lse_c"], 5.0, places=5)
        self.assertAlmostEqual(result["zero_offset_distance"], 10.0, places=5)

    def test_known_delayed_audio_offset_has_upstream_sign(self):
        visual = torch.eye(40)
        acoustic = torch.cat([torch.zeros(3, 40), visual[:-3]], dim=0)
        result = score_embeddings(visual, acoustic, vshift=5)
        self.assertEqual(result["offset_frames"], -3)
        self.assertEqual(result["offset_seconds"], -0.12)
        self.assertLess(result["lse_d"], result["zero_offset_distance"])

    def test_aligned_embeddings_have_zero_offset(self):
        visual = torch.eye(40)
        result = score_embeddings(visual, visual, vshift=5)
        self.assertEqual(result["offset_frames"], 0)
        self.assertLess(result["lse_d"], 1e-5)
        self.assertGreater(result["lse_c"], 1)

    def test_reject_invalid_embeddings(self):
        for video, audio in [
            (torch.empty(0, 3), torch.empty(0, 3)),
            (torch.ones(3, 2), torch.ones(4, 2)),
            (torch.tensor([[float("nan")]]), torch.ones(1, 1)),
            (torch.ones(1, 1), torch.tensor([[float("inf")]])),
        ]:
            with self.subTest(video=video.shape, audio=audio.shape), self.assertRaises(ValueError):
                score_embeddings(video, audio)

    def test_installed_upstream_calc_pdist_parity(self):
        default = os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/syncnet/syncnet_python"
        repo = Path(os.environ.get("SYNCNET_REPO", default))
        source = repo / "SyncNetInstance.py"
        if not source.is_file():
            self.skipTest("Install SyncNet or set SYNCNET_REPO for upstream parity")
        spec = importlib.util.spec_from_file_location("lse_test_upstream", source)
        upstream = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo))
        try:
            spec.loader.exec_module(upstream)
        finally:
            sys.path.remove(str(repo))
        generator = torch.Generator().manual_seed(7209)
        for windows, shift in [(1, 15), (7, 3), (71, 15)]:
            with self.subTest(windows=windows, shift=shift):
                visual = torch.randn(windows, 32, generator=generator)
                acoustic = torch.randn(windows, 32, generator=generator)
                reference = torch.stack(upstream.calc_pdist(visual, acoustic, shift), 1).mean(1)
                result = score_embeddings(visual, acoustic, shift)
                self.assertEqual(result["distance_by_shift"], reference.tolist())
                minimum, index = reference.min(0)
                self.assertEqual(result["lse_d"], minimum.item())
                self.assertEqual(result["lse_c"], (reference.median() - minimum).item())
                self.assertEqual(result["offset_frames"], shift - index.item())


class RecordingModel:
    def __init__(self):
        self.images = []
        self.sounds = []

    def forward_lip(self, value):
        self.images.append(value.clone())
        return value.mean(dim=(2, 3, 4))

    def forward_aud(self, value):
        self.sounds.append(value.clone())
        return value.mean(dim=(1, 3))


class PreprocessingTests(unittest.TestCase):
    def test_bgr_values_and_syncnet_tensor_shapes_are_preserved(self):
        frames = np.empty((9, 224, 224, 3), dtype=np.uint8)
        frames[:] = [13, 71, 201]
        audio = (np.sin(np.arange(9 * 640) * 0.17) * 18000).astype(np.int16)
        model = RecordingModel()
        visual, acoustic = extract_embeddings(model, frames, audio, "cpu", batch_size=3)
        self.assertEqual(visual.shape, (4, 3))
        self.assertEqual(acoustic.shape, (4, 13))
        self.assertEqual([tuple(x.shape) for x in model.images], [(3, 3, 5, 224, 224), (1, 3, 5, 224, 224)])
        self.assertEqual([tuple(x.shape) for x in model.sounds], [(3, 1, 13, 20), (1, 1, 13, 20)])
        torch.testing.assert_close(visual, torch.tensor([[13.0, 71.0, 201.0]]).repeat(4, 1))
        self.assertEqual(model.images[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(acoustic).all())

    def test_mouth_crop_float_audio_and_too_short_media_are_rejected(self):
        frames = np.zeros((9, 224, 224, 3), dtype=np.uint8)
        audio = np.zeros(9 * 640, dtype=np.int16)
        cases = [
            (frames[:, :88, :88], audio),
            (frames, audio.astype(np.float32)),
            (frames[:5], audio),
            (frames, audio[: 5 * 640]),
        ]
        for video, signal in cases:
            with self.subTest(video_shape=video.shape, audio_shape=signal.shape), self.assertRaises(ValueError):
                extract_embeddings(RecordingModel(), video, signal, "cpu")


class PairingAndReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self, **kwargs):
        defaults = {
            "manifest": None,
            "test_list": None,
            "video": None,
            "audio": None,
            "video_root": self.root / "videos",
            "audio_root": self.root / "wavs",
            "split": "test",
            "video_suffix": ".mp4",
            "limit": None,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_same_basename_in_different_videos_keeps_full_relative_pairing(self):
        test_list = self.root / "test.txt"
        test_list.write_text("person_a/001\ntest/person_b/001\n")
        samples = evaluate.load_samples(self.args(test_list=test_list))
        self.assertEqual([s["id"] for s in samples], ["test/person_a/001", "test/person_b/001"])
        self.assertEqual(samples[0]["video"], self.root / "videos/test/person_a/001.mp4")
        self.assertEqual(samples[1]["audio"], self.root / "wavs/test/person_b/001.wav")

    def test_duplicate_ids_cannot_be_hidden_by_limit(self):
        test_list = self.root / "test.txt"
        test_list.write_text("person_a/001\ntest/person_a/001\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate.load_samples(self.args(test_list=test_list, limit=1))

    def test_manifest_paths_are_relative_to_manifest(self):
        manifest = self.root / "pairs.jsonl"
        manifest.write_text(json.dumps({"id": "person_a/001", "video": "video/001.mp4", "audio": "wav/001.wav"}) + "\n")
        sample = evaluate.load_samples(self.args(manifest=manifest))[0]
        self.assertEqual(sample["video"], self.root / "video/001.mp4")
        self.assertEqual(sample["audio"], self.root / "wav/001.wav")

    def test_manifest_requires_explicit_audio_to_avoid_accidental_ground_truth(self):
        manifest = self.root / "pairs.jsonl"
        entry = {"id": "person_a/001", "video": "video/001.mp4"}
        manifest.write_text(json.dumps(entry) + "\n")
        with self.assertRaises(KeyError):
            evaluate.load_samples(self.args(manifest=manifest))
        for invalid in ["", "  ", False, 0]:
            entry["audio"] = invalid
            manifest.write_text(json.dumps(entry) + "\n")
            with self.subTest(audio=invalid), self.assertRaises(ValueError):
                evaluate.load_samples(self.args(manifest=manifest))
        entry["audio"] = None
        manifest.write_text(json.dumps(entry) + "\n")
        self.assertIsNone(evaluate.load_samples(self.args(manifest=manifest))[0]["audio"])

    def test_ids_reject_absolute_and_parent_paths(self):
        for value in ["", "/video/001", "../001", "video/../001", "video\\001"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate.safe_id(value)

    def test_summary_is_equal_clip_mean_and_reports_missing_coverage(self):
        rows = [
            {"status": "ok", "lse_d": 2.0, "lse_c": 10.0, "num_windows": 5},
            {"status": "ok", "lse_d": 10.0, "lse_c": 2.0, "num_windows": 500},
            {"status": "error", "error": "missing generated WAV"},
        ]
        result = evaluate.summarize(rows, requested=4, protocol={"test": "fixture"})
        self.assertEqual(result["lse_d"], 6.0)
        self.assertEqual(result["lse_c"], 6.0)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["complete"])
        empty = evaluate.summarize([rows[-1]], requested=1, protocol={})
        self.assertIsNone(empty["lse_d"])
        self.assertIsNone(empty["lse_c"])
        self.assertFalse(empty["complete"])

    def test_duration_and_crop_guards_prevent_scoring(self):
        video, audio = self.root / "video.mp4", self.root / "audio.wav"
        video.touch()
        audio.touch()
        sample = {"video": video, "audio": audio}
        args = SimpleNamespace(duration_policy="strict", duration_tolerance=0.1, input_kind="syncnet-crop")
        cases = [
            ([{"duration_seconds": 2.0}, {"duration_seconds": 3.0}], "duration mismatch"),
            (
                [
                    {"duration_seconds": 2.0, "width": 88, "height": 88, "r_frame_rate": "25/1"},
                    {"duration_seconds": 2.0},
                ],
                "AV-HuBERT mouth crops",
            ),
            (
                [
                    {"duration_seconds": 2.0, "width": 224, "height": 224, "r_frame_rate": "30/1"},
                    {"duration_seconds": 2.0},
                ],
                "25fps",
            ),
        ]
        for metadata, message in cases:
            with (
                self.subTest(message=message),
                patch.object(evaluate, "probe", side_effect=metadata),
                patch.object(evaluate, "score_crop") as scoring,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate.evaluate_sample(
                        sample, None, args, self.root, self.root, self.root / "log", "ffmpeg", "ffprobe"
                    )
                scoring.assert_not_called()

    def test_missing_media_fails_before_external_processes(self):
        sample = {"video": self.root / "missing.mp4", "audio": None}
        with patch.object(evaluate, "probe") as probe:
            with self.assertRaises(FileNotFoundError):
                evaluate.evaluate_sample(
                    sample, None, None, self.root, self.root, self.root / "log", "ffmpeg", "ffprobe"
                )
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
