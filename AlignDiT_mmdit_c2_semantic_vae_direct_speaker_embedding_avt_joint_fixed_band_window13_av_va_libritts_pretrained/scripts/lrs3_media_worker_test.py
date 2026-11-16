"""Regression checks for segment identity, strict crops, and cache validation."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

import lrs3_media_worker as worker
from lrs3_media_worker import (complete_segment_offsets, crop_mouth, interpolate_landmarks,
                              read_frames, valid_array, verify_original_frame_mapping)
from lrs3_media_worker_pipeline import Pipeline
from lrs3_media_early_fan import current_attempt, eligible_failures, read_complete_rows


class MediaWorkerTests(unittest.TestCase):
    def test_early_fan_requires_current_unfinished_regular_attempt(self):
        state = {"status": "running", "active_stage": "full_media_regular",
                 "stages": {"full_media_regular": {"finished": False, "attempt_token": "a1"}}}
        self.assertTrue(current_attempt(state, "a1"))
        self.assertFalse(current_attempt(state, "a2"))
        state["stages"]["full_media_regular"]["finished"] = True
        self.assertFalse(current_attempt(state, "a1"))
        state["stages"]["full_media_regular"]["finished"] = False
        state["active_stage"] = "full_media_landmark_repair"
        self.assertFalse(current_attempt(state, "a1"))

    def test_early_fan_uses_only_terminal_face_crop_failures_and_ignores_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "worker.jsonl"
            rows = [{"relative_stem": key, "sample_id": key, "status": status, "error": error}
                    for key, status, error in [("crop", "failed", "Mouth ROI outside canvas"),
                                               ("ok", "ok", "landmark"),
                                               ("oom", "failed", "CUDA out of memory"),
                                               ("mouth96", "failed", "landmark mismatch"),
                                               ("attempted", "failed", "landmark missing")]]
            report.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"unfinished":')
            source = {row["sample_id"]: {"sample_id": row["sample_id"],
                                       "width": 96 if row["sample_id"] == "mouth96" else 224} for row in rows}
            state = {"stages": {"full_media_regular": {"reports": [str(report)]}}}
            self.assertEqual(len(read_complete_rows(report)), 5)
            self.assertEqual(set(eligible_failures(state, source, {"attempted"})), {"crop"})

    def test_pipeline_decoder_threads_apply_only_to_av_and_preserve_crop_override(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.root = Path("/unused/output")
        pipeline.args = SimpleNamespace(cpu_threads=2, mean_face="mean.npy", checkpoint="model.pt",
                                        inventory="inventory.tsv", landmark_root="landmarks",
                                        original_root="original", fairseq_root="fairseq", user_dir="avhubert",
                                        av_decode_threads=0, av_warmup_frames=0)
        self.assertNotIn("--decode-threads", pipeline.command("manifest.jsonl", 0, 1))
        self.assertNotIn("--warmup-frames", pipeline.command("manifest.jsonl", 0, 1))
        pipeline.args.av_decode_threads = 4
        pipeline.args.av_warmup_frames = 750
        for stage in ("all", "features"):
            command = pipeline.command("manifest.jsonl", 0, 1, media_stage=stage)
            self.assertEqual(command[command.index("--decode-threads") + 1], "4")
            self.assertEqual(command[command.index("--warmup-frames") + 1], "750")
        crop = pipeline.command("manifest.jsonl", 0, 1, media_stage="crop", decode_threads=2)
        self.assertEqual(crop[crop.index("--decode-threads") + 1], "2")
        self.assertNotIn("--warmup-frames", crop)
        self.assertNotIn("--decode-threads", pipeline.command("manifest.jsonl", 0, 1, repair=True))
        self.assertNotIn("--warmup-frames", pipeline.command("manifest.jsonl", 0, 1, repair=True))

    def test_warmup_default_has_no_effect_and_rejects_training_mode(self):
        self.assertIsNone(worker.warmup_reader(None, 0))
        with self.assertRaisesRegex(ValueError, "evaluation-mode"):
            worker.warmup_reader(SimpleNamespace(model=SimpleNamespace(training=True)), 750)

    def test_offsets_include_segments_not_selected_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.tsv"
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["original_clip_id", "segment_id", "video_path", "frames"], delimiter="\t")
                writer.writeheader()
                for index, frames in [(2, 4), (0, 7), (1, 3)]:
                    writer.writerow(dict(original_clip_id="pretrain/id/00001", segment_id=index,
                                         video_path=f"/video/id_00001_{index}.mp4", frames=frames))
            offsets = complete_segment_offsets(path, {"pretrain/id/00001"})
            self.assertEqual(offsets[("pretrain/id/00001", "id_00001_2")], (10, 4, 14))

    def test_missing_landmarks_do_not_become_resized_faces(self):
        with self.assertRaisesRegex(ValueError, "prohibited"):
            interpolate_landmarks([None] * 10)
        with self.assertRaisesRegex(ValueError, "coverage"):
            interpolate_landmarks([np.ones((68, 2))] + [None] * 20)

    def test_crop_preserves_short_sequences_and_mouth_location(self):
        mean = np.full((68, 2), [128., 175.], dtype=np.float32)
        for i, xy in zip([33, 36, 39, 42, 45], [[128, 140], [80, 90], [105, 90], [150, 90], [175, 90]]):
            mean[i] = xy
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[174:177, 127:130] = 255
        for count in [1, 7, 12, 15]:
            crops, metrics = crop_mouth([frame] * count, np.stack([mean] * count), mean)
            self.assertEqual(len(crops), count)
            self.assertEqual(crops[0].shape, (96, 96))
            self.assertEqual(int(crops[0][48, 48]), 255)
            self.assertLess(metrics["alignment_residual_mean_px"], 1e-4)

    def test_finite_but_wrong_cache_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npy"
            np.save(path, np.zeros((5, 1024), dtype=np.float32))
            valid_array(path, 5)
            with self.assertRaises(ValueError):
                valid_array(path, 6)
            array = np.zeros((5, 1024), dtype=np.float32)
            array[1, 0] = np.nan
            np.save(path, array)
            with self.assertRaises(ValueError):
                valid_array(path, 5)

    def test_equal_frame_count_does_not_hide_wrong_temporal_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (224, 224))
            generator = np.random.default_rng(12)
            for _ in range(5):
                frame = generator.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            frames = read_frames(path, 5, (224, 224))
            metrics = verify_original_frame_mapping(path, frames[2:4], 2, 5)
            self.assertEqual(metrics["landmark_mapping_evidence"], "decoded_original_first_middle_last_frames")
            with self.assertRaisesRegex(ValueError, "temporal mapping"):
                verify_original_frame_mapping(path, frames[2:4], 1, 5)

    def test_explicit_decoder_threads_preserve_reader_pixels_and_frame_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 96))
            generator = np.random.default_rng(74)
            for _ in range(17):
                writer.write(generator.integers(0, 256, (96, 96, 3), dtype=np.uint8))
            writer.release()
            previous = worker.DECODE_THREADS
            try:
                worker.DECODE_THREADS = 0
                expected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                     for frame in worker.read_frames(path, 17, (96, 96))])
                for count in (1, 2, 4):
                    worker.DECODE_THREADS = count
                    actual = worker.load_grayscale_video(path)
                    self.assertEqual(actual.shape, (17, 96, 96))
                    self.assertEqual(actual.dtype, np.uint8)
                    np.testing.assert_array_equal(actual, expected)
            finally:
                worker.DECODE_THREADS = previous


if __name__ == "__main__":
    unittest.main()
