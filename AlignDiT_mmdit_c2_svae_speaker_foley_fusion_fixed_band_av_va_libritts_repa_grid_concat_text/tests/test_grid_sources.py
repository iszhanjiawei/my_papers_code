"""GRID regression checks for split leakage, silence and audio/video alignment."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from aligndit.script.misc.grid_sources import (
    SAMPLE_RATE, SPLIT_SUBDIR, SourceValidationError, build_source_records,
    decode_aligned_audio, executable, load_mouth_frames, mouth_frame_paths, normalize_text, probe_video,
)


class GridTextAndAlignmentTests(unittest.TestCase):
    def test_silence_tokens_are_not_spoken_and_id_checks_content(self):
        self.assertEqual(normalize_text("sil bin sp blue at f two now <sil>", "bbaf2n"),
                         "bin blue at f two now")
        with self.assertRaisesRegex(SourceValidationError, "Transcript/ID mismatch"):
            normalize_text("bin blue at f three now", "bbaf2n")

    def _decode(self, values, offset, samples):
        response = subprocess.CompletedProcess([], 0, np.asarray(values, dtype="<f4").tobytes(), b"")
        record = {"video_path": "unused.mpg", "num_samples": samples,
                  "audio_offset_seconds": offset / SAMPLE_RATE}
        with patch("aligndit.script.misc.grid_sources.subprocess.run", return_value=response), \
             patch("aligndit.script.misc.grid_sources.executable", return_value="ffmpeg"):
            return decode_aligned_audio(record)

    def test_audio_start_after_video_adds_only_leading_padding(self):
        np.testing.assert_array_equal(self._decode([1, 2, 3], 2, 6), [0, 0, 1, 2, 3, 0])

    def test_audio_start_before_video_removes_only_leading_excess(self):
        np.testing.assert_array_equal(self._decode([1, 2, 3, 4, 5], -2, 3), [3, 4, 5])

    def test_severely_trimmed_waveform_is_rejected_instead_of_padded(self):
        with self.assertRaisesRegex(SourceValidationError, "excessive alignment"):
            self._decode(np.zeros(16000), 0, 48000)

    def test_decoder_failure_provenance_does_not_change_between_processes(self):
        messages = []
        for address in ("0x7ab123", "0x592def"):
            response = subprocess.CompletedProcess([], 0, "{}", f"[mpeg1video @ {address}] damaged frame")
            with patch("aligndit.script.misc.grid_sources.subprocess.run", return_value=response), \
                 patch("aligndit.script.misc.grid_sources.executable", return_value="ffprobe"):
                with self.assertRaises(SourceValidationError) as raised:
                    probe_video("invalid.mpg")
                messages.append(str(raised.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("damaged frame", messages[0])

    def test_nonfinite_audio_is_rejected(self):
        with self.assertRaisesRegex(SourceValidationError, "non-finite"):
            self._decode([0, float("nan")], 0, 2)


class GridSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ffmpeg = executable("ffmpeg")
            cls.ffprobe = executable("ffprobe")
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / SPLIT_SUBDIR).mkdir(parents=True)
        cls.video = cls.root / "Grid_dataset_Raw/Grid_video/s1/bbaf2n.mpg"
        cls.video.parent.mkdir(parents=True)
        subprocess.run([
            cls.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=96x96:r=25:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
            "-c:v", "mpeg1video", "-c:a", "mp2", "-threads", "1", str(cls.video),
        ], capture_output=True, check=True)
        cls.mouth = cls.root / "Lip_Grid_Gray/Face_my_Grid_Gray/s1-face-bbaf2n"
        cls.mouth.mkdir(parents=True)
        for number in range(1, 76):
            Image.fromarray(np.full((96, 96), number, dtype=np.uint8)).save(cls.mouth / f"{number:06}.jpg")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        (self.root / SPLIT_SUBDIR / "train.txt").write_text(
            "s1-bbaf2n|s1|{IGNORED}|sil bin blue sp at f two now\n")
        (self.root / SPLIT_SUBDIR / "val.txt").write_text("")

    def build(self, **kwargs):
        return build_source_records(self.root, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe, workers=2, **kwargs)

    def test_real_mpg_recovery_uses_full_three_second_timeline(self):
        records, excluded, summary = self.build()
        self.assertFalse(excluded)
        self.assertEqual(summary["accepted"], {"train": 1})
        self.assertTrue(summary["video_decode_validated"])
        self.assertEqual(records[0]["decoded_video_frames"], 75)
        audio = decode_aligned_audio(records[0], ffmpeg=self.ffmpeg)
        self.assertEqual(audio.shape, (48000,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(float(np.square(audio).mean()), 0.001)
        self.assertEqual(load_mouth_frames(records[0]).shape, (75, 96, 96))
        self.assertEqual(len(records[0]["video_sha256"]), 64)
        self.assertEqual(len(records[0]["mouth_sha256"]), 64)

    def test_valid_74_frame_clip_keeps_its_actual_duration(self):
        video = self.video.with_name("bbaf4p.mpg")
        mouth = self.mouth.with_name("s1-face-bbaf4p")
        mouth.mkdir()
        try:
            subprocess.run([
                self.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=96x96:r=25:d=2.96",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2.96",
                "-frames:v", "74", "-c:v", "mpeg1video", "-c:a", "mp2", "-threads", "1", str(video),
            ], capture_output=True, check=True)
            for number in range(1, 75):
                (mouth / f"{number:06}.jpg").write_bytes((self.mouth / f"{number:06}.jpg").read_bytes())
            (self.root / SPLIT_SUBDIR / "val.txt").write_text(
                "s1-bbaf4p|s1|{}|bin blue at f four please\n")
            records, excluded, _ = self.build()
            self.assertFalse(excluded)
            record = next(record for record in records if record["split"] == "val")
            self.assertEqual(record["frame_count"], 74)
            self.assertEqual(record["duration"], 2.96)
            self.assertEqual(decode_aligned_audio(record, ffmpeg=self.ffmpeg).shape, (47360,))
        finally:
            for path in mouth.iterdir():
                path.unlink()
            mouth.rmdir()
            video.unlink(missing_ok=True)

    def test_split_overlap_is_fatal_even_outside_smoke_limit(self):
        (self.root / SPLIT_SUBDIR / "val.txt").write_text(
            "s1-bbaf3s|s1|{}|bin blue at f three soon\n"
            "s1-bbaf2n|s1|{}|bin blue at f two now\n")
        with self.assertRaisesRegex(SourceValidationError, "split-leaked"):
            self.build(limit=1)

    def test_missing_video_is_recorded_without_reassigning_split(self):
        (self.root / SPLIT_SUBDIR / "val.txt").write_text(
            "s1-bbaf3s|s1|{}|bin blue at f three soon\n")
        records, excluded, summary = self.build()
        self.assertEqual(len(records), 1)
        self.assertEqual(excluded[0]["id"], "s1/bbaf3s")
        self.assertEqual(excluded[0]["split"], "val")
        self.assertIn("Missing MPG", excluded[0]["reason"])
        self.assertEqual(summary["excluded"], {"val": 1})

    def test_74_mouth_frames_cannot_be_paired_with_75_video_frames(self):
        frame = self.mouth / "000075.jpg"
        content = frame.read_bytes()
        try:
            frame.unlink()
            records, excluded, _ = self.build()
            self.assertFalse(records)
            self.assertIn("count mismatch", excluded[0]["reason"])
        finally:
            frame.write_bytes(content)

    def test_missing_middle_mouth_frame_is_not_hidden_by_numeric_sort(self):
        frame = self.mouth / "000033.jpg"
        content = frame.read_bytes()
        try:
            frame.unlink()
            with self.assertRaisesRegex(SourceValidationError, "contiguous"):
                mouth_frame_paths(self.mouth)
        finally:
            frame.write_bytes(content)

    def test_corrupt_mouth_image_is_an_explicit_exclusion(self):
        frame = self.mouth / "000002.jpg"
        content = frame.read_bytes()
        try:
            frame.write_bytes(b"not a JPEG")
            records, excluded, _ = self.build()
            self.assertFalse(records)
            self.assertIn("Invalid mouth frame", excluded[0]["reason"])
        finally:
            frame.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
