import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aligndit.script.misc import prepare_celeb_role_pairs as pairing


class RolePairPreparationTest(unittest.TestCase):
    @staticmethod
    def clip(stem, text, seconds=3.0):
        return pairing.parse_clip(
            {
                "utterance_key": f"celebvdub/train/id_with_underscore/{stem}",
                "split": "train",
                "video_id": "id_with_underscore",
                "audio_relative_path": f"train/id_with_underscore/{stem}.wav",
                "text": text,
                "duration_seconds": seconds,
            }
        )

    @staticmethod
    def vectors(clips):
        return {clip.key: np.array([1, i / 20]) for i, clip in enumerate(clips)}

    def test_group_split_and_normalization(self):
        clip = self.clip("2_1_8", "Isn’t THAT nice!")
        self.assertEqual(clip.group, ("id_with_underscore", "2_1"))
        self.assertEqual(clip.words, ("isnt", "that", "nice"))
        with self.assertRaises(ValueError):
            pairing.parse_clip({"utterance_key": "celebvdub/test/id/2_1_8", "split": "test"})

    def test_text_overlap(self):
        for left, right in (
            ("how are you", "HOW are you?"),
            ("fine", "I am fine"),
            ("what did you see today", "well what did you see yesterday"),
            ("this is the red car parked outside", "this was the red car parked outside"),
            ("", "hello there"),
        ):
            with self.subTest(left=left, right=right):
                self.assertTrue(pairing.overlapping_text(self.clip("1_0", left).words, self.clip("1_1", right).words))
        self.assertFalse(pairing.overlapping_text(("bright", "blue", "car"), ("the", "bird", "sings")))

    def test_stable_cross_utterance_selection_and_orientation(self):
        short = self.clip("0_0", "I prefer birds", seconds=1.0)
        long = self.clip("0_1", "Where is the pencil", seconds=12.0)
        ref_a = self.clip("0_2", "Look at her dance", seconds=2.0)
        ref_b = self.clip("0_3", "Mail me your letter", seconds=10.0)
        clips = [short, long, ref_a, ref_b]
        vectors = self.vectors(clips)
        pairs, exclusions = pairing.pair_group(clips, vectors)
        by_target = {p["target_key"]: p["reference_key"] for p in pairs}
        self.assertEqual(exclusions, [])
        self.assertEqual(
            by_target[short.key], min([ref_a.key, ref_b.key], key=lambda k: pairing.reference_priority(short.key, k))
        )
        self.assertEqual(by_target[ref_b.key], ref_a.key)
        self.assertEqual(by_target[ref_a.key], ref_b.key)
        self.assertEqual(
            sorted(pairs, key=lambda p: p["target_key"]),
            sorted(pairing.pair_group(clips[::-1], vectors)[0], key=lambda p: p["target_key"]),
        )

    def test_ineligible_pairs_never_fall_back_to_self_or_different_voice(self):
        left = self.clip("0_0", "go and get the book")
        right = self.clip("0_1", "pass me the sheets")
        singleton = pairing.pair_group([left], self.vectors([left]))[1]
        self.assertEqual(singleton[0]["reason"], "no_distinct_clip_in_prefix")
        for other, right_vector, reason in (
            (self.clip("0_1", "pass me the sheets", 1.99), [1, 0.1], "no_reference_in_duration_range"),
            (right, [0, 1], "no_reference_passing_speaker_cosine"),
            (right, [1, 0], "only_identical_speaker_vectors"),
            (self.clip("0_1", "go and get the book now"), [1, 0.1], "overlapping_or_empty_transcript"),
        ):
            with self.subTest(reason=reason):
                _, excluded = pairing.pair_group([left, other], {left.key: [1, 0], other.key: right_vector})
                self.assertEqual(next(e["reason"] for e in excluded if e["target_key"] == left.key), reason)

    def test_atomic_publication_detects_conflict_and_discards_partial_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"

            def build(_root, stage, _workers):
                (stage / "pairs.jsonl").write_text("selected")
                return {"accepted": 1}

            with patch.object(pairing, "write_bundle", side_effect=build):
                pairing.prepare(root, output, 1)
                pairing.prepare(root, output, 1)
                (output / "pairs.jsonl").write_text("preexisting")
                with self.assertRaises(OSError):
                    pairing.prepare(root, output, 1)
                self.assertEqual((output / "pairs.jsonl").read_text(), "preexisting")

            def fail(_root, stage, _workers):
                (stage / "partial").write_text("pending")
                raise ValueError("wrong contract")

            with patch.object(pairing, "write_bundle", side_effect=fail), self.assertRaises(ValueError):
                pairing.prepare(root, root / "missing", 1)
            self.assertEqual(list(root.iterdir()), [output])
