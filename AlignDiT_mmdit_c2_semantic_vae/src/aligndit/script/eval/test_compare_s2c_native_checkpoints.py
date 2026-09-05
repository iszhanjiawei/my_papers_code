"""CPU regression tests for longitudinal paired units and fixture isolation."""

import tempfile
import unittest
from pathlib import Path

from aligndit.script.eval.compare_s2c_native_checkpoints import clone_fixtures, temporal_pair


class LongitudinalTests(unittest.TestCase):
    def rows(self, offset=0.0):
        return [
            {
                "branch": "svae",
                "utterance_key": str(i),
                "speaker_id": str(i),
                "subset": "dev-clean" if i < 2 else "dev-other",
                "sampling_seed": seed,
                "score": float(i + seed - 666) + offset,
            }
            for i in range(4)
            for seed in (666, 667, 668)
        ]

    def test_delta_is_later_minus_earlier_after_seed_average(self):
        result = temporal_pair(self.rows(), self.rows(0.125))["overall"]
        self.assertEqual(result["utterances"], 4)
        score = result["metrics"]["score"]
        self.assertEqual(score["earlier_mean"], 2.5)
        self.assertEqual(score["later_mean"], 2.625)
        self.assertEqual(score["later_minus_earlier"], 0.125)
        self.assertEqual(score["delta_ci95"], [0.125, 0.125])

    def test_incomplete_later_seeds_are_not_silently_averaged(self):
        with self.assertRaisesRegex(ValueError, "Incomplete seed"):
            temporal_pair(self.rows(), self.rows()[:-1])

    def test_fixtures_are_independent_copies_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            reference = base / "reference"
            for name in ("common", "svae_context_canary2", "mel_canary2", "svae_context", "mel"):
                path = reference / name / "artifact"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"original")
            target = base / "candidate"
            clone_fixtures(reference, target)
            for path in target.glob("*/artifact"):
                original = reference / path.relative_to(target)
                self.assertEqual(path.read_bytes(), original.read_bytes())
                self.assertNotEqual(path.stat().st_ino, original.stat().st_ino)
                self.assertFalse(path.is_symlink())
            (target / "common/artifact").write_bytes(b"candidate only")
            self.assertEqual((reference / "common/artifact").read_bytes(), b"original")
            with self.assertRaises(FileExistsError):
                clone_fixtures(reference, target)


if __name__ == "__main__":
    unittest.main()
