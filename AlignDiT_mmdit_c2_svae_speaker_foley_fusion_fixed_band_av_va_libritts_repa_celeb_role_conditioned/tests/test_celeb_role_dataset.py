import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aligndit.model.celeb_role_dataset import PairedCelebVDubDataset
from aligndit.model.semantic_vae_dataset import sha256_file


class FrozenPairMapTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pair_path = self.root / "pairs.jsonl"
        self.manifest = self.root / "train.jsonl"
        self.keys = [f"celebvdub/train/v/5_0_{i}" for i in range(3)]
        records = [
            {"utterance_key": k, "video_id": "v", "split": "train", "ctc_feasible_40hz": True} for k in self.keys
        ]
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.base = SimpleNamespace(
            records=records,
            manifest_path=self.manifest,
            get_frame_len=[8, 11, 15].__getitem__,
            speaker_embedding_contract={"metadata_sha256": "cam"},
        )

    def make_bundle(self, target=None, ref=None, excluded=None):
        t, r, x = self.keys
        pairs = [{"target_key": target or t, "reference_key": ref or r, "cosine": 0.7}]
        excluded = [r, x] if excluded is None else excluded
        exclusions = [{"target_key": key, "reason": "test"} for key in excluded]
        for path, rows in ((self.pair_path, pairs), (self.root / "exclusions.jsonl", exclusions)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.root / "summary.json").write_text(
            json.dumps(
                {
                    "accepted": 1,
                    "excluded": len(exclusions),
                    "settings": {"min_cosine": 0.65},
                    "provenance": {"cam_metadata_sha256": "cam"},
                    "train_manifest_sha256": sha256_file(self.manifest),
                    "pairs_sha256": sha256_file(self.pair_path),
                    "exclusions_sha256": sha256_file(self.root / "exclusions.jsonl"),
                }
            )
        )

    def test_frame_budget_counts_reference_and_target(self):
        self.make_bundle()
        dataset = PairedCelebVDubDataset(self.base, self.pair_path)
        self.assertEqual(dataset.get_frame_len(0), 19)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.ctc_feasible_count, 1)

    def test_rejects_self_reference_and_test_reference(self):
        for ref in (self.keys[0], self.keys[1].replace("/train/", "/test/")):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                self.make_bundle(ref=ref)
                PairedCelebVDubDataset(self.base, self.pair_path)

    def test_duplicates_or_missing_exclusions_cannot_change_training_coverage(self):
        for invalid in ([self.keys[1], self.keys[1]], [self.keys[0], self.keys[1]], []):
            with self.subTest(excluded=invalid), self.assertRaisesRegex(ValueError, "partition"):
                self.make_bundle(excluded=invalid)
                PairedCelebVDubDataset(self.base, self.pair_path)

    def test_edited_bundle_is_rejected(self):
        self.make_bundle()
        with self.pair_path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "pairs_sha256"):
            PairedCelebVDubDataset(self.base, self.pair_path)
