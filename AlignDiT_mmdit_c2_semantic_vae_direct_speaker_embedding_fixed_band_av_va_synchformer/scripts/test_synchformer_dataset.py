"""Dataset contracts for native Synchformer lengths and immutable clip identity."""

import unittest

import torch

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset


def item(frames, tokens, value):
    return {
        "ctc_feasible": True,
        "ctc_target_length": 1,
        "mel_spec": torch.full((64, frames), value),
        "text": "a",
        "utterance_key": f"celebvdub/train/video{int(value)}/0_0_0",
        "video": torch.full((frames, 1024), value),
        "speaker_embedding": torch.full((192,), value),
        "sync_feat": torch.full((tokens, 768), value),
    }


class SynchformerDatasetTest(unittest.TestCase):
    def test_mixed_duration_batch_keeps_independent_token_lengths(self):
        batch = SemanticVaeCelebVDubDataset.collate_fn([item(40, 8, 1.0), item(80, 24, 2.0)])
        self.assertEqual(batch["sync_lens"].tolist(), [8, 24])
        self.assertEqual(batch["mel_lengths"].tolist(), [40, 80])
        self.assertEqual(batch["sync_feat"].shape, (2, 24, 768))
        torch.testing.assert_close(batch["sync_feat"][0, :8], torch.ones(8, 768))
        self.assertEqual(batch["sync_feat"][0, 8:].count_nonzero().item(), 0)
        self.assertEqual(batch["speaker_embedding"].shape, (2, 192))

    def test_enabled_condition_cannot_silently_disappear(self):
        first, second = item(40, 8, 1.0), item(80, 24, 2.0)
        del second["sync_feat"]
        with self.assertRaisesRegex(RuntimeError, "every sample"):
            SemanticVaeCelebVDubDataset.collate_fn([first, second])

    def test_partial_segments_fail(self):
        with self.assertRaisesRegex(ValueError, "multiples of 8"):
            SemanticVaeCelebVDubDataset.collate_fn([item(40, 9, 1.0)])

    def test_same_basename_in_different_videos_has_distinct_identity(self):
        records = [
            {"utterance_key": f"celebvdub/train/{video}/0_0_0", "split": "train",
             "audio_relative_path": f"train/{video}/0_0_0.wav",
             "video_relative_path": f"train/{video}/0_0_0.npy"}
            for video in ["first", "second"]
        ]
        keys = [SemanticVaeCelebVDubDataset._synchformer_clip_key(row) for row in records]
        self.assertNotEqual(*keys)
        records[0]["video_relative_path"] = records[1]["video_relative_path"]
        with self.assertRaisesRegex(ValueError, "video identity"):
            SemanticVaeCelebVDubDataset._synchformer_clip_key(records[0])

    def test_optional_branch_does_not_change_legacy_collation(self):
        rows = [item(40, 8, 1.0), item(80, 24, 2.0)]
        for row in rows:
            del row["sync_feat"]
        batch = SemanticVaeCelebVDubDataset.collate_fn(rows)
        self.assertNotIn("sync_feat", batch)
        self.assertNotIn("sync_lens", batch)
        self.assertEqual(batch["mel_lengths"].tolist(), [40, 80])
        self.assertEqual(batch["video"].shape, (2, 80, 1024))


if __name__ == "__main__":
    unittest.main()
