"""Data contracts that prevent cross-clip conditioning and padded resampling."""

import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, SyncMLP
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import load_setting1_synchformer_features


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


class SynchformerPipelineTest(unittest.TestCase):
    def test_mixed_duration_batch_keeps_independent_token_lengths(self):
        batch = SemanticVaeCelebVDubDataset.collate_fn([item(40, 8, 1.0), item(80, 24, 2.0)])
        self.assertEqual(batch["sync_lens"].tolist(), [8, 24])
        self.assertEqual(batch["mel_lengths"].tolist(), [40, 80])
        self.assertEqual(batch["sync_feat"].shape, (2, 24, 768))
        self.assertTrue(torch.equal(batch["sync_feat"][0, :8], torch.ones(8, 768)))
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

    def test_setting1_prefix_preserves_target_alignment_after_learned_projection(self):
        # A nonzero MLP and bias expose accidental prompt leakage that the
        # zero-initialized production projection could otherwise conceal.
        torch.manual_seed(9)
        harness = SimpleNamespace(
            sync_dim=768,
            sync_pos_emb=torch.randn(1, 8, 768),
            sync_in=torch.nn.Sequential(torch.nn.Linear(768, 32), torch.nn.SiLU(), SyncMLP(32)),
        )
        feature = torch.randn(1, 24, 768)
        time = torch.randn(1, 32)
        target_audio, target_video = DiT_VT_MMDiT.get_sync_deltas(
            harness, feature, torch.tensor([24]), time, audio_len=73, video_len=73,
        )
        combined = torch.cat((torch.zeros_like(feature), feature), dim=1)
        prompt_mask = torch.arange(146).unsqueeze(0) < 73
        combined_audio, combined_video = DiT_VT_MMDiT.get_sync_deltas(
            harness, combined, torch.tensor([48]), time,
            audio_len=146, video_len=146, complementary_mask=prompt_mask,
        )
        self.assertEqual(combined_audio[:, :73].count_nonzero().item(), 0)
        self.assertEqual(combined_video[:, :73].count_nonzero().item(), 0)
        torch.testing.assert_close(combined_audio[:, 73:], target_audio, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(combined_video[:, 73:], target_video, rtol=1e-6, atol=1e-7)

    def test_historical_inference_config_does_not_require_sync_cache(self):
        config = OmegaConf.create({"model": {"arch": {}}})
        self.assertEqual(load_setting1_synchformer_features(config, [{}, {}]), ([None, None], None))


if __name__ == "__main__":
    unittest.main()
