"""CPU regressions for no-drop distributed batching and exact checkpoint resume."""

import tempfile
import signal
import unittest
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from aligndit.model.trainer_text_audio import TextAudioTrainer, create_text_audio_ema, ctc_weight, rank_batches


class TinyDataset(Dataset):
    def __len__(self):
        return 9

    def get_frame_len(self, index):
        return 2 + index % 2

    def __getitem__(self, index):
        return {"audio": torch.full((self.get_frame_len(index), 2), float(index + 1) / 10),
                "text": "ab", "speaker_embedding": torch.ones(3)}

    @staticmethod
    def collate_fn(items):
        return {"audio": pad_sequence([item["audio"] for item in items], batch_first=True),
                "lens": torch.tensor([len(item["audio"]) for item in items]),
                "text": [item["text"] for item in items],
                "speaker_embedding": torch.stack([item["speaker_embedding"] for item in items])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
        self.ctc_lambda = 0.0

    def forward(self, audio, text, lens, speaker_embedding):
        pred = audio * self.weight + torch.randn_like(audio) * 0.05
        loss = (pred - 0.3).square().mean()
        return loss, {"diff_loss": float(loss.detach()), "ctc_loss": 0.0}, audio, pred


def configuration(output, stop, resume=None):
    return OmegaConf.create({
        "seed": 666,
        "optim": {"grad_accumulation_steps": 2, "max_updates": 4, "stop_after_updates": stop,
                  "learning_rate": 0.001, "betas": [0.9, 0.99], "weight_decay": 0.01,
                  "num_warmup_updates": 2, "max_grad_norm": 1.0},
        "runtime": {"mixed_precision": "no"},
        "ema": {"beta": 0.99},
        "datasets": {"batch_size_per_gpu": 6, "max_samples": 2, "num_workers": 0},
        "model": {"ctc_lambda": 0.0, "ctc_warmup_start": 1, "ctc_warmup_end": 3},
        "logging": {"every_updates": 1},
        "validation": {"every_updates": 1, "seed": 123, "max_batches": 0},
        "ckpts": {"save_dir": str(output), "resume": str(resume) if resume else None,
                  "keep_last_n_checkpoints": 3, "save_per_updates": 2},
    })


class TrainerTests(unittest.TestCase):
    def test_ema_preserves_real_model_state_after_device_style_buffer_conversion(self):
        from aligndit.script.misc.test_text_audio_model import cfm, inputs, make_model

        model = cfm(make_model(conv_layers=4)).double()
        visual_null = model.transformer.video_embed.vid_null_emb
        self.assertFalse(visual_null.is_leaf)
        self.assertTrue(visual_null.requires_grad)
        original_state = model.state_dict()
        ema = create_text_audio_ema(model, 0.999)
        self.assertIs(model.transformer.video_embed.vid_null_emb, visual_null)
        self.assertTrue(visual_null.requires_grad)
        self.assertEqual(original_state.keys(), ema.ema_model.state_dict().keys())
        for name, value in ema.ema_model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]), name)
            self.assertNotEqual(value.data_ptr(), original_state[name].data_ptr(), name)
        self.assertTrue(all(not value.requires_grad and value.is_leaf for value in ema.ema_model.buffers()))
        self.assertTrue(all(not value.requires_grad for value in ema.ema_model.parameters()))
        arguments = {key: value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                     for key, value in inputs().items()}
        with torch.no_grad():
            expected, _ = model.transformer(**arguments)
            actual, _ = ema.ema_model.transformer(**arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            model.transformer.proj_out.weight.add_(0.1)
        ema.update()
        self.assertEqual(int(ema.step), 1)
        self.assertTrue(torch.equal(ema.ema_model.transformer.proj_out.weight, model.transformer.proj_out.weight))

    def test_distributed_batches_keep_every_sample_and_equal_steps(self):
        batches = [[i] for i in range(11)]
        shards = [rank_batches(batches, rank, 3, accumulation=2)[0] for rank in range(3)]
        self.assertEqual([len(shard) for shard in shards], [4, 4, 4])
        self.assertEqual({i for shard in shards for batch in shard for i in batch}, set(range(11)))
        self.assertEqual(rank_batches(batches, 0, 3, accumulation=2)[1], 1)
        validation = [rank_batches(batches, rank, 3, training=False)[0] for rank in range(3)]
        self.assertEqual(sum(len(shard) for shard in validation), 11)

    def test_ctc_schedule(self):
        self.assertEqual(ctc_weight(10000, 0.03, 10000, 30000), 0)
        self.assertAlmostEqual(ctc_weight(20000, 0.03, 10000, 30000), 0.015)
        self.assertEqual(ctc_weight(50000, 0.03, 10000, 30000), 0.03)

    def run_tiny(self, output, stop, resume=None):
        set_seed(666)
        cfg = configuration(output, stop, resume)
        accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
        trainer = TextAudioTrainer(TinyModel(), cfg, {"fixture": "exact-resume-v1"}, accelerator=accelerator)
        trainer.train(TinyDataset(), TinyDataset())
        return torch.load(output / f"model_{stop:08d}.pt", weights_only=True, map_location="cpu")

    def test_full_state_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            whole = self.run_tiny(root / "whole", 4)
            self.run_tiny(root / "resumed", 2)
            resumed = self.run_tiny(root / "resumed", 4, root / "resumed" / "model_00000002.pt")
            for key in ("model_state_dict", "ema_model_state_dict"):
                self.assertEqual(whole[key].keys(), resumed[key].keys())
                for name in whole[key]:
                    self.assertTrue(torch.equal(whole[key][name], resumed[key][name]), (key, name))
            self.assertEqual(whole["scheduler_state_dict"], resumed["scheduler_state_dict"])
            self.assertEqual((whole["epoch"], whole["batch_in_epoch"]),
                             (resumed["epoch"], resumed["batch_in_epoch"]))
            self.assertTrue(torch.equal(whole["rng_states"][0]["torch"], resumed["rng_states"][0]["torch"]))
            state_a = next(iter(whole["optimizer_state_dict"]["state"].values()))
            state_b = next(iter(resumed["optimizer_state_dict"]["state"].values()))
            for key in state_a:
                self.assertTrue(torch.equal(state_a[key], state_b[key]), key)

    def test_signal_saves_at_completed_update_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = configuration(root, 4)
            accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
            trainer = TextAudioTrainer(TinyModel(), cfg, {"fixture": "signal-v1"}, accelerator=accelerator)
            trainer._signal(signal.SIGTERM, None)
            trainer.train(TinyDataset(), TinyDataset())
            payload = torch.load(root / "model_00000001.pt", weights_only=True, map_location="cpu")
            self.assertEqual(payload["save_reason"], "signal")
            self.assertEqual(payload["batch_in_epoch"], 2)
            self.assertEqual(int(payload["ema_model_state_dict"]["step"]), 1)

    def test_resume_refuses_changed_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_tiny(root / "original", 2)
            cfg = configuration(root / "different", 4, root / "original" / "model_00000002.pt")
            accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
            trainer = TextAudioTrainer(TinyModel(), cfg, {"fixture": "changed-vocabulary"}, accelerator=accelerator)
            with self.assertRaisesRegex(ValueError, "Resume contract differs"):
                trainer._resume()


if __name__ == "__main__":
    unittest.main()
