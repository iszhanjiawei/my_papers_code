"""Exercise the real Chem update loop: uninterrupted versus checkpoint-resumed CPU training."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aligndit.model.trainer_chem import ChemAdaptiveBandRepaTrainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(64, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 64))

    def forward(self, inp, **kwargs):
        prediction = self.net(inp + torch.randn_like(inp) * 0.01)
        loss = prediction.square().mean() * (0.9 + 0.05 * random.random() + 0.05 * np.random.rand())
        return loss, {"diff_loss": float(loss.detach())}, inp, prediction


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def get_frame_len(self, index):
        return 8

    def __getitem__(self, index):
        return torch.full((64, 8), float(index + 1) / 10)

    @staticmethod
    def collate_fn(batch):
        count = len(batch)
        return {
            "mel": torch.stack(batch),
            "mel_lengths": torch.full((count,), 8),
            "video": torch.zeros(count, 8, 1024),
            "video_lengths": torch.full((count,), 8),
            "text": ["a"] * count,
            "text_lengths": torch.ones(count, dtype=torch.long),
        }


class TinyTrainer(ChemAdaptiveBandRepaTrainer):
    def _before_update(self, global_update):
        pass

    def _forward_diagnostics(self, loss, loss_components):
        assert torch.isfinite(loss)
        return {}

    def _clip_gradients(self):
        return float(self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))


def seed(value):
    torch.manual_seed(value)
    random.seed(value)
    np.random.seed(value)


def train(root, limit, initial_seed, write_contract=True):
    seed(initial_seed)
    root.mkdir(exist_ok=True)
    if write_contract:
        (root / "speaker_training_contract.json").write_text('{"test":"exact Chem CPU resume","horizon":5}\n')
    trainer = TinyTrainer(
        TinyModel(),
        scheduler_total_updates=5,
        epochs=200,
        learning_rate=5e-5,
        num_warmup_updates=2,
        save_per_updates=2,
        keep_last_n_checkpoints=-1,
        checkpoint_path=str(root),
        batch_size_per_gpu=24,
        batch_size_type="frame",
        max_samples=3,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        last_per_updates=2,
        log_samples=False,
        ema_kwargs={"beta": 0.999},
        accelerate_kwargs={"cpu": True},
        parent_contract_path="unused",
        expected_parent_sha256="unused",
        expected_parent_size=0,
        expected_parent_contract_sha256="unused",
        ctc_target_lambda=0.03,
        ctc_warmup_start=10000,
        ctc_warmup_end=30000,
    )
    trainer.run_until_update = limit
    trainer.train(TinyDataset(), num_workers=0, resumable_with_seed=666)
    return torch.load(root / "model_last.pt", weights_only=True, map_location="cpu")


def assert_equal(a, b, key="state"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), key
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), key
        for item in a:
            assert_equal(a[item], b[item], f"{key}.{item}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), key
        for index, (left, right) in enumerate(zip(a, b)):
            assert_equal(left, right, f"{key}[{index}]")
    else:
        assert a == b, f"{key}: {a!r} != {b!r}"


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="chem_resume_test_") as directory:
        root = Path(directory)
        continuous = train(root / "continuous", 5, 666)
        part = train(root / "resumed", 3, 666)
        assert part["update"] == 3 and part["data_position"] == {"epoch": 1, "batch": 1}
        resumed = train(root / "resumed", 5, 12345)
        assert_equal(continuous, resumed)
        assert continuous["update"] == 5
        assert (root / "continuous/model_2.pt").is_file() and (root / "continuous/model_4.pt").is_file()
        assert not (root / "continuous/model_6.pt").exists()
        # A changed immutable contract must fail before loading optimizer/model state.
        (root / "resumed/speaker_training_contract.json").write_text('{"changed": true}')
        try:
            train(root / "resumed", 5, 12345, write_contract=False)
        except RuntimeError as error:
            assert "training_contract_sha256" in str(error)
        else:
            raise AssertionError("Changed checkpoint contract was silently accepted")
        print(
            json.dumps(
                {
                    "passed": True,
                    "changed_contract_rejected": True,
                    "updates": 5,
                    "resume_update": 3,
                    "exact_model_optimizer_scheduler_ema_rng": True,
                    "crosses_epoch_boundary": True,
                }
            )
        )


if __name__ == "__main__":
    main()
