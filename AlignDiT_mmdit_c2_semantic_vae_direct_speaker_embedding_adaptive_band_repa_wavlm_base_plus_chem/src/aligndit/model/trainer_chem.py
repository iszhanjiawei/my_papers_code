"""Chem-only update horizon and complete, atomic recovery for the copied experiment."""

from __future__ import annotations

import gc
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch

from aligndit.model.trainer_semantic_vae_adaptive_band import SemanticVaeAdaptiveBandTrainer


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class ChemAdaptiveBandRepaTrainer(SemanticVaeAdaptiveBandTrainer):
    """Retain model/optimizer/EMA semantics while scheduling the Chem run by updates."""

    def __init__(self, *args, scheduler_total_updates: int, **kwargs):
        self.scheduler_total_updates = int(scheduler_total_updates)
        self.isolated_dataloader_rng = True
        if int(kwargs.get("grad_accumulation_steps", 1)) != 1:
            raise ValueError("The Chem exact-resume contract currently requires gradient accumulation = 1")
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("This Chem launcher and RNG checkpoint contract require a single GPU")

    def _contract_digest(self) -> str:
        path = Path(self.checkpoint_path) / "speaker_training_contract.json"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if not self.is_main or (not last and self.keep_last_n_checkpoints == 0):
            return
        path = Path(self.checkpoint_path) / ("model_last.pt" if last else f"model_{update}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_schema_version": 2,
            "dataset": "Chem",
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "update": int(update),
            "scheduler_total_updates": self.scheduler_total_updates,
            "batches_per_epoch": self.batches_per_epoch,
            "data_position": {"epoch": update // self.batches_per_epoch, "batch": update % self.batches_per_epoch},
            "training_contract_sha256": self._contract_digest(),
            "rng_state": capture_rng(),
        }
        temporary = path.with_suffix(".pt.tmp")
        self.accelerator.save(checkpoint, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
        if self.logger == "tensorboard":
            self.writer.flush()
        print(f"Saved complete Chem checkpoint at update {update}: {path}", flush=True)
        if not last and self.keep_last_n_checkpoints > 0:
            numbered = sorted(path.parent.glob("model_[0-9]*.pt"), key=lambda item: int(item.stem.split("_")[1]))
            for old in numbered[: -self.keep_last_n_checkpoints]:
                old.unlink()

    def load_checkpoint(self):
        root = Path(self.checkpoint_path)
        candidates = list(root.glob("model_[0-9]*.pt"))
        last = root / "model_last.pt"
        if last.is_file():
            candidates.append(last)
        if not candidates:
            return 0
        # mmap reads the small state header without loading all multi-GB tensors.
        candidates_with_updates = []
        for path in candidates:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            candidates_with_updates.append((int(state["update"]), path))
            del state
        update, path = max(candidates_with_updates, key=lambda pair: (pair[0], pair[1].name == "model_last.pt"))
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected = {
            "checkpoint_schema_version": 2,
            "dataset": "Chem",
            "scheduler_total_updates": self.scheduler_total_updates,
            "batches_per_epoch": self.batches_per_epoch,
            "training_contract_sha256": self._contract_digest(),
            "data_position": {"epoch": update // self.batches_per_epoch, "batch": update % self.batches_per_epoch},
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"Chem resume contract mismatch for {key}: expected {value}, got {state.get(key)}")
        self.accelerator.unwrap_model(self.model).load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.ema_model.load_state_dict(state["ema_model_state_dict"], strict=True)
        restore_rng(state["rng_state"])
        del state
        gc.collect()
        print(
            f"Resumed Chem model, AdamW, scheduler, EMA, data position and RNG at update {update}: {path}", flush=True
        )
        return update
