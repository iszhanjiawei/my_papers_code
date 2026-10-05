"""GRID update-bounded training with atomic, experiment-bound checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from aligndit.model.trainer_semantic_vae_adaptive_band import SemanticVaeAdaptiveBandTrainer


class GridEvenBatchSampler:
    """Pad each shuffled epoch to equally many batches per distributed rank.

    Every original batch remains present. At most world_size - 1 batches are
    repeated from the start of that epoch's deterministic shuffle.
    """

    def __init__(self, batch_sampler, world_size):
        self.batch_sampler = batch_sampler
        self.world_size = int(world_size)
        self.drop_last = True
        if self.world_size <= 0 or len(batch_sampler) == 0:
            raise ValueError("GRID distributed batch sampler requires nonempty batches and a positive world size")

    def __len__(self):
        return math.ceil(len(self.batch_sampler) / self.world_size) * self.world_size

    def set_epoch(self, epoch):
        self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        batches = list(self.batch_sampler)
        missing = len(self) - len(batches)
        return iter(batches + [batches[index % len(batches)] for index in range(missing)])


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        # DDP workers must not initialize CUDA contexts on other ranks' GPUs.
        "cuda": [torch.cuda.get_rng_state()] if torch.cuda.is_initialized() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state(state["cuda"][0])


class GridSemanticVaeTrainer(SemanticVaeAdaptiveBandTrainer):
    def _prepare_batch_sampler(self, batch_sampler):
        return GridEvenBatchSampler(batch_sampler, self.accelerator.num_processes)

    def __init__(self, *args, total_updates, run_until_update, training_contract, **kwargs):
        self.total_optimizer_updates = int(total_updates)
        self.run_until_update = int(run_until_update)
        if not 0 < self.run_until_update <= self.total_optimizer_updates:
            raise ValueError("Require 0 < run_until_update <= total_updates")
        if int(kwargs.get("grad_accumulation_steps", 1)) != 1:
            raise ValueError("GRID resumable update schedule currently requires grad_accumulation_steps=1")
        if self.total_optimizer_updates <= int(kwargs.get("num_warmup_updates", 20000)):
            raise ValueError("total_updates must exceed the configured LR warmup")
        super().__init__(*args, **kwargs)
        self.training_contract = dict(training_contract, world_size=self.accelerator.num_processes)
        self.training_contract_sha256 = contract_hash(self.training_contract)
        contract_path = Path(self.checkpoint_path) / "grid_training_contract.json"
        if self.is_main:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            if contract_path.exists():
                previous = json.loads(contract_path.read_text())
                if previous != self.training_contract:
                    raise RuntimeError(f"GRID checkpoint directory belongs to a different run: {contract_path}")
            else:
                if any(contract_path.parent.glob("*.pt")):
                    raise RuntimeError("Refusing unbound existing weights in a fresh GRID output directory")
                temporary = contract_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(self.training_contract, indent=2, ensure_ascii=False) + "\n")
                os.replace(temporary, contract_path)
        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, update, last=False):
        if not last and self.keep_last_n_checkpoints == 0:
            return
        self.accelerator.wait_for_everyone()
        state = random_state()
        states = [state]
        if self.accelerator.num_processes > 1:
            states = [None] * self.accelerator.num_processes
            torch.distributed.all_gather_object(states, state)
        if self.is_main:
            root = Path(self.checkpoint_path)
            root.mkdir(parents=True, exist_ok=True)
            path = root / ("model_last.pt" if last else f"model_{update}.pt")
            temporary = path.with_suffix(".pt.tmp")
            checkpoint = {
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_model_state_dict": self.ema_model.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "update": int(update),
                "grid_checkpoint_schema": 1,
                "training_contract_sha256": self.training_contract_sha256,
                "rng_states": states,
            }
            torch.save(checkpoint, temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, path)
            print(f"Saved GRID checkpoint at update {update}: {path}", flush=True)
            if not last and self.keep_last_n_checkpoints > 0:
                numbered = sorted(
                    (p for p in root.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()),
                    key=lambda p: int(p.stem.removeprefix("model_")),
                )
                for old in numbered[: -self.keep_last_n_checkpoints]:
                    old.unlink()
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self):
        root = Path(self.checkpoint_path)
        last = root / "model_last.pt"
        numbered = sorted(
            (p for p in root.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()),
            key=lambda p: int(p.stem.removeprefix("model_")),
        )
        path = last if last.is_file() else (numbered[-1] if numbered else None)
        if path is None:
            if any(root.glob("*.pt")) or any(root.glob("*.safetensors")):
                raise RuntimeError("GRID output directory contains unrecognized checkpoints")
            return 0
        if path.is_symlink():
            raise RuntimeError(f"GRID resume checkpoint must be a regular file: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("grid_checkpoint_schema") != 1
            or checkpoint.get("training_contract_sha256") != self.training_contract_sha256
            or len(checkpoint.get("rng_states", [])) != self.accelerator.num_processes
        ):
            raise RuntimeError(f"Checkpoint does not match this GRID training contract: {path}")
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=True)
        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_random_state(checkpoint["rng_states"][self.accelerator.process_index])
        update = int(checkpoint["update"])
        del checkpoint
        gc.collect()
        if self.is_main:
            print(f"Restored GRID model/EMA/optimizer/scheduler/RNG at update {update}", flush=True)
        return update
