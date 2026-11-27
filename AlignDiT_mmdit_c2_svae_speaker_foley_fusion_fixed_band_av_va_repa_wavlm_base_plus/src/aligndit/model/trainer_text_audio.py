"""Resumable text/audio training; every dataset item is retained by the sampler.

The dataloader is deliberately not passed to Accelerator.prepare: batches have
already been assigned to ranks here, including explicit, logged tail repeats.
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aligndit.model.libritts_text_audio_dataset import LengthBudgetBatchSampler
from aligndit.model.text_audio_checkpoint import contract_sha256


def canonical_sha256(value):
    return contract_sha256(value)


def create_text_audio_ema(model, beta):
    """Clone an EMA shadow without retaining buffer autograd graphs.

    The inherited visual null embedding is a requires-grad buffer, rather than
    a Parameter. Moving the online model to CUDA makes that buffer non-leaf and
    ordinary deepcopy fails. An EMA needs only its value; supply a detached
    clone in deepcopy's memo without modifying the online model or state keys.
    Pass the shadow explicitly so ema_pytorch cannot swallow deepcopy failures.
    """
    memo = {id(buffer): buffer.detach().clone()
            for buffer in model.buffers() if buffer.requires_grad or not buffer.is_leaf}
    shadow = deepcopy(model, memo)
    return EMA(model, ema_model=shadow, beta=float(beta), update_every=1,
               update_after_step=0, include_online_model=False)


def capture_rng_state(device=None):
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), position, has_gauss, cached],
            "torch": torch.get_rng_state(),
            # Each DDP rank uses one device. Reading all visible CUDA RNG states
            # would create unnecessary contexts on every other rank's GPU.
            "cuda": [torch.cuda.get_rng_state()] if torch.cuda.is_available()
            and (device is None or device.type == "cuda") else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != 1:
            raise ValueError("This checkpoint requires one local CUDA RNG state per rank")
        torch.cuda.set_rng_state(state["cuda"][0].cpu())


def rank_batches(batches, rank, world_size, accumulation=1, training=True):
    """Pad by repeating whole batches, never discard examples or split a batch."""
    if not batches:
        raise ValueError("The dataset produced no batches")
    if world_size < 1 or not 0 <= rank < world_size or accumulation < 1:
        raise ValueError("Invalid rank/world size/gradient accumulation")
    repeated = 0
    if training:
        multiple = world_size * accumulation
        repeated = (-len(batches)) % multiple
        batches = list(batches) + [batches[i % len(batches)] for i in range(repeated)]
    return batches[rank::world_size], repeated


def ctc_weight(update, target, start, end):
    if target <= 0 or update <= start:
        return 0.0
    if end <= start:
        return float(target)
    return float(target) * min(1.0, (update - start) / (end - start))


def make_accelerator(cfg):
    # The optional CTC branch and frozen video branch produce unused parameters.
    return Accelerator(
        gradient_accumulation_steps=int(cfg.optim.grad_accumulation_steps),
        mixed_precision=str(cfg.runtime.mixed_precision),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


class TextAudioTrainer:
    def __init__(self, model, cfg, training_contract, accelerator=None, initialization_report=None):
        self.cfg = cfg
        self.contract = training_contract
        self.contract_hash = canonical_sha256(training_contract)
        self.initialization_report = initialization_report or {}
        self.accelerator = accelerator or make_accelerator(cfg)
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.main = self.accelerator.is_main_process
        self.output = Path(str(cfg.ckpts.save_dir)).resolve()
        self.update = self.epoch = self.batch_in_epoch = 0
        self.stop_requested = False
        self.accumulation = int(cfg.optim.grad_accumulation_steps)
        self.max_updates = int(cfg.optim.max_updates)
        self.stop_after = int(cfg.optim.get("stop_after_updates") or self.max_updates)
        if self.max_updates < 1 or self.stop_after < 1 or self.accumulation < 1:
            raise ValueError("Update limits and gradient accumulation must be positive")
        if str(cfg.runtime.mixed_precision) not in {"bf16", "no"}:
            raise ValueError("This trainer supports bf16 or full precision; fp16 scaler state is not implemented")
        if int(cfg.ckpts.keep_last_n_checkpoints) < 1:
            raise ValueError("Retain at least one checkpoint")
        for section, field in (("logging", "every_updates"), ("validation", "every_updates"),
                               ("ckpts", "save_per_updates")):
            if int(cfg[section][field]) < 1:
                raise ValueError(f"{section}.{field} must be positive")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(cfg.optim.learning_rate),
                                      betas=tuple(cfg.optim.betas), weight_decay=float(cfg.optim.weight_decay))
        warmup = int(cfg.optim.num_warmup_updates)

        def lr_scale(step):
            if warmup and step < warmup:
                return max(1, step + 1) / warmup
            return max(0.0, (self.max_updates - step) / max(1, self.max_updates - warmup))

        # A plain scheduler advances exactly once per successful optimizer update.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)
        online = self.accelerator.unwrap_model(self.model)
        self.ema = create_text_audio_ema(online, cfg.ema.beta) if self.main else None
        self.writer = None
        self._original_handlers = {}

    def _collective_error(self, error):
        messages = [None] * self.world_size
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(messages, error)
        else:
            messages[0] = error
        errors = [item for item in messages if item]
        if errors:
            raise RuntimeError("; ".join(errors))

    def _setup_output(self):
        error = None
        if self.main:
            try:
                self.output.mkdir(parents=True, exist_ok=True)
                contract_path = self.output / "training_contract.json"
                resume = self.cfg.ckpts.get("resume")
                if not resume and list(self.output.glob("model_*.pt")):
                    raise ValueError("Existing checkpoints require explicit ckpts.resume or a new run directory")
                if contract_path.exists():
                    old = json.loads(contract_path.read_text())
                    if old != self.contract:
                        raise ValueError("Output directory has a different training contract")
                else:
                    contract_path.write_text(json.dumps(self.contract, ensure_ascii=False, indent=2) + "\n")
                from omegaconf import OmegaConf
                (self.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))
                (self.output / "initialization_report.json").write_text(
                    json.dumps(self.initialization_report, ensure_ascii=False, indent=2) + "\n")
            except Exception as exc:
                error = f"Output initialization failed: {exc}"
        self._collective_error(error)
        self.accelerator.wait_for_everyone()

    def _resume(self):
        path = self.cfg.ckpts.get("resume")
        if not path:
            set_seed(int(self.cfg.seed) + self.rank)
            return
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if payload.get("checkpoint_schema_version") != 1 or payload.get("stage") != "libritts_text_audio":
            raise ValueError("Resume requires a complete LibriTTS text/audio training checkpoint")
        contract = payload.get("training_contract", {})
        if canonical_sha256(contract) != payload.get("training_contract_sha256"):
            raise ValueError("Checkpoint training contract hash is invalid")
        if contract != self.contract:
            raise ValueError("Resume contract differs (data, architecture, sources or optimizer); use a new run")
        if payload.get("world_size") != self.world_size:
            raise ValueError("Exact resume requires the original world size; rank RNG and batch cursors differ")
        required = {"model_state_dict", "ema_model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                    "rng_states", "update", "epoch", "batch_in_epoch"}
        if not required.issubset(payload):
            raise ValueError(f"Incomplete resume checkpoint, missing {sorted(required - payload.keys())}")
        self.accelerator.unwrap_model(self.model).load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if self.main:
            self.ema.load_state_dict(payload["ema_model_state_dict"], strict=True)
        self.update, self.epoch, self.batch_in_epoch = (
            int(payload[key]) for key in ("update", "epoch", "batch_in_epoch"))
        if self.update < 1 or self.epoch < 0 or self.batch_in_epoch < 0:
            raise ValueError("Invalid checkpoint update/epoch/batch cursor")
        if int(payload["ema_model_state_dict"].get("step", -1)) != self.update:
            raise ValueError("Checkpoint EMA update counter differs from optimizer updates")
        self.initialization_report = payload.get("initialization_report", self.initialization_report)
        if len(payload["rng_states"]) != self.world_size or self.batch_in_epoch % self.accumulation:
            raise ValueError("Checkpoint has inconsistent rank RNG or an unfinished optimizer update")
        restore_rng_state(payload["rng_states"][self.rank])
        if self.main:
            (self.output / "initialization_report.json").write_text(
                json.dumps(self.initialization_report, ensure_ascii=False, indent=2) + "\n")
            print(f"Resumed update={self.update}, epoch={self.epoch}, next_batch={self.batch_in_epoch}", flush=True)

    def save_checkpoint(self, reason="periodic"):
        """Collect all rank RNG states; atomically publish a complete checkpoint."""
        self.accelerator.wait_for_everyone()
        local_rng = capture_rng_state(self.accelerator.device)
        rng_states = [None] * self.world_size
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(rng_states, local_rng)
        else:
            rng_states[0] = local_rng
        error = None
        if self.main:
            try:
                payload = {
                    "checkpoint_schema_version": 1, "stage": "libritts_text_audio",
                    "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                    "ema_model_state_dict": self.ema.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "update": self.update, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                    "world_size": self.world_size, "training_contract": self.contract,
                    "training_contract_sha256": self.contract_hash, "rng_states": rng_states,
                    "initialization_report": self.initialization_report, "save_reason": reason,
                }
                path = self.output / f"model_{self.update:08d}.pt"
                temporary = self.output / f".{path.name}.tmp"
                with temporary.open("wb") as handle:
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                latest_tmp = self.output / ".latest.json.tmp"
                latest_tmp.write_text(json.dumps({"checkpoint": path.name, "update": self.update,
                                                 "reason": reason, "contract_sha256": self.contract_hash}) + "\n")
                os.replace(latest_tmp, self.output / "latest.json")
                checkpoints = sorted(self.output.glob("model_[0-9]*.pt"))
                keep = int(self.cfg.ckpts.keep_last_n_checkpoints)
                for old in checkpoints[:-keep]:
                    old.unlink()
                if self.writer:
                    self.writer.flush()
                print(f"Saved {path} ({reason})", flush=True)
            except Exception as exc:
                error = f"Checkpoint save failed: {exc}"
        self._collective_error(error)
        self.accelerator.wait_for_everyone()

    def _loader(self, dataset, batches, epoch):
        # A dedicated generator prevents worker startup from consuming the model RNG.
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + epoch * 100003 + self.rank)
        workers = int(self.cfg.datasets.num_workers)
        return DataLoader(dataset, batch_sampler=batches, collate_fn=dataset.collate_fn,
                          num_workers=workers, pin_memory=self.accelerator.device.type == "cuda",
                          persistent_workers=False, generator=generator)

    def _forward(self, model, batch):
        device = self.accelerator.device
        return model(batch["audio"].to(device, non_blocking=True), text=batch["text"],
                     lens=batch["lens"].to(device, non_blocking=True),
                     speaker_embedding=batch["speaker_embedding"].to(device, non_blocking=True))

    @torch.no_grad()
    def validate(self, dataset):
        saved_rng = capture_rng_state(self.accelerator.device)
        model = self.accelerator.unwrap_model(self.model)
        training = model.training
        totals = torch.zeros(4, dtype=torch.float64, device=self.accelerator.device)
        try:
            set_seed(int(self.cfg.validation.seed) + self.rank)
            model.eval()
            sampler = LengthBudgetBatchSampler(dataset, max_frames=int(self.cfg.datasets.batch_size_per_gpu),
                                               max_samples=int(self.cfg.datasets.max_samples),
                                               shuffle=False, seed=int(self.cfg.validation.seed))
            batches = sampler.batches_for_epoch(0)
            limit = int(self.cfg.validation.get("max_batches") or 0)
            if limit:
                batches = batches[:limit]
            local, _ = rank_batches(batches, self.rank, self.world_size, training=False)
            for batch in self._loader(dataset, local, 0):
                with self.accelerator.autocast():
                    loss, components, _, _ = self._forward(model, batch)
                n = len(batch["text"])
                totals += torch.tensor([float(loss), float(components["diff_loss"]),
                                        float(components.get("ctc_loss", 0)), 1.0],
                                       device=totals.device, dtype=totals.dtype) * n
            totals = self.accelerator.reduce(totals, reduction="sum")
            if totals[3] <= 0 or not torch.isfinite(totals).all():
                raise RuntimeError("Validation produced no samples or a nonfinite loss")
            means = (totals[:3] / totals[3]).tolist()
            if self.main:
                for name, value in zip(("total", "flow", "ctc"), means):
                    self.writer.add_scalar(f"validation/{name}", value, self.update)
                self.writer.add_scalar("validation/samples", float(totals[3]), self.update)
                self.writer.flush()
                print(f"Validation update={self.update}: total={means[0]:.6f} flow={means[1]:.6f} ctc={means[2]:.6f}", flush=True)
        finally:
            model.train(training)
            restore_rng_state(saved_rng)

    def _signal(self, signum, frame):
        self.stop_requested = True

    def train(self, train_dataset, validation_dataset):
        self._setup_output()
        self._resume()
        if self.main:
            logdir = self.output / "tensorboard"
            self.writer = SummaryWriter(str(logdir), purge_step=self.update + 1 if self.update else None)
            print(f"TensorBoard logdir: {logdir}", flush=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal)
        start = time.monotonic()
        statistics = torch.zeros(6, dtype=torch.float64, device=self.accelerator.device)
        online = self.accelerator.unwrap_model(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        reason = "complete"
        try:
            while self.update < min(self.max_updates, self.stop_after):
                sampler = LengthBudgetBatchSampler(train_dataset, max_frames=int(self.cfg.datasets.batch_size_per_gpu),
                                                   max_samples=int(self.cfg.datasets.max_samples),
                                                   seed=int(self.cfg.seed), shuffle=True)
                all_batches = sampler.batches_for_epoch(self.epoch)
                batches, repeated = rank_batches(all_batches, self.rank, self.world_size, self.accumulation)
                if self.batch_in_epoch > len(batches):
                    raise ValueError("Resume batch cursor exceeds the reconstructed epoch")
                if self.main:
                    self.writer.add_scalar("data/repeated_tail_batches", repeated, self.update)
                    print(f"Epoch {self.epoch}: {len(all_batches)} original batches, {repeated} tail repeats, "
                          f"{len(batches)} batches/rank", flush=True)
                loader = self._loader(train_dataset, batches[self.batch_in_epoch:], self.epoch)
                self.model.train()
                for batch in loader:
                    online.ctc_lambda = ctc_weight(self.update, float(self.cfg.model.ctc_lambda),
                                                  int(self.cfg.model.ctc_warmup_start), int(self.cfg.model.ctc_warmup_end))
                    with self.accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            loss, components, _, _ = self._forward(self.model, batch)
                        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                        if self.accelerator.reduce(finite, reduction="sum").item() != self.world_size:
                            raise FloatingPointError("Nonfinite training loss; last complete checkpoint remains valid")
                        self.accelerator.backward(loss)
                        grad_norm = 0.0
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), float(self.cfg.optim.max_grad_norm))
                            if not torch.isfinite(torch.as_tensor(grad_norm)).all():
                                raise FloatingPointError("Nonfinite gradient; refusing to publish a damaged checkpoint")
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.batch_in_epoch += 1
                    statistics += torch.tensor([float(loss.detach()), float(components["diff_loss"]),
                                                float(components.get("ctc_loss", 0)), float(batch["lens"].sum()),
                                                sum(len(text) for text in batch["text"]), 1],
                                               device=statistics.device, dtype=statistics.dtype)
                    if not self.accelerator.sync_gradients:
                        continue
                    if self.accelerator.optimizer_step_was_skipped:
                        raise FloatingPointError("Optimizer update was skipped; resume from the last complete checkpoint")
                    self.scheduler.step()
                    self.update += 1
                    if self.main:
                        self.ema.update()
                    if self.update % int(self.cfg.logging.every_updates) == 0 or self.update == 1:
                        reduced = self.accelerator.reduce(statistics, reduction="sum")
                        memory = None
                        if self.accelerator.device.type == "cuda":
                            local_memory = torch.tensor([
                                torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                            ], dtype=torch.float64, device=self.accelerator.device) / 2**30
                            memory = self.accelerator.gather(local_memory).reshape(-1, 2).max(dim=0).values
                        seconds = max(1e-9, time.monotonic() - start)
                        if self.main:
                            if memory is not None:
                                self.writer.add_scalar("train/peak_allocated_gib", float(memory[0]), self.update)
                                self.writer.add_scalar("train/peak_reserved_gib", float(memory[1]), self.update)
                            for name, value in zip(("total", "flow", "ctc"), (reduced[:3] / reduced[5]).tolist()):
                                self.writer.add_scalar(f"train/{name}", value, self.update)
                            for name, value in {"grad_norm": float(grad_norm), "lr": self.scheduler.get_last_lr()[0],
                                                "ctc_weight": online.ctc_lambda, "frames_per_second": float(reduced[3]) / seconds,
                                                "text_tokens_per_second": float(reduced[4]) / seconds}.items():
                                self.writer.add_scalar(f"train/{name}", value, self.update)
                            self.writer.flush()
                            print(f"update={self.update} loss={float(reduced[0] / reduced[5]):.6f} "
                                  f"grad={float(grad_norm):.4f} lr={self.scheduler.get_last_lr()[0]:.3g} "
                                  f"frames/s={float(reduced[3]) / seconds:.1f}", flush=True)
                        statistics.zero_()
                        start = time.monotonic()
                    stop = torch.tensor(int(self.stop_requested), device=self.accelerator.device)
                    stop = self.accelerator.reduce(stop, reduction="sum").item() > 0
                    terminal = stop or self.update >= min(self.max_updates, self.stop_after)
                    if not terminal and self.update % int(self.cfg.validation.every_updates) == 0:
                        self.validate(validation_dataset)
                    if terminal:
                        reason = "signal" if stop else "complete"
                        self.save_checkpoint(reason)
                        return
                    if self.update % int(self.cfg.ckpts.save_per_updates) == 0:
                        self.save_checkpoint()
                self.epoch += 1
                self.batch_in_epoch = 0
            # Resuming an already complete run does not overwrite its checkpoint.
        finally:
            for signum, handler in self._original_handlers.items():
                signal.signal(signum, handler)
            if self.writer:
                self.writer.close()
            self.accelerator.end_training()
