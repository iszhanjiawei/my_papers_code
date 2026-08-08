"""Exact-update trainer for staged Semantic-VAE C2 multimodal adaptation."""

from __future__ import annotations

import gc
import math
import re
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from aligndit.model.semantic_vae_c2_stage import S3_STAGES
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.trainer import _atomic_accelerator_save


S3_STAGE_START_UPDATE = {"s3a": 0, "s3b": 5_000}
S3_STAGE_MAX_UPDATES = {"s3a": 5_000, "s3b": 195_000}
S3_FINAL_CUMULATIVE_UPDATE = {"s3a": 5_000, "s3b": 200_000}


def _scalar(value: Any, *, name: str) -> int | bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be scalar, got shape={tuple(value.shape)}")
        value = value.item()
    if type(value) not in {int, bool}:
        raise RuntimeError(f"{name} must be an int/bool scalar, got {value!r}")
    return value


class SemanticVaeC2Trainer:
    """Trainer whose resume boundary is local to S3a or S3b.

    Checkpoint names and the compatibility ``update`` field are cumulative
    across S3a+S3b, while scheduler, optimizer and EMA counters are stage
    local.  This keeps paper-facing 5k/50k/.../200k checkpoints unambiguous
    without leaking S3a optimizer state into S3b.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        accelerator: Accelerator,
        optimizer_groups: list[dict[str, Any]],
        stage: str,
        num_warmup_updates: int,
        save_per_updates: int,
        keep_last_n_checkpoints: int,
        last_per_updates: int,
        checkpoint_path: str,
        batch_size_per_gpu: int,
        batch_size_type: str,
        max_samples: int,
        grad_accumulation_steps: int,
        max_grad_norm: float,
        logger: str | None,
        run_name: str,
        ema_kwargs: dict[str, Any],
    ) -> None:
        if stage not in S3_STAGES:
            raise ValueError(f"Unknown Semantic-VAE C2 stage: {stage!r}")
        if not optimizer_groups:
            raise ValueError("S3 optimizer requires at least one parameter group")
        if grad_accumulation_steps != accelerator.gradient_accumulation_steps:
            raise ValueError("Accelerator and trainer gradient accumulation settings differ")
        if batch_size_type not in {"frame", "sample"}:
            raise ValueError(f"Unsupported batch_size_type: {batch_size_type!r}")
        if logger not in {None, "tensorboard"}:
            raise ValueError("The reproducible S3 path supports only tensorboard or null logging")

        self.accelerator = accelerator
        self.model = model
        self.stage = stage
        self.stage_start_update = S3_STAGE_START_UPDATE[stage]
        self.max_stage_updates = S3_STAGE_MAX_UPDATES[stage]
        self.final_cumulative_update = S3_FINAL_CUMULATIVE_UPDATE[stage]
        self.num_warmup_updates = int(num_warmup_updates)
        self.save_per_updates = int(save_per_updates)
        self.keep_last_n_checkpoints = int(keep_last_n_checkpoints)
        self.last_per_updates = int(last_per_updates)
        self.checkpoint_path = str(checkpoint_path)
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.batch_size_type = batch_size_type
        self.max_samples = int(max_samples)
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.logger = logger
        self.training_contract_sha256: str | None = None
        self.noise_scheduler = None

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
            print(f"Using logger={logger}; Semantic-VAE C2 stage={stage}", flush=True)
            if logger == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=f"runs/{run_name}")

        clean_groups: list[dict[str, Any]] = []
        for group in optimizer_groups:
            clean_group = dict(group)
            clean_group.pop("parameter_names", None)
            clean_groups.append(clean_group)
        self.optimizer = AdamW(clean_groups)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.scheduler = None

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    def bind_training_contract(self, contract_sha256: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", contract_sha256) is None:
            raise ValueError(f"Invalid S3 training contract SHA256: {contract_sha256!r}")
        self.training_contract_sha256 = contract_sha256

    def cumulative_update(self, stage_update: int) -> int:
        if not 0 <= stage_update <= self.max_stage_updates:
            raise ValueError(f"Invalid {self.stage} stage update: {stage_update}")
        return self.stage_start_update + stage_update

    def reset_ema_from_online(self) -> None:
        """Initialize a fresh stage-local EMA at step zero from online weights."""

        self.accelerator.wait_for_everyone()
        if self.is_main:
            self.ema_model.copy_params_from_model_to_ema()
            self.ema_model.step.zero_()
            self.ema_model.initted.fill_(False)
            if int(self.ema_model.step.item()) != 0 or bool(self.ema_model.initted.item()):
                raise RuntimeError("Failed to reset the S3 EMA to a fresh step-zero state")
        self.accelerator.wait_for_everyone()

    def _checkpoint_files(self) -> list[Path]:
        directory = Path(self.checkpoint_path)
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise NotADirectoryError(f"S3 checkpoint path is not a directory: {directory}")
        serialized = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix in {".pt", ".safetensors"}
        )
        recognized = [
            path for path in serialized if path.name == "model_last.pt" or re.fullmatch(r"model_[0-9]+\.pt", path.name)
        ]
        unexpected = sorted(set(serialized) - set(recognized))
        if unexpected:
            raise RuntimeError(f"Unexpected serialized files in S3 checkpoint directory: {unexpected}")
        return recognized

    def has_local_checkpoint(self) -> bool:
        return bool(self._checkpoint_files())

    def _validate_checkpoint(self, checkpoint: dict[str, Any], checkpoint_name: str) -> int:
        if checkpoint.get("checkpoint_schema_version") != 1:
            raise RuntimeError(f"Checkpoint {checkpoint_name} has the wrong schema version")
        if checkpoint.get("semantic_vae_c2_stage") != self.stage:
            raise RuntimeError(
                f"Checkpoint {checkpoint_name} stage mismatch: expected {self.stage!r}, "
                f"got {checkpoint.get('semantic_vae_c2_stage')!r}"
            )
        if checkpoint.get("training_contract_sha256") != self.training_contract_sha256:
            raise RuntimeError(
                f"Checkpoint {checkpoint_name} contract mismatch: expected {self.training_contract_sha256}, "
                f"got {checkpoint.get('training_contract_sha256')!r}"
            )
        stage_update = _scalar(checkpoint.get("stage_update"), name=f"{checkpoint_name} stage_update")
        if isinstance(stage_update, bool) or not 0 <= stage_update <= self.max_stage_updates:
            raise RuntimeError(f"Checkpoint {checkpoint_name} has invalid stage_update={stage_update!r}")
        cumulative_update = self.cumulative_update(stage_update)
        for field in ("cumulative_update", "update"):
            value = _scalar(checkpoint.get(field), name=f"{checkpoint_name} {field}")
            if value != cumulative_update:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_name} {field} mismatch: expected {cumulative_update}, got {value!r}"
                )
        if checkpoint.get("stage_start_update") != self.stage_start_update:
            raise RuntimeError(f"Checkpoint {checkpoint_name} has the wrong stage_start_update")
        if checkpoint.get("stage_max_updates") != self.max_stage_updates:
            raise RuntimeError(f"Checkpoint {checkpoint_name} has the wrong stage_max_updates")
        match = re.fullmatch(r"model_([0-9]+)\.pt", checkpoint_name)
        if match is not None and int(match.group(1)) != cumulative_update:
            raise RuntimeError(
                f"Checkpoint filename {checkpoint_name} does not match cumulative update {cumulative_update}"
            )

        ema_state = checkpoint.get("ema_model_state_dict")
        if not isinstance(ema_state, dict):
            raise TypeError(f"Checkpoint {checkpoint_name} has no EMA state mapping")
        ema_step = _scalar(ema_state.get("step"), name=f"{checkpoint_name} EMA step")
        if ema_step != stage_update:
            raise RuntimeError(
                f"Checkpoint {checkpoint_name} EMA step mismatch: expected {stage_update}, got {ema_step!r}"
            )
        ema_initted = _scalar(ema_state.get("initted"), name=f"{checkpoint_name} EMA initted")
        if stage_update > 0 and ema_initted is not True:
            raise RuntimeError(f"Checkpoint {checkpoint_name} EMA must be initialized after training starts")
        return stage_update

    def save_checkpoint(self, stage_update: int, *, last: bool = False) -> None:
        cumulative_update = self.cumulative_update(stage_update)
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = {
                "checkpoint_schema_version": 1,
                "semantic_vae_c2_stage": self.stage,
                "stage_start_update": self.stage_start_update,
                "stage_max_updates": self.max_stage_updates,
                "stage_update": stage_update,
                "cumulative_update": cumulative_update,
                "update": cumulative_update,
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_model_state_dict": self.ema_model.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_contract_sha256": self.training_contract_sha256,
            }
            destination = Path(self.checkpoint_path) / ("model_last.pt" if last else f"model_{cumulative_update}.pt")
            _atomic_accelerator_save(self.accelerator, checkpoint, destination)
            print(
                f"Saved {self.stage} {'last' if last else 'numbered'} checkpoint: "
                f"stage_update={stage_update}, cumulative_update={cumulative_update}",
                flush=True,
            )
            if not last and self.keep_last_n_checkpoints > 0:
                numbered = [
                    path
                    for path in Path(self.checkpoint_path).glob("model_[0-9]*.pt")
                    if re.fullmatch(r"model_[0-9]+\.pt", path.name)
                ]
                numbered.sort(key=lambda path: int(path.stem.split("_")[1]))
                while len(numbered) > self.keep_last_n_checkpoints:
                    numbered.pop(0).unlink()
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self) -> int:
        files = self._checkpoint_files()
        if not files:
            return 0
        if self.training_contract_sha256 is None:
            raise RuntimeError("Bind the immutable S3 contract before loading a local checkpoint")

        names = {path.name for path in files}
        if "model_last.pt" in names:
            checkpoint_path = Path(self.checkpoint_path) / "model_last.pt"
        else:
            checkpoint_path = max(files, key=lambda path: int(path.stem.split("_")[1]))
        self.accelerator.wait_for_everyone()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"S3 checkpoint must contain a mapping: {checkpoint_path}")
        stage_update = self._validate_checkpoint(checkpoint, checkpoint_path.name)

        model_state = checkpoint.get("model_state_dict")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if (
            not isinstance(model_state, dict)
            or not isinstance(optimizer_state, dict)
            or not isinstance(scheduler_state, dict)
        ):
            raise TypeError(f"Checkpoint {checkpoint_path} is missing exact resume-state mappings")
        self.accelerator.unwrap_model(self.model).load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        if self.scheduler is None:
            raise RuntimeError("S3 scheduler must be constructed before loading a checkpoint")
        self.scheduler.load_state_dict(scheduler_state)
        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
        del checkpoint
        gc.collect()
        self.accelerator.wait_for_everyone()
        return stage_update

    def reconcile_checkpoint_files(self, stage_update: int) -> None:
        """Repair an interrupted two-file save before training can continue.

        Every durable boundary publishes ``model_last.pt`` first and the
        paper-facing numbered checkpoint second.  If the process dies between
        those atomic writes, the loaded state is complete and can safely
        recreate the missing sibling without performing another update.
        """

        if stage_update <= 0:
            return
        cumulative_update = self.cumulative_update(stage_update)
        checkpoint_dir = Path(self.checkpoint_path)
        if not (checkpoint_dir / "model_last.pt").is_file():
            self.save_checkpoint(stage_update, last=True)
        numbered_required = cumulative_update % self.save_per_updates == 0 or stage_update == self.max_stage_updates
        if numbered_required and not (checkpoint_dir / f"model_{cumulative_update}.pt").is_file():
            self.save_checkpoint(stage_update)

    def _build_dataloader(self, dataset: Dataset, *, num_workers: int, seed: int) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        common = {
            "dataset": dataset,
            "collate_fn": dataset.collate_fn,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if self.batch_size_type == "sample":
            return DataLoader(
                **common,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        self.accelerator.even_batches = False
        batch_sampler = DynamicBatchSampler(
            SequentialSampler(dataset),
            self.batch_size_per_gpu,
            max_samples=self.max_samples,
            random_seed=seed,
            drop_residual=False,
        )
        return DataLoader(**common, batch_sampler=batch_sampler)

    def train(
        self,
        train_dataset: Dataset,
        *,
        num_workers: int,
        seed: int,
        run_until_stage_update: int,
        deterministic_update_seed: bool,
    ) -> None:
        if not 0 < run_until_stage_update <= self.max_stage_updates:
            raise ValueError(
                f"run_until_stage_update must be in [1, {self.max_stage_updates}], got {run_until_stage_update}"
            )
        if deterministic_update_seed and self.grad_accumulation_steps != 1:
            raise ValueError("deterministic_update_seed requires grad_accumulation_steps=1")

        train_dataloader = self._build_dataloader(train_dataset, num_workers=num_workers, seed=seed)
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
        total_steps = self.max_stage_updates * self.accelerator.num_processes
        if not 0 < warmup_steps < total_steps:
            raise RuntimeError(f"Invalid scheduler horizon: warmup={warmup_steps}, total={total_steps}")
        warmup = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=1e-8,
            total_iters=total_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(self.optimizer, [warmup, decay], milestones=[warmup_steps])
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)

        stage_update = self.load_checkpoint()
        self.reconcile_checkpoint_files(stage_update)
        if stage_update > run_until_stage_update:
            raise RuntimeError(
                f"Local checkpoint update {stage_update} exceeds requested stop {run_until_stage_update}"
            )
        if stage_update == run_until_stage_update:
            if (Path(self.checkpoint_path) / "model_last.pt").is_file():
                if self.is_main:
                    print(f"{self.stage} already reached stage update {stage_update}; nothing to do", flush=True)
            else:
                self.save_checkpoint(stage_update, last=True)
            self.accelerator.end_training()
            return

        steps_per_epoch = math.ceil(len(train_dataloader) / self.grad_accumulation_steps)
        start_step = stage_update * self.grad_accumulation_steps
        skipped_epoch = start_step // len(train_dataloader)
        skipped_batch = start_step % len(train_dataloader)
        skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        training_epochs = math.ceil(self.max_stage_updates / steps_per_epoch)
        last_checkpoint_update = stage_update if (Path(self.checkpoint_path) / "model_last.pt").is_file() else -1

        for epoch in range(skipped_epoch, training_epochs):
            if stage_update >= run_until_stage_update:
                break
            self.model.train()
            if epoch == skipped_epoch:
                current_dataloader = skipped_dataloader
                initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
            else:
                current_dataloader = train_dataloader
                initial = 0
            batch_sampler = getattr(train_dataloader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            elif hasattr(getattr(batch_sampler, "batch_sampler", None), "set_epoch"):
                batch_sampler.batch_sampler.set_epoch(epoch)

            progress = tqdm(
                range(steps_per_epoch),
                desc=f"{self.stage.upper()} epoch {epoch + 1}/{training_epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=initial,
            )
            for batch in current_dataloader:
                if not batch:
                    raise RuntimeError("Semantic-VAE C2 dataset produced an empty batch")
                if deterministic_update_seed:
                    update_seed = seed * 1_000_003 + stage_update * self.accelerator.num_processes
                    update_seed += self.accelerator.process_index
                    set_seed(update_seed)
                with self.accelerator.accumulate(self.model):
                    latent = batch["mel"].permute(0, 2, 1)
                    loss, components, _, _ = self.model(
                        latent,
                        text=batch["text"],
                        lens=batch["mel_lengths"],
                        text_lens=batch["text_lengths"],
                        video=batch["video"],
                        video_lens=batch["video_lengths"],
                        noise_scheduler=self.noise_scheduler,
                    )
                    if not torch.isfinite(loss):
                        keys = batch.get("utterance_keys", ("<unknown>",))
                        raise FloatingPointError(f"Non-finite {self.stage} loss for batch keys={list(keys)[:4]}")
                    self.accelerator.backward(loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()
                    stage_update += 1
                    cumulative_update = self.cumulative_update(stage_update)
                    progress.update(1)
                    progress.set_postfix(
                        stage_update=str(stage_update),
                        cumulative_update=str(cumulative_update),
                        loss=loss.item(),
                        **components,
                    )
                else:
                    cumulative_update = self.cumulative_update(stage_update)

                if self.accelerator.is_local_main_process and self.logger == "tensorboard":
                    self.writer.add_scalar("loss", loss.item(), cumulative_update)
                    for key, value in components.items():
                        self.writer.add_scalar(key, value, cumulative_update)
                    for group in self.optimizer.param_groups:
                        self.writer.add_scalar(f"lr/{group['group_name']}", group["lr"], cumulative_update)

                if self.accelerator.sync_gradients and cumulative_update % self.last_per_updates == 0:
                    self.save_checkpoint(stage_update, last=True)
                    last_checkpoint_update = stage_update
                if (
                    self.accelerator.sync_gradients
                    and self.keep_last_n_checkpoints != 0
                    and cumulative_update % self.save_per_updates == 0
                ):
                    self.save_checkpoint(stage_update)
                if stage_update >= run_until_stage_update:
                    break
            progress.close()

        if stage_update != run_until_stage_update:
            raise RuntimeError(
                f"{self.stage} stopped at stage update {stage_update}, expected {run_until_stage_update}"
            )
        if last_checkpoint_update != stage_update:
            self.save_checkpoint(stage_update, last=True)
        cumulative_update = self.cumulative_update(stage_update)
        numbered_path = Path(self.checkpoint_path) / f"model_{cumulative_update}.pt"
        if stage_update == self.max_stage_updates and not numbered_path.is_file():
            self.save_checkpoint(stage_update)
        self.accelerator.end_training()
