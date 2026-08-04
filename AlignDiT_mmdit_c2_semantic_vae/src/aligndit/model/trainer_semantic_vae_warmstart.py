"""Stage-local trainer for mel-to-Semantic-VAE representation adaptation."""

from __future__ import annotations

import math
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from aligndit.model.semantic_vae_warmstart import STAGE_NAMES, WarmStartLoadReport, load_parent_ema_weights
from aligndit.model.trainer_notext import Trainer_notext
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.trainer import _atomic_accelerator_save
from f5_tts.model.utils import default


def create_warmstart_accelerator(*, grad_accumulation_steps: int, logger: str | None) -> Accelerator:
    """Create the process group before strict per-rank parent initialization."""

    if logger not in {None, "tensorboard"}:
        raise ValueError("The reproducible warm-start path supports only tensorboard or null logging")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", "600"))
    if timeout_seconds <= 0:
        raise ValueError("NCCL_TIMEOUT must be a positive integer number of seconds")
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))
    return Accelerator(
        kwargs_handlers=[ddp_kwargs, process_group_kwargs],
        gradient_accumulation_steps=grad_accumulation_steps,
    )


def has_local_training_checkpoint(checkpoint_path: str | Path) -> bool:
    directory = Path(checkpoint_path)
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise NotADirectoryError(f"Warm-start checkpoint path is not a directory: {directory}")

    serialized_files = sorted(
        path.name for path in directory.iterdir() if path.is_file() and path.suffix in {".pt", ".safetensors"}
    )
    recognized_files = [
        name for name in serialized_files if name == "model_last.pt" or re.fullmatch(r"model_[0-9]+\.pt", name)
    ]
    unexpected_files = sorted(set(serialized_files) - set(recognized_files))
    if unexpected_files:
        # Trainer.load_checkpoint historically accepts pretrained_*.pt and
        # safetensors. They are forbidden here because a stage directory may
        # only resume its own optimizer/scheduler/EMA state.
        raise RuntimeError(
            f"Unexpected serialized files in warm-start checkpoint directory {directory}: {unexpected_files}"
        )
    return bool(recognized_files)


def initialize_parent_on_all_ranks(
    model: torch.nn.Module,
    accelerator: Accelerator,
    *,
    parent_path: str,
    stage: str,
    expected_parent_update: int,
    expected_parent_stage: str | None,
    expected_parent_contract_sha256: str | None,
    expected_parent_sha256: str,
    expected_parent_size: int,
) -> WarmStartLoadReport:
    """Load the same immutable parent independently on every rank.

    The checkpoint is memory-mapped and backed by a shared host page cache.
    Independent strict loads avoid ordering hazards from ad-hoc NCCL broadcasts
    before DDP construction and guarantee frozen parameters match on all ranks.
    """

    report = load_parent_ema_weights(
        model,
        parent_path,
        stage=stage,
        expected_parent_update=expected_parent_update,
        expected_parent_stage=expected_parent_stage,
        expected_parent_contract_sha256=expected_parent_contract_sha256,
        expected_parent_sha256=expected_parent_sha256,
        expected_parent_size=expected_parent_size,
    )
    model.to(accelerator.device)
    accelerator.wait_for_everyone()
    return report


class SemanticVaeWarmStartTrainer(Trainer_notext):
    """A Trainer_notext variant with fixed stage-local optimizer groups."""

    def __init__(
        self,
        model,
        *,
        accelerator: Accelerator,
        optimizer_groups: list[dict[str, Any]],
        stage: str,
        epochs: int,
        num_warmup_updates: int,
        save_per_updates: int,
        keep_last_n_checkpoints: int,
        checkpoint_path: str,
        batch_size_per_gpu: int,
        batch_size_type: str,
        max_samples: int,
        grad_accumulation_steps: int,
        max_grad_norm: float,
        logger: str | None,
        wandb_run_name: str,
        last_per_updates: int,
        ema_kwargs: dict[str, Any],
        projection_target_lambda: float,
        projection_ramp_updates: int,
    ) -> None:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown warm-start stage: {stage!r}")
        if not optimizer_groups:
            raise ValueError("Warm-start optimizer requires at least one parameter group")
        if grad_accumulation_steps != accelerator.gradient_accumulation_steps:
            raise ValueError("Accelerator and trainer gradient accumulation settings differ")
        if projection_target_lambda < 0 or projection_ramp_updates < 0:
            raise ValueError("Projection target and ramp updates must be non-negative")
        if stage != "s2c" and (projection_target_lambda != 0 or projection_ramp_updates != 0):
            raise ValueError("Only S2c may enable the 40-Hz HuBERT projection loss")
        if stage == "s2c" and projection_target_lambda > 0 and projection_ramp_updates <= 0:
            raise ValueError("A positive S2c projection target requires a positive ramp")

        self.accelerator = accelerator
        self.logger = logger
        self.log_samples = False
        self.model = model
        self.stage = stage
        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = checkpoint_path
        self.training_contract_sha256 = None
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.noise_scheduler = None
        self.projection_target_lambda = float(projection_target_lambda)
        self.projection_ramp_updates = int(projection_ramp_updates)

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
            print(f"Using logger: {logger}; warm-start stage={stage}", flush=True)
            if self.logger == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        clean_groups = []
        for group in optimizer_groups:
            clean_group = dict(group)
            # Parameter names are committed to the contract separately.  They
            # are not optimizer hyperparameters and would bloat every save.
            clean_group.pop("parameter_names", None)
            clean_groups.append(clean_group)
        self.optimizer = AdamW(clean_groups)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    def reset_ema_from_online(self) -> None:
        """Start a fresh stage EMA from the transferred online parameters."""

        self.accelerator.wait_for_everyone()
        if self.is_main:
            self.ema_model.copy_params_from_model_to_ema()
            self.ema_model.step.zero_()
            self.ema_model.initted.fill_(False)
        self.accelerator.wait_for_everyone()

    def projection_lambda_at(self, update: int) -> float:
        if self.stage != "s2c" or self.projection_target_lambda == 0:
            return 0.0
        fraction = min(max(update, 0) / self.projection_ramp_updates, 1.0)
        return self.projection_target_lambda * fraction

    def _set_projection_lambda(self, update: int) -> float:
        value = self.projection_lambda_at(update)
        self.accelerator.unwrap_model(self.model).proj_lambda = value
        return value

    def _validate_checkpoint_contract(self, checkpoint: dict, checkpoint_name: str) -> None:
        super()._validate_checkpoint_contract(checkpoint, checkpoint_name)
        actual_stage = checkpoint.get("warmstart_stage")
        if actual_stage != self.stage:
            raise RuntimeError(
                f"Checkpoint {checkpoint_name} warm-start stage mismatch: expected {self.stage!r}, got {actual_stage!r}"
            )

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = {
                "checkpoint_schema_version": 1,
                "warmstart_stage": self.stage,
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_model_state_dict": self.ema_model.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_contract_sha256": self.training_contract_sha256,
                "update": update,
            }
            if last:
                _atomic_accelerator_save(
                    self.accelerator,
                    checkpoint,
                    os.path.join(self.checkpoint_path, "model_last.pt"),
                )
                print(f"Saved {self.stage} last checkpoint at update {update}", flush=True)
            elif self.keep_last_n_checkpoints != 0:
                destination = os.path.join(self.checkpoint_path, f"model_{update}.pt")
                _atomic_accelerator_save(self.accelerator, checkpoint, destination)
                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        path
                        for path in Path(self.checkpoint_path).glob("model_[0-9]*.pt")
                        if re.fullmatch(r"model_[0-9]+\.pt", path.name)
                    ]
                    checkpoints.sort(key=lambda path: int(path.stem.split("_")[1]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        checkpoints.pop(0).unlink()
        self.accelerator.wait_for_everyone()

    def train(
        self,
        train_dataset: Dataset,
        *,
        num_workers: int,
        resumable_with_seed: int,
        max_updates: int,
        run_until_update: int,
        deterministic_update_seed: bool,
    ) -> None:
        if not isinstance(max_updates, int) or max_updates <= 0:
            raise ValueError(f"max_updates must be a positive integer, got {max_updates!r}")
        if not isinstance(run_until_update, int) or not 0 < run_until_update <= max_updates:
            raise ValueError(f"run_until_update must be within [1, {max_updates}], got {run_until_update!r}")
        if deterministic_update_seed and self.grad_accumulation_steps != 1:
            raise ValueError("deterministic_update_seed requires grad_accumulation_steps=1")

        generator = torch.Generator().manual_seed(resumable_with_seed)
        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            batch_sampler = DynamicBatchSampler(
                SequentialSampler(train_dataset),
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"Unknown batch_size_type: {self.batch_size_type!r}")

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = max_updates * self.accelerator.num_processes
        if total_updates <= warmup_updates:
            raise ValueError(f"Training horizon ({total_updates}) must exceed warmup ({warmup_updates})")
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        decay_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=1e-8,
            total_iters=total_updates - warmup_updates,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_updates],
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)

        start_update = self.load_checkpoint()
        if start_update > run_until_update:
            raise RuntimeError(
                f"Checkpoint update {start_update} exceeds this invocation's run_until_update={run_until_update}"
            )
        has_current_last = Path(self.checkpoint_path, "model_last.pt").is_file()
        if start_update == run_until_update:
            if has_current_last:
                if self.is_main:
                    print(f"Stage {self.stage} is already complete at update {start_update}; nothing to do", flush=True)
            else:
                self.save_checkpoint(start_update, last=True)
            self.accelerator.end_training()
            return
        global_update = start_update
        last_checkpoint_update = start_update if has_current_last else -1
        steps_per_epoch = math.ceil(len(train_dataloader) / self.grad_accumulation_steps)
        start_step = start_update * self.grad_accumulation_steps
        skipped_epoch = start_step // len(train_dataloader)
        skipped_batch = start_step % len(train_dataloader)
        skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        training_epochs = math.ceil(max_updates / steps_per_epoch)

        for epoch in range(skipped_epoch, training_epochs):
            if global_update >= run_until_update:
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

            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"{self.stage.upper()} epoch {epoch + 1}/{training_epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=initial,
            )
            for batch in current_dataloader:
                if deterministic_update_seed:
                    update_seed = (
                        resumable_with_seed * 1_000_003
                        + global_update * self.accelerator.num_processes
                        + self.accelerator.process_index
                    )
                    set_seed(update_seed)
                projection_lambda = self._set_projection_lambda(global_update)
                with self.accelerator.accumulate(self.model):
                    latent = batch["mel"].permute(0, 2, 1)
                    loss, components, _, _ = self.model(
                        latent,
                        lens=batch["mel_lengths"],
                        feature=batch["rep"],
                        feature_lens=batch["rep_lengths"],
                        noise_scheduler=self.noise_scheduler,
                    )
                    self.accelerator.backward(loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()
                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update),
                        loss=loss.item(),
                        proj_lambda=projection_lambda,
                        **components,
                    )

                if self.accelerator.is_local_main_process and self.logger == "tensorboard":
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("projection_lambda", projection_lambda, global_update)
                    for key, value in components.items():
                        self.writer.add_scalar(key, value, global_update)
                    for group in self.optimizer.param_groups:
                        self.writer.add_scalar(f"lr/{group['group_name']}", group["lr"], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)
                    last_checkpoint_update = global_update
                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)
                if global_update >= run_until_update:
                    break
            progress_bar.close()

        if global_update != run_until_update:
            raise RuntimeError(f"Stage stopped at update {global_update}, expected {run_until_update}")
        if last_checkpoint_update != global_update:
            self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()
