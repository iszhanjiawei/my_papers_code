"""Guarded one-stage C2 training for the minimal Semantic-VAE repair."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_seed

from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectC2Trainer


MINIMAL_FIX_CHECKPOINT_SCHEMA = 1
MINIMAL_FIX_POLICY = "semantic-vae40-c2-one-stage-minimal-fix-v2"


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_accelerator_save(accelerator, checkpoint: dict, destination: str) -> None:
    checkpoint_dir = os.path.dirname(destination)
    os.makedirs(checkpoint_dir, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=checkpoint_dir, delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            os.fchmod(temporary_file.fileno(), 0o644)
        accelerator.save(checkpoint, temporary_path)
        with open(temporary_path, "rb+") as temporary_file:
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(checkpoint_dir)
    except BaseException:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
            _fsync_directory(checkpoint_dir)
        raise


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def scale_safe_global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return an FP64-host aggregated L2 norm without global FP32 overflow.

    PyTorch's native clipping first forms per-parameter norms and then combines
    them on device.  Hundreds of individually finite, very large norms can make
    that final FP32 reduction overflow to ``inf``.  Combining the finite scalar
    norms with ``math.hypot`` preserves their scale and lets us fail before the
    native clip coefficient silently becomes zero.
    """

    local_norms: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.coalesce().values() if gradient.is_sparse else gradient
        # FP64 prevents a single tensor with finite BF16/FP32 elements from
        # overflowing while its norm is formed.  Only one scalar per tensor is
        # copied to the host for the scale-safe aggregate.
        local_norms.append(torch.linalg.vector_norm(gradient.detach(), ord=2, dtype=torch.float64))
    if not local_norms:
        return 0.0
    values = torch.stack(local_norms).detach().cpu().tolist()
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.hypot(*values)


@torch.no_grad()
def scale_safe_top_gradient_norms(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]], limit: int = 12
) -> list[dict[str, float | str]]:
    """Return the largest finite per-parameter gradient norms for spike forensics."""

    if limit <= 0:
        return []
    norms: list[dict[str, float | str]] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.coalesce().values() if gradient.is_sparse else gradient
        value = float(torch.linalg.vector_norm(gradient.detach(), ord=2, dtype=torch.float64).item())
        norms.append({"name": name, "norm": value})
    norms.sort(key=lambda item: float(item["norm"]), reverse=True)
    return norms[:limit]


class SemanticVaeMinimalFixC2Trainer(SemanticVaeDirectC2Trainer):
    """Original one-stage C2 policy plus deterministic numerical safeguards."""

    def __init__(
        self,
        *args,
        seed: int,
        run_until_update: int,
        global_grad_norm_warning_threshold: float,
        global_grad_norm_abort_threshold: float,
        global_grad_norm_min_threshold: float,
        post_text_rms_min: float,
        post_text_rms_max: float,
        experiment_contract: dict[str, Any],
        **kwargs,
    ) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if run_until_update <= 0:
            raise ValueError(f"run_until_update must be positive, got {run_until_update}")
        if not (
            0
            <= global_grad_norm_min_threshold
            < global_grad_norm_warning_threshold
            < global_grad_norm_abort_threshold
        ):
            raise ValueError("gradient norm thresholds must satisfy 0 <= min < warning < abort")
        if not 0 < post_text_rms_min < post_text_rms_max:
            raise ValueError("post-text RMS thresholds must satisfy 0 < min < max")
        self.repair_seed = int(seed)
        self.run_until_update = int(run_until_update)
        self.global_grad_norm_warning_threshold = float(global_grad_norm_warning_threshold)
        self.global_grad_norm_abort_threshold = float(global_grad_norm_abort_threshold)
        self.global_grad_norm_min_threshold = float(global_grad_norm_min_threshold)
        self.post_text_rms_min = float(post_text_rms_min)
        self.post_text_rms_max = float(post_text_rms_max)
        self.experiment_contract = experiment_contract
        self.experiment_contract_sha256 = _sha256_json(experiment_contract)
        self.current_update = 0
        self.last_forward_snapshot: dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        if self.grad_accumulation_steps != 1:
            raise ValueError("the reproducible minimal repair currently requires grad_accumulation_steps=1")

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = {
                "checkpoint_schema_version": MINIMAL_FIX_CHECKPOINT_SCHEMA,
                "training_policy": MINIMAL_FIX_POLICY,
                "experiment_contract_sha256": self.experiment_contract_sha256,
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_model_state_dict": self.ema_model.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "update": update,
            }
            destination = Path(self.checkpoint_path) / ("model_last.pt" if last else f"model_{update}.pt")
            _atomic_accelerator_save(self.accelerator, checkpoint, str(destination))
            print(f"Saved minimal-fix {'last' if last else 'numbered'} checkpoint at update {update}", flush=True)
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self):
        checkpoint_dir = Path(self.checkpoint_path)
        candidates = [path for path in checkpoint_dir.glob("model*.pt") if path.is_file()]
        if not candidates:
            return 0

        valid: list[tuple[int, Path]] = []
        errors: list[str] = []
        for path in candidates:
            try:
                checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                update = int(checkpoint["update"])
                if checkpoint.get("checkpoint_schema_version") != MINIMAL_FIX_CHECKPOINT_SCHEMA:
                    raise RuntimeError("wrong checkpoint schema")
                if checkpoint.get("training_policy") != MINIMAL_FIX_POLICY:
                    raise RuntimeError("wrong training policy")
                if checkpoint.get("experiment_contract_sha256") != self.experiment_contract_sha256:
                    raise RuntimeError("experiment contract mismatch")
                valid.append((update, path))
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
        if not valid:
            raise RuntimeError(f"No valid minimal-fix checkpoint found in {checkpoint_dir}: {errors}")

        update, path = max(valid, key=lambda item: item[0])
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        self.accelerator.wait_for_everyone()
        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.accelerator.wait_for_everyone()
        print(f"Loaded validated minimal-fix checkpoint {path.name} at update {update}", flush=True)
        return update

    def publish_contract(self) -> None:
        contract_path = Path(self.checkpoint_path) / "training_contract.json"
        self.accelerator.wait_for_everyone()
        if self.is_main:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(contract_path.read_text()) if contract_path.is_file() else None
            if existing is not None and existing != self.experiment_contract:
                raise RuntimeError(f"existing minimal-fix contract differs: {contract_path}")
            if existing is None:
                temporary = contract_path.with_suffix(".json.tmp")
                with temporary.open("w", encoding="utf-8") as file:
                    json.dump(self.experiment_contract, file, ensure_ascii=False, indent=2, sort_keys=True)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, contract_path)
        self.accelerator.wait_for_everyone()

    def finetune(self, pretrained_path, train_dataset, num_workers=16, resumable_with_seed: int | None = None):
        """Do not reload the parent when this exact experiment is resuming."""

        checkpoint_dir = Path(self.checkpoint_path)
        local_resume = any(
            path.is_file() and (path.name == "model_last.pt" or path.name.startswith("model_"))
            for path in checkpoint_dir.glob("model*.pt")
        )
        if not local_resume:
            self.load_pretrained(pretrained_path)
        self.train(train_dataset, num_workers, resumable_with_seed)

    def _synchronized_failure_count(self, local_failure: bool) -> int:
        flag = torch.tensor(int(local_failure), device=self.accelerator.device, dtype=torch.int32)
        return int(self.accelerator.reduce(flag, reduction="sum").item())

    def _before_update(self, global_update: int) -> None:
        self.current_update = int(global_update)
        update_seed = self.repair_seed * 1_000_003
        update_seed += global_update * self.accelerator.num_processes + self.accelerator.process_index
        set_seed(update_seed)

    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        transformer = getattr(self.accelerator.unwrap_model(self.model), "transformer", None)
        raw = getattr(transformer, "last_text_context_raw_rms", None)
        post = getattr(transformer, "last_text_context_post_rms", None)
        if not isinstance(raw, torch.Tensor) or raw.numel() != 1:
            raise RuntimeError("minimal repair requires a scalar last_text_context_raw_rms after every forward")
        if not isinstance(post, torch.Tensor) or post.numel() != 1:
            raise RuntimeError("minimal repair requires a scalar last_text_context_post_rms after every forward")
        raw_value = float(raw.detach().float().item())
        post_value = float(post.detach().float().item())
        component_values = [float(value) for value in loss_components.values()]
        local_failure = (
            not bool(torch.isfinite(loss.detach()).item())
            or any(not math.isfinite(value) for value in component_values)
            or not math.isfinite(raw_value)
            or not math.isfinite(post_value)
            or not self.post_text_rms_min <= post_value <= self.post_text_rms_max
        )
        failed_ranks = self._synchronized_failure_count(local_failure)
        if failed_ranks:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Unsafe forward state before backward: "
                f"update={self.current_update}, loss={float(loss.detach().float().item())!r}, "
                f"components={loss_components}, raw_text_rms={raw_value!r}, post_text_rms={post_value!r}, "
                f"required_post_range=[{self.post_text_rms_min}, {self.post_text_rms_max}], "
                f"failed_ranks={failed_ranks}/{self.accelerator.num_processes}"
            )
        self.last_forward_snapshot = {
            "loss": float(loss.detach().float().item()),
            "loss_components": {
                name: float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
                for name, value in loss_components.items()
            },
            "raw_text_rms": raw_value,
            "post_text_rms": post_value,
        }
        return {"text_context/raw_rms": raw_value, "text_context/post_rms": post_value}

    def _record_gradient_spike(self, safe_norm: float) -> None:
        report = {
            "current_update_before_step": self.current_update,
            "optimizer_step_after_success": self.current_update + 1,
            "global_grad_norm": safe_norm,
            "max_grad_norm": self.max_grad_norm,
            "clip_coefficient_upper_bound": min(1.0, self.max_grad_norm / safe_norm),
            "forward": self.last_forward_snapshot,
            "top_parameter_gradient_norms": scale_safe_top_gradient_norms(self.model.named_parameters()),
        }
        path = Path(self.checkpoint_path) / "gradient_spikes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        print(
            "Finite gradient spike will be clipped, not treated as numerical failure: "
            f"{json.dumps(report, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )

    def _clip_gradients(self) -> float:
        if not self.accelerator.sync_gradients:
            raise RuntimeError("minimal repair gradient guard was called before gradients synchronized")
        if self.max_grad_norm <= 0:
            raise RuntimeError("minimal repair requires max_grad_norm > 0")

        self.accelerator.unscale_gradients(self.optimizer)
        safe_norm = scale_safe_global_grad_norm(self.model.parameters())
        local_failure = (
            not math.isfinite(safe_norm)
            or safe_norm <= self.global_grad_norm_min_threshold
            or safe_norm > self.global_grad_norm_abort_threshold
        )
        failed_ranks = self._synchronized_failure_count(local_failure)
        if failed_ranks:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Unsafe scale-safe pre-clip gradient norm: "
                f"update={self.current_update}, local_norm={safe_norm!r}, "
                f"required_range=({self.global_grad_norm_min_threshold}, "
                f"{self.global_grad_norm_abort_threshold}], "
                f"failed_ranks={failed_ranks}/{self.accelerator.num_processes}"
            )

        if safe_norm > self.global_grad_norm_warning_threshold and self.accelerator.is_main_process:
            self._record_gradient_spike(safe_norm)

        native_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        native_value = float(native_norm.detach().float().item())
        failed_ranks = self._synchronized_failure_count(not math.isfinite(native_value))
        if failed_ranks:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Native gradient clipping became non-finite after a finite scale-safe check: "
                f"update={self.current_update}, safe_norm={safe_norm!r}, native_norm={native_value!r}, "
                f"failed_ranks={failed_ranks}/{self.accelerator.num_processes}"
            )
        return safe_norm
