"""Fail-closed validation for a completed Semantic-VAE warm-start stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
STAGES = ("s1", "s2a", "s2b", "s2c")
PREVIOUS_STAGE = {"s1": None, "s2a": "s1", "s2b": "s2a", "s2c": "s2b"}
REQUIRED_CHECKPOINT_KEYS = {
    "checkpoint_schema_version",
    "ema_model_state_dict",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "training_contract_sha256",
    "update",
    "warmstart_stage",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be scalar, got tensor shape {tuple(value.shape)}")
        return value.item()
    return value


def _checkpoint_identity(checkpoint_path: Path) -> tuple[str | None, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint root must be a dict: {checkpoint_path}")
    update = _scalar(checkpoint.get("update"), name=f"checkpoint update in {checkpoint_path.name}")
    if not isinstance(update, int) or isinstance(update, bool) or update < 0:
        raise RuntimeError(f"Invalid checkpoint update in {checkpoint_path}: {update!r}")
    numbered_match = re.fullmatch(r"model_([0-9]+)[.]pt", checkpoint_path.name)
    if numbered_match is not None and int(numbered_match.group(1)) != update:
        raise RuntimeError(
            f"Checkpoint filename/update mismatch for {checkpoint_path}: "
            f"filename={numbered_match.group(1)}, update={update}"
        )
    return checkpoint.get("warmstart_stage"), update


def validate_resume_checkpoint_order(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Prevent model_last.pt from silently taking precedence over a newer numbered checkpoint."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        return {"checkpoint_dir": str(checkpoint_dir), "latest_update": None, "status": "empty"}
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Warm-start checkpoint path is not a directory: {checkpoint_dir}")
    serialized = sorted(
        path for path in checkpoint_dir.iterdir() if path.is_file() and path.suffix in {".pt", ".safetensors"}
    )
    recognized = [
        path for path in serialized if path.name == "model_last.pt" or re.fullmatch(r"model_[0-9]+[.]pt", path.name)
    ]
    unexpected = sorted(path.name for path in set(serialized) - set(recognized))
    if unexpected:
        raise RuntimeError(f"Unexpected serialized files in {checkpoint_dir}: {unexpected}")
    if not recognized:
        return {"checkpoint_dir": str(checkpoint_dir), "latest_update": None, "status": "empty"}

    last_path = checkpoint_dir / "model_last.pt"
    numbered = [path for path in recognized if path.name != "model_last.pt"]
    numbered.sort(key=lambda path: int(path.stem.split("_")[1]))
    identities = {path.name: _checkpoint_identity(path) for path in ([last_path] if last_path.is_file() else [])}
    if numbered:
        identities[numbered[-1].name] = _checkpoint_identity(numbered[-1])
    stages = {stage for stage, _ in identities.values()}
    if len(stages) != 1:
        raise RuntimeError(f"Checkpoint stage mismatch in {checkpoint_dir}: {identities}")
    last_update = identities.get("model_last.pt", (None, -1))[1]
    highest_numbered_update = identities.get(numbered[-1].name, (None, -1))[1] if numbered else -1
    if last_path.is_file() and last_update < highest_numbered_update:
        raise RuntimeError(
            f"Stale model_last.pt in {checkpoint_dir}: last={last_update}, numbered={highest_numbered_update}"
        )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "latest_update": max(last_update, highest_numbered_update),
        "stage": stages.pop(),
        "status": "resumable",
    }


def validate_completed_stage_checkpoint(
    checkpoint_path: str | Path,
    contract_path: str | Path,
    *,
    expected_stage: str,
    expected_update: int,
    expected_horizon: int,
    expected_model_keys: int = 313,
    expected_world_size: int | None = None,
    expected_mixed_precision: str | None = None,
    expected_frame_budget: int | None = None,
    expected_max_samples: int | None = None,
) -> dict[str, Any]:
    """Validate stage identity, contract binding, model/EMA schema, and completion update."""

    if expected_stage not in STAGES:
        raise ValueError(f"expected_stage must be one of {STAGES}, got {expected_stage!r}")
    if not 0 < expected_update <= expected_horizon:
        raise ValueError(f"expected_update must be within [1, {expected_horizon}], got {expected_update}")
    if expected_model_keys <= 0:
        raise ValueError(f"expected_model_keys must be positive, got {expected_model_keys}")

    checkpoint_path = Path(checkpoint_path).resolve()
    contract_path = Path(contract_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {checkpoint_path}")
    if not contract_path.is_file():
        raise FileNotFoundError(f"Warm-start contract does not exist: {contract_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint root must be a dict: {checkpoint_path}")
    actual_keys = set(checkpoint)
    if actual_keys != REQUIRED_CHECKPOINT_KEYS:
        raise RuntimeError(
            "Checkpoint schema keys mismatch: "
            f"missing={sorted(REQUIRED_CHECKPOINT_KEYS - actual_keys)}, "
            f"unexpected={sorted(actual_keys - REQUIRED_CHECKPOINT_KEYS)}"
        )
    if checkpoint["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint schema version mismatch: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, got {checkpoint['checkpoint_schema_version']!r}"
        )
    if checkpoint["warmstart_stage"] != expected_stage:
        raise RuntimeError(
            f"Warm-start stage mismatch: expected {expected_stage!r}, got {checkpoint['warmstart_stage']!r}"
        )
    actual_update = _scalar(checkpoint["update"], name="checkpoint update")
    if actual_update != expected_update:
        raise RuntimeError(f"Checkpoint update mismatch: expected {expected_update}, got {actual_update!r}")

    contract_sha256 = _sha256_file(contract_path)
    if checkpoint["training_contract_sha256"] != contract_sha256:
        raise RuntimeError(
            "Checkpoint training contract mismatch: "
            f"expected {contract_sha256}, got {checkpoint['training_contract_sha256']!r}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise RuntimeError(
            "Training contract schema mismatch: "
            f"expected {CONTRACT_SCHEMA_VERSION}, got {contract.get('schema_version')!r}"
        )
    if contract.get("policy", {}).get("stage") != expected_stage:
        raise RuntimeError(
            "Training contract stage mismatch: "
            f"expected {expected_stage!r}, got {contract.get('policy', {}).get('stage')!r}"
        )
    expected_previous_stage = PREVIOUS_STAGE[expected_stage]
    configured_previous_stage = contract.get("policy", {}).get("previous_stage")
    if configured_previous_stage != expected_previous_stage:
        raise RuntimeError(
            "Training contract previous-stage mismatch: "
            f"expected {expected_previous_stage!r}, got {configured_previous_stage!r}"
        )
    configured_stage = contract.get("config", {}).get("stage", {}).get("name")
    if configured_stage != expected_stage:
        raise RuntimeError(
            f"Training contract config stage mismatch: expected {expected_stage!r}, got {configured_stage!r}"
        )
    configured_horizon = contract.get("config", {}).get("optim", {}).get("max_updates")
    if configured_horizon != expected_horizon:
        raise RuntimeError(f"Training horizon mismatch: expected {expected_horizon}, got {configured_horizon!r}")
    runtime = contract.get("distributed_runtime", {})
    datasets = contract.get("config", {}).get("datasets", {})
    expected_runtime_values = {
        "num_processes": expected_world_size,
        "mixed_precision": expected_mixed_precision,
    }
    for key, expected_value in expected_runtime_values.items():
        if expected_value is not None and runtime.get(key) != expected_value:
            raise RuntimeError(
                f"Training contract runtime {key} mismatch: expected {expected_value!r}, got {runtime.get(key)!r}"
            )
    expected_dataset_values = {
        "batch_size_per_gpu": expected_frame_budget,
        "max_samples": expected_max_samples,
    }
    for key, expected_value in expected_dataset_values.items():
        if expected_value is not None and datasets.get(key) != expected_value:
            raise RuntimeError(
                f"Training contract dataset {key} mismatch: expected {expected_value!r}, got {datasets.get(key)!r}"
            )
    if datasets.get("batch_size_type") != "frame":
        raise RuntimeError(f"Training contract must use frame batching, got {datasets.get('batch_size_type')!r}")
    audio_representation = contract.get("config", {}).get("model", {}).get("audio_representation", {})
    if audio_representation.get("channels") != 64 or audio_representation.get("frame_rate") != 40:
        raise RuntimeError(f"Unexpected Semantic-VAE audio representation: {audio_representation}")

    model_state = checkpoint["model_state_dict"]
    ema_state = checkpoint["ema_model_state_dict"]
    if not isinstance(model_state, dict) or len(model_state) != expected_model_keys:
        actual_count = len(model_state) if isinstance(model_state, dict) else None
        raise RuntimeError(f"Model state key count mismatch: expected {expected_model_keys}, got {actual_count}")
    if not isinstance(ema_state, dict):
        raise TypeError("EMA state must be a dict")
    ema_initted = bool(_scalar(ema_state.get("initted"), name="EMA initted"))
    ema_step = _scalar(ema_state.get("step"), name="EMA step")
    if not ema_initted:
        raise RuntimeError("EMA is not initialized")
    if ema_step != expected_update:
        raise RuntimeError(f"EMA step mismatch: expected {expected_update}, got {ema_step!r}")

    expected_ema_keys = {"initted", "step", *(f"ema_model.{key}" for key in model_state)}
    actual_ema_keys = set(ema_state)
    if actual_ema_keys != expected_ema_keys:
        raise RuntimeError(
            "EMA/model key mismatch: "
            f"missing={sorted(expected_ema_keys - actual_ema_keys)}, "
            f"unexpected={sorted(actual_ema_keys - expected_ema_keys)}"
        )
    for key, online_value in model_state.items():
        ema_value = ema_state[f"ema_model.{key}"]
        if not isinstance(online_value, torch.Tensor) or not isinstance(ema_value, torch.Tensor):
            raise TypeError(f"Model and EMA values must be tensors for key {key!r}")
        if online_value.shape != ema_value.shape or online_value.dtype != ema_value.dtype:
            raise RuntimeError(
                f"Model/EMA tensor metadata mismatch for {key}: "
                f"online={online_value.shape}/{online_value.dtype}, ema={ema_value.shape}/{ema_value.dtype}"
            )
        for state_name, value in (("model", online_value), ("EMA", ema_value)):
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise RuntimeError(f"Non-finite {state_name} tensor in checkpoint: {key}")
    for state_name in ("optimizer_state_dict", "scheduler_state_dict"):
        if not isinstance(checkpoint[state_name], dict) or not checkpoint[state_name]:
            raise RuntimeError(f"{state_name} must be a non-empty dict")

    return {
        "checkpoint": str(checkpoint_path),
        "contract_sha256": contract_sha256,
        "ema_step": int(ema_step),
        "model_key_count": len(model_state),
        "stage": expected_stage,
        "update": int(actual_update),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--resume-directory", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--update", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--expected-model-keys", type=int, default=313)
    parser.add_argument("--expected-world-size", type=int, default=6)
    parser.add_argument("--expected-mixed-precision", default="bf16")
    parser.add_argument("--expected-frame-budget", type=int, default=7200)
    parser.add_argument("--expected-max-samples", type=int, default=32)
    args = parser.parse_args()
    if args.resume_directory is not None:
        print(
            json.dumps(
                validate_resume_checkpoint_order(args.resume_directory),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    required = {
        "contract": args.contract,
        "stage": args.stage,
        "update": args.update,
        "horizon": args.horizon,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        parser.error(f"checkpoint validation requires: {', '.join(missing)}")
    report = validate_completed_stage_checkpoint(
        args.checkpoint,
        args.contract,
        expected_stage=args.stage,
        expected_update=args.update,
        expected_horizon=args.horizon,
        expected_model_keys=args.expected_model_keys,
        expected_world_size=args.expected_world_size,
        expected_mixed_precision=args.expected_mixed_precision,
        expected_frame_budget=args.expected_frame_budget,
        expected_max_samples=args.expected_max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
