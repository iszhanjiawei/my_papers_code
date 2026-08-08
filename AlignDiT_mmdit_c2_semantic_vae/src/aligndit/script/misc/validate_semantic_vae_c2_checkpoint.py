"""Validate a complete, strictly resumable Semantic-VAE C2 stage checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from aligndit.model.semantic_vae_c2_stage import EXPECTED_S3_TARGET_KEYS, sha256_file
from aligndit.model.trainer_semantic_vae_c2 import (
    S3_FINAL_CUMULATIVE_UPDATE,
    S3_STAGE_MAX_UPDATES,
    S3_STAGE_START_UPDATE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(S3_STAGE_MAX_UPDATES), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-stage-update", type=int)
    return parser.parse_args()


def _scalar(value: Any, *, field: str) -> int | bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{field} must be scalar, got shape={tuple(value.shape)}")
        value = value.item()
    if type(value) not in {int, bool}:
        raise RuntimeError(f"{field} must be an int/bool scalar, got {value!r}")
    return value


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular checkpoint file: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    return value


def _update_semantic_digest(digest: Any, value: Any, *, field: str) -> None:
    """Hash nested checkpoint state independent of torch.save container metadata."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite tensor in {field}")
        # Flatten first because PyTorch cannot reinterpret a 0-D tensor as a
        # byte dtype with a different element size. Optimizer step counters
        # are commonly stored as scalar tensors in otherwise valid states.
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        return
    if isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_semantic_digest(digest, key, field=f"{field}.<key>")
            _update_semantic_digest(digest, value[key], field=f"{field}.{key}")
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}[".encode())
        for index, item in enumerate(value):
            _update_semantic_digest(digest, item, field=f"{field}[{index}]")
        digest.update(b"]")
        return
    if value is None or isinstance(value, (str, int, bool)):
        digest.update(f"{type(value).__name__}:{value!r};".encode())
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite scalar in {field}: {value!r}")
        digest.update(f"float:{value.hex()};".encode())
        return
    raise TypeError(f"Unsupported checkpoint value in {field}: {type(value).__name__}")


def _semantic_digest(value: Any, *, field: str) -> str:
    digest = hashlib.sha256()
    _update_semantic_digest(digest, value, field=field)
    return digest.hexdigest()


def _expected_parameter_contract(contract: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]], int]:
    parameter_policy = contract.get("parameter_policy")
    report = parameter_policy.get("report") if isinstance(parameter_policy, dict) else None
    categories = report.get("category_names") if isinstance(report, dict) else None
    groups = parameter_policy.get("group_membership") if isinstance(parameter_policy, dict) else None
    if not isinstance(categories, dict) or not isinstance(groups, list):
        raise TypeError("S3 contract has no exact parameter/group membership policy")
    parameter_names = [name for names in categories.values() for name in names]
    if len(parameter_names) != 702 or len(set(parameter_names)) != 702:
        raise RuntimeError("S3 contract parameter schema must contain 702 unique tensors")
    world_size = contract.get("distributed_runtime", {}).get("num_processes")
    if type(world_size) is not int or world_size <= 0:
        raise RuntimeError("S3 contract has no valid distributed world size")
    return set(parameter_names), groups, world_size


def _validate_optimizer_scheduler(
    checkpoint: dict[str, Any],
    *,
    path: Path,
    contract_groups: list[dict[str, Any]],
    stage_update: int,
    world_size: int,
) -> None:
    optimizer = checkpoint.get("optimizer_state_dict")
    scheduler = checkpoint.get("scheduler_state_dict")
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise RuntimeError(f"{path.name} optimizer state has the wrong top-level schema")
    state = optimizer["state"]
    groups = optimizer["param_groups"]
    if not isinstance(state, dict) or not isinstance(groups, list) or len(groups) != len(contract_groups):
        raise RuntimeError(f"{path.name} optimizer groups disagree with the immutable contract")
    parameter_ids: list[int] = []
    for index, (actual, expected) in enumerate(zip(groups, contract_groups)):
        expected_names = expected.get("parameter_names")
        params = actual.get("params") if isinstance(actual, dict) else None
        if (
            not isinstance(expected_names, list)
            or not isinstance(params, list)
            or len(params) != len(expected_names)
            or actual.get("group_name") != expected.get("group_name")
            or float(actual.get("weight_decay", -1.0)) != float(expected.get("weight_decay", -2.0))
            or float(actual.get("initial_lr", -1.0)) != float(expected.get("lr", -2.0))
        ):
            raise RuntimeError(f"{path.name} optimizer group {index} disagrees with the immutable policy")
        if any(type(parameter_id) is not int for parameter_id in params):
            raise RuntimeError(f"{path.name} optimizer group {index} has non-integer parameter ids")
        parameter_ids.extend(params)
    if len(parameter_ids) != len(set(parameter_ids)) or not set(state).issubset(parameter_ids):
        raise RuntimeError(f"{path.name} optimizer parameter ids overlap or contain unknown state")

    if not isinstance(scheduler, dict):
        raise TypeError(f"{path.name} is missing the scheduler state mapping")
    expected_scheduler_step = stage_update * world_size
    if scheduler.get("last_epoch") != expected_scheduler_step:
        raise RuntimeError(
            f"{path.name} scheduler step mismatch: expected {expected_scheduler_step}, "
            f"got {scheduler.get('last_epoch')!r}"
        )
    scheduler_lrs = scheduler.get("_last_lr")
    optimizer_lrs = [group.get("lr") for group in groups]
    if scheduler_lrs != optimizer_lrs or len(optimizer_lrs) != len(contract_groups):
        raise RuntimeError(f"{path.name} scheduler/optimizer learning rates disagree")


def _validate_one(
    path: Path,
    *,
    stage: str,
    stage_update: int,
    cumulative_update: int,
    contract_sha256: str,
    parameter_names: set[str],
    contract_groups: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    checkpoint = _load(path)
    expected_scalars = {
        "checkpoint_schema_version": 1,
        "stage_start_update": S3_STAGE_START_UPDATE[stage],
        "stage_max_updates": S3_STAGE_MAX_UPDATES[stage],
        "stage_update": stage_update,
        "cumulative_update": cumulative_update,
        "update": cumulative_update,
    }
    for field, expected in expected_scalars.items():
        actual = _scalar(checkpoint.get(field), field=f"{path.name}:{field}")
        if actual != expected:
            raise RuntimeError(f"{path.name}:{field} mismatch: expected {expected}, got {actual!r}")
    if checkpoint.get("semantic_vae_c2_stage") != stage:
        raise RuntimeError(f"{path.name} stage mismatch")
    if checkpoint.get("training_contract_sha256") != contract_sha256:
        raise RuntimeError(f"{path.name} training contract mismatch")

    model_state = checkpoint.get("model_state_dict")
    ema_state = checkpoint.get("ema_model_state_dict")
    expected_state_keys = parameter_names | {"transformer.rotary_embed.inv_freq"}
    if not isinstance(model_state, dict) or set(model_state) != expected_state_keys:
        raise RuntimeError(
            f"{path.name} online model schema mismatch: expected {EXPECTED_S3_TARGET_KEYS} exact keys, "
            f"got {len(model_state) if isinstance(model_state, dict) else type(model_state)}"
        )
    if not isinstance(ema_state, dict):
        raise TypeError(f"{path.name} has no EMA state mapping")
    ema_model_keys = [key for key in ema_state if key.startswith("ema_model.")]
    if len(ema_model_keys) != EXPECTED_S3_TARGET_KEYS or set(ema_state) != set(ema_model_keys) | {
        "initted",
        "step",
    }:
        raise RuntimeError(f"{path.name} EMA schema mismatch")
    ema_model_state = {
        key.removeprefix("ema_model."): value for key, value in ema_state.items() if key.startswith("ema_model.")
    }
    if set(ema_model_state) != expected_state_keys:
        raise RuntimeError(f"{path.name} EMA model keys differ from the exact online-model schema")
    for key in expected_state_keys:
        online = model_state[key]
        ema = ema_model_state[key]
        if (
            not isinstance(online, torch.Tensor)
            or not isinstance(ema, torch.Tensor)
            or online.shape != ema.shape
            or online.dtype != ema.dtype
        ):
            raise RuntimeError(f"{path.name} online/EMA tensor schema mismatch for {key}")
    if _scalar(ema_state["step"], field=f"{path.name}:EMA step") != stage_update:
        raise RuntimeError(f"{path.name} EMA step mismatch")
    if _scalar(ema_state["initted"], field=f"{path.name}:EMA initted") is not True:
        raise RuntimeError(f"{path.name} EMA is not initialized")
    _validate_optimizer_scheduler(
        checkpoint,
        path=path,
        contract_groups=contract_groups,
        stage_update=stage_update,
        world_size=world_size,
    )

    resume_digests = {
        field: _semantic_digest(checkpoint[field], field=f"{path.name}:{field}")
        for field in (
            "model_state_dict",
            "optimizer_state_dict",
            "ema_model_state_dict",
            "scheduler_state_dict",
        )
    }

    result = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "stage": stage,
        "stage_update": stage_update,
        "cumulative_update": cumulative_update,
        "model_keys": len(model_state),
        "ema_step": stage_update,
        "resume_state_sha256": resume_digests,
    }
    del checkpoint
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    contract_path = checkpoint_dir / "training_contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        raise FileNotFoundError(f"Missing immutable S3 contract: {contract_path}")
    contract_sha256 = sha256_file(contract_path)
    with contract_path.open(encoding="utf-8") as file:
        contract = json.load(file)
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("Invalid S3 training contract schema")
    policy = contract.get("policy")
    if not isinstance(policy, dict) or policy.get("stage") != args.stage:
        raise RuntimeError("S3 training contract stage mismatch")
    parameter_names, contract_groups, world_size = _expected_parameter_contract(contract)

    stage_update = (
        S3_STAGE_MAX_UPDATES[args.stage] if args.expected_stage_update is None else args.expected_stage_update
    )
    if not 0 < stage_update <= S3_STAGE_MAX_UPDATES[args.stage]:
        raise ValueError(f"Invalid expected stage update: {stage_update}")
    cumulative_update = S3_STAGE_START_UPDATE[args.stage] + stage_update
    if stage_update == S3_STAGE_MAX_UPDATES[args.stage] and cumulative_update != S3_FINAL_CUMULATIVE_UPDATE[args.stage]:
        raise RuntimeError("Internal S3 cumulative update contract is inconsistent")

    serialized = sorted(
        path.name for path in checkpoint_dir.iterdir() if path.is_file() and path.suffix in {".pt", ".safetensors"}
    )
    unexpected = [
        name for name in serialized if name != "model_last.pt" and re.fullmatch(r"model_[0-9]+\.pt", name) is None
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected serialized files in {checkpoint_dir}: {unexpected}")

    numbered_path = checkpoint_dir / f"model_{cumulative_update}.pt"
    last_path = checkpoint_dir / "model_last.pt"
    save_per_updates = int(contract.get("config", {}).get("ckpts", {}).get("save_per_updates", 0))
    numbered_required = stage_update == S3_STAGE_MAX_UPDATES[args.stage] or (
        save_per_updates > 0 and cumulative_update % save_per_updates == 0
    )
    paths = [last_path]
    if numbered_required or numbered_path.exists() or numbered_path.is_symlink():
        paths.insert(0, numbered_path)
    reports = [
        _validate_one(
            path,
            stage=args.stage,
            stage_update=stage_update,
            cumulative_update=cumulative_update,
            contract_sha256=contract_sha256,
            parameter_names=parameter_names,
            contract_groups=contract_groups,
            world_size=world_size,
        )
        for path in paths
    ]
    if len(reports) == 2 and reports[0]["resume_state_sha256"] != reports[1]["resume_state_sha256"]:
        raise RuntimeError("Numbered and model_last checkpoints do not contain identical resumable state")
    print(
        json.dumps(
            {
                "contract_path": str(contract_path),
                "contract_sha256": contract_sha256,
                "stage": args.stage,
                "stage_update": stage_update,
                "cumulative_update": cumulative_update,
                "checkpoints": reports,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
