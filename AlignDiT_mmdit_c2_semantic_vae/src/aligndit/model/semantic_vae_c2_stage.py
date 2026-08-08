"""Strict checkpoint migration and parameter policies for Semantic-VAE C2.

S3a is the only representation-to-multimodal boundary: it imports the S2c
pure-audio EMA, explicitly drops the obsolete HuBERT projector, and trains
only parameters that do not belong to the imported audio path.  S3b imports
the complete S3a EMA but starts a fresh optimizer, scheduler, and EMA.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


S3_STAGES = ("s3a", "s3b")
S3A_SOURCE_KIND = "s2c_audio"
S3B_SOURCE_KIND = "s3a_c2"
ParentKind = Literal["s2c_audio", "s3a_c2"]

EXPECTED_S2C_SOURCE_KEYS = 313
EXPECTED_S3_TARGET_KEYS = 703
EXPECTED_S2C_LOADED_KEYS = 303
EXPECTED_S2C_IGNORED_KEYS = 10
EXPECTED_S3_NEW_KEYS = 400
EXPECTED_S3_PARAMETER_TENSORS = 702
EXPECTED_LOADED_AUDIO_PARAMETER_TENSORS = 302
EXPECTED_NEW_MULTIMODAL_PARAMETER_TENSORS = 400

S2C_IGNORED_PROJECTOR_KEYS = frozenset(
    {f"transformer.projectors.0.model.{layer}.{suffix}" for layer in (0, 1, 3, 4, 6) for suffix in ("bias", "weight")}
)
EMA_BOOKKEEPING_KEYS = frozenset({"initted", "step"})
NEW_MM_BLOCK_MODULES = frozenset(
    {
        "cross_attn",
        "cross_attn_ada",
        "v_attn_norm",
        "v_attn",
        "v_ff",
    }
)


@dataclass(frozen=True)
class S3MigrationReport:
    parent_kind: str
    parent_path: str
    parent_sha256: str
    parent_size: int
    parent_contract_sha256: str
    parent_stage: str
    parent_update: int
    parent_ema_step: int
    source_key_count: int
    target_key_count: int
    loaded_key_count: int
    ignored_source_keys: tuple[str, ...]
    new_target_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    loaded_numel: int
    target_numel: int

    @property
    def loaded_fraction(self) -> float:
        return self.loaded_numel / self.target_numel

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["loaded_fraction"] = self.loaded_fraction
        return result


@dataclass(frozen=True)
class S3ParameterReport:
    stage: str
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    category_names: dict[str, tuple[str, ...]]
    trainable_numel: int
    frozen_numel: int
    optimizer_group_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Parent checkpoint must contain a mapping: {path}")
    return checkpoint


def _strict_scalar(value: Any, *, name: str, expected: int | bool) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be scalar, got shape={tuple(value.shape)}")
        value = value.item()
    if type(value) is not type(expected) or value != expected:
        raise RuntimeError(f"{name} mismatch: expected {expected!r}, got {value!r}")


def _extract_ema_state(checkpoint: dict[str, Any], *, expected_step: int) -> dict[str, torch.Tensor]:
    ema_state = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema_state, dict):
        raise TypeError("Parent checkpoint has no EMA state mapping")
    if set(EMA_BOOKKEEPING_KEYS) - set(ema_state):
        raise RuntimeError("Parent EMA is missing initted/step bookkeeping")
    _strict_scalar(ema_state["initted"], name="parent EMA initted", expected=True)
    _strict_scalar(ema_state["step"], name="parent EMA step", expected=expected_step)

    model_state: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for key, value in ema_state.items():
        if key in EMA_BOOKKEEPING_KEYS:
            continue
        if not key.startswith("ema_model."):
            unexpected.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Parent EMA value for {key!r} is not a tensor")
        model_state[key.removeprefix("ema_model.")] = value
    if unexpected:
        raise RuntimeError(f"Unexpected parent EMA keys outside ema_model.*: {sorted(unexpected)}")
    return model_state


def _validate_parent_schema(
    checkpoint: dict[str, Any],
    *,
    parent_kind: ParentKind,
    expected_parent_stage: str,
    expected_parent_update: int,
    expected_parent_contract_sha256: str,
) -> int:
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise RuntimeError("Parent checkpoint_schema_version must be 1")
    if checkpoint.get("training_contract_sha256") != expected_parent_contract_sha256:
        raise RuntimeError(
            "Parent checkpoint contract mismatch: "
            f"expected {expected_parent_contract_sha256}, got {checkpoint.get('training_contract_sha256')!r}"
        )

    if parent_kind == S3A_SOURCE_KIND:
        if checkpoint.get("warmstart_stage") != expected_parent_stage:
            raise RuntimeError(
                f"S2c parent stage mismatch: expected {expected_parent_stage!r}, "
                f"got {checkpoint.get('warmstart_stage')!r}"
            )
        _strict_scalar(checkpoint.get("update"), name="S2c parent update", expected=expected_parent_update)
        return expected_parent_update

    if checkpoint.get("semantic_vae_c2_stage") != expected_parent_stage:
        raise RuntimeError(
            f"S3 parent stage mismatch: expected {expected_parent_stage!r}, "
            f"got {checkpoint.get('semantic_vae_c2_stage')!r}"
        )
    _strict_scalar(checkpoint.get("stage_update"), name="S3 parent stage_update", expected=expected_parent_update)
    expected_cumulative = 5_000
    _strict_scalar(
        checkpoint.get("cumulative_update"),
        name="S3 parent cumulative_update",
        expected=expected_cumulative,
    )
    _strict_scalar(checkpoint.get("update"), name="S3 parent update", expected=expected_cumulative)
    return expected_parent_update


def load_s3_parent_ema(
    model: nn.Module,
    parent_path: str | Path,
    *,
    parent_kind: ParentKind,
    expected_parent_stage: str,
    expected_parent_update: int,
    expected_parent_contract_sha256: str,
    expected_parent_sha256: str,
    expected_parent_size: int,
) -> S3MigrationReport:
    """Strictly migrate the permitted parent EMA tensors into ``model``."""

    if parent_kind not in {S3A_SOURCE_KIND, S3B_SOURCE_KIND}:
        raise ValueError(f"Unknown S3 parent kind: {parent_kind!r}")
    if len(expected_parent_sha256) != 64:
        raise ValueError("expected_parent_sha256 must be a SHA256 hex digest")
    if len(expected_parent_contract_sha256) != 64:
        raise ValueError("expected_parent_contract_sha256 must be a SHA256 hex digest")
    if expected_parent_size <= 0:
        raise ValueError("expected_parent_size must be positive")

    candidate = Path(parent_path)
    if candidate.is_symlink():
        raise FileNotFoundError(f"S3 parent must not be a symlink: {candidate}")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"S3 parent must be a regular file: {path}")
    identity = _file_identity(path)
    if identity[2] != expected_parent_size:
        raise RuntimeError(f"Parent size mismatch: expected {expected_parent_size}, got {identity[2]}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_parent_sha256:
        raise RuntimeError(f"Parent SHA256 mismatch: expected {expected_parent_sha256}, got {actual_sha256}")
    if _file_identity(path) != identity:
        raise RuntimeError("Parent changed while its SHA256 was being verified")

    checkpoint = _load_checkpoint(path)
    if _file_identity(path) != identity:
        raise RuntimeError("Parent changed between integrity verification and checkpoint loading")
    parent_ema_step = _validate_parent_schema(
        checkpoint,
        parent_kind=parent_kind,
        expected_parent_stage=expected_parent_stage,
        expected_parent_update=expected_parent_update,
        expected_parent_contract_sha256=expected_parent_contract_sha256,
    )
    source_state = _extract_ema_state(checkpoint, expected_step=parent_ema_step)
    target_state = model.state_dict()

    source_keys = set(source_state)
    target_keys = set(target_state)
    ignored_source = sorted(source_keys - target_keys)
    new_target = sorted(target_keys - source_keys)
    common_keys = sorted(source_keys & target_keys)
    shape_mismatches = sorted(
        key for key in common_keys if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    )

    if parent_kind == S3A_SOURCE_KIND:
        expected_counts = (
            EXPECTED_S2C_SOURCE_KEYS,
            EXPECTED_S3_TARGET_KEYS,
            EXPECTED_S2C_LOADED_KEYS,
            EXPECTED_S2C_IGNORED_KEYS,
            EXPECTED_S3_NEW_KEYS,
        )
        actual_counts = (
            len(source_state),
            len(target_state),
            len(common_keys) - len(shape_mismatches),
            len(ignored_source),
            len(new_target),
        )
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"S2c-to-S3a migration count mismatch: expected={expected_counts}, actual={actual_counts}"
            )
        if set(ignored_source) != S2C_IGNORED_PROJECTOR_KEYS:
            raise RuntimeError(
                f"Unexpected S2c-only keys: expected={sorted(S2C_IGNORED_PROJECTOR_KEYS)}, actual={ignored_source}"
            )
    else:
        if len(source_state) != EXPECTED_S3_TARGET_KEYS or len(target_state) != EXPECTED_S3_TARGET_KEYS:
            raise RuntimeError(
                f"S3a-to-S3b schema count mismatch: source={len(source_state)}, target={len(target_state)}"
            )
        if ignored_source or new_target:
            raise RuntimeError(
                f"S3a-to-S3b must be a full-key transfer: source_only={ignored_source}, target_only={new_target}"
            )
    if shape_mismatches:
        raise RuntimeError(f"S3 parent contains shape mismatches: {shape_mismatches}")

    merged_state = dict(target_state)
    loaded_numel = 0
    for key in common_keys:
        value = source_state[key]
        target = target_state[key]
        if value.shape != target.shape:
            raise RuntimeError(f"Refusing shape-changing S3 migration for {key}: {value.shape} -> {target.shape}")
        merged_state[key] = value.to(dtype=target.dtype)
        loaded_numel += target.numel()
    model.load_state_dict(merged_state, strict=True)

    return S3MigrationReport(
        parent_kind=parent_kind,
        parent_path=str(path),
        parent_sha256=actual_sha256,
        parent_size=identity[2],
        parent_contract_sha256=expected_parent_contract_sha256,
        parent_stage=expected_parent_stage,
        parent_update=expected_parent_update,
        parent_ema_step=parent_ema_step,
        source_key_count=len(source_state),
        target_key_count=len(target_state),
        loaded_key_count=len(common_keys),
        ignored_source_keys=tuple(ignored_source),
        new_target_keys=tuple(new_target),
        shape_mismatches=tuple(shape_mismatches),
        loaded_numel=loaded_numel,
        target_numel=sum(value.numel() for value in target_state.values()),
    )


def _is_new_multimodal_parameter(name: str) -> bool:
    if name.startswith(("transformer.text_embed.", "transformer.video_embed.", "transformer.projectors_ctc.")):
        return True
    prefix = "transformer.transformer_blocks."
    if not name.startswith(prefix):
        return False
    parts = name.split(".")
    if len(parts) < 4:
        raise RuntimeError(f"Cannot parse Transformer block parameter: {name!r}")
    try:
        block_index = int(parts[2])
    except ValueError as error:
        raise RuntimeError(f"Cannot parse Transformer block index: {name!r}") from error
    return block_index < 12 and parts[3] in NEW_MM_BLOCK_MODULES


def _s3b_parameter_category(name: str) -> str:
    if _is_new_multimodal_parameter(name):
        return "multimodal_new"
    if name.startswith(("transformer.input_embed.proj.", "transformer.proj_out.")):
        return "interface"
    block_prefix = "transformer.transformer_blocks."
    if name.startswith(block_prefix):
        try:
            block_index = int(name.split(".")[2])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Cannot parse audio block index from {name!r}") from error
        return "audio_blocks_0_5" if block_index < 6 else "audio_backbone_rest"
    if name.startswith("transformer."):
        return "audio_backbone_rest"
    raise RuntimeError(f"Unexpected parameter outside the Semantic-VAE C2 transformer: {name}")


def configure_s3_parameters(
    model: nn.Module,
    *,
    stage: str,
    learning_rates: dict[str, float],
    weight_decay: float,
) -> tuple[list[dict[str, Any]], S3ParameterReport]:
    """Apply S3a/S3b freezing and build complete, non-overlapping AdamW groups."""

    if stage not in S3_STAGES:
        raise ValueError(f"Unknown Semantic-VAE C2 stage: {stage!r}")
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}")

    named_parameters = list(model.named_parameters())
    if len(named_parameters) != EXPECTED_S3_PARAMETER_TENSORS:
        raise RuntimeError(
            f"Semantic-VAE C2 parameter schema drift: expected {EXPECTED_S3_PARAMETER_TENSORS} tensors, "
            f"got {len(named_parameters)}"
        )
    names = [name for name, _ in named_parameters]
    if len(names) != len(set(names)):
        raise RuntimeError("Semantic-VAE C2 model exposes duplicate parameter names")

    categories: dict[str, list[str]] = {}
    grouped_parameters: dict[tuple[str, bool], list[nn.Parameter]] = {}
    grouped_names: dict[tuple[str, bool], list[str]] = {}
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_numel = 0
    frozen_numel = 0
    new_count = 0
    loaded_count = 0

    for name, parameter in named_parameters:
        category = _s3b_parameter_category(name)
        categories.setdefault(category, []).append(name)
        is_new = category == "multimodal_new"
        new_count += int(is_new)
        loaded_count += int(not is_new)
        parameter.requires_grad = is_new if stage == "s3a" else True
        if not parameter.requires_grad:
            frozen_names.append(name)
            frozen_numel += parameter.numel()
            continue
        if category not in learning_rates:
            raise RuntimeError(f"Missing learning rate for active category {category!r} ({name})")
        learning_rate = learning_rates[category]
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"Learning rate for {category!r} must be positive, got {learning_rate!r}")
        no_decay = parameter.ndim < 2 or name.endswith(".bias")
        group_key = (category, no_decay)
        grouped_parameters.setdefault(group_key, []).append(parameter)
        grouped_names.setdefault(group_key, []).append(name)
        trainable_names.append(name)
        trainable_numel += parameter.numel()

    if new_count != EXPECTED_NEW_MULTIMODAL_PARAMETER_TENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_MULTIMODAL_PARAMETER_TENSORS} new multimodal parameter tensors, got {new_count}"
        )
    if loaded_count != EXPECTED_LOADED_AUDIO_PARAMETER_TENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_LOADED_AUDIO_PARAMETER_TENSORS} imported audio parameter tensors, got {loaded_count}"
        )
    expected_active = new_count if stage == "s3a" else len(named_parameters)
    if len(trainable_names) != expected_active:
        raise RuntimeError(
            f"Stage {stage} trainable tensor count mismatch: expected {expected_active}, got {len(trainable_names)}"
        )

    optimizer_groups: list[dict[str, Any]] = []
    optimizer_group_names: list[str] = []
    for category, no_decay in sorted(grouped_parameters, key=lambda item: (item[0], item[1])):
        group_name = f"{category}.{'no_decay' if no_decay else 'decay'}"
        optimizer_groups.append(
            {
                "params": grouped_parameters[(category, no_decay)],
                "lr": float(learning_rates[category]),
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "group_name": group_name,
                "parameter_names": tuple(grouped_names[(category, no_decay)]),
            }
        )
        optimizer_group_names.append(group_name)

    optimizer_names = [name for group in optimizer_groups for name in group["parameter_names"]]
    if len(optimizer_names) != len(set(optimizer_names)) or set(optimizer_names) != set(trainable_names):
        raise RuntimeError("S3 optimizer groups overlap or do not cover every trainable parameter exactly once")

    return optimizer_groups, S3ParameterReport(
        stage=stage,
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
        category_names={key: tuple(value) for key, value in sorted(categories.items())},
        trainable_numel=trainable_numel,
        frozen_numel=frozen_numel,
        optimizer_group_names=tuple(optimizer_group_names),
    )
