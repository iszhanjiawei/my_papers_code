"""Strict mel-to-Semantic-VAE warm-start policies.

This module deliberately contains no training loop.  It defines the only
weights that may cross representation or stage boundaries and the exact
parameter groups trained by each adaptation stage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


S1_RESET_PREFIXES = (
    "transformer.input_embed.proj.",
    "transformer.proj_out.",
    "transformer.projectors.",
)
S1_EXPECTED_SHAPE_MISMATCHES = frozenset(
    {
        "transformer.input_embed.proj.weight",
        "transformer.proj_out.weight",
        "transformer.proj_out.bias",
    }
)
EMA_BOOKKEEPING_KEYS = frozenset({"initted", "step"})
KNOWN_LEGACY_MEL_BUFFERS = frozenset(
    {
        "mel_spec.mel_stft.mel_scale.fb",
        "mel_spec.mel_stft.spectrogram.window",
    }
)
STAGE_NAMES = ("s1", "s2a", "s2b", "s2c")
S1_QK_NORM_PATTERN = re.compile(r"transformer\.transformer_blocks\.[0-9]+\.attn\.(?:q_norm|k_norm)\.weight")


@dataclass(frozen=True)
class WarmStartLoadReport:
    parent_path: str
    parent_update: int
    source_key_count: int
    target_key_count: int
    loaded_key_count: int
    reset_keys: tuple[str, ...]
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
class TrainableParameterReport:
    stage: str
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_numel: int
    frozen_numel: int
    optimizer_group_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-start parent checkpoint does not exist: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        # mmap was added after weights_only.  Keep the correctness contract on
        # older PyTorch builds while accepting the additional host-memory use.
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _extract_ema_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "ema_model_state_dict" not in checkpoint:
        raise RuntimeError("Warm-start parent checkpoint has no ema_model_state_dict")
    ema_state = checkpoint["ema_model_state_dict"]
    if not isinstance(ema_state, dict):
        raise TypeError("ema_model_state_dict must be a mapping")

    model_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    for key, value in ema_state.items():
        if key in EMA_BOOKKEEPING_KEYS:
            continue
        if not key.startswith("ema_model."):
            unexpected_keys.append(key)
            continue
        model_key = key.removeprefix("ema_model.")
        if model_key in KNOWN_LEGACY_MEL_BUFFERS:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"EMA model value for {key!r} is not a tensor")
        model_state[model_key] = value

    if unexpected_keys:
        raise RuntimeError(f"Unexpected EMA keys outside ema_model.*: {sorted(unexpected_keys)}")
    if not model_state:
        raise RuntimeError("Warm-start parent EMA contains no model tensors")
    return model_state


def _scalar_value(value: Any, *, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"Warm-start parent EMA {field} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    return value


def _validate_ema_bookkeeping(checkpoint: dict[str, Any], *, expected_step: int) -> None:
    ema_state = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema_state, dict):
        raise TypeError("ema_model_state_dict must be a mapping")
    if "initted" not in ema_state or "step" not in ema_state:
        raise RuntimeError("Warm-start parent EMA must contain initted and step bookkeeping")

    initted = _scalar_value(ema_state["initted"], field="initted")
    if not isinstance(initted, bool) or not initted:
        raise RuntimeError(f"Warm-start parent EMA is not initialized: initted={initted!r}")
    step = _scalar_value(ema_state["step"], field="step")
    if not isinstance(step, int) or isinstance(step, bool) or step != expected_step:
        raise RuntimeError(f"Warm-start parent EMA step mismatch: expected {expected_step}, got {step!r}")


def _checkpoint_update(checkpoint: dict[str, Any]) -> int:
    if "update" not in checkpoint:
        raise RuntimeError("Warm-start parent checkpoint has no exact update field")
    update = checkpoint["update"]
    if isinstance(update, torch.Tensor):
        update = update.item()
    if not isinstance(update, int) or isinstance(update, bool) or update < 0:
        raise RuntimeError(f"Invalid warm-start parent update: {update!r}")
    return update


def load_parent_ema_weights(
    model: nn.Module,
    parent_path: str | Path,
    *,
    stage: str,
    expected_parent_update: int,
    expected_parent_stage: str | None = None,
    expected_parent_contract_sha256: str | None = None,
    expected_parent_sha256: str | None = None,
    expected_parent_size: int | None = None,
) -> WarmStartLoadReport:
    """Load only the permitted EMA tensors into an unwrapped target model.

    S1 crosses representations and therefore resets both 64-D interfaces and
    the semantically incompatible 50-Hz alignment projector.  Later stages
    require an exact, full-shape transfer from the preceding 64-D stage.
    Optimizer, scheduler, online-model weights, EMA counters and source update
    are never restored here.
    """

    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown Semantic-VAE warm-start stage: {stage!r}")
    if not isinstance(expected_parent_update, int) or expected_parent_update < 0:
        raise ValueError(f"expected_parent_update must be a non-negative integer, got {expected_parent_update!r}")

    parent_path = Path(parent_path).resolve()
    identity_before = _file_identity(parent_path)
    if expected_parent_size is not None and identity_before[2] != expected_parent_size:
        raise RuntimeError(
            f"Warm-start parent size mismatch: expected {expected_parent_size}, got {identity_before[2]}"
        )
    if expected_parent_sha256 is not None:
        actual_sha256 = _sha256_file(parent_path)
        if actual_sha256 != expected_parent_sha256:
            raise RuntimeError(
                f"Warm-start parent SHA256 mismatch: expected {expected_parent_sha256}, got {actual_sha256}"
            )
        if _file_identity(parent_path) != identity_before:
            raise RuntimeError("Warm-start parent changed while its SHA256 was being verified")

    checkpoint = _load_checkpoint(parent_path)
    if _file_identity(parent_path) != identity_before:
        raise RuntimeError("Warm-start parent changed between integrity verification and checkpoint loading")
    parent_update = _checkpoint_update(checkpoint)
    if parent_update != expected_parent_update:
        raise RuntimeError(f"Warm-start parent update mismatch: expected {expected_parent_update}, got {parent_update}")
    _validate_ema_bookkeeping(checkpoint, expected_step=parent_update)
    actual_parent_stage = checkpoint.get("warmstart_stage")
    if actual_parent_stage != expected_parent_stage:
        raise RuntimeError(
            f"Warm-start parent stage mismatch: expected {expected_parent_stage!r}, got {actual_parent_stage!r}"
        )
    actual_parent_contract = checkpoint.get("training_contract_sha256")
    if actual_parent_contract != expected_parent_contract_sha256:
        raise RuntimeError(
            "Warm-start parent contract mismatch: "
            f"expected {expected_parent_contract_sha256!r}, got {actual_parent_contract!r}"
        )
    source_state = _extract_ema_model_state(checkpoint)
    target_state = model.state_dict()

    source_keys = set(source_state)
    target_keys = set(target_state)
    missing_source = sorted(target_keys - source_keys)
    extra_source = sorted(source_keys - target_keys)
    expected_target_only = (
        sorted(key for key in target_keys if S1_QK_NORM_PATTERN.fullmatch(key)) if stage == "s1" else []
    )
    if missing_source != expected_target_only or extra_source:
        raise RuntimeError(
            "Warm-start model schema mismatch: "
            f"expected_target_only={expected_target_only}, "
            f"missing_source={missing_source}, extra_source={extra_source}"
        )

    reset_prefixes = S1_RESET_PREFIXES if stage == "s1" else ()
    reset_keys = sorted(
        key
        for key in target_state
        if key.startswith(reset_prefixes) or (stage == "s1" and S1_QK_NORM_PATTERN.fullmatch(key))
    )
    shape_mismatches = sorted(
        key
        for key in target_state
        if key in source_state and tuple(source_state[key].shape) != tuple(target_state[key].shape)
    )
    expected_mismatches = S1_EXPECTED_SHAPE_MISMATCHES if stage == "s1" else frozenset()
    if set(shape_mismatches) != expected_mismatches:
        raise RuntimeError(
            "Unexpected warm-start shape mismatch set: "
            f"expected={sorted(expected_mismatches)}, actual={shape_mismatches}"
        )
    if any(key not in reset_keys for key in shape_mismatches):
        raise RuntimeError(f"A shape-mismatched tensor is not covered by the semantic reset policy: {shape_mismatches}")

    merged_state = dict(target_state)
    loaded_keys: list[str] = []
    loaded_numel = 0
    for key, target_value in target_state.items():
        if key in reset_keys:
            continue
        if key not in source_state:
            raise RuntimeError(f"Target tensor {key!r} is neither loaded nor covered by the reset policy")
        source_value = source_state[key]
        if source_value.shape != target_value.shape:
            raise RuntimeError(
                f"Refusing shape-changing warm-start for {key}: {source_value.shape} -> {target_value.shape}"
            )
        merged_state[key] = source_value.to(dtype=target_value.dtype)
        loaded_keys.append(key)
        loaded_numel += target_value.numel()

    model.load_state_dict(merged_state, strict=True)
    target_numel = sum(value.numel() for value in target_state.values())
    minimum_fraction = 0.85 if stage == "s1" else 1.0
    loaded_fraction = loaded_numel / target_numel
    if loaded_fraction + 1e-12 < minimum_fraction:
        raise RuntimeError(f"Warm-start loaded only {loaded_fraction:.6%}; required at least {minimum_fraction:.0%}")

    return WarmStartLoadReport(
        parent_path=str(Path(parent_path).resolve()),
        parent_update=parent_update,
        source_key_count=len(source_state),
        target_key_count=len(target_state),
        loaded_key_count=len(loaded_keys),
        reset_keys=tuple(reset_keys),
        shape_mismatches=tuple(shape_mismatches),
        loaded_numel=loaded_numel,
        target_numel=target_numel,
    )


def _parameter_category(name: str) -> str:
    if name.startswith(("transformer.input_embed.proj.", "transformer.proj_out.")):
        return "interface"
    if name.startswith("transformer.projectors."):
        return "projector"
    if name.startswith("transformer.transformer_blocks."):
        parts = name.split(".")
        try:
            block_index = int(parts[2])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Cannot parse Transformer block index from parameter {name!r}") from error
        return "early_backbone" if block_index < 6 else "backbone"
    if name.startswith("transformer."):
        return "backbone"
    raise RuntimeError(f"Unexpected trainable model parameter outside transformer: {name}")


def _stage_allows_parameter(stage: str, name: str) -> bool:
    if name.startswith(("transformer.input_embed.proj.", "transformer.proj_out.")):
        return True
    if stage == "s1":
        return False
    if name.startswith(("transformer.input_embed.conv_pos_embed.", "transformer.norm_out.")):
        return True
    if name.startswith("transformer.transformer_blocks."):
        block_index = int(name.split(".")[2])
        if stage == "s2a":
            return block_index >= 12
        if stage == "s2b":
            return block_index >= 6
        return stage == "s2c"
    return stage == "s2c" and name.startswith("transformer.")


def configure_stage_parameters(
    model: nn.Module,
    *,
    stage: str,
    learning_rates: dict[str, float],
    weight_decay: float,
) -> tuple[list[dict[str, Any]], TrainableParameterReport]:
    """Freeze the stage and return deterministic AdamW parameter groups."""

    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown Semantic-VAE warm-start stage: {stage!r}")
    if not isinstance(weight_decay, (float, int)) or weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}")

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("Warm-start model has no trainable parameters")
    for name, parameter in named_parameters:
        parameter.requires_grad = _stage_allows_parameter(stage, name)

    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    grouped_names: dict[tuple[str, bool], list[str]] = {}
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_numel = 0
    frozen_numel = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            frozen_names.append(name)
            frozen_numel += parameter.numel()
            continue
        category = _parameter_category(name)
        if category not in learning_rates:
            raise RuntimeError(f"Missing learning rate for active parameter category {category!r} ({name})")
        learning_rate = learning_rates[category]
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError(f"Learning rate for {category!r} must be positive, got {learning_rate!r}")
        no_decay = parameter.ndim < 2 or name.endswith(".bias")
        key = (category, no_decay)
        grouped.setdefault(key, []).append(parameter)
        grouped_names.setdefault(key, []).append(name)
        trainable_names.append(name)
        trainable_numel += parameter.numel()

    if not trainable_names:
        raise RuntimeError(f"Warm-start stage {stage} selected no parameters")

    optimizer_groups: list[dict[str, Any]] = []
    optimizer_group_names: list[str] = []
    for category, no_decay in sorted(grouped, key=lambda item: (item[0], item[1])):
        group_name = f"{category}.{'no_decay' if no_decay else 'decay'}"
        optimizer_groups.append(
            {
                "params": grouped[(category, no_decay)],
                "lr": float(learning_rates[category]),
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "group_name": group_name,
                "parameter_names": tuple(grouped_names[(category, no_decay)]),
            }
        )
        optimizer_group_names.append(group_name)

    return optimizer_groups, TrainableParameterReport(
        stage=stage,
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
        trainable_numel=trainable_numel,
        frozen_numel=frozen_numel,
        optimizer_group_names=tuple(optimizer_group_names),
    )
