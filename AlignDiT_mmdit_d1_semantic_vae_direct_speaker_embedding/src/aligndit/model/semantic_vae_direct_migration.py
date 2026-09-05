"""Strict S2c EMA migration for the Semantic-VAE Direct-D1 experiment."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


EXPECTED_SOURCE_KEYS = 313
EXPECTED_TARGET_KEYS = 559
EXPECTED_LOADED_KEYS = 303
EXPECTED_IGNORED_SOURCE_KEYS = 10
EXPECTED_NEW_TARGET_KEYS = 256
EMA_BOOKKEEPING_KEYS = frozenset({"initted", "step"})
S2C_IGNORED_PROJECTOR_KEYS = frozenset(
    {f"transformer.projectors.0.model.{layer}.{suffix}" for layer in (0, 1, 3, 4, 6) for suffix in ("bias", "weight")}
)


@dataclass(frozen=True)
class DirectD1MigrationReport:
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
    loaded_numel: int
    target_numel: int

    @property
    def loaded_fraction(self) -> float:
        return self.loaded_numel / self.target_numel

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["loaded_fraction"] = self.loaded_fraction
        return result


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_parent_artifacts(
    checkpoint_path: str | Path,
    contract_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_size: int,
    expected_contract_sha256: str,
) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path).expanduser().absolute()
    contract = Path(contract_path).expanduser().absolute()
    for path, label in ((checkpoint, "S2c checkpoint"), (contract, "S2c training contract")):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} must be a regular file: {path}")
    checkpoint = checkpoint.resolve(strict=True)
    contract = contract.resolve(strict=True)
    identity = _file_identity(checkpoint)
    if identity[2] != expected_checkpoint_size:
        raise RuntimeError(f"S2c checkpoint size mismatch: expected={expected_checkpoint_size}, got={identity[2]}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            f"S2c checkpoint SHA256 mismatch: expected={expected_checkpoint_sha256}, got={checkpoint_sha256}"
        )
    if _file_identity(checkpoint) != identity:
        raise RuntimeError("S2c checkpoint changed while its SHA256 was being verified")
    contract_sha256 = sha256_file(contract)
    if contract_sha256 != expected_contract_sha256:
        raise RuntimeError(f"S2c contract SHA256 mismatch: expected={expected_contract_sha256}, got={contract_sha256}")
    return checkpoint, contract


def load_s2c_ema_state(
    checkpoint_path: str | Path,
    *,
    expected_parent_contract_sha256: str,
    expected_parent_update: int = 70_000,
) -> tuple[dict[str, torch.Tensor], int]:
    path = Path(checkpoint_path).resolve(strict=True)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"S2c checkpoint must contain a mapping: {path}")
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise RuntimeError("S2c checkpoint_schema_version must be 1")
    if checkpoint.get("training_contract_sha256") != expected_parent_contract_sha256:
        raise RuntimeError(
            "S2c embedded contract SHA256 mismatch: "
            f"expected={expected_parent_contract_sha256}, got={checkpoint.get('training_contract_sha256')!r}"
        )
    if checkpoint.get("warmstart_stage") != "s2c" or checkpoint.get("update") != expected_parent_update:
        raise RuntimeError(
            "S2c parent identity mismatch: "
            f"stage={checkpoint.get('warmstart_stage')!r}, update={checkpoint.get('update')!r}"
        )

    ema_state = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema_state, dict):
        raise TypeError("S2c checkpoint has no EMA state mapping")
    if set(EMA_BOOKKEEPING_KEYS) - set(ema_state):
        raise RuntimeError("S2c EMA state is missing initted/step bookkeeping")
    initted = ema_state["initted"]
    step = ema_state["step"]
    if isinstance(initted, torch.Tensor):
        initted = initted.item()
    if isinstance(step, torch.Tensor):
        step = step.item()
    if initted is not True or step != expected_parent_update:
        raise RuntimeError(f"S2c EMA bookkeeping mismatch: initted={initted!r}, step={step!r}")

    source_state: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for key, value in ema_state.items():
        if key in EMA_BOOKKEEPING_KEYS:
            continue
        if not key.startswith("ema_model."):
            unexpected.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"S2c EMA value for {key!r} is not a tensor")
        source_state[key.removeprefix("ema_model.")] = value
    if unexpected:
        raise RuntimeError(f"Unexpected S2c EMA keys outside ema_model.*: {sorted(unexpected)}")
    if len(source_state) != EXPECTED_SOURCE_KEYS:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_KEYS} S2c EMA tensors, got {len(source_state)}")
    return source_state, int(step)


def migrate_s2c_ema_into_model(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    *,
    parent_path: str | Path,
    parent_sha256: str,
    parent_size: int,
    parent_contract_sha256: str,
    parent_ema_step: int,
) -> DirectD1MigrationReport:
    transformer = model.transformer
    if (
        transformer.n_mm_layers != 6
        or transformer.n_text_layers != 6
        or tuple(transformer.layer_indices_ctc) != (5, 11)
        or tuple(transformer.ctc_sampling_ratios) != (1, 1)
        or transformer.audio_video_ratio != 1
        or transformer.video_rope_scaled
        or transformer.prompt_isolated_ca
        or getattr(transformer, "text_attention_mode", "audio_only") != "audio_only"
    ):
        raise RuntimeError("S2c migration requires the D1 6-MM/12-audio architecture with 40-Hz CTC after blocks 6/12")
    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    common_keys = sorted(source_keys & target_keys)
    ignored_source = sorted(source_keys - target_keys)
    new_target = sorted(target_keys - source_keys)
    shape_mismatches = sorted(
        key for key in common_keys if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    )
    # Speaker conditioning adds exactly one zero-initialized tensor; the
    # inherited non-speaker D1 migration contract remains unchanged.
    speaker_key = "transformer.speaker_proj.weight"
    has_speaker = getattr(transformer, "speaker_proj", None) is not None
    if has_speaker:
        speaker = target_state.get(speaker_key)
        if (
            speaker is None
            or speaker_key not in new_target
            or tuple(speaker.shape) != (768, 192)
            or torch.count_nonzero(speaker).item() != 0
            or transformer.speaker_proj.bias is not None
            or transformer.speaker_condition_start_layer != 6
        ):
            raise RuntimeError(
                "S2c speaker migration requires a new, zero-initialized bias-free Linear(192, 768) "
                "conditioning the D1 audio-only blocks 6..17"
            )
    actual_counts = (
        len(source_state),
        len(target_state),
        len(common_keys) - len(shape_mismatches),
        len(ignored_source),
        len(new_target),
    )
    expected_counts = (
        EXPECTED_SOURCE_KEYS,
        EXPECTED_TARGET_KEYS + int(has_speaker),
        EXPECTED_LOADED_KEYS,
        EXPECTED_IGNORED_SOURCE_KEYS,
        EXPECTED_NEW_TARGET_KEYS + int(has_speaker),
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"S2c-to-Direct-D1 migration count mismatch: expected={expected_counts}, actual={actual_counts}"
        )
    if set(ignored_source) != S2C_IGNORED_PROJECTOR_KEYS:
        raise RuntimeError(
            f"Unexpected S2c-only keys: expected={sorted(S2C_IGNORED_PROJECTOR_KEYS)}, actual={ignored_source}"
        )
    if shape_mismatches:
        raise RuntimeError(f"S2c parent contains shape mismatches: {shape_mismatches}")

    merged_state = dict(target_state)
    loaded_numel = 0
    for key in common_keys:
        target = target_state[key]
        merged_state[key] = source_state[key].to(device=target.device, dtype=target.dtype)
        loaded_numel += target.numel()
    model.load_state_dict(merged_state, strict=True)
    return DirectD1MigrationReport(
        parent_path=str(Path(parent_path).resolve(strict=True)),
        parent_sha256=parent_sha256,
        parent_size=parent_size,
        parent_contract_sha256=parent_contract_sha256,
        parent_stage="s2c",
        parent_update=70_000,
        parent_ema_step=parent_ema_step,
        source_key_count=len(source_state),
        target_key_count=len(target_state),
        loaded_key_count=len(common_keys),
        ignored_source_keys=tuple(ignored_source),
        new_target_keys=tuple(new_target),
        loaded_numel=loaded_numel,
        target_numel=sum(value.numel() for value in target_state.values()),
    )
