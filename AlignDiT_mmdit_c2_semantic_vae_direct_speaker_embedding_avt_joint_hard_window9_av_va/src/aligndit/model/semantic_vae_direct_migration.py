"""Strict S2c EMA migration for the Semantic-VAE Direct-C2 experiment."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


EXPECTED_SOURCE_KEYS = 313
EXPECTED_TARGET_KEYS = 703
EXPECTED_LOADED_KEYS = 303
EXPECTED_IGNORED_SOURCE_KEYS = 10
EXPECTED_NEW_TARGET_KEYS = 400
EXPECTED_AVT_TARGET_KEYS = 801
EXPECTED_AVT_NEW_TARGET_KEYS = 498
EMA_BOOKKEEPING_KEYS = frozenset({"initted", "step"})
S2C_IGNORED_PROJECTOR_KEYS = frozenset(
    {f"transformer.projectors.0.model.{layer}.{suffix}" for layer in (0, 1, 3, 4, 6) for suffix in ("bias", "weight")}
)


def _expected_new_target_keys(*, text_joint_attention: bool, has_speaker: bool) -> set[str]:
    """Exact additions to the fixed 18-layer S2c parent, for either C2 topology.

    Keep the original 12 MM / 6 audio staging and all shared condition encoders.
    AVT replaces each block's eight cross-attention tensors with sixteen text
    stream tensors and adds only the two input-projection tensors.
    """
    keys = {
        f"transformer.projectors_ctc.{projector}.model.{layer}.{suffix}"
        for projector in range(2)
        for layer in (0, 1, 3, 4, 6)
        for suffix in ("weight", "bias")
    }
    keys.add("transformer.text_embed.text_embed.weight")
    for layer in range(4):
        prefix = f"transformer.text_embed.text_blocks.{layer}."
        keys.update(prefix + f"{name}.{suffix}" for name in ("dwconv", "norm", "pwconv1", "pwconv2")
                    for suffix in ("weight", "bias"))
        keys.update(prefix + "grn." + suffix for suffix in ("gamma", "beta"))

    video_prefix = "transformer.video_embed."
    keys.add(video_prefix + "vid_null_emb")
    keys.update(video_prefix + f"{name}.{suffix}"
                for name in ("proj", "conv_pos_embed.conv1d.0", "conv_pos_embed.conv1d.2",
                             "vid_conformer.after_norm", "vid_conformer.embed.out.0", "vid_conformer.embed.out.1")
                for suffix in ("weight", "bias"))
    for layer in range(2):
        prefix = video_prefix + f"vid_conformer.encoders.{layer}."
        keys.update(prefix + f"{name}.{suffix}"
                    for name in ("feed_forward.w_1", "feed_forward.w_2", "norm_ff", "norm_mha",
                                 "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v",
                                 "self_attn.linear_out")
                    for suffix in ("weight", "bias"))
        keys.update(prefix + "self_attn." + name
                    for name in ("linear_pos.weight", "pos_bias_u", "pos_bias_v"))

    for layer in range(12):
        prefix = f"transformer.transformer_blocks.{layer}."
        for stream in (("v", "t") if text_joint_attention else ("v",)):
            keys.update(prefix + f"{stream}_{name}.{suffix}"
                        for name in ("attn_norm.linear", "attn.to_q", "attn.to_k", "attn.to_v",
                                     "attn.to_out.0", "ff.ff.0.0", "ff.ff.2")
                        for suffix in ("weight", "bias"))
            keys.update(prefix + f"{stream}_attn.{name}.weight" for name in ("q_norm", "k_norm"))
        if not text_joint_attention:
            keys.update(prefix + name for name in (
                "cross_attn.q_proj_weight", "cross_attn.k_proj_weight", "cross_attn.v_proj_weight",
                "cross_attn.in_proj_bias", "cross_attn.out_proj.weight", "cross_attn.out_proj.bias",
                "cross_attn_ada.weight", "cross_attn_ada.bias",
            ))
    if text_joint_attention:
        keys.update({"transformer.text_joint_proj.weight", "transformer.text_joint_proj.bias"})
    if has_speaker:
        keys.add("transformer.speaker_proj.weight")
    return keys


@dataclass(frozen=True)
class DirectC2MigrationReport:
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
) -> DirectC2MigrationReport:
    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    common_keys = sorted(source_keys & target_keys)
    ignored_source = sorted(source_keys - target_keys)
    new_target = sorted(target_keys - source_keys)
    shape_mismatches = sorted(
        key for key in common_keys if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    )
    # The speaker snapshot adds exactly one zero-initialized tensor. Keep the
    # original strict S2c contract for all other keys and for non-speaker configs.
    speaker_key = "transformer.speaker_proj.weight"
    has_speaker = getattr(model.transformer, "speaker_proj", None) is not None
    if has_speaker:
        speaker = target_state.get(speaker_key)
        if (
            speaker is None
            or speaker_key not in new_target
            or tuple(speaker.shape) != (768, 192)
            or torch.count_nonzero(speaker).item() != 0
        ):
            raise RuntimeError("S2c speaker migration requires a new, zero-initialized Linear(192, 768) weight")
    text_joint_attention = bool(getattr(model.transformer, "text_joint_attention", False))
    expected_new_keys = _expected_new_target_keys(
        text_joint_attention=text_joint_attention, has_speaker=has_speaker
    )
    if set(new_target) != expected_new_keys:
        raise RuntimeError(
            "Unexpected S2c-to-C2 new tensor keys: "
            f"missing={sorted(expected_new_keys - set(new_target))}, "
            f"unexpected={sorted(set(new_target) - expected_new_keys)}"
        )
    # Every new modality stream retains the source audio stream's dimensions
    # and point-wise FFN. Reject shape changes even if the key counts match.
    for key in expected_new_keys:
        if not key.startswith("transformer.transformer_blocks."):
            continue
        _, _, layer_index, stream_name = key.split(".", 3)
        if not stream_name.startswith(("v_", "t_")):
            continue
        audio_key = f"transformer.transformer_blocks.{layer_index}.{stream_name[2:]}"
        if audio_key not in source_state or target_state[key].shape != source_state[audio_key].shape:
            raise RuntimeError(f"New modality tensor does not match its S2c audio counterpart: {key}")
    if text_joint_attention:
        for key, shape in (
            ("transformer.text_joint_proj.weight", (768, 512)),
            ("transformer.text_joint_proj.bias", (768,)),
        ):
            if tuple(target_state[key].shape) != shape:
                raise RuntimeError(f"AVT migration requires {key} to have shape {shape}")
    expected_target_count = EXPECTED_AVT_TARGET_KEYS if text_joint_attention else EXPECTED_TARGET_KEYS
    expected_new_count = EXPECTED_AVT_NEW_TARGET_KEYS if text_joint_attention else EXPECTED_NEW_TARGET_KEYS
    actual_counts = (
        len(source_state),
        len(target_state),
        len(common_keys) - len(shape_mismatches),
        len(ignored_source),
        len(new_target),
    )
    expected_counts = (
        EXPECTED_SOURCE_KEYS,
        expected_target_count + int(has_speaker),
        EXPECTED_LOADED_KEYS,
        EXPECTED_IGNORED_SOURCE_KEYS,
        expected_new_count + int(has_speaker),
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"S2c-to-Direct-C2 migration count mismatch: expected={expected_counts}, actual={actual_counts}"
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
    return DirectC2MigrationReport(
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
