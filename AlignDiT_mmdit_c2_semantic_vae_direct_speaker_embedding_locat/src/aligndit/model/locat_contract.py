"""Semantic guards for isolated LocAt checkpoints, independent of tensor shapes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOCAT_CONTRACT_SCHEMA = 1
CONTRACT_FILENAME = "speaker_training_contract.json"
LOCAT_CONFIG_KEYS = frozenset({
    "locat_enabled", "locat_av_enabled", "locat_va_enabled",
    "locat_audio_fps", "locat_video_fps", "locat_sigma_min_seconds",
    "locat_sigma_max_seconds", "locat_sigma_init_seconds", "locat_alpha_init",
    "locat_bias_mode", "locat_av_layers", "locat_va_layers",
})


def make_locat_contract(backbone) -> dict[str, Any] | None:
    """Record effective model semantics, including the omitted terminal VA layer."""
    config = getattr(backbone, "locat_config", {})
    if not config.get("locat_enabled", False):
        return None
    if set(config) != LOCAT_CONFIG_KEYS:
        raise RuntimeError(f"Unexpected LocAt model contract keys: {sorted(config)}")
    count = sum(
        parameter.numel() for name, parameter in backbone.named_parameters()
        if ".locat_av." in name or ".locat_va." in name
    )
    if not count:
        raise RuntimeError("Enabled LocAt model has no active predictor parameters")
    return {
        "schema_version": LOCAT_CONTRACT_SCHEMA,
        "config": dict(config),
        "parameter_count": count,
        "formula": "alpha_i * exp(-(t_key_j - t_query_i)**2 / (2*sigma_i**2))",
        "uniform_control": "alpha_i on every cross-modal key when locat_bias_mode=uniform",
        "predictor": "layer-specific query after QK normalization and before RoPE; weights shared over heads",
        "scope": "configured AV/VA blocks of joint attention; no AA/VV/text change and no PRR",
        "mass_preservation": False,
        "extra_loss": False,
    }


def read_training_contract(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_dir) / CONTRACT_FILENAME
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"Training contract is not a file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Training contract must be a JSON object: {path}")
    return value


def validate_locat_checkpoint_contract(
    backbone, checkpoint_dir: str | Path, *, require_existing: bool = True,
) -> None:
    """Reject missing or incompatible semantics even when state keys would match.

    Training may create a contract in an empty output directory, but must never
    retroactively assign a new contract to unguarded existing weight files.
    """
    requested = make_locat_contract(backbone)
    previous = read_training_contract(checkpoint_dir)
    recorded = previous.get("locat") if previous is not None else None
    if requested is None and recorded is None:
        return
    directory = Path(checkpoint_dir)
    has_weights = directory.is_dir() and any(
        entry.is_file() and entry.suffix in {".pt", ".safetensors"}
        for entry in directory.iterdir()
    )
    if previous is None and not require_existing and not has_weights:
        return
    if requested is None or recorded is None:
        raise RuntimeError(
            "LocAt checkpoint requires a matching LocAt training contract beside the weights; "
            f"refusing baseline/LocAt mixing or unguarded resume in {directory}"
        )
    if not isinstance(recorded, dict) or recorded != requested:
        raise RuntimeError(f"LocAt checkpoint/training configuration mismatch in {directory}")
