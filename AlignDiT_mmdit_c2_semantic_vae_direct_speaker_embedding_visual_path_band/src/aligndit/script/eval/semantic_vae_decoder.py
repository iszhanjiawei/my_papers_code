"""Strict loader for the Semantic-VAE decoder bound to a latent cache."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


SAMPLE_RATE = 16_000
HOP_LENGTH = 400
LATENT_DIM = 64


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def load_semantic_vae_decoder(
    *, repo: Path, checkpoint_root: Path, cache_spec: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only the pinned 1000k EMA decoder and verify every cache binding."""

    repo = repo.resolve(strict=True)
    checkpoint_root = checkpoint_root.resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dac.model.dac import DAC

    metainfo_path = checkpoint_root / "metainfo.json"
    checkpoint_path = checkpoint_root / "dac/ema_state_dict.pth"
    metainfo = read_json_object(metainfo_path)
    config = dict(metainfo.get("DAC", {}))
    bigvgan_config = Path(config["bigvgan_conf"])
    if not bigvgan_config.is_absolute():
        bigvgan_config = repo / bigvgan_config
    config["bigvgan_conf"] = str(bigvgan_config.resolve(strict=True))

    checkpoint_contract = cache_spec.get("checkpoint", {})
    source_contract = cache_spec.get("semantic_vae_source", {})
    expected_hashes = {
        metainfo_path: checkpoint_contract.get("metainfo_sha256"),
        checkpoint_path: checkpoint_contract.get("ema_sha256"),
        bigvgan_config: source_contract.get("bigvgan_config_sha256"),
    }
    for path, expected_sha256 in expected_hashes.items():
        if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Semantic-VAE decoder resource differs from latent cache contract: {path}")

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != source_contract.get("commit") or status:
        raise RuntimeError(
            f"Semantic-VAE source differs from latent cache contract: commit={commit}, status={status!r}"
        )

    model = DAC(**config)
    del model.projectors
    target_state = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict) or not bool(checkpoint.get("initted")):
        raise RuntimeError("Semantic-VAE EMA checkpoint is not an initialized mapping")
    ema_step = checkpoint.get("step")
    if isinstance(ema_step, torch.Tensor):
        ema_step = ema_step.item()
    if not isinstance(ema_step, int) or isinstance(ema_step, bool):
        raise TypeError(f"Invalid Semantic-VAE EMA step: {ema_step!r}")
    if ema_step != checkpoint_contract.get("ema_step"):
        raise RuntimeError(f"Unexpected Semantic-VAE EMA step: {ema_step!r}")

    selected: dict[str, torch.Tensor] = {}
    ignored_decoder_projection = 0
    for raw_key, value in checkpoint.items():
        key = raw_key.removeprefix("ema_model.")
        if key in {"initted", "step"} or key.startswith("projectors."):
            continue
        if key not in target_state:
            if key.startswith("decoder_proj."):
                ignored_decoder_projection += 1
                continue
            raise KeyError(f"Unexpected Semantic-VAE EMA key: {raw_key}")
        if target_state[key].shape != value.shape or target_state[key].dtype != value.dtype:
            raise ValueError(f"Semantic-VAE EMA tensor mismatch for {key}")
        selected[key] = value
    missing = sorted(set(target_state) - set(selected))
    if missing:
        raise RuntimeError(f"Semantic-VAE EMA is missing current model keys: {missing}")
    model.load_state_dict(selected, strict=True)
    if int(model.sample_rate) != SAMPLE_RATE or int(model.hop_length) != HOP_LENGTH or int(model.vae_dim) != LATENT_DIM:
        raise RuntimeError(
            f"Unexpected Semantic-VAE geometry: sample_rate={model.sample_rate}, "
            f"hop={model.hop_length}, vae_dim={model.vae_dim}"
        )
    decoder = model.decoder.eval().requires_grad_(False).to(device)
    del model, checkpoint, selected, target_state
    return decoder, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ema_step": ema_step,
        "git_commit": commit,
        "ignored_legacy_decoder_projection_keys": ignored_decoder_projection,
        "metainfo_sha256": sha256_file(metainfo_path),
        "repo": str(repo),
    }
