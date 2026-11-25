"""Initialize paired text/audio pretraining from the pinned S2c EMA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)


PRETRAIN_STAGE = "paired_libritts_text_audio"


def contract_sha256(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _plain(config):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def initialize_from_s2c(model, cfg) -> dict:
    """Load the pinned S2c EMA audio tensors; leave newly added streams initialized."""
    cfg = _plain(cfg)
    arch = cfg["model"]["arch"]
    required = {"dim": 768, "depth": 18, "heads": 12, "pe_attn_head": 1,
                "n_mm_layers": 12, "n_text_layers": 12, "text_joint_attention": True,
                "text_audio_only": True, "paired_joint_order": "at_then_av",
                "speaker_dim": 192, "speaker_condition_start_layer": 12}
    for key, expected in required.items():
        if arch.get(key) != expected:
            raise ValueError(f"Initial S2c-to-AT run requires {key}={expected!r}, got {arch.get(key)!r}")
    ckpt = cfg["ckpts"]
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    is_main = not distributed or torch.distributed.get_rank() == 0
    errors = [None]
    if is_main:
        try:
            validate_parent_artifacts(
                ckpt["pretrained_path"], ckpt["parent_contract_path"],
                expected_checkpoint_sha256=ckpt["expected_parent_sha256"],
                expected_checkpoint_size=ckpt["expected_parent_size"],
                expected_contract_sha256=ckpt["expected_parent_contract_sha256"],
            )
            parent = json.loads(Path(ckpt["parent_contract_path"]).read_text())
            source_arch = parent["config"]["model"]["arch"]
            for key in ("dim", "depth", "heads", "pe_attn_head", "qk_norm"):
                if arch.get(key) != source_arch.get(key):
                    raise ValueError(f"S2c initialization architecture mismatch: {key}")
            source_rep = parent["config"]["model"]["audio_representation"]
            for key in ("channels", "frame_rate", "sample_rate", "hop_length"):
                if cfg["model"]["audio_representation"].get(key) != source_rep.get(key):
                    raise ValueError(f"S2c initialization representation mismatch: {key}")
        except Exception as error:  # noqa: BLE001 - forward failure to every DDP rank
            errors[0] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError(errors[0])
    source, ema_step = load_s2c_ema_state(
        ckpt["pretrained_path"],
        expected_parent_contract_sha256=ckpt["expected_parent_contract_sha256"],
        expected_parent_update=ckpt.get("expected_parent_update", 70_000),
    )
    report = migrate_s2c_ema_into_model(
        model, source, parent_path=ckpt["pretrained_path"],
        parent_sha256=ckpt["expected_parent_sha256"], parent_size=ckpt["expected_parent_size"],
        parent_contract_sha256=ckpt["expected_parent_contract_sha256"], parent_ema_step=ema_step,
    ).to_dict()
    report["initialization"] = "S2c EMA audio weights; fresh text/speaker streams and optimizer"
    return report
