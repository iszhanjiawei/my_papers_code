"""Audited S2c initialization and LibriTTS AT-to-AVT EMA transfer.

Changing a modality boundary is an explicit migration, never a training resume.
Only removal of later text/video branches is permitted by the optional migration;
all audio layers, CTC taps, speaker placement and representation contracts stay fixed.
"""

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
        except Exception as error:
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


_ARCH_DEFAULTS = {"dim_head": 64, "text_mask_padding": True, "qk_norm": None,
                  "text_embedding_average_upsampling": False, "attn_mask_enabled": False,
                  "use_conformer": True,
                  "conv_layers": 0, "pe_attn_head": None, "speaker_dim": None,
                  "speaker_condition_start_layer": None, "video_dim": 1024,
                  "video_rope_scaled": True, "audio_video_ratio": 1,
                  "normalize_text_context": False, "prompt_isolated_ca": False}
_ARCH_KEYS = ("dim", "depth", "heads", "dim_head", "ff_mult", "text_dim", "text_mask_padding",
              "text_embedding_average_upsampling", "attn_mask_enabled", "use_conformer",
              "qk_norm", "conv_layers", "pe_attn_head", "layer_indices_ctc", "ctc_sampling_ratios",
              "speaker_dim", "speaker_condition_start_layer", "video_dim", "video_rope_scaled",
              "audio_video_ratio", "text_joint_attention", "normalize_text_context", "prompt_isolated_ca")
_DATA_KEYS = ("normalization_sha256", "vocab_sha256", "codec_sha256", "speaker_checkpoint_sha256")


def validate_transfer_contract(source: dict, target: dict, *, allow_layer_repartition=False) -> tuple[int, int]:
    """Validate semantics which state_dict shapes alone cannot establish."""
    if type(allow_layer_repartition) is not bool:
        raise TypeError("allow_layer_repartition must be bool")
    if source.get("contract_schema_version") != 1 or source.get("stage") != "libritts_text_audio":
        raise ValueError("Unsupported embedded AT training contract schema/stage")
    if source["model"].get("backbone") != target["model"].get("backbone"):
        raise ValueError("AT transfer backbone mismatch")
    src, dst = source["model"]["arch"], target["model"]["arch"]
    if src.get("text_audio_only") is not True:
        raise ValueError("The source checkpoint is not an explicit text/audio-only model")
    for key in _ARCH_KEYS:
        if src.get(key, _ARCH_DEFAULTS.get(key)) != dst.get(key, _ARCH_DEFAULTS.get(key)):
            raise ValueError(f"AT transfer architecture mismatch: {key}")
    src_mm, dst_mm = src["n_mm_layers"], dst["n_mm_layers"]
    if any(type(value) is not int for value in (src_mm, dst_mm, src["depth"])):
        raise ValueError("Layer counts must be integers")
    if src.get("n_text_layers", src_mm) != src_mm or dst.get("n_text_layers", dst_mm) != dst_mm:
        raise ValueError("AT/AVT joint transfer requires n_text_layers == n_mm_layers")
    if src_mm != dst_mm and not allow_layer_repartition:
        raise ValueError("Changing the AT/AVT boundary requires allow_layer_repartition=True")
    if not 0 < dst_mm <= src_mm <= src["depth"]:
        raise ValueError("Only retaining or reducing the pretrained joint layer count is supported")
    source_rep, target_rep = source["model"]["audio_representation"], target["model"]["audio_representation"]
    for key in ("channels", "frame_rate", "sample_rate", "hop_length"):
        if source_rep.get(key) != target_rep.get(key):
            raise ValueError(f"AT transfer audio representation mismatch: {key}")
    for key in _DATA_KEYS:
        a, b = source["data"].get(key), target["data"].get(key)
        if not isinstance(a, str) or len(a) != 64 or any(c not in "0123456789abcdef" for c in a) or a != b:
            raise ValueError(f"AT transfer data contract mismatch: {key}")
    return src_mm, dst_mm


def _validate_instantiated_model(model, contract):
    """Check runtime properties invisible to tensor shapes, before any mutation."""
    backbone = model.transformer
    arch = contract["model"]["arch"]
    for key in ("dim", "depth", "n_mm_layers", "n_text_layers", "text_joint_attention",
                "speaker_dim", "speaker_condition_start_layer", "audio_video_ratio",
                "video_rope_scaled", "normalize_text_context", "prompt_isolated_ca"):
        expected = arch.get(key, _ARCH_DEFAULTS.get(key))
        if key == "n_text_layers" and expected is None:
            expected = arch["n_mm_layers"]
        if getattr(backbone, key) != expected:
            raise ValueError(f"Instantiated model differs from its transfer contract: {key}")
    if backbone.text_audio_only != arch.get("text_audio_only", False):
        raise ValueError("Instantiated model differs from its transfer contract: text_audio_only")
    if backbone.text_embed.mask_padding != arch.get("text_mask_padding", True):
        raise ValueError("Instantiated model differs from its transfer contract: text_mask_padding")
    if backbone.text_embed.average_upsampling != arch.get("text_embedding_average_upsampling", False):
        raise ValueError("Instantiated model differs from its transfer contract: text_embedding_average_upsampling")
    if backbone.video_embed.use_conformer != arch.get("use_conformer", True):
        raise ValueError("Instantiated model differs from its transfer contract: use_conformer")
    if list(backbone.layer_indices_ctc) != list(arch.get("layer_indices_ctc", [6, 12])):
        raise ValueError("Instantiated model differs from its transfer contract: layer_indices_ctc")
    if list(backbone.ctc_sampling_ratios) != list(arch.get("ctc_sampling_ratios", [2, 1])):
        raise ValueError("Instantiated model differs from its transfer contract: ctc_sampling_ratios")
    for block in backbone.transformer_blocks:
        if block.attn.heads != arch["heads"] or block.attn.inner_dim // block.attn.heads != arch.get("dim_head", 64):
            raise ValueError("Instantiated model attention dimensions differ from its transfer contract")
        pe = arch.get("pe_attn_head")
        if block.attn.processor.pe_attn_head != pe or (hasattr(block, "pe_attn_head") and block.pe_attn_head != pe):
            raise ValueError("Instantiated model differs from its transfer contract: pe_attn_head")
        expected_mask = arch.get("attn_mask_enabled", False)
        if block.attn.processor.attn_mask_enabled != expected_mask or (
            hasattr(block, "attn_mask_enabled") and block.attn_mask_enabled != expected_mask
        ):
            raise ValueError("Instantiated model differs from its transfer contract: attn_mask_enabled")


def _removed_stream_key(key: str, first: int, last: int) -> bool:
    parts = key.split(".")
    return (len(parts) >= 4 and parts[:2] == ["transformer", "transformer_blocks"]
            and parts[2].isdigit() and first <= int(parts[2]) < last
            and parts[3].startswith(("v_", "t_")))


def load_text_audio_pretrained(model, path, expected_contract: dict, *, allow_layer_repartition=False) -> dict:
    """Initialize a new AT/AVT run from EMA; never restore optimizer/update state."""
    path = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("checkpoint_schema_version") != 1 or checkpoint.get("stage") != "libritts_text_audio":
        raise ValueError("Not a supported LibriTTS text/audio checkpoint")
    source_contract = checkpoint["training_contract"]
    if contract_sha256(source_contract) != checkpoint.get("training_contract_sha256"):
        raise ValueError("Embedded AT training contract hash mismatch")
    expected_contract = _plain(expected_contract)
    source_mm, target_mm = validate_transfer_contract(
        source_contract, expected_contract, allow_layer_repartition=allow_layer_repartition,
    )
    _validate_instantiated_model(model, expected_contract)
    ema = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema, dict) or "initted" not in ema or not bool(ema["initted"]):
        raise ValueError("AT checkpoint has no initialized EMA")
    if int(checkpoint.get("update", 0)) < 1 or "step" not in ema or int(ema["step"]) < 1:
        raise ValueError("AT checkpoint has no completed optimizer/EMA updates")
    if int(ema["step"]) != int(checkpoint["update"]):
        raise ValueError("AT checkpoint EMA step differs from its completed optimizer update")
    if any(key not in ("initted", "step") and not key.startswith("ema_model.") for key in ema):
        raise ValueError("Unexpected EMA state keys")
    source = {key.removeprefix("ema_model."): value for key, value in ema.items() if key.startswith("ema_model.")}
    target = model.state_dict()
    missing, extra = sorted(set(target) - set(source)), sorted(set(source) - set(target))
    if missing:
        raise ValueError(f"AT checkpoint is missing target tensors: {missing}")
    if any(not _removed_stream_key(key, target_mm, source_mm) for key in extra):
        raise ValueError(f"Unapproved tensors discarded by AT transfer: {extra}")
    mismatches = [key for key in target if not isinstance(source[key], torch.Tensor)
                  or source[key].shape != target[key].shape]
    if mismatches:
        raise ValueError(f"AT transfer tensor shape mismatch: {mismatches}")
    model.load_state_dict({key: source[key] for key in target}, strict=True)
    return {"initialization": "LibriTTS text/audio EMA; fresh downstream optimizer and update counter",
            "source_path": str(path), "source_update": checkpoint["update"],
            "source_contract_sha256": checkpoint["training_contract_sha256"],
            "source_joint_layers": source_mm, "target_joint_layers": target_mm,
            "loaded_tensors": len(target), "loaded_numel": sum(value.numel() for value in target.values()),
            "discarded_stream_tensors": extra,
            "visual_weights_pretrained": False,
            "source_ctc_lambda": source_contract["model"].get("ctc_lambda"),
            "function_preserving": False}
