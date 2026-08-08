"""Strict S3a/S3b training entrypoint for 40-Hz Semantic-VAE C2."""

from __future__ import annotations

import hashlib
import json
import os
import re
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

import accelerate
import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.dataset import SemanticVaeCelebVDubDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_c2_stage import (
    S3_STAGES,
    S3A_SOURCE_KIND,
    S3B_SOURCE_KIND,
    configure_s3_parameters,
    load_s3_parent_ema,
    sha256_file,
)
from aligndit.model.trainer_semantic_vae_c2 import (
    S3_STAGE_MAX_UPDATES,
    S3_STAGE_START_UPDATE,
    SemanticVaeC2Trainer,
)
from aligndit.model.trainer_semantic_vae_warmstart import (
    create_warmstart_accelerator,
    has_local_training_checkpoint,
)
from aligndit.script.misc.svae_cache_utils import atomic_write_json
from f5_tts.model.utils import get_tokenizer


PROJECT_ROOT = Path(str(files("aligndit").joinpath("../.."))).resolve()
os.chdir(PROJECT_ROOT)
POLICY_VERSION = "semantic-vae40-c2-s3-staged-v1"
EXPECTED_STAGE_WARMUP = {"s3a": 500, "s3b": 5_000}
EXPECTED_STAGE_LEARNING_RATES = {
    "s3a": {"multimodal_new": 5e-5},
    "s3b": {
        "multimodal_new": 5e-5,
        "interface": 2e-5,
        "audio_blocks_0_5": 5e-6,
        "audio_backbone_rest": 1e-5,
    },
}
EXPECTED_VALID_TRAIN_RECORDS = 79_508


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _line_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def _parent_metadata(stage_cfg) -> dict[str, Any]:
    parent_path = _regular_file(stage_cfg.parent_checkpoint, label="S3 parent checkpoint")
    parent_contract_path = _regular_file(stage_cfg.parent_contract_path, label="S3 parent contract")
    actual_sha256 = sha256_file(parent_path)
    actual_size = parent_path.stat().st_size
    actual_contract_sha256 = sha256_file(parent_contract_path)

    configured_sha256 = stage_cfg.get("expected_parent_sha256")
    if configured_sha256 is not None and configured_sha256 != actual_sha256:
        raise RuntimeError(f"S3 parent SHA256 mismatch: expected {configured_sha256}, got {actual_sha256}")
    configured_size = stage_cfg.get("expected_parent_size")
    if configured_size is not None and int(configured_size) != actual_size:
        raise RuntimeError(f"S3 parent size mismatch: expected {configured_size}, got {actual_size}")
    configured_contract_sha256 = stage_cfg.get("expected_parent_contract_sha256")
    if configured_contract_sha256 is not None and configured_contract_sha256 != actual_contract_sha256:
        raise RuntimeError(
            f"S3 parent contract SHA256 mismatch: expected {configured_contract_sha256}, got {actual_contract_sha256}"
        )

    return {
        "canonical_path": str(parent_path),
        "size": actual_size,
        "sha256": actual_sha256,
        "contract_path": str(parent_contract_path),
        "contract_sha256": actual_contract_sha256,
        "kind": str(stage_cfg.parent_kind),
        "expected_stage": str(stage_cfg.expected_parent_stage),
        "expected_update": int(stage_cfg.expected_parent_update),
    }


def _checkpoint_files(checkpoint_dir: Path) -> list[str]:
    if not checkpoint_dir.exists():
        return []
    return sorted(
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and (path.name == "model_last.pt" or re.fullmatch(r"model_[0-9]+\.pt", path.name))
    )


def _artifact_contract(model_cfg, dataset: SemanticVaeCelebVDubDataset) -> dict[str, Any]:
    paths = {
        "full_manifest": _regular_file(model_cfg.datasets.full_manifest_path, label="full CelebVDub manifest"),
        "train_ctc40_valid_manifest": _regular_file(
            model_cfg.datasets.manifest_path, label="CTC-valid CelebVDub train manifest"
        ),
        "ctc40_excluded_manifest": _regular_file(
            model_cfg.datasets.ctc_excluded_path, label="CTC-excluded CelebVDub train manifest"
        ),
        "ctc40_exclusion_report": _regular_file(
            model_cfg.datasets.ctc_exclusion_report_path, label="CTC exclusion report"
        ),
        "inventory_meta": _regular_file(model_cfg.datasets.inventory_meta_path, label="CelebVDub inventory metadata"),
        "latent_completion": _regular_file(
            model_cfg.datasets.latent_completion_path, label="CelebVDub latent completion marker"
        ),
        "latent_spec": _regular_file(dataset.latent_spec_path, label="CelebVDub latent immutable spec"),
        "video_40hz_completion": _regular_file(
            model_cfg.datasets.video_completion_path, label="CelebVDub 40-Hz video completion marker"
        ),
        "video_40hz_spec": _regular_file(dataset.video_spec_path, label="CelebVDub 40-Hz video immutable spec"),
        "normalization": _regular_file(model_cfg.datasets.normalization_path, label="LibriSpeech normalization"),
        "vocab": _regular_file(model_cfg.datasets.vocab_path, label="CelebVDub vocabulary"),
    }
    valid_count = _line_count(paths["train_ctc40_valid_manifest"])
    if valid_count != EXPECTED_VALID_TRAIN_RECORDS or len(dataset) != EXPECTED_VALID_TRAIN_RECORDS:
        raise RuntimeError(
            "CTC-valid CelebVDub cardinality mismatch: "
            f"expected={EXPECTED_VALID_TRAIN_RECORDS}, manifest={valid_count}, dataset={len(dataset)}"
        )
    expected_hashes = {
        "full_manifest": model_cfg.datasets.expected_full_manifest_sha256,
        "train_ctc40_valid_manifest": model_cfg.datasets.expected_train_manifest_sha256,
        "ctc40_excluded_manifest": model_cfg.datasets.expected_ctc_excluded_sha256,
        "vocab": model_cfg.datasets.expected_vocab_sha256,
    }
    for name, expected_sha256 in expected_hashes.items():
        actual_sha256 = sha256_file(paths[name])
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Immutable S3 artifact {name} SHA256 mismatch: expected={expected_sha256}, got={actual_sha256}"
            )
    excluded_count = _line_count(paths["ctc40_excluded_manifest"])
    if excluded_count != 105:
        raise RuntimeError(f"Expected 105 CTC-excluded CelebVDub records, got {excluded_count}")
    normalization_sha256 = sha256_file(paths["normalization"])
    if normalization_sha256 != model_cfg.datasets.expected_normalization_sha256:
        raise RuntimeError(
            "LibriSpeech normalization SHA256 mismatch: "
            f"expected={model_cfg.datasets.expected_normalization_sha256}, got={normalization_sha256}"
        )
    if (
        sha256_file(paths["latent_spec"]) != dataset.latent_spec_sha256
        or sha256_file(paths["video_40hz_spec"]) != dataset.video_spec_sha256
    ):
        raise RuntimeError("CelebVDub cache specs changed after dataset provenance validation")
    ctc_report = _read_json(paths["ctc40_exclusion_report"], label="CTC exclusion report")
    if (
        ctc_report.get("cache_schema_version") != 1
        or ctc_report.get("input_stride") != 1
        or ctc_report.get("train")
        != {"excluded": excluded_count, "total": valid_count + excluded_count, "valid": valid_count}
    ):
        raise RuntimeError("CelebVDub CTC exclusion report disagrees with the selected manifests")
    tokenizer_contract = ctc_report.get("tokenizer")
    if (
        not isinstance(tokenizer_contract, dict)
        or tokenizer_contract.get("sha256") != model_cfg.datasets.expected_vocab_sha256
        or tokenizer_contract.get("size") != 159
        or tokenizer_contract.get("blank_index") != 160
        or tokenizer_contract.get("output_classes") != 161
        or tokenizer_contract.get("unknown_id") != 0
    ):
        raise RuntimeError("CelebVDub CTC exclusion report vocabulary identity mismatch")
    inventory_meta = _read_json(paths["inventory_meta"], label="CelebVDub inventory metadata")
    manifest_entries = inventory_meta.get("manifests")
    if not isinstance(manifest_entries, dict):
        raise TypeError("CelebVDub inventory metadata has no manifests mapping")
    expected_manifest_entries = {
        "inventory.jsonl": (79_826, paths["full_manifest"]),
        "train_ctc40_valid.jsonl": (valid_count, paths["train_ctc40_valid_manifest"]),
        "ctc40_excluded.jsonl": (excluded_count, paths["ctc40_excluded_manifest"]),
    }
    for name, (expected_count, path) in expected_manifest_entries.items():
        entry = manifest_entries.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("count") != expected_count
            or entry.get("sha256") != sha256_file(path)
            or entry.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"CelebVDub inventory metadata entry mismatch for {name}")
    ctc_preflight = inventory_meta.get("ctc40_preflight")
    report_entry = ctc_preflight.get("report") if isinstance(ctc_preflight, dict) else None
    if (
        not isinstance(report_entry, dict)
        or report_entry.get("sha256") != sha256_file(paths["ctc40_exclusion_report"])
        or report_entry.get("size_bytes") != paths["ctc40_exclusion_report"].stat().st_size
    ):
        raise RuntimeError("CelebVDub inventory metadata does not authenticate the CTC exclusion report")

    latent_completion = _read_json(paths["latent_completion"], label="latent completion")
    if (
        latent_completion.get("cache_schema_version") != 1
        or latent_completion.get("feature") != "semantic_vae_posterior_sample_v1"
        or latent_completion.get("selection") != {"mode": "full"}
    ):
        raise RuntimeError("CelebVDub latent completion marker does not describe the authoritative full cache")
    consolidated_index = latent_completion.get("consolidated_index")
    if not isinstance(consolidated_index, dict):
        raise TypeError("CelebVDub latent completion marker has no consolidated index mapping")
    cache_root = Path(model_cfg.datasets.cache_root).resolve(strict=True)
    index_path = _regular_file(cache_root / consolidated_index.get("path", ""), label="latent consolidated index")
    if consolidated_index.get("sha256") != sha256_file(index_path):
        raise RuntimeError("CelebVDub latent consolidated index SHA256 does not match its completion marker")
    if int(consolidated_index.get("count", -1)) != int(latent_completion.get("count", -2)):
        raise RuntimeError("CelebVDub latent completion/index counts disagree")
    if int(consolidated_index.get("size_bytes", -1)) != index_path.stat().st_size:
        raise RuntimeError("CelebVDub latent consolidated index size does not match its completion marker")
    if latent_completion.get("manifest_sha256") != model_cfg.datasets.expected_full_manifest_sha256:
        raise RuntimeError("CelebVDub latent cache was not generated from the pinned full manifest")

    video_completion = _read_json(paths["video_40hz_completion"], label="40-Hz video completion")
    if (
        video_completion.get("cache_schema_version") != 1
        or video_completion.get("feature") != "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1"
        or video_completion.get("selection") != {"mode": "full"}
    ):
        raise RuntimeError("CelebVDub video completion marker does not describe the authoritative full cache")
    video_index = video_completion.get("consolidated_index")
    if not isinstance(video_index, dict):
        raise TypeError("CelebVDub video completion marker has no consolidated index mapping")
    video_index_path = _regular_file(cache_root / video_index.get("path", ""), label="video consolidated index")
    if video_index.get("sha256") != sha256_file(video_index_path):
        raise RuntimeError("CelebVDub video consolidated index SHA256 does not match its completion marker")
    if int(video_index.get("count", -1)) != int(video_completion.get("count", -2)):
        raise RuntimeError("CelebVDub video completion/index counts disagree")
    if int(video_index.get("size_bytes", -1)) != video_index_path.stat().st_size:
        raise RuntimeError("CelebVDub video consolidated index size does not match its completion marker")
    if video_completion.get("manifest_sha256") != model_cfg.datasets.expected_full_manifest_sha256:
        raise RuntimeError("CelebVDub video cache was not generated from the pinned full manifest")
    if int(video_completion.get("count", -1)) != int(latent_completion.get("count", -2)):
        raise RuntimeError("CelebVDub 40-Hz video and Semantic-VAE latent cache counts disagree")
    if int(video_completion.get("total_target_frames", -1)) != int(latent_completion.get("total_latent_frames", -2)):
        raise RuntimeError("CelebVDub 40-Hz video and Semantic-VAE latent totals disagree")

    return {
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
            for name, path in paths.items()
        },
        "latent_consolidated_index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "size": index_path.stat().st_size,
            "count": int(consolidated_index["count"]),
        },
        "video_40hz_consolidated_index": {
            "path": str(video_index_path),
            "sha256": sha256_file(video_index_path),
            "size": video_index_path.stat().st_size,
            "count": int(video_index["count"]),
        },
        "train_record_count": valid_count,
    }


def publish_training_contract(
    trainer: SemanticVaeC2Trainer,
    model_cfg,
    dataset: SemanticVaeCelebVDubDataset,
    *,
    parent_metadata: dict[str, Any],
    migration_report,
    parameter_report,
    optimizer_groups: list[dict[str, Any]],
) -> str:
    checkpoint_dir = Path(model_cfg.ckpts.save_dir)
    contract_path = checkpoint_dir / "training_contract.json"
    checkpoint_files = _checkpoint_files(checkpoint_dir)
    if checkpoint_files and not contract_path.is_file():
        raise RuntimeError(f"Refusing to resume S3 checkpoints {checkpoint_files} without {contract_path}")

    artifact_contract = _artifact_contract(model_cfg, dataset)
    source_paths = {
        "cfm_vt": PROJECT_ROOT / "src/aligndit/model/cfm_vt.py",
        "dataset": PROJECT_ROOT / "src/aligndit/model/dataset.py",
        "dit_vt_mm": PROJECT_ROOT / "src/aligndit/model/backbone/dit_vt_mm.py",
        "modules": PROJECT_ROOT / "src/aligndit/model/modules.py",
        "semantic_vae_c2_stage": PROJECT_ROOT / "src/aligndit/model/semantic_vae_c2_stage.py",
        "trainer_semantic_vae_c2": PROJECT_ROOT / "src/aligndit/model/trainer_semantic_vae_c2.py",
        "trainer_semantic_vae_warmstart": PROJECT_ROOT / "src/aligndit/model/trainer_semantic_vae_warmstart.py",
        "finetune_semantic_vae_c2": Path(__file__).resolve(),
        "stage_launcher": PROJECT_ROOT / "src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_4x4090.sh",
        "chain_launcher": PROJECT_ROOT / "src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_chain_4x4090.sh",
        "checkpoint_validator": PROJECT_ROOT / "src/aligndit/script/misc/validate_semantic_vae_c2_checkpoint.py",
        "svae_cache_utils": PROJECT_ROOT / "src/aligndit/script/misc/svae_cache_utils.py",
        "f5_cfm": PROJECT_ROOT / "src/f5_tts/model/cfm.py",
        "f5_dataset": PROJECT_ROOT / "src/f5_tts/model/dataset.py",
        "f5_modules": PROJECT_ROOT / "src/f5_tts/model/modules.py",
        "f5_trainer": PROJECT_ROOT / "src/f5_tts/model/trainer.py",
        "f5_utils": PROJECT_ROOT / "src/f5_tts/model/utils.py",
    }
    group_membership = [
        {
            "group_name": group["group_name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_names": list(group["parameter_names"]),
        }
        for group in optimizer_groups
    ]
    if trainer.is_main and migration_report is None:
        if not contract_path.is_file():
            raise RuntimeError("A fresh S3 stage has neither a migration report nor an existing contract")
        existing = _read_json(contract_path, label="existing S3 training contract")
        migration_contract = existing.get("migration_report")
        if not isinstance(migration_contract, dict):
            raise RuntimeError("Existing S3 contract has no migration report")
    elif trainer.is_main:
        migration_contract = migration_report.to_dict()
    else:
        migration_contract = None

    contract = {
        "schema_version": 1,
        "policy": {
            "version": POLICY_VERSION,
            "stage": trainer.stage,
            "stage_start_update": trainer.stage_start_update,
            "stage_max_updates": trainer.max_stage_updates,
            "final_cumulative_update": trainer.final_cumulative_update,
            "ema_weights_only_across_stages": True,
            "optimizer_scheduler_ema_reset_across_stages": True,
            "fresh_stage_ema_step": 0,
            "checkpoint_filenames_are_cumulative": True,
            "s3a_trains_only_new_multimodal_parameters": trainer.stage == "s3a",
        },
        "config": OmegaConf.to_container(model_cfg, resolve=True),
        "parent": parent_metadata,
        "migration_report": migration_contract,
        "parameter_policy": {
            "report": parameter_report.to_dict(),
            "group_membership": group_membership,
            "group_membership_sha256": _sha256_json(group_membership),
        },
        "artifacts": artifact_contract,
        "distributed_runtime": {
            "accelerate": accelerate.__version__,
            "cuda_allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "distributed_type": str(trainer.accelerator.distributed_type),
            "ema_pytorch": version("ema-pytorch"),
            "mixed_precision": trainer.accelerator.mixed_precision,
            "num_processes": trainer.accelerator.num_processes,
            "torch": torch.__version__,
        },
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    if trainer.is_main:
        result = atomic_write_json(contract_path, contract)
        print(
            f"S3 contract {'created' if result.created else 'verified'}: {result.path} sha256={result.sha256}",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
    return sha256_file(contract_path)


def _validate_fixed_stage_config(model_cfg, stage: str) -> None:
    max_updates = int(model_cfg.optim.max_updates)
    warmup = int(model_cfg.optim.num_warmup_updates)
    if max_updates != S3_STAGE_MAX_UPDATES[stage]:
        raise RuntimeError(
            f"Stage {stage} must train exactly {S3_STAGE_MAX_UPDATES[stage]} local updates, got {max_updates}"
        )
    if warmup != EXPECTED_STAGE_WARMUP[stage]:
        raise RuntimeError(f"Stage {stage} warmup must be {EXPECTED_STAGE_WARMUP[stage]}, got {warmup}")
    actual_learning_rates = {name: float(value) for name, value in model_cfg.optim.learning_rates.items()}
    if actual_learning_rates != EXPECTED_STAGE_LEARNING_RATES[stage]:
        raise RuntimeError(
            f"Stage {stage} learning-rate policy mismatch: "
            f"expected={EXPECTED_STAGE_LEARNING_RATES[stage]}, got={actual_learning_rates}"
        )
    expected_parent_kind = S3A_SOURCE_KIND if stage == "s3a" else S3B_SOURCE_KIND
    if model_cfg.stage.parent_kind != expected_parent_kind:
        raise RuntimeError(f"Stage {stage} must use parent_kind={expected_parent_kind!r}")
    if int(model_cfg.datasets.batch_size_per_gpu) != 3_600:
        raise RuntimeError("The 4x4090 S3 contract requires 3,600 latent frames/GPU")
    if int(model_cfg.datasets.max_samples) != 32:
        raise RuntimeError("The 4x4090 S3 contract requires max_samples=32")
    if int(model_cfg.datasets.expected_record_count) != EXPECTED_VALID_TRAIN_RECORDS:
        raise RuntimeError(f"Semantic-VAE C2 requires exactly {EXPECTED_VALID_TRAIN_RECORDS} training records")
    if int(model_cfg.optim.grad_accumulation_steps) != 1 or float(model_cfg.optim.weight_decay) != 0.01:
        raise RuntimeError("Semantic-VAE C2 requires grad_accumulation_steps=1 and weight_decay=0.01")
    if int(model_cfg.seed) != 666:
        raise RuntimeError("Semantic-VAE C2 S3 uses the fixed experiment seed 666")
    if int(model_cfg.model.arch.audio_video_ratio) != 1:
        raise RuntimeError("Semantic-VAE C2 requires an exact 40-Hz audio/video ratio of 1")
    if list(model_cfg.model.arch.ctc_sampling_ratios) != [1, 1]:
        raise RuntimeError("Semantic-VAE C2 CTC must remain at 40 Hz with ctc_sampling_ratios=[1, 1]")
    for required_true in (
        "text_mask_padding",
        "attn_mask_enabled",
        "always_use_attention_mask",
        "strict_audio_video_alignment",
        "mask_input_embeddings",
    ):
        if not bool(model_cfg.model.arch[required_true]):
            raise RuntimeError(f"Semantic-VAE C2 requires model.arch.{required_true}=true")
    fixed_arch = {
        "depth": 18,
        "n_mm_layers": 12,
        "n_text_layers": 12,
        "layer_indices_ctc": [6, 12],
    }
    for field, expected in fixed_arch.items():
        actual = model_cfg.model.arch[field]
        actual = list(actual) if isinstance(expected, list) else int(actual)
        if actual != expected:
            raise RuntimeError(f"Semantic-VAE C2 requires model.arch.{field}={expected}, got {actual}")
    if bool(model_cfg.model.arch.prompt_isolated_ca) or bool(model_cfg.model.arch.video_rope_scaled):
        raise RuntimeError("C2 S3 requires global text CA and a shared 40-Hz audio/video RoPE scale")
    if float(model_cfg.model.ctc_lambda) != 0.1:
        raise RuntimeError("Semantic-VAE C2 requires ctc_lambda=0.1")
    representation = model_cfg.model.audio_representation
    if (
        int(representation.channels) != 64
        or int(representation.frame_rate) != 40
        or int(representation.sample_rate) != 16_000
        or int(representation.hop_length) != 400
    ):
        raise RuntimeError("S3 requires the fixed 64-D/40-Hz/16-kHz Semantic-VAE representation contract")


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg) -> None:
    stage = str(model_cfg.stage.name)
    if stage not in S3_STAGES:
        raise ValueError(f"Unknown Semantic-VAE C2 stage {stage!r}; expected one of {S3_STAGES}")
    _validate_fixed_stage_config(model_cfg, stage)

    seed = int(model_cfg.seed)
    set_seed(seed)
    accelerator = create_warmstart_accelerator(
        grad_accumulation_steps=int(model_cfg.optim.grad_accumulation_steps),
        logger=model_cfg.ckpts.logger,
    )
    expected_world_size = os.environ.get("EXPECTED_WORLD_SIZE")
    if expected_world_size is None or not expected_world_size.isdecimal():
        raise RuntimeError("EXPECTED_WORLD_SIZE must be exported by the S3 launcher")
    if accelerator.num_processes != int(expected_world_size):
        raise RuntimeError(f"S3 world-size mismatch: expected {expected_world_size}, got {accelerator.num_processes}")

    vocab_path = _regular_file(model_cfg.datasets.vocab_path, label="CelebVDub vocabulary")
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    channels = int(model_cfg.model.audio_representation.channels)
    model = CFM_VT(
        transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=int(model_cfg.model.audio_representation.sample_rate),
            hop_length=int(model_cfg.model.audio_representation.hop_length),
        ),
        num_channels=channels,
        vocab_char_map=vocab_char_map,
        ctc_lambda=float(model_cfg.model.ctc_lambda),
        audio_video_ratio=int(model_cfg.model.arch.audio_video_ratio),
        strict_audio_video_alignment=bool(model_cfg.model.arch.strict_audio_video_alignment),
    )

    parent = _parent_metadata(model_cfg.stage)
    local_resume = has_local_training_checkpoint(model_cfg.ckpts.save_dir)
    migration_report = None
    if not local_resume:
        migration_report = load_s3_parent_ema(
            model,
            parent["canonical_path"],
            parent_kind=parent["kind"],
            expected_parent_stage=parent["expected_stage"],
            expected_parent_update=parent["expected_update"],
            expected_parent_contract_sha256=parent["contract_sha256"],
            expected_parent_sha256=parent["sha256"],
            expected_parent_size=parent["size"],
        )
    model.to(accelerator.device)
    accelerator.wait_for_everyone()

    optimizer_groups, parameter_report = configure_s3_parameters(
        model,
        stage=stage,
        learning_rates={name: float(value) for name, value in model_cfg.optim.learning_rates.items()},
        weight_decay=float(model_cfg.optim.weight_decay),
    )
    trainer = SemanticVaeC2Trainer(
        model,
        accelerator=accelerator,
        optimizer_groups=optimizer_groups,
        stage=stage,
        num_warmup_updates=int(model_cfg.optim.num_warmup_updates),
        save_per_updates=int(model_cfg.ckpts.save_per_updates),
        keep_last_n_checkpoints=int(model_cfg.ckpts.keep_last_n_checkpoints),
        last_per_updates=int(model_cfg.ckpts.last_per_updates),
        checkpoint_path=model_cfg.ckpts.save_dir,
        batch_size_per_gpu=int(model_cfg.datasets.batch_size_per_gpu),
        batch_size_type=str(model_cfg.datasets.batch_size_type),
        max_samples=int(model_cfg.datasets.max_samples),
        grad_accumulation_steps=int(model_cfg.optim.grad_accumulation_steps),
        max_grad_norm=float(model_cfg.optim.max_grad_norm),
        logger=model_cfg.ckpts.logger,
        run_name=f"{model_cfg.model.name}_{stage}_{model_cfg.datasets.name}",
        ema_kwargs=OmegaConf.to_container(model_cfg.ema, resolve=True),
    )
    if not local_resume:
        trainer.reset_ema_from_online()

    dataset = SemanticVaeCelebVDubDataset(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
        vocab_path=model_cfg.datasets.vocab_path,
        expected_normalization_sha256=model_cfg.datasets.expected_normalization_sha256,
        expected_record_count=int(model_cfg.datasets.expected_record_count),
    )
    contract_sha256 = publish_training_contract(
        trainer,
        model_cfg,
        dataset,
        parent_metadata=parent,
        migration_report=migration_report,
        parameter_report=parameter_report,
        optimizer_groups=optimizer_groups,
    )
    trainer.bind_training_contract(contract_sha256)
    if migration_report is not None and trainer.is_main:
        print(f"S3 migration report: {migration_report.to_dict()}", flush=True)

    set_seed(seed + accelerator.process_index)
    run_until_raw = os.environ.get("S3_RUN_UNTIL_STAGE_UPDATE")
    run_until = int(model_cfg.optim.max_updates) if run_until_raw is None else int(run_until_raw)
    if trainer.is_main:
        print(
            f"Stage {stage}: local scheduler horizon={trainer.max_stage_updates}; "
            f"cumulative offset={S3_STAGE_START_UPDATE[stage]}; this invocation stops at local update={run_until}",
            flush=True,
        )
    trainer.train(
        dataset,
        num_workers=int(model_cfg.datasets.num_workers),
        seed=seed,
        run_until_stage_update=run_until,
        deterministic_update_seed=bool(model_cfg.optim.deterministic_update_seed),
    )


if __name__ == "__main__":
    main()
