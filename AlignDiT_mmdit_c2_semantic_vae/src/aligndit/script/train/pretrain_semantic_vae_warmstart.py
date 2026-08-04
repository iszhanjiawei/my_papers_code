"""Staged mel-EMA warm-start for 64-D, 40-Hz Semantic-VAE audio pretraining."""

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

from aligndit.model.cfm_notext import CFM_notext
from aligndit.model.dataset import SemanticVaePretrainDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_warmstart import (
    S1_EXPECTED_SHAPE_MISMATCHES,
    S1_QK_NORM_PATTERN,
    S1_RESET_PREFIXES,
    STAGE_NAMES,
    configure_stage_parameters,
)
from aligndit.model.trainer_semantic_vae_warmstart import (
    SemanticVaeWarmStartTrainer,
    create_warmstart_accelerator,
    has_local_training_checkpoint,
    initialize_parent_on_all_ranks,
)
from aligndit.script.misc.svae_cache_utils import atomic_write_json, safe_join, sha256_file


PROJECT_ROOT = Path(str(files("aligndit").joinpath("../.."))).resolve()
os.chdir(PROJECT_ROOT)
PREVIOUS_STAGE = {"s1": None, "s2a": "s1", "s2b": "s2a", "s2c": "s2b"}
POLICY_VERSION = "mel-ema-to-svae40-staged-v1"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parent_metadata(model_cfg) -> dict[str, Any]:
    parent_path = Path(model_cfg.stage.parent_checkpoint).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Warm-start parent checkpoint does not exist: {parent_path}")
    actual_sha256 = sha256_file(parent_path)
    expected_sha256 = model_cfg.stage.get("expected_parent_sha256")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(f"Warm-start parent SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")

    parent_contract_path_raw = model_cfg.stage.get("parent_contract_path")
    parent_contract_sha256 = None
    if parent_contract_path_raw is not None:
        parent_contract_path = Path(parent_contract_path_raw).resolve()
        if not parent_contract_path.is_file():
            raise FileNotFoundError(f"Previous-stage training contract does not exist: {parent_contract_path}")
        parent_contract_sha256 = sha256_file(parent_contract_path)

    return {
        "canonical_path": str(parent_path),
        "size": parent_path.stat().st_size,
        "sha256": actual_sha256,
        "expected_update": int(model_cfg.stage.expected_parent_update),
        "expected_stage": model_cfg.stage.get("expected_parent_stage"),
        "contract_sha256": parent_contract_sha256,
    }


def _checkpoint_files(checkpoint_dir: Path) -> list[str]:
    if not checkpoint_dir.exists():
        return []
    return sorted(
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and (path.name == "model_last.pt" or re.fullmatch(r"model_[0-9]+\.pt", path.name))
    )


def publish_training_contract(
    trainer: SemanticVaeWarmStartTrainer,
    model_cfg,
    dataset: SemanticVaePretrainDataset,
    *,
    parent_metadata: dict[str, Any],
    parameter_report,
    optimizer_groups: list[dict[str, Any]],
    load_report,
) -> str:
    checkpoint_dir = Path(model_cfg.ckpts.save_dir)
    contract_path = checkpoint_dir / "training_contract.json"
    checkpoint_files = _checkpoint_files(checkpoint_dir)
    if checkpoint_files and not contract_path.is_file():
        raise RuntimeError(
            f"Refusing to resume warm-start checkpoints {checkpoint_files} without contract {contract_path}"
        )

    source_paths = {
        "cfm_notext": PROJECT_ROOT / "src/aligndit/model/cfm_notext.py",
        "dataset": PROJECT_ROOT / "src/aligndit/model/dataset.py",
        "dit_notext": PROJECT_ROOT / "src/aligndit/model/backbone/dit_notext.py",
        "modules": PROJECT_ROOT / "src/aligndit/model/modules.py",
        "semantic_vae_warmstart": PROJECT_ROOT / "src/aligndit/model/semantic_vae_warmstart.py",
        "trainer_notext": PROJECT_ROOT / "src/aligndit/model/trainer_notext.py",
        "trainer_semantic_vae_warmstart": PROJECT_ROOT / "src/aligndit/model/trainer_semantic_vae_warmstart.py",
        "pretrain_semantic_vae_warmstart": Path(__file__).resolve(),
        "pretrain_semantic_vae_warmstart_launcher": PROJECT_ROOT
        / "src/aligndit/run/train/pretrain_semantic_vae_warmstart_6xa40.sh",
        "svae_cache_utils": PROJECT_ROOT / "src/aligndit/script/misc/svae_cache_utils.py",
        "f5_cfm": PROJECT_ROOT / "src/f5_tts/model/cfm.py",
        "f5_dataset": PROJECT_ROOT / "src/f5_tts/model/dataset.py",
        "f5_dit": PROJECT_ROOT / "src/f5_tts/model/backbones/dit.py",
        "f5_modules": PROJECT_ROOT / "src/f5_tts/model/modules.py",
        "f5_trainer": PROJECT_ROOT / "src/f5_tts/model/trainer.py",
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
    if trainer.is_main and load_report is None:
        if not contract_path.is_file():
            raise RuntimeError("A fresh warm-start stage has neither a migration report nor an existing contract")
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        migration_report = existing_contract.get("migration_report")
        if migration_report is None:
            raise RuntimeError(f"Existing warm-start contract has no migration_report: {contract_path}")
    elif trainer.is_main:
        migration_report = load_report.to_dict()
    else:
        migration_report = None
    contract = {
        "cache_completion": {
            "hubert_40hz": sha256_file(safe_join(dataset.cache_root, "state/hubert_40hz/complete.json")),
            "latents": sha256_file(safe_join(dataset.cache_root, "state/latents/complete.json")),
            "normalization": dataset.normalization_sha256,
        },
        "config": OmegaConf.to_container(model_cfg, resolve=True),
        "distributed_runtime": {
            "accelerate": accelerate.__version__,
            "cuda_allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "distributed_type": str(trainer.accelerator.distributed_type),
            "ema_pytorch": version("ema-pytorch"),
            "mixed_precision": trainer.accelerator.mixed_precision,
            "num_processes": trainer.accelerator.num_processes,
            "torch": torch.__version__,
        },
        "parent": parent_metadata,
        "migration_report": migration_report,
        "policy": {
            "version": POLICY_VERSION,
            "stage": trainer.stage,
            "previous_stage": PREVIOUS_STAGE[trainer.stage],
            "s1_reset_prefixes": list(S1_RESET_PREFIXES),
            "s1_expected_shape_mismatches": sorted(S1_EXPECTED_SHAPE_MISMATCHES),
            "s1_qk_norm_target_only_pattern": S1_QK_NORM_PATTERN.pattern,
            "ema_weights_only_across_stages": True,
            "optimizer_scheduler_update_reset_across_stages": True,
            "parent_loaded_strictly_on_every_rank": True,
            "projection_execution_disabled": trainer.stage != "s2c",
        },
        "parameter_policy": {
            "report": parameter_report.to_dict(),
            "group_membership": group_membership,
            "group_membership_sha256": _sha256_json(group_membership),
        },
        "schema_version": 1,
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    if trainer.is_main:
        result = atomic_write_json(contract_path, contract)
        print(
            f"Warm-start contract {'created' if result.created else 'verified'}: {result.path} sha256={result.sha256}",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
    return sha256_file(contract_path)


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg) -> None:
    stage = str(model_cfg.stage.name)
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown warm-start stage {stage!r}; expected one of {STAGE_NAMES}")
    expected_previous = PREVIOUS_STAGE[stage]
    configured_previous = model_cfg.stage.get("expected_parent_stage")
    if configured_previous != expected_previous:
        raise RuntimeError(
            f"Stage {stage} must follow {expected_previous!r}, but config declares {configured_previous!r}"
        )

    experiment_seed = int(model_cfg.seed)
    set_seed(experiment_seed)
    accelerator_instance = create_warmstart_accelerator(
        grad_accumulation_steps=int(model_cfg.optim.grad_accumulation_steps),
        logger=model_cfg.ckpts.logger,
    )
    expected_world_size_raw = os.environ.get("EXPECTED_WORLD_SIZE")
    if expected_world_size_raw is None or not expected_world_size_raw.isdecimal():
        raise RuntimeError("EXPECTED_WORLD_SIZE must be exported by the warm-start launcher")
    if accelerator_instance.num_processes != int(expected_world_size_raw):
        raise RuntimeError(
            f"Warm-start world-size mismatch: expected {expected_world_size_raw}, "
            f"got {accelerator_instance.num_processes}"
        )

    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    channels = int(model_cfg.model.audio_representation.channels)
    model = CFM_notext(
        transformer=model_cls(**model_cfg.model.arch, mel_dim=channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=int(model_cfg.model.audio_representation.sample_rate),
            hop_length=int(model_cfg.model.audio_representation.hop_length),
        ),
        num_channels=channels,
        proj_lambda=0.0,
    )
    if stage != "s2c":
        # Keep the freshly initialized projector in the state dict while
        # avoiding its unused forward compute before S2c.
        model.transformer.layer_map = {}

    parent = _parent_metadata(model_cfg)
    local_resume = has_local_training_checkpoint(model_cfg.ckpts.save_dir)
    load_report = None
    if not local_resume:
        load_report = initialize_parent_on_all_ranks(
            model,
            accelerator_instance,
            parent_path=parent["canonical_path"],
            stage=stage,
            expected_parent_update=parent["expected_update"],
            expected_parent_stage=parent["expected_stage"],
            expected_parent_contract_sha256=parent["contract_sha256"],
            expected_parent_sha256=parent["sha256"],
            expected_parent_size=parent["size"],
        )
    else:
        model.to(accelerator_instance.device)

    learning_rates = {key: float(value) for key, value in model_cfg.optim.learning_rates.items()}
    optimizer_groups, parameter_report = configure_stage_parameters(
        model,
        stage=stage,
        learning_rates=learning_rates,
        weight_decay=float(model_cfg.optim.weight_decay),
    )
    exp_name = f"{model_cfg.model.name}_{stage}_{model_cfg.datasets.name}"
    trainer = SemanticVaeWarmStartTrainer(
        model,
        accelerator=accelerator_instance,
        optimizer_groups=optimizer_groups,
        stage=stage,
        epochs=int(model_cfg.optim.epochs),
        num_warmup_updates=int(model_cfg.optim.num_warmup_updates),
        save_per_updates=int(model_cfg.ckpts.save_per_updates),
        keep_last_n_checkpoints=int(model_cfg.ckpts.keep_last_n_checkpoints),
        checkpoint_path=model_cfg.ckpts.save_dir,
        batch_size_per_gpu=int(model_cfg.datasets.batch_size_per_gpu),
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=int(model_cfg.datasets.max_samples),
        grad_accumulation_steps=int(model_cfg.optim.grad_accumulation_steps),
        max_grad_norm=float(model_cfg.optim.max_grad_norm),
        logger=model_cfg.ckpts.logger,
        wandb_run_name=exp_name,
        last_per_updates=int(model_cfg.ckpts.last_per_updates),
        ema_kwargs=OmegaConf.to_container(model_cfg.ema, resolve=True),
        projection_target_lambda=float(model_cfg.projection.target_lambda),
        projection_ramp_updates=int(model_cfg.projection.ramp_updates),
    )

    train_dataset = SemanticVaePretrainDataset(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
    )
    contract_sha256 = publish_training_contract(
        trainer,
        model_cfg,
        train_dataset,
        parent_metadata=parent,
        parameter_report=parameter_report,
        optimizer_groups=optimizer_groups,
        load_report=load_report,
    )
    trainer.bind_training_contract(contract_sha256)
    if load_report is not None and trainer.is_main:
        print(f"Warm-start load report: {load_report.to_dict()}", flush=True)
    set_seed(experiment_seed + trainer.accelerator.process_index)
    max_updates = int(model_cfg.optim.max_updates)
    run_until_raw = os.environ.get("WARMSTART_RUN_UNTIL_UPDATE")
    run_until_update = max_updates if run_until_raw is None else int(run_until_raw)
    if trainer.is_main:
        print(
            f"Stage {stage}: scheduler horizon={max_updates}, this invocation stops at update={run_until_update}",
            flush=True,
        )
    trainer.train(
        train_dataset,
        num_workers=int(model_cfg.datasets.num_workers),
        resumable_with_seed=experiment_seed,
        max_updates=max_updates,
        run_until_update=run_until_update,
        deterministic_update_seed=bool(model_cfg.optim.deterministic_update_seed),
    )


if __name__ == "__main__":
    main()
