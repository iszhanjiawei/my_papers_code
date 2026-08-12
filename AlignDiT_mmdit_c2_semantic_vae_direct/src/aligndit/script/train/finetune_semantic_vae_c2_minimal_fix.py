"""Train the one-stage minimal repair for Semantic-VAE C2."""

from __future__ import annotations

import os
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.trainer_semantic_vae_minimal_fix import SemanticVaeMinimalFixC2Trainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def _file_contract(path: str) -> dict[str, object]:
    candidate = Path(path).resolve(strict=True)
    digest = sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(candidate), "size": candidate.stat().st_size, "sha256": digest.hexdigest()}


def _validate_repair_config(model_cfg) -> int:
    if int(model_cfg.seed) != 666:
        raise RuntimeError("Semantic-VAE minimal repair uses the fixed experiment seed 666")
    if int(model_cfg.datasets.expected_record_count) != 79_613:
        raise RuntimeError("minimal repair must use the complete 79,613-record Direct-C2 manifest")
    if float(model_cfg.optim.learning_rate) != 1e-5:
        raise RuntimeError("minimal repair requires a single global learning rate of 1e-5")
    if int(model_cfg.optim.grad_accumulation_steps) != 1:
        raise RuntimeError("minimal repair requires grad_accumulation_steps=1")
    fixed_arch = {
        "depth": 18,
        "n_mm_layers": 12,
        "n_text_layers": 12,
        "layer_indices_ctc": [6, 12],
        "ctc_sampling_ratios": [1, 1],
        "audio_video_ratio": 1,
    }
    for field, expected in fixed_arch.items():
        actual = model_cfg.model.arch[field]
        actual = list(actual) if isinstance(expected, list) else int(actual)
        if actual != expected:
            raise RuntimeError(f"minimal repair requires model.arch.{field}={expected}, got {actual}")
    if not bool(model_cfg.model.arch.normalize_text_context):
        raise RuntimeError("minimal repair requires parameter-free text-context normalization")
    if bool(model_cfg.model.arch.prompt_isolated_ca) or bool(model_cfg.model.arch.video_rope_scaled):
        raise RuntimeError("minimal repair must retain C2 global text CA and shared 40-Hz RoPE")
    if float(model_cfg.model.ctc_lambda) != 0.1:
        raise RuntimeError("minimal repair retains the original fixed CTC lambda 0.1")
    representation = model_cfg.model.audio_representation
    if (
        int(representation.channels) != 64
        or int(representation.frame_rate) != 40
        or int(representation.sample_rate) != 16_000
        or int(representation.hop_length) != 400
    ):
        raise RuntimeError("minimal repair requires the fixed 64-D/40-Hz Semantic-VAE representation")

    max_updates = int(model_cfg.optim.max_updates)
    run_until_raw = os.environ.get("MINIMAL_FIX_RUN_UNTIL_UPDATE")
    run_until = max_updates if run_until_raw is None else int(run_until_raw)
    if not 0 < run_until <= max_updates:
        raise RuntimeError(f"run limit must be in [1, {max_updates}], got {run_until}")
    return run_until


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg) -> None:
    run_until = _validate_repair_config(model_cfg)
    seed = int(model_cfg.seed)
    set_seed(seed)

    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")
    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"

    model = CFM_VT(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=audio_cfg.channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=audio_cfg.channels,
            target_sample_rate=audio_cfg.sample_rate,
            hop_length=audio_cfg.hop_length,
        ),
        num_channels=audio_cfg.channels,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=model_arc.audio_video_ratio,
        ctc_lambda=model_cfg.model.ctc_lambda,
    )

    trainer = SemanticVaeMinimalFixC2Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=model_cfg.ckpts.save_dir,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="AlignDiT",
        wandb_run_name=exp_name,
        wandb_resume_id=None,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=audio_cfg.name,
        is_local_vocoder=False,
        local_vocoder_path="",
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        ema_kwargs=model_cfg.ema,
        parent_contract_path=model_cfg.ckpts.parent_contract_path,
        expected_parent_sha256=model_cfg.ckpts.expected_parent_sha256,
        expected_parent_size=model_cfg.ckpts.expected_parent_size,
        expected_parent_contract_sha256=model_cfg.ckpts.expected_parent_contract_sha256,
        expected_parent_update=model_cfg.ckpts.expected_parent_update,
        seed=seed,
        run_until_update=run_until,
        global_grad_norm_abort_threshold=model_cfg.monitoring.global_grad_norm_abort_threshold,
        global_grad_norm_min_threshold=model_cfg.monitoring.global_grad_norm_min_threshold,
        post_text_rms_min=model_cfg.monitoring.post_text_rms_min,
        post_text_rms_max=model_cfg.monitoring.post_text_rms_max,
        experiment_contract={
            "checkpoint_schema_version": 1,
            "training_policy": "semantic-vae40-c2-one-stage-minimal-fix-v1",
            "seed": seed,
            "world_size": 4,
            "gradient_accumulation_steps": int(model_cfg.optim.grad_accumulation_steps),
            "batch_size_per_gpu": int(model_cfg.datasets.batch_size_per_gpu),
            "batch_size_type": str(model_cfg.datasets.batch_size_type),
            "max_samples": int(model_cfg.datasets.max_samples),
            "manifest": _file_contract(model_cfg.datasets.manifest_path),
            "normalization": _file_contract(model_cfg.datasets.normalization_path),
            "vocabulary": _file_contract(model_cfg.datasets.vocab_path),
            "parent_checkpoint": _file_contract(model_cfg.ckpts.pretrained_path),
            "parent_contract": _file_contract(model_cfg.ckpts.parent_contract_path),
            "resolved_config": OmegaConf.to_container(model_cfg, resolve=True),
        },
    )
    expected_world_size = os.environ.get("EXPECTED_WORLD_SIZE")
    if expected_world_size is None or not expected_world_size.isdecimal():
        raise RuntimeError("EXPECTED_WORLD_SIZE must be exported by the minimal-fix launcher")
    if trainer.accelerator.num_processes != int(expected_world_size):
        raise RuntimeError(
            f"world-size mismatch: expected {expected_world_size}, got {trainer.accelerator.num_processes}"
        )
    trainer.publish_contract()

    train_dataset = SemanticVaeCelebVDubDataset(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
        vocab_path=model_cfg.datasets.vocab_path,
        expected_manifest_sha256=model_cfg.datasets.expected_manifest_sha256,
        expected_inventory_sha256=model_cfg.datasets.expected_inventory_sha256,
        expected_normalization_sha256=model_cfg.datasets.expected_normalization_sha256,
        expected_vocab_sha256=model_cfg.datasets.expected_vocab_sha256,
        expected_record_count=model_cfg.datasets.expected_record_count,
    )
    if trainer.is_main:
        print(
            "Minimal-fix dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"single-stage run stops at exact update={run_until}",
            flush=True,
        )

    set_seed(seed + trainer.accelerator.process_index)
    trainer.finetune(
        model_cfg.ckpts.pretrained_path,
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=seed,
    )


if __name__ == "__main__":
    main()
