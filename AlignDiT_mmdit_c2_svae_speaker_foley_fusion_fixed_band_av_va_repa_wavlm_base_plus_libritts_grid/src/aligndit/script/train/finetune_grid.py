"""Fine-tune the LibriTTS-initialized Foley AV/VA model on GRID."""

import json
import math
import os
from importlib.resources import files
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.grid_dataset import GridDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def dataset_from_config(model_cfg, *, allow_partial=False):
    """Load and validate GRID data without instantiating the diffusion model."""
    repa_enabled = float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0)) > 0
    if model_cfg.datasets.cache_format == "grid_mmdit_svae":
        from aligndit.model.grid_mmdit_dataset import GridMmditDataset
        dataset_cls = GridMmditDataset
    elif model_cfg.datasets.cache_format == "grid_aligned_mpg":
        dataset_cls = GridDataset
    else:
        raise ValueError(f"Unsupported GRID cache format: {model_cfg.datasets.cache_format}")
    return dataset_cls(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
        vocab_path=model_cfg.datasets.vocab_path,
        expected_manifest_sha256=model_cfg.datasets.expected_manifest_sha256,
        expected_inventory_sha256=model_cfg.datasets.expected_inventory_sha256,
        expected_normalization_sha256=model_cfg.datasets.expected_normalization_sha256,
        expected_vocab_sha256=model_cfg.datasets.expected_vocab_sha256,
        expected_record_count=model_cfg.datasets.expected_record_count,
        expected_vae_checkpoint_sha256=model_cfg.datasets.expected_vae_checkpoint_sha256,
        expected_video_checkpoint_sha256=model_cfg.datasets.expected_video_checkpoint_sha256,
        allow_partial=allow_partial,
        speaker_embedding_cache_dir=model_cfg.datasets.speaker_embedding_cache_dir,
        speaker_embedding_dim=int(model_cfg.datasets.speaker_embedding_dim),
        speaker_embedding_model_id=model_cfg.datasets.speaker_embedding_model_id,
        speaker_embedding_checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
        repa_feature_cache_dir=(model_cfg.datasets.repa_feature_cache_dir if repa_enabled else None),
        repa_feature_dim=(int(model_cfg.datasets.repa_feature_dim) if repa_enabled else 768),
        repa_model_id=(model_cfg.datasets.repa_model_id if repa_enabled else None),
        repa_model_revision=(model_cfg.datasets.repa_model_revision if repa_enabled else None),
        repa_checkpoint_sha256=(model_cfg.datasets.repa_checkpoint_sha256 if repa_enabled else None),
        repa_teacher_layer=(int(model_cfg.datasets.repa_teacher_layer) if repa_enabled else None),
    )


def training_schedule_from_config(model_cfg, dataset, *, world_size=1):
    """Check the real dynamic sampler and Accelerate's drop-last DDP policy."""
    from torch.utils.data import SequentialSampler

    from f5_tts.model.dataset import DynamicBatchSampler

    if world_size <= 0 or int(model_cfg.optim.grad_accumulation_steps) <= 0:
        raise ValueError("world size and gradient accumulation must be positive")
    if model_cfg.datasets.batch_size_type != "frame":
        raise ValueError("The GRID entry requires frame batching to preserve the Direct-C2 experiment")
    budget = int(model_cfg.datasets.batch_size_per_gpu)
    if budget <= 0 or any(row["latent_frames"] > budget for row in dataset.records):
        raise ValueError("GRID frame budget would silently discard at least one utterance; increase it")
    sampler = DynamicBatchSampler(
        SequentialSampler(dataset),
        budget,
        max_samples=int(model_cfg.datasets.max_samples),
        random_seed=int(model_cfg.seed),
        drop_residual=False,
    )
    if sampler.drop_last is not True:
        raise RuntimeError("GRID DDP requires dropping incomplete groups of frame batches")
    per_rank_batches = len(sampler) // world_size
    if not per_rank_batches:
        raise ValueError("GRID has fewer frame batches than distributed processes")
    grad_accum = int(model_cfg.optim.grad_accumulation_steps)
    epochs = int(model_cfg.optim.epochs)
    effective_updates = math.ceil(per_rank_batches / grad_accum) * epochs
    stop = int(model_cfg.optim.run_until_update)
    warmup = int(model_cfg.optim.num_warmup_updates)
    scheduler_steps = math.ceil(len(sampler) / grad_accum) * epochs
    if stop <= 0 or effective_updates < stop:
        raise ValueError(
            f"GRID epochs/batch/GPU configuration supports {effective_updates} updates, "
            f"below run_until_update={stop}; increase optim.epochs or lower the run limit"
        )
    if warmup <= 0 or scheduler_steps <= warmup * world_size:
        raise ValueError("GRID learning-rate schedule requires positive warmup and decay durations")
    return {
        "world_size": world_size,
        "frame_batches": len(sampler),
        "per_rank_batches": per_rank_batches,
        "dropped_tail_batches_per_epoch": len(sampler) % world_size,
        "tail_policy": "drop incomplete DDP group after deterministic epoch shuffle",
        "effective_max_updates": effective_updates,
        "run_until_update": stop,
        "scheduler_total_steps": scheduler_steps,
        "scheduler_warmup_steps": warmup * world_size,
    }


def model_from_config(model_cfg):
    """Build the Foley model for the target GRID tokenizer and audio representation."""
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")

    return CFM_VT(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=audio_cfg.channels,
        ),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=audio_cfg.channels,
            target_sample_rate=audio_cfg.sample_rate,
            hop_length=audio_cfg.hop_length,
        ),
        num_channels=audio_cfg.channels,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=model_arc.audio_video_ratio,
        ctc_lambda=model_cfg.model.ctc_lambda,
        repa_lambda=model_cfg.model.repa_lambda,
    )


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name="finetune_grid")
def main(model_cfg):
    if model_cfg.ckpts.logger != "tensorboard":
        raise ValueError("GRID training requires ckpts.logger=tensorboard to record loss curves")
    set_seed(int(model_cfg.seed))
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    expected = {
        "joint_av_text_ca": True,
        "n_mm_layers": 12,
        "audio_video_ratio": 1,
        "temporal_band_enabled": True,
        "temporal_band_mode": "fixed",
        "temporal_band_bidirectional": True,
        "temporal_band_audio_fps": 40.0,
        "temporal_band_video_fps": 40.0,
        "temporal_band_fixed_offset_seconds": 0.0,
        "temporal_band_fixed_sigma_seconds": 0.1,
    }
    for key, value in expected.items():
        if model_arc.get(key) != value:
            raise ValueError(f"Foley AV/VA training requires {key}={value!r}")
    if float(audio_cfg.frame_rate) != 40 or float(model_cfg.model.repa_lambda) != 0.1:
        raise ValueError("This Foley run requires 40-Hz latents and WavLM-Base+ REPA weight 0.1")
    if model_cfg.ckpts.log_samples:
        raise ValueError("Use the Semantic-VAE inference entry for samples; inherited mel sample logging is unsupported")
    speaker_dim = int(model_arc.speaker_dim)
    if speaker_dim != 192 or speaker_dim != int(model_cfg.datasets.speaker_embedding_dim):
        raise ValueError("CAM++ model and dataset must both use speaker_embedding_dim=192")
    if int(model_arc.speaker_condition_start_layer) != 12:
        raise ValueError("Speaker conditioning must start at zero-based block 12")
    repa_lambda = float(model_cfg.model.repa_lambda)
    if int(model_arc.repa_layer) != 9 or int(model_arc.repa_target_dim) != 768 or int(model_arc.repa_projector_dim) != 2048:
        raise ValueError("GRID WavLM-Base+ REPA uses block 10 and a 768 -> 2048 -> 2048 -> 768 projector")
    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires the requested global learning rate 5e-5")
    ctc_lambda = float(model_cfg.model.ctc_lambda)
    if not math.isfinite(ctc_lambda) or ctc_lambda <= 0:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires a finite positive ctc_lambda")
    if int(model_cfg.model.ctc_warmup_start) != 10_000 or int(model_cfg.model.ctc_warmup_end) != 30_000:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires start=10000 and end=30000")

    if model_cfg.ckpts.initialization_mode != "libritts_text_audio":
        raise ValueError("GRID requires LibriTTS text/audio EMA initialization")
    if str(model_cfg.datasets.name) != "GRID":
        raise ValueError("The GRID entry requires datasets.name=GRID")
    train_dataset = dataset_from_config(model_cfg)
    if train_dataset.split != "train":
        raise ValueError("GRID training must use the train manifest; val is held out")
    schedule = training_schedule_from_config(
        model_cfg, train_dataset, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    model_cfg.datasets.expected_manifest_sha256 = train_dataset.cache_contract["manifest_sha256"]
    model_cfg.datasets.expected_inventory_sha256 = train_dataset.cache_contract["inventory_sha256"]
    model_cfg.datasets.expected_record_count = len(train_dataset)
    speaker_metadata = train_dataset.speaker_embedding_contract
    repa_metadata = train_dataset.repa_feature_contract

    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"
    model = model_from_config(model_cfg)

    trainer = SemanticVaeDirectC2SpeakerTrainer(
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
        initialization_mode=model_cfg.ckpts.initialization_mode,
        parent_contract_path=model_cfg.ckpts.parent_contract_path,
        expected_parent_sha256=model_cfg.ckpts.expected_parent_sha256,
        expected_parent_size=model_cfg.ckpts.expected_parent_size,
        expected_parent_contract_sha256=model_cfg.ckpts.expected_parent_contract_sha256,
        expected_parent_update=model_cfg.ckpts.expected_parent_update,
        ctc_target_lambda=model_cfg.model.ctc_lambda,
        ctc_warmup_start=model_cfg.model.ctc_warmup_start,
        ctc_warmup_end=model_cfg.model.ctc_warmup_end,
    )
    trainer.run_until_update = int(model_cfg.optim.run_until_update)
    if trainer.accelerator.num_processes != schedule["world_size"]:
        schedule = training_schedule_from_config(model_cfg, train_dataset, world_size=trainer.accelerator.num_processes)
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "grid_cache": train_dataset.cache_contract,
            "batch_and_schedule": schedule,
            "validation": "existing GRID val split is held out; no online validation loop",
            "speaker_cache_metadata": speaker_metadata,
            "repa_cache_metadata": repa_metadata,
            "initialization": "LibriTTS Foley text/audio EMA with fresh GRID optimizer and update count",
            "speaker_condition": "L2 CAM++ -> LibriTTS-pretrained bias-free Linear(192,768), blocks 12..17",
            "seed": int(model_cfg.seed),
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        contract["repa"] = {
            "lambda": repa_lambda,
            "schedule": "fixed from the first update",
            "student_layer_zero_based": int(model_arc.repa_layer),
            "teacher_layer": int(model_cfg.datasets.repa_teacher_layer),
            "alignment": "linear interpolation of each unpadded 50-Hz target to valid 40-Hz latent length",
            "loss": "mean 1-cosine on the flow-matching generation mask only",
            "inference": "projector weights load strictly but the auxiliary head is not called",
        }
        contract["temporal_band"] = {
            "mode": "fixed Gaussian",
            "offset_seconds": model_arc.temporal_band_fixed_offset_seconds,
            "sigma_seconds": model_arc.temporal_band_fixed_sigma_seconds,
            "scope": "AV and VA logit blocks with transposed bias; AA and VV unmodified",
            "trainable_parameters": 0,
        }
        contract["text_cross_attention"] = "joint video and audio queries attend to shared text keys and values"
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different contract: {save_dir}")
        else:
            if any(save_dir.glob("model_*.pt")):
                raise RuntimeError(f"Existing checkpoint has no training contract: {save_dir}")
            temporary = contract_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(contract_path)

        print(
            "GRID Foley CAM++/REPA dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"repa_lambda={repa_lambda}; temporal_band=fixed AV/VA",
            flush=True,
        )
    trainer.finetune(
        model_cfg.ckpts.pretrained_path,
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=int(model_cfg.seed),
    )


if __name__ == "__main__":
    main()
