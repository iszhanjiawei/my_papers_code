"""Train isolated Semantic-VAE Direct-C2 with frozen CAM++ speaker conditions."""

import json
import math
import os
from importlib.resources import files
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.repa import validate_repa_cache_metadata
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.speaker_embedding import validate_speaker_cache_metadata
from aligndit.model.trainer_semantic_vae_adaptive_band import SemanticVaeAdaptiveBandTrainer
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def check_role_training_config(cfg):
    """Reject accidental training with a mismatched parent or local-attention recipe."""
    arch = cfg.model.arch
    expected = {
        "joint_av_text_ca": True, "role_conditioned_text": True, "temporal_band_enabled": True,
        "temporal_band_mode": "fixed", "temporal_band_bidirectional": True,
        "temporal_band_audio_fps": 40.0, "temporal_band_video_fps": 40.0,
        "temporal_band_fixed_offset_seconds": 0.0, "temporal_band_fixed_sigma_seconds": 0.1,
    }
    for name, value in expected.items():
        if arch.get(name) != value:
            raise ValueError(f"This Foley role experiment requires {name}={value!r}")
    if cfg.ckpts.get("initialization_mode") != "libritts_text_audio":
        raise ValueError("The fixed AV/VA role experiment requires the LibriTTS text/audio parent")
    if any("foley_fixed_band_av_va_libritts" not in name for name in (cfg.model.name, Path(cfg.ckpts.save_dir).name)):
        raise ValueError("Use the dedicated Foley fixed-band LibriTTS model name and checkpoint directory")


def dataset_from_config(model_cfg):
    repa_enabled = float(model_cfg.model.get("repa_lambda", 0.0)) > 0
    paired = bool(model_cfg.model.arch.get("role_conditioned_text", False))
    if bool(model_cfg.datasets.get("pairing_path")) != paired:
        raise ValueError("Role-conditioned CelebV-Dub training requires both paired data and a role-aware model")
    if paired:
        check_role_training_config(model_cfg)
        if "celeb_role_conditioned" not in Path(model_cfg.ckpts.save_dir).name or (
            "celeb_role_conditioned" not in model_cfg.model.name
        ):
            raise ValueError("Paired CelebV-Dub requires its own model name and checkpoint directory")
        if model_cfg.datasets.name != "CelebVDub" or model_cfg.ckpts.logger != "tensorboard":
            raise ValueError("Role-conditioned CelebV-Dub runs require the CelebVDub split and TensorBoard")
    base = SemanticVaeCelebVDubDataset(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
        vocab_path=model_cfg.datasets.vocab_path,
        expected_manifest_sha256=model_cfg.datasets.expected_manifest_sha256,
        expected_inventory_sha256=model_cfg.datasets.expected_inventory_sha256,
        expected_normalization_sha256=model_cfg.datasets.expected_normalization_sha256,
        expected_vocab_sha256=model_cfg.datasets.expected_vocab_sha256,
        expected_record_count=model_cfg.datasets.expected_record_count,
        speaker_embedding_cache_dir=model_cfg.datasets.speaker_embedding_cache_dir,
        speaker_embedding_dim=int(model_cfg.model.arch.speaker_dim),
        speaker_embedding_model_id=model_cfg.datasets.speaker_embedding_model_id,
        speaker_embedding_checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
        repa_feature_cache_dir=(model_cfg.datasets.repa_feature_cache_dir if repa_enabled else None),
        repa_feature_dim=(int(model_cfg.datasets.repa_feature_dim) if repa_enabled else 768),
        repa_model_id=(model_cfg.datasets.repa_model_id if repa_enabled else None),
        repa_model_revision=(model_cfg.datasets.repa_model_revision if repa_enabled else None),
        repa_checkpoint_sha256=(model_cfg.datasets.repa_checkpoint_sha256 if repa_enabled else None),
        repa_teacher_layer=(int(model_cfg.datasets.repa_teacher_layer) if repa_enabled else None),
    )
    if paired:
        from aligndit.model.celeb_role_dataset import PairedCelebVDubDataset

        return PairedCelebVDubDataset(base, model_cfg.datasets.pairing_path)
    return base


def paired_schedule(model_cfg, dataset, world_size):
    """Match the trainer's frame batching, DDP tail handling and LR scheduler."""
    from torch.utils.data import SequentialSampler
    from f5_tts.model.dataset import DynamicBatchSampler

    budget = int(model_cfg.datasets.batch_size_per_gpu)
    accumulation = int(model_cfg.optim.grad_accumulation_steps)
    epochs = int(model_cfg.optim.epochs)
    if world_size < 1 or accumulation < 1 or model_cfg.datasets.batch_size_type != "frame":
        raise ValueError("Paired training requires frame batching and positive DDP size and accumulation")
    sampler = DynamicBatchSampler(
        SequentialSampler(dataset), budget, max_samples=int(model_cfg.datasets.max_samples),
        random_seed=int(model_cfg.seed), drop_residual=False,
    )
    if sum(map(len, sampler.batches)) != len(dataset):
        longest = max(dataset.get_frame_len(i) for i in range(len(dataset)))
        raise ValueError(f"Per-GPU budget {budget} would omit pairs (longest P+T is {longest})")
    batches = len(sampler)
    per_rank = batches // world_size
    available_updates = math.ceil(per_rank / accumulation) * epochs
    run_until = int(model_cfg.optim.run_until_update)
    warmup_steps = int(model_cfg.optim.num_warmup_updates) * world_size
    scheduler_steps = math.ceil(batches / accumulation) * epochs
    if not 0 < run_until <= available_updates:
        raise ValueError(f"Requested {run_until} updates, sampler permits {available_updates}; adjust optim.epochs")
    if not 0 < warmup_steps < scheduler_steps:
        raise ValueError(f"LR schedule requires positive warmup and decay, got {warmup_steps=} {scheduler_steps=}")
    return {
        "world_size": world_size, "frame_budget_per_gpu": budget, "pairs": len(dataset),
        "batches_per_epoch": batches, "batches_per_rank": per_rank, "dropped_tail_batches": batches % world_size,
        "available_updates": available_updates, "run_until_update": run_until,
        "lr_warmup_steps": warmup_steps, "lr_total_steps": scheduler_steps,
    }


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
    set_seed(int(model_cfg.seed))
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    temporal_band_enabled = bool(model_arc.get("temporal_band_enabled", False))
    fixed_band = temporal_band_enabled and model_arc.get("temporal_band_mode", "adaptive") == "fixed"
    if temporal_band_enabled:
        if int(model_arc.n_mm_layers) != 12 or int(model_arc.audio_video_ratio) != 1:
            raise ValueError("The temporal band requires 12 MM layers and aligned input rates")
        if any(float(model_arc[key]) != float(audio_cfg.frame_rate) for key in (
            "temporal_band_audio_fps", "temporal_band_video_fps"
        )):
            raise ValueError("Temporal-band rates must match the already-interpolated 40-Hz cache")
        # Never allow an enabled experiment to write into an inherited baseline
        # checkpoint directory, even when a caller selects the wrong YAML.
        if not fixed_band and ("adaptive_band" not in str(model_cfg.ckpts.save_dir) or "adaptive_band" not in str(model_cfg.model.name)):
            raise ValueError("Adaptive-band runs require dedicated model.name and ckpts.save_dir")
    if model_cfg.ckpts.log_samples:
        raise ValueError("Use the Semantic-VAE inference entry for samples; inherited mel sample logging is unsupported")
    speaker_dim = int(model_arc.speaker_dim)
    if speaker_dim != 192 or speaker_dim != int(model_cfg.datasets.speaker_embedding_dim):
        raise ValueError("CAM++ model and dataset must both use speaker_embedding_dim=192")
    if int(model_arc.speaker_condition_start_layer) != 12:
        raise ValueError("Speaker conditioning must start at zero-based block 12")
    speaker_metadata = validate_speaker_cache_metadata(
        model_cfg.datasets.speaker_embedding_cache_dir,
        expected_dim=speaker_dim,
        model_id=model_cfg.datasets.speaker_embedding_model_id,
        checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
    )
    repa_lambda = float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0))
    if not math.isfinite(repa_lambda) or repa_lambda < 0:
        raise ValueError(f"repa_lambda must be finite and non-negative, got {repa_lambda}")
    repa_enabled = repa_lambda > 0
    repa_metadata = None
    if repa_enabled:
        if "repa_wavlm_base_plus" not in str(model_cfg.ckpts.save_dir) or "repa_wavlm_base_plus" not in str(model_cfg.model.name):
            raise ValueError("REPA runs require dedicated model.name and ckpts.save_dir")
        if int(model_arc.repa_layer) != 9:
            raise ValueError("The primary REPA experiment must tap zero-based MM-DiT layer 9 (the 10th block)")
        if int(model_arc.repa_target_dim) != 768 or int(model_arc.repa_projector_dim) != 2048:
            raise ValueError("The primary WavLM-Base+ REPA projector must be 768->2048->2048->768")
        repa_metadata = validate_repa_cache_metadata(
            model_cfg.datasets.repa_feature_cache_dir,
            expected_manifest_sha256=model_cfg.datasets.expected_manifest_sha256,
            expected_count=int(model_cfg.datasets.expected_record_count),
            expected_dim=int(model_cfg.datasets.repa_feature_dim),
            model_id=model_cfg.datasets.repa_model_id,
            model_revision=model_cfg.datasets.repa_model_revision,
            checkpoint_sha256=model_cfg.datasets.repa_checkpoint_sha256,
            teacher_layer=int(model_cfg.datasets.repa_teacher_layer),
        )

    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires the requested global learning rate 5e-5")
    ctc_lambda = float(model_cfg.model.ctc_lambda)
    if not math.isfinite(ctc_lambda) or ctc_lambda <= 0:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires a finite positive ctc_lambda")
    if int(model_cfg.model.ctc_warmup_start) != 10_000 or int(model_cfg.model.ctc_warmup_end) != 30_000:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires start=10000 and end=30000")

    train_dataset = dataset_from_config(model_cfg)
    paired = bool(model_arc.get("role_conditioned_text", False))
    schedule = paired_schedule(model_cfg, train_dataset, int(os.environ.get("WORLD_SIZE", 1))) if paired else None

    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")
    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"

    model = CFM_VT(
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
        repa_lambda=repa_lambda,
    )

    trainer_cls = SemanticVaeAdaptiveBandTrainer if temporal_band_enabled and not fixed_band else SemanticVaeDirectC2SpeakerTrainer
    trainer = trainer_cls(
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
        initialization_mode=model_cfg.ckpts.get("initialization_mode", "s2c"),
        ctc_target_lambda=model_cfg.model.ctc_lambda,
        ctc_warmup_start=model_cfg.model.ctc_warmup_start,
        ctc_warmup_end=model_cfg.model.ctc_warmup_end,
    )
    trainer.run_until_update = int(model_cfg.optim.run_until_update)
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "speaker_cache_metadata": speaker_metadata,
            "repa_cache_metadata": repa_metadata,
            "initialization": (
                "LibriTTS 50k text/audio EMA, adapted vocabulary and CTC rows, downstream optimizer/update start at zero"
                if fixed_band else "same S2c 70k EMA parent as Direct-C2, new optimizer/update counter"
            ),
            "speaker_condition": (
                "L2 CAM++ -> pretrained bias-free Linear(192,768), blocks 12..17"
                if fixed_band else "L2 CAM++ -> zero-initialized bias-free Linear(192,768), blocks 12..17"
            ),
            "seed": int(model_cfg.seed),
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        if paired:
            contract["train_pairing"] = train_dataset.pairing_contract
            contract["batch_and_schedule"] = schedule
            contract["text_roles"] = "separate reference style and target lexical streams; target-only FM, CTC and REPA"
        if repa_enabled:
            contract["repa"] = {
                "lambda": repa_lambda,
                "schedule": "fixed from the first update",
                "student_layer_zero_based": int(model_arc.repa_layer),
                "teacher_layer": int(model_cfg.datasets.repa_teacher_layer),
                "alignment": (
                    "linear interpolation of the target 50-Hz teacher to its target-only 40-Hz span"
                    if paired else "linear interpolation of each unpadded 50-Hz target to valid 40-Hz latent length"
                ),
                "loss": "mean 1-cosine on the flow-matching generation mask only",
                "inference": "projector weights load strictly but the auxiliary head is not called",
            }
        if fixed_band:
            contract["temporal_band"] = {
                "mode": "fixed Gaussian", "offset_seconds": model_arc.temporal_band_fixed_offset_seconds,
                "sigma_seconds": model_arc.temporal_band_fixed_sigma_seconds,
                "scope": "AV and VA logit blocks with transposed bias; AA and VV unmodified",
                "trainable_parameters": 0,
            }
            contract["text_cross_attention"] = "both video and audio queries attend to target text, with modality-specific outputs"
        elif temporal_band_enabled:
            contract["temporal_band"] = {
                "formula": "B[i,j] = -(t_video[j] - t_audio[i] - delta[i])**2 / (2*sigma[i]**2)",
                "predictor_input": "branch-specific video embedding before all joint audio/video attention",
                "sharing": "one predictor, shared over all heads and the first 12 MM blocks",
                "scope": "audio queries to video keys only; shared joint softmax retained",
                "initialization": (
                    f"zero offset and sigma={model_arc.temporal_band_init_sigma_seconds} seconds; "
                    "not function-preserving when enabled"
                ),
                "extra_loss": False,
                "mass_preservation": False,
                "parameters": {k: v for k, v in OmegaConf.to_container(model_arc, resolve=True).items()
                               if k.startswith("temporal_band_")},
                "parameter_count": sum(p.numel() for p in model.transformer.temporal_band.parameters()),
            }
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different contract: {save_dir}")
        else:
            if paired and any(save_dir.glob("model_*.pt")):
                raise RuntimeError(f"Existing checkpoint has no source pairing/training contract: {save_dir}")
            temporary = contract_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
            os.replace(temporary, contract_path)

    if trainer.is_main:
        print(
            "Direct-C2 CAM++ speaker CTC-warmup dataset validated: "
            f"records={len(train_dataset)}, paired_roles={paired}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"repa_lambda={repa_lambda}; temporal_band={temporal_band_enabled}",
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
