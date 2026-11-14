"""Train fixed-13 AVT GRID with WavLM REPA and the complete LibriTTS 50k EMA."""

import json
import math
import os
from importlib.resources import files
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.fixed_temporal_band_contract import (
    make_fixed_band_contract,
    validate_fixed_band_training_config,
)
from aligndit.model.grid_dataset import GridDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
)
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from aligndit.model.trainer_semantic_vae_fixed_band import SemanticVaeFixedBandTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def dataset_from_config(model_cfg, *, allow_partial=False):
    """Load and validate GRID data without instantiating the diffusion model."""
    return GridDataset(
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
        repa_feature_cache_dir=model_cfg.datasets.repa_feature_cache_dir,
        repa_feature_dim=int(model_cfg.datasets.repa_feature_dim),
        repa_model_id=model_cfg.datasets.repa_model_id,
        repa_model_revision=model_cfg.datasets.repa_model_revision,
        repa_checkpoint_sha256=model_cfg.datasets.repa_checkpoint_sha256,
        repa_teacher_layer=int(model_cfg.datasets.repa_teacher_layer),
    )


def make_repa_contract(model_cfg, dataset):
    """Pin the reference GRID REPA experiment while keeping fixed-13 AVT."""
    arch, data = model_cfg.model.arch, model_cfg.datasets
    if float(model_cfg.model.repa_lambda) != 0.1:
        raise ValueError("This REPA comparison requires a fixed repa_lambda=0.1 from update zero")
    if (int(arch.repa_layer), int(arch.repa_target_dim), int(arch.repa_projector_dim)) != (9, 768, 2048):
        raise ValueError("REPA requires zero-based student block 9 and the 768->2048->2048->768 projector")
    expected_teacher = {
        "repa_feature_dim": 768,
        "repa_model_id": WAVLM_BASE_PLUS_MODEL_ID,
        "repa_model_revision": WAVLM_BASE_PLUS_REVISION,
        "repa_checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
        "repa_teacher_layer": 12,
    }
    for key, value in expected_teacher.items():
        if data.get(key) != value:
            raise ValueError(f"Pinned WavLM-Base+ REPA teacher changed: {key}")
    if dataset.repa_feature_cache_dir is None or dataset.repa_feature_contract is None:
        raise ValueError("REPA requires the audited GRID WavLM-Base+ feature cache")
    if "repa_wavlm_base_plus" not in str(model_cfg.model.name) or "repa_wavlm_base_plus" not in str(model_cfg.ckpts.save_dir):
        raise ValueError("REPA requires isolated model.name and ckpts.save_dir")
    if int(model_cfg.optim.run_until_update) != 150_000:
        raise ValueError("This REPA comparison must stop at exactly 150000 GRID updates")
    return {
        "lambda": 0.1,
        "schedule": "fixed from the first update",
        "student_layer_zero_based": 9,
        "projector": {"dimensions": [768, 2048, 2048, 768], "activation": "SiLU", "new_parameter_tensors": 6},
        "teacher_model_id": WAVLM_BASE_PLUS_MODEL_ID,
        "teacher_model_revision": WAVLM_BASE_PLUS_REVISION,
        "teacher_checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
        "teacher_layer": 12,
        "teacher_frozen": True,
        "teacher_source_audio": "complete_unmasked_waveform",
        "alignment": "linear interpolation of each unpadded 50-Hz target to valid 40-Hz latent length",
        "loss": "mean 1-cosine on the flow-matching generation mask only",
        "inference": "projector weights load strictly but the auxiliary head is not called",
    }


def training_schedule_from_config(model_cfg, dataset, *, world_size=1):
    """Check the real dynamic sampler and Accelerate's drop-last DDP policy."""
    from accelerate.data_loader import BatchSamplerShard
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
    rank_lengths = [
        len(BatchSamplerShard(
            sampler, num_processes=world_size, process_index=rank,
            split_batches=False, even_batches=False,
        ))
        for rank in range(world_size)
    ]
    if len(set(rank_lengths)) != 1:
        raise RuntimeError("GRID distributed ranks would receive different numbers of frame batches")
    per_rank_batches = rank_lengths[0]
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


def validate_grid_training_target(model_cfg, dataset, schedule):
    """Reject a held-out split or an inherited checkpoint before model allocation."""
    if str(model_cfg.datasets.name) != "GRID" or dataset.split != "train":
        raise ValueError("GRID training must use datasets.name=GRID and the train manifest; val is held out")
    if model_cfg.ckpts.get("initialization_mode") != "libritts_text_audio":
        raise ValueError("This GRID experiment requires the complete LibriTTS 50k EMA initialization")
    if "grid" not in str(model_cfg.ckpts.save_dir).lower() or "grid" not in str(model_cfg.model.name).lower():
        raise ValueError("GRID requires isolated model.name and ckpts.save_dir")
    if dataset.speaker_embedding_cache_dir is None:
        raise ValueError("This GRID experiment requires the existing CAM++ speaker cache")
    repa_contract = make_repa_contract(model_cfg, dataset)
    save_dir = Path(model_cfg.ckpts.save_dir)
    contract_path = save_dir / "speaker_training_contract.json"
    if any(save_dir.glob("*.pt")) and not contract_path.is_file():
        raise RuntimeError("Existing GRID checkpoints require their original GRID training contract")
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        expected = {
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "grid_cache": dataset.cache_contract,
            "batch_and_schedule": schedule,
            "repa_cache_metadata": dataset.repa_feature_contract,
            "repa": repa_contract,
        }
        if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Refusing to resume a different experiment or dataset in the GRID checkpoint directory")


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name="finetune_grid")
def main(model_cfg):
    if model_cfg.ckpts.logger != "tensorboard":
        raise ValueError("GRID training requires ckpts.logger=tensorboard to record loss curves")
    set_seed(int(model_cfg.seed))
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    temporal_band_enabled = validate_fixed_band_training_config(model_cfg)
    if model_cfg.ckpts.log_samples:
        raise ValueError("Use the Semantic-VAE inference entry for samples; inherited mel sample logging is unsupported")
    speaker_dim = int(model_arc.speaker_dim)
    if speaker_dim != 192 or speaker_dim != int(model_cfg.datasets.speaker_embedding_dim):
        raise ValueError("CAM++ model and dataset must both use speaker_embedding_dim=192")
    if int(model_arc.speaker_condition_start_layer) != 12:
        raise ValueError("Speaker conditioning must start at zero-based block 12")

    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires the requested global learning rate 5e-5")
    ctc_lambda = float(model_cfg.model.ctc_lambda)
    if not math.isfinite(ctc_lambda) or ctc_lambda <= 0:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires a finite positive ctc_lambda")
    if int(model_cfg.model.ctc_warmup_start) != 10_000 or int(model_cfg.model.ctc_warmup_end) != 30_000:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires start=10000 and end=30000")

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
    validate_grid_training_target(model_cfg, train_dataset, schedule)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "GRID data and distributed schedule validated before model creation: "
            + json.dumps(
                {"grid_cache": train_dataset.cache_contract, "batch_and_schedule": schedule},
                ensure_ascii=False, sort_keys=True,
            ),
            flush=True,
        )

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
        repa_lambda=float(model_cfg.model.repa_lambda),
    )

    trainer_cls = SemanticVaeFixedBandTrainer if temporal_band_enabled else SemanticVaeDirectC2SpeakerTrainer
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
    if trainer.accelerator.num_processes != schedule["world_size"]:
        raise RuntimeError("Actual distributed process count differs from the pre-model GRID sampler audit")
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        libritts_parent = model_cfg.ckpts.get("initialization_mode", "s2c") == "libritts_text_audio"
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
            "repa_cache_metadata": train_dataset.repa_feature_contract,
            "repa": make_repa_contract(model_cfg, train_dataset),
            "initialization": (
                "complete LibriTTS text/audio/speaker 50k EMA; vocabulary rows mapped to the unchanged 159-character downstream vocabulary; "
                "only the six REPA projector tensors are newly initialized; new optimizer/update counter; carry parent EMA step"
                if libritts_parent else "same S2c 70k EMA parent as Direct-C2, new optimizer/update counter"
            ),
            "speaker_condition": (
                "L2 CAM++ -> LibriTTS-pretrained bias-free Linear(192,768), blocks 12..17"
                if libritts_parent else "L2 CAM++ -> zero-initialized bias-free Linear(192,768), blocks 12..17"
            ),
            "seed": int(model_cfg.seed),
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        if temporal_band_enabled:
            contract["temporal_band"] = make_fixed_band_contract(model_arc, model.transformer.temporal_band)
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different contract: {save_dir}")
        else:
            contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")

    if trainer.is_main:
        print(
            "GRID fixed-13 AVT CAM++ speaker CTC-warmup + WavLM-Base+ REPA dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"repa_lambda={model_cfg.model.repa_lambda} from update zero; stop at 150000",
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
