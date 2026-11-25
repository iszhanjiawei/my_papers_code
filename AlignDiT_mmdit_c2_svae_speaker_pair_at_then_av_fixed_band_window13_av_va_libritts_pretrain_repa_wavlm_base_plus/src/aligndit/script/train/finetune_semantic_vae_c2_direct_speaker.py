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
from aligndit.model.paired_attention_contract import checkpoint_architecture
from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_LAYER,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
    validate_repa_cache_metadata,
)
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.speaker_embedding import validate_speaker_cache_metadata
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def validate_repa_training_config(model_cfg):
    """Validate the fixed reference REPA recipe and pinned CelebV-Dub targets."""

    repa_lambda = float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0))
    if not math.isfinite(repa_lambda) or repa_lambda < 0:
        raise ValueError(f"repa_lambda must be finite and non-negative, got {repa_lambda}")
    if repa_lambda == 0:
        return None
    expected = {
        "model.repa_lambda": 0.1,
        "model.arch.repa_layer": 9,
        "model.arch.repa_target_dim": 768,
        "model.arch.repa_projector_dim": 2048,
        "datasets.repa_feature_dim": 768,
        "datasets.repa_model_id": WAVLM_BASE_PLUS_MODEL_ID,
        "datasets.repa_model_revision": WAVLM_BASE_PLUS_REVISION,
        "datasets.repa_checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
        "datasets.repa_teacher_layer": WAVLM_BASE_PLUS_LAYER,
    }
    for key, expected_value in expected.items():
        actual = OmegaConf.select(model_cfg, key)
        if actual != expected_value:
            raise ValueError(f"REPA contract mismatch: {key}={actual!r}, expected {expected_value!r}")
    return validate_repa_cache_metadata(
        model_cfg.datasets.repa_feature_cache_dir,
        expected_manifest_sha256=model_cfg.datasets.expected_manifest_sha256,
        expected_count=int(model_cfg.datasets.expected_record_count),
        expected_dim=int(model_cfg.datasets.repa_feature_dim),
        model_id=model_cfg.datasets.repa_model_id,
        model_revision=model_cfg.datasets.repa_model_revision,
        checkpoint_sha256=model_cfg.datasets.repa_checkpoint_sha256,
        teacher_layer=int(model_cfg.datasets.repa_teacher_layer),
    )


def dataset_from_config(model_cfg):
    """Build the exact training split for launch, preflight and batch checks."""

    repa_enabled = float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0)) > 0
    return SemanticVaeCelebVDubDataset(
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


def make_repa_contract(model_cfg, dataset):
    """Describe the train-only auxiliary supervision and validated target cache."""

    if float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0)) == 0:
        return None
    if dataset.repa_feature_contract is None:
        raise RuntimeError("REPA is enabled but the training dataset has no validated teacher cache")
    return {
        "lambda": float(model_cfg.model.repa_lambda),
        "schedule": "fixed from the first update",
        "student_layer_zero_based": int(model_cfg.model.arch.repa_layer),
        "teacher_layer": int(model_cfg.datasets.repa_teacher_layer),
        "projector": "768 -> 2048 -> 2048 -> 768, SiLU",
        "alignment": "linear interpolation of each unpadded 50-Hz target to valid 40-Hz latent length",
        "loss": "mean 1-cosine on the flow-matching generation mask only",
        "inference": "projector weights load strictly but the auxiliary head is not called",
        "teacher_cache": dataset.repa_feature_contract,
    }


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
    set_seed(int(model_cfg.seed))
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    repa_metadata = validate_repa_training_config(model_cfg)
    repa_lambda = float(OmegaConf.select(model_cfg, "model.repa_lambda", default=0.0))
    if model_cfg.ckpts.log_samples:
        raise ValueError("Use the Semantic-VAE inference entry for samples; inherited mel sample logging is unsupported")
    speaker_dim = int(model_arc.speaker_dim)
    if speaker_dim != 192 or speaker_dim != int(model_cfg.datasets.speaker_embedding_dim):
        raise ValueError("CAM++ model and dataset must both use speaker_embedding_dim=192")
    if int(model_arc.speaker_condition_start_layer) != int(model_arc.n_mm_layers):
        raise ValueError("CAM++ conditioning must start at the first audio-only block")
    speaker_metadata = validate_speaker_cache_metadata(
        model_cfg.datasets.speaker_embedding_cache_dir,
        expected_dim=speaker_dim,
        model_id=model_cfg.datasets.speaker_embedding_model_id,
        checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
    )

    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires the requested global learning rate 5e-5")
    ctc_lambda = float(model_cfg.model.ctc_lambda)
    if not math.isfinite(ctc_lambda) or ctc_lambda <= 0:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires a finite positive ctc_lambda")
    if int(model_cfg.model.ctc_warmup_start) != 10_000 or int(model_cfg.model.ctc_warmup_end) != 30_000:
        raise RuntimeError("Direct-C2 CTC-warmup experiment requires start=10000 and end=30000")

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

    initialization_mode = model_cfg.ckpts.get("initialization_mode", "s2c")
    trainer = SemanticVaeDirectC2SpeakerTrainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=model_cfg.ckpts.save_dir,
        checkpoint_architecture=checkpoint_architecture(model_arc, initialization_mode),
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
        initialization_mode=initialization_mode,
        ctc_target_lambda=model_cfg.model.ctc_lambda,
        ctc_warmup_start=model_cfg.model.ctc_warmup_start,
        ctc_warmup_end=model_cfg.model.ctc_warmup_end,
    )
    trainer.run_until_update = int(model_cfg.optim.run_until_update)
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    train_dataset = dataset_from_config(model_cfg)
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paired_parent = initialization_mode == "paired_libritts_text_audio"
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "speaker_cache_metadata": speaker_metadata,
            "initialization": ("paired LibriTTS AT-only 50k EMA, new optimizer/update counter" if paired_parent
                               else "same S2c 70k EMA parent as Direct-C2, new optimizer/update counter"),
            "speaker_condition": ("L2 CAM++ -> LibriTTS-pretrained projection in the audio-only tail" if paired_parent
                                  else "L2 CAM++ -> zero-initialized bias-free Linear(192,768) in the audio-only tail"),
            "seed": int(model_cfg.seed),
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        if repa_metadata is not None:
            contract["repa_cache_metadata"] = repa_metadata
            contract["repa"] = make_repa_contract(model_cfg, train_dataset)
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different contract: {save_dir}")
        else:
            contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")

    if trainer.is_main:
        print(
            "Direct-C2 CAM++ speaker CTC-warmup dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"repa_lambda={repa_lambda}",
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
