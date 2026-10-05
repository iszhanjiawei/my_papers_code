"""Train baseline-global Direct-C2 with framewise visual modulation and frozen CAM++."""

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
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.speaker_embedding import validate_speaker_cache_metadata
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


@hydra.main(
    version_base="1.3",
    config_path=str(files("aligndit").joinpath("config")),
    config_name="finetune_celebvdub_mm_c2_svae_speaker_framewise_visual_mod",
)
def main(model_cfg):
    set_seed(int(model_cfg.seed))
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    if int(model_arc.n_mm_layers) != 12 or int(model_arc.audio_video_ratio) != 1:
        raise ValueError("The visual-modulation experiment retains 12 MM layers and aligned input rates")
    if float(audio_cfg.frame_rate) != 40.0:
        raise ValueError("Framewise visual modulation uses the aligned 40-Hz audio/video grid")
    if model_arc.get("framewise_visual_modulation") is not True:
        raise ValueError("This experiment requires framewise_visual_modulation=true")
    if str(model_arc.visual_modulation_mode) != "framewise":
        raise ValueError("This training entry requires visual_modulation_mode=framewise")
    if int(model_arc.visual_modulation_hidden_dim) != 128:
        raise ValueError("The primary visual-modulation experiment uses hidden dimension 128")
    incompatible_keys = [
        key for key in model_arc
        if key.startswith(("temporal_band_", "repa_", "infonce_"))
        or key in {
            "av_local_window_radius", "block_video_audio_attention",
            "av_visual_delta_gate_init", "vv_local_window_radius",
        }
    ]
    if incompatible_keys or any(key.startswith(("repa_", "infonce_")) for key in model_cfg.model):
        raise ValueError(f"This experiment changes only visual modulation, keeping baseline losses: {incompatible_keys}")
    experiment_tag = "framewise_visual_mod"
    if any(experiment_tag not in str(value) for value in (model_cfg.model.name, model_cfg.ckpts.save_dir)):
        raise ValueError("Framewise visual modulation requires dedicated model.name and ckpts.save_dir")
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
    )

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
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "speaker_cache_metadata": speaker_metadata,
            "initialization": "same S2c 70k EMA parent as Direct-C2, new optimizer/update counter",
            "speaker_condition": "L2 CAM++ -> zero-initialized bias-free Linear(192,768), blocks 12..17",
            "seed": int(model_cfg.seed),
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        contract["attention"] = {
            "scope": "baseline first 12 MM blocks; topology unchanged",
            "audio_to_audio": "global",
            "audio_to_video": "global; shared joint audio/video softmax",
            "video_to_audio": "global",
            "video_to_video": "global",
            "adaptive_band": False,
            "audio_video_frame_rate": float(audio_cfg.frame_rate),
        }
        contract["visual_modulation"] = {
            "enabled": True,
            "mode": str(model_arc.visual_modulation_mode),
            "hidden_dim": int(model_arc.visual_modulation_hidden_dim),
            "scope": "audio attention and feedforward AdaLN in MM blocks 0..11",
            "source": "aligned raw 1024-D input video before the video context encoder",
            "condition": "corresponding video frame plus flow-time embedding",
            "initialization": "zero output projections; baseline behavior at initialization",
            "masking": "generation region intersected with valid video/audio positions; disabled for video CFG dropout",
            "text_and_speaker": "unchanged text cross-attention and speaker tail blocks 12..17",
        }
        contract["losses"] = {
            "flow": "unchanged baseline flow matching on the generation mask",
            "ctc": "unchanged 0 through 10k, linearly to 0.03 at 30k",
            "additional_losses": [],
        }
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different contract: {save_dir}")
        else:
            contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")

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
        speaker_embedding_cache_dir=model_cfg.datasets.speaker_embedding_cache_dir,
        speaker_embedding_dim=speaker_dim,
        speaker_embedding_model_id=model_cfg.datasets.speaker_embedding_model_id,
        speaker_embedding_checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
    )
    if trainer.is_main:
        print(
            "Direct-C2 CAM++ speaker CTC-warmup dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"visual_modulation={model_arc.visual_modulation_mode}; AA/AV/VA/VV=global; no auxiliary losses",
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
