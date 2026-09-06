"""Fine-tune original D1 on Semantic-VAE latents with fixed CTC lambda 0.03."""

import math
import os
from importlib.resources import files

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectD1Trainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def validate_experiment_config(model_cfg) -> None:
    """Keep the D1 architecture and the requested Semantic-VAE experiment explicit."""

    arch = model_cfg.model.arch
    representation = model_cfg.model.audio_representation
    if (
        int(arch.depth) != 18
        or int(arch.n_mm_layers) != 6
        or int(arch.n_text_layers) != 6
        or list(arch.layer_indices_ctc) != [5, 11]
        or list(arch.ctc_sampling_ratios) != [1, 1]
        or int(arch.audio_video_ratio) != 1
        or bool(arch.prompt_isolated_ca)
        or bool(arch.video_rope_scaled)
        or arch.get("text_attention_mode", "audio_only") != "audio_only"
        or int(arch.pe_attn_head) != 1
        or arch.qk_norm != "rms_norm"
        or bool(arch.checkpoint_activations)
    ):
        raise RuntimeError("Semantic-VAE D1 requires 6 MM + 12 audio blocks, CTC after blocks 6/12, and 40-Hz strides")
    if (
        int(representation.channels) != 64
        or int(representation.frame_rate) != 40
        or int(representation.sample_rate) != 16000
        or int(representation.hop_length) != 400
    ):
        raise RuntimeError("Semantic-VAE D1 requires the fixed 64D/40Hz representation at 16 kHz")
    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise RuntimeError("Semantic-VAE D1 preserves the global learning rate 5e-5")
    ctc_lambda = float(model_cfg.model.ctc_lambda)
    if not math.isfinite(ctc_lambda) or ctc_lambda != 0.03:
        raise RuntimeError("Semantic-VAE D1 requires the requested ctc_lambda=0.03")
    if "ctc_warmup_start" in model_cfg.model or "ctc_warmup_end" in model_cfg.model:
        raise RuntimeError("This experiment fixes CTC lambda at 0.03 from the first update; no CTC warmup")
    if bool(model_cfg.ckpts.log_samples):
        raise RuntimeError("Semantic-VAE training requires log_samples=False; use the dedicated VAE inference entry")


def build_model(model_cfg, vocab_char_map: dict[str, int], vocab_size: int) -> CFM_VT:
    """Build the same validated model for training and contract smoke checks."""

    validate_experiment_config(model_cfg)
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
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
    )


@hydra.main(
    version_base="1.3",
    config_path=str(files("aligndit").joinpath("config")),
    config_name="finetune_celebvdub_mm_d1_semantic_vae_direct",
)
def main(model_cfg):
    validate_experiment_config(model_cfg)
    experiment_seed = int(model_cfg.seed)
    set_seed(experiment_seed)
    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")
    audio_cfg = model_cfg.model.audio_representation
    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"
    model = build_model(model_cfg, vocab_char_map, vocab_size)

    trainer = SemanticVaeDirectD1Trainer(
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
    )
    set_seed(experiment_seed + trainer.accelerator.process_index)
    if trainer.is_main:
        print(f"Global experiment seed={experiment_seed}; training RNG uses seed + process_index on each rank")

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
            "Original D1 Semantic-VAE fixed-CTC dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda={model_cfg.model.ctc_lambda} from update 1 (no CTC warmup)",
            flush=True,
        )
    trainer.finetune(
        model_cfg.ckpts.pretrained_path,
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=experiment_seed,
    )


if __name__ == "__main__":
    main()
