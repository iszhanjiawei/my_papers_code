"""Train the C2 architecture from scratch on cached MingTok acoustic latents."""

import os
from importlib.resources import files

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_mingtok import CFM_MingTok
from aligndit.model.dataset_mingtok import load_dataset_mingtok
from aligndit.model.trainer_mingtok import Trainer_MingTok
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
    experiment_seed = int(model_cfg.seed)
    # Construct identical scratch-initialized parameters on every rank. After
    # Accelerate initializes, switch each rank to its own reproducible stream.
    set_seed(experiment_seed)

    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    latent_name = model_cfg.model.latent.name

    exp_name = f"{model_cfg.model.name}_{latent_name}_{tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    data_dir = getattr(model_cfg.datasets, "data_dir", None)
    if data_dir and tokenizer not in ["custom", "byte"]:
        tokenizer_path = os.path.join(data_dir, f"{model_cfg.datasets.name}_{tokenizer}", "vocab.txt")
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, "custom")
    elif tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    model = CFM_MingTok(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.latent.dim,
        ),
        num_channels=model_cfg.model.latent.dim,
        audio_video_ratio=model_cfg.model.arch.audio_video_ratio,
        vocab_char_map=vocab_char_map,
        ctc_lambda=model_cfg.model.ctc_lambda,
    )

    trainer = Trainer_MingTok(
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
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=latent_name,
        is_local_vocoder=False,
        local_vocoder_path="",
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        ema_kwargs=model_cfg.ema,
        mingtok_repo_path=model_cfg.model.codec.repo_path,
        mingtok_checkpoint_dir=model_cfg.model.codec.checkpoint_dir,
        ctc_warmup_start=model_cfg.model.ctc_warmup_start,
        ctc_warmup_end=model_cfg.model.ctc_warmup_end,
    )

    rank_seed = experiment_seed + trainer.accelerator.process_index
    set_seed(rank_seed)

    if trainer.accelerator.is_main_process:
        print(f"Global experiment seed={experiment_seed}; training RNG uses seed + process_index on each rank")
        print(
            "Initialization policy: no audio-only pretrained checkpoint is loaded; "
            "start fresh unless this experiment save directory contains a resumable C2 checkpoint."
        )

    train_dataset = load_dataset_mingtok(
        model_cfg.datasets.name,
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=model_cfg.datasets.cache_dir,
        latent_dim=model_cfg.model.latent.dim,
        latent_fps=model_cfg.model.latent.fps,
        video_fps=model_cfg.datasets.video_fps,
        audio_video_ratio=model_cfg.model.arch.audio_video_ratio,
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=experiment_seed,
    )


if __name__ == "__main__":
    main()
