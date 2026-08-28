# training script.

import os
from importlib.resources import files

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.dataset import load_dataset_mel
from aligndit.model.modules import MelSpec_tacotron
from aligndit.model.trainer_vt import Trainer_VT
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
    experiment_seed = getattr(model_cfg, "seed", None)
    if experiment_seed is not None:
        experiment_seed = int(experiment_seed)
        # All ranks construct identical newly initialized parameters before DDP
        # synchronization. A rank-specific stream is selected after Trainer has
        # initialized Accelerate.
        set_seed(experiment_seed)

    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
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

    # set model
    model = CFM_VT(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_module=MelSpec_tacotron(**model_cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in model_cfg.model.mel_spec.items() if k != "mel_spec_type"},  # hack
        vocab_char_map=vocab_char_map,
        ctc_lambda=model_cfg.model.ctc_lambda,
    )

    # init trainer
    trainer = Trainer_VT(
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
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        ema_kwargs=model_cfg.ema,
    )
    if experiment_seed is not None:
        rank_seed = experiment_seed + trainer.accelerator.process_index
        set_seed(rank_seed)
        if trainer.accelerator.is_main_process:
            print(f"Global experiment seed={experiment_seed}; training RNG uses seed + process_index on each rank")

    train_dataset = load_dataset_mel(
        model_cfg.datasets.name,
        tokenizer,
        mel_spec_module=MelSpec_tacotron(**model_cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in model_cfg.model.mel_spec.items() if k != "mel_spec_type"},  # hack
        dataset_type="CustomDataset_mel_video",
        data_dir=data_dir,
    )
    init_mode = str(getattr(model_cfg.ckpts, "init_mode", "audio_pretrained"))
    pretrained_path = getattr(model_cfg.ckpts, "pretrained_path", None)
    train_kwargs = {
        "num_workers": model_cfg.datasets.num_workers,
        # Preserve historical C0-C3 ordering when no explicit global seed is
        # configured. Seeded experiments record and reuse their experiment seed here.
        "resumable_with_seed": experiment_seed if experiment_seed is not None else 666,
    }

    if init_mode == "scratch":
        if pretrained_path not in (None, ""):
            raise ValueError("ckpts.pretrained_path must be null when ckpts.init_mode=scratch")
        if trainer.accelerator.is_main_process:
            print("Initialization mode: scratch (no pretrained AlignDiT checkpoint will be loaded)")
        trainer.train(train_dataset, **train_kwargs)
    elif init_mode == "audio_pretrained":
        if not pretrained_path:
            raise ValueError("ckpts.pretrained_path is required when ckpts.init_mode=audio_pretrained")
        if trainer.accelerator.is_main_process:
            print(f"Initialization mode: audio_pretrained ({pretrained_path})")
        trainer.finetune(pretrained_path, train_dataset, **train_kwargs)
    else:
        raise ValueError(f"Unsupported ckpts.init_mode: {init_mode!r}")


if __name__ == "__main__":
    main()
