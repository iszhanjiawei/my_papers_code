# training script.

import json
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
    speaker_dim = getattr(model_arc, "speaker_dim", None)
    speaker_embedding_cache_dir = getattr(model_cfg.datasets, "speaker_embedding_cache_dir", None)
    speaker_embedding_dim = int(getattr(model_cfg.datasets, "speaker_embedding_dim", 192))
    if speaker_dim is not None:
        if int(speaker_dim) != speaker_embedding_dim:
            raise ValueError(
                f"model speaker_dim={speaker_dim} does not match dataset speaker_embedding_dim="
                f"{speaker_embedding_dim}"
            )
        if speaker_embedding_cache_dir is None:
            raise ValueError("speaker_embedding_cache_dir is required when model.arch.speaker_dim is configured")
        metadata_path = os.path.join(speaker_embedding_cache_dir, "metadata.json")
        with open(metadata_path, encoding="utf-8") as file:
            cache_metadata = json.load(file)
        if cache_metadata.get("status") != "complete":
            raise RuntimeError(
                f"speaker cache is not complete according to {metadata_path}: "
                f"status={cache_metadata.get('status')!r}"
            )
        expected_cache_contract = {
            "output_shape": [speaker_embedding_dim],
            "source_audio": "complete_unmasked_waveform",
            "model_id": getattr(model_cfg.datasets, "speaker_embedding_model_id", None),
            "checkpoint_sha256": getattr(
                model_cfg.datasets,
                "speaker_embedding_checkpoint_sha256",
                None,
            ),
        }
        for key, expected_value in expected_cache_contract.items():
            if expected_value is not None and cache_metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"speaker cache contract mismatch in {metadata_path}: {key}="
                    f"{cache_metadata.get(key)!r}, expected {expected_value!r}"
                )

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    data_dir = getattr(model_cfg.datasets, 'data_dir', None)
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
            print(
                f"Global experiment seed={experiment_seed}; "
                "training RNG uses seed + process_index on each rank"
            )

    train_dataset = load_dataset_mel(
        model_cfg.datasets.name,
        tokenizer,
        mel_spec_module=MelSpec_tacotron(**model_cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in model_cfg.model.mel_spec.items() if k != "mel_spec_type"},  # hack
        dataset_type="CustomDataset_mel_video",
        data_dir=data_dir,
        speaker_embedding_cache_dir=speaker_embedding_cache_dir,
        speaker_embedding_dim=speaker_embedding_dim,
    )
    dataset_limit = getattr(model_cfg.datasets, "limit", None)
    if dataset_limit is not None:
        dataset_limit = int(dataset_limit)
        if dataset_limit <= 0:
            raise ValueError(f"datasets.limit must be positive, got {dataset_limit}")
        train_dataset.data = train_dataset.data.select(range(min(dataset_limit, len(train_dataset.data))))
        train_dataset.durations = train_dataset.data["duration"]
        print(f"Using a gated smoke-test subset with {len(train_dataset)} samples")
    trainer.finetune(
        model_cfg.ckpts.pretrained_path,
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        # Preserve historical C0-C3 ordering when no explicit global seed is
        # configured. D0 records and reuses its experiment seed here.
        resumable_with_seed=experiment_seed if experiment_seed is not None else 666,
    )


if __name__ == "__main__":
    main()
