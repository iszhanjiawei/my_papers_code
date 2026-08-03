"""From-scratch 40 Hz Semantic-VAE audio pretraining on LibriSpeech."""

import os
import re
from importlib.resources import files
from pathlib import Path

import accelerate
import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_notext import CFM_notext
from aligndit.model.dataset import SemanticVaePretrainDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.trainer_notext import Trainer_notext
from aligndit.script.misc.svae_cache_utils import atomic_write_json, safe_join, sha256_file


PROJECT_ROOT = Path(str(files("aligndit").joinpath("../.."))).resolve()
os.chdir(PROJECT_ROOT)


def publish_training_contract(trainer: Trainer_notext, model_cfg, dataset: SemanticVaePretrainDataset) -> None:
    """Create an immutable resume contract next to checkpoints before update zero."""

    cache_root = dataset.cache_root
    checkpoint_dir = Path(model_cfg.ckpts.save_dir)
    contract_path = checkpoint_dir / "training_contract.json"
    if checkpoint_dir.exists():
        checkpoint_files = sorted(
            path.name for path in checkpoint_dir.iterdir() if path.is_file() and path.suffix in {".pt", ".safetensors"}
        )
    else:
        checkpoint_files = []
    unexpected_checkpoints = [
        name for name in checkpoint_files if name != "model_last.pt" and re.fullmatch(r"model_[0-9]+\.pt", name) is None
    ]
    if unexpected_checkpoints:
        raise RuntimeError(
            "From-scratch Semantic-VAE pretraining refuses non-training checkpoints in its save directory: "
            f"{unexpected_checkpoints}"
        )
    if checkpoint_files and not contract_path.is_file():
        raise RuntimeError(
            f"Refusing to resume {checkpoint_files} without the immutable training contract: {contract_path}"
        )

    source_paths = {
        "cfm_notext": PROJECT_ROOT / "src/aligndit/model/cfm_notext.py",
        "dataset": PROJECT_ROOT / "src/aligndit/model/dataset.py",
        "dit_notext": PROJECT_ROOT / "src/aligndit/model/backbone/dit_notext.py",
        "f5_cfm": PROJECT_ROOT / "src/f5_tts/model/cfm.py",
        "f5_dataset": PROJECT_ROOT / "src/f5_tts/model/dataset.py",
        "f5_dit": PROJECT_ROOT / "src/f5_tts/model/backbones/dit.py",
        "f5_modules": PROJECT_ROOT / "src/f5_tts/model/modules.py",
        "f5_trainer": PROJECT_ROOT / "src/f5_tts/model/trainer.py",
        "modules": PROJECT_ROOT / "src/aligndit/model/modules.py",
        "pretrain_semantic_vae": Path(__file__).resolve(),
        "svae_cache_utils": PROJECT_ROOT / "src/aligndit/script/misc/svae_cache_utils.py",
        "trainer_notext": PROJECT_ROOT / "src/aligndit/model/trainer_notext.py",
    }
    contract = {
        "cache_completion": {
            "hubert_40hz": sha256_file(safe_join(cache_root, "state/hubert_40hz/complete.json")),
            "latents": sha256_file(safe_join(cache_root, "state/latents/complete.json")),
            "normalization": dataset.normalization_sha256,
        },
        "config": OmegaConf.to_container(model_cfg, resolve=True),
        "distributed_runtime": {
            "accelerate": accelerate.__version__,
            "distributed_type": str(trainer.accelerator.distributed_type),
            "mixed_precision": trainer.accelerator.mixed_precision,
            "num_processes": trainer.accelerator.num_processes,
            "torch": torch.__version__,
        },
        "schema_version": 1,
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    if trainer.accelerator.is_main_process:
        result = atomic_write_json(contract_path, contract)
        print(
            f"Training contract {'created' if result.created else 'verified'}: {result.path} sha256={result.sha256}",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg) -> None:
    experiment_seed = int(model_cfg.seed)
    set_seed(experiment_seed)

    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    representation_name = model_cfg.model.audio_representation.name
    channels = int(model_cfg.model.audio_representation.channels)
    exp_name = f"{model_cfg.model.name}_{representation_name}_{model_cfg.datasets.name}"

    model = CFM_notext(
        transformer=model_cls(**model_arc, mel_dim=channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=int(model_cfg.model.audio_representation.sample_rate),
            hop_length=int(model_cfg.model.audio_representation.hop_length),
        ),
        num_channels=channels,
        proj_lambda=model_cfg.model.proj_lambda,
    )
    trainer = Trainer_notext(
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
        mel_spec_type=representation_name,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        ema_kwargs=model_cfg.ema,
    )
    set_seed(experiment_seed + trainer.accelerator.process_index)

    train_dataset = SemanticVaePretrainDataset(
        manifest_path=model_cfg.datasets.manifest_path,
        cache_root=model_cfg.datasets.cache_root,
        normalization_path=model_cfg.datasets.normalization_path,
    )
    publish_training_contract(trainer, model_cfg, train_dataset)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=experiment_seed,
        max_updates=int(model_cfg.optim.max_updates),
        deterministic_update_seed=bool(model_cfg.optim.deterministic_update_seed),
    )


if __name__ == "__main__":
    main()
