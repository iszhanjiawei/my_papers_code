"""Train isolated Semantic-VAE D1 with frozen CAM++ speaker conditioning."""

from __future__ import annotations

import inspect
import json
import os
from importlib.resources import files
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.speaker_embedding import validate_speaker_cache_metadata
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectD1SpeakerTrainer
from aligndit.script.train.finetune_semantic_vae_d1_direct import (
    build_model as build_d1_model,
)
from aligndit.script.train.finetune_semantic_vae_d1_direct import (
    validate_experiment_config as validate_d1_config,
)
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def validate_experiment_config(model_cfg) -> None:
    """Preserve D1/CTC and condition all twelve of D1's audio-only blocks."""

    validate_d1_config(model_cfg)
    arch = model_cfg.model.arch
    if int(arch.speaker_dim) != 192 or int(model_cfg.datasets.speaker_embedding_dim) != 192:
        raise ValueError("CAM++ model and dataset must both use speaker_embedding_dim=192")
    if int(arch.speaker_condition_start_layer) != 6:
        raise ValueError("D1 speaker conditioning must start at zero-based block 6, covering all 12 audio-only blocks")
    if int(model_cfg.optim.run_until_update) <= 0:
        raise ValueError("run_until_update must be a positive child optimizer update count")
    if str(model_cfg.ckpts.logger) != "tensorboard":
        raise ValueError("This training entry requires live TensorBoard loss logging")


def build_model(model_cfg, vocab_char_map: dict[str, int], vocab_size: int):
    """Expose the identical validated architecture for training and smoke tests."""

    validate_experiment_config(model_cfg)
    return build_d1_model(model_cfg, vocab_char_map, vocab_size)


@hydra.main(
    version_base="1.3",
    config_path=str(files("aligndit").joinpath("config")),
    config_name="finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup",
)
def main(model_cfg):
    validate_experiment_config(model_cfg)
    speaker_metadata = validate_speaker_cache_metadata(
        model_cfg.datasets.speaker_embedding_cache_dir,
        expected_dim=int(model_cfg.datasets.speaker_embedding_dim),
        model_id=model_cfg.datasets.speaker_embedding_model_id,
        checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
    )
    experiment_seed = int(model_cfg.seed)
    set_seed(experiment_seed)
    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")
    audio_cfg = model_cfg.model.audio_representation
    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"
    model = build_model(model_cfg, vocab_char_map, vocab_size)
    trainer = SemanticVaeDirectD1SpeakerTrainer(
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
    set_seed(experiment_seed + trainer.accelerator.process_index)

    dataset_config = OmegaConf.to_container(model_cfg.datasets, resolve=True)
    dataset_parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    train_dataset = SemanticVaeCelebVDubDataset(
        **{key: value for key, value in dataset_config.items() if key in dataset_parameters}
    )
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "speaker_cache_metadata": speaker_metadata,
            "speaker_cache_contract": train_dataset.speaker_embedding_contract,
            "initialization": "same S2c 70k EMA parent as Direct-D1, fresh optimizer and child update 0",
            "speaker_condition": "L2 CAM++ -> zero-initialized bias-free Linear(192,768), audio-only blocks 6..17",
            "seed": experiment_seed,
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        contract_path = save_dir / "speaker_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text(encoding="utf-8"))
            if previous != contract:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with a different speaker contract: {save_dir}")
        else:
            if any(save_dir.glob("*.pt")) or any(save_dir.glob("*.safetensors")):
                raise RuntimeError(f"Refusing to resume existing weights without a speaker training contract: {save_dir}")
            with contract_path.open("x", encoding="utf-8") as file:
                json.dump(contract, file, indent=2, ensure_ascii=False, sort_keys=True)
                file.write("\n")
        print(
            "Semantic-VAE D1 CAM++ speaker dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"CTC=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}; "
            f"stop at child update {trainer.run_until_update}; seed={experiment_seed} + rank",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
    trainer.finetune(
        model_cfg.ckpts.pretrained_path,
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=experiment_seed,
    )


if __name__ == "__main__":
    main()
