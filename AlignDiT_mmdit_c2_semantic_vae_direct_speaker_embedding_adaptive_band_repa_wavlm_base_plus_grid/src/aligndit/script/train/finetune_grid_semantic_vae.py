"""Fine-tune the unchanged MM-DiT/CAM++/adaptive-band/REPA architecture on GRID."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.grid_semantic_vae_dataset import GridSemanticVaeDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.trainer_grid_semantic_vae import GridSemanticVaeTrainer
from f5_tts.model.utils import get_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def build_dataset(config, *, split="train"):
    values = OmegaConf.to_container(config.datasets, resolve=True)
    parameters = inspect.signature(GridSemanticVaeDataset).parameters
    kwargs = {key: value for key, value in values.items() if key in parameters}
    kwargs["split"] = split
    return GridSemanticVaeDataset(**kwargs)


def build_model(config):
    """Shared with the real-data smoke entry; all network settings are inherited."""
    arc = config.model.arch
    if (
        int(arc.depth) != 18
        or int(arc.n_mm_layers) != 12
        or int(arc.n_text_layers) != 12
        or int(arc.speaker_dim) != 192
        or int(arc.speaker_condition_start_layer) != 12
        or not bool(arc.temporal_band_enabled)
        or int(arc.audio_video_ratio) != 1
        or float(arc.temporal_band_audio_fps) != 40
        or float(arc.temporal_band_video_fps) != 40
        or int(arc.repa_layer) != 9
        or int(arc.repa_target_dim) != 768
        or int(arc.repa_projector_dim) != 2048
        or float(config.model.repa_lambda) != 0.1
    ):
        raise ValueError("GRID must retain the requested 18-layer C2 speaker/adaptive-band/WavLM-Base+ architecture")
    vocabulary, size = get_tokenizer(config.datasets.vocab_path, "custom")
    audio = config.model.audio_representation
    if (audio.channels, audio.frame_rate, audio.sample_rate, audio.hop_length) != (64, 40, 16000, 400):
        raise ValueError("GRID requires the unchanged 64D/40-Hz Semantic-VAE representation")
    model_cls = hydra.utils.get_class(f"aligndit.model.{config.model.backbone}")
    return CFM_VT(
        transformer=model_cls(**arc, text_num_embeds=size, mel_dim=audio.channels),
        mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
        num_channels=audio.channels,
        vocab_char_map=vocabulary,
        audio_video_ratio=arc.audio_video_ratio,
        ctc_lambda=config.model.ctc_lambda,
        repa_lambda=config.model.repa_lambda,
    )


@hydra.main(version_base="1.3", config_path="../../config", config_name="finetune_grid_mmdit")
def main(config):
    os.chdir(PROJECT_ROOT)
    if config.datasets.name != "GRID" or "grid" not in str(config.ckpts.save_dir).lower():
        raise ValueError("GRID training requires dedicated dataset identity and checkpoint directory")
    if config.ckpts.log_samples:
        raise ValueError("The inherited mel logger cannot decode Semantic-VAE latents")
    if (
        float(config.optim.learning_rate) != 5e-5
        or float(config.model.ctc_lambda) != 0.03
        or int(config.model.ctc_warmup_start) != 10000
        or int(config.model.ctc_warmup_end) != 30000
    ):
        raise ValueError("GRID preserves parent LR=5e-5 and CTC 0->0.03 from update 10k to 30k")
    set_seed(int(config.seed))
    dataset = build_dataset(config)
    if dataset.is_subset and "smoke" not in str(config.ckpts.save_dir).lower():
        raise ValueError("Diagnostic subsets require a separate checkpoint directory containing 'smoke'")
    model = build_model(config)
    resolved = OmegaConf.to_container(config, resolve=True)
    # A shorter stopping point is resumable within the same immutable 100k
    # schedule. Changing batch size, world size, data, seed or horizon is not.
    resolved["optim"].pop("run_until_update", None)
    resolved["optim"].pop("epochs", None)
    exp_name = f"{config.model.name}_{config.model.audio_representation.name}_GRID_{config.model.tokenizer}"
    contract = {
        "schema_version": 1,
        "dataset_contract_sha256": dataset.contract_sha256,
        "config": resolved,
        "project_dir": str(PROJECT_ROOT),
        "tensorboard_logdir": str(PROJECT_ROOT / "runs" / exp_name),
        "initialization": "strict S2c 70k EMA, fresh optimizer/update; parent EMA step semantics preserved",
        "scheduler": "linear 20k warmup then decay to explicit total optimizer updates",
        "distributed_batches": "repeat at most world_size-1 shuffled batches per epoch, preserving all GRID samples",
    }
    ckpts, optim, datasets = config.ckpts, config.optim, config.datasets
    trainer = GridSemanticVaeTrainer(
        model,
        total_updates=optim.total_updates,
        run_until_update=optim.run_until_update,
        training_contract=contract,
        epochs=optim.get("epochs", 1000),
        learning_rate=optim.learning_rate,
        num_warmup_updates=optim.num_warmup_updates,
        save_per_updates=ckpts.save_per_updates,
        keep_last_n_checkpoints=ckpts.keep_last_n_checkpoints,
        checkpoint_path=ckpts.save_dir,
        batch_size_per_gpu=datasets.batch_size_per_gpu,
        batch_size_type=datasets.batch_size_type,
        max_samples=datasets.max_samples,
        grad_accumulation_steps=optim.grad_accumulation_steps,
        max_grad_norm=optim.max_grad_norm,
        logger=ckpts.logger,
        wandb_project="AlignDiT",
        wandb_run_name=exp_name,
        last_per_updates=ckpts.last_per_updates,
        log_samples=False,
        bnb_optimizer=optim.bnb_optimizer,
        mel_spec_type=config.model.audio_representation.name,
        model_cfg_dict=OmegaConf.to_container(config, resolve=True),
        ema_kwargs=config.ema,
        parent_contract_path=ckpts.parent_contract_path,
        expected_parent_sha256=ckpts.expected_parent_sha256,
        expected_parent_size=ckpts.expected_parent_size,
        expected_parent_contract_sha256=ckpts.expected_parent_contract_sha256,
        expected_parent_update=ckpts.expected_parent_update,
        ctc_target_lambda=config.model.ctc_lambda,
        ctc_warmup_start=config.model.ctc_warmup_start,
        ctc_warmup_end=config.model.ctc_warmup_end,
    )
    set_seed(int(config.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        print(
            f"Validated GRID {'diagnostic subset' if dataset.is_subset else 'full train split'}: "
            f"records={len(dataset)}, CTC feasible={dataset.ctc_feasible_count}, "
            f"horizon={optim.total_updates}, stop={optim.run_until_update}, "
            f"numbered checkpoints every {ckpts.save_per_updates}, TensorBoard={contract['tensorboard_logdir']}",
            flush=True,
        )
    trainer.finetune(
        ckpts.pretrained_path, dataset, num_workers=datasets.num_workers, resumable_with_seed=int(config.seed)
    )


if __name__ == "__main__":
    main()
