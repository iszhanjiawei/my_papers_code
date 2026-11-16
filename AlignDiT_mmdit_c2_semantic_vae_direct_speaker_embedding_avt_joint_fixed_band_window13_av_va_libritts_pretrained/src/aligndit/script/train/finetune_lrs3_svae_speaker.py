"""Train the frozen LRS3 Semantic-VAE/AV-HuBERT/CAM++ inventory."""

import inspect
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
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.lrs3_semantic_vae_dataset import SemanticVaeLRS3Dataset
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from aligndit.model.trainer_semantic_vae_fixed_band import SemanticVaeFixedBandTrainer
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
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
    if str(model_cfg.ckpts.logger) != "tensorboard":
        raise ValueError("LRS3 training and smoke tests require TensorBoard loss logging")
    dataset_parameters = inspect.signature(SemanticVaeLRS3Dataset).parameters
    dataset_config = OmegaConf.to_container(model_cfg.datasets, resolve=True)
    train_dataset = SemanticVaeLRS3Dataset(
        **{key: value for key, value in dataset_config.items() if key in dataset_parameters}
    )
    if train_dataset.split != "train":
        raise ValueError("Training requires the LRS3 train split")

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
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        libritts_parent = model_cfg.ckpts.get("initialization_mode", "s2c") == "libritts_text_audio"
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract_config = OmegaConf.to_container(model_cfg, resolve=True)
        # Extending a run's stop limit preserves optimizer/scheduler semantics.
        # Everything affecting samples, updates or losses remains frozen.
        contract_config["optim"].pop("run_until_update", None)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": contract_config,
            "data_contract": train_dataset.data_contract,
            "initialization": (
                "complete LibriTTS text/audio/speaker 50k EMA; vocabulary rows mapped to unchanged 159-character IDs; "
                "new optimizer/update counter; carry parent EMA step"
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
            "LRS3 CAM++ speaker CTC-warmup dataset validated: "
            f"records={len(train_dataset)}, CTC feasible={train_dataset.ctc_feasible_count}, "
            f"CTC zero_infinity-only={train_dataset.ctc_infeasible_count}; "
            f"ctc_lambda=0 through update {model_cfg.model.ctc_warmup_start}, "
            f"linear to {model_cfg.model.ctc_lambda} at update {model_cfg.model.ctc_warmup_end}",
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
