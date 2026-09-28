"""Train the isolated AV-HuBERT audio-teacher InfoNCE experiment."""

from __future__ import annotations

import copy
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
from aligndit.model.trainer_semantic_vae_direct_speaker_avhubert_infonce import (
    SemanticVaeDirectC2SpeakerAVHuBERTInfoNCETrainer,
)
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("aligndit").joinpath("../..")))


def training_identity(config: dict, teacher_contract: dict) -> dict:
    """Keep model/data/optimizer semantics fixed while allowing a later run cap.

    Runtime log locations, checkpoint frequency, workers and requested stopping
    update do not change the inherited 200-epoch optimization horizon.
    """
    identity = copy.deepcopy(config)
    identity["optim"].pop("run_until_update", None)
    identity["datasets"].pop("num_workers", None)
    for key in ("save_dir", "save_per_updates", "last_per_updates", "keep_last_n_checkpoints", "logger", "log_samples"):
        identity["ckpts"].pop(key, None)
    return {
        "schema_version": 1,
        "configuration": identity,
        "audio_teacher_contract": teacher_contract,
        "teacher_timeline": "native 25 Hz; student at (j + 0.5) * 1.6 - 0.5 on the 40-Hz grid",
        "student": "block 11 raw text-CA output, before residual/gate, Linear(768,1024)",
        "reduction": "valid-anchor mean per rank, then ordinary DDP rank mean; no cross-clip or cross-rank negatives",
        "loss_mask": "generation region and valid teacher prefix; skip text/video-dropped branches",
        "warmup": "target * min(completed_optimizer_updates / 10000, 1); first forward has zero weight",
    }


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")), config_name=None)
def main(model_cfg):
    set_seed(int(model_cfg.seed))
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    audio_cfg = model_cfg.model.audio_representation
    if model_cfg.ckpts.log_samples:
        raise ValueError("Use the isolated Semantic-VAE inference entry for samples")
    if model_cfg.ckpts.logger != "tensorboard":
        raise ValueError("This experiment requires TensorBoard recording")
    if int(model_arc.speaker_dim) != 192 or int(model_cfg.datasets.speaker_embedding_dim) != 192:
        raise ValueError("CAM++ conditioning must use 192 dimensions")
    if int(model_arc.speaker_condition_start_layer) != 12:
        raise ValueError("Speaker conditioning must start at zero-based block 12")
    if int(model_arc.context_alignment_layer) != 11 or int(model_arc.context_alignment_dim) != 1024:
        raise ValueError("InfoNCE requires the raw text-CA context at zero-based block 11 and a 1024-D teacher")
    if float(model_cfg.optim.learning_rate) != 5e-5:
        raise ValueError("The paired baseline comparison retains learning_rate=5e-5")
    if int(model_cfg.model.ctc_warmup_start) != 10000 or int(model_cfg.model.ctc_warmup_end) != 30000:
        raise ValueError("The inherited CTC schedule must retain the 10k/30k warmup boundaries")
    if float(model_cfg.model.ctc_lambda) != 0.03:
        raise ValueError("The inherited CTC target must remain 0.03")
    if int(model_cfg.model.infonce_warmup_updates) != 10000:
        raise ValueError("InfoNCE warmup must last 10000 completed optimizer updates")
    if not math.isclose(float(model_cfg.model.infonce_lambda), 0.05) or not math.isclose(
        float(model_cfg.model.infonce_temperature), 0.07
    ):
        raise ValueError("This InfoNCE experiment fixes target weight=0.05 and temperature=0.07")
    if int(model_cfg.model.infonce_min_negative_frames) != 5:
        raise ValueError("InfoNCE negatives must be at least five native 25-Hz teacher frames away")
    if int(model_cfg.optim.run_until_update) <= 0:
        raise ValueError("run_until_update must be positive")

    speaker_metadata = validate_speaker_cache_metadata(
        model_cfg.datasets.speaker_embedding_cache_dir,
        expected_dim=192,
        model_id=model_cfg.datasets.speaker_embedding_model_id,
        checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
    )
    # Check cache identity before allocating DDP. Full coverage is audited once
    # before launch; each consumed teacher file is validated by the dataset.
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
        speaker_embedding_dim=192,
        speaker_embedding_model_id=model_cfg.datasets.speaker_embedding_model_id,
        speaker_embedding_checkpoint_sha256=model_cfg.datasets.speaker_embedding_checkpoint_sha256,
        audio_teacher_cache_dir=model_cfg.datasets.audio_teacher_cache_dir,
        audio_teacher_audio_root=model_cfg.datasets.audio_teacher_audio_root,
        audio_teacher_expected_identity=model_cfg.datasets.audio_teacher_expected_identity,
    )
    resolved_config = OmegaConf.to_container(model_cfg, resolve=True)
    identity = training_identity(resolved_config, train_dataset.audio_teacher_contract)
    vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.vocab_path, "custom")
    exp_name = f"{model_cfg.model.name}_{audio_cfg.name}_{model_cfg.datasets.name}_{model_cfg.model.tokenizer}"
    model = CFM_VT(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=audio_cfg.channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=audio_cfg.channels, target_sample_rate=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length
        ),
        num_channels=audio_cfg.channels,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=model_arc.audio_video_ratio,
        ctc_lambda=model_cfg.model.ctc_lambda,
        infonce_lambda=model_cfg.model.infonce_lambda,
        infonce_temperature=model_cfg.model.infonce_temperature,
        infonce_min_negative_frames=model_cfg.model.infonce_min_negative_frames,
    )
    trainer = SemanticVaeDirectC2SpeakerAVHuBERTInfoNCETrainer(
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
        log_samples=False,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=audio_cfg.name,
        is_local_vocoder=False,
        local_vocoder_path="",
        model_cfg_dict=resolved_config,
        ema_kwargs=model_cfg.ema,
        parent_contract_path=model_cfg.ckpts.parent_contract_path,
        expected_parent_sha256=model_cfg.ckpts.expected_parent_sha256,
        expected_parent_size=model_cfg.ckpts.expected_parent_size,
        expected_parent_contract_sha256=model_cfg.ckpts.expected_parent_contract_sha256,
        expected_parent_update=model_cfg.ckpts.expected_parent_update,
        ctc_target_lambda=model_cfg.model.ctc_lambda,
        ctc_warmup_start=model_cfg.model.ctc_warmup_start,
        ctc_warmup_end=model_cfg.model.ctc_warmup_end,
        infonce_target_lambda=model_cfg.model.infonce_lambda,
        infonce_warmup_updates=model_cfg.model.infonce_warmup_updates,
        infonce_training_contract=identity,
    )
    trainer.run_until_update = int(model_cfg.optim.run_until_update)
    set_seed(int(model_cfg.seed) + trainer.accelerator.process_index)
    if trainer.is_main:
        save_dir = Path(model_cfg.ckpts.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "experiment": str(model_cfg.model.name),
            "project_dir": str(Path.cwd()),
            "config": resolved_config,
            "training_identity": identity,
            "speaker_cache_metadata": speaker_metadata,
            "initialization": "same S2c 70k EMA parent as the speaker baseline; new optimizer/update counter",
            "tensorboard_logdir": str(Path("runs", exp_name).resolve()),
        }
        contract_path = save_dir / "avhubert_infonce_training_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
            if previous.get("training_identity") != identity:
                raise RuntimeError(f"Refusing to reuse a checkpoint directory with different training semantics: {save_dir}")
        else:
            contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
        print(
            f"AV-HuBERT InfoNCE dataset validated: records={len(train_dataset)}, "
            f"CTC feasible={train_dataset.ctc_feasible_count}; "
            "teacher=audio-only/25Hz/1024D, tap=block11 raw text-CA, tau=0.07, gap=5 frames; "
            f"InfoNCE 0->0.05 over 10k completed updates; stop={trainer.run_until_update}; "
            f"TensorBoard={contract['tensorboard_logdir']}",
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
