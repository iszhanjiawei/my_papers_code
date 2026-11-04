"""Train an architecture-compatible LibriTTS text/audio MM-DiT checkpoint."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from aligndit.model.cfm_text_audio import CFMTextAudio
from aligndit.model.libritts_text_audio_dataset import LibriTTSTextAudioDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.text_audio_checkpoint import initialize_from_s2c
from aligndit.model.trainer_text_audio import TextAudioTrainer, canonical_sha256, make_accelerator


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    # aligndit is a namespace package: str(files(...)) may be MultiplexedPath(...).
    package = Path(__file__).resolve().parents[2]
    sources = list((package / "model").glob("*.py"))
    sources += list((package / "model" / "backbone").glob("*.py"))
    sources += [Path(__file__)]
    project_src = package.parent
    sources += list((project_src / "f5_tts" / "model").glob("*.py"))
    return {str(path.relative_to(project_src)): file_sha256(path) for path in sorted(set(sources))}


def build_contract(cfg, train_dataset, validation_dataset):
    data = dict(train_dataset.contract)
    data.update({"cache_contract_sha256": canonical_sha256(train_dataset.contract),
                 "normalization_sha256": file_sha256(train_dataset.normalization_path),
                 "vocab_sha256": file_sha256(train_dataset.vocab_path),
                 "train_records": len(train_dataset), "validation_records": len(validation_dataset),
                 "validation_contract_sha256": canonical_sha256(validation_dataset.contract)})
    if not data.get("codec_sha256") or not data.get("speaker_checkpoint_sha256"):
        raise ValueError("The feature cache must record codec and speaker checkpoint SHA-256")
    optimization = OmegaConf.to_container(cfg.optim, resolve=True)
    # The stop boundary does not alter the LR horizon or optimizer state.
    optimization.pop("stop_after_updates", None)
    return {
        "contract_schema_version": 1, "stage": "libritts_text_audio",
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": data,
        "initialization": {"type": "s2c_ema", "parent_sha256": str(cfg.ckpts.expected_parent_sha256),
                           "parent_update": int(cfg.ckpts.expected_parent_update),
                           "parent_contract_sha256": str(cfg.ckpts.expected_parent_contract_sha256)},
        "optimization": optimization,
        "batching": {"frame_budget_per_rank": int(cfg.datasets.batch_size_per_gpu),
                     "max_samples_per_rank": int(cfg.datasets.max_samples), "tail_policy": "repeat_batches_no_drop",
                     "overflow_policy": "singleton"},
        "ema": OmegaConf.to_container(cfg.ema, resolve=True),
        "seed": int(cfg.seed), "precision": str(cfg.runtime.mixed_precision),
        "validation": OmegaConf.to_container(cfg.validation, resolve=True),
        "source_sha256": source_hashes(), "torch_version": str(torch.__version__),
    }


@hydra.main(version_base="1.3", config_path=str(files("aligndit").joinpath("config")),
            config_name="pretrain_libritts_text_audio")
def main(cfg):
    set_seed(int(cfg.seed))
    accelerator = make_accelerator(cfg)
    if accelerator.num_processes != int(cfg.runtime.expected_world_size):
        raise ValueError("World size does not match runtime.expected_world_size; use the launcher or override explicitly")
    arch = cfg.model.arch
    if (int(arch.depth), int(arch.n_mm_layers), int(arch.n_text_layers), int(arch.pe_attn_head),
        int(arch.speaker_condition_start_layer)) != (18, 12, 12, 1, 12):
        raise ValueError("This pretraining recipe requires 12 AT + 6 A blocks, PE head 1 and speaker in blocks 12..17")
    if not arch.text_audio_only or not arch.text_joint_attention or not arch.text_mask_padding:
        raise ValueError("Text/audio pretraining requires real AT joint attention and masked text padding")
    if arch.get("temporal_band_enabled", False):
        raise ValueError("Audio/video temporal priors are inactive in text/audio pretraining")
    if not bool(cfg.datasets.require_complete):
        raise ValueError("Training requires an audited complete feature cache; partial caches are not allowed")
    train = LibriTTSTextAudioDataset(str(cfg.datasets.cache_root), split="train", require_complete=True)
    validation = LibriTTSTextAudioDataset(str(cfg.datasets.cache_root), split="val", require_complete=True)
    if len(train) < 1 or len(validation) < 1:
        raise ValueError("Training and validation splits must both contain examples")
    if train.vocab_char_map != validation.vocab_char_map:
        raise ValueError("Train/validation vocabularies differ")
    contract = build_contract(cfg, train, validation)
    audio = cfg.model.audio_representation
    cls = hydra.utils.get_class(f"aligndit.model.{cfg.model.backbone}")
    model = CFMTextAudio(
        transformer=cls(**arch, text_num_embeds=int(train.vocab_size), mel_dim=int(audio.channels)),
        mel_spec_module=PrecomputedAudioRepresentation(int(audio.channels), int(audio.sample_rate), int(audio.hop_length)),
        num_channels=int(audio.channels), vocab_char_map=train.vocab_char_map,
        audio_drop_prob=float(cfg.model.audio_drop_prob), cond_drop_prob=float(cfg.model.cond_drop_prob),
        text_drop_prob=float(cfg.model.text_drop_prob), ctc_lambda=float(cfg.model.ctc_lambda),
        frac_lengths_mask=tuple(cfg.model.frac_lengths_mask),
    )
    if cfg.ckpts.get("resume"):
        # Full optimizer/EMA/RNG restoration happens in the trainer. Parent weights
        # are deliberately not reloaded when resuming this already initialized run.
        report = {"mode": "resume", "checkpoint": str(cfg.ckpts.resume)}
    else:
        report = initialize_from_s2c(model, cfg)
    trainer = TextAudioTrainer(model, cfg, contract, accelerator=accelerator, initialization_report=report)
    if accelerator.is_main_process:
        print(json.dumps({"stage": "libritts_text_audio", "train_records": len(train),
                          "validation_records": len(validation), "world_size": accelerator.num_processes,
                          "contract_sha256": canonical_sha256(contract), "output": str(cfg.ckpts.save_dir)}, indent=2), flush=True)
    trainer.train(train, validation)


if __name__ == "__main__":
    main()
