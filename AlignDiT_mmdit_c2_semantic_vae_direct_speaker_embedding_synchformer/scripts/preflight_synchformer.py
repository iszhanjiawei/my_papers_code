#!/usr/bin/env python
"""Validate the complete training contract without creating CUDA models or a run.

Use --audit-cache to read all training feature arrays as well as completion
metadata. Hydra overrides after -- use the same syntax as the train launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.semantic_vae_direct_migration import validate_parent_artifacts


CONFIG_NAME = "finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_synchformer"


def validate(config, *, audit_cache: bool = False) -> dict:
    arch, data, ckpts = config.model.arch, config.datasets, config.ckpts
    if int(arch.get("sync_dim", 0)) != 768 or int(data.synchformer_dim) != 768:
        raise ValueError("This entry requires model.arch.sync_dim=datasets.synchformer_dim=768")
    if int(arch.speaker_dim) != 192 or int(arch.speaker_condition_start_layer) != 12:
        raise ValueError("The CAM++ speaker architecture must match the parent experiment")
    if ckpts.logger != "tensorboard" or ckpts.log_samples:
        raise ValueError("Synchformer training requires TensorBoard and log_samples=false")
    if float(config.optim.learning_rate) != 5e-5:
        raise ValueError("Keep the parent experiment learning rate 5e-5")
    if int(config.model.ctc_warmup_start) != 10000 or int(config.model.ctc_warmup_end) != 30000:
        raise ValueError("Keep the parent experiment CTC warmup 10k..30k")
    dataset = SemanticVaeCelebVDubDataset(
        manifest_path=data.manifest_path,
        cache_root=data.cache_root,
        normalization_path=data.normalization_path,
        vocab_path=data.vocab_path,
        expected_manifest_sha256=data.expected_manifest_sha256,
        expected_inventory_sha256=data.expected_inventory_sha256,
        expected_normalization_sha256=data.expected_normalization_sha256,
        expected_vocab_sha256=data.expected_vocab_sha256,
        expected_record_count=data.expected_record_count,
        speaker_embedding_cache_dir=data.speaker_embedding_cache_dir,
        speaker_embedding_dim=data.speaker_embedding_dim,
        speaker_embedding_model_id=data.speaker_embedding_model_id,
        speaker_embedding_checkpoint_sha256=data.speaker_embedding_checkpoint_sha256,
        synchformer_cache_dir=data.synchformer_cache_dir,
        synchformer_dim=data.synchformer_dim,
        synchformer_checkpoint_sha256=data.synchformer_checkpoint_sha256,
    )
    validate_parent_artifacts(
        ckpts.pretrained_path,
        ckpts.parent_contract_path,
        expected_checkpoint_sha256=ckpts.expected_parent_sha256,
        expected_checkpoint_size=ckpts.expected_parent_size,
        expected_contract_sha256=ckpts.expected_parent_contract_sha256,
    )
    audit = None
    if audit_cache:
        audit = {
            "speaker": dataset.audit_speaker_embedding_cache(),
            "synchformer": dataset.audit_synchformer_cache(),
        }
    # At least one real collated sample exercises latent/video/speaker/sync IO.
    sample = dataset.collate_fn([dataset[0]])
    exp_name = (
        f"{config.model.name}_{config.model.audio_representation.name}_"
        f"{data.name}_{config.model.tokenizer}"
    )
    return {
        "ready": True,
        "train_count": len(dataset),
        "sync_contract": dataset.synchformer_contract,
        "parent_checkpoint": str(ckpts.pretrained_path),
        "checkpoint_dir": str(ckpts.save_dir),
        "tensorboard_logdir": str(Path(__file__).resolve().parents[1] / "runs" / exp_name),
        "sample_shapes": {key: list(value.shape) for key, value in sample.items() if hasattr(value, "shape")},
        "full_array_audit": audit,
        "config": OmegaConf.to_container(config, resolve=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-cache", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[1] / "src/aligndit/config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=CONFIG_NAME, overrides=args.overrides)
    try:
        result = validate(config, audit_cache=args.audit_cache)
    except Exception as error:
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"ready": False, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
