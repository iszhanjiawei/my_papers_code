"""Validate the fixed13 LibriTTS CelebV-Dub REPA launch before allocating DDP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
import torch

from aligndit.model.fixed_temporal_band_contract import validate_fixed_band_training_config
from aligndit.model.libritts_pretrained_migration import validate_libritts_checkpoint_artifact
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import (
    dataset_from_config, make_repa_contract, validate_repa_training_config,
)

PROJECT = Path(__file__).resolve().parents[1]


def audit_teacher(dataset):
    """Read every teacher feature; reject missing, malformed or nonfinite targets."""
    def inspect(record):
        feature = dataset._load_repa_feature_array(record)
        return feature.shape[0]
    start = time.monotonic()
    total_frames = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, frames in enumerate(pool.map(inspect, dataset.records), 1):
            total_frames += frames
            if count % 10000 == 0:
                print(f"REPA cache verified {count}/{len(dataset)}", flush=True)
    report = dict(complete=True, valid=count, total_frames=total_frames,
                  elapsed_seconds=time.monotonic() - start,
                  contract=dataset.repa_feature_contract)
    coverage = json.loads((dataset.repa_feature_cache_dir / "coverage_report.json").read_text())
    if total_frames != coverage["total_frames"]:
        raise ValueError("Audited teacher-frame total differs from extraction coverage")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--audit-teacher", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    if args.num_processes != 4 or torch.cuda.device_count() < args.num_processes:
        raise ValueError("The unchanged CelebV-Dub recipe requires four visible GPUs")
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        cfg = compose(config_name="finetune_celebvdub_repa", overrides=overrides)
    if cfg.ckpts.logger != "tensorboard":
        raise ValueError("TensorBoard loss recording is required")
    validate_fixed_band_training_config(cfg)
    validate_repa_training_config(cfg)
    dataset = dataset_from_config(cfg)
    parent = validate_libritts_checkpoint_artifact(cfg.ckpts.pretrained_path, cfg)
    batch = dataset.collate_fn([dataset[i] for i in (0, len(dataset) // 2, len(dataset) - 1)])
    run_name = f"{cfg.model.name}_{cfg.model.audio_representation.name}_{cfg.datasets.name}_{cfg.model.tokenizer}"
    report = dict(result="PASS", records=len(dataset), parent=parent,
                  repa=make_repa_contract(cfg, dataset),
                  batch_latent_shape=list(batch["mel"].shape),
                  batch_teacher_shape=list(batch["repa_features"].shape),
                  checkpoint_dir=str(cfg.ckpts.save_dir), run_name=run_name,
                  tensorboard_logdir=str(PROJECT / "runs" / run_name),
                  epochs=int(cfg.optim.epochs), run_until_update=int(cfg.optim.run_until_update))
    if args.audit_teacher:
        report["teacher_audit"] = audit_teacher(dataset)
        (PROJECT / "logs").mkdir(exist_ok=True)
        (PROJECT / "logs/celeb_repa_teacher_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
