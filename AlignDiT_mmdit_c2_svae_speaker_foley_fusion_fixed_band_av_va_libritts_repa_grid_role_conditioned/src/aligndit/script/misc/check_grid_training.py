"""Check the exact GRID launch configuration without constructing a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from aligndit.model.semantic_vae_direct_migration import validate_parent_artifacts
from aligndit.script.train.finetune_grid import dataset_from_config, training_schedule_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    if args.num_processes < 1 or args.num_processes > torch.cuda.device_count():
        raise ValueError(f"Requested {args.num_processes} workers; visible CUDA devices={torch.cuda.device_count()}")
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[2] / "config")):
        config = compose(config_name="finetune_grid", overrides=overrides)
    if config.ckpts.logger != "tensorboard":
        raise ValueError("GRID training requires ckpts.logger=tensorboard for loss curves")
    dataset = dataset_from_config(config)
    if dataset.split != "train":
        raise ValueError("GRID training must use train.jsonl; val is held out")
    schedule = training_schedule_from_config(config, dataset, world_size=args.num_processes)
    ckpt = config.ckpts
    validate_parent_artifacts(
        ckpt.pretrained_path, ckpt.parent_contract_path,
        expected_checkpoint_sha256=ckpt.expected_parent_sha256,
        expected_checkpoint_size=ckpt.expected_parent_size,
        expected_contract_sha256=ckpt.expected_parent_contract_sha256,
    )
    # Decode a real cached batch, checking lengths, dimensions, finite values
    # and speaker normalization before committing GPU memory to the full model.
    batch = dataset.collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
    run_name = f"{config.model.name}_{config.model.audio_representation.name}_{config.datasets.name}_{config.model.tokenizer}"
    print(json.dumps({
        "result": "PASS", "records": len(dataset), "schedule": schedule,
        "batch_latent_shape": list(batch["mel"].shape),
        "batch_video_shape": list(batch["video"].shape),
        "checkpoint_dir": str(ckpt.save_dir), "run_name": run_name,
        "tensorboard_logdir": str((Path("runs") / run_name).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
