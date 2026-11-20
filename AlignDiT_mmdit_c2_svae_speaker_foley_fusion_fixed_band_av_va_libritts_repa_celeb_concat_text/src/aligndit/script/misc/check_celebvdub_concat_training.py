"""Validate a single-utterance CelebV-Dub training run without initializing a GPU or modifying its checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.libritts_pretrained_migration import validate_libritts_checkpoint_artifact
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import dataset_from_config, training_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="finetune_celebvdub_concat_text", overrides=overrides)
    dataset = dataset_from_config(cfg)
    schedule = training_schedule(cfg, dataset, args.num_processes)
    ckpt = cfg.ckpts
    validate_libritts_checkpoint_artifact(ckpt.pretrained_path, cfg)
    # Exercise the same item loader and collation path used inside DDP.
    batch = dataset.collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
    if "prompt_lens" in batch or "reference_text" in batch:
        raise ValueError("Expected one transcript and one utterance per training item")
    if not torch.equal(batch["mel_lengths"], batch["video_lengths"]):
        raise ValueError("Audio/video geometry mismatch")
    if not all(batch["text"]) or not torch.all(batch["mel_lengths"] > 0):
        raise ValueError("Missing text or audio in training batch")

    destination = Path(ckpt.save_dir)
    contract_path = destination / "speaker_training_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        if (previous.get("config") != OmegaConf.to_container(cfg, resolve=True)
                or previous.get("batch_and_schedule") != schedule):
            raise ValueError(f"Existing run has a different configuration or DDP schedule: {destination}")
    elif any(destination.glob("model_*.pt")):
        raise ValueError(f"Existing checkpoints have no training contract: {destination}")

    name = f"{cfg.model.name}_{cfg.model.audio_representation.name}_{cfg.datasets.name}_{cfg.model.tokenizer}"
    print(json.dumps({
        "result": "PASS", "training_launched": False, "training_utterances": len(dataset),
        "batch_and_schedule": schedule,
        "sample_utterance_frames": batch["mel_lengths"].tolist(),
        "sample_speaker_shape": list(batch["speaker_embedding"].shape),
        "sample_teacher_shape": list(batch["repa_features"].shape) if "repa_features" in batch else None,
        "checkpoint_dir": str(destination), "tensorboard_logdir": str((Path("runs") / name).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
