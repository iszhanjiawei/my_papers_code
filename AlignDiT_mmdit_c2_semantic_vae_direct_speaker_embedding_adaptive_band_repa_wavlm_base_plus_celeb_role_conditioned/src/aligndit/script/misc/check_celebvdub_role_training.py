"""Validate a paired CelebV-Dub training run without initializing a GPU or modifying its checkpoints."""
import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from aligndit.model.semantic_vae_direct_migration import validate_parent_artifacts
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import dataset_from_config, paired_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="finetune_celebvdub_role_conditioned", overrides=overrides)
    if not cfg.model.arch.role_conditioned_text:
        raise ValueError("This launcher requires role-conditioned text")
    paired = dataset_from_config(cfg)
    schedule = paired_schedule(cfg, paired, args.num_processes)
    ckpt = cfg.ckpts
    validate_parent_artifacts(
        ckpt.pretrained_path, ckpt.parent_contract_path,
        expected_checkpoint_sha256=ckpt.expected_parent_sha256,
        expected_checkpoint_size=ckpt.expected_parent_size,
        expected_contract_sha256=ckpt.expected_parent_contract_sha256,
    )
    # Exercise the same item loader and collation path used inside DDP.
    batch = paired.collate_fn([paired[i] for i in range(min(2, len(paired)))])
    p = batch["prompt_lens"]
    t = batch["mel_lengths"] - p
    if not torch.all((p > 0) & (t > 0)) or not torch.equal(batch["mel_lengths"], batch["video_lengths"]):
        raise ValueError("Invalid prompt/target geometry")
    if any(torch.count_nonzero(batch["video"][i, :length]) for i, length in enumerate(p)):
        raise ValueError("Reference region must have empty video conditions")
    if len(batch["reference_text"]) != len(batch["text"]) or not all(batch["reference_text"]):
        raise ValueError("Missing transcript role in training batch")

    destination = Path(ckpt.save_dir)
    contract_path = destination / "speaker_training_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        if (previous.get("train_pairing") != paired.pairing_contract
                or previous.get("config") != OmegaConf.to_container(cfg, resolve=True)
                or previous.get("batch_and_schedule") != schedule):
            raise ValueError(f"Existing run has a different pairing manifest, configuration or DDP schedule: {destination}")
    elif any(destination.glob("model_*.pt")):
        raise ValueError(f"Existing checkpoints have no pairing contract: {destination}")

    name = f"{cfg.model.name}_{cfg.model.audio_representation.name}_{cfg.datasets.name}_{cfg.model.tokenizer}"
    print(json.dumps({
        "result": "PASS", "training_launched": False, "training_pairs": len(paired),
        "train_pairing": paired.pairing_contract, "batch_and_schedule": schedule,
        "sample_prompt_frames": p.tolist(), "sample_target_frames": t.tolist(),
        "sample_speaker_shape": list(batch["speaker_embedding"].shape),
        "sample_teacher_shape": list(batch["repa_features"].shape) if "repa_features" in batch else None,
        "checkpoint_dir": str(destination), "tensorboard_logdir": str((Path("runs") / name).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
