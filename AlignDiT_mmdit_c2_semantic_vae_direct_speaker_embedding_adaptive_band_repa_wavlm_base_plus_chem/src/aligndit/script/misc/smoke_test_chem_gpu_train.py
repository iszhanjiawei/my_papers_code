"""Run a real Chem training checkpoint canary, then resume the identical run contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from aligndit.model.trainer_chem import ChemAdaptiveBandRepaTrainer
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import main as train_main


class StopAfterFirstCheckpoint(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["first", "resume"])
    parser.add_argument("--max-memory-gib", type=float, default=22.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real Chem training validation requires CUDA")
    torch.cuda.set_per_process_memory_fraction(
        args.max_memory_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("src/aligndit/config").resolve())):
        config = compose(
            config_name="finetune_chem_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus",
            overrides=[
                "optim.run_until_update=2",
                "model.name=Chem_adaptive_band_repa_wavlm_base_plus_smoke",
                "ckpts.save_dir=output/chem_adaptive_band_repa_wavlm_base_plus_smoke",
                "ckpts.last_per_updates=1",
                "ckpts.save_per_updates=2",
                "datasets.num_workers=0",
            ],
        )
    root = Path(config.ckpts.save_dir)
    original_save = ChemAdaptiveBandRepaTrainer.save_checkpoint

    def save_then_stop(trainer, update, last=False):
        original_save(trainer, update, last)
        if update == 1 and last:
            if trainer.logger == "tensorboard":
                trainer.writer.flush()
                trainer.writer.close()
            raise StopAfterFirstCheckpoint

    if args.phase == "first":
        if (root / "model_last.pt").exists():
            raise RuntimeError("First phase requires a fresh diagnostic checkpoint directory")
        try:
            with patch.object(ChemAdaptiveBandRepaTrainer, "save_checkpoint", save_then_stop):
                train_main.__wrapped__(config)
        except StopAfterFirstCheckpoint:
            pass
        else:
            raise AssertionError("Canary did not stop at update 1")
        expected_update = 1
    else:
        state = torch.load(root / "model_last.pt", map_location="cpu", weights_only=True, mmap=True)
        assert state["update"] == 1
        del state
        train_main.__wrapped__(config)
        expected_update = 2
    state = torch.load(root / "model_last.pt", map_location="cpu", weights_only=True, mmap=True)
    assert state["update"] == expected_update
    assert state["scheduler_total_updates"] == 120000
    assert state["optimizer_state_dict"]["state"]
    print(
        json.dumps(
            {
                "passed": True,
                "phase": args.phase,
                "update": expected_update,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "checkpoint_path": str(root / "model_last.pt"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
