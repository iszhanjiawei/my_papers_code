"""CPU checks for strict-window config, checkpoint provenance, and diagnostics.

Run from this experiment root with ``PYTHONPATH=src python
src/aligndit/script/misc/smoke_test_hard_window_contract.py``. Temporary sidecars
exercise semantic compatibility without loading checkpoints or training data.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from aligndit.model.trainer_semantic_vae_adaptive_band import SemanticVaeAdaptiveBandTrainer
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    load_composed_config,
    validate_fixed_band_contract,
)
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import validate_temporal_band_config


RADIUS_KEY = "temporal_band_fixed_window_radius_frames"
PROJECT = Path(__file__).resolve().parents[4]


def config_for(save_dir: Path):
    return OmegaConf.create({
        "model": {
            "name": "AlignDiT_hard_window5_av_va",
            "audio_representation": {"frame_rate": 40},
            "arch": {
                "n_mm_layers": 12,
                "audio_video_ratio": 1,
                "temporal_band_enabled": True,
                "temporal_band_mode": "fixed",
                "temporal_band_bidirectional": True,
                "temporal_band_audio_fps": 40.0,
                "temporal_band_video_fps": 40.0,
                "temporal_band_fixed_offset_seconds": 0.0,
                "temporal_band_fixed_sigma_seconds": 0.1,
                RADIUS_KEY: 2,
            },
        },
        "ckpts": {"save_dir": str(save_dir)},
    })


def assert_rejected(call, message_fragment: str):
    try:
        call()
    except (ValueError, TypeError, RuntimeError) as error:
        assert message_fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection containing {message_fragment!r}")


def write_contract(checkpoint: Path, config):
    contract = {
        "temporal_band": {
            "mode": "fixed",
            "parameters": OmegaConf.to_container(config.model.arch, resolve=True),
            "parameter_count": 0,
        },
    }
    (checkpoint.parent / "speaker_training_contract.json").write_text(json.dumps(contract))


def test_checkpoint_semantic_compatibility():
    with TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "model_3.pt"
        config = config_for(Path(temporary) / "hard_window5_av_va")
        assert_rejected(lambda: validate_fixed_band_contract(config, checkpoint), "matching fixed-band training contract")
        write_contract(checkpoint, config)
        validate_fixed_band_contract(config, checkpoint)

        # A legacy sidecar has no radius field. Matching Gaussian inference is
        # valid, but identical state-dict keys must not permit hard-mask reuse.
        legacy = copy.deepcopy(config)
        del legacy.model.arch[RADIUS_KEY]
        write_contract(checkpoint, legacy)
        validate_fixed_band_contract(legacy, checkpoint)
        assert_rejected(lambda: validate_fixed_band_contract(config, checkpoint), RADIUS_KEY)
        write_contract(checkpoint, config)
        assert_rejected(lambda: validate_fixed_band_contract(legacy, checkpoint), RADIUS_KEY)

        for bad_radius in (1, 3, True, 2.0, 1.5, -1, "2"):
            bad = copy.deepcopy(config)
            bad.model.arch[RADIUS_KEY] = bad_radius
            write_contract(checkpoint, config)
            assert_rejected(lambda: validate_fixed_band_contract(bad, checkpoint), RADIUS_KEY)
            write_contract(checkpoint, bad)
            assert_rejected(lambda: validate_fixed_band_contract(config, checkpoint), RADIUS_KEY)

        # Check direction semantics too: an AV-only checkpoint cannot silently
        # become the requested AV+VA experiment, even when the radius matches.
        wrong_direction = copy.deepcopy(config)
        wrong_direction.model.arch.temporal_band_bidirectional = False
        write_contract(checkpoint, wrong_direction)
        assert_rejected(lambda: validate_fixed_band_contract(config, checkpoint), "temporal_band_bidirectional")
        wrong_direction.model.arch.temporal_band_bidirectional = 1
        write_contract(checkpoint, wrong_direction)
        assert_rejected(lambda: validate_fixed_band_contract(config, checkpoint), "must be a bool")


def test_training_config_safety():
    with TemporaryDirectory() as temporary:
        save_dir = Path(temporary) / "hard_window5_av_va"
        config = config_for(save_dir)
        assert validate_temporal_band_config(config) == (True, "fixed", True, 2)
        for bad_radius in (1, 3, True, 2.0, 1.5, -1, "2"):
            bad = copy.deepcopy(config)
            bad.model.arch[RADIUS_KEY] = bad_radius
            assert_rejected(lambda: validate_temporal_band_config(bad), "requires window radius 2")

        cases = (
            ("model.arch.temporal_band_enabled", False, "enabled fixed bidirectional"),
            ("model.arch.temporal_band_mode", "adaptive", "enabled fixed bidirectional"),
            ("model.arch.temporal_band_bidirectional", False, "enabled fixed bidirectional"),
            ("model.arch.temporal_band_bidirectional", 1, "must be a bool"),
            ("model.arch.temporal_band_fixed_offset_seconds", 0.025, "zero temporal offset"),
            ("model.arch.temporal_band_audio_fps", 25.0, "rates must match"),
            ("model.arch.temporal_band_video_fps", 25.0, "rates must match"),
            ("model.name", "AlignDiT_fixed_band_av_va", "dedicated model.name and ckpts.save_dir"),
            ("ckpts.save_dir", str(Path(temporary) / "fixed_band_av_va"), "dedicated model.name and ckpts.save_dir"),
        )
        for field, value, expected in cases:
            bad = copy.deepcopy(config)
            OmegaConf.update(bad, field, value)
            assert_rejected(lambda: validate_temporal_band_config(bad), expected)
        missing = copy.deepcopy(config)
        del missing.model.arch[RADIUS_KEY]
        assert_rejected(lambda: validate_temporal_band_config(missing), "require a hard temporal window")
        wrong_rate = copy.deepcopy(config)
        wrong_rate.model.audio_representation.frame_rate = 25
        wrong_rate.model.arch.temporal_band_audio_fps = 25
        wrong_rate.model.arch.temporal_band_video_fps = 25
        assert_rejected(lambda: validate_temporal_band_config(wrong_rate), "40")

        save_dir.mkdir()
        (save_dir / "model_3.pt").touch()
        assert_rejected(lambda: validate_temporal_band_config(config), "without their original training contract")


def test_composed_experiment_config():
    config = load_composed_config(
        PROJECT / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_hard_window5_av_va.yaml"
    )
    assert "hard_window5_av_va" in config.model.name
    assert "hard_window5_av_va" in config.ckpts.save_dir
    assert config.model.audio_representation.frame_rate == 40
    with TemporaryDirectory() as temporary:
        config.ckpts.save_dir = str(Path(temporary) / "hard_window5_av_va")
        assert validate_temporal_band_config(config) == (True, "fixed", True, 2)


def test_hard_window_diagnostics():
    trainer = object.__new__(SemanticVaeAdaptiveBandTrainer)
    backbone = SimpleNamespace(
        temporal_band=SimpleNamespace(window_radius_frames=2),
        last_temporal_band_valid_mask=torch.tensor([[True, True, False]]),
        last_temporal_band_offset_seconds=torch.zeros(1, 3),
        last_temporal_band_sigma_seconds=torch.full((1, 3), 0.1),
        speaker_proj=torch.nn.Linear(192, 4, bias=False),
    )
    trainer.model = SimpleNamespace(transformer=backbone)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.current_ctc_lambda = 0.0
    diagnostics = trainer._forward_diagnostics(torch.tensor(2.0), {"flow_loss": 2.0})
    assert diagnostics["temporal_band/window_radius_frames"] == 2
    assert diagnostics["temporal_band/window_size_frames"] == 5
    assert diagnostics["temporal_band/offset_ms_mean"] == 0
    assert not any("sigma" in key for key in diagnostics)
    # Gaussian compatibility still exposes the meaningful sigma statistic.
    backbone.temporal_band.window_radius_frames = None
    diagnostics = trainer._forward_diagnostics(torch.tensor(2.0), {"flow_loss": 2.0})
    assert diagnostics["temporal_band/sigma_ms_mean"] == 100
    assert "temporal_band/window_size_frames" not in diagnostics


def main():
    torch.set_num_threads(1)
    for test in (
        test_checkpoint_semantic_compatibility,
        test_training_config_safety,
        test_composed_experiment_config,
        test_hard_window_diagnostics,
    ):
        test()
        print(f"PASS {test.__name__}", flush=True)
    print("Hard-window checkpoint/config/diagnostics contracts passed.", flush=True)


if __name__ == "__main__":
    main()
