"""CPU checks for fixed-band experiment identity; no dataset or GPU is needed."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.fixed_temporal_band_contract import (
    FIXED_BAND_KEYS,
    make_fixed_band_contract,
    validate_fixed_band_contract,
    validate_fixed_band_training_config,
)


class FixedBandContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = OmegaConf.create({
            "model": {
                "name": "avt_joint_fixed_band_av_va_test",
                "audio_representation": {"frame_rate": 40.0},
                "arch": {
                    "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
                    "audio_video_ratio": 1, "text_joint_attention": True,
                    "temporal_band_enabled": True, "temporal_band_mode": "fixed",
                    "temporal_band_audio_fps": 40.0, "temporal_band_video_fps": 40.0,
                    "temporal_band_fixed_offset_seconds": 0.0,
                    "temporal_band_fixed_sigma_seconds": 0.100,
                    "temporal_band_bidirectional": True,
                },
            },
            "ckpts": {"save_dir": str(self.root / "avt_joint_fixed_band_av_va_test")},
        })
        self.save_dir = Path(self.config.ckpts.save_dir)
        self.save_dir.mkdir()
        self.checkpoint = self.save_dir / "model_150000.pt"
        self.sidecar = self.save_dir / "speaker_training_contract.json"
        self.band = FixedTemporalBand(dim=64)
        self.contract = {
            "config": OmegaConf.to_container(self.config, resolve=True),
            "temporal_band": make_fixed_band_contract(self.config.model.arch, self.band),
        }

    def write_contract(self, contract=None):
        self.sidecar.write_text(json.dumps(self.contract if contract is None else contract))

    def test_valid_training_and_inference_contract(self):
        self.assertTrue(validate_fixed_band_training_config(self.config))
        self.write_contract()
        validate_fixed_band_contract(self.config, self.checkpoint)
        self.assertEqual(self.contract["temporal_band"]["parameter_count"], 0)
        for zero_block in ("AA", "VV", "AT", "VT", "TA", "TV", "TT"):
            self.assertIn(zero_block, self.contract["temporal_band"]["scope"])

    def test_training_rejects_invalid_flags_and_topology(self):
        cases = (
            ("temporal_band_enabled", False), ("temporal_band_enabled", "true"),
            ("temporal_band_bidirectional", False), ("temporal_band_bidirectional", 1),
            ("temporal_band_mode", "adaptive"), ("text_joint_attention", False),
            ("text_joint_attention", 1), ("depth", 20), ("n_mm_layers", 10),
            ("n_text_layers", 18), ("audio_video_ratio", 2),
            ("temporal_band_audio_fps", 25.0), ("temporal_band_video_fps", 25.0),
            ("temporal_band_fixed_offset_seconds", float("nan")),
            ("temporal_band_fixed_sigma_seconds", 0.0),
            ("temporal_band_fixed_sigma_seconds", float("inf")),
        )
        for key, value in cases:
            config = copy.deepcopy(self.config)
            config.model.arch[key] = value
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                validate_fixed_band_training_config(config)

    def test_training_rejects_wrong_names_paths_and_existing_weights(self):
        for location in ("model.name", "ckpts.save_dir"):
            config = copy.deepcopy(self.config)
            OmegaConf.update(config, location, "original_avt_experiment")
            with self.subTest(location=location), self.assertRaises(ValueError):
                validate_fixed_band_training_config(config)
        self.checkpoint.write_bytes(b"identity check only; not a checkpoint")
        with self.assertRaises(RuntimeError):
            validate_fixed_band_training_config(self.config)

    def test_inference_rejects_missing_or_original_avt_sidecar(self):
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config, self.checkpoint)
        self.write_contract({"config": self.contract["config"]})
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config, self.checkpoint)

    def test_inference_rejects_every_changed_band_parameter(self):
        changed_values = (False, "adaptive", 25.0, 25.0, 0.025, 0.050, False)
        for key, value in zip(FIXED_BAND_KEYS, changed_values):
            contract = copy.deepcopy(self.contract)
            contract["temporal_band"]["parameters"][key] = value
            self.write_contract(contract)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_inference_rejects_legacy_av_only_and_nonzero_parameter_count(self):
        for alteration in ("legacy", "count", "bool_count"):
            contract = copy.deepcopy(self.contract)
            if alteration == "legacy":
                del contract["temporal_band"]["parameters"]["temporal_band_bidirectional"]
            else:
                contract["temporal_band"]["parameter_count"] = False if alteration == "bool_count" else 4
            self.write_contract(contract)
            with self.subTest(alteration=alteration), self.assertRaises(RuntimeError):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_inference_rejects_disabled_bias_on_fixed_weights(self):
        self.write_contract()
        config = copy.deepcopy(self.config)
        config.model.arch.temporal_band_enabled = False
        config.model.arch.temporal_band_bidirectional = False
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(config, self.checkpoint)

    def test_inference_rejects_cross_attention_or_contradictory_sidecars(self):
        for field in ("nested_topology", "topology", "nested_geometry"):
            contract = copy.deepcopy(self.contract)
            if field == "nested_topology":
                contract["config"]["model"]["arch"]["text_joint_attention"] = False
            elif field == "topology":
                contract["temporal_band"]["text_joint_attention"] = False
            else:
                contract["config"]["model"]["arch"]["temporal_band_fixed_sigma_seconds"] = 0.2
            self.write_contract(contract)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_contract_rejects_hidden_band_state(self):
        self.band.register_buffer("unexpected_state", torch.ones(1))
        with self.assertRaises(RuntimeError):
            make_fixed_band_contract(self.config.model.arch, self.band)

    def test_disabled_parent_training_is_unchanged(self):
        config = copy.deepcopy(self.config)
        config.model.arch.temporal_band_enabled = False
        config.model.arch.temporal_band_bidirectional = False
        config.model.name = "original_avt"
        config.ckpts.save_dir = str(self.root / "original_avt")
        self.assertFalse(validate_fixed_band_training_config(config))
        validate_fixed_band_contract(config, self.checkpoint)

    def test_composed_experiment_config_is_accepted(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            config = compose(config_name="finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_av_va")
        config.ckpts.save_dir = str(self.save_dir)
        self.assertTrue(validate_fixed_band_training_config(config))
        self.assertTrue(config.model.arch.text_joint_attention)


if __name__ == "__main__":
    unittest.main(verbosity=2)
