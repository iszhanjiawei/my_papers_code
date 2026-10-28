"""CPU identity/config/diagnostic checks for the AVT joint nine-frame window."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.fixed_temporal_band_contract import (
    RADIUS_KEY,
    make_fixed_band_contract,
    validate_fixed_band_contract,
    validate_fixed_band_training_config,
)
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer
from aligndit.model.trainer_semantic_vae_fixed_band import SemanticVaeFixedBandTrainer


class HardWindow9ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "avt_joint_hard_window9_av_va_test"
        self.root.mkdir()
        self.checkpoint = self.root / "model_150000.pt"
        self.sidecar = self.root / "speaker_training_contract.json"
        self.config = OmegaConf.create({
            "model": {
                "name": "avt_joint_hard_window9_av_va_test",
                "audio_representation": {"frame_rate": 40.0},
                "arch": {
                    "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
                    "audio_video_ratio": 1, "text_joint_attention": True,
                    "temporal_band_enabled": True, "temporal_band_mode": "fixed",
                    "temporal_band_audio_fps": 40.0, "temporal_band_video_fps": 40.0,
                    "temporal_band_fixed_offset_seconds": 0.0,
                    "temporal_band_fixed_sigma_seconds": 0.100,
                    "temporal_band_bidirectional": True, RADIUS_KEY: 4,
                },
            },
            "ckpts": {"save_dir": str(self.root)},
        })
        self.band = FixedTemporalBand(dim=64, window_radius_frames=4)
        self.contract = self.make_sidecar(self.config, self.band)

    @staticmethod
    def make_sidecar(config, band):
        return {
            "config": OmegaConf.to_container(config, resolve=True),
            "temporal_band": make_fixed_band_contract(config.model.arch, band),
        }

    def write_contract(self, contract=None):
        self.sidecar.write_text(json.dumps(self.contract if contract is None else contract))

    def test_hard_contract_records_exact_geometry_and_shared_avt_scope(self):
        self.assertTrue(validate_fixed_band_training_config(self.config))
        self.write_contract()
        validate_fixed_band_contract(self.config, self.checkpoint)
        band = self.contract["temporal_band"]
        self.assertEqual(band["parameters"][RADIUS_KEY], 4)
        self.assertEqual(band["window_radius_frames"], 4)
        self.assertEqual(band["window_size_frames"], 9)
        self.assertEqual(band["kernel"], "hard_window")
        self.assertEqual(band["attention_mode"], "avt_joint")
        self.assertFalse(band["gaussian_sigma_active"])
        self.assertEqual(band["parameter_count"], 0)
        for block in ("AA", "VV", "AT", "VT", "TA", "TV", "TT"):
            self.assertIn(block, band["scope"])

    def test_training_rejects_wrong_radius_types_geometry_and_topology(self):
        cases = [(RADIUS_KEY, value) for value in (None, 2, 5, 4.0, True, -1, "4")]
        cases.extend((
            ("temporal_band_audio_fps", 25.0), ("temporal_band_video_fps", 25.0),
            ("temporal_band_fixed_offset_seconds", 0.025), ("temporal_band_bidirectional", False),
            ("temporal_band_enabled", False), ("text_joint_attention", False),
            ("depth", 20), ("n_mm_layers", 11), ("n_text_layers", 11), ("audio_video_ratio", 2),
        ))
        for key, value in cases:
            config = copy.deepcopy(self.config)
            config.model.arch[key] = value
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                validate_fixed_band_training_config(config)

    def test_training_rejects_inherited_paths_names_and_unlabeled_weights(self):
        for location in ("model.name", "ckpts.save_dir"):
            config = copy.deepcopy(self.config)
            OmegaConf.update(config, location, "avt_joint_fixed_band_av_va_test")
            with self.subTest(location=location), self.assertRaises(ValueError):
                validate_fixed_band_training_config(config)
        self.checkpoint.write_bytes(b"identity check only")
        with self.assertRaises(RuntimeError):
            validate_fixed_band_training_config(self.config)

    def test_gaussian_hard_mismatch_rejected_in_both_directions(self):
        gaussian = copy.deepcopy(self.config)
        del gaussian.model.arch[RADIUS_KEY]
        gaussian_contract = self.make_sidecar(gaussian, FixedTemporalBand(dim=64))
        self.write_contract(gaussian_contract)
        with self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
            validate_fixed_band_contract(self.config, self.checkpoint)
        self.write_contract()
        with self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
            validate_fixed_band_contract(gaussian, self.checkpoint)
        self.write_contract(gaussian_contract)
        validate_fixed_band_contract(gaussian, self.checkpoint)
        gaussian.model.arch[RADIUS_KEY] = None
        validate_fixed_band_contract(gaussian, self.checkpoint)

    def test_changed_radius_rejected_in_both_directions(self):
        for radius in (2, 5):
            changed = copy.deepcopy(self.config)
            changed.model.arch[RADIUS_KEY] = radius
            self.write_contract()
            with self.subTest(radius=radius), self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
                validate_fixed_band_contract(changed, self.checkpoint)
            self.write_contract(self.make_sidecar(changed, FixedTemporalBand(dim=64, window_radius_frames=radius)))
            with self.subTest(radius=radius), self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_hard_sidecar_requires_radius_in_parameters_and_resolved_config(self):
        for target in ("parameters", "resolved"):
            contract = copy.deepcopy(self.contract)
            values = (contract["temporal_band"]["parameters"] if target == "parameters"
                      else contract["config"]["model"]["arch"])
            del values[RADIUS_KEY]
            self.write_contract(contract)
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_hard_sidecar_rejects_contradictory_semantics(self):
        cases = (("kernel", "gaussian"), ("attention_mode", "av_joint_text_cross"),
                 ("window_radius_frames", 2), ("window_size_frames", 5),
                 ("window_size_frames", 9.0), ("gaussian_sigma_active", True),
                 ("boundary_policy", "shift to full width"))
        for key, value in cases:
            contract = copy.deepcopy(self.contract)
            contract["temporal_band"][key] = value
            self.write_contract(contract)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_missing_contract_and_invalid_radius_types_are_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config, self.checkpoint)
        for radius in (4.0, True, "4", -1):
            contract = copy.deepcopy(self.contract)
            contract["temporal_band"]["parameters"][RADIUS_KEY] = radius
            self.write_contract(contract)
            with self.subTest(radius=radius), self.assertRaisesRegex(RuntimeError, RADIUS_KEY):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_contract_creation_checks_actual_band(self):
        for band in (FixedTemporalBand(dim=64), FixedTemporalBand(dim=64, window_radius_frames=2),
                     FixedTemporalBand(dim=64, window_radius_frames=4, sigma_seconds=0.2)):
            with self.subTest(radius=band.window_radius_frames), self.assertRaises(RuntimeError):
                make_fixed_band_contract(self.config.model.arch, band)
        self.band.register_buffer("unexpected_state", torch.ones(1))
        with self.assertRaises(RuntimeError):
            make_fixed_band_contract(self.config.model.arch, self.band)

    def test_hard_diagnostics_report_width_without_active_sigma(self):
        backbone = SimpleNamespace(
            temporal_band=self.band,
            last_temporal_band_valid_mask=torch.tensor([[True, False]]),
            last_temporal_band_offset_seconds=torch.zeros(1, 2),
            last_temporal_band_sigma_seconds=torch.full((1, 2), 0.1),
        )
        # Use the concrete subclass instance without allocating data or GPU state.
        trainer = SemanticVaeFixedBandTrainer.__new__(SemanticVaeFixedBandTrainer)
        trainer.model = SimpleNamespace(transformer=backbone)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        with patch.object(SemanticVaeDirectC2SpeakerTrainer, "_forward_diagnostics", side_effect=lambda *_: {}):
            diagnostics = trainer._forward_diagnostics(torch.tensor(1.0), {})
            self.assertEqual(diagnostics["temporal_band/window_radius_frames"], 4.0)
            self.assertEqual(diagnostics["temporal_band/window_size_frames"], 9.0)
            self.assertEqual(diagnostics["temporal_band/offset_ms_mean"], 0.0)
            self.assertFalse(any("sigma" in key for key in diagnostics))
            backbone.temporal_band = FixedTemporalBand(dim=64)
            diagnostics = trainer._forward_diagnostics(torch.tensor(1.0), {})
            self.assertAlmostEqual(diagnostics["temporal_band/sigma_ms_mean"], 100.0)
            self.assertFalse(any("window_size" in key for key in diagnostics))

    def test_composed_hard9_config_is_accepted(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            config = compose(config_name="finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va")
        config.ckpts.save_dir = str(self.root)
        self.assertTrue(validate_fixed_band_training_config(config))
        self.assertEqual(config.model.arch[RADIUS_KEY], 4)
        self.assertTrue(config.model.arch.text_joint_attention)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
