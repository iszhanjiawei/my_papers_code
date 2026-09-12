"""CPU-only positive and adversarial checks for LocAt sidecars and migration."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from aligndit.model.locat_contract import make_locat_contract, validate_locat_checkpoint_contract
from aligndit.model.semantic_vae_direct_migration import validate_locat_initialization


def fixture(*, va=False, enabled=True):
    config = {
        "locat_enabled": enabled, "locat_av_enabled": True, "locat_va_enabled": va,
        "locat_audio_fps": 40.0, "locat_video_fps": 40.0,
        "locat_sigma_min_seconds": 0.025, "locat_sigma_max_seconds": 0.400,
        "locat_sigma_init_seconds": 0.100, "locat_alpha_init": 0.1,
        "locat_bias_mode": "gaussian", "locat_av_layers": 12 if enabled else 0,
        "locat_va_layers": 11 if enabled and va else 0,
    }
    backbone = nn.Module()
    backbone.locat_config = config
    backbone.transformer_blocks = nn.ModuleList([nn.Module() for _ in range(12)])
    for direction in ("av", "va"):
        for layer in range(config[f"locat_{direction}_layers"]):
            predictor = nn.Module()
            predictor.log_sigma = nn.Linear(64, 1)
            predictor.log_alpha = nn.Linear(64, 1)
            nn.init.zeros_(predictor.log_sigma.weight)
            nn.init.zeros_(predictor.log_alpha.weight)
            nn.init.constant_(predictor.log_sigma.bias, math.log(0.075 / 0.300))
            nn.init.constant_(predictor.log_alpha.bias, math.log(math.expm1(0.1)))
            setattr(backbone.transformer_blocks[layer], f"locat_{direction}", predictor)
    model = nn.Module()
    model.transformer = backbone
    return model


class LocAtContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="locat-contract-")
        self.directory = Path(self.temporary.name)
        self.model = fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self, value=None):
        value = {"locat": make_locat_contract(self.model.transformer)} if value is None else value
        (self.directory / "speaker_training_contract.json").write_text(json.dumps(value), encoding="utf-8")

    def check(self, backbone=None, **kwargs):
        return validate_locat_checkpoint_contract(backbone or self.model.transformer, self.directory, **kwargs)

    def test_exact_av_and_bidirectional_counts(self):
        self.assertEqual(make_locat_contract(self.model.transformer)["parameter_count"], 1560)
        self.assertEqual(make_locat_contract(fixture(va=True).transformer)["parameter_count"], 2990)

    def test_new_empty_training_directory_allowed(self):
        self.check(require_existing=False)

    def test_inference_requires_existing_sidecar(self):
        with self.assertRaises(RuntimeError):
            self.check()

    def test_existing_weights_without_sidecar_rejected_for_resume(self):
        (self.directory / "model_last.pt").touch()
        with self.assertRaises(RuntimeError):
            self.check(require_existing=False)

    def test_matching_contract_allowed_for_inference_and_resume(self):
        self.publish()
        self.check()
        self.check(require_existing=False)

    def test_semantic_changes_without_tensor_shape_changes_rejected(self):
        self.publish()
        for key, value in (
            ("locat_bias_mode", "uniform"), ("locat_audio_fps", 25.0),
            ("locat_sigma_max_seconds", 0.500), ("locat_alpha_init", 0.2),
        ):
            backbone = copy.deepcopy(self.model.transformer)
            backbone.locat_config[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.check(backbone)

    def test_direction_change_rejected(self):
        self.publish()
        with self.assertRaises(RuntimeError):
            self.check(fixture(va=True).transformer)

    def test_baseline_cannot_load_locat_sidecar(self):
        self.publish()
        with self.assertRaises(RuntimeError):
            self.check(fixture(enabled=False).transformer)

    def test_baseline_without_locat_contract_unchanged(self):
        self.check(fixture(enabled=False).transformer)

    def test_locat_cannot_load_baseline_sidecar(self):
        self.publish({"experiment": "baseline"})
        with self.assertRaises(RuntimeError):
            self.check()

    def test_tampered_parameter_count_rejected(self):
        contract = {"locat": make_locat_contract(self.model.transformer)}
        contract["locat"]["parameter_count"] = 1
        self.publish(contract)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_malformed_contract_rejected(self):
        self.publish([])
        with self.assertRaises(TypeError):
            self.check()


class LocAtMigrationTests(unittest.TestCase):
    def setUp(self):
        self.model = fixture()
        self.state = self.model.state_dict()

    def check(self, state=None, new_target=None):
        state = self.state if state is None else state
        return validate_locat_initialization(self.model, state, set(state) if new_target is None else new_target)

    def test_exact_initialized_av_keys(self):
        self.assertEqual(len(self.check()), 48)

    def test_exact_initialized_bidirectional_keys(self):
        model = fixture(va=True)
        self.assertEqual(len(validate_locat_initialization(model, model.state_dict(), set(model.state_dict()))), 92)

    def test_unknown_key_rejected(self):
        self.state["transformer.transformer_blocks.0.locat_av.unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_missing_key_rejected(self):
        self.state.pop(next(iter(self.state)))
        with self.assertRaises(RuntimeError):
            self.check()

    def test_parent_locat_key_rejected(self):
        new_target = set(self.state)
        new_target.remove(next(iter(self.state)))
        with self.assertRaises(RuntimeError):
            self.check(new_target=new_target)

    def test_changed_weight_rejected(self):
        self.state["transformer.transformer_blocks.0.locat_av.log_sigma.weight"][0, 0] = 1.0
        with self.assertRaises(RuntimeError):
            self.check()

    def test_wrong_bias_rejected(self):
        self.state["transformer.transformer_blocks.0.locat_av.log_alpha.bias"].zero_()
        with self.assertRaises(RuntimeError):
            self.check()

    def test_wrong_shape_rejected(self):
        self.state["transformer.transformer_blocks.0.locat_av.log_sigma.weight"] = torch.zeros(1, 32)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_nonfinite_initialization_rejected(self):
        self.state["transformer.transformer_blocks.0.locat_av.log_sigma.weight"][0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            self.check()

    def test_disabled_model_rejects_stray_locat_keys(self):
        self.model.transformer.locat_config["locat_enabled"] = False
        with self.assertRaises(RuntimeError):
            self.check()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
