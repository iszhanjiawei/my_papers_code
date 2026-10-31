"""Objective-only changes must never accept weights by tensor shape alone."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src/aligndit/model/wavlm_tra_contract.py"
spec = importlib.util.spec_from_file_location("wavlm_tra_contract", MODULE_PATH)
contract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract_module)


def fixture_config() -> dict:
    return {
        "model": {
            "repa_lambda": 0.1, "repa_schedule": "taro_clean_fraction", "repa_schedule_eps": 1e-8,
            "arch": {
                "dim": 768, "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
                "text_joint_attention": True, "temporal_band_enabled": True,
                "temporal_band_mode": "fixed", "temporal_band_bidirectional": True,
                "temporal_band_fixed_window_radius_frames": 4,
                "repa_layer": 9, "repa_target_dim": 768, "repa_projector_dim": 2048,
            },
        },
        "datasets": {
            "repa_feature_cache_dir": "/deliberately/absent/training/cache",
            "repa_feature_dim": 768, "repa_teacher_layer": 12,
            "repa_model_id": contract_module.WAVLM_MODEL_ID,
            "repa_model_revision": contract_module.WAVLM_MODEL_REVISION,
            "repa_checkpoint_sha256": contract_module.WAVLM_CHECKPOINT_SHA256,
        },
    }


class WavlmTraContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint_path = Path(self.directory.name) / "model_150000.pt"
        self.config = fixture_config()
        self.expected = contract_module.build_wavlm_tra_contract(self.config)
        self.checkpoint = {"wavlm_tra_contract": copy.deepcopy(self.expected), "update": 150000}
        self.sidecar = {
            "config": copy.deepcopy(self.config), "repa": copy.deepcopy(self.expected),
            "repa_cache_metadata": {
                "feature": "wavlm_hidden_state", "model_id": contract_module.WAVLM_MODEL_ID,
                "model_revision": contract_module.WAVLM_MODEL_REVISION,
                "checkpoint_sha256": contract_module.WAVLM_CHECKPOINT_SHA256,
                "teacher_layer": 12, "output_dim": 768, "output_dtype": "float16",
                "source_audio": "complete_unmasked_waveform", "sample_rate": 16000,
                "frame_rate_hz": 50.0, "status": "complete",
            },
        }
        self.write_sidecar()

    def write_sidecar(self):
        (self.checkpoint_path.parent / "speaker_training_contract.json").write_text(json.dumps(self.sidecar))

    def validate(self):
        return contract_module.validate_wavlm_tra_inference_contract(self.config, self.checkpoint_path, self.checkpoint)

    def test_valid_without_teacher_cache_or_checkpoint_file_access(self):
        self.assertFalse(Path(self.config["datasets"]["repa_feature_cache_dir"]).exists())
        self.assertFalse(self.checkpoint_path.exists())
        self.assertEqual(self.validate(), self.expected)

    def test_serializes_direction_reduction_and_teacher(self):
        self.assertEqual(self.expected["implemented_weight"], "t/(1+eps)")
        self.assertFalse(self.expected["weight_normalization"])
        self.assertIn("total_valid_generated_frames", self.expected["reduction"])
        self.assertEqual(self.expected["teacher"]["teacher_layer"], 12)
        self.assertEqual(self.expected["student_layer_zero_based"], 9)
        self.assertEqual(self.expected["projector"]["dimensions"], [768, 2048, 2048, 768])

    def test_plain_hard9_checkpoint_rejected_even_with_matching_sidecar(self):
        self.checkpoint.pop("wavlm_tra_contract")
        with self.assertRaisesRegex(RuntimeError, "missing embedded"):
            self.validate()

    def test_constant_repa_embedded_contract_rejected(self):
        self.checkpoint["wavlm_tra_contract"]["schedule"] = "fixed from the first update"
        with self.assertRaisesRegex(RuntimeError, "Embedded.*mismatch"):
            self.validate()

    def test_changed_schedule_epsilon_rejected(self):
        self.config["model"]["repa_schedule_eps"] = 1e-5
        with self.assertRaisesRegex(RuntimeError, "Embedded.*mismatch"):
            self.validate()

    def test_changed_loss_coefficient_rejected(self):
        self.config["model"]["repa_lambda"] = 0.2
        with self.assertRaisesRegex(RuntimeError, "Embedded.*mismatch"):
            self.validate()

    def test_reversed_direction_or_normalization_cannot_share_identity(self):
        for key, value in (("implemented_weight", "(1-t)/(1+eps)"), ("weight_normalization", True)):
            with self.subTest(key=key):
                self.checkpoint["wavlm_tra_contract"] = copy.deepcopy(self.expected)
                self.checkpoint["wavlm_tra_contract"][key] = value
                with self.assertRaisesRegex(RuntimeError, "Embedded.*mismatch"):
                    self.validate()

    def test_contract_bool_and_integer_are_not_interchangeable(self):
        self.checkpoint["wavlm_tra_contract"]["weight_normalization"] = 0
        with self.assertRaisesRegex(RuntimeError, "Embedded.*mismatch"):
            self.validate()

    def test_missing_sidecar_rejected(self):
        (self.checkpoint_path.parent / "speaker_training_contract.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "requires its training sidecar"):
            self.validate()

    def test_sidecar_repa_mismatch_rejected(self):
        self.sidecar["repa"]["schedule"] = "constant"
        self.write_sidecar()
        with self.assertRaisesRegex(RuntimeError, "sidecar.*mismatch"):
            self.validate()

    def test_sidecar_resolved_config_mismatch_rejected(self):
        self.sidecar["config"]["model"]["repa_lambda"] = 0.2
        self.write_sidecar()
        with self.assertRaisesRegex(RuntimeError, "Resolved training.*mismatch"):
            self.validate()

    def test_teacher_receipt_mismatch_rejected(self):
        self.sidecar["repa_cache_metadata"]["checkpoint_sha256"] = "0" * 64
        self.write_sidecar()
        with self.assertRaisesRegex(RuntimeError, "cache metadata mismatch"):
            self.validate()

    def test_fixed_repa_config_cannot_be_mislabelled_as_tra(self):
        self.config["model"]["repa_schedule"] = "fixed"
        with self.assertRaisesRegex(RuntimeError, "requires model.repa_schedule"):
            contract_module.build_wavlm_tra_contract(self.config)

    def test_invalid_coefficient_and_epsilon_rejected(self):
        for key in ("repa_lambda", "repa_schedule_eps"):
            for value in (0, -1, float("nan"), float("inf"), True, "0.1"):
                with self.subTest(key=key, value=value):
                    config = fixture_config()
                    config["model"][key] = value
                    with self.assertRaises(ValueError):
                        contract_module.build_wavlm_tra_contract(config)

    def test_different_window_architecture_and_teacher_layer_rejected(self):
        for key, value in (("temporal_band_fixed_window_radius_frames", 2), ("repa_layer", 3), ("n_mm_layers", 10)):
            with self.subTest(key=key):
                config = fixture_config()
                config["model"]["arch"][key] = value
                with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
                    contract_module.build_wavlm_tra_contract(config)
        self.config["datasets"]["repa_teacher_layer"] = 6
        with self.assertRaisesRegex(RuntimeError, "teacher mismatch"):
            contract_module.build_wavlm_tra_contract(self.config)


if __name__ == "__main__":
    unittest.main()
