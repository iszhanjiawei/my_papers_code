"""Exercise actual inherited checkpoint hooks without a trainer launch or GPUs."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_wavlm_tra_contract import fixture_config
from torch import nn

from aligndit.model.trainer_semantic_vae_fixed_band import SemanticVaeFixedBandTrainer
from aligndit.model.wavlm_tra_contract import build_wavlm_tra_contract


class FakeAccelerator:
    is_main_process = True

    @staticmethod
    def wait_for_everyone():
        pass

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def save(checkpoint, path):
        torch.save(checkpoint, path)


class TrainerCheckpointHookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trainer = object.__new__(SemanticVaeFixedBandTrainer)
        trainer = self.trainer
        trainer.model = nn.Linear(2, 2)
        trainer.model.repa_lambda = 0.1
        trainer.ema_model = nn.Linear(2, 2)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.25)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 10)
        trainer.accelerator = FakeAccelerator()
        trainer.checkpoint_path = self.directory.name
        trainer.keep_last_n_checkpoints = -1
        trainer.grad_accumulation_steps = 1
        trainer.wavlm_tra_contract = build_wavlm_tra_contract(fixture_config())
        self.path = Path(self.directory.name) / "model_last.pt"

    def test_actual_inherited_save_and_resume_preserve_contract_and_state(self):
        trainer = self.trainer
        expected_online = copy.deepcopy(trainer.model.state_dict())
        expected_ema = copy.deepcopy(trainer.ema_model.state_dict())
        trainer.save_checkpoint(123, last=True)
        checkpoint = torch.load(self.path, weights_only=True)
        self.assertEqual(checkpoint["wavlm_tra_contract"], trainer.wavlm_tra_contract)
        self.assertEqual(checkpoint["update"], 123)
        with torch.no_grad():
            trainer.model.weight.fill_(99)
            trainer.ema_model.weight.fill_(88)
        trainer.optimizer.param_groups[0]["lr"] = 0.9
        self.assertEqual(trainer.load_checkpoint(), 123)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.25)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected_online[key], rtol=0, atol=0)
        for key, value in trainer.ema_model.state_dict().items():
            torch.testing.assert_close(value, expected_ema[key], rtol=0, atol=0)

    def test_resume_rejects_missing_or_changed_identity_before_mutating_state(self):
        trainer = self.trainer
        trainer.save_checkpoint(123, last=True)
        original = torch.load(self.path, weights_only=True)
        for missing in (True, False):
            with self.subTest(missing=missing):
                checkpoint = copy.deepcopy(original)
                if missing:
                    del checkpoint["wavlm_tra_contract"]
                else:
                    checkpoint["wavlm_tra_contract"]["implemented_weight"] = "1-t"
                torch.save(checkpoint, self.path)
                with (
                    patch.object(trainer.model, "load_state_dict", side_effect=AssertionError("model mutated")),
                    patch.object(trainer.ema_model, "load_state_dict", side_effect=AssertionError("EMA mutated")),
                    patch.object(trainer.optimizer, "load_state_dict", side_effect=AssertionError("optimizer mutated")),
                    self.assertRaises(RuntimeError),
                ):
                    trainer.load_checkpoint()

    def test_active_repa_cannot_save_without_objective_identity(self):
        del self.trainer.wavlm_tra_contract
        with self.assertRaisesRegex(RuntimeError, "must include"):
            self.trainer.save_checkpoint(123, last=True)
        self.assertFalse(self.path.exists())

    def test_disabled_repa_retains_original_checkpoint_schema(self):
        self.trainer.model.repa_lambda = 0.0
        del self.trainer.wavlm_tra_contract
        self.trainer.save_checkpoint(123, last=True)
        checkpoint = torch.load(self.path, weights_only=True)
        self.assertNotIn("wavlm_tra_contract", checkpoint)
        self.assertEqual(self.trainer.load_checkpoint(), 123)


if __name__ == "__main__":
    unittest.main()
