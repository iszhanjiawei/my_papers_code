import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from aligndit.script.misc.validate_semantic_vae_warmstart_checkpoint import (
    validate_completed_stage_checkpoint,
    validate_resume_checkpoint_order,
)


class SemanticVaeWarmStartCheckpointValidatorTest(unittest.TestCase):
    def _fixture(self, root: Path, *, stage: str = "s2a", update: int = 10_000):
        contract_path = root / "training_contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": {"stage": stage, "previous_stage": "s1"},
                    "distributed_runtime": {"mixed_precision": "bf16", "num_processes": 6},
                    "config": {
                        "stage": {"name": stage},
                        "optim": {"max_updates": update},
                        "datasets": {
                            "batch_size_per_gpu": 7200,
                            "batch_size_type": "frame",
                            "max_samples": 32,
                        },
                        "model": {"audio_representation": {"channels": 64, "frame_rate": 40}},
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        model_state = {"weight": torch.ones(2, 3), "bias": torch.zeros(2)}
        checkpoint = {
            "checkpoint_schema_version": 1,
            "warmstart_stage": stage,
            "model_state_dict": model_state,
            "optimizer_state_dict": {"param_groups": [{}]},
            "ema_model_state_dict": {
                "initted": torch.tensor(True),
                "step": torch.tensor(update),
                **{f"ema_model.{key}": value.clone() for key, value in model_state.items()},
            },
            "scheduler_state_dict": {"last_epoch": update},
            "training_contract_sha256": contract_sha256,
            "update": update,
        }
        checkpoint_path = root / "model_last.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path, contract_path

    def test_valid_completed_stage_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, contract_path = self._fixture(Path(directory))
            report = validate_completed_stage_checkpoint(
                checkpoint_path,
                contract_path,
                expected_stage="s2a",
                expected_update=10_000,
                expected_horizon=10_000,
                expected_model_keys=2,
            )
        self.assertEqual(report["stage"], "s2a")
        self.assertEqual(report["update"], 10_000)
        self.assertEqual(report["ema_step"], 10_000)

    def test_stage_update_contract_and_ema_mismatches_fail_closed(self):
        mutators = {
            "stage": lambda checkpoint: checkpoint.update(warmstart_stage="s2b"),
            "update": lambda checkpoint: checkpoint.update(update=9_999),
            "contract": lambda checkpoint: checkpoint.update(training_contract_sha256="0" * 64),
            "ema_step": lambda checkpoint: checkpoint["ema_model_state_dict"].update(step=torch.tensor(9_999)),
            "ema_initted": lambda checkpoint: checkpoint["ema_model_state_dict"].update(initted=torch.tensor(False)),
            "non_finite": lambda checkpoint: checkpoint["ema_model_state_dict"]["ema_model.weight"].fill_(float("nan")),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                checkpoint_path, contract_path = self._fixture(Path(directory))
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                mutate(checkpoint)
                torch.save(checkpoint, checkpoint_path)
                with self.assertRaises(RuntimeError):
                    validate_completed_stage_checkpoint(
                        checkpoint_path,
                        contract_path,
                        expected_stage="s2a",
                        expected_update=10_000,
                        expected_horizon=10_000,
                        expected_model_keys=2,
                    )

    def test_stale_model_last_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            last_path, _ = self._fixture(directory, update=500)
            checkpoint = torch.load(last_path, weights_only=True)
            checkpoint["update"] = 1_000
            checkpoint["ema_model_state_dict"]["step"] = torch.tensor(1_000)
            torch.save(checkpoint, directory / "model_1000.pt")
            with self.assertRaisesRegex(RuntimeError, "Stale model_last"):
                validate_resume_checkpoint_order(directory)

            (directory / "model_1000.pt").unlink()
            checkpoint["update"] = 500
            checkpoint["ema_model_state_dict"]["step"] = torch.tensor(500)
            torch.save(checkpoint, directory / "model_500.pt")
            report = validate_resume_checkpoint_order(directory)
            self.assertEqual(report["latest_update"], 500)


if __name__ == "__main__":
    unittest.main()
