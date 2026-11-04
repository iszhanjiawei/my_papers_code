"""CPU regression checks for the isolated 17-to-13-frame training curriculum.

Run from this experiment directory with PYTHONPATH=src. No data or GPU needed.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.fixed_temporal_band_contract import (
    make_fixed_band_contract,
    validate_fixed_band_contract,
    validate_fixed_band_training_config,
)
from aligndit.model.temporal_band_curriculum import CosineBandSchedule
from aligndit.model.trainer_semantic_vae_cosine_band import SemanticVaeCosineBandTrainer


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_cosine_band_window17to13_av_va"
PARENT_CONFIG_NAME = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window17_av_va"
MARKER = "avt_joint_cosine_band_window17to13_av_va"


def schedule() -> CosineBandSchedule:
    return CosineBandSchedule(
        start_sigma_seconds=0.200,
        end_sigma_seconds=0.150,
        start_update=10_000,
        end_update=50_000,
    )


def compose_config(name: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        return compose(config_name=name)


def fake_trainer(*, is_main: bool = True):
    """Exercise the production hook without allocating a backbone or optimizer."""
    trainer = SemanticVaeCosineBandTrainer.__new__(SemanticVaeCosineBandTrainer)
    trainer.model = SimpleNamespace(
        transformer=SimpleNamespace(temporal_band=FixedTemporalBand(dim=8, sigma_seconds=0.150)),
        ctc_lambda=0.03,
    )
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model, is_main_process=is_main)
    if is_main:
        trainer.ema_model = SimpleNamespace(
            ema_model=SimpleNamespace(
                transformer=SimpleNamespace(temporal_band=FixedTemporalBand(dim=8, sigma_seconds=0.150))
            ),
            # The inherited parent EMA has already seen 70k updates. It must
            # not advance this fresh experiment's window curriculum.
            step=torch.tensor(70_000),
        )
    trainer.temporal_band_schedule = schedule()
    trainer.ctc_target_lambda = 0.03
    trainer.ctc_warmup_start = 10_000
    trainer.ctc_warmup_end = 30_000
    trainer.current_ctc_lambda = 0.0
    return trainer


class CurriculumScheduleTests(unittest.TestCase):
    def test_boundary_values_and_cosine_quarters(self):
        expected = {
            1: 0.200,
            10_000: 0.200,
            20_000: 0.175 + 0.025 / math.sqrt(2.0),
            30_000: 0.175,
            40_000: 0.175 - 0.025 / math.sqrt(2.0),
            50_000: 0.150,
            150_000: 0.150,
        }
        for update, sigma in expected.items():
            with self.subTest(update=update):
                self.assertAlmostEqual(schedule().sigma_at(update), sigma, places=14)

    def test_monotonic_bounded_width(self):
        values = [schedule().sigma_at(update) for update in range(1, 150_001, 101)]
        self.assertTrue(all(0.150 <= value <= 0.200 for value in values))
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_invalid_schedule_rejected(self):
        good = dict(start_sigma_seconds=0.2, end_sigma_seconds=0.15, start_update=10_000, end_update=50_000)
        for key, value in (
            ("start_sigma_seconds", float("nan")),
            ("end_sigma_seconds", 0.0),
            ("start_update", -1),
            ("end_update", 10_000),
        ):
            kwargs = {**good, key: value}
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                CosineBandSchedule(**kwargs)


class CurriculumTrainerTests(unittest.TestCase):
    def test_optimizer_update_index_and_parent_ctc_hook(self):
        trainer = fake_trainer()
        for completed, sigma, ctc in (
            (0, 0.200, 0.0),
            (9_999, 0.200, 0.0),
            (19_999, 0.175 + 0.025 / math.sqrt(2.0), 0.015),
            (29_999, 0.175, 0.030),
            (49_999, 0.150, 0.030),
            (149_999, 0.150, 0.030),
        ):
            with self.subTest(completed_updates=completed):
                trainer._before_update(completed)
                self.assertAlmostEqual(trainer.model.transformer.temporal_band.sigma_seconds, sigma)
                self.assertAlmostEqual(trainer.ema_model.ema_model.transformer.temporal_band.sigma_seconds, sigma)
                self.assertAlmostEqual(trainer.current_band_sigma_seconds, sigma)
                self.assertEqual(trainer.completed_updates, completed)
                self.assertAlmostEqual(trainer.model.ctc_lambda, ctc)
                self.assertAlmostEqual(trainer.current_ctc_lambda, ctc)

    def test_gradient_accumulation_and_resume_use_completed_updates(self):
        continuing = fake_trainer()
        continuing._before_update(20_000)
        # The repeated hook calls represent microbatches from one optimizer
        # update: their schedule must not advance independently.
        for _ in range(4):
            continuing._before_update(29_999)
            self.assertAlmostEqual(continuing.current_band_sigma_seconds, 0.175)
        resumed = fake_trainer()
        resumed._before_update(29_999)
        self.assertEqual(resumed.current_band_sigma_seconds, continuing.current_band_sigma_seconds)
        self.assertEqual(resumed.current_ctc_lambda, continuing.current_ctc_lambda)
        self.assertEqual(int(resumed.ema_model.step), 70_000)

    def test_non_main_ddp_rank_does_not_require_an_ema_copy(self):
        trainer = fake_trainer(is_main=False)
        trainer._before_update(29_999)
        self.assertAlmostEqual(trainer.model.transformer.temporal_band.sigma_seconds, 0.175)

    def test_schedule_changes_actual_bias_without_parameter_or_rng_changes(self):
        trainer = fake_trainer()
        band = trainer.model.transformer.temporal_band
        video = torch.zeros(2, 16, 8)
        rng_state = torch.get_rng_state().clone()
        trainer._before_update(0)
        wide = band.bias(*band(video, 16), 16)
        trainer._before_update(49_999)
        narrow = band.bias(*band(video, 16), 16)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        self.assertEqual(band.state_dict(), {})
        self.assertEqual(list(band.parameters()), [])
        self.assertTrue(torch.equal(wide.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 16)))
        self.assertTrue(torch.equal(narrow.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 16)))
        off_diagonal = ~torch.eye(16, dtype=torch.bool)
        self.assertTrue(torch.all(narrow[:, off_diagonal] < wide[:, off_diagonal]))


class CurriculumContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = compose_config(CONFIG_NAME)
        self.config.ckpts.save_dir = str(Path(temporary.name) / MARKER)
        self.root = Path(self.config.ckpts.save_dir)
        self.root.mkdir()
        self.checkpoint = self.root / "model_150000.pt"
        self.sidecar = self.root / "speaker_training_contract.json"
        self.band = FixedTemporalBand(dim=8, sigma_seconds=0.150)
        self.contract = {
            "config": OmegaConf.to_container(self.config, resolve=True),
            "temporal_band": make_fixed_band_contract(self.config.model.arch, self.band, schedule()),
        }

    def write_contract(self, contract=None):
        self.sidecar.write_text(json.dumps(self.contract if contract is None else contract))

    def test_matching_contract_and_target_inference_width(self):
        self.assertTrue(validate_fixed_band_training_config(self.config))
        self.write_contract()
        validate_fixed_band_contract(self.config, self.checkpoint)
        self.assertEqual(self.contract["temporal_band"]["mode"], "cosine_curriculum")
        self.assertEqual(self.contract["temporal_band"]["parameter_count"], 0)
        self.assertEqual(self.config.model.arch.temporal_band_fixed_sigma_seconds, 0.150)
        self.assertEqual(self.band.sigma_seconds, 0.150)

    def test_missing_or_fixed_window_contract_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config, self.checkpoint)
        old = copy.deepcopy(self.contract)
        old["temporal_band"]["mode"] = "fixed"
        self.write_contract(old)
        with self.assertRaises(RuntimeError):
            validate_fixed_band_contract(self.config, self.checkpoint)

    def test_schedule_and_nested_config_mismatches_rejected(self):
        for location in ("recorded_schedule", "nested_config", "missing_schedule", "inference_width"):
            contract = copy.deepcopy(self.contract)
            if location == "recorded_schedule":
                contract["temporal_band"]["curriculum"]["start_update"] = 20_000
            elif location == "nested_config":
                contract["config"]["temporal_band_curriculum"]["end_update"] = 60_000
            elif location == "missing_schedule":
                del contract["temporal_band"]["curriculum"]
            else:
                contract["temporal_band"]["parameters"]["temporal_band_fixed_sigma_seconds"] = 0.200
            self.write_contract(contract)
            with self.subTest(location=location), self.assertRaises(RuntimeError):
                validate_fixed_band_contract(self.config, self.checkpoint)

    def test_training_rejects_inference_width_different_from_target(self):
        self.config.model.arch.temporal_band_fixed_sigma_seconds = 0.200
        with self.assertRaises((ValueError, RuntimeError)):
            validate_fixed_band_training_config(self.config)

    def test_config_preserves_all_other_parent_settings(self):
        parent = OmegaConf.to_container(compose_config(PARENT_CONFIG_NAME), resolve=True)
        current = OmegaConf.to_container(compose_config(CONFIG_NAME), resolve=True)
        self.assertEqual(current["optim"]["run_until_update"], 150_000)
        self.assertEqual(current["model"]["arch"]["temporal_band_fixed_sigma_seconds"], 0.150)
        self.assertEqual(CosineBandSchedule.from_config(current.pop("temporal_band_curriculum")), schedule())
        for section, key in (("model", "name"), ("ckpts", "save_dir"), ("optim", "run_until_update")):
            current[section][key] = parent[section][key]
        current["model"]["arch"]["temporal_band_fixed_sigma_seconds"] = parent["model"]["arch"]["temporal_band_fixed_sigma_seconds"]
        self.assertEqual(current, parent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
