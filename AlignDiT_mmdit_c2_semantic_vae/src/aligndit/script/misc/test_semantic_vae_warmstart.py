import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW

from aligndit.model.backbone.dit_notext import DiT_noText
from aligndit.model.cfm_notext import CFM_notext
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_warmstart import (
    S1_EXPECTED_SHAPE_MISMATCHES,
    S1_QK_NORM_PATTERN,
    configure_stage_parameters,
    load_parent_ema_weights,
)
from aligndit.model.trainer_semantic_vae_warmstart import (
    SemanticVaeWarmStartTrainer,
    has_local_training_checkpoint,
)


def _make_model(*, channels: int, qk_norm: str | None, projector_strides: tuple[int, int]) -> CFM_notext:
    transformer = DiT_noText(
        dim=16,
        depth=18,
        heads=2,
        dim_head=8,
        dropout=0.0,
        ff_mult=2,
        mel_dim=channels,
        qk_norm=qk_norm,
        pe_attn_head=1,
        attn_mask_enabled=True,
        mask_during_training=True,
        layer_indices=[12],
        projector_dim=16,
        z_dim=8,
        projector_strides=projector_strides,
        padding_safe_projector=True,
    )
    return CFM_notext(
        transformer=transformer,
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=16_000,
            hop_length=160 if channels == 80 else 400,
        ),
        num_channels=channels,
        proj_lambda=0.0,
    )


def _ema_state(model: torch.nn.Module, value_offset: float = 0.0, *, step: int = 500_000) -> dict[str, torch.Tensor]:
    result = {"initted": torch.tensor(True), "step": torch.tensor(step)}
    for key, value in model.state_dict().items():
        tensor = value.detach().clone()
        if tensor.is_floating_point():
            tensor = tensor + value_offset
        result[f"ema_model.{key}"] = tensor
    return result


def _write_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    update: int,
    stage: str | None,
    contract_sha256: str | None,
    ema_offset: float = 0.0,
) -> None:
    online_state = {
        key: value.detach().clone() + 17 if value.is_floating_point() else value.detach().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "update": update,
            "warmstart_stage": stage,
            "training_contract_sha256": contract_sha256,
            "model_state_dict": online_state,
            "ema_model_state_dict": _ema_state(model, ema_offset, step=update),
        },
        path,
    )


def _optimizer_parameter_ids(groups) -> list[int]:
    return [id(parameter) for group in groups for parameter in group["params"]]


class SemanticVaeWarmStartTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        random.seed(1234)

    def test_s1_loads_only_permitted_mel_ema_tensors(self):
        source = _make_model(channels=80, qk_norm=None, projector_strides=(2, 1))
        target = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        target_initial = {key: value.detach().clone() for key, value in target.state_dict().items()}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "mel.pt"
            _write_checkpoint(
                checkpoint_path,
                source,
                update=500_000,
                stage=None,
                contract_sha256=None,
                ema_offset=0.25,
            )
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            report = load_parent_ema_weights(
                target,
                checkpoint_path,
                stage="s1",
                expected_parent_update=500_000,
                expected_parent_sha256=checkpoint_sha256,
                expected_parent_size=checkpoint_path.stat().st_size,
            )

        source_ema = {
            key.removeprefix("ema_model."): value
            for key, value in _ema_state(source, 0.25).items()
            if key.startswith("ema_model.")
        }
        self.assertEqual(report.source_key_count, 277)
        self.assertEqual(report.target_key_count, 313)
        self.assertEqual(report.loaded_key_count, 263)
        self.assertEqual(len(report.reset_keys), 50)
        self.assertEqual(set(report.shape_mismatches), S1_EXPECTED_SHAPE_MISMATCHES)
        # The exact parameter fraction depends on the hidden width.  The
        # production 768-D model is checked separately against 88.6682%; this
        # compact fixture should only prove that the reset is non-empty and
        # that most parameters still come from the parent EMA.
        self.assertGreater(report.loaded_fraction, 0.85)
        self.assertLess(report.loaded_fraction, 1.0)

        reset_keys = set(report.reset_keys)
        for key, value in target.state_dict().items():
            if key in reset_keys:
                torch.testing.assert_close(value, target_initial[key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, source_ema[key], rtol=0, atol=0)
        qk_norm_keys = [key for key in reset_keys if S1_QK_NORM_PATTERN.fullmatch(key)]
        self.assertEqual(len(qk_norm_keys), 36)

    def test_s1_schema_drift_fails_closed(self):
        source = _make_model(channels=80, qk_norm=None, projector_strides=(2, 1))
        target = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing_path = directory / "missing.pt"
            _write_checkpoint(missing_path, source, update=500_000, stage=None, contract_sha256=None)
            checkpoint = torch.load(missing_path, weights_only=True)
            checkpoint["ema_model_state_dict"].pop("ema_model.transformer.time_embed.time_mlp.0.weight")
            torch.save(checkpoint, missing_path)
            with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                load_parent_ema_weights(target, missing_path, stage="s1", expected_parent_update=500_000)

            extra_path = directory / "extra.pt"
            _write_checkpoint(extra_path, source, update=500_000, stage=None, contract_sha256=None)
            checkpoint = torch.load(extra_path, weights_only=True)
            checkpoint["ema_model_state_dict"]["ema_model.transformer.unexpected.weight"] = torch.ones(1)
            torch.save(checkpoint, extra_path)
            with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                load_parent_ema_weights(target, extra_path, stage="s1", expected_parent_update=500_000)

            mismatch_path = directory / "mismatch.pt"
            _write_checkpoint(mismatch_path, source, update=500_000, stage=None, contract_sha256=None)
            checkpoint = torch.load(mismatch_path, weights_only=True)
            key = "ema_model.transformer.time_embed.time_mlp.0.weight"
            checkpoint["ema_model_state_dict"][key] = checkpoint["ema_model_state_dict"][key][:-1]
            torch.save(checkpoint, mismatch_path)
            with self.assertRaisesRegex(RuntimeError, "shape mismatch set"):
                load_parent_ema_weights(target, mismatch_path, stage="s1", expected_parent_update=500_000)

            stale_ema_path = directory / "stale_ema.pt"
            _write_checkpoint(stale_ema_path, source, update=500_000, stage=None, contract_sha256=None)
            checkpoint = torch.load(stale_ema_path, weights_only=True)
            checkpoint["ema_model_state_dict"]["step"] = torch.tensor(499_999)
            torch.save(checkpoint, stale_ema_path)
            with self.assertRaisesRegex(RuntimeError, "EMA step mismatch"):
                load_parent_ema_weights(target, stale_ema_path, stage="s1", expected_parent_update=500_000)

            uninitted_ema_path = directory / "uninitted_ema.pt"
            _write_checkpoint(uninitted_ema_path, source, update=500_000, stage=None, contract_sha256=None)
            checkpoint = torch.load(uninitted_ema_path, weights_only=True)
            checkpoint["ema_model_state_dict"]["initted"] = torch.tensor(False)
            torch.save(checkpoint, uninitted_ema_path)
            with self.assertRaisesRegex(RuntimeError, "EMA is not initialized"):
                load_parent_ema_weights(target, uninitted_ema_path, stage="s1", expected_parent_update=500_000)

    def test_later_stage_requires_exact_adjacent_ema_parent(self):
        source = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        target = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        contract_sha = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "s1.pt"
            _write_checkpoint(
                checkpoint_path,
                source,
                update=10_000,
                stage="s1",
                contract_sha256=contract_sha,
                ema_offset=0.5,
            )
            report = load_parent_ema_weights(
                target,
                checkpoint_path,
                stage="s2a",
                expected_parent_update=10_000,
                expected_parent_stage="s1",
                expected_parent_contract_sha256=contract_sha,
            )
            self.assertEqual(report.loaded_key_count, 313)
            self.assertEqual(report.loaded_fraction, 1.0)
            with self.assertRaisesRegex(RuntimeError, "parent stage mismatch"):
                load_parent_ema_weights(
                    target,
                    checkpoint_path,
                    stage="s2a",
                    expected_parent_update=10_000,
                    expected_parent_stage="s2b",
                    expected_parent_contract_sha256=contract_sha,
                )
            with self.assertRaisesRegex(RuntimeError, "parent contract mismatch"):
                load_parent_ema_weights(
                    target,
                    checkpoint_path,
                    stage="s2a",
                    expected_parent_update=10_000,
                    expected_parent_stage="s1",
                    expected_parent_contract_sha256="b" * 64,
                )

    def test_stage_freezing_and_optimizer_groups_are_exact(self):
        model = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        stage_settings = {
            "s1": {"interface": 1e-4},
            "s2a": {"interface": 5e-5, "backbone": 1e-5},
            "s2b": {"interface": 3e-5, "backbone": 1e-5},
            "s2c": {"interface": 2e-5, "projector": 2e-5, "early_backbone": 5e-6, "backbone": 1e-5},
        }
        reports = {}
        for stage, learning_rates in stage_settings.items():
            groups, report = configure_stage_parameters(
                model,
                stage=stage,
                learning_rates=learning_rates,
                weight_decay=0.01,
            )
            reports[stage] = report
            optimizer_ids = _optimizer_parameter_ids(groups)
            trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
            self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
            self.assertEqual(set(optimizer_ids), set(trainable_ids))
            for group in groups:
                if group["group_name"].endswith("no_decay"):
                    self.assertEqual(group["weight_decay"], 0.0)

        self.assertEqual(
            set(reports["s1"].trainable_names),
            {
                "transformer.input_embed.proj.weight",
                "transformer.input_embed.proj.bias",
                "transformer.proj_out.weight",
                "transformer.proj_out.bias",
            },
        )
        self.assertNotIn("transformer.transformer_blocks.11.attn.to_q.weight", reports["s2a"].trainable_names)
        self.assertIn("transformer.transformer_blocks.12.attn.to_q.weight", reports["s2a"].trainable_names)
        self.assertNotIn("transformer.transformer_blocks.5.attn.to_q.weight", reports["s2b"].trainable_names)
        self.assertIn("transformer.transformer_blocks.6.attn.to_q.weight", reports["s2b"].trainable_names)
        self.assertEqual(reports["s2c"].frozen_numel, 0)

    def test_s1_second_step_reaches_new_input_interface_only(self):
        model = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        model.transformer.layer_map = {}
        groups, _ = configure_stage_parameters(
            model,
            stage="s1",
            learning_rates={"interface": 1e-3},
            weight_decay=0.01,
        )
        optimizer = AdamW(groups)
        input_before = model.transformer.input_embed.proj.weight.detach().clone()
        output_before = model.transformer.proj_out.weight.detach().clone()
        frozen = model.transformer.transformer_blocks[0].attn.to_q.weight
        frozen_before = frozen.detach().clone()
        latent = torch.randn(2, 12, 64)
        lengths = torch.tensor([12, 10])
        feature = torch.randn(2, 12, 8)

        for _ in range(2):
            loss, _, _, _ = model(latent, lens=lengths, feature=feature, feature_lens=lengths)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.assertFalse(torch.equal(model.transformer.proj_out.weight, output_before))
        self.assertFalse(torch.equal(model.transformer.input_embed.proj.weight, input_before))
        torch.testing.assert_close(frozen, frozen_before, rtol=0, atol=0)
        self.assertIsNone(frozen.grad)

    def test_s2c_projector_receives_gradient_and_lambda_ramps(self):
        model = _make_model(channels=64, qk_norm="rms_norm", projector_strides=(1, 1))
        configure_stage_parameters(
            model,
            stage="s2c",
            learning_rates={"interface": 2e-5, "projector": 2e-5, "early_backbone": 5e-6, "backbone": 1e-5},
            weight_decay=0.01,
        )
        model.proj_lambda = 0.1
        lengths = torch.tensor([12, 10])
        loss, _, _, _ = model(
            torch.randn(2, 12, 64),
            lens=lengths,
            feature=torch.randn(2, 12, 8),
            feature_lens=lengths,
        )
        loss.backward()
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.transformer.projectors.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0)

        trainer = SemanticVaeWarmStartTrainer.__new__(SemanticVaeWarmStartTrainer)
        trainer.stage = "s2c"
        trainer.projection_target_lambda = 0.1
        trainer.projection_ramp_updates = 5000
        self.assertEqual(trainer.projection_lambda_at(0), 0.0)
        self.assertAlmostEqual(trainer.projection_lambda_at(2500), 0.05)
        self.assertAlmostEqual(trainer.projection_lambda_at(5000), 0.1)
        self.assertAlmostEqual(trainer.projection_lambda_at(50_000), 0.1)

    def test_local_checkpoint_detection_and_stage_contract_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.assertFalse(has_local_training_checkpoint(directory))
            (directory / "pretrained_external.pt").touch()
            with self.assertRaisesRegex(RuntimeError, "Unexpected serialized files"):
                has_local_training_checkpoint(directory)
            (directory / "pretrained_external.pt").unlink()
            (directory / "model_last.pt").touch()
            self.assertTrue(has_local_training_checkpoint(directory))

        trainer = SemanticVaeWarmStartTrainer.__new__(SemanticVaeWarmStartTrainer)
        trainer.stage = "s1"
        trainer.training_contract_sha256 = "a" * 64
        trainer._validate_checkpoint_contract(
            {
                "checkpoint_schema_version": 1,
                "training_contract_sha256": "a" * 64,
                "warmstart_stage": "s1",
            },
            "model_last.pt",
        )
        with self.assertRaisesRegex(RuntimeError, "warm-start stage mismatch"):
            trainer._validate_checkpoint_contract(
                {
                    "checkpoint_schema_version": 1,
                    "training_contract_sha256": "a" * 64,
                    "warmstart_stage": "s2a",
                },
                "model_last.pt",
            )


if __name__ == "__main__":
    unittest.main()
