"""CPU checks for strict AT EMA transfer and declared 18-layer AVT ablations.

Run from this snapshot: PYTHONPATH=src python -m unittest
aligndit.script.misc.test_text_audio_transfer -v
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_text_audio import CFMTextAudio
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.fixed_temporal_band_contract import (
    make_fixed_band_contract,
    validate_fixed_band_contract,
    validate_fixed_band_training_config,
)
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.text_audio_checkpoint import (
    _validate_instantiated_model,
    contract_sha256,
    load_text_audio_pretrained,
    validate_transfer_contract,
)
from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectC2Trainer
from aligndit.script.misc.smoke_test_avt_joint import BASE_ARCH


def tiny_arch(joint_layers=12, text_audio_only=True):
    return {key: value for key, value in {
        **BASE_ARCH, "depth": 18, "n_mm_layers": joint_layers,
        "n_text_layers": joint_layers, "speaker_condition_start_layer": 12,
        "layer_indices_ctc": [6, 12], "text_mask_padding": True,
        "text_audio_only": text_audio_only,
    }.items() if key not in ("mel_dim", "text_num_embeds")}


def contract(arch):
    return {
        "contract_schema_version": 1, "stage": "libritts_text_audio",
        "model": {"backbone": "DiT_VT_MMDiT", "arch": copy.deepcopy(arch),
                  "ctc_lambda": 0.0,
                  "audio_representation": {"channels": 64, "frame_rate": 40,
                                           "sample_rate": 16000, "hop_length": 400}},
        "data": dict.fromkeys(("normalization_sha256", "vocab_sha256", "codec_sha256",
                               "speaker_checkpoint_sha256"), "a" * 64),
    }


def model(arch, vocab_size=16):
    cls = CFMTextAudio if arch.get("text_audio_only", False) else CFM_VT
    return cls(transformer=DiT_VT_MMDiT(**arch, mel_dim=64, text_num_embeds=vocab_size),
               mel_spec_module=PrecomputedAudioRepresentation(64, 16000, 400), num_channels=64)


class TextAudioTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "model_7.pt"
        self.source_contract = contract(tiny_arch())
        self.source = model(self.source_contract["model"]["arch"])
        self.payload = {
            "checkpoint_schema_version": 1, "stage": "libritts_text_audio", "update": 7,
            "training_contract": self.source_contract,
            "training_contract_sha256": contract_sha256(self.source_contract),
            "ema_model_state_dict": {
                "initted": torch.tensor(True), "step": torch.tensor(7),
                **{"ema_model." + key: value.clone() for key, value in self.source.state_dict().items()},
            },
            # Deliberately different online tensors prove the loader uses EMA.
            "model_state_dict": {key: torch.zeros_like(value) for key, value in self.source.state_dict().items()},
        }

    def save(self):
        torch.save(self.payload, self.path)

    def test_all_18_audio_layers_survive_12_to_12_8_6_and_only_removed_streams_are_discarded(self):
        self.save()
        source = self.source.state_dict()
        for joint in (12, 8, 6):
            with self.subTest(joint=joint):
                target_contract = contract(tiny_arch(joint, False))
                target = model(target_contract["model"]["arch"])
                report = load_text_audio_pretrained(target, self.path, target_contract,
                                                   allow_layer_repartition=joint != 12)
                self.assertEqual(target.transformer.speaker_condition_start_layer, 12)
                self.assertEqual(report["source_update"], 7)
                self.assertFalse(report["visual_weights_pretrained"])
                self.assertEqual(report["source_ctc_lambda"], 0.0)
                self.assertEqual(report["loaded_tensors"], len(target.state_dict()))
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, source[key], rtol=0, atol=0, msg=key)
                expected = sorted(key for key in source if key not in target.state_dict())
                self.assertEqual(report["discarded_stream_tensors"], expected)
                self.assertEqual(len(expected), (12 - joint) * 32)
                for key in expected:
                    pieces = key.split(".")
                    self.assertTrue(joint <= int(pieces[2]) < 12)
                    self.assertTrue(pieces[3].startswith(("t_", "v_")))
                for layer in range(18):
                    key = f"transformer.transformer_blocks.{layer}.attn.to_q.weight"
                    torch.testing.assert_close(target.state_dict()[key], source[key], rtol=0, atol=0)

    def test_repartition_requires_opt_in_and_cannot_invent_joint_layers(self):
        with self.assertRaisesRegex(ValueError, "allow_layer_repartition"):
            validate_transfer_contract(self.source_contract, contract(tiny_arch(8, False)))
        with self.assertRaisesRegex(ValueError, "retaining or reducing"):
            validate_transfer_contract(self.source_contract, contract(tiny_arch(13, False)),
                                       allow_layer_repartition=True)
        with self.assertRaises(TypeError):
            validate_transfer_contract(self.source_contract, contract(tiny_arch()), allow_layer_repartition=1)

    def test_equal_shapes_cannot_hide_semantic_or_data_contract_changes(self):
        for field, value in (("pe_attn_head", None), ("text_mask_padding", False),
                             ("attn_mask_enabled", False), ("video_rope_scaled", True),
                             ("speaker_condition_start_layer", 13), ("layer_indices_ctc", [5, 11]),
                             ("text_embedding_average_upsampling", True)):
            target = contract(tiny_arch(12, False))
            target["model"]["arch"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_transfer_contract(self.source_contract, target)
        for field in self.source_contract["data"]:
            target = contract(tiny_arch(12, False))
            target["data"][field] = "b" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_transfer_contract(self.source_contract, target)

    def test_runtime_rope_and_mask_must_match_declared_contract_before_loading(self):
        self.save()
        target_contract = contract(tiny_arch(12, False))
        for field in ("pe_attn_head", "attn_mask_enabled"):
            target = model(target_contract["model"]["arch"])
            attention = target.transformer.transformer_blocks[15].attn.processor
            setattr(attention, field, None if field == "pe_attn_head" else False)
            before = target.transformer.input_embed.proj.weight.detach().clone()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                load_text_audio_pretrained(target, self.path, target_contract)
            torch.testing.assert_close(target.transformer.input_embed.proj.weight, before, rtol=0, atol=0)

    def test_checkpoint_rejects_bad_ema_and_modified_contract(self):
        original = copy.deepcopy(self.payload)
        target_contract = contract(tiny_arch(12, False))
        target = model(target_contract["model"]["arch"])
        for problem in ("hash", "uninitialized", "zero_update", "ema_step", "unexpected_ema", "stage"):
            self.payload = copy.deepcopy(original)
            ema = self.payload["ema_model_state_dict"]
            if problem == "hash":
                self.payload["training_contract"]["model"]["arch"]["pe_attn_head"] = 2
            elif problem == "uninitialized":
                ema["initted"] = torch.tensor(False)
            elif problem == "zero_update":
                self.payload["update"] = 0
            elif problem == "ema_step":
                ema["step"] = torch.tensor(6)
            elif problem == "unexpected_ema":
                ema["mystery"] = torch.tensor(1)
            else:
                self.payload["stage"] = "audio_only"
            self.save()
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                load_text_audio_pretrained(target, self.path, target_contract)

    def test_missing_extra_and_wrong_shape_tensors_fail_without_partial_loading(self):
        original = copy.deepcopy(self.payload)
        target_contract = contract(tiny_arch(8, False))
        target = model(target_contract["model"]["arch"])
        before = {key: value.clone() for key, value in target.state_dict().items()}
        for problem in ("missing", "extra", "shape"):
            self.payload = copy.deepcopy(original)
            ema = self.payload["ema_model_state_dict"]
            if problem == "missing":
                del ema["ema_model.transformer.speaker_proj.weight"]
            elif problem == "extra":
                ema["ema_model.transformer.transformer_blocks.10.attn.extra"] = torch.zeros(1)
            else:
                ema["ema_model.transformer.proj_out.weight"] = torch.zeros(1)
            self.save()
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                load_text_audio_pretrained(target, self.path, target_contract, allow_layer_repartition=True)
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0, msg=key)

    def test_downstream_ema_resets_step_and_uses_loaded_online_weights(self):
        self.save()
        target_contract = contract(tiny_arch(8, False))
        trainer = SemanticVaeDirectC2Trainer.__new__(SemanticVaeDirectC2Trainer)
        trainer.initialization_mode = "libritts_text_audio"
        trainer.text_audio_contract = target_contract
        trainer.allow_layer_repartition = True
        trainer.model = model(target_contract["model"]["arch"])
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value, is_main_process=True,
                                              wait_for_everyone=Mock())
        trainer.ema_model = SimpleNamespace(ema_model=copy.deepcopy(trainer.model),
                                           initted=torch.tensor(False), step=torch.tensor(77))
        trainer.checkpoint_path = str(self.root / "downstream")
        trainer.load_pretrained(self.path)
        self.assertTrue(trainer.ema_model.initted.item())
        self.assertEqual(trainer.ema_model.step.item(), 0)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, trainer.ema_model.ema_model.state_dict()[key], rtol=0, atol=0)
        report = json.loads((Path(trainer.checkpoint_path) / "parent_migration.json").read_text())
        self.assertEqual(report["source_update"], 7)

    def test_original_s2c_dispatch_retains_parent_step_and_strict_migration_calls(self):
        trainer = SemanticVaeDirectC2Trainer.__new__(SemanticVaeDirectC2Trainer)
        trainer.initialization_mode = "s2c"
        trainer.parent_contract_path = "parent_contract.json"
        trainer.expected_parent_sha256 = "a" * 64
        trainer.expected_parent_size = 100
        trainer.expected_parent_contract_sha256 = "b" * 64
        trainer.expected_parent_update = 70000
        trainer.model = torch.nn.Linear(2, 2)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value, is_main_process=True,
                                              wait_for_everyone=Mock())
        trainer.ema_model = SimpleNamespace(ema_model=copy.deepcopy(trainer.model),
                                           initted=torch.tensor(False), step=torch.tensor(0))
        trainer.checkpoint_path = str(self.root / "s2c")
        report = SimpleNamespace(source_key_count=313, target_key_count=802, loaded_key_count=303,
                                 ignored_source_keys=tuple(range(10)), new_target_keys=tuple(range(499)),
                                 to_dict=lambda: {"parent": "s2c"})
        module = "aligndit.model.trainer_semantic_vae_direct"
        with patch(module + ".validate_parent_artifacts") as validate, \
             patch(module + ".load_s2c_ema_state", return_value=({}, 70000)) as load, \
             patch(module + ".migrate_s2c_ema_into_model", return_value=report) as migrate, \
             patch.object(trainer, "_load_text_audio_pretrained") as transfer, \
             patch("torch.cuda.is_available", return_value=False):
            trainer.load_pretrained("parent.pt")
        transfer.assert_not_called()
        validate.assert_called_once()
        load.assert_called_once_with("parent.pt", expected_parent_contract_sha256="b" * 64,
                                     expected_parent_update=70000)
        self.assertEqual(migrate.call_count, 2)
        self.assertIs(migrate.call_args_list[0].args[0], trainer.model)
        self.assertIs(migrate.call_args_list[1].args[0], trainer.ema_model.ema_model)
        self.assertEqual(trainer.ema_model.step.item(), 70000)
        self.assertTrue(trainer.ema_model.initted.item())


class TextAudioConfigTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def configs(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        names = ("pretrain_libritts_text_audio", "finetune_celebvdub_from_libritts_at",
                 "finetune_celebvdub_from_libritts_at_mm8", "finetune_celebvdub_from_libritts_at_mm6")
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            return [compose(config_name=name) for name in names]

    def test_actual_12_at_6_a_contract_matches_all_declared_downstream_configs(self):
        configs = self.configs()
        source = contract(OmegaConf.to_container(configs[0].model.arch, resolve=True))
        source["model"] = OmegaConf.to_container(configs[0].model, resolve=True)
        source_shapes = None
        for i, cfg in enumerate(configs):
            actual_contract = contract(OmegaConf.to_container(cfg.model.arch, resolve=True))
            actual_contract["model"] = OmegaConf.to_container(cfg.model, resolve=True)
            with torch.device("meta"):
                instantiated = model(actual_contract["model"]["arch"], vocab_size=168)
            _validate_instantiated_model(instantiated, actual_contract)
            shapes = {key: value.shape for key, value in instantiated.state_dict().items()}
            if i == 0:
                source_shapes = shapes
                self.assertEqual(len(shapes), 802)
                continue
            with self.subTest(joint=cfg.model.arch.n_mm_layers):
                self.assertEqual(validate_transfer_contract(source, actual_contract,
                    allow_layer_repartition=cfg.ckpts.allow_layer_repartition), (12, cfg.model.arch.n_mm_layers))
                self.assertTrue(cfg.model.arch.text_mask_padding)
                self.assertFalse(cfg.model.arch.text_audio_only)
                self.assertIn("seed666_fp32/vocab.txt", str(cfg.datasets.vocab_path))
                cfg.ckpts.save_dir = str(self.root / cfg.model.name)
                self.assertTrue(validate_fixed_band_training_config(cfg))
                self.assertTrue(all(source_shapes[key] == shape for key, shape in shapes.items()))

    def test_at_initialized_inference_rejects_same_shape_rope_drift(self):
        cfg = self.configs()[2]
        cfg.ckpts.save_dir = str(self.root / cfg.model.name)
        destination = Path(cfg.ckpts.save_dir)
        destination.mkdir()
        band = FixedTemporalBand(dim=768, sigma_seconds=0.15)
        sidecar = {"config": OmegaConf.to_container(cfg, resolve=True),
                   "temporal_band": make_fixed_band_contract(cfg.model.arch, band)}
        (destination / "speaker_training_contract.json").write_text(json.dumps(sidecar))
        validate_fixed_band_contract(cfg, destination / "model.pt")
        cfg.model.arch.pe_attn_head = None
        with self.assertRaisesRegex(RuntimeError, "pe_attn_head"):
            validate_fixed_band_contract(cfg, destination / "model.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
