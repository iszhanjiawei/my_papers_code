"""Audit fixed-13 configuration and LibriTTS initialization without training.

Run from this snapshot with ``PYTHONPATH=src python scripts/test_libritts_parent.py
--real-parent``. The optional integration check reads the real checkpoint,
constructs the online and EMA models on CPU, and compares every loaded tensor.
It performs no forward/backward passes or optimizer updates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from aligndit.model import libritts_pretrained_migration as migration
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from f5_tts.model.trainer import Trainer
from f5_tts.model.utils import get_tokenizer


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
BASELINE_PROJECT = "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va"
BASELINE_REF = "82bd7b1"
BASELINE_CONFIG = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va"
CONFIG_NAME = BASELINE_CONFIG + "_libritts_pretrained"
REAL_PARENT = False


def baseline_file(relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REF}:{BASELINE_PROJECT}/{relative_path}"],
        cwd=REPOSITORY,
    )


def baseline_config(name: str):
    own = OmegaConf.create(baseline_file(f"src/aligndit/config/{name}.yaml").decode())
    defaults = own.pop("defaults", [])
    merged = OmegaConf.create()
    for entry in defaults:
        merged = OmegaConf.merge(merged, own if entry == "_self_" else baseline_config(entry))
    if "_self_" not in defaults:
        merged = OmegaConf.merge(merged, own)
    return merged


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def resolved_config():
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        return compose(config_name=CONFIG_NAME)


class ConfigurationAndVocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = resolved_config()

    def test_resolved_configuration_preserves_fixed13_recipe(self):
        baseline = baseline_config(BASELINE_CONFIG)
        baseline.pop("hydra", None)
        expected = OmegaConf.to_container(baseline, resolve=True)
        actual = OmegaConf.to_container(self.config, resolve=True)
        allowed = {
            "model.name", "ckpts.save_dir", "ckpts.initialization_mode",
            "ckpts.pretrained_path", "ckpts.expected_parent_sha256", "ckpts.expected_parent_size",
            "ckpts.expected_parent_update", "ckpts.parent_contract_path",
            "ckpts.expected_parent_contract_sha256",
        }
        changed = set(differences(expected, actual))
        self.assertTrue(changed <= allowed, f"Unexpected configuration changes: {sorted(changed - allowed)}")
        for section in ("datasets", "optim", "ema"):
            self.assertEqual(actual[section], expected[section], section)
        self.assertEqual(actual["model"]["arch"], expected["model"]["arch"])
        self.assertEqual(actual["model"]["arch"]["temporal_band_fixed_sigma_seconds"], 0.15)
        self.assertEqual(actual["model"]["arch"]["depth"], 18)
        self.assertEqual(actual["model"]["arch"]["n_mm_layers"], 12)
        self.assertEqual(actual["model"]["arch"]["text_mask_padding"], False)
        self.assertEqual(actual["optim"]["run_until_update"], 200000)

    def test_runtime_model_and_training_loop_are_byte_identical(self):
        paths = (
            "src/aligndit/model/backbone/dit_vt_mm.py",
            "src/aligndit/model/cfm_vt.py",
            "src/aligndit/model/modules.py",
            "src/aligndit/model/fixed_temporal_band.py",
            "src/aligndit/model/semantic_vae_dataset.py",
            "src/aligndit/model/speaker_embedding.py",
            "src/aligndit/model/trainer_vt.py",
            "src/aligndit/model/trainer_semantic_vae_direct_ctc_warmup.py",
            "src/aligndit/model/trainer_semantic_vae_direct_speaker.py",
            "src/aligndit/model/trainer_semantic_vae_fixed_band.py",
            "src/f5_tts/model/backbones/dit.py",
            "src/f5_tts/model/trainer.py",
        )
        for relative_path in paths:
            with self.subTest(file=relative_path):
                self.assertEqual((PROJECT / relative_path).read_bytes(), baseline_file(relative_path))

    def test_vocabulary_mapping_preserves_character_and_special_rows(self):
        mapping = migration.build_vocabulary_mapping(self.config.datasets.vocab_path)
        # The hash-verified source vocabulary appends twelve characters to the
        # original 159. Embedding row zero is filler; CTC uses unshifted IDs.
        self.assertEqual(list(mapping["embedding_source_rows"]), list(range(160)))
        self.assertEqual(list(mapping["ctc_source_rows"]), list(range(159)) + [171, 172])
        target = Path(self.config.datasets.vocab_path).read_bytes()
        self.assertEqual(hashlib.sha256(target).hexdigest(), migration.TARGET_VOCAB_SHA256)
        extras = ("(", ")", "/", ":", ";", "[", "]", "{", "}", "ê", "œ", "\u0304")
        reconstructed = target + "".join(token + "\n" for token in extras).encode()
        self.assertEqual(hashlib.sha256(reconstructed).hexdigest(), migration.SOURCE_VOCAB_SHA256)

    def test_vocabulary_with_wrong_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "vocab.txt"
            wrong.write_text(" \na\nb\n")
            with self.assertRaises((RuntimeError, ValueError)):
                migration.build_vocabulary_mapping(wrong)

    def test_empty_checkpoint_directory_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            holder = SimpleNamespace(checkpoint_path=directory)
            self.assertEqual(Trainer.load_checkpoint(holder), 0)


class RealParentMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_PARENT:
            raise unittest.SkipTest("Use --real-parent to audit the actual 5 GB checkpoint")
        torch.set_num_threads(4)
        cls.config = resolved_config()
        cls.path = Path(cls.config.ckpts.pretrained_path)
        cls.checkpoint = torch.load(cls.path, map_location="cpu", weights_only=True, mmap=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "checkpoint"):
            del cls.checkpoint
            gc.collect()

    def test_wrong_checkpoint_sha256_is_rejected(self):
        config = copy.deepcopy(self.config)
        config.ckpts.expected_parent_sha256 = "0" * 64
        with self.assertRaises((RuntimeError, ValueError)):
            migration.validate_libritts_checkpoint_artifact(self.path, config)
        with patch.object(migration, "sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                migration.validate_libritts_checkpoint_artifact(self.path, self.config)

    def test_wrong_parent_update_is_rejected(self):
        wrong = dict(self.checkpoint)
        wrong["update"] = 49999
        with patch.object(migration.torch, "load", return_value=wrong):
            with self.assertRaises((RuntimeError, ValueError)):
                migration.load_libritts_ema_into_model(
                    torch.nn.Module(), self.path, self.config, verify_checkpoint_hash=False
                )

    def test_wrong_embedded_contract_is_rejected(self):
        wrong = dict(self.checkpoint)
        wrong["training_contract_sha256"] = "0" * 64
        with patch.object(migration.torch, "load", return_value=wrong):
            with self.assertRaises((RuntimeError, ValueError)):
                migration.load_libritts_ema_into_model(
                    torch.nn.Module(), self.path, self.config, verify_checkpoint_hash=False
                )

    def test_all_online_and_ema_tensors_load_exactly_without_optimizer_resume(self):
        config = self.config
        torch.manual_seed(int(config.seed))
        token_map, token_count = get_tokenizer(config.datasets.vocab_path, "custom")
        backbone = hydra.utils.get_class(f"aligndit.model.{config.model.backbone}")
        audio = config.model.audio_representation
        online = CFM_VT(
            transformer=backbone(**config.model.arch, text_num_embeds=token_count, mel_dim=audio.channels),
            mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
            num_channels=audio.channels,
            vocab_char_map=token_map,
            audio_video_ratio=config.model.arch.audio_video_ratio,
            ctc_lambda=config.model.ctc_lambda,
        )
        ema_model = copy.deepcopy(online)
        optimizer = torch.optim.AdamW(online.parameters(), lr=float(config.optim.learning_rate))
        self.assertEqual(len(optimizer.state), 0)
        online_report = migration.load_libritts_ema_into_model(online, self.path, config)
        ema_report = migration.load_libritts_ema_into_model(
            ema_model, self.path, config, verify_checkpoint_hash=False
        )
        self.assertEqual(online_report, ema_report)
        self.assertEqual(online_report["parent_ema_step"], 50000)
        self.assertEqual(online_report["downstream_initial_update"], 0)
        self.assertFalse(online_report["optimizer_restored"])
        source = {
            key.removeprefix("ema_model."): value
            for key, value in self.checkpoint["ema_model_state_dict"].items()
            if key.startswith("ema_model.")
        }
        online_state = online.state_dict()
        ema_state = ema_model.state_dict()
        self.assertEqual(len(online_state), 802)
        self.assertEqual(set(online_state), set(source))
        embedding_key = "transformer.text_embed.text_embed.weight"
        ctc_keys = {
            f"transformer.projectors_ctc.{head}.model.6.{kind}"
            for head in (0, 1) for kind in ("weight", "bias")
        }
        remapped = []
        for key, value in online_state.items():
            expected = source[key]
            if key == embedding_key:
                expected = expected[:160]
                remapped.append(key)
            elif key in ctc_keys:
                expected = expected[torch.tensor(list(range(159)) + [171, 172])]
                remapped.append(key)
            with self.subTest(tensor=key):
                self.assertTrue(torch.equal(value, expected), f"Online mismatch: {key}")
                self.assertTrue(torch.equal(ema_state[key], expected), f"EMA mismatch: {key}")
        self.assertEqual(len(remapped), 5)
        self.assertEqual(len(optimizer.state), 0, "Parent optimizer must not be resumed")
        self.assertEqual(optimizer.param_groups[0]["lr"], float(config.optim.learning_rate))
        self.assertFalse(list(online.transformer.temporal_band.parameters()))
        self.assertFalse(online.transformer.temporal_band.state_dict())
        print(json.dumps({
            "result": "PASS", "online_tensors": 802, "ema_tensors": 802,
            "shape_exact_tensors": 797, "vocabulary_remapped_tensors": remapped,
            "optimizer_state_entries": len(optimizer.state), "optimizer_updates_performed": 0,
            "migration": {key: online_report[key] for key in (
                "parent_path", "parent_sha256", "parent_ema_step", "downstream_initial_update",
                "source_key_count", "target_key_count", "loaded_key_count", "shape_identical_key_count",
                "ignored_source_keys", "new_target_keys", "optimizer_restored",
            )},
        }, indent=2, ensure_ascii=False), flush=True)
        del optimizer, online_state, ema_state, online, ema_model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parent", action="store_true")
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    arguments, remaining = parser.parse_known_args()
    REAL_PARENT = arguments.real_parent
    BASELINE_REF = arguments.baseline_ref
    unittest.main(argv=[__file__, *remaining], verbosity=2)
