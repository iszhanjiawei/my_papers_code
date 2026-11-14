"""Independent contracts for adding REPA to the fixed-13 LibriTTS GRID run.

Run with ``PYTHONPATH=src python scripts/test_grid_repa_transfer.py``. Optional
``--real-parent`` verifies every online/EMA tensor using the real 5 GB parent;
``--audit-cache`` hashes all 29,531 training teacher features and source WAVs.
No training optimizer updates or GPU operations are performed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from aligndit.model import libritts_pretrained_migration as migration
from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.repa import masked_repa_cosine_loss, validate_repa_feature_array
from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectC2Trainer
from aligndit.script.train.finetune_grid import dataset_from_config, training_schedule_from_config
from f5_tts.model.trainer import Trainer
from f5_tts.model.utils import get_tokenizer


PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT.parent / (
    "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_"
    "fixed_band_window13_av_va_libritts_pretrained_grid"
)
REFERENCE = PROJECT.parent / (
    "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_"
    "repa_wavlm_base_plus_grid"
)
REAL_PARENT = AUDIT_CACHE = False
REPA_KEYS = {
    f"transformer.repa_projector.{layer}.{kind}"
    for layer in (0, 2, 4) for kind in ("weight", "bias")
}


def config(project=PROJECT):
    with initialize_config_dir(version_base="1.3", config_dir=str(project / "src/aligndit/config")):
        return compose(config_name="finetune_grid")


def changes(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        changed = set()
        for key in left.keys() | right.keys():
            name = f"{prefix}.{key}" if prefix else key
            changed.update({name} if key not in left or key not in right else changes(left[key], right[key], name))
        return changed
    return set() if left == right else {prefix}


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def miniature(*, repa=True, checkpoint=False):
    """Keep all 12 AVT + 6 audio blocks, reducing only widths for CPU checks."""
    torch.manual_seed(712)
    model = DiT_VT_MMDiT(
        dim=32, depth=18, heads=4, dim_head=8, ff_mult=2, mel_dim=64,
        text_num_embeds=16, text_dim=16, text_mask_padding=False,
        qk_norm="rms_norm", conv_layers=1, pe_attn_head=1,
        attn_mask_enabled=True, checkpoint_activations=checkpoint,
        use_conformer=False, layer_indices_ctc=[6, 12], ctc_sampling_ratios=[1, 1],
        n_mm_layers=12, n_text_layers=12, prompt_isolated_ca=False,
        audio_video_ratio=1, video_dim=16, video_rope_scaled=False,
        text_joint_attention=True, speaker_dim=192, speaker_condition_start_layer=12,
        temporal_band_enabled=True, temporal_band_mode="fixed",
        temporal_band_fixed_sigma_seconds=0.15, temporal_band_fixed_offset_seconds=0.0,
        temporal_band_audio_fps=40.0, temporal_band_video_fps=40.0,
        temporal_band_bidirectional=True,
        **({"repa_layer": 9, "repa_target_dim": 24, "repa_projector_dim": 48} if repa else {}),
    )
    # Deterministic nonzero inherited weights avoid vacuous zero-output tests;
    # a separate RNG makes these independent of projector initialization.
    generator = torch.Generator().manual_seed(918)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "repa_projector" not in name and (
                "attn_norm.linear" in name or "norm_out.linear" in name
                or "proj_out" in name or "speaker_proj" in name
            ):
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
    return model


def inputs():
    torch.manual_seed(103)
    mask = torch.arange(12)[None] < torch.tensor([12, 10])[:, None]
    text_mask = torch.arange(4)[None] < torch.tensor([4, 3])[:, None]
    generated = mask.clone()
    generated[:, :3] = False
    return dict(
        x=torch.randn(2, 12, 64), cond=torch.randn(2, 12, 64),
        text=torch.randint(0, 16, (2, 4)).masked_fill(~text_mask, -1),
        video=torch.randn(2, 12, 16), time=torch.tensor([0.2, 0.8]),
        mask=mask, text_mask=text_mask, video_mask=mask.clone(),
        complementary_mask=mask & ~generated, generation_mask=generated,
        speaker_embedding=torch.randn(2, 192), cache=False,
    )


class MechanicsTests(unittest.TestCase):
    def test_exact_reference_loss_and_teacher_contract(self):
        relative = "src/aligndit/model/repa.py"
        self.assertEqual((PROJECT / relative).read_bytes(), (REFERENCE / relative).read_bytes())

    def test_interpolation_masks_padding_and_teacher_detachment(self):
        torch.manual_seed(99)
        student = torch.randn(2, 6, 7, requires_grad=True)
        teacher = torch.randn(2, 8, 7, requires_grad=True)
        student_lens, teacher_lens = torch.tensor([6, 4]), torch.tensor([8, 5])
        generated = torch.tensor([[False, False, True, True, True, True],
                                  [False, True, True, True, True, True]])
        active = generated & (torch.arange(6)[None] < student_lens[:, None])
        expected_sum = torch.tensor(0.)
        for index, (slen, tlen) in enumerate(zip(student_lens, teacher_lens)):
            target = F.interpolate(teacher[index, :tlen].detach().T[None], size=int(slen),
                                   mode="linear", align_corners=False)[0].T
            chosen = active[index, :slen]
            expected_sum += F.cosine_similarity(student[index, :slen][chosen], target[chosen], dim=-1).sum()
        expected = 1 - expected_sum / active.sum()
        actual = masked_repa_cosine_loss(student, teacher, teacher_lens, student_lens, generated)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0))
        altered_student, altered_teacher = student.detach().clone(), teacher.detach().clone()
        altered_student[~active] = 1e6
        altered_teacher[1, 5:] = -1e6
        self.assertTrue(torch.equal(actual.detach(), masked_repa_cosine_loss(
            altered_student, altered_teacher, teacher_lens, student_lens, generated)))
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(torch.count_nonzero(student.grad[~active]).item(), 0)
        self.assertTrue(torch.isfinite(student.grad[active]).all())
        self.assertGreater(torch.count_nonzero(student.grad[active]).item(), 0)
        with self.assertRaisesRegex(RuntimeError, "no valid frames"):
            masked_repa_cosine_loss(student, teacher, teacher_lens, student_lens, torch.zeros_like(generated))

    def test_avt_tenth_block_tap_and_inference_neutrality(self):
        baseline, augmented = miniature(repa=False).eval(), miniature().eval()
        common = baseline.state_dict()
        self.assertEqual(set(augmented.state_dict()) - set(common), {key.removeprefix("transformer.") for key in REPA_KEYS})
        self.assertTrue(all(torch.equal(value, augmented.state_dict()[key]) for key, value in common.items()))
        batch, captured = inputs(), {}
        handle = augmented.transformer_blocks[9].register_forward_hook(
            lambda _module, _args, result: captured.update(audio=result[0].detach(), streams=len(result)))
        try:
            with torch.inference_mode():
                expected, expected_ctc = baseline(**batch)
                actual, actual_ctc, projection = augmented(**batch, return_repa=True)
        finally:
            handle.remove()
        self.assertGreater(torch.count_nonzero(expected).item(), 0)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(captured["streams"], 3)
        with torch.inference_mode():
            self.assertTrue(torch.equal(projection, augmented.repa_projector(captured["audio"])))
        for layer in expected_ctc:
            for key in ("z_tilde", "z_lens"):
                self.assertTrue(torch.equal(expected_ctc[layer][key], actual_ctc[layer][key]))
        self.assertFalse(list(augmented.temporal_band.parameters()))
        self.assertFalse(augmented.temporal_band.state_dict())
        self.assertTrue(torch.all(augmented.last_temporal_band_sigma_seconds == 0.15))
        # Sampling has no teacher dependency or projector forward at either CFG setting.
        plain = CFM_VT(transformer=baseline, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03)
        with_repa = CFM_VT(transformer=augmented, num_channels=64, audio_video_ratio=1,
                           ctc_lambda=0.03, repa_lambda=0.1)
        sample = dict(cond=batch["cond"][:, :3], text=batch["text"], video=batch["video"],
                      duration=torch.tensor([12, 10]), lens=torch.tensor([3, 3]),
                      speaker_embedding=batch["speaker_embedding"], steps=1, use_epss=False, seed=0)
        with patch.object(augmented.repa_projector, "forward", side_effect=AssertionError("sampling used REPA head")):
            for guidance in (0.0, 1.0):
                expected, _ = plain.sample(**sample, cfg_strength=guidance, cfg_strength_v=guidance)
                actual, _ = with_repa.sample(**sample, cfg_strength=guidance, cfg_strength_v=guidance)
                self.assertTrue(torch.equal(expected, actual))

    def test_repa_gradients_reach_projector_and_backbone_with_checkpointing(self):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                backbone = miniature(checkpoint=checkpoint)
                model = CFM_VT(transformer=backbone, num_channels=64, audio_video_ratio=1,
                               ctc_lambda=0.03, repa_lambda=0.1,
                               audio_drop_prob=0., cond_drop_prob=0., text_drop_prob=0., video_drop_prob=0.)
                batch = inputs()
                teacher = torch.randn(2, 15, 24, requires_grad=True)
                with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                    loss, components, _, _ = model(
                        inp=batch["x"], text=batch["text"], video=batch["video"],
                        lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 3]),
                        video_lens=torch.tensor([12, 10]), speaker_embedding=batch["speaker_embedding"],
                        repa_features=teacher, repa_feature_lens=torch.tensor([15, 12]))
                self.assertEqual(set(components), {"diff_loss", "ctc_loss", "repa_loss"})
                self.assertTrue(torch.isfinite(loss))
                expected_loss = components["diff_loss"] + .03 * components["ctc_loss"] + .1 * components["repa_loss"]
                self.assertAlmostEqual(loss.item(), expected_loss, places=5)
                loss.backward()
                self.assertIsNone(teacher.grad)
                for name, parameter in backbone.repa_projector.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)
                # REPA alone must supervise upstream audio representations, independently of flow/CTC.
                backbone.zero_grad(set_to_none=True)
                _, _, prediction = backbone(**batch, return_repa=True)
                auxiliary = masked_repa_cosine_loss(prediction, teacher, torch.tensor([15, 12]),
                                                   torch.tensor([12, 10]), batch["generation_mask"])
                auxiliary.backward()
                upstream = backbone.input_embed.proj.weight.grad
                self.assertIsNotNone(upstream)
                self.assertTrue(torch.isfinite(upstream).all())
                self.assertGreater(torch.count_nonzero(upstream).item(), 0)

    def test_transfer_allowlist_rejects_missing_extra_and_malformed_parameters(self):
        source = {f"original.{index}": torch.empty(0, device="meta") for index in range(802)}
        new = {key: torch.empty(shape, device="meta") for key, shape in migration.REPA_PROJECTOR_SHAPES.items()}
        target = {**source, **new}
        self.assertEqual(set(migration.validate_repa_transfer_keys(source, target)), REPA_KEYS)
        malformed = [
            {**target, "unexpected.weight": torch.empty(0)},
            {key: value for key, value in target.items() if key != "original.0"},
            {**target, "transformer.repa_projector.0.weight": torch.empty(1, device="meta")},
        ]
        for candidate in malformed:
            with self.assertRaises(ValueError):
                migration.validate_repa_transfer_keys(source, candidate)


class ExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.dataset = dataset_from_config(cls.config)

    def test_only_repa_outputs_and_stop_change_configuration(self):
        baseline = OmegaConf.to_container(config(BASELINE), resolve=True)
        actual = OmegaConf.to_container(self.config, resolve=True)
        allowed = {"model.name", "ckpts.save_dir", "optim.run_until_update", "model.repa_lambda",
                   "model.arch.repa_layer", "model.arch.repa_target_dim", "model.arch.repa_projector_dim"}
        allowed.update(f"datasets.{key}" for key in (
            "repa_feature_cache_dir", "repa_feature_dim", "repa_model_id", "repa_model_revision",
            "repa_checkpoint_sha256", "repa_teacher_layer"))
        self.assertFalse(changes(baseline, actual) - allowed, sorted(changes(baseline, actual) - allowed))
        self.assertEqual(actual["optim"]["run_until_update"], 150000)
        self.assertEqual(actual["optim"]["epochs"], 1000)
        self.assertEqual(actual["ckpts"]["pretrained_path"], baseline["ckpts"]["pretrained_path"])
        self.assertEqual(actual["model"]["repa_lambda"], .1)
        for key, value in {"repa_layer": 9, "repa_target_dim": 768, "repa_projector_dim": 2048,
                           "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
                           "text_joint_attention": True, "temporal_band_fixed_sigma_seconds": .15,
                           "temporal_band_bidirectional": True}.items():
            self.assertEqual(actual["model"]["arch"][key], value, key)
        self.assertNotEqual(actual["ckpts"]["save_dir"], baseline["ckpts"]["save_dir"])
        self.assertNotEqual(actual["model"]["name"], baseline["model"]["name"])
        reference = OmegaConf.to_container(config(REFERENCE), resolve=True)
        for key in ("repa_feature_dim", "repa_model_id", "repa_model_revision", "repa_checkpoint_sha256", "repa_teacher_layer"):
            self.assertEqual(actual["datasets"][key], reference["datasets"][key], key)

    def test_original_learning_rate_horizon_is_preserved(self):
        schedule = training_schedule_from_config(self.config, self.dataset, world_size=4)
        self.assertEqual(schedule["frame_batches"], 985)
        self.assertEqual(schedule["per_rank_batches"], 246)
        self.assertEqual(schedule["effective_max_updates"], 246000)
        self.assertEqual(schedule["scheduler_total_steps"], 985000)
        self.assertEqual(schedule["scheduler_warmup_steps"], 80000)
        self.assertEqual(schedule["run_until_update"], 150000)

    def test_training_split_and_real_batch_repa_binding(self):
        dataset = self.dataset
        self.assertEqual(len(dataset), 29531)
        self.assertEqual(dataset.split, "train")
        self.assertTrue(all(row["split"] == "train" for row in dataset.records))
        self.assertEqual(dataset.cache_contract["manifest_sha256"], "46532a02d9d9ef03127aa358bab4788640575f19b3d11efc850086ef107298e7")
        indices = (0, len(dataset) // 2, len(dataset) - 1)
        samples = [dataset[index] for index in indices]
        batch = dataset.collate_fn(samples)
        self.assertEqual(batch["repa_features"].shape[0], 3)
        self.assertEqual(batch["repa_features"].shape[-1], 768)
        self.assertTrue(torch.isfinite(batch["repa_features"]).all())
        for index, original in enumerate(indices):
            row = dataset.records[original]
            raw = np.load(dataset.cache_root / "repa" / Path(row["audio_relative_path"]).with_suffix(".npy"), allow_pickle=False)
            self.assertEqual(batch["repa_feature_lengths"][index].item(), len(raw))
            np.testing.assert_array_equal(batch["repa_features"][index, :len(raw)].numpy(), raw)

    def test_all_teacher_cache_arrays_and_audio_bindings(self):
        if not AUDIT_CACHE:
            self.skipTest("Pass --audit-cache for the complete 29,531-artifact audit")
        dataset = self.dataset
        fingerprint = dataset.inventory_metadata["extraction_sha256"]
        for index, row in enumerate(dataset.records, 1):
            relative = Path(row["audio_relative_path"])
            feature = dataset.cache_root / "repa" / relative.with_suffix(".npy")
            self.assertTrue(feature.is_file() and not feature.is_symlink(), str(feature))
            raw = np.load(feature, allow_pickle=False)
            validate_repa_feature_array(raw, source=feature)
            expected_frames = (row["original_num_samples"] - 400) // 320 + 1
            self.assertEqual(len(raw), expected_frames, str(feature))
            receipt = json.loads((dataset.cache_root / "receipts/repa" / relative.with_suffix(".json")).read_text())
            for key, value in {"utterance_key": row["utterance_key"], "source_sha256": row["source_sha256"],
                               "extraction_sha256": fingerprint, "stage": "repa", "artifact_sha256": sha256(feature)}.items():
                self.assertEqual(receipt[key], value, f"{feature}: {key}")
            audio = dataset.cache_root / "audio" / relative
            self.assertEqual(receipt["audio_sha256"], sha256(audio), str(audio))
            if index % 5000 == 0:
                print(f"Verified {index}/{len(dataset)} REPA arrays and actual source-audio hashes", flush=True)


class RealParentTests(unittest.TestCase):
    def test_real_parent_all_802_tensors_plus_fresh_head_and_ema_carry(self):
        if not REAL_PARENT:
            self.skipTest("Pass --real-parent to verify the real LibriTTS checkpoint and online/EMA tensors")
        cfg = config()
        torch.manual_seed(int(cfg.seed))
        token_map, token_count = get_tokenizer(cfg.datasets.vocab_path, "custom")
        backbone = hydra.utils.get_class(f"aligndit.model.{cfg.model.backbone}")
        audio = cfg.model.audio_representation
        online = CFM_VT(
            transformer=backbone(**cfg.model.arch, text_num_embeds=token_count, mel_dim=audio.channels),
            mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
            num_channels=audio.channels, vocab_char_map=token_map,
            audio_video_ratio=cfg.model.arch.audio_video_ratio,
            ctc_lambda=cfg.model.ctc_lambda, repa_lambda=cfg.model.repa_lambda,
        )
        fresh_head = {key: value.clone() for key, value in online.state_dict().items() if key in REPA_KEYS}
        ema = SimpleNamespace(ema_model=copy.deepcopy(online), initted=torch.tensor(False), step=torch.tensor(0))
        optimizer = torch.optim.AdamW(online.parameters(), lr=float(cfg.optim.learning_rate))
        accelerator = SimpleNamespace(unwrap_model=lambda model: model, wait_for_everyone=lambda: None)
        with tempfile.TemporaryDirectory(prefix="grid_repa_parent_") as directory:
            holder = SimpleNamespace(is_main=True, accelerator=accelerator, model=online, ema_model=ema,
                                     parent_model_cfg=cfg, checkpoint_path=directory)
            self.assertEqual(Trainer.load_checkpoint(holder), 0)
            SemanticVaeDirectC2Trainer._load_libritts_pretrained(holder, cfg.ckpts.pretrained_path)
            report = json.loads((Path(directory) / "parent_migration.json").read_text())
        self.assertEqual(report["loaded_key_count"], 802)
        self.assertEqual(report["target_key_count"], 808)
        self.assertEqual(set(report["new_target_keys"]), REPA_KEYS)
        self.assertEqual(report["shape_identical_key_count"], 797)
        self.assertEqual(report["downstream_initial_update"], 0)
        self.assertEqual(report["parent_ema_step"], 50000)
        self.assertFalse(report["optimizer_restored"])
        self.assertEqual(optimizer.state, {})
        self.assertTrue(ema.initted.item())
        self.assertEqual(ema.step.item(), 50000)
        checkpoint = torch.load(cfg.ckpts.pretrained_path, map_location="cpu", weights_only=True, mmap=True)
        source = {key.removeprefix("ema_model."): value for key, value in checkpoint["ema_model_state_dict"].items()
                  if key.startswith("ema_model.")}
        online_state, ema_state = online.state_dict(), ema.ema_model.state_dict()
        for key, actual in online_state.items():
            if key in REPA_KEYS:
                expected = fresh_head[key]
            elif key == migration.EMBEDDING_KEY:
                expected = source[key][:160]
            elif key in migration.CTC_ROW_KEYS:
                expected = source[key][torch.tensor(list(range(159)) + [171, 172])]
            else:
                expected = source[key]
            self.assertTrue(torch.equal(actual, expected), f"Online mismatch: {key}")
            self.assertTrue(torch.equal(ema_state[key], expected), f"EMA mismatch: {key}")
        self.assertFalse(list(online.transformer.temporal_band.parameters()))
        self.assertFalse(online.transformer.temporal_band.state_dict())
        print(json.dumps({"result": "PASS", "loaded_parent_tensors": 802, "unchanged_fresh_projector_tensors": 6,
                          "online_ema_exact_tensors": 808, "optimizer_state_entries": 0, "initial_update": 0,
                          "ema_step": 50000, "parent_sha256": report["parent_sha256"]}, indent=2), flush=True)
        del online, ema, optimizer, checkpoint, source, online_state, ema_state
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parent", action="store_true")
    parser.add_argument("--audit-cache", action="store_true")
    arguments, remaining = parser.parse_known_args()
    REAL_PARENT, AUDIT_CACHE = arguments.real_parent, arguments.audit_cache
    torch.set_num_threads(4)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
