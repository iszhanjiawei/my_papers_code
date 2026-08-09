from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aligndit.model.backbone.dit_vt_mm import (
    AudioInputEmbedding_MM,
    DiT_VT_MMDiT,
    MMDiTBlock_VT,
)
from aligndit.model.cfm_vt import CFM_VT, _ctc_min_input_lengths, _validate_strict_audio_video_alignment
from aligndit.model.dataset import (
    SEMANTIC_VAE_FEATURE,
    VIDEO_40HZ_FEATURE,
    SemanticVaeCelebVDubDataset,
    interpolate_video_to_latent_frames,
)
from aligndit.model.modules import DownsampleLayer, PrecomputedAudioRepresentation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _completion(
    feature: str,
    manifest_sha256: str,
    frames: int,
    index_path: Path,
    spec_path: Path,
    cache_root: Path,
) -> dict:
    index_sha256 = _sha256(index_path)
    index_records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    value = {
        "cache_schema_version": 1,
        "consolidated_index": {
            "count": 2,
            "path": index_path.relative_to(cache_root).as_posix(),
            "sha256": index_sha256,
            "size_bytes": index_path.stat().st_size,
        },
        "count": 2,
        "feature": feature,
        "manifest_sha256": manifest_sha256,
        "ordered_index_sha256": index_sha256,
        "selection": {"mode": "full"},
        "spec_sha256": _sha256(spec_path),
        "total_npy_size_bytes": sum(record["size_bytes"] for record in index_records),
    }
    if feature == SEMANTIC_VAE_FEATURE:
        value["total_latent_frames"] = frames
    else:
        value["total_source_frames"] = frames
        value["total_target_frames"] = frames
    return value


def _build_tiny_cache(root: Path, *, valid_contract: bool = True) -> tuple[dict, str]:
    cache_root = root / "cache"
    manifests = cache_root / "manifests"
    vocab_path = root / "vocab.txt"
    vocab_path.write_text(" \na\nb\n", encoding="utf-8")
    records = [
        {
            "ctc_adjacent_repeats": 1,
            "ctc_feasible_40hz": valid_contract,
            "ctc_min_input_frames": 3,
            "ctc_target_length": 2,
            "latent_frames": 4,
            "latent_relative_path": "latents/train/v0/a.npy",
            "split": "train",
            "text": "aa",
            "utterance_key": "celebvdub/train/v0/a",
            "video_40hz_relative_path": "video_40hz/train/v0/a.npy",
            "video_frames_25hz": 4,
        },
        {
            "ctc_adjacent_repeats": 0,
            "ctc_feasible_40hz": True,
            "ctc_min_input_frames": 2,
            "ctc_target_length": 2,
            "latent_frames": 2,
            "latent_relative_path": "latents/train/v0/b.npy",
            "split": "train",
            "text": "ab",
            "utterance_key": "celebvdub/train/v0/b",
            "video_40hz_relative_path": "video_40hz/train/v0/b.npy",
            "video_frames_25hz": 2,
        },
    ]
    selected_manifest = manifests / "train_ctc40_valid.jsonl"
    inventory_manifest = manifests / "inventory.jsonl"
    _write_jsonl(selected_manifest, records)
    _write_jsonl(inventory_manifest, records)
    selected_sha256 = _sha256(selected_manifest)
    inventory_sha256 = _sha256(inventory_manifest)
    inventory_meta_path = manifests / "inventory_meta.json"
    _write_json(
        inventory_meta_path,
        {
            "base_posterior_seed": 666,
            "ctc40_preflight": {
                "train_excluded": 105,
                "train_valid": 2,
                "vocab_sha256": _sha256(vocab_path),
            },
            "manifests": {
                "inventory.jsonl": {
                    "count": 2,
                    "path": "manifests/inventory.jsonl",
                    "sha256": inventory_sha256,
                    "size_bytes": inventory_manifest.stat().st_size,
                },
                "train_ctc40_valid.jsonl": {
                    "count": 2,
                    "path": "manifests/train_ctc40_valid.jsonl",
                    "sha256": selected_sha256,
                    "size_bytes": selected_manifest.stat().st_size,
                },
            },
            "latent_spec": {
                "dimension": 64,
                "dtype": "float32",
                "frame_rate_hz": 40.0,
                "hop_length_samples": 400,
                "mode": "fixed_posterior_sample",
                "sample_rate": 16000,
            },
            "total_latent_frames": 6,
        },
    )

    latent_a = np.full((4, 64), 3.0, dtype=np.float32)
    latent_b = np.full((2, 64), 5.0, dtype=np.float32)
    video_a = np.arange(4 * 1024, dtype=np.float32).reshape(4, 1024)
    video_b = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024)
    for relative_path, array in (
        (records[0]["latent_relative_path"], latent_a),
        (records[1]["latent_relative_path"], latent_b),
        (records[0]["video_40hz_relative_path"], video_a),
        (records[1]["video_40hz_relative_path"], video_b),
    ):
        path = cache_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)

    inventory_contract = {
        "count": 2,
        "inventory_metadata_path": str(inventory_meta_path),
        "inventory_metadata_sha256": _sha256(inventory_meta_path),
        "path": str(inventory_manifest),
        "sha256": inventory_sha256,
    }
    latent_spec = cache_root / "state/latents/spec.json"
    video_spec = cache_root / "state/video_40hz/spec.json"
    _write_json(
        latent_spec,
        {
            "cache_schema_version": 1,
            "checkpoint": {
                "config_sha256": "c12ba35b4035f97808dabaac4f254bd4e32b1dc5fba0840168ae0c41859d0235",
                "ema_sha256": "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516",
                "ema_step": 1000014,
                "metainfo_sha256": "24b8ff09360cdfe8a38e61862bf185c2130ef45a15e6f235bbbae8af8065c851",
            },
            "extraction": {
                "dtype": "float32",
                "golden_self_test": {
                    "raw_latent_sha256": "7e3bd4e044b7f0a4f1d0295ece831e639a1e7abd78f13fd3ce85e3e1b9feccce",
                    "utterance_key": "celebvdub/train/saP2eLOlPAc/0_0_0",
                },
                "latent_dim": 64,
                "latent_layout": "[time,channel]",
                "posterior_noise": "torch.randn with per-utterance CUDA Generator",
                "protocol": SEMANTIC_VAE_FEATURE,
                "sample_rate": 16000,
                "vae_hop_length": 400,
            },
            "manifest": inventory_contract,
            "semantic_vae_source": {
                "bigvgan_config_sha256": "a11e013f623eedc55b2410d48cbd810322df03658377806d16ab396369525618",
                "commit": "5bcca91fe8b65c0e52c5ee141968f98662dc4792",
                "working_tree_clean": True,
            },
        },
    )
    _write_json(
        video_spec,
        {
            "cache_schema_version": 1,
            "feature": VIDEO_40HZ_FEATURE,
            "interpolation": {
                "align_corners": False,
                "input_dtype": "float32",
                "input_frame_rate_hz": 25,
                "input_layout": "[time,channel]",
                "mode": "linear",
                "output_dtype": "float32",
                "output_frame_rate_hz": 40,
                "output_layout": "[time,channel]",
                "size": "record.latent_frames",
                "video_dim": 1024,
            },
            "manifest": {**inventory_contract, "total_target_frames": 6},
        },
    )

    latent_index = cache_root / "state/latents/index.jsonl"
    video_index = cache_root / "state/video_40hz/index.jsonl"
    latent_entries = []
    video_entries = []
    for record in records:
        latent_path = cache_root / record["latent_relative_path"]
        video_path = cache_root / record["video_40hz_relative_path"]
        latent_entries.append(
            {
                "feature": SEMANTIC_VAE_FEATURE,
                "latent_dim": 64,
                "latent_frames": record["latent_frames"],
                "relative_path": record["latent_relative_path"],
                "sha256": _sha256(latent_path),
                "size_bytes": latent_path.stat().st_size,
                "utterance_key": record["utterance_key"],
            }
        )
        video_entries.append(
            {
                "feature": VIDEO_40HZ_FEATURE,
                "relative_path": record["video_40hz_relative_path"],
                "sha256": _sha256(video_path),
                "size_bytes": video_path.stat().st_size,
                "source_frames": record["latent_frames"],
                "source_sha256": "2" * 64,
                "source_size_bytes": 1,
                "target_frames": record["latent_frames"],
                "utterance_key": record["utterance_key"],
                "video_dim": 1024,
            }
        )
    _write_jsonl(latent_index, latent_entries)
    _write_jsonl(video_index, video_entries)
    _write_json(
        cache_root / "state/latents/complete.json",
        _completion(SEMANTIC_VAE_FEATURE, inventory_sha256, 6, latent_index, latent_spec, cache_root),
    )
    _write_json(
        cache_root / "state/video_40hz/complete.json",
        _completion(VIDEO_40HZ_FEATURE, inventory_sha256, 6, video_index, video_spec, cache_root),
    )

    normalization_path = root / "train_normalization.json"
    _write_json(
        normalization_path,
        {
            "cache_schema_version": 1,
            "channel_count": 64,
            "count": 1,
            "feature": SEMANTIC_VAE_FEATURE,
            "frame_count": 1,
            "latent_complete_sha256": "0" * 64,
            "mean": [1.0] * 64,
            "method": "per_channel_population_mean_std_float64_welford_v1",
            "scope": "train",
            "std": [2.0] * 64,
            "train_manifest_sha256": "1" * 64,
        },
    )
    kwargs = {
        "cache_root": cache_root,
        "expected_normalization_sha256": _sha256(normalization_path),
        "expected_inventory_count": 2,
        "expected_record_count": 2,
        "manifest_path": selected_manifest,
        "normalization_path": normalization_path,
        "vocab_path": vocab_path,
    }
    return kwargs, selected_sha256


class SemanticVaeDatasetTests(unittest.TestCase):
    def test_normalizes_valid_frames_and_zero_pads_aligned_modalities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs, _ = _build_tiny_cache(Path(temp_dir))
            dataset = SemanticVaeCelebVDubDataset(**kwargs)
            first, second = dataset[0], dataset[1]
            self.assertEqual(first["mel_spec"].shape, (64, 4))
            torch.testing.assert_close(first["mel_spec"], torch.ones(64, 4))
            torch.testing.assert_close(second["mel_spec"], torch.full((64, 2), 2.0))

            batch = dataset.collate_fn([first, second])
            self.assertEqual(batch["mel"].shape, (2, 64, 4))
            self.assertEqual(batch["video"].shape, (2, 4, 1024))
            self.assertTrue(torch.equal(batch["mel_lengths"], batch["video_lengths"]))
            self.assertTrue(torch.equal(batch["audio_mask"], batch["video_mask"]))
            self.assertTrue(torch.count_nonzero(batch["mel"][1, :, 2:]) == 0)
            self.assertTrue(torch.count_nonzero(batch["video"][1, 2:]) == 0)

    def test_rejects_record_outside_immutable_ctc_valid_subset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs, _ = _build_tiny_cache(Path(temp_dir), valid_contract=False)
            with self.assertRaisesRegex(ValueError, "CTC feasibility"):
                SemanticVaeCelebVDubDataset(**kwargs)

    def test_rejects_authenticated_index_that_disagrees_with_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs, _ = _build_tiny_cache(Path(temp_dir))
            cache_root = kwargs["cache_root"]
            index_path = cache_root / "state/latents/index.jsonl"
            entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
            entries[0]["latent_frames"] = 3
            _write_jsonl(index_path, entries)
            complete_path = cache_root / "state/latents/complete.json"
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            completion["ordered_index_sha256"] = _sha256(index_path)
            completion["consolidated_index"]["sha256"] = _sha256(index_path)
            completion["consolidated_index"]["size_bytes"] = index_path.stat().st_size
            _write_json(complete_path, completion)
            with self.assertRaisesRegex(RuntimeError, "full manifest"):
                SemanticVaeCelebVDubDataset(**kwargs)

    def test_rejects_wrong_semantic_vae_provenance_even_when_completion_authenticates_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs, _ = _build_tiny_cache(Path(temp_dir))
            cache_root = kwargs["cache_root"]
            spec_path = cache_root / "state/latents/spec.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["checkpoint"]["ema_step"] = 999
            _write_json(spec_path, spec)
            complete_path = cache_root / "state/latents/complete.json"
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            completion["spec_sha256"] = _sha256(spec_path)
            _write_json(complete_path, completion)
            with self.assertRaisesRegex(RuntimeError, "wrong checkpoint"):
                SemanticVaeCelebVDubDataset(**kwargs)

    def test_interpolation_contract_is_linear_with_align_corners_false(self):
        source = torch.zeros(2, 1024, dtype=torch.float32)
        source[1] = 10.0
        aligned = interpolate_video_to_latent_frames(source, 4)
        expected = torch.tensor([0.0, 2.5, 7.5, 10.0])
        torch.testing.assert_close(aligned[:, 0], expected)


class SemanticVaeModelTests(unittest.TestCase):
    def test_text_context_is_parameter_free_normalized_and_padding_safe_for_every_ca(self):
        torch.manual_seed(2)
        model = DiT_VT_MMDiT(
            dim=16,
            depth=2,
            heads=2,
            dim_head=8,
            ff_mult=2,
            mel_dim=64,
            text_num_embeds=3,
            text_dim=16,
            text_mask_padding=True,
            attn_mask_enabled=True,
            use_conformer=False,
            layer_indices_ctc=(),
            ctc_sampling_ratios=(1, 1),
            n_mm_layers=1,
            n_text_layers=2,
            prompt_isolated_ca=False,
            audio_video_ratio=1,
            video_dim=1024,
            strict_audio_video_alignment=True,
            mask_input_embeddings=True,
            always_use_attention_mask=True,
        ).eval()
        state_keys_before = tuple(model.state_dict())
        self.assertFalse(any("text_context" in key for key in state_keys_before))
        raw_contexts = []
        ca_contexts = []

        def capture_raw_context(_module, _inputs, output):
            raw_contexts.append(output.detach().clone())

        def capture_ca_context(_module, inputs):
            ca_contexts.append(inputs[1].detach().clone())

        model.text_embed.register_forward_hook(capture_raw_context)
        for block in model.transformer_blocks:
            block.cross_attn.register_forward_pre_hook(capture_ca_context)

        audio_mask = torch.ones(2, 4, dtype=torch.bool)
        text = torch.tensor([[0, 1, 2, -1, -1], [0, -1, -1, -1, -1]])
        text_mask = text != -1
        with torch.no_grad():
            model(
                x=torch.randn(2, 4, 64),
                cond=torch.randn(2, 4, 64),
                text=text,
                video=torch.randn(2, 4, 1024),
                time=torch.rand(2),
                mask=audio_mask,
                text_mask=text_mask,
                video_mask=audio_mask,
                complementary_mask=torch.zeros_like(audio_mask),
                generation_mask=audio_mask,
            )

        self.assertEqual(len(raw_contexts), 1)
        self.assertEqual(len(ca_contexts), 2)
        raw_context = raw_contexts[0]
        cropped_mask = text_mask[:, : raw_context.shape[1]]
        expected = F.layer_norm(raw_context, (raw_context.shape[-1],), weight=None, bias=None, eps=1e-6)
        expected = expected.masked_fill(~cropped_mask.unsqueeze(-1), 0.0)
        for ca_context in ca_contexts:
            torch.testing.assert_close(ca_context, expected)

        valid_context = expected[cropped_mask]
        torch.testing.assert_close(valid_context.mean(dim=-1), torch.zeros(valid_context.shape[0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            valid_context.square().mean(dim=-1).sqrt(),
            torch.ones(valid_context.shape[0]),
            atol=2e-5,
            rtol=0,
        )
        self.assertTrue(torch.count_nonzero(expected[~cropped_mask]) == 0)
        torch.testing.assert_close(
            model.last_text_context_raw_rms,
            raw_context[cropped_mask].float().square().mean().sqrt(),
        )
        torch.testing.assert_close(model.last_text_context_post_rms, valid_context.float().square().mean().sqrt())
        self.assertEqual(tuple(model.state_dict()), state_keys_before)

        invalid_tail_mask = text_mask.clone()
        invalid_tail_mask[:, 3:] = False
        invalid_tail_mask[0, 4] = True
        with self.assertRaisesRegex(ValueError, "valid tokens beyond"):
            model._normalize_text_context(raw_context, invalid_tail_mask)

    def test_padding_safe_ctc_projector_is_batch_invariant(self):
        torch.manual_seed(0)
        projector = DownsampleLayer((1, 1), 4, 4, 3, padding_safe=True).eval()
        short = torch.randn(1, 5, 4)
        long = torch.randn(1, 9, 4)
        padded_short = F.pad(short, (0, 0, 0, 4), value=99.0)
        batch = torch.cat([padded_short, long])
        alone, alone_lens = projector(short, torch.tensor([5]))
        together, together_lens = projector(batch, torch.tensor([5, 9]))
        self.assertTrue(torch.equal(alone_lens, torch.tensor([5])))
        self.assertTrue(torch.equal(together_lens, torch.tensor([5, 9])))
        torch.testing.assert_close(together[0, :5], alone[0])
        self.assertTrue(torch.count_nonzero(together[0, 5:]) == 0)

    def test_masked_audio_conv_position_embedding_is_batch_invariant(self):
        torch.manual_seed(1)
        embedding = AudioInputEmbedding_MM(4, 16).eval()
        short_x, short_cond = torch.randn(1, 5, 4), torch.randn(1, 5, 4)
        alone = embedding(short_x, short_cond, mask=torch.ones(1, 5, dtype=torch.bool))
        padded_x = F.pad(short_x, (0, 0, 0, 4), value=77.0)
        padded_cond = F.pad(short_cond, (0, 0, 0, 4), value=-55.0)
        padded_mask = torch.tensor([[True] * 5 + [False] * 4])
        batched = embedding(padded_x, padded_cond, mask=padded_mask)
        torch.testing.assert_close(batched[:, :5], alone)
        self.assertTrue(torch.count_nonzero(batched[:, 5:]) == 0)

    def test_joint_attention_keeps_broadcast_key_mask(self):
        block = MMDiTBlock_VT(
            dim=8,
            heads=2,
            dim_head=4,
            dropout=0.0,
            attn_mask_enabled=True,
            text_dim=8,
            prompt_isolated_ca=False,
        ).eval()
        captured = {}

        def fake_attention(query, key, value, *, attn_mask, dropout_p, is_causal):
            captured["shape"] = tuple(attn_mask.shape)
            return torch.zeros_like(query)

        with mock.patch(
            "aligndit.model.backbone.dit_vt_mm.F.scaled_dot_product_attention",
            side_effect=fake_attention,
        ):
            block.joint_attn(
                torch.randn(2, 3, 8),
                torch.randn(2, 3, 8),
                mask=torch.tensor([[True, True, False], [True, True, True]]),
                v_mask=torch.tensor([[True, True, False], [True, True, True]]),
            )
        self.assertEqual(captured["shape"], (2, 1, 1, 6))

    def test_strict_model_uses_40hz_ctc_and_trainable_video_null(self):
        model = DiT_VT_MMDiT(
            dim=16,
            depth=2,
            heads=2,
            dim_head=8,
            ff_mult=2,
            mel_dim=64,
            text_num_embeds=3,
            text_dim=16,
            text_mask_padding=True,
            attn_mask_enabled=True,
            use_conformer=False,
            layer_indices_ctc=(0,),
            ctc_sampling_ratios=(1, 1),
            n_mm_layers=1,
            n_text_layers=1,
            prompt_isolated_ca=False,
            audio_video_ratio=1,
            video_dim=1024,
            strict_audio_video_alignment=True,
            mask_input_embeddings=True,
            always_use_attention_mask=True,
        )
        self.assertIsInstance(model.video_embed.vid_null_emb, nn.Parameter)
        self.assertEqual(model.projectors_ctc[0].sampling_ratios, (1, 1))
        self.assertTrue(model.projectors_ctc[0].padding_safe)

        audio_mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
        text = torch.tensor([[0, 1, 2], [0, 1, -1]])
        output, intermediates = model(
            x=torch.randn(2, 7, 64),
            cond=torch.randn(2, 7, 64),
            text=text,
            video=torch.randn(2, 7, 1024),
            time=torch.rand(2),
            mask=audio_mask,
            text_mask=text != -1,
            video_mask=audio_mask,
            complementary_mask=torch.zeros_like(audio_mask),
            generation_mask=audio_mask,
        )
        self.assertEqual(output.shape, (2, 7, 64))
        self.assertTrue(torch.count_nonzero(output[1, 5:]) == 0)
        self.assertTrue(torch.equal(intermediates[0]["z_lens"], torch.tensor([7, 5])))

        cfm = CFM_VT(
            transformer=model,
            mel_spec_module=PrecomputedAudioRepresentation(64, 16_000, 400),
            num_channels=64,
            vocab_char_map={" ": 0, "a": 1, "b": 2},
            ctc_lambda=0.1,
            audio_video_ratio=1,
            strict_audio_video_alignment=True,
        )
        loss, components, _, _ = cfm(
            torch.randn(2, 7, 64),
            text=["ab", "aa"],
            video=torch.randn(2, 7, 1024),
            lens=torch.tensor([7, 5]),
            text_lens=torch.tensor([2, 2]),
            video_lens=torch.tensor([7, 5]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(components["ctc_loss"], 0.0)
        loss.backward()

    def test_strict_cfm_helpers_reject_misalignment_and_count_repeats(self):
        text = torch.tensor([[1, 1, 2, -1], [2, 1, 1, 1]])
        lengths = torch.tensor([3, 4])
        self.assertTrue(torch.equal(_ctc_min_input_lengths(text, lengths), torch.tensor([4, 6])))
        audio = torch.zeros(2, 5, 64)
        video = torch.zeros(2, 5, 1024)
        _validate_strict_audio_video_alignment(audio, video, torch.tensor([5, 3]), torch.tensor([5, 3]), 1)
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            _validate_strict_audio_video_alignment(audio, video, torch.tensor([5, 3]), torch.tensor([5, 4]), 1)


if __name__ == "__main__":
    unittest.main()
