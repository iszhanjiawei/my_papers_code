"""Exercise the text-only path and its hand-off to the audio/video task."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.libritts_pretrained_migration import load_libritts_ema_into_model
from aligndit.model.semantic_vae_direct_migration import sha256_file
from aligndit.model.text_audio_checkpoint import contract_sha256


class FoleyTextPretrainingTests(unittest.TestCase):
    def setUp(self):
        self.arch = dict(dim=32, depth=3, heads=2, dim_head=16, ff_mult=2, text_dim=16,
                         n_mm_layers=2, n_text_layers=2, layer_indices_ctc=[1], ctc_sampling_ratios=[1],
                         use_conformer=False, video_dim=12, audio_video_ratio=1, video_rope_scaled=False,
                         speaker_dim=4, speaker_condition_start_layer=2, prompt_isolated_ca=False,
                         conv_layers=1, pe_attn_head=1, qk_norm="rms_norm", joint_av_text_ca=True,
                         dropout=0.0, text_mask_padding=True, attn_mask_enabled=True)
        self.src_arch = dict(self.arch, text_audio_only=True, temporal_band_enabled=False)
        self.dst_arch = dict(self.arch, text_mask_padding=False, temporal_band_enabled=True,
                             temporal_band_mode="fixed", temporal_band_bidirectional=True,
                             repa_layer=1, repa_target_dim=12, repa_projector_dim=32)
        self.parent = DiT_VT_MMDiT(**self.src_arch, mel_dim=8, text_num_embeds=4)

    def test_audio_queries_use_text_and_ignore_video(self):
        model = self.parent
        # The new gates begin at zero; exercise conditions after the gates have opened.
        with torch.no_grad():
            for p in model.parameters():
                if not p.count_nonzero():
                    p.normal_(std=0.05)
        valid = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
        kw = dict(x=torch.randn(2, 8, 8), cond=torch.randn(2, 8, 8), time=torch.rand(2),
                  video=None, speaker_embedding=torch.randn(2, 4),
                  mask=valid, generation_mask=valid, text=torch.tensor([[0, 1], [2, -1]]))
        x, _ = model(**kw)
        altered, _ = model(**{**kw, "text": torch.tensor([[1, 0], [3, -1]])})
        self.assertGreater((x - altered).norm().item(), 0.0)
        self.assertEqual(x[~valid].count_nonzero().item(), 0)
        x[valid].square().mean().backward()
        for block in model.transformer_blocks[:2]:
            self.assertFalse(block.cross_attn.v_cond_cross_q.weight.requires_grad)
            for proj in (block.cross_attn.text_cross_kv, block.cross_attn.audio_cross_q):
                self.assertGreater(proj.weight.grad.norm().item(), 0)
        self.assertTrue(all(not p.requires_grad for p in model.video_embed.parameters()))

    def test_ema_transfer_retains_foley_text_weights_and_remaps_ctc_blank(self):
        parent, target = nn.Module(), nn.Module()
        parent.transformer = self.parent
        target.transformer = DiT_VT_MMDiT(**self.dst_arch, mel_dim=8, text_num_embeds=2)
        repa = target.transformer.repa_projector[0].weight.detach().clone()
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "src_vocab").write_text("a\nb\nc\nd\n")
            (folder / "dst_vocab").write_text("c\na\n")
            features = {"latent": {"feature": "posterior", "checkpoint_sha256": "codec"},
                        "speaker": {"checkpoint_sha256": "speaker"}}
            protocol = {"feature_contract": features, "latent_normalized": False, "normalization_sha256": "norm"}
            meta = {"dataset": "GRID", "normalization_sha256": "norm", "feature_contract": features,
                    "extraction_protocol": protocol, "extraction_sha256": contract_sha256(protocol)}
            (folder / "manifests").mkdir()
            inventory_meta = folder / "manifests/inventory_meta.json"
            inventory_meta.write_text(json.dumps(meta))
            source_model = {"name": "test", "arch": self.src_arch, "ctc_lambda": 0.0, "audio_representation": {}}
            contract = {"stage": "libritts_text_audio", "model": source_model,
                        "initialization": {"type": "s2c_ema", "parent_sha256": "parent"},
                        "data": {"normalization_sha256": "norm", "speaker_checkpoint_sha256": "speaker",
                                 "posterior_protocol": "posterior", "codec_sha256": "codec",
                                 "vocab_sha256": sha256_file(folder / "src_vocab")}}
            weights = {"ema_model." + k: v for k, v in parent.state_dict().items()}
            checkpoint = {"stage": "libritts_text_audio", "training_contract": contract, "update": 50,
                          "training_contract_sha256": contract_sha256(contract), "checkpoint_schema_version": 1,
                          "ema_model_state_dict": {**weights, "step": torch.tensor(50), "initted": True}}
            torch.save(checkpoint, folder / "model.pt")
            config = {"model": {"arch": self.dst_arch, "audio_representation": {}},
                      "ckpts": {"expected_parent_update": 50, "libritts_run_name": "test", "s2c_sha256": "parent",
                                "libritts_vocab_path": folder / "src_vocab"},
                      "datasets": {"name": "GRID", "cache_root": folder, "expected_normalization_sha256": "norm",
                                   "speaker_embedding_checkpoint_sha256": "speaker", "vocab_path": folder / "dst_vocab",
                                   "expected_vocab_sha256": sha256_file(folder / "dst_vocab")}}
            report = load_libritts_ema_into_model(target, folder / "model.pt", config)
            self.assertEqual(len(report["new_target_keys"]), 6)
            self.assertEqual(report["loaded_key_count"], len(parent.state_dict()))
            torch.testing.assert_close(target.transformer.repa_projector[0].weight, repa)
            torch.testing.assert_close(
                target.transformer.text_embed.text_embed.weight, self.parent.text_embed.text_embed.weight[[0, 3, 1]])
            for key, value in target.state_dict().items():
                if ".cross_attn." in key:
                    torch.testing.assert_close(value, parent.state_dict()[key])
            for suffix in ("weight", "bias"):
                key = f"transformer.projectors_ctc.0.model.3.{suffix}"
                torch.testing.assert_close(target.state_dict()[key], parent.state_dict()[key][[2, 0, 4, 5]])
            features["latent"]["checkpoint_sha256"] = "different-codec"
            meta["extraction_sha256"] = contract_sha256(protocol)
            inventory_meta.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "codec_sha256"):
                load_libritts_ema_into_model(target, folder / "model.pt", config)

            config["datasets"]["cache_format"] = "grid_mmdit_svae"
            local = {"dataset": "GRID", "complete": True, "normalization": {"sha256": "norm"},
                     "latent_spec": {"mode": "fixed_posterior_sample"},
                     "speaker_spec": {"checkpoint_sha256": "speaker"},
                     "semantic_vae_contract": {"extraction": {"protocol": "posterior"},
                                               "checkpoint": {"ema_sha256": "codec"}}}
            local_path = folder / "data_contract.json"
            local_path.write_text(json.dumps(local))
            (folder / "complete.json").write_text(json.dumps({"contract_sha256": sha256_file(local_path)}))
            local_report = load_libritts_ema_into_model(target, folder / "model.pt", config)
            self.assertEqual(local_report["row_remappings"], report["row_remappings"])
