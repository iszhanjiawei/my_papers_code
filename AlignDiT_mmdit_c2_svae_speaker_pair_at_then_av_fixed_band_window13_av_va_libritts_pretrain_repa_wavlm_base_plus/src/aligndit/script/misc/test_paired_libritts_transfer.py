"""CPU checks for masked AdamW decay, stage identity and semantic vocabulary transfer."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.libritts_pretrained_migration import (
    CODEC_SHA256,
    CTC_ROW_KEYS,
    EMBEDDING_KEY,
    NORMALIZATION_SHA256,
    REPA_PROJECTOR_SHAPES,
    validate_transfer_keys,
    S2C_SHA256,
    SOURCE_VOCAB_SHA256,
    SPEAKER_SHA256,
    load_libritts_ema_into_model,
)
from aligndit.model.text_audio_checkpoint import PRETRAIN_STAGE, contract_sha256
from aligndit.model.trainer_text_audio import paired_adamw
from aligndit.script.misc.smoke_test_avt_joint import make_model


class TransferTests(unittest.TestCase):
    def test_shared_projection_decay_matches_adamw_on_at_heads_only(self):
        wrapper = nn.Module()
        wrapper.transformer = make_model(paired_joint_order="at_then_av", text_audio_only=True)
        optimizer = paired_adamw(wrapper, lr=0.01, betas=(0.9, 0.99), weight_decay=0.2)
        frozen = []
        active = []
        for block in wrapper.transformer.transformer_blocks[:wrapper.transformer.n_mm_layers]:
            rows = [p for m in (block.attn.to_q, block.attn.to_k, block.attn.to_v) for p in m.parameters()]
            for parameter, axis in [*((p, 0) for p in rows), (block.attn.to_out[0].weight, 1)]:
                parameter.grad = torch.zeros_like(parameter)
                inactive_slice, active_slice = parameter.chunk(2, dim=axis)
                frozen.append((inactive_slice, inactive_slice.detach().clone()))
                active.append((active_slice, active_slice.detach().clone() * 0.998))
        optimizer.step()
        for actual, expected in frozen + active:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_repa_transfer_allows_only_the_new_projector(self):
        with torch.device("meta"):
            source = {str(i): torch.empty(1) for i in range(802)}
            head = {key: torch.empty(shape) for key, shape in REPA_PROJECTOR_SHAPES.items()}
            target = {**source, **head}
            self.assertEqual(set(validate_transfer_keys(source, target, repa_enabled=True)), set(head))
            invalid_targets = [
                {key: val for key, val in target.items() if key != "0"},
                {**target, "unexpected": torch.empty(1)},
                {**target, "transformer.repa_projector.0.weight": torch.empty(1)},
            ]
            for invalid in invalid_targets:
                with self.assertRaises(ValueError):
                    validate_transfer_keys(source, invalid, repa_enabled=True)
            with self.assertRaises(ValueError):
                validate_transfer_keys(source, target, repa_enabled=False)

    def test_strict_full_layout_with_vocab_remapping_and_legacy_rejection(self):
        cfg_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
            source = compose(config_name="pretrain_libritts_text_audio")
            dest = compose(config_name="finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup")
        spec_dir = Path(dest.datasets.cache_root) / "state/latents/spec.json"
        if not Path(dest.datasets.vocab_path).is_file() or not spec_dir.is_file():
            self.skipTest("CelebVDub dataset contracts are unavailable on this server")
        contract = {
            "stage": PRETRAIN_STAGE, "contract_schema_version": 1,
            "model": OmegaConf.to_container(source.model, resolve=True),
            "initialization": {"type": "s2c_ema", "parent_sha256": S2C_SHA256, "parent_update": 70000},
            "data": {"normalization_sha256": NORMALIZATION_SHA256, "vocab_sha256": SOURCE_VOCAB_SHA256,
                     "codec_sha256": CODEC_SHA256, "semantic_vae_checkpoint_sha256": CODEC_SHA256,
                     "speaker_checkpoint_sha256": SPEAKER_SHA256,
                     "posterior_protocol": json.loads(spec_dir.read_text())["extraction"]["protocol"]},
        }
        # Construct the true 802-tensor layouts on meta. Expanded views provide
        # compact real values on disk, including meaningful vocabulary row IDs.
        with torch.device("meta"):
            source_net = nn.Module()
            source_net.transformer = DiT_VT_MMDiT(**source.model.arch, text_num_embeds=171, mel_dim=64)
            target_net = nn.Module()
            target_net.transformer = DiT_VT_MMDiT(**dest.model.arch, text_num_embeds=159, mel_dim=64)
        payload = {"initted": torch.tensor(True), "step": torch.tensor(50000)}
        for i, (key, tensor) in enumerate(source_net.state_dict().items()):
            if key in CTC_ROW_KEYS or key == EMBEDDING_KEY:
                rows = torch.arange(tensor.shape[0], dtype=tensor.dtype)
                value = rows.reshape(rows.numel(), *([1] * (tensor.ndim - 1))).expand(tensor.shape)
            else:
                value = torch.tensor(float(i)).expand(tensor.shape)
            payload["ema_model." + key] = value
        ckpt = {"stage": PRETRAIN_STAGE, "checkpoint_schema_version": 1, "update": 50000,
                "training_contract": contract, "training_contract_sha256": contract_sha256(contract),
                "ema_model_state_dict": payload}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({**ckpt, "stage": "libritts_text_audio"}, path)
            with self.assertRaisesRegex(ValueError, "stage"):
                load_libritts_ema_into_model(target_net, path, dest)
            torch.save(ckpt, path)
            # Assign real CPU tensors to meta slots. Delegate strict checks to
            # Module.load_state_dict while avoiding a second full model allocation.
            load_state = target_net.load_state_dict
            target_net.load_state_dict = lambda state, strict: load_state(state, strict=strict, assign=True)
            report = load_libritts_ema_into_model(target_net, path, dest)
        self.assertEqual(report["loaded_key_count"], 802)
        self.assertEqual(set(report["row_remappings"]), CTC_ROW_KEYS | {EMBEDDING_KEY})
        weights = target_net.state_dict()
        for name, mapping in report["row_remappings"].items():
            rows = weights[name].reshape(len(weights[name]), -1)[:, 0]
            torch.testing.assert_close(rows, torch.tensor(mapping["source_rows"], dtype=rows.dtype))
        self.assertEqual(int(weights["transformer.projectors_ctc.0.model.6.bias"][-1]), 172)


if __name__ == "__main__":
    unittest.main()
