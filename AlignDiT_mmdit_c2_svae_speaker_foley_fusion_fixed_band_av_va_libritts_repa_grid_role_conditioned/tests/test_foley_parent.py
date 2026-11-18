"""Parent schema tests and optional byte-for-byte check with the real 50k EMA."""

import hashlib
import os
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.libritts_pretrained_migration import (
    CTC_ROW_KEYS,
    EMBEDDING_KEY,
    _expected_unshared_keys,
    load_libritts_ema_into_model,
    validate_transfer_keys,
)
from aligndit.model.modules import PrecomputedAudioRepresentation
from f5_tts.model.utils import get_tokenizer


class LibriTTSRoleFusionTests(unittest.TestCase):
    def test_architecture_difference_is_closed(self):
        ignored, new = _expected_unshared_keys()
        self.assertEqual((len(ignored), len(new)), (194, 223))
        shared = {f"shared.{i}" for i in range(608)}
        source, target = shared | ignored, shared | new
        validate_transfer_keys(source, target)
        with self.assertRaisesRegex(ValueError, "parent-only"):
            validate_transfer_keys(source, target - {"shared.0"})
        with self.assertRaisesRegex(ValueError, "target-only"):
            validate_transfer_keys(source, target | {"transformer.unexpected"})

    @unittest.skipUnless(os.getenv("ALIGNDIT_VERIFY_LIBRITTS_PARENT") == "1", "explicit real-parent test")
    def test_real_ema_tensor_migration(self):
        config_dir = Path(__file__).resolve().parents[1] / "src/aligndit/config"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name="finetune_grid")
        vocab, size = get_tokenizer(cfg.datasets.vocab_path, "custom")
        audio = cfg.model.audio_representation
        model = CFM_VT(
            transformer=DiT_VT_MMDiT(**cfg.model.arch, text_num_embeds=size, mel_dim=audio.channels),
            mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
            num_channels=audio.channels, vocab_char_map=vocab, audio_video_ratio=cfg.model.arch.audio_video_ratio,
            ctc_lambda=cfg.model.ctc_lambda, repa_lambda=cfg.model.repa_lambda,
        )
        state = model.state_dict()
        _, fresh = _expected_unshared_keys()
        def checksum(key):
            return hashlib.sha256(state[key].numpy()).digest()
        before = {key: checksum(key) for key in fresh}
        report = load_libritts_ema_into_model(model, cfg.ckpts.pretrained_path, cfg)
        self.assertEqual(report["loaded_key_count"], 608)
        self.assertEqual(report["shape_identical_key_count"], 603)
        self.assertEqual(report["parent_ema_step"], 50000)
        self.assertEqual(report["target_key_count"], 831)
        self.assertEqual(set(report["row_remappings"]), {EMBEDDING_KEY, *CTC_ROW_KEYS})
        self.assertEqual(before, {key: checksum(key) for key in fresh})
        self.assertEqual(report["vocabulary"]["ctc_blank"], {"source": 172, "target": 160})
        source = torch.load(cfg.ckpts.pretrained_path, mmap=True, weights_only=True, map_location="cpu")
        for key in report["loaded_keys"]:
            expected = source["ema_model_state_dict"]["ema_model." + key]
            if key in report["row_remappings"]:
                expected = expected[report["row_remappings"][key]["source_rows"]]
            self.assertTrue(torch.equal(state[key], expected), key)
        self.assertGreater(int(torch.count_nonzero(state["transformer.speaker_proj.weight"])), 0)
