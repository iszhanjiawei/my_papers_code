"""Guard experiment semantics that cannot be inferred from parameter shapes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from aligndit.model.synchformer_features import PREPROCESSING
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    validate_fixed_band_contract,
    validate_synchformer_training_contract,
)


class ExperimentContractTests(unittest.TestCase):
    def setUp(self):
        config_dir = Path(__file__).resolve().parents[1] / 'src/aligndit/config'
        with initialize_config_dir(version_base='1.3', config_dir=str(config_dir)):
            self.config = compose(config_name='finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va_synchformer')
            self.parent = compose(config_name='finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va')
        arch = self.config.model.arch
        self.contract = {
            'temporal_band': {
                'mode': 'fixed', 'parameter_count': 0,
                'parameters': {k: v for k, v in arch.items() if k.startswith('temporal_band_')},
            },
            'synchformer_condition': {
                'dimension': 768, 'scope': 'audio_only_timestep_adaln_v1',
                'checkpoint_sha256': self.config.datasets.synchformer_checkpoint_sha256,
                'cache_contract': {
                    'preprocessing': PREPROCESSING,
                    'inventory_sha256': self.config.datasets.expected_inventory_sha256,
                },
            },
        }

    def check_contract(self, contract):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_10.pt'
            if contract is not None:
                (checkpoint.parent / 'speaker_training_contract.json').write_text(json.dumps(contract))
            validate_fixed_band_contract(self.config, checkpoint)
            validate_synchformer_training_contract(self.config, checkpoint)

    def test_config_preserves_baseline_except_sync_and_run_identity(self):
        self.assertEqual(dict(self.config.optim), dict(self.parent.optim))
        for key, value in self.parent.model.arch.items():
            self.assertEqual(self.config.model.arch[key], value, key)
        self.assertEqual(set(self.config.model.arch) - set(self.parent.model.arch), {'sync_dim'})
        self.assertNotEqual(self.config.model.name, self.parent.model.name)
        self.assertNotEqual(self.config.ckpts.save_dir, self.parent.ckpts.save_dir)
        self.assertEqual(self.config.model.arch.temporal_band_fixed_sigma_seconds, 0.1)
        self.assertTrue(self.config.model.arch.temporal_band_bidirectional)

    def test_training_rejects_disabled_sync_and_original_checkpoint_destination(self):
        # Stop before any cache/model/sidecar work: neither override may write
        # an incompatible model into the original experiment's checkpoint dir.
        from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import main
        from unittest.mock import patch
        from omegaconf import OmegaConf
        for mode in ("disabled_sync", "old_destination", "fractional_dim"):
            config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
            if mode == "disabled_sync":
                config.model.arch.sync_dim = 0
            elif mode == "fractional_dim":
                config.model.arch.sync_dim = 768.5
            else:
                config.ckpts.save_dir = self.parent.ckpts.save_dir
            with self.subTest(mode=mode), patch(
                "aligndit.script.train.finetune_semantic_vae_c2_direct_speaker.validate_speaker_cache_metadata",
                side_effect=AssertionError("must reject before reading cache or writing artifacts"),
            ), self.assertRaises(ValueError):
                main.__wrapped__(config)

    def test_matching_contract(self):
        self.check_contract(self.contract)

    def test_missing_or_old_audio_video_scope_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_contract(None)
        for mode in ('missing', 'audio_video'):
            contract = copy.deepcopy(self.contract)
            if mode == 'missing':
                del contract['synchformer_condition']
            else:
                contract['synchformer_condition']['scope'] = mode
            with self.assertRaises(RuntimeError):
                self.check_contract(contract)

    def test_encoder_preprocessing_and_inventory_drift_rejected(self):
        for field in ('checkpoint_sha256', 'preprocessing', 'inventory_sha256'):
            with self.subTest(field=field):
                contract = copy.deepcopy(self.contract)
                condition = contract['synchformer_condition']
                if field == 'checkpoint_sha256':
                    condition[field] = 'wrong'
                else:
                    condition['cache_contract'][field] = 'wrong'
                with self.assertRaises(RuntimeError):
                    self.check_contract(contract)

    def test_av_only_or_changed_width_rejected_with_same_tensor_shapes(self):
        for field, value in [('temporal_band_bidirectional', False), ('temporal_band_fixed_sigma_seconds', .2)]:
            contract = copy.deepcopy(self.contract)
            contract['temporal_band']['parameters'][field] = value
            with self.assertRaises(RuntimeError):
                self.check_contract(contract)


if __name__ == '__main__':
    unittest.main()
