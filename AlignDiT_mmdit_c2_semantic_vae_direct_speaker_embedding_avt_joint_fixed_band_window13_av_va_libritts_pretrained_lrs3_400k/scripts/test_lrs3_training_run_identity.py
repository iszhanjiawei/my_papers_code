"""Run naming checks without loading data, a model, or starting training."""
import argparse
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from aligndit.model.fixed_temporal_band_contract import validate_fixed_band_training_config
from launch_lrs3_training import PROJECT, load_config, training_event_dir


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.name = 'LRS3_full_ddp_smoke_001_avt_joint_fixed_band_window13_av_va'
        self.args = argparse.Namespace(cache_root=None, checkpoint_dir=Path(self.temp.name)/self.name,
            run_until_update=2, epochs=None, variant='strict', run_name=self.name)

    def test_smoke_identity_isolated_and_training_semantics_preserved(self):
        config = load_config(self.args)
        self.assertTrue(validate_fixed_band_training_config(config))
        self.assertEqual(config.model.name, self.name)
        self.assertEqual(config.ckpts.save_dir, str(self.args.checkpoint_dir))
        self.assertEqual(config.optim.run_until_update, 2)
        self.assertEqual(config.datasets.batch_size_per_gpu, 3600)
        self.assertEqual(config.datasets.max_samples, 32)
        self.assertEqual((config.model.ctc_warmup_start, config.model.ctc_warmup_end), (10000, 30000))
        expected = PROJECT/'runs'/f'{self.name}_semantic_vae_40hz_LRS3_char'
        self.assertEqual(training_event_dir(config), expected)
        frozen = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
        entry_name = f'{frozen.model.name}_{frozen.model.audio_representation.name}_{frozen.datasets.name}_{frozen.model.tokenizer}'
        self.assertEqual(expected, PROJECT/'runs'/entry_name)

    def test_default_identity_is_400k_and_different_from_smoke(self):
        smoke = training_event_dir(load_config(self.args))
        del self.args.run_name  # Existing callers need no new argument.
        baseline = load_config(self.args)
        self.assertNotEqual(training_event_dir(baseline), smoke)
        self.assertEqual(training_event_dir(baseline), PROJECT/'runs'/
            'AlignDiT_LRS3_400k_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_semantic_vae_40hz_LRS3_char')

    def test_path_traversal_and_injection_and_missing_marker_rejected(self):
        for name in ['../'+self.name, self.name+'/child', self.name+'\\child', self.name+' bad',
                     self.name+'${oc.env:HOME}', self.name+',x=1', 'unmarked_smoke', '']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.args.run_name = name
                load_config(self.args)

    def test_fixed_band_checkpoint_path_guard_is_preserved(self):
        self.args.checkpoint_dir = Path(self.temp.name)/'unmarked_checkpoint'
        config = load_config(self.args)
        with self.assertRaisesRegex(ValueError, 'require dedicated'):
            validate_fixed_band_training_config(config)


if __name__ == '__main__':
    unittest.main()
