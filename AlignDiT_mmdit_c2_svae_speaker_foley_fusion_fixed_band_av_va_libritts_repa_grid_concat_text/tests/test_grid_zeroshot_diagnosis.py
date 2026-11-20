"""Condition interventions must preserve sample identity and sampling geometry."""
import tempfile
import unittest
from pathlib import Path

from aligndit.script.eval.diagnose_grid_zeroshot import (
    VARIANTS, condition_record, frozen_bytes, select_rows,
)


def row(speaker, clip='a'):
    return dict(audio_relative_path=f'val/{speaker}/{clip}.wav', utterance_key=f'grid/{speaker}/{clip}',
        reference_id=f'ref/{speaker}', text='bin white by a two again', target_video_sha256='video',
        latent_frames=120, original_num_samples=48000,
        reference=dict(ref_text='a short reference', ref_frames=86, ref_source_audio_sha256='audio',
                       ref_latent_sha256='latent', ref_speaker_sha256=f'cam/{speaker}'))


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(f's{i}') for i in range(1, 34)]
        selected, swaps = select_rows(self.rows, 1)
        self.assertEqual(selected, self.rows)
        self.selection = dict(cam_swapped_reference_ids=swaps)
        self.refs = {r['reference_id']: r['reference'] for r in self.rows}

    def test_each_intervention_keeps_original_prompt_and_target_geometry(self):
        for item in self.rows:
            for variant in VARIANTS:
                c = condition_record(item, variant, self.selection, self.refs)
                self.assertEqual((c['prompt_frames'], c['target_frames'], c['sampler_frames']), (86, 120, 206))
                self.assertEqual(c['reference_latent_sha256'], 'latent')
                self.assertEqual(c['reference_id'], item['reference_id'])
                if variant == 'cam_swapped':
                    self.assertNotEqual(c['speaker_condition_reference_id'], item['reference_id'])
                if variant in ('video_disabled', 'ref_text_video_disabled'):
                    self.assertEqual(c['ignore_modality'], 'video')
                if variant == 'prompt_zero':
                    self.assertFalse(c['no_ref_audio'])
                    self.assertEqual(c['speaker_condition_reference_id'], item['reference_id'])
                if variant == 'reference_disabled':
                    self.assertTrue(c['no_ref_audio'])
                    self.assertIsNone(c['speaker_condition_reference_id'])

    def test_first_n_selection_does_not_depend_on_grouping(self):
        interleaved = self.rows + [row(f's{i}', 'b') for i in range(1, 34)]
        selected, swaps = select_rows(interleaved, 2)
        self.assertEqual(selected, interleaved)
        self.assertEqual(len(set(swaps.values())), 33)
        with self.assertRaises(ValueError):
            select_rows(interleaved, 3)

    def test_existing_artifacts_must_be_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'artifact'
            frozen_bytes(path, b'original')
            frozen_bytes(path, b'original')
            with self.assertRaises(RuntimeError):
                frozen_bytes(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
