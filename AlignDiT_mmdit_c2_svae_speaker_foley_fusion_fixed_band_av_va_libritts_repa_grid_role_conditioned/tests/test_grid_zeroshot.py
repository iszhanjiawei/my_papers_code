import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave

from aligndit.script.eval.grid_s1_common import publish, sha
from aligndit.script.eval.grid_zeroshot_common import (
    PORTABLE_SHA256, PROTOCOL, check_output, frozen_document, normalize_wer, summarize, wer_variants,
)


class GridZeroshotTests(unittest.TestCase):
    def test_three_wer_protocols_preserve_then_remove_silence(self):
        variants = wer_variants('sp A2 sil', 'a two')
        self.assertEqual(variants['original']['truth'], 'sp a2 sil')
        self.assertEqual(variants['digits_only']['truth'], 'sp a two sil')
        self.assertEqual(variants['digits_silence']['truth'], 'a two')
        self.assertEqual(variants['digits_only']['word_edit_distance'], 2)
        self.assertEqual(variants['digits_silence']['word_edit_distance'], 0)
        self.assertEqual(variants['digits_silence']['reference_word_count'], 2)

    def test_multi_digit_letter_and_punctuation_match_supplied_protocol(self):
        self.assertEqual(normalize_wer('SP 12 a2, isn\'t silence; sil', 'digits_silence'), 'one two a two isnt silence')
        self.assertEqual(normalize_wer('sp 12 a2, sil', 'original'), 'sp 12 a2 sil')
        self.assertEqual(normalize_wer('SP 12 a2, sil', 'digits_only'), 'sp one two a two sil')

    def test_unknown_protocol_and_empty_truth_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_wer('a', 'unknown')
        with self.assertRaises(ValueError):
            wer_variants('sp sil', 'hello')

    def test_corpus_wer_uses_word_counts_not_mean_of_sentences(self):
        rows = [dict(id='one', wer_variants=wer_variants('a 2', 'a two')),
                dict(id='two', wer_variants=wer_variants('sp a 2', 'a 3'))]
        result = summarize(rows, 'wer')['wer_variants']
        self.assertEqual(result['original']['word_edit_distance'], 3)
        self.assertEqual(result['original']['reference_word_count'], 5)
        self.assertEqual(result['original']['wer'], .6)
        self.assertEqual(result['digits_only']['wer'], .4)
        self.assertEqual(result['digits_silence']['wer'], .25)

    def test_full_frozen_document_matches_independent_supplied_rebuilder(self):
        directory = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/projects/alignDiT_idea6/zero-shot-Grid')
        document = directory/'GRID目标_CelebVDub参考_zero-shot复现说明.md'
        if not document.exists():
            self.skipTest('User-supplied frozen document is not installed')
        portable, blob, refs, assignments, _ = frozen_document(document)
        spec = importlib.util.spec_from_file_location('supplied_rebuilder', directory/'rebuild_from_document.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        other, other_blob, protocol, _ = module.load_document(document)
        self.assertEqual((portable, blob), (other, other_blob))
        self.assertEqual(len(portable), 3281)
        self.assertEqual(len(refs), 33)
        self.assertEqual(assignments, protocol['assignments'])
        import hashlib
        self.assertEqual(hashlib.sha256(blob).hexdigest(), PORTABLE_SHA256)
        # Compare every raw reference string against the supplied normalization,
        # so subtle legacy punctuation or digit handling cannot drift silently.
        spec = importlib.util.spec_from_file_location('supplied_evaluator', directory/'evaluate_zeroshot.py')
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        for row in portable:
            for variant in ('original', 'digits_only', 'digits_silence'):
                self.assertEqual(normalize_wer(row['text'], variant), evaluator.normalize_text(row['text'], variant))

    def test_receipt_rejects_reference_text_swap_and_time_offset_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(audio_relative_path='val/s1/example.wav', utterance_key='grid/s1/example',
                       text='set blue by k four soon', latent_frames=4, original_num_samples=1600,
                       reference_id='celebvdub/test/external/clip',
                       reference=dict(ref_frames=3, ref_latent_sha256='a', ref_speaker_sha256='b'))
            publish(root/'protocol.json', dict(protocol=PROTOCOL, text_conditioning='target-only',
                                              reference_transcript_used_for_generation=False))
            path = root/row['audio_relative_path']
            path.parent.mkdir(parents=True)
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\x00\x00'*1600)
            receipt = dict(id='val/s1/example', model_text=row['text'], text_conditioning='target-only',
                           reference_id=row['reference_id'], reference_latent_sha256='a', reference_speaker_sha256='b',
                           prompt_frames=3, target_frames=4, sampler_frames=7, sha256=sha(path),
                           protocol_sha256=sha(root/'protocol.json'))
            target = root/'receipts/val/s1/example.json'
            publish(target, receipt)
            self.assertEqual(check_output(root, row), receipt)
            for change in [dict(model_text='reference  '+row['text']), dict(reference_id='grid/s1/other'),
                           dict(prompt_frames=4), dict(sampler_frames=8), dict(reference_speaker_sha256='target-speaker')]:
                publish(target, dict(receipt, **change))
                with self.assertRaises(RuntimeError):
                    check_output(root, row)


if __name__ == '__main__':
    unittest.main()
