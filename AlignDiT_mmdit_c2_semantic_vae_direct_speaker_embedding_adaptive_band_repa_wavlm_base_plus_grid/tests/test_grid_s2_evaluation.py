import unittest
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import time
import wave
from unittest.mock import patch

import numpy as np

from aligndit.script.eval.eval_grid_s1 import whisper_model_slot
from aligndit.script.eval.grid_s1_common import (
    identity, normalize_text, ensure_contract, read, sha, publish, output_record, verify_outputs,
    validate_text_conditioning,
)
from aligndit.script.eval.grid_s2_common import PROTOCOL, select_references
from aligndit.script.eval.infer_grid_s2 import sampling_inputs
from aligndit.script.eval.run_grid_s1 import main as pipeline_main


def concurrent_slot_holder(directory):
    with whisper_model_slot('test', directory=Path(directory), poll_seconds=.01):
        start=time.monotonic_ns()
        time.sleep(.05)
        end=time.monotonic_ns()
    return start,end


def concurrent_contract_writer(path):
    ensure_contract(Path(path), {'protocol': 'test', 'seed': 0})
    return sha(path)


def row(speaker, utterance, text):
    return {'speaker': speaker, 'audio_relative_path': f'val/{speaker}/{utterance}.wav',
            'utterance_key': f'grid/{speaker}/{utterance}', 'text': text}


class GridS2EvaluationTests(unittest.TestCase):
    def test_target_only_changes_text_with_identical_prompt_video_and_time_grid(self):
        rng = np.random.default_rng(7)
        mean, std = rng.normal(size=64), rng.uniform(.5, 2, size=64)
        for prompt_frames, target_frames in ((119, 120), (120, 120), (2, 3)):
            reference = rng.normal(size=(prompt_frames, 64)).astype(np.float32)
            video = rng.normal(size=(target_frames, 1024)).astype(np.float32)
            reference_text, target_text = 'place red in m zero again', 'set blue by k four soon'
            baseline_text, baseline = sampling_inputs(reference, video, reference_text, target_text, mean, std)
            text, target_only = sampling_inputs(reference, video, reference_text, target_text, mean, std, 'target-only')
            self.assertEqual(baseline_text, reference_text + '  ' + target_text)
            self.assertEqual(text, target_text)
            np.testing.assert_array_equal(target_only[0], baseline[0])
            np.testing.assert_array_equal(target_only[1], baseline[1])
            self.assertEqual(target_only[2:], baseline[2:])
            self.assertEqual(target_only[2:4], (prompt_frames, target_frames))
            np.testing.assert_array_equal(target_only[1][prompt_frames:prompt_frames+target_frames], video)
            self.assertEqual(np.count_nonzero(target_only[1][:prompt_frames]), 0)

    def test_old_reference_target_contract_and_receipt_still_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, target, receipt = self.output_fixture(directory, target_only=False)
            before = (root/'protocol.json').read_bytes()
            validate_text_conditioning('s2', 'reference-target', root, require_contract=True)
            self.assertEqual(output_record(root, target, PROTOCOL), receipt)
            self.assertEqual((root/'protocol.json').read_bytes(), before)
            with self.assertRaisesRegex(RuntimeError, 'text conditioning'):
                validate_text_conditioning('s2', 'target-only', root)

    def output_fixture(self, directory, *, target_only):
        root = Path(directory)
        target = {**row('s1', 'target', 'set blue by k four soon'), 'original_num_samples': 1600, 'latent_frames': 4}
        contract = {'protocol': PROTOCOL}
        if target_only:
            contract.update(text_conditioning='target-only', model_text='supplied target GT dialogue only',
                            reference_transcript_used_for_generation=False)
        publish(root/'reference_pairs.json', {})
        contract['reference_pairs_sha256'] = sha(root/'reference_pairs.json')
        publish(root/'protocol.json', contract)
        path = root/target['audio_relative_path']
        path.parent.mkdir(parents=True)
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00'*target['original_num_samples'])
        receipt = {'id': identity(target), 'sha256': sha(path), 'protocol_sha256': sha(root/'protocol.json'),
                   'reference_id': 'val/s1/ref', 'prompt_frames': 4, 'target_frames': 4, 'sampler_frames': 8}
        if target_only:
            receipt.update(text_conditioning='target-only', model_text=target['text'])
        publish(root/'receipts'/(identity(target)+'.json'), receipt)
        return root, target, receipt

    def test_verify_rejects_wrong_missing_or_concatenated_target_only_model_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root, target, receipt = self.output_fixture(directory, target_only=True)
            target_id = identity(target)
            pair_map = {target_id: {'reference_id': 'val/s1/ref'}}
            references = {target_id: {'latent_frames': 4}}
            with patch('aligndit.script.eval.grid_s1_common.load_rows', return_value=([target], {})), \
                    patch('aligndit.script.eval.grid_s2_common.load_pairs', return_value=(pair_map, references)):
                _, summary = verify_outputs(root, root, protocol=PROTOCOL)
                self.assertEqual(summary['count'], 1)
                for invalid in ({k: v for k, v in receipt.items() if k != 'model_text'},
                                {**receipt, 'model_text': 'reference  ' + target['text']},
                                {**receipt, 'text_conditioning': 'reference-target'}):
                    publish(root/'receipts'/(target_id+'.json'), invalid)
                    with self.assertRaisesRegex(RuntimeError, 'Target-only model text'):
                        verify_outputs(root, root, protocol=PROTOCOL)

    def test_setting1_rejects_target_only_before_workers_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new_output'
            with patch('sys.argv', ['run_grid_s1', '--output-dir', str(output), '--text-conditioning', 'target-only']), \
                    patch('aligndit.script.eval.run_grid_s1.workers') as workers:
                with self.assertRaisesRegex(ValueError, 'requires --setting s2'):
                    pipeline_main()
                workers.assert_not_called()
                self.assertFalse(output.exists())

    def test_pipeline_resume_refuses_mode_collision_at_every_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.output_fixture(directory, target_only=True)
            for stage in ('infer', 'sim', 'wer', 'emosim', 'avsync', 'merge'):
                argv = ['run_grid_s1', '--setting', 's2', '--output-dir', str(root), '--start-stage', stage]
                with patch('sys.argv', argv), patch('aligndit.script.eval.run_grid_s1.workers') as workers:
                    with self.assertRaisesRegex(RuntimeError, 'text conditioning'):
                        pipeline_main()
                    workers.assert_not_called()
            validate_text_conditioning('s2', 'target-only', root, require_contract=True)

    def test_whisper_model_slots_limit_concurrent_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            with ProcessPoolExecutor(max_workers=4) as executor:
                intervals=list(executor.map(concurrent_slot_holder, [directory]*8))
            events=sorted([(start,1) for start,end in intervals]+[(end,-1) for start,end in intervals])
            active=peak=0
            for _,delta in events:
                active+=delta;peak=max(peak,active)
            self.assertEqual(peak,2)
            self.assertEqual(active,0)

    def test_parallel_workers_publish_one_immutable_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'protocol.json')
            with ProcessPoolExecutor(max_workers=4) as executor:
                hashes = list(executor.map(concurrent_contract_writer, [path]*32))
            self.assertEqual(len(set(hashes)), 1)
            self.assertEqual(read(path), {'protocol':'test', 'seed':0})
            with self.assertRaises(RuntimeError):
                ensure_contract(path, {'protocol':'another'})

    def test_different_text_same_speaker_permutation_is_reproducible(self):
        rows = [row(s, f'clip{i}', f'bin white by {i} two again') for s in ('s1','s2') for i in range(7)]
        first, second = select_references(rows), select_references(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(Counter(identity(r) for r in first.values()), Counter(identity(r) for r in rows))
        for target in rows:
            ref = first[identity(target)]
            self.assertNotEqual(identity(target), identity(ref))
            self.assertNotEqual(normalize_text(target['text']), normalize_text(ref['text']))
            self.assertEqual(target['speaker'], ref['speaker'])

    def test_duplicate_dialogue_is_never_the_reference(self):
        rows = [row('s1', f'clip{i}', text) for i,text in enumerate(['A two','a 2','B three','C four','D five','E six'])]
        for target in rows:
            self.assertNotEqual(normalize_text(target['text']), normalize_text(select_references(rows)[identity(target)]['text']))

    def test_impossible_independent_reference_is_rejected(self):
        for rows in ([row('s1','x','a')], [row('s1','x','a'),row('s1','y','A')]):
            with self.assertRaises(ValueError):
                select_references(rows)


if __name__ == '__main__':
    unittest.main()
