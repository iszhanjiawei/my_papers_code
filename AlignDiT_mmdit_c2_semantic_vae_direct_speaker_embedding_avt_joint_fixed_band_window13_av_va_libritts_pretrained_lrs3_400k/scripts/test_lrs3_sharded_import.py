"""Physical-copy, source-provenance and exact interpolation regression tests."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import lrs3_sharded_import as importer
from aligndit.lrs3_imported_source import validate_import_completion


def npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.origin, self.output = self.base / 'origin', self.base / 'new'
        self.origin.mkdir(); self.output.mkdir()
        self.contract_sha = 'a' * 64
        self.stem = 'train/pretrain/example_00001'
        self.row = {'sample_id': 'example_00001', 'relative_stem': self.stem,
                    'original_clip_id': 'pretrain/example/00001', 'split': 'train',
                    'utterance_key': 'lrs3/' + self.stem, 'posterior_seed': 123,
                    'width': 96, 'height': 96, 'video_frames': 4,
                    'original_num_samples': 2800, 'latent_frames': 7,
                    'audio_path': str(self.base / 'original.wav'),
                    'video_path': str(self.base / 'original.mp4')}
        for name, directory, suffix in [('audio', 'audio', '.wav'), ('latent', 'latents', '.npy'),
                ('speaker', 'speaker_embeddings', '.npy'), ('mouth', 'video_mouth', '.mp4'),
                ('video_25hz', 'avhubert_video_25hz', '.npy'), ('video_40hz', 'video_40hz', '.npy')]:
            self.row[name + '_relative_path'] = directory + '/' + self.stem + suffix
        sf.write(self.row['audio_path'], np.linspace(-.1, .1, 2800, dtype=np.float32), 16000, subtype='PCM_16')
        writer = cv2.VideoWriter(self.row['video_path'], cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 96))
        self.assertTrue(writer.isOpened())
        for i in range(4):
            writer.write(np.full((96, 96, 3), i * 30, dtype=np.uint8))
        writer.release()
        mouth = self.origin / self.row['mouth_relative_path']
        mouth.parent.mkdir(parents=True); mouth.symlink_to(self.row['video_path'])
        speaker = np.zeros(192, dtype=np.float32); speaker[0] = 1
        npy(self.origin / self.row['speaker_relative_path'], speaker)
        npy(self.origin / self.row['latent_relative_path'], np.ones((7, 64), dtype=np.float32))
        self.video25 = np.arange(4 * 1024, dtype=np.float32).reshape(4, 1024) / 123
        npy(self.origin / self.row['video_25hz_relative_path'], self.video25)
        self.expected40 = F.interpolate(torch.from_numpy(self.video25).T.unsqueeze(0),
                size=7, mode='linear', align_corners=False)[0].T.contiguous().numpy()
        npy(self.origin / self.row['video_40hz_relative_path'], self.expected40)
        self.audio_receipt_rel = f'state/audio_receipts/{self.stem}.json'
        importer.write_json(self.origin / self.audio_receipt_rel,
            {'identity': importer.audio_identity(self.row, importer.digest(self.row['audio_path'])),
             'outputs': {name: importer.digest(self.origin / self.row[name + '_relative_path'])
                         for name in ('latent', 'speaker')}})
        feature_hash = importer.digest(self.origin / self.row['video_25hz_relative_path'])
        importer.write_json(self.origin / f'reports/media_receipts/{self.stem}.json',
            {'relative_stem': self.stem, 'source_path': self.row['video_path'],
             'source_sha256': importer.digest(self.row['video_path']), 'protocol_sha256': importer.PROTOCOL,
             'features_complete': True, 'video_frames': 4, 'feature_sha256': feature_hash})
        self.receipt40_rel = f'state/video40_receipts/{self.stem}.json'
        importer.write_json(self.origin / self.receipt40_rel,
            {'identity': importer.video40_identity(self.row, feature_hash),
             'outputs': {'video_40hz': importer.digest(self.origin / self.row['video_40hz_relative_path'])}})
        self.proof = {key: self.row[key] for key in ('sample_id', 'relative_stem', 'original_clip_id', 'video_frames')}
        self.proof.update(status='ok', feature_sha256=feature_hash)

    def tearDown(self):
        self.temp.cleanup()

    def run_row(self, exclusion=None):
        return importer.process_row(self.row, self.proof, self.origin, self.output,
                                    self.contract_sha, exclusion)

    def test_copies_verified_assets_without_original_inode_aliases(self):
        original = {p: importer.digest(p) for p in self.origin.rglob('*') if p.is_file()}
        result = self.run_row()
        self.assertEqual(result['video40_action'], 'copied')
        self.assertEqual(set(result['files']), importer.expected_files(self.row, False))
        importer.validate_local_receipt(self.row, result, self.output, self.contract_sha, None)
        for relative, info in result['files'].items():
            target = self.output / relative
            self.assertFalse(target.is_symlink())
            self.assertEqual(importer.digest(target), info['sha256'])
            if info['source_path']:
                self.assertNotEqual((target.stat().st_dev, target.stat().st_ino),
                    (Path(info['source_path']).stat().st_dev, Path(info['source_path']).stat().st_ino))
        self.assertEqual(original, {p: importer.digest(p) for p in original})

    def test_missing_video40_recomputes_exact_existing_math(self):
        (self.origin / self.row['video_40hz_relative_path']).unlink()
        result = self.run_row()
        self.assertEqual(result['video40_action'], 'recomputed')
        actual = np.load(self.output / self.row['video_40hz_relative_path'], allow_pickle=False)
        np.testing.assert_array_equal(actual, self.expected40)
        entry = result['files'][self.row['video_40hz_relative_path']]
        self.assertEqual(entry['source_path'], str(self.origin / self.row['video_25hz_relative_path']))

    def test_hash_matching_nonfinite_old_video40_still_recomputed(self):
        path = self.origin / self.row['video_40hz_relative_path']
        bad = self.expected40.copy(); bad[0, 0] = np.nan; npy(path, bad)
        receipt = importer.read_json(self.origin / self.receipt40_rel)
        receipt['outputs']['video_40hz'] = importer.digest(path)
        importer.write_json(self.origin / self.receipt40_rel, receipt)
        result = self.run_row()
        self.assertEqual(result['video40_action'], 'recomputed')
        np.testing.assert_array_equal(np.load(self.output / self.row['video_40hz_relative_path']), self.expected40)

    def test_quality_exclusion_still_copies_audio_latent_speaker(self):
        self.proof['status'] = 'failed'
        exclusion = {'relative_stem': self.stem, 'reason': 'no_valid_landmarks'}
        result = self.run_row(exclusion)
        self.assertEqual(result['status'], 'quality_excluded')
        self.assertEqual(set(result['files']), importer.expected_files(self.row, True))
        self.assertFalse((self.output / self.row['video_40hz_relative_path']).exists())
        importer.validate_local_receipt(self.row, result, self.output, self.contract_sha, exclusion)

    def test_pinned_media_proof_cannot_be_replaced_by_new_receipt(self):
        self.proof['feature_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'independent proof'):
            self.run_row()
        self.assertFalse((self.output / f'reports/import_receipts/{self.stem}.json').exists())

    def test_bad_audio_protocol_never_becomes_complete(self):
        receipt = importer.read_json(self.origin / self.audio_receipt_rel)
        receipt['identity']['posterior_seed'] += 1
        importer.write_json(self.origin / self.audio_receipt_rel, receipt)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.run_row()
        self.assertFalse((self.output / f'reports/import_receipts/{self.stem}.json').exists())

    def test_local_resumption_rejects_tampering_and_missing_assets(self):
        result = self.run_row()
        path = self.output / self.row['latent_relative_path']
        npy(path, np.zeros((7, 64), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'Changed local'):
            importer.validate_local_receipt(self.row, result, self.output, self.contract_sha, None)

    def test_snapshot_rejects_wrong_hash_and_physical_source_symlink(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            importer.snapshot_bytes(self.row['audio_path'], '0' * 64)
        link = self.base / 'link.npy'; link.symlink_to(self.origin / self.row['latent_relative_path'])
        with self.assertRaisesRegex(ValueError, 'physical source'):
            importer.snapshot_bytes(link, regular=True)

    def test_paths_cannot_escape_or_alias_origin(self):
        with self.assertRaises(ValueError):
            importer.safe(self.output, '../origin/file')
        (self.output / 'alias').symlink_to(self.origin, target_is_directory=True)
        with self.assertRaises(ValueError):
            importer.safe(self.output, 'alias/file.npy')

    def test_owner_lock_is_not_inherited_and_owner_unlocks(self):
        lock = self.output / 'owner.lock'
        code = ('import os,sys; target=sys.argv[1]; '
                'print(any(os.path.realpath("/proc/self/fd/"+x)==target for x in os.listdir("/proc/self/fd")))')
        with importer.owned_lock(lock):
            result = subprocess.check_output([sys.executable, '-c', code, str(lock)], close_fds=True, text=True)
            self.assertEqual(result.strip(), 'False')
            with self.assertRaises(BlockingIOError):
                with importer.owned_lock(lock):
                    pass
        with importer.owned_lock(lock):
            pass

    def create_finished_shard(self, *, stale_guard=False):
        root = self.output
        directory, data_root, complete_path = importer.shard_paths(root, 0)
        result = importer.process_row(self.row, self.proof, self.origin, data_root, self.contract_sha)
        receipt_rel = f'reports/import_receipts/{self.stem}.json'
        record = {'sample_id': self.row['sample_id'], 'relative_stem': self.stem,
                  'status': 'ok', 'video40_action': result['video40_action'],
                  'import_receipt_path': receipt_rel,
                  'import_receipt_sha256': importer.digest(data_root / receipt_rel)}
        manifest_rel, proof_rel = importer.RUN + '/inputs/shard0000.jsonl', importer.RUN + '/inputs/shard0000.proof.jsonl'
        importer.atomic_bytes(root / manifest_rel, importer.payload(self.row))
        importer.atomic_bytes(root / proof_rel, importer.payload(self.proof))
        exclusion_rel = importer.RUN + '/exclusions.json'; importer.write_json(root / exclusion_rel, {})
        importer.atomic_bytes(root / 'manifests/source_inventory.jsonl', importer.payload(self.row))
        plan = {'schema_version': 1, 'policy': importer.POLICY, 'scope': 'full',
                'destination_root': str(root), 'origin_root': str(self.origin), 'num_shards': 1,
                'requested': 1, 'quality_excluded': 0, 'import_contract_sha256': self.contract_sha,
                'source_manifest_sha256': importer.digest(root / 'manifests/source_inventory.jsonl'),
                'exclusions': exclusion_rel, 'exclusions_sha256': importer.digest(root / exclusion_rel),
                'shards': [{'index': 0, 'manifest': manifest_rel,
                    'manifest_sha256': importer.digest(root / manifest_rel), 'proof': proof_rel,
                    'proof_sha256': importer.digest(root / proof_rel), 'requested': 1}]}
        importer.write_json(root / importer.PLAN, plan)
        plan_sha = importer.digest(root / importer.PLAN)
        importer.atomic_bytes(directory / 'records.jsonl', importer.payload(record))
        importer.write_json(complete_path, {'schema_version': 1, 'policy': importer.POLICY,
            'plan_sha256': plan_sha, 'shard': 0, 'import_contract_sha256': self.contract_sha,
            'requested': 1, 'processed': 1, 'failed': 0, 'quality_excluded': 0,
            'records_sha256': importer.digest(directory / 'records.jsonl')})
        if stale_guard:
            importer.write_json(directory / 'WRITE_ACTIVE.json',
                {'pid': 99999999, 'sid': 99999999, 'plan_sha256': plan_sha})
        contract = {'_sha256': self.contract_sha, 'source_manifest_sha256': plan['source_manifest_sha256'],
                    'origin_root': str(self.origin), 'destination_root': str(root),
                    'terminal': {'requested': 1, 'verified': 1, 'excluded_count': 0}}
        return plan, plan_sha, contract

    def test_completed_shard_recovers_interruption_before_guard_removal(self):
        plan, fingerprint, contract = self.create_finished_shard(stale_guard=True)
        with patch.object(importer, 'get_plan', return_value=(plan, fingerprint, contract)):
            importer.worker(self.output, 0, fingerprint)
        self.assertFalse((importer.shard_paths(self.output, 0)[0] / 'WRITE_ACTIVE.json').exists())

    def test_unknown_guard_is_never_removed_even_with_completed_data(self):
        plan, fingerprint, contract = self.create_finished_shard(stale_guard=True)
        guard = importer.shard_paths(self.output, 0)[0] / 'WRITE_ACTIVE.json'
        importer.write_json(guard, {'pid': 99999999, 'plan_sha256': 'foreign'})
        with patch.object(importer, 'get_plan', return_value=(plan, fingerprint, contract)):
            with self.assertRaisesRegex(ValueError, 'unknown writer'):
                importer.worker(self.output, 0, fingerprint)
        self.assertTrue(guard.exists())

    def test_merge_completion_matches_shared_api_and_physical_layout(self):
        plan, fingerprint, contract = self.create_finished_shard()
        with patch.object(importer, 'get_plan', return_value=(plan, fingerprint, contract)), \
             patch.object(importer, 'validate_imported_source', return_value=contract):
            importer.merge(self.output, fingerprint, threads=2)
        validated = validate_import_completion(self.output, contract)
        self.assertEqual(set(validated['receipts']), {self.row['sample_id']})
        self.assertEqual(validated['scope']['source_manifest_sha256'], plan['source_manifest_sha256'])
        local = self.output / self.row['latent_relative_path']
        shard = importer.shard_paths(self.output, 0)[1] / self.row['latent_relative_path']
        original = self.origin / self.row['latent_relative_path']
        self.assertEqual(local.stat().st_ino, shard.stat().st_ino)
        self.assertNotEqual(local.stat().st_ino, original.stat().st_ino)

    def test_incomplete_or_duplicate_shard_id_set_cannot_merge(self):
        plan, fingerprint, contract = self.create_finished_shard()
        directory = importer.shard_paths(self.output, 0)[0]
        record = importer.read_rows(directory / 'records.jsonl')[0]
        importer.atomic_bytes(directory / 'records.jsonl', importer.payload(record) * 2)
        completion = importer.read_json(directory / 'complete.json')
        completion['records_sha256'] = importer.digest(directory / 'records.jsonl')
        importer.write_json(directory / 'complete.json', completion)
        with self.assertRaisesRegex(ValueError, 'exact requested ID set'):
            importer.check_shard(self.output, plan, fingerprint, 0, validate_files=False)
        self.assertFalse((self.output / importer.COMPLETION).exists())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
