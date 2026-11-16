"""CPU tests for explicit quality closure; all writes use temporary fixtures."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aligndit.lrs3_media_quality import (
    EXCLUSIONS, TERMINAL, PROTOCOL, build_terminal, load_terminal, row_hash, scope, sha256, snapshot,
)
import lrs3_seal_media_quality as seal
from lrs3_finish_production import Finisher, cache_status
from lrs3_build_inventory import write_json, write_jsonl


class QualityClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'state/locks').mkdir(parents=True)
        self.directory = self.root / 'reports/media_full_pipeline'
        self.canonical = self.root / 'manifests/source_inventory.jsonl'
        self.reviewed = self.root / 'reviewed.json'
        self.guard = self.root / 'state/video_25hz/WRITE_ACTIVE.json'
        self.records = []
        for key in ('good', 'bad'):
            source = self.root / (key + '.mp4')
            source.write_bytes(key.encode())
            stem = 'train/pretrain/' + key
            self.records.append({'sample_id': key, 'relative_stem': stem, 'original_clip_id': 'pretrain/v/' + key,
                'video_frames': 25, 'latent_frames': 40, 'width': 224, 'video_path': str(source),
                'text_status': 'original_pipeline', 'utterance_key': 'lrs3/' + stem,
                'video_25hz_relative_path': 'avhubert_video_25hz/' + stem + '.npy',
                'video_40hz_relative_path': 'video_40hz/' + stem + '.npy'})
        write_jsonl(self.canonical, self.records)
        self.source_sha = sha256(self.canonical)
        write_json(self.root / 'reports/source_contract.json', {'source_inventory_sha256': self.source_sha,
            'source_media_pairs': 2, 'original_clip_count': 2})
        self.failure = {**self.records[1], 'status': 'failed',
                        'error': "ValueError('Landmark coverage requires review: missing=14')"}
        self.proof = self.directory / 'independent_verification.jsonl'
        write_jsonl(self.proof, [{**self.records[0], 'status': 'ok'}, self.failure])
        self.summary = {'complete': False, 'requested': 2, 'verified': 1, 'failed': 1, 'original_ids': 2,
                        'complete_original_ids': 1, 'protocol_sha256': PROTOCOL}
        write_json(self.directory / 'summary.json', self.summary)
        self.stages = {}
        for name in ('full_media_landmark_repair', 'full_media_retry', 'full_media_retry_long'):
            report = self.root / 'reports' / (name + '.jsonl')
            write_jsonl(report, [self.failure])
            self.stages[name] = {'finished': True, 'reports': [str(report)],
                'processes': [{'report': str(report), 'exit_code': 1, 'pid': -1}]}
        self.state = {'status': 'finished_with_failures', 'summary': self.summary, 'manifest': str(self.canonical),
                      'manifest_sha256': self.source_sha, 'stages': self.stages}
        write_json(self.directory / 'state.json', self.state)
        write_json(self.reviewed, [{'sample_id': 'bad', 'reason': 'landmark_coverage_insufficient'}])
        write_json(self.guard, {'pid': -1, 'stage': 'media'})

    def invoke(self, sealing=True):
        argv = ['seal', '--root', str(self.root), '--exclusions', str(self.reviewed)] + (['--seal'] if sealing else [])
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(seal, 'active_media_writers', return_value=[]), \
             mock.patch('builtins.print'):
            seal.main()
        return load_terminal(self.root)

    def test_readonly_proposal_preserves_guard_and_all_media_state(self):
        before = {str(p): p.read_bytes() for p in (self.guard, self.proof, self.directory / 'state.json')}
        self.assertIsNone(self.invoke(False))
        self.assertFalse((self.root / EXCLUSIONS).exists())
        self.assertEqual(before, {name: Path(name).read_bytes() for name in before})

    def test_seal_is_exact_audited_terminal_and_restart_is_idempotent(self):
        before = {str(p): p.read_bytes() for p in (self.proof, self.directory / 'state.json', self.directory / 'summary.json')}
        terminal = self.invoke()
        self.assertEqual((terminal['requested'], terminal['verified'], terminal['excluded_count']), (2, 1, 1))
        self.assertFalse(terminal['media_complete'])
        self.assertFalse(self.guard.exists())
        self.assertEqual(before, {name: Path(name).read_bytes() for name in before})
        self.assertEqual(self.invoke()['_sha256'], terminal['_sha256'])

    def test_exact_review_set_and_allowed_reason_are_required(self):
        for value in ([], [{'sample_id':'bad','reason':'IO'}],
                      [{'sample_id':'good','reason':'landmark_coverage_insufficient'}],
                      [{'sample_id':'bad','reason':'landmark_coverage_insufficient'}] * 2):
            write_json(self.reviewed, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_terminal(self.root, self.reviewed)
        self.assertTrue(self.guard.exists())

    def test_incomplete_or_noncanonical_independent_proof_is_rejected(self):
        for rows in ([self.failure], [{**self.records[0], 'status':'ok'}, {**self.failure, 'sample_id':'unknown'}]):
            write_jsonl(self.proof, rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build_terminal(self.root, self.reviewed)

    def test_prior_quality_failure_cannot_hide_latest_io_failure(self):
        path = self.stages['full_media_retry_long']['reports'][0]
        write_jsonl(Path(path), [{**self.failure, 'error': "OSError('disk full')"}])
        with self.assertRaisesRegex(ValueError, 'Latest media retry'):
            build_terminal(self.root, self.reviewed)

    def test_fan_io_or_oom_cannot_be_hidden_by_legacy_retry_quality(self):
        path = Path(self.stages['full_media_landmark_repair']['reports'][0])
        for error in ("OSError('disk full')", "OutOfMemoryError('CUDA out of memory')", "RuntimeError('unknown')"):
            write_jsonl(path,[{**self.failure,'error':error}])
            with self.subTest(error=error), self.assertRaisesRegex(ValueError,'Latest formal FAN failure'):
                build_terminal(self.root,self.reviewed)
        self.assertTrue(self.guard.exists())

    def test_successful_fan_then_source_bound_crop_geometry_is_allowed(self):
        fan_path = Path(self.stages['full_media_landmark_repair']['reports'][0])
        fan = {**self.records[1],'status':'ok','source_sha256':sha256(self.records[1]['video_path'])}
        write_jsonl(fan_path,[fan])
        geometry = {**self.failure,'error':"ValueError('Mouth ROI outside canonical canvas: (1, 2)')"}
        for name in ('full_media_retry','full_media_retry_long'):
            write_jsonl(Path(self.stages[name]['reports'][0]),[geometry])
        write_json(self.reviewed,[{'sample_id':'bad','reason':'mouth_geometry_invalid'}])
        terminal = self.invoke()
        self.assertEqual(terminal['excluded'][0]['fan_evidence']['status'],'ok')
        self.assertEqual(terminal['excluded'][0]['retry_evidence']['status'],'failed')
        self.assertEqual(terminal['excluded'][0]['quality_evidence'],terminal['excluded'][0]['retry_evidence'])

    def test_successful_fan_cannot_authorize_a_legacy_coverage_failure(self):
        fan_path = Path(self.stages['full_media_landmark_repair']['reports'][0])
        write_jsonl(fan_path,[{**self.records[1],'status':'ok','source_sha256':sha256(self.records[1]['video_path'])}])
        with self.assertRaisesRegex(ValueError,'only a source-bound final crop geometry'):
            build_terminal(self.root,self.reviewed)

    def test_consumer_revalidates_fan_chain_even_if_all_hashes_are_recomputed(self):
        self.invoke()
        path = Path(self.stages['full_media_landmark_repair']['reports'][0])
        failure = {**self.failure,'error':"OSError('disk full')"}
        write_jsonl(path,[failure])
        document = json.loads((self.root / EXCLUSIONS).read_text())
        document['excluded'][0]['fan_evidence'].update(record_sha256=row_hash(failure),error=failure['error'])
        write_json(self.root / EXCLUSIONS,document)
        terminal = json.loads((self.root / TERMINAL).read_text())
        terminal['evidence_files'] = [snapshot(ref['path']) for ref in terminal['evidence_files']]
        write_json(self.root / TERMINAL,terminal)
        with self.assertRaisesRegex(ValueError,'Latest formal FAN failure'):load_terminal(self.root)

    def test_fake_embedded_quality_string_is_not_accepted(self):
        for stage in self.stages.values():
            write_jsonl(Path(stage['reports'][0]), [{**self.failure,
                'error': "OSError('Landmark coverage requires review: disk missing')"}])
        with self.assertRaises(ValueError):build_terminal(self.root, self.reviewed)

    def test_unfinished_recovery_cannot_be_sealed(self):
        self.state['stages']['full_media_retry']['finished'] = False
        write_json(self.directory / 'state.json', self.state)
        with self.assertRaises(ValueError):self.invoke()
        self.assertTrue(self.guard.exists())
        self.assertFalse((self.root / TERMINAL).exists())

    def test_evidence_source_and_explicit_hash_tampering_is_rejected(self):
        terminal = self.invoke()
        with self.assertRaises(ValueError):load_terminal(self.root, expected_sha256='0' * 64)
        with self.assertRaises(ValueError):load_terminal(self.root, source_sha256='0' * 64)
        for path in (self.reviewed, self.proof, Path(self.records[1]['video_path'])):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):load_terminal(self.root)
            path.write_bytes(original)
        self.assertEqual(load_terminal(self.root)['_sha256'], terminal['_sha256'])

    def test_interrupted_publication_never_looks_complete(self):
        write_json(self.root / EXCLUSIONS, {'policy':'unused'})
        with self.assertRaises(ValueError):load_terminal(self.root)

    def test_active_writer_rejects_sealing_without_removing_guard(self):
        argv = ['seal', '--root', str(self.root), '--exclusions', str(self.reviewed), '--seal']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(seal, 'active_media_writers', return_value=[123]), \
             self.assertRaises(RuntimeError):seal.main()
        self.assertTrue(self.guard.exists())
        self.assertFalse((self.root / TERMINAL).exists())

    def test_writer_scope_supports_equals_relative_and_symlink_paths(self):
        link = self.root.parent / (self.root.name + '_link')
        link.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(link.unlink)
        for command in (['--root',str(self.root)], ['--output-root='+str(self.root)],
                        ['--output-root',self.root.name], ['--root='+str(link)]):
            self.assertTrue(seal.command_targets_root(command,self.root,self.root.parent))
        self.assertFalse(seal.command_targets_root(['--output-root=/elsewhere'],self.root,self.root.parent))

    def test_quality_video40_worker_skips_even_stale_valid_arrays(self):
        import lrs3_cache_worker as worker
        terminal = self.invoke()
        for row in self.records:
            path = self.root / row['video_25hz_relative_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ones((25, 1024), dtype=np.float32))
        bad_out = self.root / self.records[1]['video_40hz_relative_path']
        bad_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(bad_out, np.full((40, 1024), 7, dtype=np.float32))
        previous = sha256(bad_out)
        argv = ['worker', '--root', str(self.root), '--manifest', str(self.canonical), '--stage', 'video40',
                '--device', 'cpu', '--media-quality-terminal', str(self.root / TERMINAL),
                '--media-quality-terminal-sha256', terminal['_sha256']]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(worker, 'init_spec', return_value={}), \
             mock.patch('builtins.print'):worker.main()
        summary = json.loads(next((self.root / 'reports').glob('video40.*.summary.json')).read_text())
        self.assertEqual((summary['requested'], summary['succeeded'], summary['quality_excluded'], summary['failed']), (2,1,1,0))
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['processing_complete'])
        self.assertEqual(sha256(bad_out), previous)
        self.assertFalse((self.root / 'state/video40_receipts' / (self.records[1]['relative_stem']+'.json')).exists())

    def test_parallel_counts_exact_exclusions_and_rejects_wrong_worker_reason(self):
        import lrs3_cache_parallel as coordinator
        terminal = self.invoke()
        calls = []
        wrong = [False]
        def worker(command, **kwargs):
            rank = int(command[command.index('--rank')+1])
            manifest = Path(command[command.index('--manifest')+1])
            key = 'good' if rank == 0 else 'bad'
            result = {'sample_id':key, 'status':'ok' if rank == 0 else 'excluded'}
            if rank:
                result.update(reason='media_quality:' + ('wrong' if wrong[0] else 'landmark_coverage_insufficient'),
                              media_quality_terminal_sha256=terminal['_sha256'])
            path = self.root / 'reports' / ('video40.fixture'+str(len(calls))+'.jsonl')
            write_jsonl(path,[result])
            write_json(path.with_suffix('.summary.json'), {'manifest_sha256':sha256(manifest),'progress_path':str(path)})
            calls.append(command)
            return SimpleNamespace(pid=-1,poll=lambda:0,returncode=0)
        argv = ['parallel','--root',str(self.root),'--manifest',str(self.canonical),'--stage','video40',
                '--gpus','cpu,cpu','--name','quality_final','--media-quality-terminal',str(self.root / TERMINAL),
                '--media-quality-terminal-sha256',terminal['_sha256']]
        with mock.patch.object(sys,'argv',argv), mock.patch.object(coordinator.subprocess,'Popen',side_effect=worker), \
             mock.patch('builtins.print'):coordinator.main()
        summary = json.loads((self.root / 'reports/quality_final/summary.json').read_text())
        self.assertEqual((summary['requested'],summary['succeeded'],summary['quality_excluded'],summary['failed']),(2,1,1,0))
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['processing_complete'])
        self.assertFalse((self.root / 'state/video_40hz/WRITE_ACTIVE.json').exists())
        self.assertTrue(all('--media-quality-terminal-sha256' in command for command in calls))
        wrong[0] = True
        argv[argv.index('--name')+1] = 'quality_reject'
        with mock.patch.object(sys,'argv',argv), mock.patch.object(coordinator.subprocess,'Popen',side_effect=worker), \
             self.assertRaisesRegex(RuntimeError,'unauthorized quality exclusion'):coordinator.main()
        self.assertTrue((self.root / 'state/video_40hz/WRITE_ACTIVE.json').exists())

    def make_finisher(self):
        args = argparse.Namespace(root=self.root, name='finish', existing_wave=[], audio_run='audio', workers=8,
                                  finalizer_workers=16, poll_seconds=.001, wave_seconds=300, min_wave=10000)
        lock = (self.root / 'state/locks/finish.lock').open('a')
        self.addCleanup(lock.close)
        return Finisher(args, lock.fileno())

    def test_finisher_accepts_exact_closure_and_uses_unique_canonical_attempt(self):
        self.invoke()
        finish = self.make_finisher()
        self.assertTrue(finish.media_done())
        finish.state['stages']['finish_canonical_video40'] = {'status':'failed', 'log':'retained.log'}
        previous = dict(finish.state['stages']['finish_canonical_video40'])
        with mock.patch.object(finish, 'audio_done', return_value=True), \
             mock.patch.object(finish, 'text_done', return_value=True), \
             mock.patch.object(finish, 'wave') as wave, mock.patch.object(finish, 'publish'):
            finish.run_all()
        entry = wave.call_args.args[0]
        self.assertIn('_canonical_video40_quality_', entry['name'])
        self.assertTrue(entry['canonical_quality'])
        self.assertEqual(entry['count'], 2)
        self.assertEqual(finish.state['stages']['finish_canonical_video40'], previous)
        resumed = self.make_finisher()
        self.assertTrue(resumed.media_done())

    def test_quality_completion_cannot_claim_full_cache_success(self):
        terminal = self.invoke()
        summary = {'stage':'video40','source_manifest':str(self.canonical),'manifest_sha256':self.source_sha,
            'requested':2,'succeeded':1,'quality_excluded':1,'failed':0,'complete':False,'processing_complete':True,
            'finished_unix':1,'media_quality_terminal_sha256':terminal['_sha256']}
        self.assertTrue(cache_status(summary,'video40',self.canonical,self.source_sha,2,terminal))
        for change in ({'complete':True}, {'quality_excluded':0}, {'succeeded':2}, {'failed':1},
                       {'media_quality_terminal_sha256':'bad'}, {'finished_unix':None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cache_status({**summary,**change},'video40',self.canonical,self.source_sha,2,terminal)
        with self.assertRaises(RuntimeError):cache_status(summary,'video40',self.canonical,self.source_sha,2)

    def test_failed_canonical_guard_transition_retains_old_evidence(self):
        self.invoke()
        finish = self.make_finisher()
        finish.media_done()
        run = self.root / 'reports/finish_canonical_video40'
        progress = run / 'progress.jsonl'
        write_jsonl(progress,[{'sample_id':'good','status':'ok'},{'sample_id':'bad','status':'error'}])
        worker = run / 'worker.json'
        write_json(worker, {'progress_path':str(progress)})
        summary = {'stage':'video40','source_manifest':str(self.canonical),'manifest_sha256':self.source_sha,
            'requested':2,'succeeded':1,'failed':1,'complete':False,'finished_unix':1,'pid':-1,
            'rounds':[{'workers':[{'pid':-1}], 'summaries':[str(worker)]}]}
        write_json(run / 'summary.json', summary)
        guard = self.root / 'state/video_40hz/WRITE_ACTIVE.json'
        write_json(guard, {'pid':-1,'run':str(run),'stage':'video40'})
        previous = guard.read_bytes()
        self.assertTrue(finish.quiet(['video40']))
        self.assertEqual(guard.read_bytes(), previous)
        self.assertFalse(json.loads((run / 'summary.json').read_text())['complete'])
        write_jsonl(progress,[{'sample_id':'good','status':'error'},{'sample_id':'bad','status':'error'}])
        with self.assertRaises(ValueError):finish.quiet(['video40'])


class ConsumerQualityTests(unittest.TestCase):
    def test_finalizer_rejects_sealed_id_even_when_all_cache_receipts_pass(self):
        import test_lrs3_cache_integrity as fixtures
        import lrs3_finalize_cache as finalizer
        fixture = fixtures.PublicationIntegrityTests('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        row, reasons = fixture.validate()
        self.assertFalse(reasons)
        exclusion = {key: row[key] for key in ('sample_id','relative_stem','original_clip_id')}
        exclusion['reason'] = 'landmark_coverage_insufficient'
        checked, reasons = finalizer.validate(fixture.row,fixture.root,{c:i for i,c in enumerate('helo')},{},True,
            media_quality_exclusions={row['sample_id']:exclusion})
        self.assertEqual(reasons, ['media_quality:landmark_coverage_insufficient'])
        self.assertEqual(len(checked['cache_sha256']),4)

    def test_loader_requires_scope_pin_and_rejects_excluded_stem(self):
        import test_lrs3_semantic_vae_dataset as fixtures
        import aligndit.model.lrs3_semantic_vae_dataset as loader
        fixture = fixtures.LRS3ContractTests('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        bad = fixture.rows[0]
        terminal = {'_path':str(fixture.root / TERMINAL),'_sha256':'a'*64,'requested':4,'verified':3,
                    'excluded_count':1,'evidence_files':[],
                    'excluded':[{'sample_id':'bad','relative_stem':bad['relative_stem']}]}
        with mock.patch.object(loader,'load_terminal',return_value=terminal):
            with self.assertRaisesRegex(RuntimeError,'fixed media quality'):fixture.dataset()
            for name in ('inventory_meta.json','cache_completion.json'):
                path = fixture.manifests / name
                value = json.loads(path.read_text());value['media_quality']=scope(terminal);fixture.write(path,value)
            with self.assertRaisesRegex(ValueError,'sealed media quality exclusion'):fixture.dataset()


if __name__ == '__main__':unittest.main()
