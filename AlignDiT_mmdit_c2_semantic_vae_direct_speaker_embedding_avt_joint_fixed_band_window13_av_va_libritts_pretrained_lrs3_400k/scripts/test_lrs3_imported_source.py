"""Real physical importer -> publication -> loader integration, entirely in temp dirs."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import soundfile as sf
import torch

import lrs3_sharded_import as importer
import lrs3_finalize_cache as finalizer
import test_lrs3_media_quality as quality_fixture
from aligndit.lrs3_imported_source import (
    ARTIFACTS, CONTRACT, COMPLETION, LEDGER, PLAN, build_import_contract, read_import_contract,
    validate_imported_source, imported_source_scope, validate_import_completion, validate_import_receipt,
)
from aligndit.lrs3_media_quality import load_terminal, excluded_map, sha256
from aligndit.model.lrs3_semantic_vae_dataset import SemanticVaeLRS3Dataset
from aligndit.model.semantic_vae_dataset import SEMANTIC_VAE_FEATURE


class ImportedSourceTests(unittest.TestCase):
    def setUp(self):
        self.q = quality_fixture.QualityClosureTests()
        self.q.setUp()
        self.addCleanup(self.q.doCleanups)
        self.origin = self.q.root
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'destination'
        self.root.mkdir()
        self.rows = self.q.records
        for row in self.rows:
            stem = row['relative_stem']
            row.update(split='train', source_split='pretrain', text='aa', audio_seconds=1.0,
                       video_seconds=1.0, duration_eligible=True, latent_dim=64, video_dim=1024,
                       posterior_seed=666, original_num_samples=16000, height=96, width=96,
                       label_source='fixture', label_validation='fixture')
            if row['sample_id']=='bad':row.update(width=224,height=224)
            for name, folder, suffix in [('audio','audio','.wav'),('latent','latents','.npy'),
                    ('speaker','speaker_embeddings','.npy'),('mouth','video_mouth','.mp4')]:
                row[name+'_relative_path'] = folder + '/' + stem + suffix
            row['audio_path'] = str(self.origin / (row['sample_id'] + '.wav'))
            sf.write(row['audio_path'], np.linspace(-.1,.1,16000,dtype=np.float32), 16000, subtype='PCM_16')
            writer = cv2.VideoWriter(row['video_path'], cv2.VideoWriter_fourcc(*'mp4v'),25,(row['width'],row['height']))
            self.assertTrue(writer.isOpened())
            for i in range(25): writer.write(np.full((row['height'],row['width'],3),i,dtype=np.uint8))
            writer.release()
            mouth = self.origin / row['mouth_relative_path']; mouth.parent.mkdir(parents=True,exist_ok=True)
            mouth.symlink_to(row['video_path'])
            for key, array in [('latent',np.ones((40,64),dtype=np.float32)),
                    ('speaker',np.eye(1,192,dtype=np.float32).reshape(192)),
                    ('video_25hz',np.ones((25,1024),dtype=np.float32)),
                    ('video_40hz',np.ones((40,1024),dtype=np.float32))]:
                path=self.origin/row[key+'_relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,array,allow_pickle=False)
            self.write(self.origin/f'state/audio_receipts/{stem}.json',
                {'identity':importer.audio_identity(row,sha256(row['audio_path'])),
                 'outputs':{key:sha256(self.origin/row[key+'_relative_path']) for key in ('latent','speaker')}})
            feature_hash=sha256(self.origin/row['video_25hz_relative_path'])
            self.write(self.origin/f'reports/media_receipts/{stem}.json',
                {'relative_stem':stem,'source_path':row['video_path'],'source_sha256':sha256(row['video_path']),
                 'protocol_sha256':importer.PROTOCOL,'features_complete':True,'video_frames':25,
                 'feature_sha256':feature_hash,'crop_kind':'source_96'})
            self.write(self.origin/f'state/video40_receipts/{stem}.json',
                {'identity':importer.video40_identity(row,feature_hash),
                 'outputs':{'video_40hz':sha256(self.origin/row['video_40hz_relative_path'])}})
        importer.atomic_bytes(self.q.canonical,b''.join(importer.payload(row) for row in self.rows))
        source_hash=sha256(self.q.canonical)
        self.write(self.origin/'reports/source_contract.json',
                   {'schema_version':1,'source_inventory_sha256':source_hash,'source_media_pairs':2,'original_clip_count':2})
        self.q.state['manifest_sha256']=source_hash
        self.write(self.q.directory/'state.json',self.q.state)
        good={**self.rows[0],'status':'ok','feature_sha256':sha256(self.origin/self.rows[0]['video_25hz_relative_path'])}
        importer.atomic_bytes(self.q.proof,importer.payload(good)+importer.payload(self.q.failure))
        self.write(self.origin/'state/latents/spec.json',{'dataset_root':str(self.origin)})
        self.terminal=self.q.invoke()
        self.vocab=self.base/'zjw524/projects/data/CelebVDub_char/vocab.txt'
        self.vocab.parent.mkdir(parents=True);self.vocab.write_text(' \na\n')
        self.norm=self.base/'norm.json'
        self.write(self.norm,{'channel_count':64,'feature':SEMANTIC_VAE_FEATURE,
            'method':'per_channel_population_mean_std_float64_welford_v1','scope':'train',
            'mean':[0.0]*64,'std':[1.0]*64})

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

    def contract_only(self):
        value=build_import_contract(self.origin,self.root)
        for relative in ARTIFACTS.values():
            destination=self.root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.origin/relative,destination)
        self.write(self.root/CONTRACT,value)
        return validate_imported_source(self.root)

    def import_all(self, *, pilot=False, recompute=False):
        if recompute: (self.origin/self.rows[0]['video_40hz_relative_path']).unlink()
        selection=None
        if pilot:
            selection=self.base/'selection.jsonl';importer.atomic_bytes(selection,importer.payload(self.rows[0]))
        with patch('builtins.print'):
            importer.prepare(self.root,self.origin,2,selection)
            pin=sha256(self.root/PLAN)
            for index in range(2): importer.worker(self.root,index,pin)
            importer.merge(self.root,pin,threads=2)
        contract=validate_imported_source(self.root)
        return contract

    def publish(self, name='full', pseudo=False):
        args=['finalizer','--root',str(self.root),'--source-manifest',str(self.root/'manifests/source_inventory.jsonl'),
              '--name',name,'--mode','full','--workers','2']
        if pseudo:args+=['--text-policy','source_plus_asr_pseudo','--strict-baseline','manifests/full']
        with patch.object(sys,'argv',args),patch.dict(os.environ,{'ROOT_PREFIX':str(self.base)}),patch('builtins.print'):
            finalizer.main()

    def dataset(self,name='full',pseudo=False):
        return SemanticVaeLRS3Dataset(manifest_path=self.root/f'manifests/{name}/train.jsonl',cache_root=self.root,
            normalization_path=self.norm,vocab_path=self.vocab,expected_normalization_sha256=sha256(self.norm),
            expected_vocab_sha256=sha256(self.vocab),expected_selection='full',
            expected_text_policy='source_plus_asr_pseudo' if pseudo else 'source_grounded',
            speaker_embedding_model_id='iic/speech_campplus_sv_zh_en_16k-common_advanced',
            speaker_embedding_checkpoint_sha256='92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199')

    def test_real_full_import_finalizer_loader_and_pseudo_keep_exact_closures(self):
        contract=self.import_all(recompute=True)
        complete=validate_import_completion(self.root,contract,self.rows)
        self.assertEqual(set(complete['receipts']),{'good','bad'})
        self.assertEqual((complete['scope']['media_success'],complete['scope']['quality_excluded']),(1,1))
        exclusions=excluded_map(contract['terminal'])
        for row in self.rows:
            validate_import_receipt(self.root,row,contract,complete['receipts'][row['sample_id']],
                                    exclusion=exclusions.get(row['sample_id']))
        self.publish()
        data=self.dataset()
        self.assertEqual(len(data),1)
        batch=data.collate_fn([data[0]])
        self.assertEqual(tuple(batch['mel'].shape),(1,64,40))
        self.assertEqual(tuple(batch['video'].shape),(1,40,1024))
        self.assertEqual(data.data_contract['imported_source'],imported_source_scope(contract))
        self.assertEqual(data.data_contract['imported_cache'],complete['scope'])
        self.assertFalse((self.root/self.rows[0]['audio_relative_path']).is_symlink())
        quality=importer.read_rows(self.root/'reports/full_sample_quality.jsonl')
        self.assertEqual([row['sample_id'] for row in quality if row['ready']],['good'])
        self.assertTrue(all(row.get('import_receipt_sha256') for row in quality))
        self.assertTrue(any(x.startswith('media_quality:') for x in quality[1]['reasons']))
        self.publish('full_pseudo',pseudo=True)
        pseudo=self.dataset('full_pseudo',pseudo=True)
        self.assertEqual(len(pseudo),1)
        self.assertEqual(data.data_contract['imported_source'],pseudo.data_contract['imported_source'])
        for split in ('valid','test'):
            self.assertEqual((self.root/f'manifests/full/{split}.jsonl').read_bytes(),
                             (self.root/f'manifests/full_pseudo/{split}.jsonl').read_bytes())

    def test_original_evidence_history_stays_at_origin_and_exact_copies_are_required(self):
        before=(self.origin/ARTIFACTS['media_quality_terminal_sha256']).read_bytes()
        contract=self.contract_only()
        terminal=load_terminal(self.root,expected_sha256=sha256(self.origin/ARTIFACTS['media_quality_terminal_sha256']))
        self.assertEqual(terminal['_sha256'],self.terminal['_sha256'])
        self.assertEqual(terminal['_imported_source']['origin_root'],str(self.origin))
        self.assertEqual(before,(self.root/ARTIFACTS['media_quality_terminal_sha256']).read_bytes())
        self.assertEqual(before,(self.origin/ARTIFACTS['media_quality_terminal_sha256']).read_bytes())
        with self.assertRaisesRegex(ValueError,'pinned SHA256'):validate_imported_source(self.root,'0'*64)
        self.assertEqual(read_import_contract(self.root)['_sha256'],contract['_sha256'])

    def test_mutated_origin_local_copy_or_contract_fails_closed(self):
        self.contract_only()
        for root in (self.origin,self.root):
            for relative in ARTIFACTS.values():
                path=root/relative;before=path.read_bytes();path.write_bytes(before+b' ')
                with self.subTest(root=root,relative=relative),self.assertRaises(ValueError):
                    validate_imported_source(self.root)
                path.write_bytes(before)
        path=self.root/CONTRACT;value=json.loads(path.read_bytes());value['source_manifest_sha256']='0'*64
        self.write(path,value)
        with self.assertRaises(ValueError):validate_imported_source(self.root)

    def test_symlink_and_nested_or_cyclic_origins_rejected(self):
        self.contract_only()
        relative=ARTIFACTS['source_contract_sha256'];path=self.root/relative;path.unlink();path.symlink_to(self.origin/relative)
        with self.assertRaisesRegex(ValueError,'physical regular'):validate_imported_source(self.root)
        path.unlink();shutil.copyfile(self.origin/relative,path)
        for value in ({}, {'origin_root':str(self.root)}):
            self.write(self.origin/CONTRACT,value)
            with self.assertRaisesRegex(ValueError,'Nested or cyclic'):validate_imported_source(self.root)
        (self.origin/CONTRACT).unlink()
        with self.assertRaisesRegex(ValueError,'disjoint'):build_import_contract(self.origin,self.origin)
        nested=self.origin/'nested';nested.mkdir()
        with self.assertRaisesRegex(ValueError,'disjoint'):build_import_contract(self.origin,nested)

    def test_real_pilot_completion_cannot_authorize_full_publication(self):
        contract=self.import_all(pilot=True)
        with self.assertRaisesRegex(ValueError,'exact full-source'):validate_import_completion(self.root,contract)
        with self.assertRaisesRegex(ValueError,'exact full-source'):self.publish()
        self.assertFalse((self.root/'manifests/full/cache_completion.json').exists())

    def test_full_completion_and_plan_semantics_are_not_replaceable_by_matching_hashes(self):
        contract=self.import_all()
        path=self.root/COMPLETION;original=path.read_bytes()
        for field,value in [('scope','pilot'),('status','running'),('processed',1),('failed',1),('quality_excluded',0)]:
            completion=json.loads(original);completion[field]=value;self.write(path,completion)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_import_completion(self.root,contract)
        path.write_bytes(original)
        plan=json.loads((self.root/PLAN).read_bytes());plan['scope']='pilot';self.write(self.root/PLAN,plan)
        completion=json.loads(original);completion['plan_sha256']=sha256(self.root/PLAN);self.write(path,completion)
        with self.assertRaisesRegex(ValueError,'contradicts'):validate_import_completion(self.root,contract)

    def test_duplicate_foreign_or_missing_ledger_ids_rejected_even_with_rehashed_completion(self):
        contract=self.import_all();ledger=importer.read_rows(self.root/LEDGER)
        for rows in ([ledger[0]], [ledger[0],ledger[0]], [ledger[0],{**ledger[1],'sample_id':'foreign'}]):
            importer.atomic_bytes(self.root/LEDGER,b''.join(importer.payload(row) for row in rows))
            completion=importer.read_json(self.root/COMPLETION);completion['ledger_sha256']=sha256(self.root/LEDGER)
            self.write(self.root/COMPLETION,completion)
            with self.subTest(rows=rows),self.assertRaises(ValueError):validate_import_completion(self.root,contract)

    def test_tampered_import_receipt_aborts_publication_instead_of_excluding_it(self):
        self.import_all()
        path=self.root/f'reports/import_receipts/{self.rows[0]["relative_stem"]}.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Immutable imported cache provenance failed'):self.publish()
        self.assertFalse((self.root/'manifests/full/cache_completion.json').exists())

    def test_changed_local_asset_symlink_or_original_inode_cannot_be_imported(self):
        contract=self.import_all();complete=validate_import_completion(self.root,contract)
        row=self.rows[0];entry=complete['receipts']['good'];path=self.root/row['latent_relative_path']
        before=path.read_bytes();path.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'differs from its receipt'):
            validate_import_receipt(self.root,row,contract,entry)
        path.unlink();path.symlink_to(self.origin/row['latent_relative_path'])
        with self.assertRaisesRegex(ValueError,'physical regular'):validate_import_receipt(self.root,row,contract,entry)
        path.unlink();os.link(self.origin/row['latent_relative_path'],path)
        with self.assertRaisesRegex(ValueError,'writable inodes'):validate_import_receipt(self.root,row,contract,entry)

    def test_loader_rejects_changed_import_scope_or_published_receipt_binding(self):
        self.import_all();self.publish()
        path=self.root/'manifests/full/inventory_meta.json';original=path.read_bytes()
        meta=json.loads(original);meta['imported_source']['contract_sha256']='0'*64;self.write(path,meta)
        with self.assertRaisesRegex(RuntimeError,'exact imported'):self.dataset()
        path.write_bytes(original)
        manifest=self.root/'manifests/full/train.jsonl';row=importer.read_rows(manifest)[0]
        row['import_receipt_sha256']='0'*64;importer.atomic_bytes(manifest,importer.payload(row))
        meta=json.loads(original);meta['manifests']['train.jsonl'].update(sha256=sha256(manifest),size_bytes=manifest.stat().st_size)
        self.write(path,meta)
        with self.assertRaisesRegex(ValueError,'immutable import receipt ledger'):self.dataset()


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
