"""Audit each LRS3 cache and publish frozen, explicitly scoped train manifests."""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lrs3_build_inventory import clean_text, digest, write_json, write_jsonl
from aligndit.lrs3_media_quality import evidence_paths, excluded_map, load_terminal, scope as media_quality_scope

LATENT_SPEC = {'dimension':64,'dtype':'float32','frame_rate_hz':40.0,
               'hop_length_samples':400,'mode':'fixed_posterior_sample','sample_rate':16000}

SOURCE_KINDS = {'existing_transcript', 'source_asr_consensus', 'source_asr_mfa_consensus',
                'source_asr_ctc_consensus', 'asr_pseudo'}
PSEUDO_THRESHOLDS = {
    'min_word_probability': ('minimum', 0.50),
    'mean_word_probability': ('minimum', 0.80),
    'min_segment_avg_logprob': ('minimum', -0.75),
    'max_no_speech_probability': ('maximum', 0.30),
    'max_compression_ratio': ('maximum', 2.4),
    'ctc_min_char_probability': ('minimum', 0.80),
    'ctc_mean_char_probability': ('minimum', 0.97),
}


def check_pseudo_label(label, row, text_policy):
    if text_policy != 'source_plus_asr_pseudo' or row['split'] != 'train':
        raise ValueError('ASR pseudo labels require the explicit training-only extension policy')
    if (label.get('independent_content_verified') is not False or
        label.get('acceptance_policy_version') != 'lrs3_asr_pseudo_p1_v1'):
        raise ValueError('Pseudo label has missing or incompatible policy provenance')
    confidence = label.get('confidence', {})
    for key, (bound, threshold) in PSEUDO_THRESHOLDS.items():
        value = confidence.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or nonfinite pseudo confidence: {key}')
        if (bound == 'minimum' and value < threshold) or (bound == 'maximum' and value > threshold):
            raise ValueError(f'Pseudo confidence below acceptance policy: {key}')
    gates = label.get('quality_gates', {})
    required = ['ctc_text_exact', 'group_contiguous_support', 'word_text_exact',
                'positive_word_durations', 'within_pcm_bounds', 'group_boundary_consistent']
    if not all(gates.get(key) is True for key in required):
        raise ValueError('Pseudo label lacks required independent content and boundary checks')
    if len(clean_text(label['text']).split()) < 3 or row['audio_seconds'] < 1.0:
        raise ValueError('Pseudo label is too short for this acceptance policy')
    refs = label.get('evidence_refs', [])
    if not refs or any(not ref.get('path') or len(ref.get('sha256', '')) != 64 for ref in refs):
        raise ValueError('Pseudo label lacks evidence references')


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def read_rows_snapshot(path):
    payload = Path(path).read_bytes()
    return [json.loads(line) for line in payload.splitlines() if line.strip()], hashlib.sha256(payload).hexdigest()


def frozen_recovery(path, source_hash, missing_pairs, missing_groups):
    labels, fingerprint = read_rows_snapshot(path)
    frozen_path = path.parent / 'frozen_export.json'
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    summary = frozen.get('summary', {})
    if (summary.get('processing_complete') is not True or summary.get('pending_segments') != 0 or
        summary.get('groups_requested') != missing_groups or
        summary.get('groups_complete') != missing_groups or
        summary.get('segments_requested') != missing_pairs or
        frozen.get('source_manifest_sha256') != source_hash or
        frozen.get('files', {}).get(path.name) != fingerprint):
        raise ValueError('Recovery labels are not a complete frozen export for this source scope')
    accounted = sum(summary.get(key, 0) for key in
                    ['strict_segments','pseudo_p1_segments','pseudo_p2_candidate_segments','excluded_segments'])
    if accounted != missing_pairs:
        raise ValueError('Frozen recovery export does not account for every requested segment')
    return labels, fingerprint, frozen_path, hashlib.sha256(frozen_bytes).hexdigest()


def text_policy_contract(root, policy, baseline):
    result = {'schema_version': 1, 'policy': policy, 'version': 'lrs3_text_policy_v1',
              'pseudo_scope': 'none' if policy == 'source_grounded' else 'train_only',
              'strict_baseline': None,
              'pseudo_acceptance_policy': 'lrs3_asr_pseudo_p1_v1',
              'pseudo_confidence_thresholds': PSEUDO_THRESHOLDS,
              'confidence_note': 'Model scores are evidence; they are not calibrated text correctness probabilities.'}
    if policy == 'source_grounded':
        if baseline is not None:raise ValueError('A strict baseline is only used for a pseudo extension')
        return result, None
    if baseline is None:raise ValueError('The pseudo extension requires a frozen strict baseline')
    path = root / baseline
    if path.resolve() == (root / 'manifests/full_pseudo').resolve():
        raise ValueError('A pseudo extension cannot be its own strict baseline')
    meta = json.loads((path / 'inventory_meta.json').read_text())
    receipt = json.loads((path / 'cache_completion.json').read_text())
    if (meta.get('text_policy') != 'source_grounded' or receipt.get('status') != 'complete' or
        meta.get('selection', {}).get('mode') != 'full'):
        raise ValueError('The extension baseline must be a completed strict full publication')
    for filename, entry in meta['manifests'].items():
        if digest(path / filename) != entry['sha256']:
            raise ValueError('Strict baseline manifest hash changed')
    policy_sha = digest(path / 'text_policy.json')
    if policy_sha != meta.get('text_policy_sha256') or policy_sha != receipt.get('text_policy_sha256'):
        raise ValueError('Strict baseline policy hash changed')
    if digest(path / 'inventory.jsonl') != receipt.get('inventory_sha256'):
        raise ValueError('Strict baseline completion inventory hash changed')
    result['strict_baseline'] = {
        'manifest_dir': str(path.resolve().relative_to(root.resolve())),
        'inventory_meta_sha256': digest(path / 'inventory_meta.json'),
        'inventory_sha256': digest(path / 'inventory.jsonl'),
        'train_sha256': digest(path / 'train.jsonl'),
        'valid_sha256': digest(path / 'valid.jsonl'),
        'test_sha256': digest(path / 'test.jsonl'),
        'text_policy_sha256': policy_sha,
    }
    return result, path


def validate(row, root, vocabulary, recovered, strict_provenance=False, text_policy='source_grounded',
             media_quality_exclusions=None):
    r = dict(row)
    r['source_kind'] = 'existing_transcript' if r['text'] else 'unresolved'
    reasons = []
    excluded = (media_quality_exclusions or {}).get(r['sample_id'])
    if excluded is not None:
        if any(excluded[key] != r[key] for key in ['relative_stem','original_clip_id']):
            raise ValueError('Media quality exclusion identity differs from the canonical row')
        reasons.append('media_quality:' + excluded['reason'])
        r['media_quality_exclusion'] = excluded
    if not r['text'] and r['sample_id'] in recovered:
        label = recovered[r['sample_id']]
        if label.get('original_clip_id') != r['original_clip_id'] or label.get('accepted') is not True:
            raise ValueError('Recovery label identity or acceptance differs')
        kind = label.get('source_kind', 'source_asr_consensus')
        if kind not in SOURCE_KINDS - {'existing_transcript'}:
            raise ValueError('Unknown recovery source kind')
        if kind == 'asr_pseudo':
            check_pseudo_label(label, r, text_policy)
        if strict_provenance and label.get('source_audio_sha256') != digest(r['audio_path']):
            raise ValueError('Recovery label does not bind the actual source audio')
        r.update(text=clean_text(label['text']),
                 text_status='asr_pseudo' if kind == 'asr_pseudo' else 'recovered_consensus',
                 source_kind=kind, independent_content_verified=False,
                 label_source=label['source'],label_validation=label['quality'],
                 recovery_evidence=label)
        for key in ['acceptance_policy_version', 'confidence', 'evidence_refs']:
            if key in label:r[key]=label[key]
    if not r['text']:reasons.append('text_unresolved')
    r['oov_characters'] = sorted(set(r['text']) - set(vocabulary))
    if r['oov_characters']:reasons.append('text_oov')
    if not r['duration_eligible']:reasons.append('duration_outside_0.4_30_seconds')
    if abs(r['audio_seconds'] - r['video_seconds']) > 0.10:reasons.append('audio_video_duration_delta')
    ids=[vocabulary.get(c,0) for c in r['text']]
    repeats=sum(a==b for a,b in zip(ids,ids[1:]))
    r.update(ctc_target_length=len(ids),ctc_adjacent_repeats=repeats,
             ctc_min_input_frames=len(ids)+repeats,
             ctc_feasible_40hz=bool(r['text']) and r['latent_frames']>=len(ids)+repeats)
    hashes={}
    for name,shape in [('video_25hz',(r['video_frames'],1024)),('latent',(r['latent_frames'],64)),
                       ('video_40hz',(r['latent_frames'],1024)),('speaker',(192,))]:
        path=root/r[f'{name}_relative_path']
        try:
            if path.is_symlink() or not path.is_file():raise ValueError('missing_or_symlink')
            a=np.load(path,allow_pickle=False)
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('shape_dtype_or_finite')
            if name=='speaker' and abs(float(np.linalg.norm(a))-1)>1e-4:raise ValueError('not_unit_norm')
            hashes[name]=digest(path)
        except (OSError,ValueError) as error:
            reasons.append(f'{name}:{error}')
    mouth=root/r['mouth_relative_path']
    if not mouth.is_file():reasons.append('mouth_missing')
    media={}
    try:
        media=json.loads((root/'reports/media_receipts'/f'{r["relative_stem"]}.json').read_text())
        if strict_provenance:
            if (media['source_path']!=r['video_path'] or
                media['source_sha256']!=digest(r['video_path']) or
                media['feature_sha256']!=hashes.get('video_25hz') or
                media['protocol_sha256']!='29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f' or
                media.get('features_complete') is not True or media['video_frames']!=r['video_frames']):
                raise ValueError('source_or_protocol_mismatch')
    except (OSError,ValueError,KeyError) as error:
        if strict_provenance:reasons.append(f'media_provenance:{error}')
    if strict_provenance:
        try:
            audio=root/r['audio_relative_path']
            if not audio.is_file() or audio.resolve()!=Path(r['audio_path']).resolve():
                raise ValueError('audio_link_mismatch')
            receipt=json.loads((root/'state/audio_receipts'/f'{r["relative_stem"]}.json').read_text())
            expected={'audio_sha256':digest(audio),'utterance_key':r['utterance_key'],
                      'posterior_seed':r['posterior_seed'],
                      'posterior_protocol':'semantic_vae_posterior_sample_v1',
                      'codec_sha256':'7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516',
                      'campplus_sha256':'92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199',
                      'speaker_protocol':'full_waveform_10s_repeat_fbank_mean_average_l2_v1'}
            if receipt['identity']!=expected or receipt['outputs']!={k:hashes.get(k) for k in ['latent','speaker']}:
                raise ValueError('source_or_protocol_mismatch')
            r['source_audio_sha256']=expected['audio_sha256']
        except (OSError,ValueError,KeyError) as error:reasons.append(f'audio_provenance:{error}')
        try:
            receipt=json.loads((root/'state/video40_receipts'/f'{r["relative_stem"]}.json').read_text())
            expected={'source_sha256':hashes.get('video_25hz'),
                      'protocol':'avhubert_video_25hz_to_40hz_linear_align_corners_false_v1',
                      'utterance_key':r['utterance_key'],'target_frames':r['latent_frames']}
            if receipt['identity']!=expected or receipt['outputs']!={'video_40hz':hashes.get('video_40hz')}:
                raise ValueError('source_or_protocol_mismatch')
        except (OSError,ValueError,KeyError) as error:reasons.append(f'video40_provenance:{error}')
    r['media_preprocessing']={'crop_kind':media.get('crop_kind'),
                              'protocol_sha256':media.get('protocol_sha256'),
                              'source_sha256':media.get('source_sha256')}
    r['cache_sha256']=hashes
    return r,reasons


def main():
    prefix=os.environ.get('ROOT_PREFIX','')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(prefix+'/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1'))
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--recovered-labels',type=Path,action='append',default=[])
    p.add_argument('--name',required=True,help='Unique publication directory under manifests/')
    p.add_argument('--mode',choices=['pilot','full'],required=True)
    p.add_argument('--text-policy',choices=['source_grounded','source_plus_asr_pseudo'],default='source_grounded')
    p.add_argument('--strict-baseline',type=Path,help='Frozen strict manifest directory, relative to cache root')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if '/' in args.name or args.name in {'.','..'}:raise ValueError('Invalid publication name')
    manifest_dir=args.root/'manifests'/args.name
    locks=[]
    for stage in ['audio','video40','media','publication']:
        path=args.root/f'state/locks/{stage}.lock';path.parent.mkdir(parents=True,exist_ok=True)
        lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    for feature in ['latents','video_25hz','video_40hz','speaker_embeddings']:
        if (args.root/f'state/{feature}/WRITE_ACTIVE.json').exists():
            raise RuntimeError(f'Cache writes are unfinished: {feature}')
    if (manifest_dir/'cache_completion.json').exists():raise FileExistsError('Publication is immutable; choose another name')
    policy, baseline_path = text_policy_contract(args.root,args.text_policy,args.strict_baseline)
    if baseline_path is not None and baseline_path.resolve()==manifest_dir.resolve():
        raise ValueError('An extension cannot overwrite its strict baseline')
    rows,source_hash=read_rows_snapshot(args.source_manifest)
    input_pins={str(args.source_manifest):source_hash}
    source_contract_hash=None
    media_quality=None
    if args.mode=='full':
        contract_path=args.root/'reports/source_contract.json'
        contract_bytes=contract_path.read_bytes()
        contract=json.loads(contract_bytes)
        source_contract_hash=hashlib.sha256(contract_bytes).hexdigest()
        canonical=args.root/'manifests/source_inventory.jsonl'
        if (source_hash!=contract.get('source_inventory_sha256') or digest(canonical)!=source_hash or
            len(rows)!=contract['source_media_pairs'] or
            len({r['original_clip_id'] for r in rows})!=contract['original_clip_count']):
            raise ValueError('Full source inventory differs from the canonical source contract')
        input_pins.update({str(contract_path):source_contract_hash,str(canonical):source_hash})
        media_quality=load_terminal(args.root,source_sha256=source_hash)
        for path in evidence_paths(media_quality):
            input_pins[str(path)]=digest(path)
        if any(not r['text'] for r in rows) and not args.recovered_labels:
            raise ValueError('Full publication requires completed text recovery accounting for missing transcripts')
    for key in ['sample_id','utterance_key','relative_stem']:
        if len({r[key] for r in rows})!=len(rows):raise ValueError(f'Duplicate source {key}')
    source_videos={s:{r['original_clip_id'].split('/')[1] for r in rows if r['split']==s}
                   for s in ['train','valid','test']}
    if source_videos['test'] & (source_videos['train'] | source_videos['valid']):
        raise ValueError('A test source video also appears in training or validation')
    recovered={}
    checked_evidence=set()
    known_source_ids={r['sample_id'] for r in rows}
    media_exclusions=excluded_map(media_quality)
    if not media_exclusions.keys() <= known_source_ids:
        raise ValueError('Media quality exclusions contain noncanonical source IDs')
    quality_scope=media_quality_scope(media_quality)
    if baseline_path is not None:
        baseline_meta=json.loads((baseline_path/'inventory_meta.json').read_text())
        if baseline_meta.get('media_quality') != quality_scope:
            raise ValueError('Pseudo extension and strict baseline have different media quality closures')
    missing_pairs=sum(not r['text'] for r in rows)
    missing_groups=len({r['original_clip_id'] for r in rows if not r['text']})
    label_pins=[]
    for path in args.recovered_labels:
        if args.mode=='full':
            labels,fingerprint,frozen_path,frozen_hash=frozen_recovery(path,source_hash,missing_pairs,missing_groups)
            input_pins[str(frozen_path)]=frozen_hash
        else:
            labels,fingerprint=read_rows_snapshot(path)
        input_pins[str(path)]=fingerprint
        label_pins.append({'path':str(path),'sha256':fingerprint})
        for r in labels:
            if r.get('accepted') is True:
                if r['sample_id'] not in known_source_ids:raise ValueError('Recovery label outside requested source scope')
                if r.get('source_kind')=='asr_pseudo' and args.text_policy!='source_plus_asr_pseudo':
                    raise ValueError('Strict publication cannot ingest ASR pseudo labels')
                for ref in r.get('evidence_refs',[]):
                    key=(ref['path'],ref['sha256'])
                    if key not in checked_evidence:
                        if digest(ref['path'])!=ref['sha256']:raise ValueError('Recovery evidence hash changed')
                        checked_evidence.add(key)
                previous=recovered.get(r['sample_id'])
                if previous and clean_text(previous['text'])!=clean_text(r['text']):
                    raise ValueError(f'Conflicting accepted recovery labels: {r["sample_id"]}')
                recovered[r['sample_id']]=r
    policy['recovered_label_files']=label_pins
    vocab_path=Path(prefix+'/zjw524/projects/data/CelebVDub_char/vocab.txt')
    vocabulary={c:i for i,c in enumerate(vocab_path.read_text().splitlines())}
    accepted=[]; quality=[]; reasons_count=collections.Counter(); originals={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=pool.map(lambda r:validate(r,args.root,vocabulary,recovered,args.mode=='full',args.text_policy,media_exclusions),rows)
        for i,(r,reasons) in enumerate(results):
            oid=r['original_clip_id']
            original=originals.setdefault(oid,{'original_clip_id':oid,'split':r['split'],'source_pairs':0,
                       'ready_pairs':0,'source_audio_seconds':0.0,'ready_audio_seconds':0.0,
                       'cache_ready_pairs':0,'ctc_infeasible_pairs':0,
                       'text_status_counts':collections.Counter(),'source_kind_counts':collections.Counter(),
                       'crop_kind_counts':collections.Counter(),
                       'pending_reasons':collections.Counter(),'samples':[]})
            if original['split']!=r['split']:raise ValueError(f'Original clip crosses splits: {oid}')
            original['source_pairs']+=1;original['source_audio_seconds']+=r['audio_seconds']
            original['text_status_counts'][r['text_status']]+=1
            original['source_kind_counts'][r['source_kind']]+=1
            original['crop_kind_counts'][r['media_preprocessing']['crop_kind'] or 'unknown']+=1
            original['ctc_infeasible_pairs']+=not r['ctc_feasible_40hz']
            original['cache_ready_pairs']+=len(r['cache_sha256'])==4 and not any(
                'provenance:' in x or x.startswith('media_quality:') for x in reasons)
            original['samples'].append({'sample_id':r['sample_id'],'ready':not reasons,'reasons':reasons})
            if not reasons:
                accepted.append(r);original['ready_pairs']+=1;original['ready_audio_seconds']+=r['audio_seconds']
            else:
                reasons_count.update(reasons);original['pending_reasons'].update(reasons)
            quality.append({'sample_id':r['sample_id'],'original_clip_id':oid,'split':r['split'],
                            'ready':not reasons,'reasons':reasons,'text_status':r['text_status'],
                            'source_kind':r['source_kind'],
                            'text':r['text'],'label_source':r['label_source'],'label_validation':r['label_validation'],
                            'independent_content_verified':r.get('independent_content_verified',False),
                            'acceptance_policy_version':r.get('acceptance_policy_version'),
                            'confidence':r.get('confidence'), 'evidence_refs':r.get('evidence_refs',[]),
                            'audio_seconds':r['audio_seconds'],'video_seconds':r['video_seconds'],
                            'ctc_feasible_40hz':r['ctc_feasible_40hz'],'media_preprocessing':r['media_preprocessing'],
                            'cache_sha256':r['cache_sha256']})
            if 'media_quality_exclusion' in r:
                quality[-1]['media_quality_exclusion']=r['media_quality_exclusion']
            if (i+1)%1000==0:print(json.dumps({'audited':i+1,'total':len(rows),'accepted':len(accepted)}),flush=True)
    if not accepted:raise RuntimeError('No training-ready samples; no manifest published')
    if baseline_path is not None:
        accepted_by_id={r['sample_id']:r for r in accepted}
        baseline_ids=set()
        for previous in read_rows(baseline_path/'inventory.jsonl'):
            baseline_ids.add(previous['sample_id'])
            if accepted_by_id.get(previous['sample_id'])!=previous:
                raise ValueError('Pseudo extension changed or removed a strict baseline record')
        if any(r['source_kind']!='asr_pseudo' or r['split']!='train'
               for sid,r in accepted_by_id.items() if sid not in baseline_ids):
            raise ValueError('A pseudo extension may only add explicitly marked training pseudo labels')
    for path,fingerprint in input_pins.items():
        if digest(path)!=fingerprint:raise ValueError('A publication input changed during the cache audit')
    entries={};split_counts={}
    for split in ['train','valid','test']:
        selected=[r for r in accepted if r['split']==split]
        path=manifest_dir/f'{split}.jsonl';write_jsonl(path,selected)
        feasible=sum(r['ctc_feasible_40hz'] for r in selected)
        split_counts[split]=len(selected)
        entries[path.name]={'count':len(selected),'sha256':digest(path),'size_bytes':path.stat().st_size,
                            'ctc_feasible_count':feasible,'ctc_infeasible_count':len(selected)-feasible}
        if baseline_path is not None and split in {'valid','test'}:
            if digest(path)!=digest(baseline_path/path.name):
                raise ValueError('Pseudo extension changed a validation or test manifest')
    inv=manifest_dir/'inventory.jsonl';write_jsonl(inv,accepted)
    entries[inv.name]={'count':len(accepted),'sha256':digest(inv),'size_bytes':inv.stat().st_size}
    write_json(manifest_dir/'text_policy.json',policy)
    policy_sha=digest(manifest_dir/'text_policy.json')
    source_kind_counts={s:dict(collections.Counter(r['source_kind'] for r in accepted if r['split']==s))
                        for s in ['train','valid','test']}
    meta={'schema_version':1,'dataset':'LRS3','status':'complete','selection':{'mode':args.mode},
          'text_policy':args.text_policy,'text_policy_sha256':policy_sha,'source_kind_counts':source_kind_counts,
          'manifests':entries,'latent_spec':LATENT_SPEC,'source_manifest_sha256':source_hash,
          'source_contract_sha256':source_contract_hash,'source_original_ids':len(originals),
          'source_pairs':len(rows),'training_ready_pairs':len(accepted),'excluded_pairs':len(rows)-len(accepted),
          'all_requested_source_pairs_ready':len(accepted)==len(rows),'vocab_sha256':digest(vocab_path)}
    if quality_scope is not None:meta['media_quality']=quality_scope
    write_json(manifest_dir/'inventory_meta.json',meta)
    speaker_dir=manifest_dir/'speaker_contract'
    speaker={'schema_version':1,'status':'complete','model_id':'iic/speech_campplus_sv_zh_en_16k-common_advanced',
             'checkpoint_sha256':'92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199',
             'sample_rate':16000,'fbank_dim':80,'fbank_mean_normalization':'per_chunk_per_mel_over_time',
             'chunk_seconds':10,'max_seconds':90,'padding':'repeat_complete_utterance_to_chunk_multiple',
             'chunk_aggregation':'arithmetic_mean_then_l2_normalize','output_shape':[192],'output_dtype':'float32',
             'source_audio':'complete_unmasked_waveform','expected_count':len(accepted),'coverage_report':'coverage_report.json'}
    coverage={'complete':True,'expected':len(accepted),'valid':len(accepted),'missing':[],'invalid':[],'extra':[],
              'split_counts':split_counts,'scope':'exact published manifest; cache root may contain additional pending samples'}
    write_json(speaker_dir/'metadata.json',speaker);write_json(speaker_dir/'coverage_report.json',coverage)
    total_frames=sum(r['latent_frames'] for r in accepted)
    receipt={'schema_version':1,'status':'complete','selection':{'mode':args.mode},
             'source_contract_sha256':source_contract_hash,'source_original_ids':len(originals),
             'text_policy':args.text_policy,'text_policy_sha256':policy_sha,
             'inventory_sha256':digest(inv),'count':len(accepted),'features':{
                 'latents':{'feature':'semantic_vae_posterior_sample_v1','count':len(accepted),'total_frames':total_frames},
                 'video_40hz':{'feature':'avhubert_video_25hz_to_40hz_linear_align_corners_false_v1','count':len(accepted),'total_frames':total_frames},
                 'speaker_embeddings':{'count':len(accepted),'dimension':192}},
             'speaker_metadata_dir':str(speaker_dir),
             'speaker_metadata_sha256':digest(speaker_dir/'metadata.json'),
             'speaker_coverage_sha256':digest(speaker_dir/'coverage_report.json')}
    if quality_scope is not None:receipt['media_quality']=quality_scope
    write_jsonl(args.root/f'reports/{args.name}_sample_quality.jsonl',quality)
    for o in originals.values():
        o['fully_covered']=o['source_pairs']==o['ready_pairs']
        o['pending_reasons']=dict(o['pending_reasons'])
        o['text_status_counts']=dict(o['text_status_counts'])
        o['source_kind_counts']=dict(o['source_kind_counts'])
        o['crop_kind_counts']=dict(o['crop_kind_counts'])
        o['ready_duration_fraction']=o['ready_audio_seconds']/o['source_audio_seconds'] if o['source_audio_seconds'] else 0.0
    write_jsonl(args.root/f'reports/{args.name}_original_coverage.jsonl',originals.values())
    coverage_tsv=args.root/f'reports/{args.name}_original_coverage.tsv'
    with coverage_tsv.with_suffix('.tsv.tmp').open('w') as handle:
        fields=['original_clip_id','split','source_pairs','ready_pairs','cache_ready_pairs',
                'source_audio_seconds','ready_audio_seconds','ready_duration_fraction',
                'fully_covered','ctc_infeasible_pairs','text_status_counts','source_kind_counts',
                'crop_kind_counts','pending_reasons']
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
        for original in originals.values():
            writer.writerow({key:json.dumps(original[key],ensure_ascii=False) if isinstance(original[key],dict)
                             else original[key] for key in fields})
    coverage_tsv.with_suffix('.tsv.tmp').replace(coverage_tsv)
    summary={'selection':args.mode,'source_pairs':len(rows),'ready_pairs':len(accepted),
             'text_policy':args.text_policy,'text_policy_sha256':policy_sha,'source_kind_counts':source_kind_counts,
             'all_requested_source_pairs_ready':len(accepted)==len(rows),'split_counts':split_counts,
             'audio_hours_by_split':{s:sum(r['audio_seconds'] for r in accepted if r['split']==s)/3600 for s in split_counts},
             'text_sources':dict(collections.Counter(r['text_status'] for r in accepted)),
             'ctc_infeasible_ready_pairs':sum(not r['ctc_feasible_40hz'] for r in accepted),
             'source_audio_hours':sum(r['audio_seconds'] for r in rows)/3600,
             'source_video_hours':sum(r['video_seconds'] for r in rows)/3600,
             'excluded_audio_hours':sum(q['audio_seconds'] for q in quality if not q['ready'])/3600,
             'pending_reasons':dict(reasons_count),'original_ids':len(originals),
             'fully_covered_original_ids':sum(o['fully_covered'] for o in originals.values()),
             'manifest_dir':str(manifest_dir),'source_manifest_sha256':source_hash,
             'source_contract_sha256':source_contract_hash,'publication_input_pins':input_pins,
             'source_videos_by_split':{s:len(v) for s,v in source_videos.items()},
             'test_source_video_overlap_count':0,
             'train_valid_source_video_overlap_count':len(source_videos['train']&source_videos['valid']),
             'limitations':['Text consensus is automatic evidence, not human verification.',
                            'Duration and feature-length agreement alone do not prove lip/audio synchronization.',
                            'Completion receipt covers only the published accepted inventory; exclusions remain explicit.']}
    if quality_scope is not None:summary['media_quality']=quality_scope
    write_json(args.root/f'reports/{args.name}_readiness.json',summary)
    for path,fingerprint in input_pins.items():
        if digest(path)!=fingerprint:raise ValueError('A publication input changed before completion')
    write_json(manifest_dir/'cache_completion.json',receipt) # Publication completion marker is last.
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
