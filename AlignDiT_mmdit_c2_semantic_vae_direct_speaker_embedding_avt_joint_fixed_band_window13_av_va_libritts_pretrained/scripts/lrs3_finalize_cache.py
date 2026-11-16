"""Audit each LRS3 cache and publish frozen, explicitly scoped train manifests."""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lrs3_build_inventory import clean_text, digest, write_json, write_jsonl

LATENT_SPEC = {'dimension':64,'dtype':'float32','frame_rate_hz':40.0,
               'hop_length_samples':400,'mode':'fixed_posterior_sample','sample_rate':16000}


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def validate(row, root, vocabulary, recovered, strict_provenance=False):
    r = dict(row)
    reasons = []
    if not r['text'] and r['sample_id'] in recovered:
        label = recovered[r['sample_id']]
        if label.get('original_clip_id') != r['original_clip_id'] or label.get('accepted') is not True:
            raise ValueError('Recovery label identity or acceptance differs')
        r.update(text=clean_text(label['text']),text_status='recovered_consensus',
                 label_source=label['source'],label_validation=label['quality'],
                 recovery_evidence=label)
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
    rows=read_rows(args.source_manifest)
    for key in ['sample_id','utterance_key','relative_stem']:
        if len({r[key] for r in rows})!=len(rows):raise ValueError(f'Duplicate source {key}')
    recovered={}
    for path in args.recovered_labels:
        for r in read_rows(path):
            if r.get('accepted') is True:
                previous=recovered.get(r['sample_id'])
                if previous and clean_text(previous['text'])!=clean_text(r['text']):
                    raise ValueError(f'Conflicting accepted recovery labels: {r["sample_id"]}')
                recovered[r['sample_id']]=r
    vocab_path=Path(prefix+'/zjw524/projects/data/CelebVDub_char/vocab.txt')
    vocabulary={c:i for i,c in enumerate(vocab_path.read_text().splitlines())}
    accepted=[]; quality=[]; reasons_count=collections.Counter(); originals={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=pool.map(lambda r:validate(r,args.root,vocabulary,recovered,args.mode=='full'),rows)
        for i,(r,reasons) in enumerate(results):
            oid=r['original_clip_id']
            original=originals.setdefault(oid,{'original_clip_id':oid,'split':r['split'],'source_pairs':0,
                       'ready_pairs':0,'source_audio_seconds':0.0,'ready_audio_seconds':0.0,
                       'cache_ready_pairs':0,'ctc_infeasible_pairs':0,
                       'text_status_counts':collections.Counter(),'crop_kind_counts':collections.Counter(),
                       'pending_reasons':collections.Counter(),'samples':[]})
            if original['split']!=r['split']:raise ValueError(f'Original clip crosses splits: {oid}')
            original['source_pairs']+=1;original['source_audio_seconds']+=r['audio_seconds']
            original['text_status_counts'][r['text_status']]+=1
            original['crop_kind_counts'][r['media_preprocessing']['crop_kind'] or 'unknown']+=1
            original['ctc_infeasible_pairs']+=not r['ctc_feasible_40hz']
            original['cache_ready_pairs']+=len(r['cache_sha256'])==4 and not any('provenance:' in x for x in reasons)
            original['samples'].append({'sample_id':r['sample_id'],'ready':not reasons,'reasons':reasons})
            if not reasons:
                accepted.append(r);original['ready_pairs']+=1;original['ready_audio_seconds']+=r['audio_seconds']
            else:
                reasons_count.update(reasons);original['pending_reasons'].update(reasons)
            quality.append({'sample_id':r['sample_id'],'original_clip_id':oid,'split':r['split'],
                            'ready':not reasons,'reasons':reasons,'text_status':r['text_status'],
                            'text':r['text'],'label_source':r['label_source'],'label_validation':r['label_validation'],
                            'audio_seconds':r['audio_seconds'],'video_seconds':r['video_seconds'],
                            'ctc_feasible_40hz':r['ctc_feasible_40hz'],'media_preprocessing':r['media_preprocessing'],
                            'cache_sha256':r['cache_sha256']})
            if (i+1)%1000==0:print(json.dumps({'audited':i+1,'total':len(rows),'accepted':len(accepted)}),flush=True)
    if not accepted:raise RuntimeError('No training-ready samples; no manifest published')
    entries={};split_counts={}
    for split in ['train','valid','test']:
        selected=[r for r in accepted if r['split']==split]
        path=manifest_dir/f'{split}.jsonl';write_jsonl(path,selected)
        feasible=sum(r['ctc_feasible_40hz'] for r in selected)
        split_counts[split]=len(selected)
        entries[path.name]={'count':len(selected),'sha256':digest(path),'size_bytes':path.stat().st_size,
                            'ctc_feasible_count':feasible,'ctc_infeasible_count':len(selected)-feasible}
    inv=manifest_dir/'inventory.jsonl';write_jsonl(inv,accepted)
    entries[inv.name]={'count':len(accepted),'sha256':digest(inv),'size_bytes':inv.stat().st_size}
    meta={'schema_version':1,'dataset':'LRS3','status':'complete','selection':{'mode':args.mode},
          'manifests':entries,'latent_spec':LATENT_SPEC,'source_manifest_sha256':digest(args.source_manifest),
          'source_pairs':len(rows),'training_ready_pairs':len(accepted),'excluded_pairs':len(rows)-len(accepted),
          'all_requested_source_pairs_ready':len(accepted)==len(rows),'vocab_sha256':digest(vocab_path)}
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
             'inventory_sha256':digest(inv),'count':len(accepted),'features':{
                 'latents':{'feature':'semantic_vae_posterior_sample_v1','count':len(accepted),'total_frames':total_frames},
                 'video_40hz':{'feature':'avhubert_video_25hz_to_40hz_linear_align_corners_false_v1','count':len(accepted),'total_frames':total_frames},
                 'speaker_embeddings':{'count':len(accepted),'dimension':192}},
             'speaker_metadata_dir':str(speaker_dir),
             'speaker_metadata_sha256':digest(speaker_dir/'metadata.json'),
             'speaker_coverage_sha256':digest(speaker_dir/'coverage_report.json')}
    write_jsonl(args.root/f'reports/{args.name}_sample_quality.jsonl',quality)
    for o in originals.values():
        o['fully_covered']=o['source_pairs']==o['ready_pairs']
        o['pending_reasons']=dict(o['pending_reasons'])
        o['text_status_counts']=dict(o['text_status_counts'])
        o['crop_kind_counts']=dict(o['crop_kind_counts'])
        o['ready_duration_fraction']=o['ready_audio_seconds']/o['source_audio_seconds'] if o['source_audio_seconds'] else 0.0
    write_jsonl(args.root/f'reports/{args.name}_original_coverage.jsonl',originals.values())
    summary={'selection':args.mode,'source_pairs':len(rows),'ready_pairs':len(accepted),
             'all_requested_source_pairs_ready':len(accepted)==len(rows),'split_counts':split_counts,
             'audio_hours_by_split':{s:sum(r['audio_seconds'] for r in accepted if r['split']==s)/3600 for s in split_counts},
             'text_sources':dict(collections.Counter(r['text_status'] for r in accepted)),
             'ctc_infeasible_ready_pairs':sum(not r['ctc_feasible_40hz'] for r in accepted),
             'source_audio_hours':sum(r['audio_seconds'] for r in rows)/3600,
             'source_video_hours':sum(r['video_seconds'] for r in rows)/3600,
             'excluded_audio_hours':sum(q['audio_seconds'] for q in quality if not q['ready'])/3600,
             'pending_reasons':dict(reasons_count),'original_ids':len(originals),
             'fully_covered_original_ids':sum(o['fully_covered'] for o in originals.values()),
             'manifest_dir':str(manifest_dir),'source_manifest_sha256':digest(args.source_manifest),
             'limitations':['Text consensus is automatic evidence, not human verification.',
                            'Duration and feature-length agreement alone do not prove lip/audio synchronization.',
                            'Completion receipt covers only the published accepted inventory; exclusions remain explicit.']}
    write_json(args.root/f'reports/{args.name}_readiness.json',summary)
    write_json(manifest_dir/'cache_completion.json',receipt) # Publication completion marker is last.
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
