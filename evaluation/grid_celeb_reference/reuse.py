"""Reuse only hash-verified input artifacts after input-only selection repair."""
import argparse
import json
from pathlib import Path
import shutil

from protocol import DEFAULT_RUN, DATA, rows, sha, save_json


def main(source, dest):
    previous = {r['target_id']: r for r in rows(source)}
    current = rows(dest)
    old_refs = {r['ref_id']:r for r in previous.values()}
    new_refs = {r['ref_id']:r for r in current}
    if set(old_refs) != set(new_refs):
        raise ValueError('Reference selection changed')
    meta = json.loads((source/'references/features.json').read_text())
    for record in meta['records']:
        old, new = old_refs[record['ref_id']], new_refs[record['ref_id']]
        if old['ref_audio_sha256'] != new['ref_audio_sha256']:
            raise ValueError('Canonical reference changed')
        for key, hashkey in [('ref_latent_path','latent_sha256'),('ref_speaker_path','speaker_sha256')]:
            if sha(old[key]) != record[hashkey]:
                raise ValueError('Changed reference cache')
            path = Path(new[key])
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            shutil.copy2(old[key],path)
    save_json(dest/'references/features.json',meta)
    copied = 0
    for row in current:
        old = previous.get(row['target_id'])
        record_path = source/'assets/records'/(row['target_id']+'.json')
        if old is None or not record_path.exists():
            continue
        if old['source_video_sha256'] != row['source_video_sha256'] or old['native_frames'] != row['native_frames']:
            raise ValueError('Target input changed')
        record = json.loads(record_path.read_text())
        paths = {k:row[k] for k in ('mouth_video','lip_feature','face_feature','video_25hz_path','target_video_path')}
        paths['gt_audio']=str(dest/'assets/gt'/(row['target_id']+'.wav'))
        for key, expected in record['artifact_sha256'].items():
            if sha(record['artifact_paths'][key]) != expected:
                raise ValueError('Unverified visual artifact')
            path = Path(paths[key])
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            shutil.copy2(record['artifact_paths'][key],path)
        record['artifact_paths'] = paths
        record['reused_verified_record_from'] = str(record_path)
        save_json(dest/'assets/records'/(row['target_id']+'.json'),record)
        copied += 1
    save_json(dest/'selection_repair.json',{'source_manifest_sha256':sha(source/'inference.jsonl'),
              'new_manifest_sha256':sha(dest/'inference.jsonl'), 'references_unchanged':True,
              'no_formal_model_outputs_reused':True, 'verified_visual_records_reused':copied,
              'removed_target_ids':sorted(set(previous)-{r['target_id'] for r in current}),
              'added_target_ids':sorted({r['target_id'] for r in current}-set(previous)),
              'reason':'input-only exclusion of incomplete raw GRID videos before formal evaluation'})
    print(f'Reused 66 identical references and {copied} verified input-only visual records',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=DATA/'evaluations/grid660_celeb66_seed0_v1')
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    a=p.parse_args()
    main(a.source,a.run)
