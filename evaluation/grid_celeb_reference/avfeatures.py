"""Joint AV-HuBERT features; target GT is read only in this scoring stage."""
import argparse
from pathlib import Path
import sys
import numpy as np

from protocol import DEFAULT_RUN, DATA, PROJECT, add_repa, rows, sha, save_json


def main(run, method, limit=0):
    add_repa()
    sys.path.insert(0,str(DATA/'av_hubert/fairseq/fairseq'))
    sys.path.insert(0,str(DATA/'av_hubert/avhubert/avhubert'))
    import fairseq
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=str(DATA/'av_hubert/avhubert/avhubert')))
    from aligndit.script.misc.extract_avhubert import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    reader=AVHubertFeatureReader(str(PROJECT.parent.parent/'alignDiT_pretrain_models/large_vox_iter5.pt'),custom_utils=avhubert_utils)
    label=method+(f'_smoke{limit}' if limit else '')
    root=run/'assets/gt_avhubert' if method=='gt' else run/'outputs'/label/'avhubert_feat'
    records=[]
    items=rows(run,evaluation=True)
    if limit:
        items=items[:limit]
    for i,row in enumerate(items,1):
        dest=root/(row['target_id']+'.npy')
        audio=Path(row['target_gt_audio']) if method=='gt' else run/'outputs'/label/(row['target_id']+'.wav')
        record_path=dest.with_suffix('.json')
        if dest.exists() and record_path.exists():
            import json
            previous=json.loads(record_path.read_text())
            if previous['audio_sha256']!=sha(audio) or previous['mouth_sha256']!=sha(row['mouth_video']) or previous['feature_sha256']!=sha(dest):
                raise ValueError('Changed resumed AV feature')
        else:
            feat=reader.get_feats((row['mouth_video'],str(audio))).cpu().float().numpy()
            if feat.shape!=(row['native_frames'],1024) or not np.isfinite(feat).all():
                raise ValueError('Invalid joint AV feature: '+row['target_id'])
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.with_suffix('.tmp').open('wb') as f:
                np.save(f,feat,allow_pickle=False)
            dest.with_suffix('.tmp').replace(dest)
            previous={'target_id':row['target_id'],'audio_sha256':sha(audio),'mouth_sha256':sha(row['mouth_video']),'feature_sha256':sha(dest)}
            save_json(record_path,previous)
        records.append(previous)
        print(f'AV features {method} {i}/{len(items)}: {row["target_id"]}',flush=True)
    save_json(root/'summary.json',{'count':len(records),'partial_smoke_test':bool(limit),'records':records})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--method',choices=['gt','repa','baseline','styledubber','produbber'],required=True)
    p.add_argument('--limit',type=int,choices=[0,1,33],default=0)
    a=p.parse_args()
    main(a.run,a.method,a.limit)
