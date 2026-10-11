"""The historical metric workers, with audited 660-pair GRID comparators."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from protocol import DEFAULT_RUN, DATA, PROJECT, PROTOCOL, add_repa, rows, sha, save_json


def validate(run, method, items):
    out=run/'outputs'/method
    summary=json.loads((out/'inference_summary.json').read_text())
    if summary['protocol']!=PROTOCOL or summary['count']!=660 or summary['partial_smoke_test'] or summary['pair_manifest_sha256']!=sha(run/'pairs.jsonl'):
        raise ValueError('Incomplete or different inference protocol')
    records={r['target_id']:r for r in summary['outputs']}
    actual={str(p.relative_to(out)) for p in (out/'test').rglob('*.wav')}
    if set(records)!={r['target_id'] for r in items} or actual!={r['target_id']+'.wav' for r in items}:
        raise ValueError('Output coverage differs from frozen targets')
    for row in items:
        record=records[row['target_id']]
        path=out/(row['target_id']+'.wav')
        wave,sr=sf.read(path,dtype='float32')
        if sr!=16000 or wave.shape!=(row['target_num_samples'],) or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError('Invalid generated waveform')
        if record['ref_id']!=row['ref_id'] or sha(path)!=record['sha256']:
            raise ValueError('Output identity/hash mismatch')
    return out,summary


def main(run, method, task, gt=False):
    add_repa()
    from aligndit.script.eval import eval_celebvdub_grid_reference as native
    items=rows(run,evaluation=True)
    if gt:
        if task!='wer':
            raise ValueError('GT diagnostic supports WER only')
        out=run/'assets/gt'
        inference={'checkpoint':None}
    else:
        out,inference=validate(run,method,items)
    summary_path=out/f'_{task}_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if summary['manifest_sha256']!=sha(run/'pairs.jsonl') or summary['count']!=660:
            raise ValueError('Existing metric belongs to different protocol')
        print('Already verified metric',task,summary[task],flush=True)
        return
    test_set=[]
    for row in items:
        reference=row['ref_audio'] if task=='sim' else row['target_gt_audio']
        test_set.append((str(out/(row['target_id']+'.wav')),reference,row['target_text']))
    args=argparse.Namespace(eval_task=task,gen_wav_dir=out,
          wavlm_ckpt=PROJECT.parent.parent/'alignDiT_pretrain_models/wavlm_large_finetune.pth',
          wavlm_base_ckpt=PROJECT.parent.parent/'alignDiT_pretrain_models/wavlm_large_s3prl.pt',
          asr_ckpt=DATA/'faster-whisper-large-v3',emo_ckpt=DATA/'emotion2vec_plus_large',gt_av_feat=run/'assets/gt_avhubert')
    if task=='avsync':
        native.validate_av_features(items,out,args.gt_av_feat)
    results=native.attach_identities(items,native.run_worker(args,test_set),task)
    summary=native.summarize(results,task)
    summary.update(protocol=PROTOCOL,task=task,method='GRID_GT' if gt else method,
       manifest_sha256=sha(run/'pairs.jsonl'),checkpoint=inference['checkpoint'],
       metric_reference={'sim':'generated versus paired CelebV-Dub reference',
                         'wer':'generated ASR versus GRID dialogue; native English normalization',
                         'emosim':'generated versus time-aligned GRID GT; emotion2vec classifier scores',
                         'emoembed':'generated versus time-aligned GRID GT; emotion2vec utterance embeddings',
                         'avsync':'joint AV-HuBERT cosine, generated versus GRID GT on the identical mouth video'}[task])
    (out/f'_{task}_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in results))
    save_json(summary_path,summary)
    print(f'{method} {task}: {summary[task]:.5f}, count=660',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--method',choices=['repa','baseline','styledubber','produbber'],default='repa')
    p.add_argument('--task',choices=['sim','wer','emosim','emoembed','avsync'],required=True)
    p.add_argument('--gt',action='store_true')
    a=p.parse_args()
    main(a.run,a.method,a.task,a.gt)
