"""Freeze a 213-clip training-membership diagnostic set before model evaluation."""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def runs(text):
    tokens = [m.group().replace('’', "'").casefold() for m in TOKEN_RE.finditer(text)]
    result = []
    i = 0
    while i < len(tokens):
        end = i + 1
        while end < len(tokens) and tokens[end] == tokens[i]:
            end += 1
        if end-i >= 2 and any(c.isalpha() for c in tokens[i]):
            result.append(dict(word=tokens[i], count=end-i, start_token=i, end_token=end))
        i = end
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=666)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cache = a.data_root/'CelebVDub_svae1000k_sample_seed666_fp32'
    source = cache/'manifests/train.jsonl'
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    test = [json.loads(s) for s in (cache/'manifests/test.jsonl').read_text().splitlines() if s.strip()]
    candidates = [r for r in rows if runs(r['text'])]
    # Historical D1 requires combined prompt+target duration in [1,40] seconds.
    eligible = [r for r in candidates if 13 <= r['video_frames_25hz'] <= 500 and 0.5 <= r['duration_seconds'] <= 20
                and not re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]', r['text'])]
    eligible.sort(key=lambda r:r['utterance_key'])
    random.Random(a.seed).shuffle(eligible)
    selected = eligible[:213]
    assert len(selected) == 213
    root = a.data_root/'CelebVDub'
    # Freeze selection without consulting model outputs or recognition scores.
    for r in selected:
        k = r['utterance_key'].removeprefix('celebvdub/')
        for f in [root/'audio'/(k+'.wav'),root/'video'/(k+'.mp4'),root/'avhubert_video_feat'/(k+'.npy'),
                  root/'mel_tacotron'/(k+'.npy'),root/'campplus_spk_emb_zh_en_16k'/(k+'.npy'),
                  cache/r['latent_relative_path'],cache/r['video_40hz_relative_path']]:
            assert f.is_file(), f
        v=np.load(root/'avhubert_video_feat'/(k+'.npy'),mmap_mode='r')
        assert v.shape == (r['video_frames_25hz'],1024), (k,v.shape)
    a.output.mkdir(parents=True)
    selected.sort(key=lambda r:r['utterance_key'])
    (a.output/'clips.lst').write_text(''.join(r['utterance_key'].removeprefix('celebvdub/train/')+'\n' for r in selected))
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    details=[]
    for r in selected:
        k=r['utterance_key'].removeprefix('celebvdub/')
        details.append(dict(utterance_id=k,text=r['text'],duration_seconds=r['duration_seconds'],adjacent_runs=runs(r['text']),source_split='train'))
        for category,ext in [('audio','.wav'),('video','.mp4'),('avhubert_video_feat','.npy'),('mel_tacotron','.npy')]:
            target=a.output/'CelebVDub'/category/(k+ext)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(root/category/(k+ext))
        textpath=a.output/'CelebVDub/text'/(k+'.txt'); textpath.parent.mkdir(parents=True,exist_ok=True)
        textpath.write_text(r['text']+'\n')
    (a.output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    counts=collections.Counter(run['word'] for r in details for run in r['adjacent_runs'])
    durations=[r['duration_seconds'] for r in selected]
    summary=dict(name='CelebVDub_train_adjacent_repeat213_seed666',purpose='training-set diagnostic, not held-out generalization',
        train_count=len(rows),train_adjacent_samples=len(candidates),formal_test_count=len(test),formal_test_adjacent_samples=sum(bool(runs(r['text'])) for r in test),
        eligible_count=len(eligible),selected_count=213,seed=a.seed,selection='uniform shuffle of sorted eligible clip IDs; first 213; no model-score selection',
        eligibility='native video 13..500 frames; audio 0.5..20 seconds; exclude CJK/Hangul transcripts for existing English ASR; all required media/features present',
        transcript_verification='manifest labels; not manually listened; punctuation ignored for adjacent-word detection',
        source_manifest=str(source),source_manifest_sha256=sha(source),selected_manifest_sha256=sha(a.output/'manifest.jsonl'),list_sha256=sha(a.output/'clips.lst'),
        distinct_videos=len({r['video_id'] for r in selected}),repeated_runs=sum(counts.values()),repeated_words=dict(counts.most_common()),
        duration_seconds=dict(min=min(durations),max=max(durations),mean=float(np.mean(durations)),median=float(np.median(durations))))
    (a.output/'dataset_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
