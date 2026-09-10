"""Transcript-only occurrence diagnostics; retain standard corpus WER separately."""
import argparse
import json
from pathlib import Path
from build_repeat213 import TOKEN_RE, runs


def tokens(text):
    return [m.group().replace('’', "'").casefold() for m in TOKEN_RE.finditer(text)]


def align(ref,hyp):
    # Levenshtein alignment; ties prefer diagonal, then deletion, then insertion.
    n,m=len(ref),len(hyp)
    cost=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): cost[i][0]=i
    for j in range(m+1): cost[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost[i][j]=min(cost[i-1][j-1]+(ref[i-1]!=hyp[j-1]),cost[i-1][j]+1,cost[i][j-1]+1)
    mapping={}; i,j=n,m
    while i or j:
        if i and j and cost[i][j]==cost[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
            mapping[i-1]=(j-1,'equal' if ref[i-1]==hyp[j-1] else 'substitution');i-=1;j-=1
        elif i and cost[i][j]==cost[i-1][j]+1:
            mapping[i-1]=(None,'deletion');i-=1
        else: j-=1
    return mapping


def score(raw_truth,raw_hypo):
    ref,hyp=tokens(raw_truth),tokens(raw_hypo)
    mapping=align(ref,hyp)
    details=[]
    for run in runs(raw_truth):
        mapped=[mapping[i] for i in range(run['start_token'],run['end_token'])]
        matched=[j for j,kind in mapped if kind=='equal']
        complete=len(matched)==run['count'] and matched==list(range(matched[0],matched[0]+len(matched)))
        exact=complete and (matched[0]==0 or hyp[matched[0]-1]!=run['word']) and (matched[-1]+1==len(hyp) or hyp[matched[-1]+1]!=run['word'])
        details.append(dict(**run,matched=len(matched),deletions=sum(kind=='deletion' for _,kind in mapped),substitutions=sum(kind=='substitution' for _,kind in mapped),exact_run_preserved=bool(exact)))
    return details


def main():
    p=argparse.ArgumentParser();p.add_argument('wer_jsonl',type=Path);a=p.parse_args()
    rows=[json.loads(s) for s in a.wer_jsonl.read_text().splitlines() if s.startswith('{')]
    assert len(rows)==213 and len({r['utterance_id'] for r in rows})==213
    output=[]
    for r in rows:
        details=score(r['raw_truth'],r['raw_hypo'])
        assert details, r['utterance_id']
        output.append(dict(utterance_id=r['utterance_id'],raw_truth=r['raw_truth'],raw_hypo=r['raw_hypo'],runs=details))
    allruns=[d for r in output for d in r['runs']]
    count=sum(r['count'] for r in allruns)
    summary=dict(samples=len(rows),runs=len(allruns),repeated_reference_tokens=count,
        matched_tokens=sum(r['matched'] for r in allruns),deleted_tokens=sum(r['deletions'] for r in allruns),substituted_tokens=sum(r['substitutions'] for r in allruns),
        exact_runs=sum(r['exact_run_preserved'] for r in allruns),
        exact_run_rate=sum(r['exact_run_preserved'] for r in allruns)/len(allruns),
        repeated_token_deletion_rate=sum(r['deletions'] for r in allruns)/count,
        repeated_token_substitution_rate=sum(r['substitutions'] for r in allruns)/count,
        method='casefold/apostrophe-preserving audit tokenizer; word Levenshtein with diagonal/deletion/insertion tie order; exact contiguous repeated run with no extra same-word neighbor',
        limitation='ASR transcript diagnostic, not human listening or time-alignment ground truth; alignment ambiguity possible')
    a.wer_jsonl.with_name('_repeat_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    a.wer_jsonl.with_name('_repeat_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
