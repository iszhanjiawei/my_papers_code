"""Validate complete model results and compare fixed reference-defined subgroups."""
import argparse
import json
from pathlib import Path
import numpy as np
from jiwer import compute_measures
from validate_tpca_eval import validate


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]


def index_runs(rows):
    return {(r['utterance_id'],d['start_token'],d['end_token'],d['word']):d for r in rows for d in r['runs']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',type=Path)
    p.add_argument('--specs',nargs='+',default=['d1_tpca_150000','d1_tpca_200000','c2_svae_speaker_200000'])
    a=p.parse_args();root=a.dataset
    gt_root=root/'results/ground_truth'
    gt_summary=validate(gt_root,'wer',root/'clips.lst','train',root/'CelebVDub/avhubert_feat')
    gt_transcripts={r['utterance_id']:r for r in read_rows(gt_root/'_wer_results.jsonl')}
    gt_runs=index_runs(read_rows(gt_root/'_repeat_details.jsonl'))
    confirmed={k for k,v in gt_runs.items() if v['exact_run_preserved']}
    dataset_rows=read_rows(root/'samples.jsonl')
    ordinary={r['utterance_id'] for r in dataset_rows if max(d['count'] for d in r['adjacent_runs'])<=5}
    result={'dataset':json.loads((root/'dataset_summary.json').read_text()),'gt_wer':gt_summary,
            'gt_repetition':json.loads((gt_root/'_repeat_summary.json').read_text()),'models':{}}
    for spec in a.specs:
        out=root/'results'/spec
        details={stage:validate(out,stage,root/'clips.lst','train',root/'CelebVDub/avhubert_feat') for stage in ['wav','features','sim','wer','emosim','avsync']}
        wer_rows=read_rows(out/'_wer_results.jsonl')
        assert all(r['truth']==gt_transcripts[r['utterance_id']]['truth'] for r in wer_rows)
        model_runs=index_runs(read_rows(out/'_repeat_details.jsonl'))
        assert set(model_runs)==set(gt_runs)
        matched=[model_runs[k] for k in confirmed]
        denom=sum(d['count'] for d in matched)
        details['repetition']=json.loads((out/'_repeat_summary.json').read_text())
        details['gt_asr_exact_run_subset']={'runs':len(matched),'repeated_reference_tokens':denom,
            'exact_runs':sum(d['exact_run_preserved'] for d in matched),
            'exact_run_rate':sum(d['exact_run_preserved'] for d in matched)/len(matched),
            'deleted_tokens':sum(d['deletions'] for d in matched),
            'repeated_token_deletion_rate':sum(d['deletions'] for d in matched)/denom,
            'definition':'Only reference occurrences whose complete run is preserved by ASR on original GT; not a manually verified subset'}
        details['non_extreme_max_run_5']={}
        for metric in ['sim','wer','emosim','avsync']:
            rows=[r for r in read_rows(out/f'_{metric}_results.jsonl') if r['utterance_id'] in ordinary]
            assert len(rows)==210
            val=compute_measures([r['truth'] for r in rows],[r['hypo'] for r in rows])['wer'] if metric=='wer' else float(np.mean([r[metric] for r in rows]))
            details['non_extreme_max_run_5'][metric]=val
        (out/'verified_summary.json').write_text(json.dumps(details,indent=2)+'\n')
        result['models'][spec]=details
    name='comparison_summary.json' if len(a.specs)==3 else 'partial_comparison_summary.json'
    (root/name).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for spec,d in result['models'].items():
        print(spec,{k:round(d[k]['value'],5) for k in ('wer','sim','emosim','avsync')},d['gt_asr_exact_run_subset'])

if __name__=='__main__': main()
