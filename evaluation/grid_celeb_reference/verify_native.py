"""Independent CPU coverage/hash/aggregation checks for the native 660 rerun."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from grid_wer import normalize
from native_frontend import NATIVE_RUN, verify_audit
from protocol import rows, sha, save_json, DEFAULT_RUN

TASKS = ('sim','wer','emosim','emoembed','avsync')


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a,1):
        current = [i]
        for j, y in enumerate(b,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous = current
    return previous[-1]


def read_results(path, ids):
    result = [json.loads(s) for s in path.read_text().splitlines()]
    if len(result) != 660 or {r['target_id'] for r in result} != ids:
        raise ValueError('Incomplete metric: '+str(path))
    return result


def main(run):
    original = {r['target_id']:r for r in rows(DEFAULT_RUN)}
    verification = {'complete':False,'requested_per_model':660,'models':{},'aggregation_independently_recomputed':True}
    for method in ('baseline','repa'):
        child = run/method
        audit = verify_audit(child,method)
        inference = rows(child)
        ids = {r['target_id'] for r in inference}
        out = child/'outputs'/method
        summary = json.loads((out/'inference_summary.json').read_text())
        records = {r['target_id']:r for r in summary['outputs']}
        wavs = {p.relative_to(out).with_suffix('').as_posix() for p in (out/'test').rglob('*.wav')}
        if summary['count'] != 660 or set(records) != ids or wavs != ids or summary['partial_smoke_test']:
            raise ValueError('Generated WAV coverage differs from frozen targets')
        if summary['pair_manifest_sha256'] != sha(child/'pairs.jsonl'):
            raise ValueError('Inference manifest changed')
        for row in inference:
            old = original[row['target_id']]
            for key in ('target_text','source_video_sha256','ref_id','ref_text','ref_audio_sha256'):
                if row[key] != old[key]:
                    raise ValueError('Pairing/utterance changed')
            if 'target_gt_audio' in row:
                raise ValueError('GT acoustic path exposed to inference')
            record = records[row['target_id']]
            wave, sr = sf.read(out/(row['target_id']+'.wav'),dtype='float32')
            if sr != 16000 or wave.shape != (48000,) or not np.isfinite(wave).all() or not np.any(wave):
                raise ValueError('Invalid native waveform')
            if sha(out/(row['target_id']+'.wav')) != record['sha256'] or record['ref_id'] != row['ref_id']:
                raise ValueError('Native waveform identity/hash mismatch')
            if record['tail_adjustment_samples_16k'] != 0:
                raise ValueError('Native output required a tail adjustment')
            vis_record = json.loads((child/'assets/records'/(row['target_id']+'.json')).read_text())
            if vis_record['frontend_method'] != method or vis_record['audio_used_for_visual_condition']:
                raise ValueError('Wrong frontend/target acoustic conditioning')
            for k,v in vis_record['artifact_sha256'].items():
                if sha(vis_record['artifact_paths'][k]) != v:
                    raise ValueError('Native visual hash mismatch')
            for root in (child/'assets/gt_avhubert',out/'avhubert_feat'):
                feat = np.load(root/(row['target_id']+'.npy'),allow_pickle=False)
                if feat.shape != (75,1024) or feat.dtype != np.float32 or not np.isfinite(feat).all():
                    raise ValueError('Invalid scoring AV feature')
        values = {}
        for task in TASKS:
            results = read_results(out/f'_{task}_results.jsonl',ids)
            metric_summary = json.loads((out/f'_{task}_summary.json').read_text())
            if metric_summary['count'] != 660 or metric_summary['manifest_sha256'] != sha(child/'pairs.jsonl'):
                raise ValueError('Metric summary protocol mismatch')
            if task == 'wer':
                edits = sum(distance(r['truth'].split(),r['hypo'].split()) for r in results)
                words = sum(len(r['truth'].split()) for r in results)
                if words != 3960:
                    raise ValueError('Expected 3960 GRID reference words')
                value = edits/words
                normalized_edits = sum(distance(normalize(r['truth']).split(),normalize(r['hypo']).split()) for r in results)
                normalized_summary = json.loads((out/'_wer_grid_normalized_summary.json').read_text())
                if abs(normalized_edits/3960-normalized_summary['wer']) > 1e-12:
                    raise ValueError('Normalized WER aggregation mismatch')
                values['wer_grid_normalized'] = normalized_edits/3960
                values['normalized_word_edits'] = normalized_edits
            else:
                if not all(math.isfinite(r[task]) for r in results):
                    raise ValueError('Nonfinite metric')
                value = math.fsum(r[task] for r in results)/660
            if abs(value-metric_summary[task]) > 1e-12:
                raise ValueError('Metric aggregate differs from independent recomputation')
            values[task] = value
        if method == 'baseline':
            if len(audit['baseline_native_reference_cache']) != 66:
                raise ValueError('Incomplete native cached Mel reference audit')
            for record in audit['baseline_native_reference_cache']:
                if sha(record['mel_source']) != record['mel_sha256'] or sha(record['mel_path']) != record['mel_sha256']:
                    raise ValueError('Baseline Mel differs from native training cache')
        else:
            for record in audit['references']:
                row = next(r for r in inference if r['ref_id'] == record['ref_id'])
                if sha(row['ref_latent_path']) != record['latent_sha256'] or sha(row['ref_speaker_path']) != record['speaker_sha256']:
                    raise ValueError('REPA reference differs from native training cache')
        verification['models'][method] = {'wav_count':660,'metric_counts':{task:660 for task in TASKS},
            'all_waveforms_48000_samples':True,'all_native_tails_unadjusted':True,'pairing_unchanged':True,
            'target_acoustic_inputs_used':False,'values':values}
    verification['complete'] = True
    save_json(run/'verification.json',verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=NATIVE_RUN)
    main(p.parse_args().run)
