#!/usr/bin/env python3
"""Recompute full baseline GT-dialogue coverage, provenance and four metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import string
import numpy as np
import soundfile as sf
from jiwer import compute_measures
from zhon.hanzi import punctuation

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / 'Video-to-Speech-benchmark'
PACKAGE = HERE.parents[1] / 'Setting2-Dubbing-test'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def normalized(text):
    for char in punctuation + string.punctuation:
        text = text.replace(char, '')
    return text.replace('  ', ' ').lower()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', action='append')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (BENCH / 'setting2/evaluation.jsonl').read_text().splitlines()]
    original = {r['id']: r for r in map(json.loads, (BENCH / 'setting2/manifest.jsonl').read_text().splitlines())}
    package = {r['id']: r for r in map(json.loads, (PACKAGE / 'inference.jsonl').read_text().splitlines())}
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == 115 and set(package) == set(ids)
    verified = {}
    for method in args.method or ['AlignDiT_100k', 'AlignDiT_150k', 'AlignDiT_200k']:
        root = BENCH / 'results/setting2_gttext_dubbing' / method
        cfg = json.loads((root / 'run_config.json').read_text())
        assert cfg['task'] == 'Setting2-GTText-Dubbing'
        assert cfg['target_gt_text_read'] and cfg['target_gt_text_as_target_condition']
        assert not cfg['target_gt_audio_read'] and not cfg['target_acoustic_cache_read'] and not cfg['vsr_used']
        assert cfg['reference_policy'] == 'cross-reference' and not cfg['oracle_reference']
        assert cfg['seed'] == 0 and cfg['steps'] == 32 and cfg['cfg_t'] == 5 and cfg['cfg_v'] == 2
        assert cfg['use_ema'] and cfg['ode_method'] == 'euler' and cfg['sway'] == -1
        assert cfg['checkpoint_update'] == int(method.split('_')[1][:-1]) * 1000
        assert cfg['checkpoint_sha256'] == sha(cfg['checkpoint'])
        assert cfg['adapter_sha256'] == sha(HERE / 'infer_aligndit_baseline.py')
        for relative, digest in cfg['source_sha256'].items():
            assert sha(Path(cfg['baseline_repo']) / relative) == digest
        assert cfg['manifest_sha256'] == sha(PACKAGE / 'aligndit_inference.jsonl')
        progress = json.loads((root / 'progress.json').read_text())
        assert progress['requested'] == progress['complete'] == 115 and not progress['failures']
        hashes = {}
        audio_samples = 0
        for row in rows:
            ident = row['id']
            wav = root / (ident + '.wav')
            meta = json.loads(wav.with_suffix('.json').read_text())
            inputs = meta['inputs']
            assert inputs['run'] == cfg and inputs['id'] == ident
            assert inputs['reference_id'] == row['reference_id'] == package[ident]['reference_id'] != ident
            assert inputs['target_text'] == original[ident]['target_text'].strip().lower()
            assert inputs['input_sha256']['target_text_path'] == original[ident]['text_sha256']
            assert inputs['input_sha256']['reference_audio'] == original[ident]['reference_audio_sha256']
            assert inputs['input_sha256']['video_feature'] == sha(PACKAGE / package[ident]['video_feature'])
            assert not any(k in inputs for k in ['target_audio', 'gt_audio', 'target_latent', 'gt_av_feature'])
            hashes[ident] = sha(wav)
            assert hashes[ident] == meta['audio_sha256']
            audio, rate = sf.read(wav, always_2d=True)
            assert rate == 16000 and audio.shape == (package[ident]['target_samples_16khz'], 1)
            assert np.isfinite(audio).all() and np.square(audio).mean() > 0
            audio_samples += len(audio)
            gt = np.load(row['gt_av_feature'], allow_pickle=False)
            gen = np.load(root / 'avhubert_feat' / (ident + '.npy'), allow_pickle=False)
            assert gt.shape == gen.shape and gt.ndim == 2 and np.isfinite(gen).all()
        assert audio_samples == 8501120
        metrics = {}
        for metric, key in [('wer', 'wer'), ('spksim', 'sim'), ('emosim', 'emosim'), ('avsync', 'avsync')]:
            report = json.loads((root / f'setting2_{metric}.json').read_text())
            assert report['n'] == 115 and [r['id'] for r in report['results']] == ids
            assert report['generated_audio_sha256'] == hashes
            assert report['manifest_sha256'] == sha(BENCH / 'setting2/evaluation.jsonl')
            assert report['evaluator_sha256'] == sha(BENCH / 'scripts/evaluate_setting2.py')
            assert all(result['reference_id'] == row['reference_id'] for row, result in zip(rows, report['results']))
            if metric == 'wer':
                for row, result in zip(rows, report['results']):
                    assert result['raw_truth'] == row['target_text'] and result['truth'] == normalized(row['target_text'])
                corpus = compute_measures([r['truth'] for r in report['results']], [r['hypo'] for r in report['results']])
                score = corpus['wer'] * 100
                assert corpus['hits'] + corpus['substitutions'] + corpus['deletions'] == 1721
                metrics['corpus_wer_counts'] = {k: corpus[k] for k in ['hits', 'substitutions', 'deletions', 'insertions']}
            else:
                values = [r[key] for r in report['results']]
                assert np.isfinite(values).all()
                score = float(np.mean(values))
            assert abs(score - report['score']) < 1e-10
            metrics[metric] = score
        record = {'complete': True, 'n': 115, 'duration_seconds': audio_samples / 16000,
                  'task': cfg['task'], 'target_gt_dialogue_verified': True, 'target_acoustic_inputs_absent': True,
                  'four_metrics_independently_recomputed': True, 'metrics': metrics,
                  'run_config_sha256': sha(root / 'run_config.json')}
        (root / 'evaluation_verification.json').write_text(json.dumps(record, indent=2) + '\n')
        verified[method] = record
    print(json.dumps(verified, indent=2))

if __name__ == '__main__':
    main()
