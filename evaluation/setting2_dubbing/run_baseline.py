#!/usr/bin/env python3
"""Generate and score all three baseline EMA checkpoints on GT-text Setting 2."""
from pathlib import Path
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BENCH = PROJECT / 'Video-to-Speech-benchmark'
RESULTS = BENCH / 'results/setting2_gttext_dubbing'

def main():
    logs = BENCH / 'logs/setting2_gttext_dubbing_baseline'
    logs.mkdir(parents=True, exist_ok=True)
    state = {'task': 'Setting2-GTText-Dubbing', 'expected': 115, 'complete': False, 'stages': {}}
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               PYTHONPATH=str(PROJECT / 'aligndit-baseline_1003/alignDiT_baseline/AlignDiT/src'))
    for step in (100000, 150000, 200000):
        tag = f'AlignDiT_{step // 1000}k'
        output = RESULTS / tag
        state['current'] = tag
        (RESULTS / 'baseline_pipeline_progress.json').write_text(json.dumps(state, indent=2) + '\n')
        print(f'Starting {tag}: supplied GT dialogue and cross-utterance reference', flush=True)
        command = [sys.executable, '-u', str(HERE / 'infer_aligndit_baseline.py'),
                   '--checkpoint', str(PROJECT / f'aligndit-baseline_1003/ckpts/model_{step}.pt'),
                   '--expected-update', str(step), '--output-dir', str(output)]
        with (logs / f'{tag}.log').open('a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-u', str(BENCH / 'scripts/run_setting2_evaluation.py'),
                            '--generated', str(output), '--with-emosim'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-u', str(HERE / 'verify_baseline.py'), '--method', tag],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['stages'][tag] = {'complete': True}
        print(f'Completed {tag}', flush=True)
    summary = {'task': state['task'], 'n': 115, 'complete': True, 'methods': {}}
    for tag in state['stages']:
        summary['methods'][tag] = json.loads((RESULTS / tag / 'evaluation_verification.json').read_text())
    (RESULTS / 'baseline_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state.update(complete=True, current='complete')
    (RESULTS / 'baseline_pipeline_progress.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
