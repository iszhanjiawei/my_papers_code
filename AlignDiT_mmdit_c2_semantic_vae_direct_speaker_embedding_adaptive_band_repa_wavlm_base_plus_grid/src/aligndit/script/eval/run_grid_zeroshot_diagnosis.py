"""Run paired speaker diagnostics in an independent session (launch with setsid)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from aligndit.script.eval.grid_s1_common import identity, publish, read, sha
from aligndit.script.eval.diagnose_grid_zeroshot import (
    VARIANTS, check_diagnostic_output, load_selection, validate_sources,
)


def run(a):
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    live = []
    started = time.time()
    state = dict(complete=False, phase='prepare', started_at=time.time(),
                 source_sha256=sha(__file__), children=[])

    def status(phase, **kwargs):
        state.update(phase=phase, elapsed_seconds=time.time()-started, **kwargs)
        publish(root / 'pipeline_status.json', state)
        print(json.dumps({k: v for k, v in state.items() if k != 'children'}), flush=True)

    def launch(module, task, step, gpu, extra=(), name=None):
        label = name or f'{step}_{task}'
        command = [sys.executable, '-u', '-m', 'aligndit.script.eval.'+module,
                   '--task', task, '--step', str(step), '--prior-root', str(a.prior_root.resolve()),
                   '--output-root', str(root), *extra]
        env = dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(gpu))
        path = logs / (label+'.log')
        handle = path.open('a')
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        live.append(process)
        state['children'].append(dict(pid=process.pid, name=label, log=str(path), command=command, gpu=gpu))
        publish(root / 'pipeline_status.json', state)
        return process, label

    def wait_all(jobs):
        pending = list(jobs)
        while pending:
            for job in pending[:]:
                process, label = job
                code = process.poll()
                if code is None:
                    continue
                pending.remove(job)
                if code:
                    raise RuntimeError(f'{label} failed with exit code {code}; see logs/{label}.log')
            if pending:
                time.sleep(2)

    generation = 'diagnose_grid_zeroshot'
    scoring = 'score_grid_zeroshot_diagnosis'
    try:
        for step in (100000, 150000):
            wait_all([launch(generation, 'prepare', step, 0)])
        status('baseline_reproduction')
        wait_all([launch(generation, 'infer', step, gpu,
                         ['--variants', 'baseline_regenerated', '--limit', '3'], f'{step}_reproduce')
                  for step, gpu in ((100000, 0), (150000, 1))])
        rows, selection = load_selection(root)
        for step in (100000, 150000):
            for row in rows[:3]:
                receipt = check_diagnostic_output(root, step, 'baseline_regenerated', row, selection)
                if not receipt['byte_identical_to_prior']:
                    raise RuntimeError('Fresh baseline differs from the previous evaluated WAV')
        status('reference_reconstruction')
        wait_all([launch(generation, 'reconstruct', 150000, 0)])
        status('infer')
        jobs = []
        for step, gpu in ((100000, 0), (150000, 1)):
            for shard in range(a.workers_per_gpu):
                jobs.append(launch(generation, 'infer', step, gpu,
                    ['--variants', 'all', '--nshard', str(a.workers_per_gpu), '--shard', str(shard)],
                    f'{step}_infer_{shard}'))
        wait_all(jobs)
        status('verify_generation')
        verified = {}
        for step in (100000, 150000):
            validate_sources(read(root / f'ema_{step}/protocol.json'))
            for row in rows:
                noise_hashes = set()
                for variant in VARIANTS:
                    receipt = check_diagnostic_output(root, step, variant, row, selection)
                    if variant != 'target_only':
                        noise_hashes.add(receipt['target_noise_sha256'])
                if len(noise_hashes) != 1:
                    raise RuntimeError('Paired intervention initial target noise differs')
            for variant in VARIANTS:
                actual = {str(p.relative_to(root / f'ema_{step}' / variant))
                          for p in (root / f'ema_{step}' / variant).glob('val/*/*.wav')}
                if actual != {r['audio_relative_path'] for r in rows}:
                    raise RuntimeError('Unexpected generation coverage')
            verified[str(step)] = dict(count_per_variant=len(rows), variants=list(VARIANTS),
                                       baseline_reproduced=3, paired_target_noise_identical=True)
        publish(root / 'generation_verification.json', verified)
        status('speaker_embeddings_and_asr')
        # Serialize embedding jobs across checkpoints to reuse shared controls
        # without concurrent writes to their content-addressed feature cache.
        embed_job = launch(scoring, 'embed', 100000, 0)
        wer_job = launch(scoring, 'wer', 100000, 1)
        next_embed = next_wer = False
        while not (next_embed and next_wer):
            if not next_embed and embed_job[0].poll() is not None:
                wait_all([embed_job])
                embed_job = launch(scoring, 'embed', 150000, 0)
                next_embed = True
            if not next_wer and wer_job[0].poll() is not None:
                wait_all([wer_job])
                wer_job = launch(scoring, 'wer', 150000, 1)
                next_wer = True
            if not (next_embed and next_wer):
                time.sleep(2)
        wait_all([embed_job, wer_job])
        status('summarize')
        for step in (100000, 150000):
            wait_all([launch(scoring, 'summarize', step, 0, ['--require-wer'])])
            validate_sources(read(root / f'ema_{step}/protocol.json'))
            result = read(root / f'ema_{step}/diagnostic_scores/diagnosis_summary.json')
            if result['count'] != len(rows) or not result['wer_complete']:
                raise RuntimeError('Incomplete final diagnosis')
        status('complete', complete=True)
    except BaseException as error:
        for process in live:
            if process.poll() is None:
                process.terminate()
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    a = parser.parse_args()
    if not 1 <= a.workers_per_gpu <= 4 or a.prior_root.resolve() == a.output_root.resolve():
        parser.error('Use 1–4 workers per GPU and a separate diagnostic output root')
    os.chdir(Path(__file__).parents[4])
    run(a)


if __name__ == '__main__':
    main()
