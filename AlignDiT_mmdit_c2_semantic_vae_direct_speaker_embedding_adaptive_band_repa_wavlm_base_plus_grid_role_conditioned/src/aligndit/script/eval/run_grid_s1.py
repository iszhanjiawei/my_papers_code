"""Run resumable GRID inference and four metrics in detached GPU workers."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from aligndit.script.eval.grid_s1_common import (
    ROLE_MODE, TEXT_CONDITIONING, grid_config_path, metric_summary, publish, read, sha, validate_text_conditioning,
)


def workers(a, stage):
    gpus = a.gpus.split(',')
    count = len(gpus) * a.workers_per_gpu
    jobs = []
    try:
        for shard in range(count):
            log = a.output_dir/f'{stage}_worker{shard}.log'
            if stage == 'infer':
                command = [sys.executable, '-u', '-m', f'aligndit.script.eval.infer_grid_{a.setting}',
                           '--checkpoint', str(a.checkpoint), '--step', str(a.step), '--config', str(a.config)]
                if a.setting == 's2':
                    command += ['--text-conditioning', a.text_conditioning]
            else:
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_s1', '--setting', a.setting, '--task', stage,
                           '--text-conditioning', a.text_conditioning]
                summary_path = a.output_dir/f'{stage}_shard{shard}_summary.json'
                results_path = a.output_dir/f'{stage}_shard{shard}.jsonl'
                if summary_path.exists() and results_path.exists():
                    summary = read(summary_path)
                    results = [json.loads(line) for line in results_path.read_text().splitlines()]
                    if (summary['nshard'] == count and summary['inference_summary_sha256'] == sha(a.output_dir/'inference_summary.json')
                            and all(summary[k] == v for k, v in metric_summary(results, stage).items())):
                        print(f'Reuse verified {stage} shard {shard}', flush=True)
                        continue
                    raise RuntimeError(f'Stale metric results: {summary_path}')
            command += ['--output-dir', str(a.output_dir), '--nshard', str(count), '--shard', str(shard)]
            if a.limit:
                command += ['--limit', str(a.limit)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[shard//a.workers_per_gpu],
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
            handle = log.open('a')
            job = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            handle.close()
            jobs.append(job)
            print(f'{stage} shard={shard} gpu={env["CUDA_VISIBLE_DEVICES"]} pid={job.pid}', flush=True)
        publish(a.output_dir/'active_workers.json', {'stage': stage, 'pids': [j.pid for j in jobs]})
        pending = list(jobs)
        while pending:
            for job in pending[:]:
                result = job.poll()
                if result is not None:
                    if result:
                        raise RuntimeError(f'{stage} worker {job.pid} exited {result}; inspect worker logs')
                    pending.remove(job)
            if pending:
                time.sleep(1)
    finally:
        for job in jobs:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
        for job in jobs:
            if job.poll() is None:
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait()


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    prefix = os.environ.get('ROOT_PREFIX', '')
    data = Path(prefix+'/zjw524/projects/data')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting', choices=['s1','s2'], default='s1')
    p.add_argument('--text-conditioning', choices=TEXT_CONDITIONING)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--step', type=int, default=100000)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--workers-per-gpu', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--start-stage', choices=['infer', 'sim', 'wer', 'emosim', 'avsync', 'merge'], default='infer')
    a = p.parse_args()
    if a.text_conditioning is None:
        a.text_conditioning = ROLE_MODE if a.setting == 's2' else 'reference-target'
    validate_text_conditioning(a.setting, a.text_conditioning, a.output_dir, require_contract=a.start_stage != 'infer')
    if a.workers_per_gpu < 1 or (a.limit and a.limit < len(a.gpus.split(','))*a.workers_per_gpu):
        raise ValueError('Too few clips for worker count')
    a.config = (a.config or grid_config_path(a.text_conditioning)).resolve(strict=True)
    if a.start_stage == 'infer':
        from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import load_composed_config
        config = load_composed_config(a.config)
        if bool(config.model.arch.get('role_conditioned_text', False)) != (a.text_conditioning == ROLE_MODE):
            raise ValueError('GRID text conditioning is incompatible with the checkpoint architecture config')
        a.checkpoint = (a.checkpoint or Path(config.ckpts.save_dir) / f'model_{a.step}.pt').resolve(strict=True)
    if a.text_conditioning == ROLE_MODE and (a.output_dir/'protocol.json').is_file():
        saved = read(a.output_dir/'protocol.json')
        if saved.get('config_sha256') != sha(a.config) or saved['checkpoint']['update'] != a.step:
            raise RuntimeError('Role-conditioned output directory belongs to another config or checkpoint step')
        if a.checkpoint is not None and sha(a.checkpoint) != saved['checkpoint']['sha256']:
            raise RuntimeError('Role-conditioned output directory belongs to another checkpoint')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.setting == 's2':
        from aligndit.script.eval.grid_s2_common import prepare_pairs
        prepare_pairs(data/'GRID_svae1000k_sample_seed666_fp32', a.output_dir)
    stages = ['infer', 'audit', 'sim', 'wer', 'emosim', 'avsync', 'merge']
    selected = stages[stages.index(a.start_stage):]
    try:
        for stage in selected:
            publish(a.output_dir/'pipeline_status.json', {'stage': stage, 'complete': False, 'pid': os.getpid()})
            if stage not in ('audit', 'merge'):
                workers(a, stage)
            if stage == 'audit':
                subprocess.run([sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_s1',
                                '--task', 'audit', '--setting', a.setting, '--text-conditioning', a.text_conditioning,
                                '--output-dir', str(a.output_dir)], check=True)
            if stage in ('infer', 'merge'):
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_s1', '--task', 'verify' if stage=='infer' else 'merge',
                           '--setting', a.setting, '--text-conditioning', a.text_conditioning,
                           '--output-dir', str(a.output_dir), '--nshard', str(len(a.gpus.split(','))*a.workers_per_gpu)]
                if a.limit:
                    command += ['--limit', str(a.limit)]
                subprocess.run(command, check=True)
        publish(a.output_dir/'pipeline_status.json', {'stage': 'complete', 'complete': True, 'pid': os.getpid()})
    except BaseException as e:
        publish(a.output_dir/'pipeline_status.json', {'stage': stage, 'complete': False, 'error': str(e), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
