"""Run resumable GRID inference and four metrics in detached GPU workers."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from aligndit.script.eval.grid_s1_common import metric_summary, publish, read, sha


def workers(a, stage):
    gpus = a.gpus.split(',')
    count = len(gpus) * a.workers_per_gpu
    jobs = []
    try:
        for shard in range(count):
            log = a.output_dir/f'{stage}_worker{shard}.log'
            if stage == 'infer':
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.infer_grid_s1',
                           '--checkpoint', str(a.checkpoint), '--step', str(a.step)]
            else:
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_s1', '--task', stage]
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
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=data/'ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char/model_100000.pt')
    p.add_argument('--step', type=int, default=100000)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--workers-per-gpu', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--start-stage', choices=['infer', 'sim', 'wer', 'emosim', 'avsync', 'merge'], default='infer')
    a = p.parse_args()
    if a.workers_per_gpu < 1 or (a.limit and a.limit < len(a.gpus.split(','))*a.workers_per_gpu):
        raise ValueError('Too few clips for worker count')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stages = ['infer', 'sim', 'wer', 'emosim', 'avsync', 'merge']
    selected = stages[stages.index(a.start_stage):]
    try:
        for stage in selected:
            publish(a.output_dir/'pipeline_status.json', {'stage': stage, 'complete': False, 'pid': os.getpid()})
            if stage != 'merge':
                workers(a, stage)
            if stage in ('infer', 'merge'):
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_s1', '--task', 'verify' if stage=='infer' else 'merge',
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
