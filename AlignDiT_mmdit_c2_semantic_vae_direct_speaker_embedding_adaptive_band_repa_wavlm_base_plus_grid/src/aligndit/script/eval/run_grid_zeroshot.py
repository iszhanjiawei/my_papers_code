"""Run one EMA checkpoint on one GPU, with four independent resumable shards."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from aligndit.script.eval.grid_s1_common import identity, publish, read, sha
from aligndit.script.eval.grid_zeroshot_common import load_prepared, summarize

STAGES = ['infer', 'verify', 'audit', 'sim', 'wer', 'emosim', 'avsync', 'merge', 'independent-verify']


def worker_command(a, stage):
    command = [sys.executable, '-u', '-m', 'aligndit.script.eval.eval_grid_zeroshot', '--task', stage,
        '--shared-dir', str(a.shared_dir), '--output-dir', str(a.output_dir), '--checkpoint', str(a.checkpoint),
        '--step', str(a.step), '--nshard', str(a.workers_per_gpu)]
    if a.smoke:
        command += ['--smoke']
    return command


def workers(a, stage, rows):
    jobs = []
    try:
        for shard in range(a.workers_per_gpu):
            if stage != 'infer':
                result_path = a.output_dir/f'{stage}_shard{shard}.jsonl'
                summary_path = a.output_dir/f'{stage}_shard{shard}_summary.json'
                if result_path.exists() and summary_path.exists():
                    result = [json.loads(line) for line in result_path.read_text().splitlines()]
                    summary = read(summary_path)
                    if (summary['nshard'] != a.workers_per_gpu or summary['inference_summary_sha256'] != sha(a.output_dir/'inference_summary.json')
                            or [r['id'] for r in result] != [identity(r) for r in rows[shard::a.workers_per_gpu]]
                            or any(summary.get(k) != v for k, v in summarize(result, stage).items())):
                        raise RuntimeError('Cached metric shard differs from requested evaluation')
                    print(f'Reuse verified {stage} shard {shard}', flush=True)
                    continue
            command = worker_command(a, stage)+['--shard', str(shard)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
            with (a.output_dir/f'{stage}_worker{shard}.log').open('a') as log:
                job = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            jobs.append(job)
            print(f'{stage} shard={shard} physical_gpu={a.gpu} pid={job.pid}', flush=True)
        publish(a.output_dir/'active_workers.json', dict(stage=stage, pids=[job.pid for job in jobs]))
        pending = list(jobs)
        while pending:
            for job in pending[:]:
                status = job.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f'{stage} worker {job.pid} exited {status}; inspect worker log')
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


def link_gt_cache(a, rows, metadata):
    if not a.gt_joint_source:
        return
    source_root = a.gt_joint_source.parent.parent
    verified, protocol = read(source_root/'verification.json'), read(source_root/'protocol.json')
    if (verified['complete'] is not True or verified['succeeded'] != 3274
            or protocol['manifest_sha256'] != metadata['source_cache_manifest_sha256']):
        raise RuntimeError('GT cache source is not verified for these same GRID targets')
    for row in rows:
        relative = Path(identity(row)+'.npy')
        for suffix in ('.npy', '.json'):
            source = a.gt_joint_source/relative.with_suffix(suffix)
            destination = a.output_dir/'avhubert_joint/gt'/relative.with_suffix(suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha(destination) != sha(source):
                    raise RuntimeError('Existing target GT joint feature differs from verified source')
            else:
                os.link(source, destination)
    publish(a.output_dir/'gt_joint_cache_reuse.json', dict(source=str(a.gt_joint_source), count=len(rows),
            method='hard links to verified immutable GT joint features and receipts; evaluator revalidates each binding/hash'))


def capture_sources(a):
    project = Path(__file__).parents[4]
    files = sorted(p for root in (project/'src/aligndit', project/'src/f5_tts/eval') for p in root.rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml', '.yml'))
    actual = [dict(path=str(p), sha256=sha(p)) for p in files]
    path = a.output_dir/'launch_source_snapshot.json'
    if path.exists():
        if read(path)['source_files'] != actual:
            raise RuntimeError('Source files changed since this run was launched')
    else:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        publish(path, dict(git_revision=revision, source_files=actual))


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    base = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/projects/data')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--step', type=int, choices=[100000, 150000], required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--workers-per-gpu', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-stage', choices=STAGES, default='infer')
    p.add_argument('--gt-joint-source', type=Path, default=base/'evaluations/GRID_s2_target_only_100k_seed0_nfe32_cfg5_2/avhubert_joint/gt')
    a = p.parse_args()
    if a.workers_per_gpu < 1 or ',' in a.gpu:
        raise ValueError('Use one physical GPU per checkpoint and positive shard count')
    rows, metadata = load_prepared(a.shared_dir, a.smoke)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    capture_sources(a)
    link_gt_cache(a, rows, metadata)
    stage = a.start_stage
    try:
        for stage in STAGES[STAGES.index(a.start_stage):]:
            publish(a.output_dir/'pipeline_status.json', dict(stage=stage, complete=False, pid=os.getpid(), gpu=a.gpu))
            if stage in ('infer', 'sim', 'wer', 'emosim', 'avsync'):
                workers(a, stage, rows)
            elif stage == 'independent-verify':
                command = [sys.executable, '-u', '-m', 'aligndit.script.eval.verify_grid_zeroshot',
                    '--shared-dir', str(a.shared_dir), '--output-dir', str(a.output_dir), '--nshard', str(a.workers_per_gpu)]
                if a.smoke:
                    command += ['--smoke']
                with (a.output_dir/'independent_verify.log').open('a') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                subprocess.run(worker_command(a, stage), check=True)
        publish(a.output_dir/'pipeline_status.json', dict(stage='complete', complete=True, pid=os.getpid(), gpu=a.gpu))
    except BaseException as error:
        publish(a.output_dir/'pipeline_status.json', dict(stage=stage, complete=False, error=str(error), pid=os.getpid(), gpu=a.gpu))
        raise


if __name__ == '__main__':
    main()
