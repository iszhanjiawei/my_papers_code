"""Run resumable LRS3 audio or 40-Hz cache workers with explicit write locks.

Launch this coordinator with setsid. It retains every worker log, retries only
failed sample IDs, and reports success only after every requested ID succeeded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lrs3_build_inventory import digest, write_json, write_jsonl


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--stage', choices=['audio', 'video40'], required=True)
    parser.add_argument('--gpus', default='0,0,1,1,3,3', help='One entry per worker; CPU stage uses only the count.')
    parser.add_argument('--name', required=True)
    parser.add_argument('--retries', type=int, default=1)
    args = parser.parse_args()
    if '/' in args.name or args.name in {'.', '..'}:
        raise ValueError('Invalid run name')
    devices = args.gpus.split(',')
    project = Path(__file__).resolve().parents[1]
    run = args.root / 'reports' / args.name
    run.mkdir(parents=True, exist_ok=True)
    locks = args.root / 'state/locks'
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / f'{args.stage}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    features = ['latents', 'speaker_embeddings'] if args.stage == 'audio' else ['video_40hz']
    active_paths = [args.root / f'state/{feature}/WRITE_ACTIVE.json' for feature in features]
    all_rows = rows(args.manifest)
    by_id = {row['sample_id']: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError('Duplicate sample identities')
    state = {'stage': args.stage, 'pid': os.getpid(), 'started_unix': time.time(),
             'source_manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest),
             'requested': len(all_rows), 'succeeded': 0, 'complete': False, 'rounds': []}
    for path in active_paths:
        write_json(path, {'pid': os.getpid(), 'run': str(run), 'stage': args.stage})
    write_json(run / 'summary.json', state)
    succeeded = set()
    manifest = args.manifest
    for attempt in range(args.retries + 1):
        round_state = {'attempt': attempt, 'manifest': str(manifest), 'workers': []}
        state['rounds'].append(round_state)
        before = set((args.root / 'reports').glob(f'{args.stage}.*.summary.json'))
        processes = []
        for rank, device in enumerate(devices):
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=device if args.stage == 'audio' else '',
                               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               PYTHONPATH=str(project / 'src'))
            for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
                environment.pop(key, None)
            command = [sys.executable, '-u', str(project / 'scripts/lrs3_cache_worker.py'),
                       '--root', str(args.root), '--manifest', str(manifest), '--stage', args.stage,
                       '--rank', str(rank), '--world-size', str(len(devices)), '--threads', '2']
            log_path = run / f'round{attempt}.rank{rank}.log'
            log = log_path.open('w')
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(lock.fileno(),))
            processes.append((process, log))
            round_state['workers'].append({'rank': rank, 'device': device, 'pid': process.pid,
                                           'log': str(log_path), 'command': command})
        write_json(run / 'summary.json', state)
        while any(process.poll() is None for process, _ in processes):
            for worker, (process, _) in zip(round_state['workers'], processes):
                worker['returncode'] = process.poll()
            state['heartbeat_unix'] = time.time()
            write_json(run / 'summary.json', state)
            time.sleep(10)
        for worker, (process, log) in zip(round_state['workers'], processes):
            worker['returncode'] = process.returncode
            log.close()
        expected_hash = digest(manifest)
        summaries = set((args.root / 'reports').glob(f'{args.stage}.*.summary.json')) - before
        for summary_path in sorted(summaries):
            summary = json.loads(summary_path.read_text())
            if summary['manifest_sha256'] != expected_hash:
                raise RuntimeError('Unexpected concurrent worker summary')
            for result in rows(Path(summary['progress_path'])):
                if result['status'] == 'ok':
                    if result['sample_id'] not in by_id:
                        raise RuntimeError('Worker returned an unknown identity')
                    succeeded.add(result['sample_id'])
        round_state['summaries'] = [str(path) for path in sorted(summaries)]
        pending = [row for row in all_rows if row['sample_id'] not in succeeded]
        state.update(succeeded=len(succeeded), failed=len(pending))
        write_json(run / 'summary.json', state)
        if not pending:
            break
        manifest = run / f'retry{attempt + 1}.jsonl'
        write_jsonl(manifest, pending)
        print(json.dumps({'stage': args.stage, 'attempt': attempt, 'succeeded': len(succeeded),
                          'remaining': len(pending)}), flush=True)
    state.update(complete=len(succeeded) == len(all_rows), finished_unix=time.time())
    write_json(run / 'summary.json', state)
    print(json.dumps(state, indent=2), flush=True)
    if not state['complete']:
        raise SystemExit(1)  # Keep WRITE_ACTIVE guards until a successful repair.
    for path in active_paths:
        path.unlink()


if __name__ == '__main__':
    main()
