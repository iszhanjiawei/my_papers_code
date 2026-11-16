"""Run resumable LRS3 audio or 40-Hz cache workers with explicit write locks.

Launch this coordinator with setsid. It retains every worker log, retries only
failed sample IDs, and reports success only after every requested ID succeeded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from lrs3_build_inventory import digest, write_json, write_jsonl
from aligndit.lrs3_media_quality import TERMINAL, excluded_map, load_terminal


FINISH_FD_ENV = 'LRS3_INHERITED_FINISH_LOCK_FD'


class ShutdownRequested(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'Graceful shutdown requested by signal {signum}')


def process_table():
    """Local process identities; start times protect against recycled PIDs."""
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), int(fields[2]), fields[19], fields[0])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


class ChildSupervisor:
    """Own child lifetimes before unlocking an inherited flock description.

    Finisher children get separate sessions, so orphaned stage workers remain
    observable as group members even if their coordinator dies first. Cache
    workers stay in their coordinator's group and are individually reaped.
    """
    def __init__(self, *, isolated_children=False, grace_seconds=30):
        self.isolated_children = isolated_children
        self.grace_seconds = grace_seconds
        self.signum = None
        self.children = []
        self.known = {}
        self.groups = set()

    @contextmanager
    def signals(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        def requested(signum, _frame):
            if self.signum is None:
                self.signum = signum
        for sig in previous:
            signal.signal(sig, requested)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def check(self):
        if self.signum is not None:
            raise ShutdownRequested(self.signum)

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
        self.check()

    def spawn(self, command, **kwargs):
        self.check()
        if self.isolated_children:
            kwargs['start_new_session'] = True
        child = subprocess.Popen(command, **kwargs)
        self.children.append(child)  # Signals set a flag; registration cannot be interrupted.
        if self.isolated_children:
            self.groups.add(child.pid)
        self.live()
        self.check()
        return child

    def live(self):
        table = process_table()
        for child in self.children:
            child.poll()  # Reap direct children, including already exited workers.
            if child.pid in table and child.pid not in self.known:
                self.known[child.pid] = table[child.pid][2]
        for group in list(self.groups):
            if not any(value[1] == group for value in table.values()):
                self.groups.discard(group)  # Do not later mistake a recycled group ID for our child.
        members = {pid for pid, value in table.items() if value[1] in self.groups}
        frontier = {pid for pid, start in self.known.items() if pid in table and table[pid][2] == start}
        while frontier:
            found = {pid for pid, value in table.items() if value[0] in frontier} - members
            members.update(found)
            frontier = found
        for pid in members:
            self.known.setdefault(pid, table[pid][2])
        return {pid: table[pid] for pid, start in self.known.items()
                if pid in table and table[pid][2] == start and table[pid][3] not in {'Z', 'X'}}

    def _signal(self, signum):
        live = self.live()
        for pid in live:
            # Recheck the birth identity immediately before signalling.
            current = process_table().get(pid)
            if current and current[2] == self.known[pid] and current[3] not in {'Z', 'X'}:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass

    def stop_and_reap(self):
        started = time.monotonic()
        live = self.live()
        if live:
            # Give a stage coordinator time to reap workers and explicitly unlock.
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            deadline = time.monotonic() + self.grace_seconds
            while self.live() and time.monotonic() < deadline:
                time.sleep(.1)
            if self.live():
                self._signal(signal.SIGTERM)
                deadline = time.monotonic() + self.grace_seconds
                while self.live() and time.monotonic() < deadline:
                    time.sleep(.1)
            if self.live():
                self._signal(signal.SIGKILL)
            # Never explicitly unlock while a registered writer is still alive,
            # including an uninterruptible process. Keep polling with the FD open.
            while self.live():
                time.sleep(.2)
        for child in self.children:
            if child.poll() is None:
                child.wait()
        return {'all_registered_descendants_exited': True,
                'registered_pids': sorted(self.known),
                'children': [{'pid': child.pid, 'returncode': child.returncode} for child in self.children],
                'elapsed_seconds': time.monotonic() - started}


def inherited_finish_fd(root):
    value = os.environ.get(FINISH_FD_ENV)
    if value is None:
        return None
    fd = int(value)
    actual, expected = os.fstat(fd), (root / 'state/locks/finish.lock').stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Inherited finish descriptor does not refer to this cache finish lock')
    return fd


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--stage', choices=['audio', 'video40'], required=True)
    parser.add_argument('--gpus', default='0,0,1,1,3,3', help='One entry per worker; CPU stage uses only the count.')
    parser.add_argument('--name', required=True)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--media-quality-terminal', type=Path)
    parser.add_argument('--media-quality-terminal-sha256')
    args = parser.parse_args()
    if '/' in args.name or args.name in {'.', '..'}:
        raise ValueError('Invalid run name')
    return args


def run_cache(args, lock, children, context, finish_fd):
    children.check()
    devices = args.gpus.split(',')
    project = Path(__file__).resolve().parents[1]
    run = args.root / 'reports' / args.name
    run.mkdir(parents=True, exist_ok=True)
    features = ['latents', 'speaker_embeddings'] if args.stage == 'audio' else ['video_40hz']
    active_paths = [args.root / f'state/{feature}/WRITE_ACTIVE.json' for feature in features]
    all_rows = rows(args.manifest)
    by_id = {row['sample_id']: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError('Duplicate sample identities')
    quality = load_terminal(args.root) if args.stage == 'video40' else None
    if quality is not None or args.media_quality_terminal or args.media_quality_terminal_sha256:
        if (args.stage != 'video40' or quality is None or args.media_quality_terminal is None
                or args.media_quality_terminal.resolve() != (args.root / TERMINAL).resolve()
                or args.media_quality_terminal_sha256 != quality['_sha256']
                or digest(args.manifest) != quality['source_manifest_sha256']
                or len(all_rows) != quality['requested']):
            raise ValueError('Media quality exclusions require an explicit canonical full pass with matching hash')
    quality_excluded = excluded_map(quality)
    if not quality_excluded.keys() <= by_id.keys():
        raise ValueError('Quality exclusions are outside canonical video40 scope')
    state = {'stage': args.stage, 'pid': os.getpid(), 'started_unix': time.time(),
             'source_manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest),
             'requested': len(all_rows), 'succeeded': 0, 'complete': False, 'rounds': [],
             'media_quality_terminal_sha256': quality['_sha256'] if quality else None}
    context.update(state=state, summary=run / 'summary.json')
    children.check()
    for path in active_paths:
        write_json(path, {'pid': os.getpid(), 'run': str(run), 'stage': args.stage})
    write_json(run / 'summary.json', state)
    succeeded, excluded = set(), set()
    manifest = args.manifest
    for attempt in range(args.retries + 1):
        children.check()
        round_state = {'attempt': attempt, 'manifest': str(manifest), 'workers': []}
        state['rounds'].append(round_state)
        before = set((args.root / 'reports').glob(f'{args.stage}.*.summary.json'))
        processes = []
        for rank, device in enumerate(devices):
            children.check()
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=device if args.stage == 'audio' else '',
                               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               PYTHONPATH=str(project / 'src'))
            for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
                environment.pop(key, None)
            command = [sys.executable, '-u', str(project / 'scripts/lrs3_cache_worker.py'),
                       '--root', str(args.root), '--manifest', str(manifest), '--stage', args.stage,
                       '--rank', str(rank), '--world-size', str(len(devices)), '--threads', '2']
            if quality:
                command += ['--media-quality-terminal', str(args.root / TERMINAL),
                            '--media-quality-terminal-sha256', quality['_sha256']]
            log_path = run / f'round{attempt}.rank{rank}.log'
            log = log_path.open('w')
            context.setdefault('logs', []).append(log)
            passed_fds = (lock.fileno(),) + ((finish_fd,) if finish_fd is not None else ())
            process = children.spawn(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=passed_fds)
            processes.append((process, log))
            round_state['workers'].append({'rank': rank, 'device': device, 'pid': process.pid,
                                           'log': str(log_path), 'command': command})
        write_json(run / 'summary.json', state)
        while any(process.poll() is None for process, _ in processes):
            children.check()
            for worker, (process, _) in zip(round_state['workers'], processes):
                worker['returncode'] = process.poll()
            state['heartbeat_unix'] = time.time()
            write_json(run / 'summary.json', state)
            children.sleep(10)
        children.check()
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
                    if result['sample_id'] not in by_id or result['sample_id'] in quality_excluded:
                        raise RuntimeError('Worker returned an unknown identity')
                    succeeded.add(result['sample_id'])
                elif result['status'] == 'excluded':
                    expected = quality_excluded.get(result['sample_id'])
                    if (expected is None or result.get('reason') != 'media_quality:' + expected['reason']
                            or result.get('media_quality_terminal_sha256') != quality['_sha256']):
                        raise RuntimeError('Worker returned an unauthorized quality exclusion')
                    excluded.add(result['sample_id'])
        round_state['summaries'] = [str(path) for path in sorted(summaries)]
        pending = [row for row in all_rows if row['sample_id'] not in succeeded | excluded]
        state.update(succeeded=len(succeeded), quality_excluded=len(excluded), failed=len(pending))
        write_json(run / 'summary.json', state)
        if not pending:
            break
        manifest = run / f'retry{attempt + 1}.jsonl'
        write_jsonl(manifest, pending)
        print(json.dumps({'stage': args.stage, 'attempt': attempt, 'succeeded': len(succeeded),
                          'remaining': len(pending)}), flush=True)
    children.check()
    if children.live():
        raise RuntimeError('Workers left live descendants after cache processing')
    if excluded != quality_excluded.keys():
        raise RuntimeError('Final video40 exclusion set differs from the reviewed quality closure')
    state.update(complete=len(succeeded) == len(all_rows),
                 processing_complete=len(succeeded | excluded) == len(all_rows), finished_unix=time.time())
    write_json(run / 'summary.json', state)
    print(json.dumps(state, indent=2), flush=True)
    if not state['processing_complete']:
        raise SystemExit(1)  # Keep WRITE_ACTIVE guards until a successful repair.
    for path in active_paths:
        path.unlink()


def main():
    args = parse_args()
    locks = args.root / 'state/locks'
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / f'{args.stage}.lock').open('a')
    children, context = ChildSupervisor(), {}
    acquired = False
    finish_fd = None
    with children.signals():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            finish_fd = inherited_finish_fd(args.root)
            run_cache(args, lock, children, context, finish_fd)
        except BaseException as error:
            if 'state' in context:
                context['state'].update(complete=False, processing_complete=False,
                    status='interrupted' if isinstance(error, (ShutdownRequested, KeyboardInterrupt)) else 'failed',
                    error=f'{type(error).__name__}: {error}', stopped_unix=time.time())
                if isinstance(error, ShutdownRequested):
                    context['state']['interrupted_signal'] = error.signum
                write_json(context['summary'], context['state'])
            if isinstance(error, ShutdownRequested):
                raise SystemExit(128 + error.signum) from error
            raise
        finally:
            shutdown = children.stop_and_reap()
            try:
                for log in context.get('logs', []):
                    log.close()
                if 'state' in context:
                    context['state']['shutdown'] = shutdown
                    returns = {child.pid: child.returncode for child in children.children}
                    for round_state in context['state'].get('rounds', []):
                        for worker in round_state['workers']:
                            worker['returncode'] = returns.get(worker['pid'], worker.get('returncode'))
                    write_json(context['summary'], context['state'])
            finally:
                if acquired:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
                # This descriptor shares the parent's lock description. Only its
                # owner (the finisher) may explicitly unlock it after we have exited.
                if finish_fd is not None:
                    os.close(finish_fd)


if __name__ == '__main__':
    main()
