"""Resume a reviewed LRS3 cutover only after its ORIGINAL locks become free.

This waiter never deletes/replaces locks or guards. It launches the new cache
coordinator first, verifies its normal guard and worker registration, then
launches the finisher. Existing interrupted outputs are not marked complete.
The reviewed JSON plan and source revision are mandatory explicit pins.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def original_locks_free(root, identities):
    """Probe both existing inodes; unlock only descriptors we acquired."""
    with ExitStack() as stack:
        for name, expected in zip(('finish', 'video40'), identities):
            # r+ deliberately refuses to create a missing/replaced lock file.
            handle = stack.enter_context((root / f'state/locks/{name}.lock').open('r+'))
            actual = os.fstat(handle.fileno())
            if (actual.st_dev, actual.st_ino) != expected:
                raise RuntimeError('Opened lock descriptor is not the archived original inode')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            stack.callback(fcntl.flock, handle, fcntl.LOCK_UN)
    return True


def validate_guard(summary, guard, plan, pid, expected_count):
    expected_run = Path(plan['expected_cache_summary']).parent
    if (summary.get('pid') != pid or summary.get('stage') != 'video40'
            or summary.get('manifest_sha256') != plan['source_manifest_sha256']
            or summary.get('media_quality_terminal_sha256') != plan['media_quality_terminal_sha256']
            or summary.get('requested') != expected_count
            or guard != {'pid': pid, 'run': str(expected_run), 'stage': 'video40'}):
        raise RuntimeError('New cache coordinator/guard differs from the reviewed cutover')
    workers = (summary.get('rounds') or [{}])[0].get('workers', [])
    expected = len(plan['new_cache_command'][plan['new_cache_command'].index('--gpus') + 1].split(','))
    if len(workers) != expected:
        return False
    if {worker['rank'] for worker in workers} != set(range(expected)):
        raise RuntimeError('New cache worker registration is incomplete or duplicated')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--approval-receipt', type=Path, required=True)
    parser.add_argument('--lock-evidence', type=Path, required=True)
    parser.add_argument('--lock-evidence-sha256', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('Reviewed cutover plan changed')
    plan = json.loads(args.plan.read_bytes())
    approval = json.loads(args.approval_receipt.read_bytes())
    if approval.get('root_approved') is not True or approval.get('plan_sha256') != args.plan_sha256:
        raise ValueError('Cutover lacks its exact execution receipt')
    if Path(plan['working_directory']).resolve() != PROJECT:
        raise ValueError('Cutover belongs to another project')
    for key, script in [('new_cache_command', 'lrs3_cache_parallel.py'),
                        ('new_finisher_command', 'lrs3_finish_production.py')]:
        command = plan[key]
        if Path(command[2]).resolve() != PROJECT / 'scripts' / script:
            raise ValueError('Unexpected cutover entry point')
    root = Path(plan['new_cache_command'][plan['new_cache_command'].index('--root') + 1]).resolve(strict=True)
    expected_count = json.loads((root / 'reports/source_contract.json').read_bytes())['source_media_pairs']
    lock_paths = [root / f'state/locks/{name}.lock' for name in ('finish', 'video40')]
    if digest(args.lock_evidence) != args.lock_evidence_sha256:
        raise ValueError('Archived original lock identity evidence changed')
    evidence = json.loads(args.lock_evidence.read_bytes())
    device = os.makedev(evidence['filesystem']['device_major'], evidence['filesystem']['device_minor'])
    identities = [(device, evidence['lock_inodes'][path.name]) for path in lock_paths]
    old_guard_path = root / 'state/video_40hz/WRITE_ACTIVE.json'
    old_guard_hash = approval['old_snapshots']['old_guard']['sha256']
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists():
        raise FileExistsError('Use a new waiter state path; preserve prior handoff evidence')
    singleton = args.state.with_suffix('.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopping.append(number))
    state = {'pid': os.getpid(), 'sid': os.getsid(0), 'started_unix': time.time(),
             'plan_sha256': args.plan_sha256, 'source_revision': args.source_revision,
             'lock_evidence_sha256': args.lock_evidence_sha256,
             'original_lock_identities': identities, 'status': 'waiting_for_original_locks',
             'cache_started': False, 'finisher_started': False}
    def save(**updates):
        state.update(updates, heartbeat_unix=time.time())
        write(args.state, state)
    def check_inputs():
        if [(p.stat().st_dev, p.stat().st_ino) for p in lock_paths] != identities:
            raise RuntimeError('Original lock inode changed; automatic cutover refused')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
        if revision != args.source_revision:
            raise RuntimeError('Project revision changed; review a new handoff')
        changed = subprocess.check_output(['git', 'status', '--porcelain', '--', 'scripts', 'src'],
                                         cwd=PROJECT, text=True)
        if changed.strip():
            raise RuntimeError('Project source has uncommitted changes')
        if digest(args.plan) != args.plan_sha256:
            raise RuntimeError('Reviewed cutover plan changed while waiting')
        if digest(args.lock_evidence) != args.lock_evidence_sha256:
            raise RuntimeError('Archived original lock evidence changed while waiting')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if stat[0] != 'Z' and int(stat[3]) == plan['old_session_id']:
                raise RuntimeError('Old production session still has a live process')
    try:
        save()
        while not stopping:
            check_inputs()
            if original_locks_free(root, identities):
                break
            save()
            time.sleep(args.poll_seconds)
        if stopping:
            save(status='waiter_stopped_before_launch', signal=stopping[0])
            return
        check_inputs()
        if digest(old_guard_path) != old_guard_hash:
            raise RuntimeError('Old writer guard changed; investigate the intervening writer')
        if Path(plan['expected_cache_summary']).exists() or Path(plan['expected_new_finisher_state']).exists():
            raise RuntimeError('New cutover run already exists; refusing duplicate launch')
        if digest(root / 'manifests/source_inventory.jsonl') != plan['source_manifest_sha256']:
            raise RuntimeError('Canonical source changed')
        if digest(root / 'reports/media_full_pipeline/quality_terminal.json') != plan['media_quality_terminal_sha256']:
            raise RuntimeError('Reviewed media quality closure changed')
        env = dict(os.environ, **plan['environment'])
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key, None)
        log_path = args.state.parent / 'new_cache.log'
        with log_path.open('x') as log:
            cache = subprocess.Popen(plan['new_cache_command'], cwd=PROJECT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(status='waiting_for_new_cache_guard', cache_started=True, cache_pid=cache.pid,
             cache_sid=os.getsid(cache.pid), cache_log=str(log_path))
        deadline = time.monotonic() + 600
        while not stopping:
            if cache.poll() is not None:
                raise RuntimeError(f'New cache exited before handoff: {cache.returncode}')
            summary_path = Path(plan['expected_cache_summary'])
            if summary_path.exists():
                summary = json.loads(summary_path.read_bytes())
                guard = json.loads(old_guard_path.read_bytes())
                if validate_guard(summary, guard, plan, cache.pid, expected_count):
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError('New cache did not register its guard/workers within 600 seconds')
            time.sleep(2)
        if stopping:
            save(status='waiter_stopped_cache_continues', signal=stopping[0])
            return
        check_inputs()
        log_path = args.state.parent / 'new_finisher.log'
        with log_path.open('x') as log:
            finisher = subprocess.Popen(plan['new_finisher_command'], cwd=PROJECT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(status='handoff_started', finisher_started=True, finisher_pid=finisher.pid,
             finisher_sid=os.getsid(finisher.pid), finisher_log=str(log_path))
        # Follow startup to an actual persisted state; both writers remain in
        # their own sessions after the waiter exits. Completion is NOT implied.
        deadline = time.monotonic() + 180
        while not Path(plan['expected_new_finisher_state']).exists():
            if finisher.poll() is not None:
                raise RuntimeError(f'New finisher exited during startup: {finisher.returncode}')
            if time.monotonic() >= deadline:
                raise RuntimeError('New finisher did not publish its startup state')
            time.sleep(2)
        started = json.loads(Path(plan['expected_new_finisher_state']).read_bytes())
        if started.get('pid') != finisher.pid or started.get('status') != 'running':
            raise RuntimeError('New finisher state does not match its process')
        save(status='handoff_complete_production_running', finished_unix=time.time())
    except BaseException as error:
        save(status='failed_requires_review', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        fcntl.flock(singleton, fcntl.LOCK_UN)
        singleton.close()


if __name__ == '__main__':
    main()
