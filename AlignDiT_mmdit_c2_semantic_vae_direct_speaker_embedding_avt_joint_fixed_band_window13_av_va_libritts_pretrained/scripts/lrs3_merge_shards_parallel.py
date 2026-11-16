"""Resume a physical LRS3 merge with isolated, non-overlapping worker processes.

One coordinator proves the complete canonical partition and publishes the sole
ledger. Children own individual shard locks, inherit no controller locks, and
check every copied asset before linking its existing immutable inode. Existing
matching destination files are verified and retained on resume.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import lrs3_sharded_import as imp


STATE = imp.RUN + '/parallel_merge_state.json'


def collect_partition(root, plan, fingerprint):
    canonical_rows = imp.read_rows(root / imp.ARTIFACTS['source_manifest_sha256'])
    canonical = {row['sample_id']: row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError('Duplicate canonical source identity')
    all_records, seen, references = [], set(), []
    for index in range(plan['num_shards']):
        imp.check_stop()
        directory, _, complete_path = imp.shard_paths(root, index)
        # Only the compact manifests are checked here. Each locked child checks
        # every per-ID receipt and asset; the parent never scans them serially.
        with imp.owned_lock(directory / 'worker.lock'):
            if (directory / 'WRITE_ACTIVE.json').exists():
                raise ValueError('An import shard is still being written')
            complete = imp.read_json(complete_path)
            records_path = directory / 'records.jsonl'
            rows, _, _ = imp.shard_inputs(root, plan, index)
            records = imp.read_rows(records_path)
            ids = {row['sample_id'] for row in rows}
            record_ids = {row['sample_id'] for row in records}
            if (ids & seen or len(record_ids) != len(records) or record_ids != ids
                    or complete.get('policy') != imp.POLICY
                    or complete.get('plan_sha256') != fingerprint
                    or complete.get('import_contract_sha256') != plan['import_contract_sha256']
                    or complete.get('shard') != index or complete.get('failed') != 0
                    or complete.get('requested') != len(rows) or complete.get('processed') != len(rows)
                    or complete.get('records_sha256') != imp.digest(records_path)):
                raise ValueError('Shard completion or partition differs from the immutable plan')
            if any(canonical.get(row['sample_id']) != row for row in rows):
                raise ValueError('Shard row differs from canonical source')
            seen.update(ids)
            all_records.extend(records)
            references.append({'index': index, 'count': len(rows),
                               'completion_sha256': imp.digest(complete_path),
                               'records_sha256': imp.digest(records_path)})
    if (len(seen) != plan['requested']
            or plan['scope'] == 'full' and seen != canonical.keys()):
        raise ValueError('Shard partition is not the exact requested canonical ID set')
    return all_records, references


def worker(root, fingerprint, index):
    plan, fingerprint, contract = imp.get_plan(root, fingerprint)
    forbidden = {str(root / imp.RUN / name) for name in ('coordinator.lock', 'merge.lock')}
    for fd in Path('/proc/self/fd').glob('*'):
        try:
            if str(fd.readlink()) in forbidden:
                raise ValueError('Merge child inherited a controller lock descriptor')
        except FileNotFoundError:
            pass
    directory, data_root, completion_path = imp.shard_paths(root, index)
    with imp.owned_lock(directory / 'worker.lock'):
        rows, records = imp.check_shard(root, plan, fingerprint, index, validate_files=False)
        by_id = {row['sample_id']: row for row in rows}
        exclusions = imp.read_json(root / plan['exclusions'])
        guard = directory / 'MERGE_ACTIVE.json'
        if guard.exists() and imp.read_json(guard).get('plan_sha256') != fingerprint:
            raise ValueError('Unrecognized previous merge guard')
        imp.write_json(guard, {'pid': os.getpid(), 'plan_sha256': fingerprint})
        for position, record in enumerate(records, 1):
            imp.check_stop()
            imp.merge_one(root, data_root, by_id[record['sample_id']], record,
                          contract['_sha256'], exclusions.get(record['sample_id']))
            if position % 50 == 0:
                imp.write_json(directory / 'merge_progress.json',
                    {'processed': position, 'requested': len(records), 'pid': os.getpid(),
                     'plan_sha256': fingerprint, 'heartbeat_unix': time.time()})
        imp.check_stop()
        imp.get_plan(root, fingerprint)
        result = {'status': 'complete', 'plan_sha256': fingerprint, 'shard': index,
                  'requested': len(records), 'processed': len(records), 'failed': 0,
                  'completion_sha256': imp.digest(completion_path),
                  'records_sha256': imp.digest(directory / 'records.jsonl'),
                  'controller_lock_fds_inherited': False, 'finished_unix': time.time()}
        imp.write_json(directory / 'merge_completion.json', result)
        guard.unlink()


def merge(root, fingerprint, workers):
    if workers < 1:
        raise ValueError('Worker count must be positive')
    plan, fingerprint, contract = imp.get_plan(root, fingerprint)
    with ExitStack() as locks:
        for name in ('coordinator.lock', 'merge.lock'):
            locks.enter_context(imp.owned_lock(root / imp.RUN / name))
        if (root / imp.COMPLETION).exists():
            raise FileExistsError('Merged import is already immutable')
        state = {'status': 'validating_partition', 'pid': os.getpid(), 'sid': os.getsid(0),
                 'plan_sha256': fingerprint, 'workers': workers, 'active': [], 'finished': [],
                 'started_unix': time.time()}
        imp.write_json(root / STATE, state)
        try:
            imp.validate_imported_source(root, contract['_sha256'])
            records, references = collect_partition(root, plan, fingerprint)
            guard = root / 'state/video_40hz/WRITE_ACTIVE.json'
            if guard.exists() and imp.read_json(guard).get('plan_sha256') != fingerprint:
                raise ValueError('Cannot resume a merge for another plan')
            imp.write_json(guard, {'pid': os.getpid(), 'stage': 'parallel_import_merge',
                                  'plan_sha256': fingerprint})
        except BaseException as error:
            state.update(status='stopped_or_failed', error=f'{type(error).__name__}: {error}',
                         finished_unix=time.time())
            imp.write_json(root / STATE, state)
            raise
        state['status'] = 'merging'
        todo, active, failures = list(range(plan['num_shards'])), {}, []
        try:
            while todo or active:
                imp.check_stop()
                while todo and len(active) < workers:
                    imp.check_stop()
                    index = todo.pop(0)
                    log_path = root / imp.RUN / f'merge{index:04d}.{time.time_ns()}.log'
                    log = log_path.open('x')
                    command = [sys.executable, '-u', str(Path(__file__).resolve()),
                               '--root', str(root), '--plan-sha256', fingerprint,
                               '--worker-shard', str(index)]
                    try:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, close_fds=True)
                    except BaseException:
                        log.close()
                        raise
                    active[index] = child, log, str(log_path)
                for index, (child, log, log_path) in list(active.items()):
                    code = child.poll()
                    if code is not None:
                        child.wait(); log.close(); active.pop(index)
                        state['finished'].append({'shard': index, 'pid': child.pid,
                                                  'returncode': code, 'log': log_path})
                        if code != 0:
                            failures.append(index)
                state.update(active=[{'shard': i, 'pid': p.pid, 'log': log_path}
                                     for i, (p, _, log_path) in active.items()],
                             heartbeat_unix=time.time())
                imp.write_json(root / STATE, state)
                if failures:
                    raise RuntimeError(f'Merge shards failed: {failures}; outputs retained for review')
                if active:
                    time.sleep(.5)
            # Every child is reaped. Reacquiring each shard lock also rules out
            # a still-running owner from an interrupted earlier coordinator.
            for reference in references:
                directory, _, completed = imp.shard_paths(root, reference['index'])
                with imp.owned_lock(directory / 'worker.lock'):
                    merged = imp.read_json(directory / 'merge_completion.json')
                    if ((directory / 'MERGE_ACTIVE.json').exists()
                            or merged.get('status') != 'complete' or merged.get('failed') != 0
                            or merged.get('plan_sha256') != fingerprint
                            or merged.get('shard') != reference['index']
                            or merged.get('processed') != reference['count']
                            or merged.get('requested') != reference['count']
                            or merged.get('controller_lock_fds_inherited') is not False
                            or merged.get('completion_sha256') != reference['completion_sha256']
                            or imp.digest(completed) != reference['completion_sha256']
                            or merged.get('records_sha256') != reference['records_sha256']
                            or imp.digest(directory / 'records.jsonl') != reference['records_sha256']):
                        raise ValueError('Merged shard closure changed or is incomplete')
            imp.check_stop()
            imp.validate_imported_source(root, contract['_sha256'])
            imp.get_plan(root, fingerprint)
            records.sort(key=lambda row: row['sample_id'])
            counts = {key: sum(row['status'] == value for row in records)
                      for key, value in [('media_success', 'ok'), ('quality_excluded', 'quality_excluded')]}
            if (counts['quality_excluded'] != plan['quality_excluded']
                    or sum(counts.values()) != plan['requested']):
                raise ValueError('Merged quality accounting differs from the plan')
            imp.atomic_bytes(root / imp.LEDGER, b''.join(imp.payload(row) for row in records), identical_only=True)
            completion = {'schema_version': 1, 'policy': imp.POLICY, 'scope': plan['scope'],
                'status': 'complete', 'plan_sha256': fingerprint, 'import_contract_sha256': contract['_sha256'],
                'source_manifest_sha256': plan['source_manifest_sha256'], 'requested': plan['requested'],
                'processed': len(records), **counts, 'failed': 0,
                'copied_video40': sum(row['video40_action'] == 'copied' for row in records),
                'recomputed_video40': sum(row['video40_action'] == 'recomputed' for row in records),
                'ledger_path': imp.LEDGER, 'ledger_sha256': imp.digest(root / imp.LEDGER),
                'records_sha256': imp.digest(root / imp.LEDGER), 'total': len(records),
                'merge_strategy': 'nonoverlapping_shard_processes', 'merge_workers': workers,
                'finished_unix': time.time()}
            guard.unlink()
            imp.write_json(root / imp.COMPLETION, completion, identical_only=True)
            state.update(status='merged', active=[], finished_unix=time.time())
            imp.write_json(root / STATE, state)
            print(json.dumps(completion), flush=True)
        except BaseException as error:
            for child, _, _ in active.values():
                if child.poll() is None:
                    child.terminate()
            for index, (child, log, log_path) in active.items():
                code = child.wait(); log.close()
                state['finished'].append({'shard': index, 'pid': child.pid,
                                          'returncode': code, 'log': log_path,
                                          'stopped_by_coordinator': True})
            state.update(status='stopped_or_failed', active=[], error=f'{type(error).__name__}: {error}',
                         finished_unix=time.time())
            imp.write_json(root / STATE, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--worker-shard', type=int)
    args = parser.parse_args()
    root = args.root.absolute()
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError('Destination must be a physical absolute root')
    imp.install_signals()
    if args.worker_shard is None:
        merge(root, args.plan_sha256, args.workers)
    else:
        worker(root, args.plan_sha256, args.worker_shard)


if __name__ == '__main__':
    main()
