#!/usr/bin/env python3
"""Repair terminal crop failures from one active AV attempt in bounded FAN batches.

This optional overlap worker never changes AV outputs or quality thresholds.
Only failed rows already emitted by the *current* regular attempt are eligible.
The formal pipeline shares fan_repair.lock and later reuses verified repairs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lrs3_media_worker import atomic_json, sha256
from lrs3_media_worker_pipeline import save_manifest


SCRIPTS = Path(__file__).resolve().parent
CROP_ERRORS = ("landmark", "mouth roi", "alignment transform", ".pkl")


def current_attempt(state, token):
    stage = state.get("stages", {}).get("full_media_regular", {})
    return (state.get("status") == "running" and state.get("active_stage") == "full_media_regular"
            and not stage.get("finished") and stage.get("attempt_token") == token)


def read_complete_rows(path):
    """A writer may be between write() and newline; ignore only its final fragment."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def eligible_failures(state, source_rows, attempted):
    failed = {}
    for report in state["stages"]["full_media_regular"]["reports"]:
        for result in read_complete_rows(report):
            key = result.get("relative_stem")
            source = source_rows.get(key)
            if (key in attempted or source is None or source["width"] != 224
                    or result.get("status") != "failed"
                    or not any(word in result.get("error", "").lower() for word in CROP_ERRORS)):
                continue
            if result.get("sample_id") != source["sample_id"]:
                raise ValueError(f"Report/source identity mismatch: {key}")
            failed[key] = result
    return failed


def run_batch(args, root, directory, manifest, rows, fan_fd, coordinator_fd):
    batch_dir = directory / manifest.stem
    batch_dir.mkdir(parents=True, exist_ok=False)
    gpu_map = args.gpus.split(",")[:len(rows)]
    children = []
    for rank, gpu in enumerate(gpu_map):
        command = [sys.executable, "-u", str(SCRIPTS / "lrs3_media_worker_repair.py"),
                   "--manifest", str(manifest), "--output-root", str(root),
                   "--num-shards", str(len(gpu_map)), "--shard", str(rank),
                   "--cpu-threads", str(args.cpu_threads)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(args.cpu_threads),
                   MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        log_path = batch_dir / f"rank{rank}.log"
        log = log_path.open("w")
        process = subprocess.Popen(command, env=env, cwd=SCRIPTS.parent, stdout=log,
                                   stderr=subprocess.STDOUT, pass_fds=(fan_fd, coordinator_fd))
        suffix = f"_shard{rank}" if len(gpu_map) > 1 else ""
        report = root / "reports" / f"media_fan_repair_{manifest.stem}{suffix}.jsonl"
        receipt = {"pid": process.pid, "sid": os.getsid(process.pid), "gpu": gpu,
                   "rank": rank, "command": command, "log": str(log_path), "report": str(report)}
        children.append((process, log, receipt))
        atomic_json(batch_dir / "processes.json", [entry for _, _, entry in children])
    # Wait for every child before evaluating failures; locks remain inherited even
    # if this coordinator is interrupted or killed.
    while any(process.poll() is None for process, _, _ in children):
        time.sleep(2)
    totals = Counter()
    for process, log, receipt in children:
        receipt["exit_code"] = process.returncode
        log.close()
        report = Path(receipt["report"])
        summary = json.loads(report.with_suffix(".summary.json").read_text())
        expected = {row["relative_stem"] for row in rows[receipt["rank"]::len(gpu_map)]}
        results = {row["relative_stem"]: row for row in read_complete_rows(report)}
        counts = Counter(row["status"] for row in results.values())
        if (set(results) != expected or summary.get("requested") != len(expected)
                or summary.get("succeeded") != counts["ok"] or summary.get("failed") != counts["failed"]
                or process.returncode != int(counts["failed"] > 0)):
            raise RuntimeError(f"Early FAN child coverage/completion mismatch: {receipt}")
        totals.update(counts)
    atomic_json(batch_dir / "processes.json", [entry for _, _, entry in children])
    summary = {"requested": len(rows), "succeeded": totals["ok"], "failed": totals["failed"],
               "finished_at": time.time(), "manifest": str(manifest), "manifest_sha256": sha256(manifest)}
    atomic_json(batch_dir / "summary.json", summary)
    return summary


def main(args):
    root = Path(args.output_root).resolve()
    source_path = Path(args.source_manifest).resolve()
    state_path = root / "reports" / "media_full_pipeline" / "state.json"
    lock_dir = root / "state" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "early_fan_coordinator.lock").open("a") as coordinator_lock:
        fcntl.flock(coordinator_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(state_path.read_text())
        if current.get("manifest_sha256") != sha256(source_path):
            raise ValueError("Early FAN source manifest differs from the active media pipeline")
        if not current_attempt(current, args.attempt_token):
            raise ValueError("Requested regular AV attempt is not currently running")
        source = {}
        with source_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["width"] == 224:
                    if row["relative_stem"] in source:
                        raise ValueError("Duplicate source identities")
                    source[row["relative_stem"]] = row
        directory = root / "reports" / "media_early_fan" / args.attempt_token
        directory.mkdir(parents=True, exist_ok=True)
        own_state_path = directory / "state.json"
        own = json.loads(own_state_path.read_text()) if own_state_path.exists() else {"batches": [], "attempted": []}
        attempted = set(own["attempted"])
        own.update(status="running", coordinator_pid=os.getpid(), coordinator_sid=os.getsid(0),
                   attempt_token=args.attempt_token, started_at=time.time(), source_manifest=str(source_path),
                   source_manifest_sha256=sha256(source_path), gpus=args.gpus, batch_size=args.batch_size)
        atomic_json(own_state_path, own)
        try:
            while True:
                current = json.loads(state_path.read_text())
                if not current_attempt(current, args.attempt_token):
                    own.update(status="finished_collection", finished_at=time.time(),
                               stop_reason="regular_attempt_finished_or_changed")
                    break
                # Avoid working from stale state after an unexpected coordinator
                # failure. Normal pipeline updates occur every five seconds.
                if time.time() - current.get("updated_at", 0) > 90:
                    raise RuntimeError("Active media coordinator state has not advanced for 90 seconds")
                failed = eligible_failures(current, source, attempted)
                if not failed:
                    time.sleep(args.poll_seconds)
                    continue
                with (lock_dir / "fan_repair.lock").open("a") as fan_lock:
                    try:
                        fcntl.flock(fan_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(args.poll_seconds)
                        continue
                    # The formal stage may have advanced while we waited for the
                    # lock. It must never be followed by a new early repair batch.
                    current = json.loads(state_path.read_text())
                    if not current_attempt(current, args.attempt_token):
                        continue
                    failed = eligible_failures(current, source, attempted)
                    keys = sorted(failed)[:args.batch_size]
                    if not keys:
                        continue
                    token = f"early_fan_{args.attempt_token}_b{time.time_ns()}"
                    manifest = root / "manifests" / "media_early_fan" / (token + ".jsonl")
                    rows = [source[key] for key in keys]
                    save_manifest(manifest, rows)
                    save_manifest(manifest.with_suffix(".eligibility.jsonl"), [failed[key] for key in keys])
                    own.update(active_batch=str(manifest), updated_at=time.time())
                    atomic_json(own_state_path, own)
                    result = run_batch(args, root, directory, manifest, rows,
                                       fan_lock.fileno(), coordinator_lock.fileno())
                    own["batches"].append(result)
                    attempted.update(keys)
                    own.update(attempted=sorted(attempted), active_batch=None, updated_at=time.time())
                    atomic_json(own_state_path, own)
                    print(json.dumps(result), flush=True)
                if args.max_batches and len(own["batches"]) >= args.max_batches:
                    own.update(status="finished_collection", finished_at=time.time(), stop_reason="batch_limit")
                    break
        except Exception as error:
            own.update(status="failed", error=repr(error), finished_at=time.time())
            atomic_json(own_state_path, own)
            raise
        own.update(requested=sum(batch["requested"] for batch in own["batches"]),
                   succeeded=sum(batch["succeeded"] for batch in own["batches"]),
                   failed=sum(batch["failed"] for batch in own["batches"]))
        atomic_json(own_state_path, own)
        print(json.dumps(own), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--attempt-token", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Stop after this many batches; zero follows the entire regular attempt")
    args = parser.parse_args()
    if (not args.attempt_token.startswith("a") or not args.attempt_token[1:].isdigit()
            or args.batch_size < 1 or args.cpu_threads < 1 or args.poll_seconds <= 0 or args.max_batches < 0
            or not all(gpu.isdigit() for gpu in args.gpus.split(","))):
        parser.error("Invalid attempt token, GPU list, batch size, thread count, or polling interval")
    main(args)
