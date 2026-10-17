#!/usr/bin/env python
"""Run extraction workers with failure propagation and host-memory limits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0.0


def cgroup_memory_fraction() -> float | None:
    root = Path("/sys/fs/cgroup")
    try:
        limit = (root / "memory.max").read_text().strip()
        if limit == "max":
            return None
        return int((root / "memory.current").read_text()) / int(limit)
    except (OSError, ValueError, ZeroDivisionError):
        return None


def supervise(commands: list[tuple[list[str], dict[str, str], Path]], *, max_rss_mib: float) -> None:
    processes: list[subprocess.Popen] = []
    handles = []
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Extraction supervisor received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupted)
    try:
        for rank, (command, env, path) in enumerate(commands):
            handle = path.open("w")
            handles.append(handle)
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            print(f"Synchformer rank={rank} gpu={env.get('CUDA_VISIBLE_DEVICES', '')} pid={process.pid} log={path}", flush=True)
        last_report = 0.0
        while True:
            active_rss = []
            for rank, process in enumerate(processes):
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"rank {rank} (PID {process.pid}) exited with code {code}")
                    continue
                resident = rss_mib(process.pid)
                active_rss.append(resident)
                if resident > max_rss_mib:
                    raise MemoryError(f"rank {rank} RSS {resident:.0f} MiB exceeds {max_rss_mib:.0f} MiB guard")
            if not active_rss:
                return
            pressure = cgroup_memory_fraction()
            if pressure is not None and pressure >= 0.90:
                raise MemoryError(f"cgroup memory usage {pressure:.1%} reached the 90% guard")
            if time.monotonic() - last_report >= 30:
                print(json.dumps({"active_workers": len(active_rss), "rss_sum_mib": round(sum(active_rss)), "rss_max_mib": round(max(active_rss)), "cgroup_fraction": pressure}), flush=True)
                last_report = time.monotonic()
            time.sleep(2)
    finally:
        # Only terminate workers created by this supervisor; completed cache files remain reusable.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--workers-per-gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-worker-rss-mib", type=float, default=6144)
    parser.add_argument("extract_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if min(args.workers_per_gpu, args.batch_size, args.num_threads, args.max_worker_rss_mib) <= 0:
        parser.error("worker, batch, thread and RSS limits must be positive")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",")]
    if not all(gpu_ids):
        parser.error("--gpus must contain nonempty GPU IDs")
    worker_gpus = [gpu for gpu in gpu_ids for _ in range(args.workers_per_gpu)]
    extra = args.extract_args[1:] if args.extract_args[:1] == ["--"] else args.extract_args
    args.log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("extract_synchformer.py")
    commands = []
    for rank, gpu in enumerate(worker_gpus):
        command = [sys.executable, "-u", str(script), "--device", "cuda:0", "--batch-size", str(args.batch_size), "--rank", str(rank), "--world-size", str(len(worker_gpus)), "--num-threads", str(args.num_threads), *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        commands.append((command, env, args.log_dir / f"rank{rank}.log"))
    try:
        supervise(commands, max_rss_mib=args.max_worker_rss_mib)
    except (RuntimeError, MemoryError, InterruptedError) as error:
        raise SystemExit(f"Extraction stopped: {error}. All remaining workers stopped; rerun to resume validated caches.") from error


if __name__ == "__main__":
    main()
