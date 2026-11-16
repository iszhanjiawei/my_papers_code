#!/usr/bin/env python3
"""Measure one/two independent AV-HuBERT workers on one visible GPU.

Uses real source videos and the production extractor, without writing or changing
training feature caches. Model startup is excluded by a shared readiness barrier.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from lrs3_media_worker import atomic_json, load_reader


def child(args):
    import torch
    cv2.setNumThreads(args.cpu_threads)
    directory = Path(args.report_dir)
    reader = load_reader(args)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    # Warm a real maximum-length input before synchronizing startup.
    longest = max(rows, key=lambda row: row["video_frames"])
    reader.get_feats((longest["video_path"], None))
    torch.cuda.synchronize()
    (directory / f"ready_{args.rank}").write_text(str(time.time()))
    deadline = time.monotonic() + 180
    while not all((directory / f"ready_{rank}").exists() for rank in range(args.workers)):
        if time.monotonic() > deadline:
            raise TimeoutError("A benchmark worker did not reach the ready barrier")
        time.sleep(.05)
    tasks = (rows * args.repeats)[args.rank::args.workers]
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    durations = []
    frames = 0
    for row in tasks:
        begin = time.monotonic()
        features = reader.get_feats((row["video_path"], None)).detach().cpu().numpy()
        if features.shape != (row["video_frames"], 1024) or not np.isfinite(features).all():
            raise ValueError("Benchmark feature validation failed")
        durations.append(time.monotonic() - begin)
        frames += row["video_frames"]
    torch.cuda.synchronize()
    finish = time.time()
    atomic_json(directory / f"worker_{args.rank}.json", {
        "rank": args.rank, "samples": len(tasks), "video_frames": frames,
        "started_at": start, "finished_at": finish, "elapsed_seconds": finish - start,
        "mean_sample_seconds": float(np.mean(durations)),
        "max_sample_seconds": float(np.max(durations)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "longest_input_frames": longest["video_frames"], "cpu_threads": args.cpu_threads})


def main(args):
    root = Path(args.report_dir)
    summaries = []
    for workers in [1, 2]:
        directory = root / f"workers_{workers}"
        directory.mkdir(parents=True, exist_ok=True)
        for ready in directory.glob("ready_*"):
            ready.unlink()
        children = []
        logs = []
        for rank in range(workers):
            command = [sys.executable, "-u", __file__, "--child", "--workers", str(workers),
                       "--rank", str(rank), "--manifest", args.manifest, "--report-dir", str(directory),
                       "--checkpoint", args.checkpoint, "--fairseq-root", args.fairseq_root,
                       "--user-dir", args.user_dir, "--repeats", str(args.repeats),
                       "--cpu-threads", str(args.cpu_threads)]
            log = (directory / f"worker_{rank}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in children]
        for log in logs:
            log.close()
        if any(codes):
            raise RuntimeError(f"Benchmark workers failed: {codes}, see {directory}")
        results = [json.loads((directory / f"worker_{rank}.json").read_text()) for rank in range(workers)]
        elapsed = max(r["finished_at"] for r in results) - min(r["started_at"] for r in results)
        total_frames = sum(r["video_frames"] for r in results)
        summary = {"workers": workers, "samples": sum(r["samples"] for r in results),
                   "elapsed_seconds": elapsed, "video_hours": total_frames / 25 / 3600,
                   "frames_per_second": total_frames / elapsed,
                   "realtime_multiple": total_frames / 25 / elapsed,
                   "sum_peak_reserved_mib": sum(r["peak_reserved_mib"] for r in results),
                   "worker_results": results}
        summaries.append(summary)
        atomic_json(root / "summary.json", summaries)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--fairseq-root", default=prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
    parser.add_argument("--user-dir", default=prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--child", action="store_true")
    arguments = parser.parse_args()
    child(arguments) if arguments.child else main(arguments)
