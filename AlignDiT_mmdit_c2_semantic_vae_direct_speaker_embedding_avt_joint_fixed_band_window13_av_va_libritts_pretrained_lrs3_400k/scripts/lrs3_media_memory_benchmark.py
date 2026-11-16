#!/usr/bin/env python3
"""Measure AV-HuBERT allocator growth on real varied-length mouth sequences.

Read-only production inputs; output contains measurements and equality checks,
never production receipts or features. Run multiple ranks with the same barrier
directory to measure concurrent workers after independent model warmup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import lrs3_media_worker as media


def main(args):
    directory = Path(args.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    media.DECODE_THREADS = args.decode_threads
    media.cv2.setNumThreads(args.cpu_threads)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    reader = media.load_reader(args)
    longest = max(rows, key=lambda row: row["video_frames"])
    warmup = None
    if args.warmup == "fixed":
        warmup = media.warmup_reader(reader, args.warmup_frames)
    elif args.warmup != "none":
        warm = longest if args.warmup == "longest" else rows[0]
        reader.get_feats((warm["video_mouth_path"], None)).detach().cpu()
        torch.cuda.synchronize()
    if args.cache_limit_mib and torch.cuda.memory_reserved() > args.cache_limit_mib * 1024**2:
        torch.cuda.empty_cache()
    if args.barrier_dir:
        barrier = Path(args.barrier_dir)
        barrier.mkdir(parents=True, exist_ok=True)
        (barrier / f"ready_{args.rank}").write_text(str(os.getpid()))
        deadline = time.monotonic() + 180
        while not all((barrier / f"ready_{rank}").exists() for rank in range(args.workers)):
            if time.monotonic() > deadline:
                raise TimeoutError("Another memory benchmark worker did not reach the barrier")
            time.sleep(.05)
    tasks = (rows * args.repeats)[args.rank::args.workers]
    trace_path = directory / f"rank{args.rank}.jsonl"
    started = time.time()
    total_frames = clears = 0
    peak_allocated = peak_reserved = 0
    min_free = float("inf")
    try:
        with trace_path.open("w", buffering=1) as stream:
            for index, row in enumerate(tasks):
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                media.read_frames(row["video_mouth_path"], row["video_frames"], (96, 96))
                array = reader.get_feats((row["video_mouth_path"], None)).detach().cpu().numpy().astype(np.float32)
                if args.synchronize_step:
                    torch.cuda.synchronize()
                extract_seconds = time.perf_counter() - begin
                expected = np.load(row["video_feature_path"], allow_pickle=False, mmap_mode="r")
                actual_sha = hashlib.sha256(array.tobytes()).hexdigest()
                expected_sha = hashlib.sha256(expected.tobytes()).hexdigest()
                if array.shape != expected.shape or actual_sha != expected_sha:
                    raise ValueError(f"Production AV feature equality failed: {row['sample_id']}")
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                clip_peak = torch.cuda.max_memory_allocated() / 1024**2
                reserved_peak = torch.cuda.max_memory_reserved() / 1024**2
                free = torch.cuda.mem_get_info()[0] / 1024**2
                peak_allocated = max(peak_allocated, clip_peak)
                peak_reserved = max(peak_reserved, reserved_peak)
                min_free = min(min_free, free)
                clear_seconds = 0
                if args.cache_limit_mib and reserved > args.cache_limit_mib:
                    before_clear = time.perf_counter()
                    torch.cuda.empty_cache()
                    clear_seconds = time.perf_counter() - before_clear
                    clears += 1
                total_frames += row["video_frames"]
                record = {"index": index, "sample_id": row["sample_id"], "frames": row["video_frames"],
                          "extract_seconds": extract_seconds, "step_seconds": time.perf_counter() - begin,
                          "allocated_mib": allocated, "reserved_mib": reserved,
                          "clip_peak_allocated_mib": clip_peak, "clip_peak_reserved_mib": reserved_peak,
                          "reserved_after_clear_mib": torch.cuda.memory_reserved() / 1024**2,
                          "global_free_mib": free, "clear_seconds": clear_seconds,
                          "feature_sha256_equal": True, "timestamp": time.time()}
                stream.write(json.dumps(record) + "\n")
        finished = time.time()
        summary = {"rank": args.rank, "workers": args.workers, "samples": len(tasks),
                   "video_frames": total_frames, "started_at": started, "finished_at": finished,
                   "elapsed_seconds": finished - started, "clips_per_second": len(tasks) / (finished - started),
                   "frames_per_second": total_frames / (finished - started), "all_features_bitwise_equal": True,
                   "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
                   "minimum_global_free_mib": min_free, "cache_clears": clears,
                   "cache_limit_mib": args.cache_limit_mib, "memory_fraction": args.memory_fraction,
                   "longest_input_frames": longest["video_frames"], "decode_threads": args.decode_threads,
                   "warmup": args.warmup, "warmup_measurement": warmup,
                   "synchronize_step": args.synchronize_step,
                   "manifest_sha256": media.sha256(args.manifest)}
        media.atomic_json(directory / f"rank{args.rank}.summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        media.atomic_json(directory / f"rank{args.rank}.failure.json",
                          {"error": repr(error), "timestamp": time.time(), "rank": args.rank,
                           "memory_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                           "memory_reserved_mib": torch.cuda.memory_reserved() / 1024**2})
        raise


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--fairseq-root", default=prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
    parser.add_argument("--user-dir", default=prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--warmup", choices=["none", "first", "longest", "fixed"], default="longest")
    parser.add_argument("--warmup-frames", type=int, default=750)
    parser.add_argument("--synchronize-step", action="store_true")
    parser.add_argument("--cache-limit-mib", type=float, default=0)
    parser.add_argument("--memory-fraction", type=float, default=.6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--barrier-dir")
    args = parser.parse_args()
    if not (0 <= args.rank < args.workers and 0 < args.memory_fraction <= 1
            and args.cache_limit_mib >= 0 and args.repeats > 0):
        parser.error("Invalid worker count, rank, memory fraction, cache limit, or repetitions")
    main(args)
