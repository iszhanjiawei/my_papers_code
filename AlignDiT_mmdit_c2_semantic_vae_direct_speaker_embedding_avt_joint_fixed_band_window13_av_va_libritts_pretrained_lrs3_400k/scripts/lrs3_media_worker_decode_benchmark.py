#!/usr/bin/env python3
"""Benchmark decoder threading through the actual media worker in isolated roots.

One model stays loaded while fresh mouth videos, features, and receipts are
created for each variant. Decoder changes are confined to this process. Timing
excludes model/checkpoint initialization and includes the complete per-item
worker path. Feature and mouth hashes are compared across variants.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np

import lrs3_media_worker as worker


def main(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    random.Random(5873).shuffle(rows)
    picked = []
    bins = defaultdict(int)
    for row in rows:
        frames = int(row["video_frames"])
        if frames > 750:
            continue
        bucket = (int(row["width"]), min(frames // 125, 3))
        if bucket[0] in (96, 224) and bins[bucket] < args.per_bucket:
            picked.append(row)
            bins[bucket] += 1
        if len(picked) == args.per_bucket * 8:
            break
    if len(picked) != args.per_bucket * 8:
        raise ValueError(f"Insufficient stratified examples: {dict(bins)}")
    picked.sort(key=lambda row: (int(row["width"]), row["sample_id"]))
    manifest = root / "benchmark_inputs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in picked))
    namespace = argparse.Namespace(
        manifest=str(manifest), output_root="", inventory=args.inventory,
        landmark_root=args.landmark_root, original_root=args.original_root,
        checkpoint=args.checkpoint, fairseq_root=args.fairseq_root,
        user_dir=args.user_dir, mean_face=args.mean_face, num_shards=1,
        shard=0, limit=0, cpu_threads=2, stage="all")
    cv2.setNumThreads(2)
    protocol, protocol_hash = worker.media_protocol(args.checkpoint, args.mean_face)
    reader = worker.load_reader(namespace)
    import torch
    # Warm up real 25 Hz inference; no production cache is accessed or changed.
    warm = next(row for row in picked if int(row["width"]) == 96)
    reader.get_feats((warm["video_path"], None)).detach().cpu().numpy()
    worker.load_reader = lambda unused: reader
    worker.media_protocol = lambda *_: (copy.deepcopy(protocol), protocol_hash)

    capture = cv2.VideoCapture
    decoder_threads = None
    observed = defaultdict(set)

    def open_capture(path, *positional, **keywords):
        if positional or keywords:
            raise ValueError("Unexpected VideoCapture arguments in benchmark")
        cap = (capture(path) if decoder_threads is None else
               capture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads]))
        observed[str(decoder_threads)].add(cap.get(cv2.CAP_PROP_N_THREADS))
        return cap

    cv2.VideoCapture = open_capture
    current_timing = defaultdict(float)

    def timed(original, name, synchronize=False):
        def call(*positional, **keywords):
            started = time.perf_counter()
            try:
                result = original(*positional, **keywords)
                if synchronize:
                    torch.cuda.synchronize()
                return result
            finally:
                current_timing[name] += time.perf_counter() - started
        return call

    for name in ("read_frames", "crop_mouth", "verify_original_frame_mapping", "write_video", "sha256", "complete_segment_offsets"):
        setattr(worker, name, timed(getattr(worker, name), name))
    reader.get_feats = timed(reader.get_feats, "get_feats_including_decode", synchronize=True)
    reader.custom_utils.load_video = timed(reader.custom_utils.load_video, "reader_load_video")
    trials = []
    reference = None
    for repeat in range(args.repeats):
        # Alternate order to reduce page-cache and coexistence timing bias.
        order = [None, 1, 2, 4] if repeat % 2 == 0 else [4, 2, 1, None]
        for decoder_threads in order:
            label = f"repeat{repeat}_threads{decoder_threads or 'default'}"
            namespace.output_root = str(root / label)
            current_timing.clear()
            started = time.perf_counter()
            with (root / f"{label}.log").open("w", buffering=1) as log:
                with contextlib.redirect_stdout(log):
                    exit_code = worker.main(namespace)
            elapsed = time.perf_counter() - started
            report_path = Path(namespace.output_root) / "reports" / "media_benchmark_inputs_all_shard0.jsonl"
            reports = [json.loads(line) for line in report_path.read_text().splitlines()]
            hashes = {}
            for row in reports:
                if row["status"] == "ok":
                    hashes[row["relative_stem"]] = {
                        "feature": worker.sha256(row["video_feature_path"]),
                        "mouth": worker.sha256(row["video_mouth_path"])}
            if reference is None:
                reference = hashes
            mismatches = [key for key, value in hashes.items() if reference.get(key) != value]
            result = {"variant": label, "decoder_threads": decoder_threads,
                      "observed_decoder_threads": sorted(observed[str(decoder_threads)]),
                      "elapsed_seconds": elapsed, "requested": len(picked),
                      "succeeded": len(hashes), "failed": len(picked) - len(hashes),
                      "clips_per_second": len(picked) / elapsed,
                      "video_realtime_factor": sum(int(row["video_frames"]) for row in picked) / 25 / elapsed,
                      "nested_inclusive_function_seconds": dict(current_timing),
                      "hash_mismatches": mismatches, "success_set_matches_reference": set(hashes) == set(reference),
                      "worker_exit_code": exit_code}
            trials.append(result)
            worker.atomic_json(root / "summary.json", {
                "protocol_sha256": protocol_hash, "manifest": str(manifest),
                "trial_function_times_are_nested_inclusive": True, "trials": trials,
                "model_initialization_excluded": True, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
            print(json.dumps(result), flush=True)
    return int(any(t["hash_mismatches"] or not t["success_set_matches_reference"] for t in trials))


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mean-face", required=True)
    parser.add_argument("--inventory", default=prefix + "/zjw524/datasets/LRS3/DARKNSU_extracted/audit/all_pairs_manifest.tsv")
    parser.add_argument("--landmark-root", default=prefix + "/zjw524/datasets/LRS3/landmarks/LRS3_landmarks")
    parser.add_argument("--original-root", default=prefix + "/zjw524/datasets/LRS3/ainncy")
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--fairseq-root", default=prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
    parser.add_argument("--user-dir", default=prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert")
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    raise SystemExit(main(parser.parse_args()))
