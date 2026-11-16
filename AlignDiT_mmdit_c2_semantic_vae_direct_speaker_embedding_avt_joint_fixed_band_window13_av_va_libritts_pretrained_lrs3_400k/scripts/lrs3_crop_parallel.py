#!/usr/bin/env python3
"""CPU-only crop scaling in fresh isolated scratch caches.

Checks exact mouth-video hashes across process counts using the original media
worker crop path. Does not write production caches, extract AV features, or
perform FAN repair. Production staging belongs to lrs3_media_worker_pipeline.py.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PROTOCOL = "29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(args):
    import lrs3_media_worker as media
    protocol, fingerprint = media.media_protocol(args.checkpoint, args.mean_face)
    if fingerprint != args.expected_protocol_sha256:
        raise ValueError(f"Unexpected original media protocol: {fingerprint}")
    return {"protocol": protocol, "protocol_sha256": fingerprint,
            "worker_source_sha256": sha256(PROJECT / "scripts/lrs3_media_worker.py"),
            "source_manifest_sha256": sha256(args.manifest),
            "source_inventory_sha256": sha256(args.inventory),
            "checkpoint": str(Path(args.checkpoint).resolve()), "mean_face": str(Path(args.mean_face).resolve()),
            "checkpoint_stat": list(Path(args.checkpoint).stat()[6:9]),
            "mean_face_stat": list(Path(args.mean_face).stat()[6:9])}


def select_records(args):
    records = []
    seen = set()
    for row in rows(args.manifest):
        if int(row["width"]) != 224 or int(row["height"]) != 224:
            continue
        identity = row["relative_stem"]
        if identity in seen or Path(identity).is_absolute() or ".." in Path(identity).parts:
            raise ValueError(f"Duplicate or unsafe source identity: {identity}")
        seen.add(identity)
        records.append(row)
    if args.mode == "benchmark":
        if args.samples % 4:
            raise ValueError("Benchmark sample count must be divisible by four")
        bins = {index: [] for index in range(4)}
        for row in records:
            frames = int(row["video_frames"])
            if 0 < frames <= 750:
                bins[min(frames // 125, 3)].append(row)
        records = []
        generator = random.Random(94281)
        for bucket, candidates in bins.items():
            generator.shuffle(candidates)
            if len(candidates) < args.samples // 4:
                raise ValueError(f"Insufficient examples in frame bucket {bucket}")
            records.extend(candidates[:args.samples // 4])
        generator.shuffle(records)
    return records


def filtered_inventory(source, target, originals):
    # Include every segment of each requested original, preserving full offsets.
    with Path(source).open(newline="") as stream, target.open("w", newline="") as destination:
        reader = csv.DictReader(stream, delimiter="\t")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row["original_clip_id"] in originals:
                writer.writerow(row)


def worker(args):
    # These libraries are imported only after the parent has fixed all thread
    # environment variables. CPU-only even if the user's shell exposes GPUs.
    import copy
    import cv2
    import lrs3_media_worker as media
    spec = json.loads(Path(args.spec).read_text())
    if sha256(args.spec) != args.spec_sha256:
        raise ValueError("Pinned crop protocol specification changed")
    if sha256(PROJECT / "scripts/lrs3_media_worker.py") != spec["worker_source_sha256"]:
        raise ValueError("Media worker code changed after crop plan creation")
    for name in ("checkpoint", "mean_face"):
        path = Path(spec[name])
        if list(path.stat()[6:9]) != spec[name + "_stat"]:
            raise ValueError(f"Pinned {name} changed after protocol validation")
    capture = cv2.VideoCapture
    def open_capture(path, *positional, **keywords):
        if positional or keywords:
            return capture(path, *positional, **keywords)
        return capture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, args.decode_threads])
    cv2.VideoCapture = open_capture
    # Each protocol file was hashed once by the coordinator before any process
    # starts. Reusing that verified identity avoids N redundant multi-GB reads.
    def cached_protocol(checkpoint, mean_face):
        if str(Path(checkpoint).resolve()) != spec["checkpoint"] or str(Path(mean_face).resolve()) != spec["mean_face"]:
            raise ValueError("Worker assets differ from the pinned crop plan")
        return copy.deepcopy(spec["protocol"]), spec["protocol_sha256"]
    media.media_protocol = cached_protocol
    namespace = argparse.Namespace(manifest=args.manifest, output_root=args.output_root,
        inventory=args.inventory, landmark_root=args.landmark_root, original_root=args.original_root,
        checkpoint=spec["checkpoint"], mean_face=spec["mean_face"], num_shards=1, shard=0, limit=0,
        cpu_threads=args.cpu_threads, decode_threads=args.decode_threads, stage="crop")
    return media.main(namespace)


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def execute(args, records, spec, directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    root = directory / "cache"
    state_path = directory / "state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    identity = {"protocol_sha256": spec["protocol_sha256"], "source_manifest_sha256": spec["source_manifest_sha256"],
                "worker_source_sha256": spec["worker_source_sha256"], "processes": count,
                "cpu_threads": args.cpu_threads, "decode_threads": args.decode_threads,
                "sample_ids_sha256": hashlib.sha256(json.dumps([r["sample_id"] for r in records]).encode()).hexdigest()}
    if previous and previous["identity"] != identity:
        raise ValueError("Crop run identity changed; use a new report directory")
    spec_path = directory / "protocol.json"
    save(spec_path, spec)
    spec_hash = sha256(spec_path)
    inventory = directory / "complete_groups.tsv"
    filtered_inventory(args.inventory, inventory, {r["original_clip_id"] for r in records})
    children = []
    started = time.perf_counter()
    cpu_before = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    state = {"identity": identity, "coordinator_pid": os.getpid(), "requested": len(records),
             "started_unix": time.time(), "status": "running", "processes": [], "peak_worker_rss_kib": 0,
             "peak_total_workers_rss_kib": 0, "cpu_only": True}
    with ExitStack() as stack:
        lock_path = root / "state/locks/media.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context(lock_path.open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard = root / "state/video_25hz/WRITE_ACTIVE.json"
        save(guard, {"crop_coordinator": str(directory.resolve()), "pid": os.getpid(), "protocol_sha256": spec["protocol_sha256"],
                     "started_unix": time.time(), "stage": "cpu_crop_only"})
        env = os.environ.copy()
        for name in list(env):
            if name.lower().endswith("_proxy"):
                env.pop(name)
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        for rank in range(count):
            partition = records[rank::count]
            if not partition:
                continue
            manifest = directory / f"crop_partition_{rank:03d}.jsonl"
            write_rows(manifest, partition)
            log_path = directory / f"worker_{rank:03d}.log"
            log = stack.enter_context(log_path.open("a"))
            command = [sys.executable, "-u", str(SCRIPT), "--mode", "_worker", "--manifest", str(manifest),
                "--output-root", str(root), "--inventory", str(inventory), "--landmark-root", args.landmark_root,
                "--original-root", args.original_root, "--spec", str(spec_path), "--spec-sha256", spec_hash,
                "--cpu-threads", str(args.cpu_threads), "--decode-threads", str(args.decode_threads)]
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, pass_fds=(lock.fileno(),))
            receipt = {"rank": rank, "pid": process.pid, "requested": len(partition), "log": str(log_path),
                       "summary": str(root / "reports" / f"media_{manifest.stem}_crop_shard0.summary.json")}
            children.append((process, receipt))
            state["processes"].append(receipt)
        save(state_path, state)
        stop_requested = False
        def stop_after_batch(unused_signal, unused_frame):
            nonlocal stop_requested
            stop_requested = True
        previous_handler = signal.signal(signal.SIGTERM, stop_after_batch)
        last_saved = 0
        try:
            while any(process.poll() is None for process, _ in children):
                usage = [rss_kib(process.pid) for process, _ in children]
                state["peak_worker_rss_kib"] = max(state["peak_worker_rss_kib"], max(usage, default=0))
                state["peak_total_workers_rss_kib"] = max(state["peak_total_workers_rss_kib"], sum(usage))
                if time.time() - last_saved > 5:
                    state.update({"updated_unix": time.time(), "elapsed_seconds": time.perf_counter() - started,
                                  "stop_requested_after_workers_finish": stop_requested})
                    save(state_path, state)
                    last_saved = time.time()
                time.sleep(.2)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        succeeded = failed = 0
        hashes = {}
        errors = {}
        for process, receipt in children:
            receipt["exit_code"] = process.wait()
            summary_path = Path(receipt["summary"])
            if receipt["exit_code"] not in (0, 1) or not summary_path.exists():
                raise RuntimeError(f"Crop worker crashed: {receipt}")
            summary = json.loads(summary_path.read_text())
            if summary["requested"] != receipt["requested"]:
                raise RuntimeError("Crop summary count changed")
            succeeded += summary["succeeded"]
            failed += summary["failed"]
            # Reports append across attempts: only each identity's latest result
            # participates in the current receipt.
            latest = {r["relative_stem"]: r for r in rows(summary["report"])}
            for key, row in latest.items():
                if row["status"] == "ok":
                    hashes[key] = sha256(row["video_mouth_path"])
                else:
                    errors[key] = row["error"]
        elapsed = time.perf_counter() - started
        cpu_after = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
        state["cgroup_cpu_stat_delta"] = {key: int(value) - int(cpu_before.get(key, 0)) for key, value in cpu_after.items()}
        state["cgroup_stats_include_concurrent_production_jobs"] = True
        state.update({"status": "complete" if failed == 0 else "complete_with_landmark_failures", "elapsed_seconds": elapsed,
            "succeeded": succeeded, "failed": failed, "requested_clips_per_second": len(records) / elapsed,
            "succeeded_clips_per_second": succeeded / elapsed, "source_video_frames": sum(int(r["video_frames"]) for r in records),
            "source_video_seconds": sum(int(r["video_frames"]) for r in records) / 25, "mouth_sha256": hashes,
            "failures": errors, "finished_unix": time.time(), "landmark_failures_require_fan_or_exclusion": True})
        save(state_path, state)
        guard.unlink()
    return state


def main(args):
    if args.mode == "_worker":
        return worker(args)
    root = Path(args.output_root).resolve()
    if args.mode == "benchmark" and root.exists():
        raise ValueError("Benchmark output must be a fresh scratch directory")
    root.mkdir(parents=True, exist_ok=True)
    spec = prepare(args)
    records = select_records(args)
    if not records:
        raise ValueError("No 224x224 source records selected")
    write_rows(root / "benchmark_inputs.jsonl", records)
    counts = [int(value) for value in args.benchmark_processes.split(",")]
    trials = []
    reference = None
    for count in counts:
        report = execute(args, records, spec, root / f"processes_{count:02d}", count)
        if reference is None:
            reference = report
        report["hash_mismatches"] = sorted(key for key, value in report["mouth_sha256"].items() if reference["mouth_sha256"].get(key) != value)
        report["success_set_matches_reference"] = set(report["mouth_sha256"]) == set(reference["mouth_sha256"])
        report["failure_set_matches_reference"] = set(report["failures"]) == set(reference["failures"])
        concise = {key: value for key, value in report.items() if key not in ("mouth_sha256", "failures")}
        trials.append(concise)
        save(root / "summary.json", {"spec": spec, "samples": len(records), "trials": trials,
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "production_jobs_continued_during_measurement": True})
        print(json.dumps(concise), flush=True)
    return int(any(t["hash_mismatches"] or not t["success_set_matches_reference"] or not t["failure_set_matches_reference"] for t in trials))


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["benchmark", "_worker"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory", default=prefix + "/zjw524/datasets/LRS3/DARKNSU_extracted/audit/all_pairs_manifest.tsv")
    parser.add_argument("--landmark-root", default=prefix + "/zjw524/datasets/LRS3/landmarks/LRS3_landmarks")
    parser.add_argument("--original-root", default=prefix + "/zjw524/datasets/LRS3/ainncy")
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--mean-face", default=prefix + "/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1/assets/20words_mean_face.npy")
    parser.add_argument("--expected-protocol-sha256", default=PROTOCOL)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--benchmark-processes", default="6,16,32,48")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    arguments = parser.parse_args()
    if arguments.cpu_threads < 1 or arguments.decode_threads < 1:
        parser.error("Process and thread counts must be positive")
    raise SystemExit(main(arguments))
