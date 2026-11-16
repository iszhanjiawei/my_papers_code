#!/usr/bin/env python3
"""Durable LRS3 media production: parallel AV, long clips, FAN repair, verification.

Launch this coordinator under setsid. Children inherit its detached session, log
to separate files, and receive explicit GPU assignments. Completed stages are
recorded and resumed; all failures and all prior attempts remain on disk.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from lrs3_media_worker import atomic_json, media_protocol, sha256, valid_array


SCRIPTS = Path(__file__).resolve().parent


def save_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def verify_one(task):
    row, output_root, expected_protocol, decoder_threads = task
    root = Path(output_root)
    identity = row["relative_stem"]
    result = {"sample_id": row["sample_id"], "relative_stem": identity,
              "original_clip_id": row["original_clip_id"], "split": row["split"],
              "source_split": row["source_split"], "video_frames": int(row["video_frames"])}
    try:
        receipt = json.loads((root / "reports" / "media_receipts" / (identity + ".json")).read_text())
        if receipt.get("protocol_sha256") != expected_protocol or not receipt.get("features_complete"):
            raise ValueError("Missing or unexpected complete-cache protocol receipt")
        if receipt.get("source_sha256") != sha256(row["video_path"]):
            raise ValueError("Source-video checksum differs from cache receipt")
        feature = root / "avhubert_video_25hz" / (identity + ".npy")
        valid_array(feature, row["video_frames"])
        if sha256(feature) != receipt.get("feature_sha256"):
            raise ValueError("Feature checksum differs from cache receipt")
        repair_sha = receipt.get("landmark_repair_sha256")
        if repair_sha and sha256(root / "repaired_landmarks" / (identity + ".npy")) != repair_sha:
            raise ValueError("Repaired-landmark checksum differs from cache receipt")
        mouth = root / "video_mouth" / (identity + ".mp4")
        options = [cv2.CAP_PROP_N_THREADS, decoder_threads] if hasattr(cv2, "CAP_PROP_N_THREADS") else []
        capture = cv2.VideoCapture(str(mouth), cv2.CAP_FFMPEG, options)
        count = 0
        try:
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 25) > .001:
                raise ValueError("Mouth video is unreadable or not 25 fps")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (96, 96):
                    raise ValueError("Decoded mouth video is not 96x96")
                count += 1
        finally:
            capture.release()
        if count != row["video_frames"]:
            raise ValueError(f"Decoded mouth frame mismatch: {count} != {row['video_frames']}")
        result.update({"status": "ok", "crop_kind": receipt["crop_kind"],
                       "feature_sha256": receipt["feature_sha256"],
                       "landmark_mapping_evidence": receipt.get("landmark_mapping_evidence"),
                       "mouth_path": str(mouth), "feature_path": str(feature)})
    except Exception as error:
        result.update({"status": "failed", "error": repr(error)})
    return result


class Pipeline:
    def __init__(self, args, lock_fd=None):
        self.args = args
        self.lock_fd = lock_fd
        self.root = Path(args.output_root)
        self.protocol, self.protocol_hash = media_protocol(args.checkpoint, args.mean_face)
        if self.protocol_hash != args.expected_protocol_sha256:
            raise ValueError(f"Media protocol changed: {self.protocol_hash} != {args.expected_protocol_sha256}")
        self.directory = self.root / "reports" / "media_full_pipeline"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logs = self.root / "logs" / "media_full_pipeline"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.manifests = self.root / "manifests" / "media_full_pipeline"
        self.manifests.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"stages": {}}
        fingerprint = sha256(args.manifest)
        if self.state.get("manifest_sha256", fingerprint) != fingerprint:
            raise ValueError("Source manifest changed: use a new output version")
        self.state.update({"manifest": str(Path(args.manifest).resolve()), "manifest_sha256": fingerprint,
                           "coordinator_pid": os.getpid(), "coordinator_sid": os.getsid(0),
                           "updated_at": time.time(), "status": "running"})
        atomic_json(self.state_path, self.state)

    def command(self, manifest, rank, count, repair=False, media_stage="all", cpu_threads=None,
                decode_threads=None):
        script = "lrs3_media_worker_repair.py" if repair else "lrs3_media_worker.py"
        command = [sys.executable, "-u", str(SCRIPTS / script), "--manifest", str(manifest),
                   "--output-root", str(self.root), "--num-shards", str(count), "--shard", str(rank),
                   "--cpu-threads", str(cpu_threads or self.args.cpu_threads)]
        if not repair:
            command += ["--mean-face", self.args.mean_face, "--checkpoint", self.args.checkpoint,
                        "--inventory", self.args.inventory, "--landmark-root", self.args.landmark_root,
                        "--original-root", self.args.original_root,
                        "--fairseq-root", self.args.fairseq_root, "--user-dir", self.args.user_dir,
                        "--stage", media_stage]
            if decode_threads is not None:
                command += ["--decode-threads", str(decode_threads)]
        return command

    def run_stage(self, name, rows, gpu_map, repair=False, media_stage="all", separate_manifests=False,
                  cpu_threads=None, decode_threads=None):
        manifest = self.manifests / (name + ".jsonl")
        save_manifest(manifest, rows)
        fingerprint = sha256(manifest)
        previous = self.state["stages"].get(name, {})
        if previous.get("finished") and previous.get("manifest_sha256") == fingerprint:
            print(json.dumps({"stage": name, "resumed_finished_stage": True}), flush=True)
            return previous["reports"]
        if not rows:
            self.state["stages"][name] = {"finished": True, "manifest_sha256": fingerprint,
                                          "reports": [], "requested": 0, "processes": []}
            atomic_json(self.state_path, self.state)
            return []
        gpu_map = gpu_map[:min(len(rows), len(gpu_map))]
        count = len(gpu_map)
        stage = {"finished": False, "manifest_sha256": fingerprint, "requested": len(rows),
                 "started_at": time.time(), "attempt_token": f"a{time.time_ns()}",
                 "processes": [], "reports": []}
        if previous:
            self.state.setdefault("stage_attempt_history", {}).setdefault(name, []).append(previous)
        self.state["stages"][name] = stage
        self.state["active_stage"] = name
        children = []
        attempt_name = f"{name}__{stage['attempt_token']}"
        attempt_manifest = self.manifests / (attempt_name + ".jsonl")
        if not separate_manifests:
            save_manifest(attempt_manifest, rows)
        for rank, gpu in enumerate(gpu_map):
            child_manifest = attempt_manifest
            child_rank, child_count = rank, count
            if separate_manifests:
                # CPU crop stages use small physical shards, avoiding dozens of
                # copies of the complete source manifest in worker memory.
                child_manifest = self.manifests / f"{attempt_name}_part{rank:02d}.jsonl"
                save_manifest(child_manifest, rows[rank::count])
                child_rank, child_count = 0, 1
            command = self.command(child_manifest, child_rank, child_count, repair, media_stage,
                                   cpu_threads, decode_threads)
            env = os.environ.copy()
            threads = cpu_threads or self.args.cpu_threads
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(threads),
                        "MKL_NUM_THREADS": str(threads), "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(SCRIPTS.parent / "src")})
            if media_stage == "crop":
                env["OPENBLAS_NUM_THREADS"] = "1"
            log_path = self.logs / f"{name}_shard{rank}.log"
            log = log_path.open("a")
            process = subprocess.Popen(command, env=env, cwd=SCRIPTS.parent, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=() if self.lock_fd is None else (self.lock_fd,))
            suffix = f"_shard{child_rank}" if child_count > 1 else ""
            report_name = (f"media_fan_repair_{child_manifest.stem}{suffix}.jsonl" if repair
                           else f"media_{child_manifest.stem}_{media_stage}_shard{child_rank}.jsonl")
            stage["reports"].append(str(self.root / "reports" / report_name))
            receipt = {"pid": process.pid, "sid": os.getsid(process.pid), "gpu": gpu,
                       "rank": rank, "command": command, "log": str(log_path),
                       "manifest": str(child_manifest), "manifest_sha256": sha256(child_manifest),
                       "report": str(self.root / "reports" / report_name),
                       "expected_records": len(rows[rank::count])}
            stage["processes"].append(receipt)
            children.append((process, log, receipt))
            atomic_json(self.state_path, self.state)
        print(json.dumps({"stage": name, "requested": len(rows), "pids": [p.pid for p, _, _ in children]}), flush=True)
        pending = children[:]
        while pending:
            next_pending = []
            for process, log, receipt in pending:
                code = process.poll()
                if code is None:
                    next_pending.append((process, log, receipt))
                else:
                    receipt.update({"exit_code": code, "finished_at": time.time()})
                    log.close()
            pending = next_pending
            self.state["updated_at"] = time.time()
            atomic_json(self.state_path, self.state)
            if pending:
                time.sleep(5)
        # Each attempt has unique reports. A stale prior summary can never make
        # a crashed worker appear complete; per-item failures may return 1.
        missing_summaries = [str(Path(p).with_suffix(".summary.json")) for p in stage["reports"]
                             if not Path(p).with_suffix(".summary.json").exists()]
        if missing_summaries:
            raise RuntimeError(f"Workers did not finish their reports: {missing_summaries}")
        for receipt in stage["processes"]:
            summary = json.loads(Path(receipt["report"]).with_suffix(".summary.json").read_text())
            expected = {row["relative_stem"] for row in rows[receipt["rank"]::count]}
            results = self.latest_results([receipt["report"]])
            counts = Counter(row["status"] for row in results.values())
            if (receipt["exit_code"] not in (0, 1) or set(results) != expected
                    or summary.get("requested") != len(expected)
                    or summary.get("succeeded") != counts["ok"]
                    or summary.get("failed") != counts["failed"]
                    or counts["ok"] + counts["failed"] != len(expected)
                    or receipt["exit_code"] != int(counts["failed"] > 0)):
                raise RuntimeError(f"Worker completion/coverage mismatch: {receipt}")
            if not repair:
                protocol = json.loads(Path(receipt["report"]).with_suffix(".protocol.json").read_text())
                if (summary.get("protocol_sha256") != self.protocol_hash
                        or protocol.get("protocol_sha256") != self.protocol_hash
                        or protocol.get("manifest_sha256") != receipt["manifest_sha256"]):
                    raise RuntimeError(f"Worker completion protocol/manifest mismatch: {receipt}")
        stage.update({"finished": True, "finished_at": time.time()})
        atomic_json(self.state_path, self.state)
        return stage["reports"]

    @staticmethod
    def latest_results(paths):
        latest = {}
        for path in paths:
            if Path(path).exists():
                with open(path) as stream:
                    for line in stream:
                        row = json.loads(line)
                        latest[row["relative_stem"]] = row
        return latest

    def run(self):
        args = self.args
        rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        if len({row["relative_stem"] for row in rows}) != len(rows):
            raise ValueError("Duplicate source manifest identities")
        regular = [row for row in rows if row["video_frames"] <= 750]
        long = [row for row in rows if row["video_frames"] > 750]
        gpu_map = args.gpu_map.split(",")
        repair_gpus = args.repair_gpus.split(",")
        if args.crop_workers:
            # This stage finishes before any GPU media writer starts. Existing
            # verified receipts are reused, including complete AV feature caches.
            self.run_stage("full_media_cpu_crop", [row for row in rows if row["width"] == 224],
                           [""] * args.crop_workers, media_stage="crop", separate_manifests=True,
                           cpu_threads=1, decode_threads=args.crop_decode_threads)
        first_paths = self.run_stage("full_media_regular", regular, gpu_map)
        first_paths += self.run_stage("full_media_long", long, [args.long_gpu])
        first = self.latest_results(first_paths)
        missing = [row["relative_stem"] for row in rows if row["relative_stem"] not in first]
        if missing:
            raise RuntimeError(f"First pass did not account for {len(missing)} inputs")
        failed_rows = [row for row in rows if first[row["relative_stem"]]["status"] != "ok"]
        failures = []
        repair_rows = []
        for row in failed_rows:
            failure = first[row["relative_stem"]]
            error = failure.get("error", "").lower()
            crop_error = any(word in error for word in ["landmark", "mouth roi", "alignment transform", ".pkl"])
            category = ("fan_repair" if row["width"] == 224 and crop_error
                        else "gpu_memory_serial_retry" if "out of memory" in error
                        else "other_explicit_retry")
            failures.append({**failure, "repair_classification": category})
            if category == "fan_repair":
                repair_rows.append(row)
        save_manifest(self.directory / "first_pass_failures.jsonl", failures)
        save_manifest(self.manifests / "first_pass_failed_inputs.jsonl", failed_rows)
        self.run_stage("full_media_landmark_repair", repair_rows, repair_gpus, repair=True)
        # One AV worker per card for retry; longer clips remain strictly serial.
        retry_regular = [row for row in failed_rows if row["video_frames"] <= 750]
        retry_long = [row for row in failed_rows if row["video_frames"] > 750]
        retry_paths = self.run_stage("full_media_retry", retry_regular, repair_gpus)
        retry_paths += self.run_stage("full_media_retry_long", retry_long, [args.long_gpu])
        latest = {**first, **self.latest_results(retry_paths)}
        unresolved = [latest[row["relative_stem"]] for row in rows if latest[row["relative_stem"]]["status"] != "ok"]
        save_manifest(self.directory / "unresolved_worker_failures.jsonl", unresolved)
        self.verify(rows)

    def verify(self, rows):
        protocol_hash = self.protocol_hash
        report = self.directory / "independent_verification.jsonl"
        existing = self.latest_results([report])
        pending = [row for row in rows if existing.get(row["relative_stem"], {}).get("status") != "ok"]
        self.state["active_stage"] = "independent_verification"
        atomic_json(self.state_path, self.state)
        with report.open("a", buffering=1) as stream:
            with ProcessPoolExecutor(max_workers=self.args.verify_workers) as pool:
                tasks = ((row, str(self.root), protocol_hash, self.args.verify_decoder_threads) for row in pending)
                for index, result in enumerate(pool.map(verify_one, tasks, chunksize=8)):
                    existing[result["relative_stem"]] = result
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if (index + 1) % 1000 == 0:
                        print(json.dumps({"stage": "independent_verification", "processed": index + 1,
                                          "pending_at_start": len(pending)}), flush=True)
        counts, crop_kinds = Counter(), Counter()
        originals = defaultdict(lambda: {"expected": 0, "verified": 0, "failed": [], "video_frames": 0})
        for row in rows:
            result = existing[row["relative_stem"]]
            counts[result["status"]] += 1
            original = originals[row["original_clip_id"]]
            original["expected"] += 1
            original["video_frames"] += row["video_frames"]
            if result["status"] == "ok":
                original["verified"] += 1
                crop_kinds[result["crop_kind"]] += 1
            else:
                original["failed"].append({"sample_id": row["sample_id"], "error": result.get("error")})
        save_manifest(self.directory / "original_id_media_coverage.jsonl", [
            {"original_clip_id": key, **value, "complete": value["verified"] == value["expected"]}
            for key, value in sorted(originals.items())])
        summary = {"requested": len(rows), "verified": counts["ok"], "failed": counts["failed"],
                   "complete": counts["ok"] == len(rows), "original_ids": len(originals),
                   "complete_original_ids": sum(v["expected"] == v["verified"] for v in originals.values()),
                   "crop_kinds": dict(crop_kinds), "protocol_sha256": protocol_hash,
                   "checks": ["source_sha256", "feature_sha256", "feature_float32_Tx1024_finite",
                              "complete_decoding_mouth_96x96_25fps_frame_count", "repair_sha256_if_present"]}
        atomic_json(self.directory / "summary.json", summary)
        self.state.update({"status": "complete" if summary["complete"] else "finished_with_failures",
                           "active_stage": None, "finished_at": time.time(), "summary": summary})
        atomic_json(self.state_path, self.state)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mean-face", required=True)
    parser.add_argument("--gpu-map", default="0,0,1,1,3,3")
    parser.add_argument("--repair-gpus", default="0,1,3")
    parser.add_argument("--long-gpu", default="0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--crop-workers", type=int, default=0,
                        help="Run a separate CPU face-to-mouth stage before GPU extraction")
    parser.add_argument("--crop-decode-threads", type=int, default=2)
    parser.add_argument("--verify-workers", type=int, default=12)
    parser.add_argument("--verify-decoder-threads", type=int, default=2)
    parser.add_argument("--expected-protocol-sha256", default="29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f")
    parser.add_argument("--inventory", default=prefix + "/zjw524/datasets/LRS3/DARKNSU_extracted/audit/all_pairs_manifest.tsv")
    parser.add_argument("--landmark-root", default=prefix + "/zjw524/datasets/LRS3/landmarks/LRS3_landmarks")
    parser.add_argument("--original-root", default=prefix + "/zjw524/datasets/LRS3/ainncy")
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--fairseq-root", default=prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
    parser.add_argument("--user-dir", default=prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert")
    arguments = parser.parse_args()
    if arguments.crop_workers < 0 or arguments.crop_decode_threads < 1:
        parser.error("crop-workers must be nonnegative and crop-decode-threads positive")
    state_root = Path(arguments.output_root) / "state"
    lock_path = state_root / "locks" / "media.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock = lock_path.open("a")
    fcntl.flock(stage_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = state_root / "video_25hz" / "WRITE_ACTIVE.json"
    atomic_json(marker, {"pid": os.getpid(), "sid": os.getsid(0), "started_at": time.time(),
                        "manifest": str(Path(arguments.manifest).resolve()),
                        "expected_protocol_sha256": arguments.expected_protocol_sha256})
    pipeline = None
    try:
        pipeline = Pipeline(arguments, stage_lock.fileno())
        pipeline.run()
    except Exception as error:
        if pipeline is not None:
            pipeline.state.update({"status": "failed", "coordinator_error": repr(error), "updated_at": time.time()})
            atomic_json(pipeline.state_path, pipeline.state)
        raise
    if pipeline.state["status"] == "complete":
        marker.unlink()
    raise SystemExit(0 if pipeline.state["status"] == "complete" else 1)
