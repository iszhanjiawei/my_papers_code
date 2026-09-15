#!/usr/bin/env python3
"""Resumable sharded full-RGB Synchformer extraction and exhaustive audits."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime
import errno
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import torch

from aligndit.model.synchformer_features import (
    CHECKPOINT_SHA256, MODEL_ID, PREPROCESSING, SCHEMA_VERSION,
    FrozenSynchformerExtractor, atomic_json, cache_path, canonical_clip_key,
    default_checkpoint_path, load_synchformer_payload, read_inventory,
    save_synchformer_feature, sha256_file, source_identity,
)


def parser():
    root = os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects/data"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inventory", type=Path, default=Path(root) / "CelebVDub_svae1000k_sample_seed666_fp32/manifests/inventory.jsonl")
    result.add_argument("--video-root", type=Path, default=Path(root) / "CelebVDub/video")
    result.add_argument("--cache-dir", type=Path, default=Path(root) / "CelebVDub/synchformer_25fps_16f_stride8")
    result.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    result.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=8, help="number of 16-frame segments per forward")
    result.add_argument("--rank", "--shard-index", type=int, default=0)
    result.add_argument("--world-size", "--num-shards", type=int, default=1)
    result.add_argument("--limit", type=int, default=0, help="debug subset; cannot produce a complete coverage certificate")
    result.add_argument("--num-threads", type=int, default=2)
    result.add_argument("--initialize-only", action="store_true", help="initialize cache metadata once before launching extraction workers")
    result.add_argument("--audit-only", action="store_true", help="fully validate every feature tensor, source fingerprint and duration")
    result.add_argument("--audit-workers", type=int, default=1, help="read-only audit processes (spawn); each subprocess uses one Torch thread")
    result.add_argument("--audit-chunksize", type=int, default=32, help="records assigned per audit process task")
    result.add_argument("--source-audit-only", action="store_true", help="check RGB source coverage before weights are available")
    result.add_argument("--log-every", type=int, default=25)
    return result


def clip_key(record):
    return canonical_clip_key((record.get("utterance_key") or record["audio_relative_path"]))


def common_metadata(args, records):
    return {"schema_version": SCHEMA_VERSION, "model_id": MODEL_ID,
            "checkpoint_sha256": args.checkpoint_sha256, "preprocessing": PREPROCESSING,
            "inventory_sha256": sha256_file(args.inventory), "expected_count": len(records),
            "inventory_path": str(args.inventory), "video_root": str(args.video_root),
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}



def read_cache_metadata(path, *, required=True, attempts=20, retry_seconds=0.1):
    """Open directly and retry transient shared-filesystem visibility failures.

    An exists()/read() pair can race with replacement on shared filesystems.
    Workers never create or replace metadata; the launcher initializes it first.
    """
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            failure = error
        except OSError as error:
            if error.errno != errno.ESTALE:
                raise
            failure = error
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    if not required and isinstance(failure, FileNotFoundError):
        return None
    raise RuntimeError(f"Cannot read stable cache metadata {path}; run --initialize-only before workers") from failure


def prepare_cache_metadata(args, records, *, initialize=False):
    """Parent owns initialization; extraction workers only validate identity."""
    metadata = common_metadata(args, records)
    metadata_path = args.cache_dir / "metadata.json"
    if initialize:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    existing = read_cache_metadata(metadata_path, required=not initialize)
    if existing is not None:
        for field in ("schema_version", "model_id", "checkpoint_sha256", "preprocessing", "inventory_sha256"):
            if existing.get(field) != metadata[field]:
                raise ValueError(f"Cache directory metadata {field} mismatch; choose a separate cache directory")
    if initialize:
        # Preserve matching bytes/mtime so concurrent readers never observe a
        # gratuitous replacement. A preceding --limit smoke may change count.
        if existing is None or existing.get("expected_count") != metadata["expected_count"]:
            atomic_json(metadata_path, metadata)
        # Only the initializing parent invalidates the old coverage certificate.
        (args.cache_dir / "coverage_report.json").unlink(missing_ok=True)
    elif existing.get("expected_count") != metadata["expected_count"]:
        raise ValueError("Cache metadata count differs; run --initialize-only before workers")
    return metadata


def audit_record(record, video_root, cache_dir, checkpoint_sha256, source_only):
    """One unchanged full validation; only small scalars leave the process."""
    key = clip_key(record)
    video = video_root / (key + ".mp4")
    try:
        identity = source_identity(video, key)
        if identity["size_bytes"] <= 0:
            raise ValueError("empty source video")
        tokens = 0
        if not source_only:
            payload = load_synchformer_payload(cache_dir, key, checkpoint_sha256, video_path=video)
            meta = payload["metadata"]
            # RGB stream frames are discrete (40 ms); allow two frames for
            # encoder/sample-boundary rounding, never a 15 s truncation.
            expected_duration = float(record["duration_seconds"])
            if abs(meta["duration_seconds"] - expected_duration) > 0.081:
                raise ValueError(f"RGB/audio duration mismatch: {meta['duration_seconds']} vs {expected_duration}")
            tokens = payload["features"].shape[0]
        return key, tokens, None, None
    except FileNotFoundError as error:
        return key, 0, "missing", str(error)
    except Exception as error:
        return key, 0, "invalid", str(error)


_AUDIT_SETTINGS = None


def initialize_audit_worker(video_root, cache_dir, checkpoint_sha256, source_only):
    """Spawned read-only workers never construct an encoder or touch CUDA."""
    global _AUDIT_SETTINGS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _AUDIT_SETTINGS = (video_root, cache_dir, checkpoint_sha256, source_only)


def audit_worker(record):
    return audit_record(record, *_AUDIT_SETTINGS)


def audit_results(args, records, source_only=False):
    """Yield in manifest order, sharing exactly the same checks in both modes."""
    settings = (args.video_root, args.cache_dir, args.checkpoint_sha256, source_only)
    workers = getattr(args, "audit_workers", 1)
    chunksize = getattr(args, "audit_chunksize", 32)
    if workers < 1 or chunksize < 1:
        raise ValueError("Audit worker count and chunksize must be positive")
    if workers == 1:
        for record in records:
            yield audit_record(record, *settings)
    else:
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_audit_worker, initargs=settings) as executor:
            # map preserves input order; tensors stay local to each worker.
            yield from executor.map(audit_worker, records, chunksize=chunksize)


def audit(args, records, source_only=False):
    start = time.monotonic()
    metadata = common_metadata(args, records)
    valid_keys, errors = [], []
    counts = Counter()
    splits = Counter()
    total_tokens = 0
    for index, result in enumerate(audit_results(args, records, source_only)):
        key, tokens, error_kind, error_text = result
        if error_kind is None:
            valid_keys.append(key)
            splits[key.split("/")[0]] += 1
            total_tokens += tokens
        else:
            counts[error_kind] += 1
            errors.append({"clip_key": key, "error": error_text, "kind": error_kind})
        if (index + 1) % max(1, args.log_every * 100) == 0:
            print(json.dumps({"stage": "source_audit" if source_only else "audit", "checked": index + 1, "valid": len(valid_keys), **counts}), flush=True)
    report = {**metadata, "complete": len(valid_keys) == len(records) and not args.limit,
              "valid": len(valid_keys), "missing": counts["missing"], "invalid": counts["invalid"],
              "valid_by_split": dict(splits), "valid_keys": valid_keys, "errors": errors,
              "feature_tokens": total_tokens, "elapsed_seconds": time.monotonic() - start}
    output = args.cache_dir / ("source_coverage_report.json" if source_only else "coverage_report.json")
    if args.limit:
        output = output.with_name(output.stem + ".subset.json")
    atomic_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"valid_keys", "errors"}}, sort_keys=True), flush=True)
    print(f"Report: {output}", flush=True)
    return 0 if len(valid_keys) == len(records) else 1


def extract(args, records):
    metadata = prepare_cache_metadata(args, records, initialize=args.world_size == 1)
    selected = records[args.rank::args.world_size]
    counts = Counter()
    model = None
    start = time.monotonic()
    path = args.cache_dir / f"manifest.rank{args.rank:02d}.jsonl"
    with path.open("a", buffering=1) as log:
        for index, record in enumerate(selected):
            key = clip_key(record)
            video = args.video_root / (key + ".mp4")
            item_start = time.monotonic()
            entry = {"clip_key": key, "rank": args.rank, "world_size": args.world_size}
            try:
                reused = False
                if cache_path(args.cache_dir, key).is_file():
                    try:
                        payload = load_synchformer_payload(args.cache_dir, key, args.checkpoint_sha256, video_path=video)
                        if abs(payload["metadata"]["duration_seconds"] - float(record["duration_seconds"])) > 0.081:
                            raise ValueError("Cached RGB/audio duration mismatch")
                        reused = True
                    except Exception:
                        pass  # Invalid or incomplete cache is regenerated atomically.
                if not reused:
                    if model is None:
                        model = FrozenSynchformerExtractor(args.checkpoint, device=args.device,
                            batch_size=args.batch_size, expected_checkpoint_sha256=args.checkpoint_sha256)
                    payload = model.extract(video, key)
                    payload["metadata"]["inventory_sha256"] = metadata["inventory_sha256"]
                    payload["metadata"]["audio_duration_seconds"] = float(record["duration_seconds"])
                    if abs(payload["metadata"]["duration_seconds"] - float(record["duration_seconds"])) > 0.081:
                        raise ValueError(f"RGB/audio duration mismatch: {payload['metadata']['duration_seconds']} vs {record['duration_seconds']}")
                    save_synchformer_feature(args.cache_dir, key, payload)
                status = "reused" if reused else "created"
                counts[status] += 1
                entry.update(status=status, shape=list(payload["features"].shape), elapsed_seconds=time.monotonic()-item_start)
            except Exception as error:
                counts["failed"] += 1
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(json.dumps(entry), flush=True)
                if model is None:  # Missing/bad checkpoint is a global error, not 80k sample failures.
                    log.write(json.dumps(entry) + "\n")
                    raise
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            log.write(json.dumps(entry) + "\n")
            if (index + 1) % args.log_every == 0 or index == 0 or index + 1 == len(selected):
                print(json.dumps({"rank": args.rank, "done": index + 1, "total": len(selected),
                    **counts, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    summary = {**metadata, "rank": args.rank, "world_size": args.world_size, "assigned": len(selected),
               **counts, "elapsed_seconds": time.monotonic() - start}
    atomic_json(args.cache_dir / f"summary.rank{args.rank:02d}.json", summary)
    return int(counts["failed"] > 0)


def main():
    args = parser().parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size or args.batch_size < 1 or args.log_every < 1 or args.audit_workers < 1 or args.audit_chunksize < 1:
        raise ValueError("Invalid worker/batch/log arguments")
    torch.set_num_threads(args.num_threads)
    records = read_inventory(args.inventory)
    if args.limit:
        records = records[:args.limit]
    if args.initialize_only:
        metadata = prepare_cache_metadata(args, records, initialize=True)
        print(json.dumps({"stage": "initialized", "expected_count": metadata["expected_count"], "cache_dir": str(args.cache_dir)}), flush=True)
        return 0
    if args.audit_only or args.source_audit_only:
        return audit(args, records, source_only=args.source_audit_only)
    return extract(args, records)


if __name__ == "__main__":
    sys.exit(main())
