"""Audit existing teacher-cache coverage; never compute or repair features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aligndit.model.audio_teacher_cache import DEFAULT_AUDIO_TEACHER_IDENTITY, AudioTeacherCache


def main():
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(prefix + "/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32/manifests/train.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(prefix + "/zjw524/projects/data/CelebVDub/avhubert_audio_teacher_cache")
        / DEFAULT_AUDIO_TEACHER_IDENTITY,
    )
    parser.add_argument("--audio-root", type=Path, default=Path(prefix + "/zjw524/datasets/CelebV-Dub"))
    parser.add_argument("--expected-identity", default=DEFAULT_AUDIO_TEACHER_IDENTITY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=0, help="0 checks all; otherwise use even-spacing samples")
    parser.add_argument(
        "--check-features", action="store_true", help="Also load each selected NPZ and check its values"
    )
    parser.add_argument("--output", type=Path, help="Optional audit JSON report; no cache file is ever written")
    args = parser.parse_args()
    if args.workers < 1 or args.sample_count < 0:
        parser.error("workers must be positive and sample-count nonnegative")
    started = time.monotonic()
    manifest_bytes = args.manifest.read_bytes()
    records = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    selected = records
    if 0 < args.sample_count < len(records):
        if args.sample_count == 1:
            selected = [records[0]]
        else:
            selected = [records[i * (len(records) - 1) // (args.sample_count - 1)] for i in range(args.sample_count)]
    reader = AudioTeacherCache(args.cache_dir, args.audio_root, expected_identity=args.expected_identity)
    cache_keys = {path.stem for path in reader.cache_dir.glob("*/*.npz")}

    def inspect(record):
        try:
            path, _, _, _ = reader.entry_for_record(record)
            if path.stem not in cache_keys or path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"Missing regular cache file: {path}")
            if args.check_features:
                reader.load(record)
            return path.stem, None
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return None, {"utterance_key": record.get("utterance_key"), "error": str(error)}

    hits = set()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for cache_key, error in pool.map(inspect, selected):
            if error is not None:
                failures.append(error)
            else:
                hits.add(cache_key)
    report = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_records": len(records),
        "checked_records": len(selected),
        "cache_npz_count": len(cache_keys),
        "matched_unique_cache_keys": len(hits),
        "missing_or_invalid_count": len(failures),
        "check_feature_contents": args.check_features,
        "extra_cache_keys": len(cache_keys - hits) if len(selected) == len(records) else None,
        "failures": failures,
        "teacher_contract": reader.contract,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
