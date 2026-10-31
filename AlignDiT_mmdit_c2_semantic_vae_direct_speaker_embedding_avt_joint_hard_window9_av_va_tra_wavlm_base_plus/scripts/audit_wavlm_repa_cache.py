"""Read every frozen WavLM target and validate exact training-manifest coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aligndit.model.repa import validate_repa_cache_metadata, validate_repa_feature_array
from aligndit.model.semantic_vae_dataset import CELEBVDUB_TRAIN_COUNT, _safe_join


def audit_cache(cache_dir: Path, manifest_path: Path, *, workers: int = 16) -> dict:
    started = time.monotonic()
    manifest_bytes = manifest_path.read_bytes()
    records = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    metadata = validate_repa_cache_metadata(
        cache_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_count=CELEBVDUB_TRAIN_COUNT,
    )
    if len(records) != CELEBVDUB_TRAIN_COUNT:
        raise RuntimeError(f"Expected {CELEBVDUB_TRAIN_COUNT} training records, got {len(records)}")
    source_ids = [record["utterance_key"] for record in records]
    paths = [Path(record["audio_relative_path"]) for record in records]
    if len(set(source_ids)) != len(records) or len(set(paths)) != len(records):
        raise RuntimeError("Duplicate training source IDs or audio paths")
    if any(path.is_absolute() or path.parts[0] != "train" or path.suffix != ".wav" for path in paths):
        raise RuntimeError("REPA source paths must be relative train/*.wav paths")
    expected = {str(path.with_suffix(".npy")) for path in paths}
    actual = {str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*.npy")}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(f"REPA coverage differs: missing={missing[:10]}, extra={extra[:10]}")
    total_frames, valid = 0, 0
    minimum_frames, maximum_frames = None, 0
    def validate_feature(path: Path) -> int:
        feature_path = _safe_join(cache_dir, str(path.with_suffix(".npy")), label="REPA cache feature")
        if feature_path.is_symlink() or not feature_path.is_file():
            raise RuntimeError(f"REPA feature must be a regular file: {feature_path}")
        feature = np.load(feature_path, allow_pickle=False)
        validate_repa_feature_array(feature, source=feature_path)
        return feature.shape[0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for frames in executor.map(validate_feature, paths):
            total_frames += frames
            minimum_frames = frames if minimum_frames is None else min(minimum_frames, frames)
            maximum_frames = max(maximum_frames, frames)
            valid += 1
            if valid % 5000 == 0:
                print(f"Validated REPA features {valid}/{len(records)}", flush=True)
    coverage = json.loads((cache_dir / "coverage_report.json").read_text())
    if total_frames != coverage.get("total_frames"):
        raise RuntimeError(f"Total-frame mismatch: actual={total_frames}, coverage={coverage.get('total_frames')}")
    return {
        "complete": True,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "expected": len(records),
        "valid": valid,
        "missing": [],
        "extra": [],
        "invalid": [],
        "unique_source_ids": len(set(source_ids)),
        "source_ids_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "total_frames": total_frames,
        "minimum_frames": minimum_frames,
        "maximum_frames": maximum_frames,
        "metadata_sha256": hashlib.sha256((cache_dir / "metadata.json").read_bytes()).hexdigest(),
        "coverage_report_sha256": hashlib.sha256((cache_dir / "coverage_report.json").read_bytes()).hexdigest(),
        "teacher_metadata": metadata,
        "checks": ["exact complete source-ID coverage", "all shapes [frames,768]", "all float16", "all finite"],
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    root_prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(root_prefix + "/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16"))
    parser.add_argument("--manifest", type=Path,
                        default=Path(root_prefix + "/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32/manifests/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = audit_cache(args.cache_dir.resolve(strict=True), args.manifest.resolve(strict=True), workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
