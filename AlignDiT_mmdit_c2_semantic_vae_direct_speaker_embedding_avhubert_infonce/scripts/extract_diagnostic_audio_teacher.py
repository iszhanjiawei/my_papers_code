"""Extract held-out audio-only AV-HuBERT targets into an isolated diagnostic cache.

Run with this snapshot's ``PYTHONPATH=src``. No generator, optimizer or training
cache is changed. Completed outputs are verified before being reused. A complete
matching run, including ``--verify-only``, does not load the encoder or use CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from aligndit.model.audio_teacher_cache import DEFAULT_AUDIO_TEACHER_IDENTITY, teacher_frame_lengths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def hubert_source(root: Path) -> Path:
    for candidate in (root, root / "avhubert", root / "avhubert/avhubert"):
        if (candidate / "hubert.py").is_file() and (candidate / "__init__.py").is_file():
            return candidate / "hubert.py"
    raise FileNotFoundError(f"Cannot locate AV-HuBERT user module beneath {root}")


def parse_args() -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    data_root = Path(prefix + "/zjw524/projects/data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Isolated diagnostic output directory")
    parser.add_argument("--teacher-module", type=Path, required=True, help="Donor avhubert_teacher.py")
    parser.add_argument(
        "--teacher-checkpoint", type=Path,
        default=Path(prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt"),
    )
    parser.add_argument("--avhubert-root", type=Path, default=data_root / "av_hubert")
    parser.add_argument(
        "--manifest", type=Path,
        default=data_root / "CelebVDub_svae1000k_sample_seed666_fp32/manifests/test.jsonl",
    )
    parser.add_argument("--test-list", type=Path, default=data_root / "celebvdub_test_s1.lst")
    parser.add_argument("--audio-root", type=Path, default=Path(prefix + "/zjw524/datasets/CelebV-Dub"))
    parser.add_argument("--expected-identity", default=DEFAULT_AUDIO_TEACHER_IDENTITY)
    parser.add_argument("--expected-count", type=int, default=213)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true", help="Validate a complete output without extraction")
    args = parser.parse_args()
    if min(args.expected_count, args.microbatch_size, args.torch_threads) < 1:
        parser.error("counts, microbatch size and thread count must be positive")
    for key in ("teacher_module", "teacher_checkpoint", "avhubert_root", "manifest", "test_list", "audio_root"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    return args


def prepare_records(args: argparse.Namespace) -> list[dict]:
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    clips = [line.strip() for line in args.test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_clip = {}
    for row in rows:
        key = row["utterance_key"]
        if not key.startswith("celebvdub/test/") or row.get("split") != "test":
            raise ValueError(f"Expected a held-out test record: {key}")
        clip = key.removeprefix("celebvdub/test/")
        if clip in by_clip:
            raise ValueError(f"Duplicate utterance: {key}")
        by_clip[clip] = row
    if len(rows) != args.expected_count or len(clips) != args.expected_count or set(clips) != set(by_clip):
        raise ValueError("Test manifest and list must contain the same expected number of unique utterances")
    prepared = []
    for clip in clips:
        row = by_clip[clip]
        relative = Path(row["audio_relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != f"test/{clip}.wav":
            raise ValueError(f"Invalid source relative path: {relative}")
        audio_path = (args.audio_root / relative).resolve(strict=True)
        audio_path.relative_to(args.audio_root)
        info = sf.info(audio_path)
        samples = int(row["original_num_samples"])
        if row.get("sample_rate") != 16000 or math.ceil(info.frames * 16000 / info.samplerate) != samples:
            raise ValueError(f"Resampled waveform length does not match the latent manifest: {audio_path}")
        stored, valid = teacher_frame_lengths(samples)
        if not 0 < valid <= stored:
            raise ValueError(f"No fully supported teacher frames: {audio_path}")
        feature_relative = Path("teacher_features") / relative.with_suffix(".npz")
        identity = file_identity(audio_path)
        prepared.append({
            "utterance_key": row["utterance_key"],
            "feature_path": str(args.output / feature_relative),
            "feature_relative_path": feature_relative.as_posix(),
            "stored_length": stored, "valid_length": valid,
            "source_waveform": str(audio_path), "source_sha256": sha256(audio_path),
            "source_size_bytes": identity["size_bytes"], "source_mtime_ns": identity["mtime_ns"],
            "source_sample_rate": info.samplerate, "source_channels": info.channels,
            "source_samples": info.frames, "resampled": info.samplerate != 16000,
            "mono_averaged": info.channels != 1, "resampled_samples": samples,
        })
    return prepared


def validate_feature(record: dict, expected_hash: str | None = None) -> np.ndarray:
    path = Path(record["feature_path"])
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing regular diagnostic feature: {path}")
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Diagnostic feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        if features.dtype != np.float16 or features.shape != (record["stored_length"], 1024):
            raise ValueError(f"Unexpected diagnostic feature shape/dtype: {path}")
        if not np.isfinite(features).all():
            raise ValueError(f"Nonfinite diagnostic feature: {path}")
        for key in ("stored_length", "valid_length"):
            if int(data[key]) != record[key]:
                raise ValueError(f"Diagnostic {key} mismatch: {path}")
        if str(data["utterance_key"]) != record["utterance_key"]:
            raise ValueError(f"Diagnostic utterance mismatch: {path}")
        return features.copy()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    records = prepare_records(args)
    manifest_path = args.output / "teacher_manifest.json"
    provenance = {
        "teacher_identity": args.expected_identity,
        "donor_helper_sha256": sha256(args.teacher_module),
        "input_manifest_sha256": sha256(args.manifest),
        "test_list_sha256": sha256(args.test_list),
    }
    completed = {}
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in provenance.items():
            if previous.get(key) != value:
                raise ValueError(f"Existing diagnostic provenance differs: {key}")
        metadata = previous["teacher_metadata"]
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if identity != args.expected_identity or metadata["checkpoint"] != file_identity(args.teacher_checkpoint):
            raise ValueError("Existing teacher metadata/checkpoint identity differs")
        if metadata["hubert_source_sha256"] != sha256(hubert_source(args.avhubert_root)):
            raise ValueError("AV-HuBERT encoder source changed")
        old_records = {r["utterance_key"]: r for r in previous["records"]}
        if len(old_records) != len(previous["records"]) or set(old_records) - {r["utterance_key"] for r in records}:
            raise ValueError("Existing diagnostic manifest contains duplicate or unexpected utterances")
        for record in records:
            old = old_records.get(record["utterance_key"])
            if old is None:
                continue
            for key, value in record.items():
                if key != "feature_path" and old.get(key) != value:
                    raise ValueError(f"Existing source/feature contract differs for {record['utterance_key']}: {key}")
            validate_feature(record, old["feature_sha256"])
            completed[record["utterance_key"]] = {**record, "feature_sha256": old["feature_sha256"]}
    pending = [r for r in records if r["utterance_key"] not in completed]
    if not pending:
        print(json.dumps({"event": "verified_complete", "count": len(records), "encoder_loaded": False,
                          "manifest": str(manifest_path), "elapsed_seconds": time.monotonic() - started}))
        return
    if args.verify_only:
        raise RuntimeError(f"Diagnostic output is incomplete: {len(pending)} records require extraction")

    import torch

    torch.set_num_threads(args.torch_threads)
    spec = importlib.util.spec_from_file_location("diagnostic_avhubert_teacher", args.teacher_module)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = helper.AVHubertAudioTeacher(
        checkpoint_path=str(args.teacher_checkpoint), avhubert_root=str(args.avhubert_root),
        device=args.device, cache_dir=str(args.output / "teacher_cache"), microbatch_size=args.microbatch_size,
    )
    if helper._identity_key(teacher.identity) != args.expected_identity:
        raise ValueError("Extracted teacher identity differs from the pinned training teacher")
    report = {
        **provenance, "status": "in_progress", "teacher_metadata": teacher.identity,
        "teacher_cache_root": str(teacher.cache_dir), "donor_helper": str(args.teacher_module),
        "input_manifest": str(args.manifest), "test_list": str(args.test_list),
        "scope": "Exploratory post-hoc diagnosis on existing held-out test split; no training or training-cache mutation",
        "stored_dtype": "float16", "valid_frame_rule": "stack4_support_640j_to_640j_plus_880_within_original_samples",
        "extraction_device": args.device, "microbatch_size": args.microbatch_size,
    }
    for start in range(0, len(pending), args.microbatch_size):
        batch = pending[start:start + args.microbatch_size]
        features, lengths = teacher.encode([r["source_waveform"] for r in batch])
        for index, record in enumerate(batch):
            stored = int(lengths[index])
            if stored != record["stored_length"]:
                raise ValueError(f"Teacher/manifest length mismatch: {record['utterance_key']}")
            array = features[index, :stored].float().cpu().numpy().astype(np.float16)
            if array.shape != (stored, 1024) or not np.isfinite(array).all():
                raise ValueError(f"Invalid extracted teacher values: {record['utterance_key']}")
            path = Path(record["feature_path"])
            if path.exists():
                # Recover an interruption between NPZ publication and manifest update.
                if not np.array_equal(validate_feature(record), array):
                    raise ValueError(f"Refusing to replace a different existing feature: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                try:
                    with temporary.open("wb") as handle:
                        np.savez(handle, features=array, valid_length=np.int64(record["valid_length"]),
                                 stored_length=np.int64(stored), utterance_key=np.str_(record["utterance_key"]))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            validate_feature(record)
            completed[record["utterance_key"]] = {**record, "feature_sha256": sha256(path)}
        del features, lengths
        report.update(
            records=[completed[r["utterance_key"]] for r in records if r["utterance_key"] in completed],
            count=len(completed), resampled_count=sum(r["resampled"] for r in completed.values()),
            mono_averaged_count=sum(r["mono_averaged"] for r in completed.values()),
            elapsed_seconds=time.monotonic() - started,
            status="complete" if len(completed) == len(records) else "in_progress",
        )
        atomic_json(manifest_path, report)
        print(json.dumps({"event": report["status"], "completed": len(completed), "total": len(records),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
