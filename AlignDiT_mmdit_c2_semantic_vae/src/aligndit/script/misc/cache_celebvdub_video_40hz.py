"""Cache exact-length 40-Hz CelebV-Dub AV-HuBERT features with an auditable protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import Self

from aligndit.script.misc.svae_cache_utils import (
    CACHE_SCHEMA_VERSION,
    JsonlProgressWriter,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npy,
    canonical_json,
    durable_unlink,
    prefixed_path,
    quarantine_file,
    read_append_only_jsonl,
    read_jsonl,
    safe_join,
    sha256_file,
    stable_utterance_seed,
    validate_attempt_id,
    validate_npy,
)


FEATURE = "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1"
VIDEO_DIM = 1024
EXPECTED_MANIFEST_SHA256 = "a6478cce785748cbcefd87af54eafa9f654d735afa1c41b8f846e041cbc1286d"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class VideoProgressEntry:
    utterance_key: str
    relative_path: str
    sha256: str
    size_bytes: int
    source_frames: int
    source_sha256: str
    source_size_bytes: int
    target_frames: int

    def as_record(self) -> dict[str, Any]:
        return {
            "feature": FEATURE,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_frames": self.source_frames,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "target_frames": self.target_frames,
            "utterance_key": self.utterance_key,
            "video_dim": VIDEO_DIM,
        }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub/avhubert_video_feat"),
    )
    parser.add_argument("--attempt-id", default=os.environ.get("VIDEO40CACHE_ATTEMPT_ID"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="Development-only: process the first N manifest records.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--progress-fsync-interval", type=int, default=100)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--distributed-timeout-minutes", type=int, default=24 * 60)
    parser.add_argument(
        "--acknowledge-stale-write-attempt",
        help="Replace a stale guard only after confirming that its recorded writer is no longer alive.",
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def initialize_distributed(device_choice: str, timeout_minutes: int) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed coordinates rank={rank}, world_size={world_size}")
    use_cuda = device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available() or not 0 <= local_rank < torch.cuda.device_count():
            raise RuntimeError(f"CUDA worker {local_rank} is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        initialized_here = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def runtime_contract(context: DistributedContext) -> dict[str, Any]:
    device: dict[str, Any] = {"type": context.device.type}
    if context.device.type == "cuda":
        properties = torch.cuda.get_device_properties(context.local_rank)
        device.update({"capability": [properties.major, properties.minor], "name": properties.name})
    return {
        "device": device,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    required = {
        "latent_frames",
        "utterance_key",
        "video_40hz_relative_path",
        "video_dim",
        "video_frames_25hz",
        "video_relative_path",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Manifest record is missing video cache fields {sorted(missing)}: {record}")
    stable_utterance_seed(0, str(record["utterance_key"]))
    if int(record["video_dim"]) != VIDEO_DIM or int(record["video_frames_25hz"]) <= 0:
        raise ValueError(f"Invalid source video contract: {record}")
    if int(record["latent_frames"]) <= 0:
        raise ValueError(f"Invalid target video length: {record}")
    source = Path(str(record["video_relative_path"]))
    destination = Path(str(record["video_40hz_relative_path"]))
    for name, path in (("source", source), ("destination", destination)):
        if (
            path.is_absolute()
            or not path.parts
            or path.as_posix() != str(record[f"video_{'relative' if name == 'source' else '40hz_relative'}_path"])
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in path.as_posix()
            or path.suffix != ".npy"
        ):
            raise ValueError(f"Invalid {name} video path for {record['utterance_key']}: {path}")
    if destination.parts[0] != "video_40hz":
        raise ValueError(f"40-Hz output must stay below video_40hz/: {destination}")


def build_spec(
    args: argparse.Namespace,
    context: DistributedContext,
    manifest: Path,
    video_root: Path,
) -> dict[str, Any]:
    metadata_path = manifest.parent / "inventory_meta.json"
    metadata = read_json_object(metadata_path)
    entry = metadata.get("manifests", {}).get(manifest.name)
    if not isinstance(entry, dict):
        raise TypeError(f"Manifest is not registered by {metadata_path}: {manifest.name}")
    actual_manifest_hash = sha256_file(manifest)
    if entry.get("sha256") != actual_manifest_hash or actual_manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Manifest hash mismatch: metadata={entry.get('sha256')}, actual={actual_manifest_hash}, "
            f"expected={args.expected_manifest_sha256}"
        )
    video_spec = metadata.get("video_feature_spec", {})
    expected_video_spec = {"dimension": VIDEO_DIM, "dtype": "float32", "frame_rate_hz": 25, "root": str(video_root)}
    if video_spec != expected_video_spec:
        raise RuntimeError(f"Manifest AV-HuBERT source contract mismatch: {video_spec} != {expected_video_spec}")
    script_path = Path(__file__).resolve(strict=True)
    utility_path = script_path.with_name("svae_cache_utils.py").resolve(strict=True)
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "execution": runtime_contract(context),
        "feature": FEATURE,
        "interpolation": {
            "align_corners": False,
            "input_dtype": "float32",
            "input_frame_rate_hz": 25,
            "input_layout": "[time,channel]",
            "mode": "linear",
            "output_dtype": "float32",
            "output_frame_rate_hz": 40,
            "output_layout": "[time,channel]",
            "size": "record.latent_frames",
            "video_dim": VIDEO_DIM,
        },
        "manifest": {
            "count": int(entry["count"]),
            "inventory_metadata_path": str(metadata_path),
            "inventory_metadata_sha256": sha256_file(metadata_path),
            "path": str(manifest),
            "sha256": actual_manifest_hash,
            "total_target_frames": int(metadata["total_latent_frames"]),
        },
        "source": {
            "root": str(video_root),
        },
        "source_code": {
            "cache_utility_path": str(utility_path),
            "cache_utility_sha256": sha256_file(utility_path),
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
        },
    }


def publish_or_validate_spec(
    args: argparse.Namespace,
    context: DistributedContext,
    manifest: Path,
    video_root: Path,
) -> dict[str, Any]:
    spec_path = args.cache_root.absolute() / "state" / "video_40hz" / "spec.json"
    if context.is_main:
        proposed = build_spec(args, context, manifest, video_root)
        if args.validate_only:
            stored = read_json_object(spec_path)
            if stored != proposed:
                raise RuntimeError("Stored video-40Hz spec differs from the current immutable resources/runtime")
        else:
            result = atomic_write_json(spec_path, proposed)
            print(f"[rank 0] video-40Hz spec {'published' if result.created else 'verified'}: {result.path}")
    barrier(context)
    spec = read_json_object(spec_path)
    if spec.get("execution") != runtime_contract(context):
        raise RuntimeError(f"Rank {context.rank} runtime differs from the immutable video cache spec")
    for path_key, hash_key in (
        ("script_path", "script_sha256"),
        ("cache_utility_path", "cache_utility_sha256"),
    ):
        path = Path(spec["source_code"][path_key])
        if sha256_file(path) != spec["source_code"][hash_key]:
            raise RuntimeError(f"Video cache source code changed after spec publication: {path}")
    if sha256_file(Path(spec["manifest"]["path"])) != spec["manifest"]["sha256"]:
        raise RuntimeError("Video cache manifest changed after spec publication")
    if sha256_file(Path(spec["manifest"]["inventory_metadata_path"])) != spec["manifest"]["inventory_metadata_sha256"]:
        raise RuntimeError("Video cache inventory metadata changed after spec publication")
    return spec


def acquire_guard(args: argparse.Namespace, context: DistributedContext) -> Path | None:
    if args.validate_only:
        return None
    guard_path = args.cache_root.absolute() / "state" / "video_40hz" / "WRITE_ACTIVE.json"
    if context.is_main:
        if guard_path.exists() or guard_path.is_symlink():
            guard = read_json_object(guard_path)
            guarded_attempt = guard.get("attempt_id")
            if args.acknowledge_stale_write_attempt != guarded_attempt:
                raise RuntimeError(
                    f"Video cache has an active/stale writer {guarded_attempt!r}; after confirming it is dead, pass "
                    f"--acknowledge-stale-write-attempt {guarded_attempt} with a new attempt id"
                )
            if args.attempt_id == guarded_attempt:
                raise ValueError("A stale guard must be replaced with a different attempt id")
            quarantine_root = args.cache_root.absolute() / "quarantine" / "video_40hz_write_guards" / args.attempt_id
            quarantine_file(guard_path, quarantine_root, "WRITE_ACTIVE.json")
        result = atomic_write_json(
            guard_path,
            {
                "attempt_id": args.attempt_id,
                "feature": FEATURE,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "world_size": context.world_size,
            },
        )
        if not result.created:
            raise RuntimeError(f"Video cache write guard already exists: {guard_path}")
    barrier(context)
    if read_json_object(guard_path).get("attempt_id") != args.attempt_id:
        raise RuntimeError("Video cache write guard belongs to a different attempt")
    return guard_path


def release_guard(guard_path: Path | None, context: DistributedContext) -> None:
    if guard_path is None:
        return
    barrier(context)
    if context.is_main:
        durable_unlink(guard_path)
    barrier(context)


def parse_progress(record: Mapping[str, Any], source: Path) -> VideoProgressEntry:
    required = {
        "feature",
        "relative_path",
        "sha256",
        "size_bytes",
        "source_frames",
        "source_sha256",
        "source_size_bytes",
        "target_frames",
        "utterance_key",
        "video_dim",
    }
    if set(record) != required or record.get("feature") != FEATURE or record.get("video_dim") != VIDEO_DIM:
        raise ValueError(f"Invalid video progress schema in {source}: {record}")
    for field in ("sha256", "source_sha256"):
        if not isinstance(record[field], str) or not SHA256_PATTERN.fullmatch(record[field]):
            raise ValueError(f"Invalid {field} in {source}: {record[field]!r}")
    entry = VideoProgressEntry(
        utterance_key=str(record["utterance_key"]),
        relative_path=str(record["relative_path"]),
        sha256=str(record["sha256"]),
        size_bytes=int(record["size_bytes"]),
        source_frames=int(record["source_frames"]),
        source_sha256=str(record["source_sha256"]),
        source_size_bytes=int(record["source_size_bytes"]),
        target_frames=int(record["target_frames"]),
    )
    stable_utterance_seed(0, entry.utterance_key)
    if min(entry.size_bytes, entry.source_frames, entry.source_size_bytes, entry.target_frames) <= 0:
        raise ValueError(f"Invalid video progress dimensions in {source}: {record}")
    path = Path(entry.relative_path)
    if path.is_absolute() or path.parts[0] != "video_40hz" or path.suffix != ".npy":
        raise ValueError(f"Invalid video progress output path in {source}: {path}")
    return entry


def load_progress_index(cache_root: Path) -> dict[str, VideoProgressEntry]:
    index: dict[str, VideoProgressEntry] = {}
    paths: list[tuple[Path, bool]] = []
    consolidated = cache_root / "state" / "video_40hz" / "index.jsonl"
    if consolidated.exists() or consolidated.is_symlink():
        paths.append((consolidated, False))
    progress_root = cache_root / "state" / "video_40hz" / "progress"
    if progress_root.exists():
        if not progress_root.is_dir() or progress_root.is_symlink():
            raise RuntimeError(f"Video progress root is not a regular directory: {progress_root}")
        paths.extend((path, True) for path in sorted(progress_root.glob("*.jsonl")))
    for path, append_only in paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Video progress/index must be a regular file: {path}")
        iterator = read_append_only_jsonl(path) if append_only else read_jsonl(path)
        for record in iterator:
            entry = parse_progress(record, path)
            previous = index.get(entry.utterance_key)
            if previous is not None and previous != entry:
                raise RuntimeError(f"Conflicting video cache progress for {entry.utterance_key}")
            index[entry.utterance_key] = entry
    return index


def selected_records(manifest: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(read_jsonl(manifest)):
        if limit is not None and index >= limit:
            break
        yield record


def load_local_records(
    manifest: Path,
    limit: int | None,
    context: DistributedContext,
) -> tuple[list[dict[str, Any]], int]:
    local: list[dict[str, Any]] = []
    count = 0
    seen: set[str] = set()
    for selected_index, record in enumerate(selected_records(manifest, limit)):
        validate_manifest_record(record)
        key = str(record["utterance_key"])
        if key in seen:
            raise ValueError(f"Duplicate manifest key: {key}")
        seen.add(key)
        if selected_index % context.world_size == context.rank:
            local.append(record)
        count += 1
    expected_local = max(0, (count + context.world_size - 1 - context.rank) // context.world_size)
    if len(local) != expected_local:
        raise AssertionError(f"Video shard mismatch on rank {context.rank}: {len(local)} != {expected_local}")
    return local, count


def interpolate_video(source: np.ndarray, target_frames: int, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.array(source, dtype=np.float32, copy=True)).transpose(0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        aligned = F.interpolate(tensor, size=target_frames, mode="linear", align_corners=False)
    output = aligned.squeeze(0).transpose(0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)
    if output.shape != (target_frames, VIDEO_DIM) or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid interpolated AV-HuBERT output: {output.shape}/{output.dtype}")
    return output


def process_record(
    record: Mapping[str, Any],
    *,
    video_root: Path,
    cache_root: Path,
    device: torch.device,
    known: VideoProgressEntry | None,
    validate_only: bool,
) -> tuple[VideoProgressEntry, str]:
    source_path = safe_join(video_root, str(record["video_relative_path"]))
    if not source_path.is_file() or source_path.is_symlink():
        raise FileNotFoundError(f"AV-HuBERT source is not a regular file: {source_path}")
    source = np.load(source_path, allow_pickle=False, mmap_mode="r")
    expected_source_shape = (int(record["video_frames_25hz"]), VIDEO_DIM)
    if source.shape != expected_source_shape or source.dtype != np.float32 or not np.isfinite(source).all():
        raise ValueError(f"Invalid AV-HuBERT source {source_path}: {source.shape}/{source.dtype}")
    source_sha256 = sha256_file(source_path)
    source_size_bytes = source_path.stat().st_size
    destination = safe_join(cache_root, str(record["video_40hz_relative_path"]))
    target_frames = int(record["latent_frames"])
    expected_common = {
        "relative_path": str(record["video_40hz_relative_path"]),
        "source_frames": expected_source_shape[0],
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
        "target_frames": target_frames,
        "utterance_key": str(record["utterance_key"]),
    }
    if destination.exists() or destination.is_symlink():
        validation = validate_npy(
            destination,
            expected_shape=(target_frames, VIDEO_DIM),
            expected_dtype=np.float32,
        )
        entry = VideoProgressEntry(
            sha256=validation.sha256,
            size_bytes=validation.size_bytes,
            **expected_common,
        )
        if known is not None:
            if entry != known:
                raise RuntimeError(f"Stored video output/progress mismatch for {record['utterance_key']}")
            return entry, "validated" if validate_only else "resumed"
        if validate_only:
            raise RuntimeError(f"Unauthenticated video output has no progress entry: {destination}")
        expected = interpolate_video(source, target_frames, device)
        existing = np.load(destination, allow_pickle=False)
        if not np.array_equal(existing, expected):
            raise RuntimeError(f"Unauthenticated video output differs from deterministic interpolation: {destination}")
        return entry, "recovered"
    if known is not None:
        raise FileNotFoundError(f"Video progress points to a missing output: {destination}")
    if validate_only:
        raise FileNotFoundError(f"Video output is missing during read-only validation: {destination}")
    output = interpolate_video(source, target_frames, device)
    result = atomic_write_npy(destination, output)
    return (
        VideoProgressEntry(
            sha256=result.sha256,
            size_bytes=result.size_bytes,
            **expected_common,
        ),
        "created" if result.created else "recovered",
    )


def scan_output_paths(cache_root: Path) -> set[str]:
    output_root = cache_root / "video_40hz"
    if not output_root.exists():
        return set()
    if not output_root.is_dir() or output_root.is_symlink():
        raise RuntimeError(f"Video-40Hz output root is not a regular directory: {output_root}")
    paths: set[str] = set()
    for directory, directory_names, filenames in os.walk(output_root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise RuntimeError(f"Symlink directory below video-40Hz cache: {directory_path}")
        for name in filenames:
            path = directory_path / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Non-regular video cache output: {path}")
            paths.add(path.relative_to(cache_root).as_posix())
    return paths


def finalize(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    manifest: Path,
    selected_count: int,
    known: Mapping[str, VideoProgressEntry],
) -> dict[str, Any]:
    selection = {"mode": "first_n", "limit": args.limit} if args.limit is not None else {"mode": "full"}
    records: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    expected_keys: set[str] = set()
    total_source_frames = 0
    total_target_frames = 0
    total_size_bytes = 0
    digest = hashlib.sha256()
    index_size_bytes = 0
    for record in selected_records(manifest, args.limit):
        key = str(record["utterance_key"])
        entry = known.get(key)
        if entry is None:
            raise RuntimeError(f"Missing video cache progress after extraction: {key}")
        if entry.relative_path != record["video_40hz_relative_path"] or entry.target_frames != record["latent_frames"]:
            raise RuntimeError(f"Video cache progress/manifest mismatch: {key}")
        index_record = entry.as_record()
        encoded = f"{canonical_json(index_record)}\n".encode()
        digest.update(encoded)
        index_size_bytes += len(encoded)
        records.append(index_record)
        expected_keys.add(key)
        expected_paths.add(entry.relative_path)
        total_source_frames += entry.source_frames
        total_target_frames += entry.target_frames
        total_size_bytes += entry.size_bytes
    if len(records) != selected_count:
        raise RuntimeError(f"Video cache final count mismatch: {len(records)} != {selected_count}")
    is_full = args.limit is None
    if is_full:
        if set(known) != expected_keys or scan_output_paths(args.cache_root.absolute()) != expected_paths:
            raise RuntimeError("Full video cache contains unexpected progress keys or output paths")
        if total_target_frames != int(spec["manifest"]["total_target_frames"]):
            raise RuntimeError("Full video cache target-frame total differs from the immutable inventory")
    summary: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "count": len(records),
        "feature": FEATURE,
        "manifest_sha256": spec["manifest"]["sha256"],
        "ordered_index_sha256": digest.hexdigest(),
        "selection": selection,
        "spec_sha256": sha256_file(args.cache_root.absolute() / "state" / "video_40hz" / "spec.json"),
        "total_npy_size_bytes": total_size_bytes,
        "total_source_frames": total_source_frames,
        "total_target_frames": total_target_frames,
    }
    if not is_full:
        return summary
    index_path = args.cache_root.absolute() / "state" / "video_40hz" / "index.jsonl"
    if args.validate_only:
        if index_path.stat().st_size != index_size_bytes or sha256_file(index_path) != digest.hexdigest():
            raise RuntimeError("Stored video cache consolidated index differs from deterministic manifest order")
    else:
        result = atomic_write_jsonl(index_path, records)
        if result.sha256 != digest.hexdigest() or result.size_bytes != index_size_bytes:
            raise AssertionError("Published video cache index differs from its in-memory digest")
    summary["consolidated_index"] = {
        "count": len(records),
        "path": "state/video_40hz/index.jsonl",
        "sha256": digest.hexdigest(),
        "size_bytes": index_size_bytes,
    }
    return summary


def main() -> None:
    args = get_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}")
    if args.progress_fsync_interval <= 0 or args.distributed_timeout_minutes <= 0:
        raise ValueError("Progress fsync interval and distributed timeout must be positive")
    if not SHA256_PATTERN.fullmatch(args.expected_manifest_sha256):
        raise ValueError(f"Invalid expected manifest SHA256: {args.expected_manifest_sha256!r}")
    if args.attempt_id is None and not args.validate_only:
        raise ValueError("--attempt-id (or VIDEO40CACHE_ATTEMPT_ID) is required for writing")
    if args.attempt_id is not None:
        args.attempt_id = validate_attempt_id(args.attempt_id)
    if args.acknowledge_stale_write_attempt is not None:
        args.acknowledge_stale_write_attempt = validate_attempt_id(args.acknowledge_stale_write_attempt)
    args.cache_root = args.cache_root.expanduser().absolute()
    manifest = (args.manifest or args.cache_root / "manifests" / "inventory.jsonl").expanduser().resolve(strict=True)
    video_root = args.video_root.expanduser().resolve(strict=True)

    context = initialize_distributed(args.device, args.distributed_timeout_minutes)
    guard_path: Path | None = None
    try:
        spec = publish_or_validate_spec(args, context, manifest, video_root)
        guard_path = acquire_guard(args, context)
        local_records, selected_count = load_local_records(manifest, args.limit, context)
        if args.limit is None and selected_count != int(spec["manifest"]["count"]):
            raise RuntimeError("Full video selection count differs from the immutable inventory")
        known = load_progress_index(args.cache_root)
        counters = {"created": 0, "recovered": 0, "resumed": 0, "validated": 0}
        if args.validate_only:
            writer: Any = _NullProgressWriter()
        else:
            progress_path = (
                args.cache_root
                / "state"
                / "video_40hz"
                / "progress"
                / (f"{args.attempt_id}.rank-{context.rank:05d}-of-{context.world_size:05d}.jsonl")
            )
            writer = JsonlProgressWriter(progress_path, args.progress_fsync_interval)
        with writer:
            iterator = tqdm(
                local_records,
                desc=f"Video 25->40 Hz rank {context.rank}",
                disable=not context.is_main,
                dynamic_ncols=True,
            )
            for record in iterator:
                entry, status = process_record(
                    record,
                    video_root=video_root,
                    cache_root=args.cache_root,
                    device=context.device,
                    known=known.get(record["utterance_key"]),
                    validate_only=args.validate_only,
                )
                counters[status] += 1
                if not args.validate_only:
                    writer.append(entry.as_record())
                known[entry.utterance_key] = entry
        print(f"[rank {context.rank}] video-40Hz complete local={len(local_records)} counters={counters}", flush=True)
        barrier(context)
        if context.is_main:
            merged = load_progress_index(args.cache_root)
            summary = finalize(args, spec, manifest, selected_count, merged)
            if args.limit is None:
                complete_path = args.cache_root / "state" / "video_40hz" / "complete.json"
                if args.validate_only:
                    if read_json_object(complete_path) != summary:
                        raise RuntimeError("Stored video cache completion marker differs from read-only validation")
                else:
                    result = atomic_write_json(complete_path, summary)
                    print(
                        f"[rank 0] video-40Hz completion {'published' if result.created else 'verified'}: {result.path}",
                        flush=True,
                    )
            else:
                print(f"[rank 0] partial video-40Hz selection validated: {summary}", flush=True)
        release_guard(guard_path, context)
        guard_path = None
    finally:
        if context.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


class _NullProgressWriter:
    def __enter__(self) -> Self:
        return self

    def append(self, record: Mapping[str, Any]) -> None:
        raise RuntimeError("Read-only validation cannot append progress")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


if __name__ == "__main__":
    main()
