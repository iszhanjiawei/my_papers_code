"""Extract a deterministic, auditable Semantic-VAE posterior-sample cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
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
import torchaudio
from torch import nn
from tqdm import tqdm
from typing_extensions import Self

from aligndit.script.misc.svae_cache_utils import (
    BASE_POSTERIOR_SEED,
    CACHE_SCHEMA_VERSION,
    SAMPLE_RATE,
    SEMANTIC_VAE_HOP_LENGTH,
    SEMANTIC_VAE_LATENT_DIM,
    JsonlProgressWriter,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npy,
    canonical_json,
    durable_unlink,
    expected_latent_frames,
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


EXTRACTION_PROTOCOL = "semantic_vae_posterior_sample_v1"
OFFICIAL_EMA_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
OFFICIAL_METAINFO_SHA256 = "24b8ff09360cdfe8a38e61862bf185c2130ef45a15e6f235bbbae8af8065c851"
OFFICIAL_CONFIG_SHA256 = "c12ba35b4035f97808dabaac4f254bd4e32b1dc5fba0840168ae0c41859d0235"
OFFICIAL_BIGVGAN_CONFIG_SHA256 = "a11e013f623eedc55b2410d48cbd810322df03658377806d16ab396369525618"
OFFICIAL_MANIFEST_SHA256 = "65c1332f9852bb84ddba8cfef8359cf5f2c7195a593d4e24087eb6c60d1dabe5"
OFFICIAL_SEMANTIC_VAE_COMMIT = "5bcca91fe8b65c0e52c5ee141968f98662dc4792"
OFFICIAL_EMA_STEP = 1_000_014
LIGHT_STATE_CONTRACT_SHA256 = "470cd7036e6c296855a89e971c02142edba0c564f868cb6a86cfbd54037b5366"
GOLDEN_UTTERANCE_KEY = "train-clean-100/103/1240/103-1240-0015"
GOLDEN_POSTERIOR_SEED = 3_920_034_511_769_737_100
GOLDEN_RAW_LATENT_SHA256 = "e3de5ff47682f97e063c6aaeaee9cec195ebdb34e1bce964c4a10d2912114f3f"
CHECKPOINT_GROUP_COUNTS = {"decoder": 783, "decoder_proj": 22, "projectors": 10, "posterior": 145}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MANIFEST_FIELDS = {
    "audio_relative_path",
    "latent_dim",
    "latent_frames",
    "latent_relative_path",
    "num_channels",
    "original_num_samples",
    "padded_num_samples",
    "posterior_seed",
    "sample_rate",
    "split",
    "subset",
    "utterance_key",
}


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
class ProgressEntry:
    utterance_key: str
    relative_path: str
    sha256: str
    size_bytes: int
    latent_frames: int
    latent_dim: int

    def as_record(self) -> dict[str, Any]:
        return {
            "feature": EXTRACTION_PROTOCOL,
            "latent_dim": self.latent_dim,
            "latent_frames": self.latent_frames,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "utterance_key": self.utterance_key,
        }


class SemanticVaePosterior(nn.Module):
    """The exact 145-key posterior path, excluding unused decoder/projector weights."""

    def __init__(self, encoder_class: type[nn.Module], attention_class: type[nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder_class(d_model=64, strides=[4, 4, 5, 5], d_latent=1024)
        self.pre_block = attention_class(1024, 64, num_heads=8)
        self.fc_mu = nn.Linear(64, 64)
        self.fc_var = nn.Linear(64, 64)

    def stats(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(waveform).transpose(1, 2)
        hidden = self.pre_block(hidden)
        mu = self.fc_mu(hidden)
        log_var = torch.clamp(self.fc_var(hidden), min=-12, max=12)
        return mu, log_var


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=prefixed_path("datasets/LibriSpeech"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=prefixed_path("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <cache-root>/manifests/inventory.jsonl.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=prefixed_path("projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k"),
    )
    parser.add_argument(
        "--semantic-vae-repo",
        type=Path,
        default=prefixed_path("projects/alignDiT_idea6/papers_codes/Semantic-VAE"),
    )
    parser.add_argument(
        "--attempt-id",
        default=os.environ.get("SVAECACHE_ATTEMPT_ID"),
        help="Unique single-launch ID shared by every rank; progress logs are immutable per attempt.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Development-only: extract the first N manifest records.")
    selection.add_argument(
        "--utterance-key",
        action="append",
        default=None,
        help="Development-only: select one exact key; repeat to select multiple keys.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Offline, single-rank mode: quarantine invalid outputs and regenerate them.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read-only verification against progress hashes; do not load the codec or write progress.",
    )
    parser.add_argument("--progress-fsync-interval", type=int, default=100)
    parser.add_argument("--expected-checkpoint-sha256", default=OFFICIAL_EMA_SHA256)
    parser.add_argument("--expected-manifest-sha256", default=OFFICIAL_MANIFEST_SHA256)
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=24 * 60,
        help="Collective timeout; deliberately covers long, imbalanced resume and final-audit phases.",
    )
    parser.add_argument(
        "--acknowledge-stale-write-attempt",
        help="Explicitly replace a stale WRITE_ACTIVE guard only when its recorded attempt ID matches this value.",
    )
    return parser.parse_args()


def initialize_distributed(*, require_cuda: bool, timeout_minutes: int) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed coordinates rank={rank}, world_size={world_size}")
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Semantic-VAE extraction requires CUDA")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible CUDA devices")
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
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def distributed_barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def run_checked(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def selection_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}")
        return {"mode": "first_n", "limit": args.limit}
    if args.utterance_key:
        keys = sorted(args.utterance_key)
        if len(keys) != len(set(keys)):
            raise ValueError("--utterance-key values must be unique")
        for key in keys:
            stable_utterance_seed(0, key)
        return {"mode": "utterance_keys", "keys": keys}
    return {"mode": "full"}


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def validate_semantic_vae_metainfo(metainfo: Mapping[str, Any], semantic_repo: Path) -> Path:
    dac = metainfo.get("DAC")
    if not isinstance(dac, dict):
        raise TypeError("Semantic-VAE metainfo is missing the DAC object")
    expected = {
        "encoder_dim": 64,
        "encoder_rates": [4, 4, 5, 5],
        "sample_rate": SAMPLE_RATE,
        "vae_dim": SEMANTIC_VAE_LATENT_DIM,
        "attn_proj": True,
    }
    mismatches = {key: (dac.get(key), value) for key, value in expected.items() if dac.get(key) != value}
    if mismatches:
        raise ValueError(f"Unexpected Semantic-VAE architecture in metainfo: {mismatches}")
    if int(np.prod(dac["encoder_rates"])) != SEMANTIC_VAE_HOP_LENGTH:
        raise ValueError(f"Semantic-VAE hop is not {SEMANTIC_VAE_HOP_LENGTH}: {dac['encoder_rates']}")
    bigvgan_relative = dac.get("bigvgan_conf")
    if not isinstance(bigvgan_relative, str):
        raise TypeError("Semantic-VAE metainfo has no bigvgan_conf path")
    return safe_join(semantic_repo, bigvgan_relative)


def manifest_metadata(manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = manifest.parent / "inventory_meta.json"
    metadata = read_json_object(metadata_path)
    entry = metadata.get("manifests", {}).get(manifest.name)
    if not isinstance(entry, dict):
        raise TypeError(f"{manifest.name} is not registered in {metadata_path}")
    return metadata, entry


def cuda_numerics() -> dict[str, bool]:
    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }


def runtime_versions() -> dict[str, Any]:
    return {
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
    }


def build_and_verify_spec(args: argparse.Namespace, context: DistributedContext) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve(strict=True)
    cache_root = args.cache_root.resolve() if args.cache_root.exists() else args.cache_root.absolute()
    checkpoint_root = args.checkpoint_root.resolve(strict=True)
    semantic_repo = args.semantic_vae_repo.resolve(strict=True)
    manifest = (args.manifest or cache_root / "manifests" / "inventory.jsonl").resolve(strict=True)
    manifest_meta, manifest_entry = manifest_metadata(manifest)
    manifest_meta_path = manifest.parent / "inventory_meta.json"
    expected_manifest_contract = {
        "base_posterior_seed": BASE_POSTERIOR_SEED,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
    }
    manifest_mismatches = {
        key: (manifest_meta.get(key), expected)
        for key, expected in expected_manifest_contract.items()
        if manifest_meta.get(key) != expected
    }
    if manifest_mismatches:
        raise RuntimeError(f"Manifest metadata contract mismatch: {manifest_mismatches}")
    if manifest_entry.get("sha256") != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Manifest sidecar does not match the pinned manifest SHA256: "
            f"{manifest_entry.get('sha256')} != {args.expected_manifest_sha256}"
        )
    metainfo_path = checkpoint_root / "metainfo.json"
    config_path = checkpoint_root / "config.json"
    checkpoint_path = checkpoint_root / "dac" / "ema_state_dict.pth"
    metainfo = read_json_object(metainfo_path)
    bigvgan_path = validate_semantic_vae_metainfo(metainfo, semantic_repo)

    semantic_commit = run_checked(["git", "rev-parse", "HEAD"], semantic_repo)
    semantic_status = run_checked(["git", "status", "--porcelain", "--untracked-files=normal"], semantic_repo)
    if semantic_commit != OFFICIAL_SEMANTIC_VAE_COMMIT:
        raise RuntimeError(
            f"Semantic-VAE source commit mismatch: expected {OFFICIAL_SEMANTIC_VAE_COMMIT}, got {semantic_commit}"
        )
    if semantic_status:
        raise RuntimeError(f"Semantic-VAE source tree must be clean; git status:\n{semantic_status}")

    expected_resource_hashes = {
        checkpoint_path: args.expected_checkpoint_sha256,
        metainfo_path: OFFICIAL_METAINFO_SHA256,
        config_path: OFFICIAL_CONFIG_SHA256,
        bigvgan_path: OFFICIAL_BIGVGAN_CONFIG_SHA256,
        manifest: args.expected_manifest_sha256,
    }
    resource_hashes: dict[str, str] = {}
    for path, expected_hash in expected_resource_hashes.items():
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"Invalid expected SHA256 for {path}: {expected_hash!r}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}")
        resource_hashes[str(path)] = actual_hash

    manifest_line_count = sum(1 for line in manifest.open("rb") if line.strip())
    if manifest_line_count != int(manifest_entry["count"]):
        raise RuntimeError(
            f"Manifest count mismatch for {manifest}: metadata={manifest_entry['count']}, actual={manifest_line_count}"
        )
    manifest_meta_hash = sha256_file(manifest_meta_path)
    extractor_path = Path(__file__).resolve()
    utility_path = extractor_path.with_name("svae_cache_utils.py")

    properties = torch.cuda.get_device_properties(context.local_rank)
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "checkpoint": {
            "config_path": str(config_path),
            "config_sha256": resource_hashes[str(config_path)],
            "ema_path": str(checkpoint_path),
            "ema_sha256": resource_hashes[str(checkpoint_path)],
            "ema_step": OFFICIAL_EMA_STEP,
            "metainfo_path": str(metainfo_path),
            "metainfo_sha256": resource_hashes[str(metainfo_path)],
        },
        "dataset_root": str(dataset_root),
        "extraction": {
            "code": {
                "extractor_path": str(extractor_path),
                "extractor_sha256": sha256_file(extractor_path),
                "utility_path": str(utility_path),
                "utility_sha256": sha256_file(utility_path),
            },
            "cuda_numerics": cuda_numerics(),
            "device_capability": [properties.major, properties.minor],
            "device_name": properties.name,
            "dtype": "float32",
            "first_forward_warmup": {"num_samples": SEMANTIC_VAE_HOP_LENGTH, "value": 0.0},
            "golden_self_test": {
                "raw_latent_sha256": GOLDEN_RAW_LATENT_SHA256,
                "utterance_key": GOLDEN_UTTERANCE_KEY,
            },
            "latent_dim": SEMANTIC_VAE_LATENT_DIM,
            "latent_layout": "[time,channel]",
            "posterior_noise": "torch.randn with per-utterance CUDA Generator",
            "protocol": EXTRACTION_PROTOCOL,
            "sample_rate": SAMPLE_RATE,
            "vae_hop_length": SEMANTIC_VAE_HOP_LENGTH,
        },
        "manifest": {
            "count": int(manifest_entry["count"]),
            "inventory_metadata_path": str(manifest_meta_path),
            "inventory_metadata_sha256": manifest_meta_hash,
            "path": str(manifest),
            "sha256": resource_hashes[str(manifest)],
        },
        "runtime": runtime_versions(),
        "semantic_vae_source": {
            "bigvgan_config_path": str(bigvgan_path),
            "bigvgan_config_sha256": resource_hashes[str(bigvgan_path)],
            "commit": semantic_commit,
            "repo": str(semantic_repo),
            "working_tree_clean": True,
        },
    }


def validate_stored_spec_resources(args: argparse.Namespace, spec: Mapping[str, Any]) -> None:
    if spec.get("extraction", {}).get("protocol") != EXTRACTION_PROTOCOL:
        raise RuntimeError(f"Stored cache uses a different extraction protocol: {spec}")
    if Path(spec.get("dataset_root", "")).resolve() != args.dataset_root.resolve(strict=True):
        raise RuntimeError("Stored dataset root does not match this launch")
    if spec.get("checkpoint", {}).get("ema_sha256") != args.expected_checkpoint_sha256:
        raise RuntimeError("Stored checkpoint hash does not match --expected-checkpoint-sha256")
    if spec.get("manifest", {}).get("sha256") != args.expected_manifest_sha256:
        raise RuntimeError("Stored manifest hash does not match --expected-manifest-sha256")

    extraction_code = spec.get("extraction", {}).get("code", {})
    files_and_hashes = {
        Path(spec["checkpoint"]["config_path"]): spec["checkpoint"]["config_sha256"],
        Path(spec["checkpoint"]["ema_path"]): spec["checkpoint"]["ema_sha256"],
        Path(spec["checkpoint"]["metainfo_path"]): spec["checkpoint"]["metainfo_sha256"],
        Path(spec["manifest"]["inventory_metadata_path"]): spec["manifest"]["inventory_metadata_sha256"],
        Path(spec["manifest"]["path"]): spec["manifest"]["sha256"],
        Path(spec["semantic_vae_source"]["bigvgan_config_path"]): spec["semantic_vae_source"]["bigvgan_config_sha256"],
        Path(extraction_code["extractor_path"]): extraction_code["extractor_sha256"],
        Path(extraction_code["utility_path"]): extraction_code["utility_sha256"],
    }
    for path, expected_hash in files_and_hashes.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Stored cache resource changed: {path}: expected {expected_hash}, got {actual_hash}")

    semantic_repo = Path(spec["semantic_vae_source"]["repo"])
    semantic_commit = run_checked(["git", "rev-parse", "HEAD"], semantic_repo)
    semantic_status = run_checked(["git", "status", "--porcelain", "--untracked-files=normal"], semantic_repo)
    if semantic_commit != spec["semantic_vae_source"]["commit"] or semantic_status:
        raise RuntimeError(
            f"Semantic-VAE source changed after cache spec publication: commit={semantic_commit}, status={semantic_status!r}"
        )


def validate_rank_runtime(spec: Mapping[str, Any], context: DistributedContext) -> None:
    if context.device.type != "cuda":
        return
    properties = torch.cuda.get_device_properties(context.local_rank)
    actual = {
        "cuda_numerics": cuda_numerics(),
        "device_capability": [properties.major, properties.minor],
        "device_name": properties.name,
        "runtime": runtime_versions(),
    }
    expected = {
        "cuda_numerics": spec["extraction"]["cuda_numerics"],
        "device_capability": spec["extraction"]["device_capability"],
        "device_name": spec["extraction"]["device_name"],
        "runtime": spec["runtime"],
    }
    if actual != expected:
        raise RuntimeError(f"Rank {context.rank} runtime differs from the immutable cache spec: {actual} != {expected}")


def publish_or_validate_spec(args: argparse.Namespace, context: DistributedContext) -> dict[str, Any]:
    cache_root = args.cache_root.absolute()
    spec_path = cache_root / "state" / "latents" / "spec.json"
    if context.is_main:
        if args.validate_only:
            if not spec_path.is_file() or spec_path.is_symlink():
                raise FileNotFoundError(f"Read-only validation requires an existing regular spec: {spec_path}")
            spec = read_json_object(spec_path)
            validate_stored_spec_resources(args, spec)
            print(f"[rank 0] read-only spec verified: {spec_path}", flush=True)
        else:
            spec = build_and_verify_spec(args, context)
            result = atomic_write_json(spec_path, spec)
            print(
                f"[rank 0] latent spec {'published' if result.created else 'verified'}: "
                f"{result.path} sha256={result.sha256}",
                flush=True,
            )
    distributed_barrier(context)
    stored = read_json_object(spec_path)
    if Path(stored.get("dataset_root", "")).resolve() != args.dataset_root.resolve(strict=True):
        raise RuntimeError(f"Stored dataset root does not match this launch: {spec_path}")
    validate_rank_runtime(stored, context)
    return stored


def acquire_write_guard(args: argparse.Namespace, context: DistributedContext, cache_root: Path) -> Path | None:
    if args.validate_only:
        return None
    guard_path = cache_root / "state" / "latents" / "WRITE_ACTIVE.json"
    if context.is_main:
        if guard_path.exists() or guard_path.is_symlink():
            if not guard_path.is_file() or guard_path.is_symlink():
                raise RuntimeError(f"Write guard is not a regular file: {guard_path}")
            guard = read_json_object(guard_path)
            guarded_attempt = guard.get("attempt_id")
            if args.acknowledge_stale_write_attempt != guarded_attempt:
                raise RuntimeError(
                    f"Cache has an active/stale write guard from attempt {guarded_attempt!r}: {guard_path}. "
                    "Confirm that no writer is alive, then pass "
                    f"--acknowledge-stale-write-attempt {guarded_attempt} with a new --attempt-id."
                )
            if args.attempt_id == guarded_attempt:
                raise ValueError("A stale write guard must be resumed with a new, unique --attempt-id")
            quarantine_root = cache_root / "quarantine" / "write_guards" / args.attempt_id
            quarantined = quarantine_file(guard_path, quarantine_root, "WRITE_ACTIVE.json")
            print(f"[rank 0] quarantined acknowledged stale write guard at {quarantined}", flush=True)
        guard = {
            "attempt_id": args.attempt_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "protocol": EXTRACTION_PROTOCOL,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "world_size": context.world_size,
        }
        result = atomic_write_json(guard_path, guard)
        if not result.created:
            raise RuntimeError(f"A write guard unexpectedly already exists: {guard_path}")
        print(f"[rank 0] acquired write guard: {guard_path}", flush=True)
    distributed_barrier(context)
    guard = read_json_object(guard_path)
    if guard.get("attempt_id") != args.attempt_id:
        raise RuntimeError(f"Write guard belongs to another attempt: {guard}")
    return guard_path


def release_write_guard(guard_path: Path | None, context: DistributedContext) -> None:
    if guard_path is None:
        return
    distributed_barrier(context)
    if context.is_main:
        durable_unlink(guard_path)
        print(f"[rank 0] released write guard: {guard_path}", flush=True)
    distributed_barrier(context)


def prepare_write_state(
    args: argparse.Namespace,
    context: DistributedContext,
    cache_root: Path,
    selection: Mapping[str, Any],
) -> None:
    complete_path = cache_root / "state" / "latents" / "complete.json"
    if context.is_main and (complete_path.exists() or complete_path.is_symlink()):
        if selection["mode"] != "full":
            raise RuntimeError("A completed full cache cannot be reopened by a partial writing selection")
        if not complete_path.is_file() or complete_path.is_symlink():
            raise RuntimeError(f"Completion marker is not a regular file: {complete_path}")
        quarantine_root = cache_root / "quarantine" / "completion_markers" / args.attempt_id
        quarantined = quarantine_file(complete_path, quarantine_root, "complete.json")
        print(f"[rank 0] invalidated completion marker during write: {quarantined}", flush=True)
    distributed_barrier(context)


def scan_latent_tree(cache_root: Path) -> dict[str, Path]:
    latent_root = cache_root / "latents"
    if not latent_root.exists():
        return {}
    if not latent_root.is_dir() or latent_root.is_symlink():
        raise RuntimeError(f"Latent output root is not a regular directory: {latent_root}")
    files: dict[str, Path] = {}
    for directory, directory_names, filenames in os.walk(latent_root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            child = directory_path / name
            if child.is_symlink():
                raise RuntimeError(f"Symlink directory in latent cache: {child}")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Non-regular file in latent cache: {path}")
            relative = path.relative_to(cache_root).as_posix()
            if relative in files:
                raise RuntimeError(f"Duplicate path while scanning latent cache: {relative}")
            files[relative] = path
    return files


def quarantine_orphans_for_repair(
    args: argparse.Namespace,
    cache_root: Path,
    expected_paths: set[str],
) -> int:
    quarantined_count = 0
    for relative, path in scan_latent_tree(cache_root).items():
        if relative in expected_paths:
            continue
        quarantine_root = cache_root / "quarantine" / "orphans" / args.attempt_id
        destination = quarantine_file(path, quarantine_root, relative)
        print(f"[repair] quarantined orphan {path} -> {destination}", flush=True)
        quarantined_count += 1
    return quarantined_count


def validate_manifest_record(
    record: dict[str, Any],
    dataset_root: Path,
    base_seed: int,
    *,
    check_source: bool,
) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Manifest record is missing fields {sorted(missing)}: {record}")
    key = record["utterance_key"]
    if not isinstance(key, str):
        raise TypeError(f"utterance_key must be a string: {record}")
    if record["sample_rate"] != SAMPLE_RATE or record["num_channels"] != 1:
        raise ValueError(f"Manifest audio contract mismatch for {key}")
    num_samples = record["original_num_samples"]
    frames = expected_latent_frames(num_samples)
    if record["latent_frames"] != frames or record["padded_num_samples"] != frames * SEMANTIC_VAE_HOP_LENGTH:
        raise ValueError(f"Manifest frame contract mismatch for {key}")
    if record["latent_dim"] != SEMANTIC_VAE_LATENT_DIM:
        raise ValueError(f"Manifest latent dimension mismatch for {key}")
    if record["posterior_seed"] != stable_utterance_seed(base_seed, key):
        raise ValueError(f"Manifest posterior seed mismatch for {key}")
    audio_relative = Path(record["audio_relative_path"])
    latent_path = Path(record["latent_relative_path"])
    if (
        audio_relative.is_absolute()
        or audio_relative.as_posix() != record["audio_relative_path"]
        or any(part in {"", ".", ".."} for part in audio_relative.parts)
        or "\\" in record["audio_relative_path"]
    ):
        raise ValueError(f"Invalid source audio path for {key}: {record['audio_relative_path']}")
    if (
        latent_path.is_absolute()
        or latent_path.as_posix() != record["latent_relative_path"]
        or any(part in {"", ".", ".."} for part in latent_path.parts)
        or "\\" in record["latent_relative_path"]
        or latent_path.parts[0] != "latents"
        or latent_path.suffix != ".npy"
    ):
        raise ValueError(f"Invalid latent cache path for {key}: {latent_path}")
    if check_source:
        audio_path = safe_join(dataset_root, record["audio_relative_path"])
        if not audio_path.is_file() or audio_path.is_symlink():
            raise FileNotFoundError(f"Manifest source audio is not a regular file: {audio_path}")


def iter_selected_records(manifest: Path, selection: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    mode = selection["mode"]
    requested_keys = set(selection.get("keys", []))
    found_keys: set[str] = set()
    for index, record in enumerate(read_jsonl(manifest)):
        if mode == "first_n" and index >= selection["limit"]:
            break
        if mode == "utterance_keys" and record.get("utterance_key") not in requested_keys:
            continue
        if mode == "utterance_keys":
            found_keys.add(record["utterance_key"])
        yield record
    if mode == "utterance_keys" and found_keys != requested_keys:
        raise KeyError(f"Manifest does not contain requested keys: {sorted(requested_keys - found_keys)}")


def load_local_records(
    manifest: Path,
    selection: Mapping[str, Any],
    dataset_root: Path,
    base_seed: int,
    context: DistributedContext,
) -> tuple[list[dict[str, Any]], int]:
    local: list[dict[str, Any]] = []
    selected_count = 0
    seen_keys: set[str] = set()
    for selected_index, record in enumerate(iter_selected_records(manifest, selection)):
        owned = selected_index % context.world_size == context.rank
        if context.is_main or owned:
            validate_manifest_record(record, dataset_root, base_seed, check_source=owned)
        if context.is_main:
            key = record["utterance_key"]
            if key in seen_keys:
                raise ValueError(f"Duplicate utterance key in selected manifest: {key}")
            seen_keys.add(key)
        if owned:
            local.append(record)
        selected_count += 1
    expected_local_count = (selected_count + context.world_size - 1 - context.rank) // context.world_size
    if len(local) != max(0, expected_local_count):
        raise AssertionError(
            f"Exact shard count failed for rank {context.rank}: expected {expected_local_count}, got {len(local)}"
        )
    return local, selected_count


def parse_progress_entry(record: Mapping[str, Any], source: Path) -> ProgressEntry:
    required = {
        "feature",
        "latent_dim",
        "latent_frames",
        "relative_path",
        "sha256",
        "size_bytes",
        "utterance_key",
    }
    if set(record) != required:
        raise ValueError(f"Unexpected progress fields in {source}: {sorted(record)}")
    if record["feature"] != EXTRACTION_PROTOCOL:
        raise ValueError(f"Wrong progress feature in {source}: {record['feature']!r}")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.fullmatch(record["sha256"]):
        raise ValueError(f"Invalid SHA256 in {source}: {record['sha256']!r}")
    entry = ProgressEntry(
        utterance_key=str(record["utterance_key"]),
        relative_path=str(record["relative_path"]),
        sha256=record["sha256"],
        size_bytes=int(record["size_bytes"]),
        latent_frames=int(record["latent_frames"]),
        latent_dim=int(record["latent_dim"]),
    )
    stable_utterance_seed(0, entry.utterance_key)
    if entry.size_bytes <= 0 or entry.latent_frames <= 0 or entry.latent_dim != SEMANTIC_VAE_LATENT_DIM:
        raise ValueError(f"Invalid dimensions or size in {source}: {record}")
    path = Path(entry.relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != entry.relative_path
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in entry.relative_path
        or path.parts[0] != "latents"
        or path.suffix != ".npy"
    ):
        raise ValueError(f"Invalid progress cache path in {source}: {entry.relative_path}")
    return entry


def load_progress_index(progress_root: Path) -> dict[str, ProgressEntry]:
    index: dict[str, ProgressEntry] = {}
    if not progress_root.exists():
        return index
    if progress_root.is_symlink() or not progress_root.is_dir():
        raise RuntimeError(f"Progress root must be a regular directory: {progress_root}")
    for progress_path in sorted(progress_root.glob("*.jsonl")):
        if progress_path.is_symlink() or not progress_path.is_file():
            raise RuntimeError(f"Progress log must be a regular file: {progress_path}")
        for record in read_append_only_jsonl(progress_path):
            entry = parse_progress_entry(record, progress_path)
            previous = index.get(entry.utterance_key)
            if previous is not None and previous != entry:
                raise RuntimeError(
                    f"Conflicting progress for {entry.utterance_key}: {previous} versus {entry} in {progress_path}"
                )
            index[entry.utterance_key] = entry
    return index


def load_consolidated_index(cache_root: Path) -> dict[str, ProgressEntry]:
    index_path = cache_root / "state" / "latents" / "index.jsonl"
    index: dict[str, ProgressEntry] = {}
    if index_path.exists() or index_path.is_symlink():
        if not index_path.is_file() or index_path.is_symlink():
            raise RuntimeError(f"Consolidated index must be a regular file: {index_path}")
        for record in read_jsonl(index_path):
            entry = parse_progress_entry(record, index_path)
            if entry.utterance_key in index:
                raise RuntimeError(f"Duplicate key in consolidated index: {entry.utterance_key}")
            index[entry.utterance_key] = entry
    return index


def load_known_index(cache_root: Path) -> dict[str, ProgressEntry]:
    index = load_consolidated_index(cache_root)

    progress = load_progress_index(cache_root / "state" / "latents" / "progress")
    for key, entry in progress.items():
        previous = index.get(key)
        if previous is not None and previous != entry:
            raise RuntimeError(f"Progress conflicts with consolidated index for {key}: {entry} != {previous}")
        index[key] = entry
    return index


def load_known_index_for_repair(
    args: argparse.Namespace,
    cache_root: Path,
    spec: Mapping[str, Any],
) -> dict[str, ProgressEntry]:
    index_path = cache_root / "state" / "latents" / "index.jsonl"
    quarantined_completion = cache_root / "quarantine" / "completion_markers" / args.attempt_id / "complete.json"
    index: dict[str, ProgressEntry] = {}
    index_error: Exception | None = None
    if index_path.exists() or index_path.is_symlink():
        try:
            if not quarantined_completion.is_file() or quarantined_completion.is_symlink():
                raise RuntimeError("No trustworthy pre-repair completion marker binds the consolidated index")
            completion = read_json_object(quarantined_completion)
            consolidated = completion.get("consolidated_index")
            expected_completion_fields = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "count": int(spec["manifest"]["count"]),
                "feature": EXTRACTION_PROTOCOL,
                "manifest_sha256": spec["manifest"]["sha256"],
                "selection": {"mode": "full"},
                "spec_sha256": sha256_file(cache_root / "state" / "latents" / "spec.json"),
            }
            mismatches = {
                key: (completion.get(key), expected)
                for key, expected in expected_completion_fields.items()
                if completion.get(key) != expected
            }
            if mismatches or not isinstance(consolidated, dict):
                raise RuntimeError(f"Completion marker cannot authenticate the consolidated index: {mismatches}")
            actual_size = index_path.stat().st_size
            actual_hash = sha256_file(index_path)
            if (
                consolidated.get("path") != "state/latents/index.jsonl"
                or consolidated.get("size_bytes") != actual_size
                or consolidated.get("sha256") != actual_hash
                or consolidated.get("count") != completion.get("count")
            ):
                raise RuntimeError(
                    f"Consolidated index is not bound by completion: completion={consolidated}, "
                    f"actual_size={actual_size}, actual_sha256={actual_hash}"
                )
            index = load_consolidated_index(cache_root)
            if len(index) != consolidated["count"]:
                raise RuntimeError(f"Consolidated index count mismatch: {len(index)} != {consolidated['count']}")
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            index_error = error
    if index_error is not None:
        if not index_path.is_file() or index_path.is_symlink():
            raise RuntimeError(
                f"Cannot safely quarantine invalid consolidated index {index_path}: {index_error}"
            ) from index_error
        quarantine_root = cache_root / "quarantine" / "indexes" / args.attempt_id
        destination = quarantine_file(index_path, quarantine_root, "index.jsonl")
        print(
            f"[repair] quarantined untrusted consolidated index {index_path} -> {destination}: {index_error}",
            flush=True,
        )
        index = {}

    progress_root = cache_root / "state" / "latents" / "progress"
    if not progress_root.exists():
        return index
    if not progress_root.is_dir() or progress_root.is_symlink():
        raise RuntimeError(f"Progress root must be a regular directory: {progress_root}")
    for progress_path in sorted(progress_root.glob("*.jsonl")):
        try:
            candidate: dict[str, ProgressEntry] = {}
            for record in read_append_only_jsonl(progress_path):
                entry = parse_progress_entry(record, progress_path)
                previous_in_file = candidate.get(entry.utterance_key)
                if previous_in_file is not None and previous_in_file != entry:
                    raise RuntimeError(f"Conflicting entries inside {progress_path} for {entry.utterance_key}")
                previous = index.get(entry.utterance_key)
                if previous is not None and previous != entry:
                    raise RuntimeError(f"Progress conflicts with authoritative index for {entry.utterance_key}")
                candidate[entry.utterance_key] = entry
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            if not progress_path.is_file() or progress_path.is_symlink():
                raise RuntimeError(f"Cannot safely quarantine invalid progress log {progress_path}: {error}") from error
            quarantine_root = cache_root / "quarantine" / "progress" / args.attempt_id
            destination = quarantine_file(progress_path, quarantine_root, progress_path.name)
            print(f"[repair] quarantined invalid progress {progress_path} -> {destination}: {error}", flush=True)
            continue
        index.update(candidate)
    return index


def state_contract_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(f"{key}\t{tensor.dtype}\t{tuple(tensor.shape)}\n".encode())
    return digest.hexdigest()


def import_semantic_vae_classes(semantic_repo: Path) -> tuple[type[nn.Module], type[nn.Module]]:
    sys.path.insert(0, str(semantic_repo))
    try:
        import dac
        from dac.model.attn_proj import AttnProjection
        from dac.model.dac import Encoder
    finally:
        sys.path.pop(0)
    imported_root = Path(dac.__file__).resolve().parent.parent
    if imported_root != semantic_repo:
        raise RuntimeError(f"Imported Semantic-VAE from {imported_root}, expected {semantic_repo}")
    return Encoder, AttnProjection


def load_posterior_model(
    spec: Mapping[str, Any],
    device: torch.device,
    dataset_root: Path,
) -> SemanticVaePosterior:
    semantic_repo = Path(spec["semantic_vae_source"]["repo"])
    checkpoint_path = Path(spec["checkpoint"]["ema_path"])
    encoder_class, attention_class = import_semantic_vae_classes(semantic_repo)
    model = SemanticVaePosterior(encoder_class, attention_class)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected an EMA state mapping in {checkpoint_path}")
    metadata_keys = {key for key in checkpoint if not key.startswith("ema_model.")}
    if metadata_keys != {"initted", "step"}:
        raise RuntimeError(f"Unexpected EMA metadata keys: {sorted(metadata_keys)}")
    initted = checkpoint["initted"]
    step = checkpoint["step"]
    if (
        not isinstance(initted, torch.Tensor)
        or initted.shape != torch.Size([])
        or initted.dtype != torch.bool
        or not isinstance(step, torch.Tensor)
        or step.shape != torch.Size([])
        or step.dtype != torch.int64
    ):
        raise RuntimeError(f"Unexpected EMA metadata tensors: initted={initted!r}, step={step!r}")
    if not bool(initted.item()) or int(step.item()) != OFFICIAL_EMA_STEP:
        raise RuntimeError(f"Unexpected EMA state: initted={initted}, step={step}")

    stripped = {
        key.removeprefix("ema_model."): value for key, value in checkpoint.items() if key.startswith("ema_model.")
    }
    grouped: dict[str, dict[str, torch.Tensor]] = {name: {} for name in CHECKPOINT_GROUP_COUNTS}
    for key, value in stripped.items():
        if key.startswith(("encoder.", "pre_block.", "fc_mu.", "fc_var.")):
            grouped["posterior"][key] = value
        elif key.startswith("decoder."):
            grouped["decoder"][key] = value
        elif key.startswith("decoder_proj."):
            grouped["decoder_proj"][key] = value
        elif key.startswith("projectors."):
            grouped["projectors"][key] = value
        else:
            raise RuntimeError(f"Unexpected EMA model key: {key}")
    counts = {name: len(values) for name, values in grouped.items()}
    if counts != CHECKPOINT_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected checkpoint group counts: expected {CHECKPOINT_GROUP_COUNTS}, got {counts}")

    model_state = model.state_dict()
    posterior_state = grouped["posterior"]
    if set(model_state) != set(posterior_state):
        raise RuntimeError(
            f"Posterior key mismatch: missing={sorted(set(model_state) - set(posterior_state))}, "
            f"unexpected={sorted(set(posterior_state) - set(model_state))}"
        )
    for key, expected_tensor in model_state.items():
        checkpoint_tensor = posterior_state[key]
        if checkpoint_tensor.shape != expected_tensor.shape or checkpoint_tensor.dtype != expected_tensor.dtype:
            raise RuntimeError(
                f"Posterior tensor contract mismatch for {key}: checkpoint "
                f"{checkpoint_tensor.dtype}{tuple(checkpoint_tensor.shape)}, model "
                f"{expected_tensor.dtype}{tuple(expected_tensor.shape)}"
            )
    contract_hash = state_contract_hash(posterior_state)
    if contract_hash != LIGHT_STATE_CONTRACT_SHA256:
        raise RuntimeError(
            f"Posterior state contract hash mismatch: expected {LIGHT_STATE_CONTRACT_SHA256}, got {contract_hash}"
        )
    model.load_state_dict(posterior_state, strict=True)
    del checkpoint, stripped, grouped, posterior_state
    gc.collect()
    model.eval().requires_grad_(False).to(device)

    # The scripted Snake kernel changes its first CUDA result during lazy initialization.
    # A fixed discarded forward makes every real utterance independent of rank and resume position.
    with torch.inference_mode():
        warmup = torch.zeros(1, 1, SEMANTIC_VAE_HOP_LENGTH, dtype=torch.float32, device=device)
        model.stats(warmup)
    torch.cuda.synchronize(device)
    run_golden_self_test(model, dataset_root, device)
    return model


def load_and_validate_waveform(record: Mapping[str, Any], dataset_root: Path, device: torch.device) -> torch.Tensor:
    audio_path = safe_join(dataset_root, record["audio_relative_path"])
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != record["sample_rate"] or sample_rate != SAMPLE_RATE:
        raise ValueError(f"Decoded sample rate mismatch for {audio_path}: {sample_rate}")
    if waveform.ndim != 2 or waveform.shape[0] != record["num_channels"] or waveform.shape[0] != 1:
        raise ValueError(f"Decoded channel mismatch for {audio_path}: shape={tuple(waveform.shape)}")
    if waveform.shape[1] != record["original_num_samples"]:
        raise ValueError(
            f"Decoded sample count mismatch for {audio_path}: expected {record['original_num_samples']}, "
            f"got {waveform.shape[1]}"
        )
    waveform = waveform.mean(dim=0, keepdim=True).unsqueeze(0)
    right_pad = record["padded_num_samples"] - waveform.shape[-1]
    if not 0 <= right_pad < SEMANTIC_VAE_HOP_LENGTH:
        raise ValueError(f"Invalid right padding for {record['utterance_key']}: {right_pad}")
    return F.pad(waveform, (0, right_pad)).to(device=device, dtype=torch.float32, non_blocking=True)


@torch.inference_mode()
def extract_latent(
    model: SemanticVaePosterior,
    waveform: torch.Tensor,
    seed: int,
    expected_frames: int,
    device: torch.device,
) -> np.ndarray:
    mu, log_var = model.stats(waveform)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
    latent = mu + torch.exp(0.5 * log_var) * noise
    if latent.shape != (1, expected_frames, SEMANTIC_VAE_LATENT_DIM):
        raise RuntimeError(
            f"Semantic-VAE produced {tuple(latent.shape)}, expected {(1, expected_frames, SEMANTIC_VAE_LATENT_DIM)}"
        )
    array = latent.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise FloatingPointError("Semantic-VAE produced a non-finite latent")
    return array


def run_golden_self_test(model: SemanticVaePosterior, dataset_root: Path, device: torch.device) -> None:
    record = {
        "audio_relative_path": ("train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac"),
        "latent_frames": 153,
        "num_channels": 1,
        "original_num_samples": 60_960,
        "padded_num_samples": 61_200,
        "sample_rate": SAMPLE_RATE,
        "utterance_key": GOLDEN_UTTERANCE_KEY,
    }
    waveform = load_and_validate_waveform(record, dataset_root, device)
    latent = extract_latent(model, waveform, GOLDEN_POSTERIOR_SEED, 153, device)
    actual_hash = hashlib.sha256(latent.tobytes(order="C")).hexdigest()
    if actual_hash != GOLDEN_RAW_LATENT_SHA256:
        raise RuntimeError(
            f"Semantic-VAE golden self-test failed: expected {GOLDEN_RAW_LATENT_SHA256}, got {actual_hash}"
        )


def entry_from_result(record: Mapping[str, Any], sha256: str, size_bytes: int) -> ProgressEntry:
    return ProgressEntry(
        utterance_key=record["utterance_key"],
        relative_path=record["latent_relative_path"],
        sha256=sha256,
        size_bytes=size_bytes,
        latent_frames=record["latent_frames"],
        latent_dim=SEMANTIC_VAE_LATENT_DIM,
    )


def validate_existing_output(
    path: Path,
    record: Mapping[str, Any],
    progress: ProgressEntry | None,
) -> ProgressEntry:
    validation = validate_npy(
        path,
        expected_shape=(record["latent_frames"], SEMANTIC_VAE_LATENT_DIM),
        expected_dtype=np.float32,
    )
    entry = entry_from_result(record, validation.sha256, validation.size_bytes)
    if progress is not None and entry != progress:
        raise RuntimeError(
            f"Cache file no longer matches progress for {record['utterance_key']}: {entry} != {progress}"
        )
    return entry


def process_record(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    dataset_root: Path,
    cache_root: Path,
    progress: ProgressEntry | None,
    model_holder: list[SemanticVaePosterior],
    spec: Mapping[str, Any],
) -> tuple[ProgressEntry, str]:
    output_path = safe_join(cache_root, record["latent_relative_path"])
    if args.validate_only:
        if progress is None:
            raise FileNotFoundError(f"No progress hash exists for {record['utterance_key']}")
        return validate_existing_output(output_path, record, progress), "validated"

    if output_path.is_file() and not output_path.is_symlink():
        try:
            existing = validate_existing_output(output_path, record, progress)
        except (OSError, ValueError, RuntimeError) as error:
            if not args.repair:
                raise RuntimeError(
                    f"Invalid existing latent for {record['utterance_key']}; rerun offline with --repair: {error}"
                ) from error
            quarantine_root = cache_root / "quarantine" / "latents" / args.attempt_id
            quarantined = quarantine_file(output_path, quarantine_root, record["latent_relative_path"])
            print(f"[rank {context.rank}] quarantined {output_path} -> {quarantined}: {error}", flush=True)
        else:
            if progress is not None:
                return existing, "resumed"
            # A file without a committed progress hash may be from a crash window.
            # Recompute below and require atomic no-clobber to prove identical content.
            if args.repair:
                quarantine_root = cache_root / "quarantine" / "latents" / args.attempt_id
                quarantined = quarantine_file(output_path, quarantine_root, record["latent_relative_path"])
                print(
                    f"[rank {context.rank}] quarantined unindexed output {output_path} -> {quarantined}",
                    flush=True,
                )
    elif output_path.exists() or output_path.is_symlink():
        raise RuntimeError(f"Latent output is not a regular file: {output_path}")

    if not model_holder:
        model_holder.append(load_posterior_model(spec, context.device, dataset_root))
    waveform = load_and_validate_waveform(record, dataset_root, context.device)
    array = extract_latent(
        model_holder[0],
        waveform,
        record["posterior_seed"],
        record["latent_frames"],
        context.device,
    )
    result = atomic_write_npy(output_path, array)
    entry = entry_from_result(record, result.sha256, result.size_bytes)
    if progress is not None and entry != progress:
        raise RuntimeError(
            f"Regenerated latent differs from prior progress for {record['utterance_key']}: {entry} != {progress}"
        )
    return entry, "created" if result.created else "recovered"


def validate_final_index(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    selection: Mapping[str, Any],
    selected_count: int,
    progress_index: Mapping[str, ProgressEntry],
) -> dict[str, Any]:
    cache_root = args.cache_root.absolute().resolve()
    dataset_root = args.dataset_root.resolve(strict=True)
    manifest = Path(spec["manifest"]["path"])
    base_seed = int(read_json_object(manifest.parent / "inventory_meta.json")["base_posterior_seed"])
    expected_paths: set[str] = set()
    expected_keys: set[str] = set()
    index_digest = hashlib.sha256()
    index_size_bytes = 0
    index_records: list[dict[str, Any]] = []
    count = 0
    total_frames = 0
    total_size_bytes = 0
    for record in iter_selected_records(manifest, selection):
        validate_manifest_record(record, dataset_root, base_seed, check_source=False)
        key = record["utterance_key"]
        entry = progress_index.get(key)
        if entry is None:
            raise RuntimeError(f"Missing progress entry after extraction: {key}")
        expected_entry_path = record["latent_relative_path"]
        if (
            entry.relative_path != expected_entry_path
            or entry.latent_frames != record["latent_frames"]
            or entry.latent_dim != SEMANTIC_VAE_LATENT_DIM
        ):
            raise RuntimeError(f"Progress contract mismatch after extraction: {key}: {entry}")
        index_record = entry.as_record()
        encoded_index_record = f"{canonical_json(index_record)}\n".encode()
        index_digest.update(encoded_index_record)
        index_size_bytes += len(encoded_index_record)
        index_records.append(index_record)
        expected_paths.add(expected_entry_path)
        expected_keys.add(key)
        count += 1
        total_frames += entry.latent_frames
        total_size_bytes += entry.size_bytes
    if count != selected_count:
        raise RuntimeError(f"Final selected count mismatch: expected {selected_count}, got {count}")

    actual_files = scan_latent_tree(cache_root)
    actual_paths = set(actual_files)
    missing = expected_paths - actual_paths
    if missing:
        raise RuntimeError(f"Final cache is missing selected paths: {sorted(missing)[:20]}")

    is_full = selection["mode"] == "full"
    if is_full:
        unexpected_progress = set(progress_index) - expected_keys
        unexpected_paths = actual_paths - expected_paths
        if unexpected_progress or unexpected_paths:
            raise RuntimeError(
                f"Full cache has unexpected progress/paths: progress={sorted(unexpected_progress)[:20]}, "
                f"paths={sorted(unexpected_paths)[:20]}"
            )
        for relative, path in actual_files.items():
            if Path(relative).suffix != ".npy":
                raise RuntimeError(f"Unexpected non-NPY file in full latent cache: {path}")

    summary = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "count": count,
        "feature": EXTRACTION_PROTOCOL,
        "manifest_sha256": spec["manifest"]["sha256"],
        "ordered_index_sha256": index_digest.hexdigest(),
        "selection": selection,
        "spec_sha256": sha256_file(cache_root / "state" / "latents" / "spec.json"),
        "total_latent_frames": total_frames,
        "total_npy_size_bytes": total_size_bytes,
    }
    if not is_full:
        return summary

    index_path = cache_root / "state" / "latents" / "index.jsonl"
    if args.validate_only:
        if not index_path.is_file() or index_path.is_symlink():
            raise FileNotFoundError(f"Full read-only validation requires the consolidated index: {index_path}")
        actual_index_size = index_path.stat().st_size
        actual_index_hash = sha256_file(index_path)
        if actual_index_size != index_size_bytes or actual_index_hash != index_digest.hexdigest():
            raise RuntimeError(
                f"Consolidated index mismatch: expected {index_size_bytes}/{index_digest.hexdigest()}, "
                f"got {actual_index_size}/{actual_index_hash}"
            )
    else:
        index_result = atomic_write_jsonl(index_path, index_records)
        if index_result.size_bytes != index_size_bytes or index_result.sha256 != index_digest.hexdigest():
            raise AssertionError("Published consolidated index does not match the in-memory deterministic index")
    summary["consolidated_index"] = {
        "count": count,
        "path": "state/latents/index.jsonl",
        "sha256": index_digest.hexdigest(),
        "size_bytes": index_size_bytes,
    }
    return summary


def main() -> None:
    args = get_args()
    selection = selection_spec(args)
    if args.attempt_id is None and not args.validate_only:
        raise ValueError("--attempt-id (or SVAECACHE_ATTEMPT_ID) is required for every writing launch")
    if args.attempt_id is not None:
        args.attempt_id = validate_attempt_id(args.attempt_id)
    if args.acknowledge_stale_write_attempt is not None:
        args.acknowledge_stale_write_attempt = validate_attempt_id(args.acknowledge_stale_write_attempt)
    if not SHA256_PATTERN.fullmatch(args.expected_checkpoint_sha256):
        raise ValueError(f"Invalid --expected-checkpoint-sha256: {args.expected_checkpoint_sha256!r}")
    if not SHA256_PATTERN.fullmatch(args.expected_manifest_sha256):
        raise ValueError(f"Invalid --expected-manifest-sha256: {args.expected_manifest_sha256!r}")
    if args.distributed_timeout_minutes <= 0:
        raise ValueError("--distributed-timeout-minutes must be positive")
    if args.validate_only and args.repair:
        raise ValueError("--validate-only and --repair are mutually exclusive")
    if args.validate_only and args.acknowledge_stale_write_attempt is not None:
        raise ValueError("Read-only validation cannot acknowledge or replace a write guard")
    if args.repair and selection["mode"] != "full":
        raise ValueError("Offline repair requires the full manifest selection")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("highest")
    context = initialize_distributed(
        require_cuda=not args.validate_only,
        timeout_minutes=args.distributed_timeout_minutes,
    )
    guard_path: Path | None = None
    try:
        if args.repair and context.world_size != 1:
            raise ValueError("--repair is deliberately restricted to offline single-rank execution")
        spec = publish_or_validate_spec(args, context)
        cache_root = args.cache_root.absolute().resolve()
        complete_path = cache_root / "state" / "latents" / "complete.json"
        active_guard_path = cache_root / "state" / "latents" / "WRITE_ACTIVE.json"
        if args.validate_only and (active_guard_path.exists() or active_guard_path.is_symlink()):
            raise RuntimeError(f"Read-only validation refuses to race an active/stale writer: {active_guard_path}")
        if (
            not args.validate_only
            and selection["mode"] != "full"
            and (complete_path.exists() or complete_path.is_symlink())
        ):
            raise RuntimeError("A completed full cache cannot be reopened by a partial writing selection")
        guard_path = acquire_write_guard(args, context, cache_root)
        if not args.validate_only:
            prepare_write_state(args, context, cache_root, selection)
        dataset_root = Path(spec["dataset_root"])
        manifest = Path(spec["manifest"]["path"])
        manifest_meta = read_json_object(manifest.parent / "inventory_meta.json")
        base_seed = int(manifest_meta["base_posterior_seed"])
        local_records, selected_count = load_local_records(
            manifest,
            selection,
            dataset_root,
            base_seed,
            context,
        )
        if selection["mode"] == "full" and selected_count != int(spec["manifest"]["count"]):
            raise RuntimeError(
                f"Full selection count differs from immutable manifest spec: {selected_count} != "
                f"{spec['manifest']['count']}"
            )
        progress_root = cache_root / "state" / "latents" / "progress"
        progress_index = (
            load_known_index_for_repair(args, cache_root, spec) if args.repair else load_known_index(cache_root)
        )
        model_holder: list[SemanticVaePosterior] = []
        counters = {"created": 0, "recovered": 0, "resumed": 0, "validated": 0, "orphans_quarantined": 0}

        if args.repair:
            expected_paths = {record["latent_relative_path"] for record in local_records}
            counters["orphans_quarantined"] = quarantine_orphans_for_repair(args, cache_root, expected_paths)

        if args.validate_only:
            writer_context: Any = _NullProgressWriter()
        else:
            progress_path = progress_root / (
                f"{args.attempt_id}.rank-{context.rank:05d}-of-{context.world_size:05d}.jsonl"
            )
            writer_context = JsonlProgressWriter(progress_path, args.progress_fsync_interval)

        with writer_context as writer:
            iterator = tqdm(
                local_records,
                desc=f"Semantic-VAE rank {context.rank}",
                disable=not context.is_main,
                dynamic_ncols=True,
            )
            for record in iterator:
                entry, status = process_record(
                    record,
                    args=args,
                    context=context,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    progress=progress_index.get(record["utterance_key"]),
                    model_holder=model_holder,
                    spec=spec,
                )
                counters[status] += 1
                if not args.validate_only:
                    writer.append(entry.as_record())
                progress_index[entry.utterance_key] = entry

        print(
            f"[rank {context.rank}] complete local={len(local_records)} selected_global={selected_count} "
            f"counters={counters}",
            flush=True,
        )
        distributed_barrier(context)
        if context.is_main:
            merged_progress = load_known_index(cache_root)
            completion = validate_final_index(args, spec, selection, selected_count, merged_progress)
            if args.validate_only:
                if selection["mode"] == "full":
                    if not complete_path.is_file() or complete_path.is_symlink():
                        raise RuntimeError(f"Completion marker must be a regular file: {complete_path}")
                    stored_completion = read_json_object(complete_path)
                    if stored_completion != completion:
                        raise RuntimeError(
                            f"Completion marker does not match read-only validation: {stored_completion} != {completion}"
                        )
                print(f"[rank 0] read-only validation complete: {completion}", flush=True)
            elif selection["mode"] == "full":
                result = atomic_write_json(complete_path, completion)
                print(
                    f"[rank 0] completion {'published' if result.created else 'verified'}: "
                    f"{result.path} sha256={result.sha256}",
                    flush=True,
                )
            else:
                print(
                    f"[rank 0] partial selection validated without publishing a completion marker: {completion}",
                    flush=True,
                )
        distributed_barrier(context)
        release_write_guard(guard_path, context)
        guard_path = None
    finally:
        destroy_distributed(context)


class _NullProgressWriter:
    def __enter__(self) -> Self:
        return self

    def append(self, record: Mapping[str, Any]) -> None:
        raise RuntimeError("validate-only mode cannot append progress")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


if __name__ == "__main__":
    main()
