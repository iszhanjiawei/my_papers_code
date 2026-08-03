"""Extract deterministic HuBERT-large targets aligned to the 40 Hz Semantic-VAE cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from typing_extensions import Self

from aligndit.script.misc.extract_librispeech_svae_latents import (
    OFFICIAL_MANIFEST_SHA256,
    SHA256_PATTERN,
    DistributedContext,
    destroy_distributed,
    distributed_barrier,
    initialize_distributed,
    iter_selected_records,
    load_local_records,
    read_json_object,
    selection_spec,
)
from aligndit.script.misc.svae_cache_utils import (
    CACHE_SCHEMA_VERSION,
    HUBERT_HIDDEN_DIM,
    SAMPLE_RATE,
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
    validate_attempt_id,
    validate_npy,
)


EXTRACTION_PROTOCOL = "hubert_large_ll60k_last_hidden_40hz_linear_v1"
STATE_RELATIVE_ROOT = Path("state/hubert_40hz")
FEATURE_RELATIVE_ROOT = "hubert_40hz"
OFFICIAL_MODEL_SHA256 = "45a299050945479a68cffe2ab7a63fd08931718ba18f05a42dbb86a5164178e0"
OFFICIAL_CONFIG_SHA256 = "52e48f7f44bf7f1be327cc3e82368681acf0cc8c5538d757eae9e6685bfe2b16"
OFFICIAL_PREPROCESSOR_SHA256 = "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"
OFFICIAL_LATENT_COMPLETE_SHA256 = "e255f8ddea5181436283510538ad1bd6bf6808bbe61d3081f3f38977c91be69b"
OFFICIAL_LATENT_INDEX_SHA256 = "94006f150a97361b7d4a8241afea1d18875c631c16b981e0c3b249ff68e303b8"
OFFICIAL_LATENT_SPEC_SHA256 = "f6d87ad506a4e20ebf5e3ced11b81590dae7ff6ca92f3aaa256062b9ec7a7ced"
GOLDEN_UTTERANCE_KEY = "train-clean-100/103/1240/103-1240-0015"
GOLDEN_NUM_SAMPLES = 60_960
GOLDEN_NATIVE_FRAMES = 190
GOLDEN_TARGET_FRAMES = 153
GOLDEN_NATIVE_RAW_SHA256 = "790c470369fe5d94828ac233213099acddaf6833461da53a664bf9f487bd120b"
GOLDEN_ALIGNED_RAW_SHA256 = "f816736e744e5db01b100df05b76d2093465208c1466cbd998ba5a9d5b535b3c"
HUBERT_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
HUBERT_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


@dataclass(frozen=True)
class ProgressEntry:
    utterance_key: str
    relative_path: str
    sha256: str
    size_bytes: int
    native_frames: int
    target_frames: int
    feature_dim: int

    def as_record(self) -> dict[str, Any]:
        return {
            "feature": EXTRACTION_PROTOCOL,
            "feature_dim": self.feature_dim,
            "native_frames": self.native_frames,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "target_frames": self.target_frames,
            "utterance_key": self.utterance_key,
        }


class NullProgressWriter:
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=prefixed_path("datasets/LibriSpeech"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=prefixed_path("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--hubert-root",
        type=Path,
        default=prefixed_path("projects/alignDiT_idea6/hubert-large-ll60k"),
    )
    parser.add_argument("--attempt-id", default=os.environ.get("HUBERT40_ATTEMPT_ID"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--utterance-key", action="append", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--progress-fsync-interval", type=int, default=100)
    parser.add_argument("--expected-manifest-sha256", default=OFFICIAL_MANIFEST_SHA256)
    parser.add_argument("--expected-model-sha256", default=OFFICIAL_MODEL_SHA256)
    parser.add_argument("--distributed-timeout-minutes", type=int, default=24 * 60)
    parser.add_argument("--acknowledge-stale-write-attempt")
    return parser.parse_args()


def hubert_native_frames(num_samples: int) -> int:
    if num_samples < 400:
        raise ValueError(f"HuBERT requires at least 400 samples, got {num_samples}")
    frames = num_samples
    for kernel, stride in zip(HUBERT_CONV_KERNEL, HUBERT_CONV_STRIDE, strict=True):
        frames = (frames - kernel) // stride + 1
    if frames <= 0:
        raise ValueError(f"Invalid HuBERT frame count for {num_samples} samples: {frames}")
    return frames


def feature_relative_path(record: Mapping[str, Any]) -> str:
    key = record["utterance_key"]
    parts = key.split("/")
    if len(parts) != 4 or any(part in {"", ".", ".."} for part in parts) or "\\" in key:
        raise ValueError(f"Invalid utterance key: {key!r}")
    return f"{FEATURE_RELATIVE_ROOT}/{key}.npy"


def manifest_metadata(manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = read_json_object(manifest.parent / "inventory_meta.json")
    entry = metadata.get("manifests", {}).get(manifest.name)
    if not isinstance(entry, dict):
        raise TypeError(f"Manifest is not registered in inventory_meta.json: {manifest}")
    return metadata, entry


def validate_model_config(config: Mapping[str, Any], preprocessor: Mapping[str, Any]) -> None:
    expected_config = {
        "architectures": ["HubertModel"],
        "conv_kernel": list(HUBERT_CONV_KERNEL),
        "conv_stride": list(HUBERT_CONV_STRIDE),
        "hidden_size": HUBERT_HIDDEN_DIM,
        "model_type": "hubert",
        "num_hidden_layers": 24,
    }
    config_mismatches = {
        key: (config.get(key), value) for key, value in expected_config.items() if config.get(key) != value
    }
    expected_preprocessor = {
        "do_normalize": True,
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "sampling_rate": SAMPLE_RATE,
    }
    preprocessor_mismatches = {
        key: (preprocessor.get(key), value)
        for key, value in expected_preprocessor.items()
        if preprocessor.get(key) != value
    }
    if config_mismatches or preprocessor_mismatches:
        raise RuntimeError(
            f"Unexpected HuBERT resource contract: config={config_mismatches}, preprocessor={preprocessor_mismatches}"
        )


def runtime_contract(context: DistributedContext) -> dict[str, Any]:
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_capability": list(torch.cuda.get_device_capability(context.device)),
        "device_name": torch.cuda.get_device_name(context.device),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }


def build_spec(args: argparse.Namespace, context: DistributedContext) -> dict[str, Any]:
    cache_root = args.cache_root.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    hubert_root = args.hubert_root.resolve(strict=True)
    manifest = (args.manifest or cache_root / "manifests/inventory.jsonl").resolve(strict=True)
    _metadata, manifest_entry = manifest_metadata(manifest)
    model_path = hubert_root / "pytorch_model.bin"
    config_path = hubert_root / "config.json"
    preprocessor_path = hubert_root / "preprocessor_config.json"
    latent_complete = cache_root / "state/latents/complete.json"
    latent_index = cache_root / "state/latents/index.jsonl"
    latent_spec = cache_root / "state/latents/spec.json"
    resources = {
        manifest: args.expected_manifest_sha256,
        model_path: args.expected_model_sha256,
        config_path: OFFICIAL_CONFIG_SHA256,
        preprocessor_path: OFFICIAL_PREPROCESSOR_SHA256,
        latent_complete: OFFICIAL_LATENT_COMPLETE_SHA256,
        latent_index: OFFICIAL_LATENT_INDEX_SHA256,
        latent_spec: OFFICIAL_LATENT_SPEC_SHA256,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for path, expected_hash in resources.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required regular resource is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            mismatches[str(path)] = (actual_hash, expected_hash)
    if mismatches:
        raise RuntimeError(f"Pinned HuBERT extraction resource mismatch: {mismatches}")
    if manifest_entry.get("sha256") != args.expected_manifest_sha256:
        raise RuntimeError("Manifest sidecar SHA does not match the pinned inventory")
    config = read_json_object(config_path)
    preprocessor = read_json_object(preprocessor_path)
    validate_model_config(config, preprocessor)
    completion = read_json_object(latent_complete)
    if (
        completion.get("count") != int(manifest_entry["count"])
        or completion.get("manifest_sha256") != args.expected_manifest_sha256
        or completion.get("consolidated_index", {}).get("sha256") != OFFICIAL_LATENT_INDEX_SHA256
    ):
        raise RuntimeError("Completed Semantic-VAE cache is not bound to the authoritative manifest/index")
    source_path = Path(__file__).resolve(strict=True)
    common_path = source_path.with_name("svae_cache_utils.py")
    manifest_helpers_path = source_path.with_name("extract_librispeech_svae_latents.py")
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "extraction": {
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "feature_dim": HUBERT_HIDDEN_DIM,
            "interpolation": {"align_corners": False, "mode": "linear"},
            "layer": "last_hidden_state",
            "output_dtype": "float32",
            "output_layout": "[time,feature]",
            "protocol": EXTRACTION_PROTOCOL,
            "source_rate": "native HuBERT convolution length (~50 Hz)",
            "target_rate": "exact Semantic-VAE latent length (40 Hz)",
        },
        "golden": {
            "aligned_raw_sha256": GOLDEN_ALIGNED_RAW_SHA256,
            "native_raw_sha256": GOLDEN_NATIVE_RAW_SHA256,
            "native_frames": GOLDEN_NATIVE_FRAMES,
            "target_frames": GOLDEN_TARGET_FRAMES,
            "utterance_key": GOLDEN_UTTERANCE_KEY,
        },
        "hubert": {
            "config_path": str(config_path),
            "config_sha256": OFFICIAL_CONFIG_SHA256,
            "model_path": str(model_path),
            "model_sha256": args.expected_model_sha256,
            "preprocessor_path": str(preprocessor_path),
            "preprocessor_sha256": OFFICIAL_PREPROCESSOR_SHA256,
            "root": str(hubert_root),
        },
        "latent_cache": {
            "complete_path": str(latent_complete),
            "complete_sha256": OFFICIAL_LATENT_COMPLETE_SHA256,
            "index_path": str(latent_index),
            "index_sha256": OFFICIAL_LATENT_INDEX_SHA256,
            "spec_path": str(latent_spec),
            "spec_sha256": OFFICIAL_LATENT_SPEC_SHA256,
        },
        "manifest": {
            "count": int(manifest_entry["count"]),
            "inventory_metadata_path": str(manifest.parent / "inventory_meta.json"),
            "inventory_metadata_sha256": sha256_file(manifest.parent / "inventory_meta.json"),
            "path": str(manifest),
            "sha256": args.expected_manifest_sha256,
        },
        "runtime": runtime_contract(context),
        "source": {
            "common_path": str(common_path),
            "common_sha256": sha256_file(common_path),
            "extractor_path": str(source_path),
            "extractor_sha256": sha256_file(source_path),
            "manifest_helpers_path": str(manifest_helpers_path),
            "manifest_helpers_sha256": sha256_file(manifest_helpers_path),
        },
    }


def validate_stored_spec(args: argparse.Namespace, spec: Mapping[str, Any], context: DistributedContext) -> None:
    if spec.get("manifest", {}).get("sha256") != args.expected_manifest_sha256:
        raise RuntimeError("Stored HuBERT spec uses a different manifest")
    if spec.get("hubert", {}).get("model_sha256") != args.expected_model_sha256:
        raise RuntimeError("Stored HuBERT spec uses a different model")
    paths_and_hashes = {
        Path(spec["manifest"]["path"]): spec["manifest"]["sha256"],
        Path(spec["manifest"]["inventory_metadata_path"]): spec["manifest"]["inventory_metadata_sha256"],
        Path(spec["hubert"]["config_path"]): spec["hubert"]["config_sha256"],
        Path(spec["hubert"]["model_path"]): spec["hubert"]["model_sha256"],
        Path(spec["hubert"]["preprocessor_path"]): spec["hubert"]["preprocessor_sha256"],
        Path(spec["latent_cache"]["complete_path"]): spec["latent_cache"]["complete_sha256"],
        Path(spec["latent_cache"]["index_path"]): spec["latent_cache"]["index_sha256"],
        Path(spec["latent_cache"]["spec_path"]): spec["latent_cache"]["spec_sha256"],
        Path(spec["source"]["common_path"]): spec["source"]["common_sha256"],
        Path(spec["source"]["extractor_path"]): spec["source"]["extractor_sha256"],
        Path(spec["source"]["manifest_helpers_path"]): spec["source"]["manifest_helpers_sha256"],
    }
    for path, expected in paths_and_hashes.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"HuBERT cache resource changed after spec publication: {path}")
    if not args.validate_only and runtime_contract(context) != spec["runtime"]:
        raise RuntimeError(f"Rank {context.rank} runtime differs from the immutable HuBERT spec")


def publish_or_validate_spec(args: argparse.Namespace, context: DistributedContext, cache_root: Path) -> dict[str, Any]:
    spec_path = cache_root / STATE_RELATIVE_ROOT / "spec.json"
    if context.is_main:
        if args.validate_only:
            if not spec_path.is_file() or spec_path.is_symlink():
                raise FileNotFoundError(f"Read-only validation requires a regular spec: {spec_path}")
            spec = read_json_object(spec_path)
        else:
            spec = build_spec(args, context)
            result = atomic_write_json(spec_path, spec)
            print(f"[rank 0] HuBERT spec {'published' if result.created else 'verified'}: {result.path}", flush=True)
    distributed_barrier(context)
    stored = read_json_object(spec_path)
    validate_stored_spec(args, stored, context)
    return stored


def acquire_write_guard(args: argparse.Namespace, context: DistributedContext, cache_root: Path) -> Path | None:
    if args.validate_only:
        return None
    guard_path = cache_root / STATE_RELATIVE_ROOT / "WRITE_ACTIVE.json"
    if context.is_main:
        if guard_path.exists() or guard_path.is_symlink():
            if not guard_path.is_file() or guard_path.is_symlink():
                raise RuntimeError(f"HuBERT write guard is not a regular file: {guard_path}")
            guarded_attempt = read_json_object(guard_path).get("attempt_id")
            if args.acknowledge_stale_write_attempt != guarded_attempt:
                raise RuntimeError(
                    f"HuBERT cache has an active/stale writer {guarded_attempt!r}; verify no writer is alive, then "
                    "acknowledge that exact attempt using a new --attempt-id"
                )
            if args.attempt_id == guarded_attempt:
                raise ValueError("A stale HuBERT guard must be resumed with a new attempt ID")
            quarantine_root = cache_root / "quarantine/hubert_40hz/write_guards" / args.attempt_id
            quarantine_file(guard_path, quarantine_root, "WRITE_ACTIVE.json")
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
            raise RuntimeError(f"HuBERT write guard already exists: {guard_path}")
        print(f"[rank 0] acquired HuBERT write guard: {guard_path}", flush=True)
    distributed_barrier(context)
    if read_json_object(guard_path).get("attempt_id") != args.attempt_id:
        raise RuntimeError("HuBERT write guard belongs to another attempt")
    return guard_path


def release_write_guard(guard_path: Path | None, context: DistributedContext) -> None:
    if guard_path is None:
        return
    distributed_barrier(context)
    if context.is_main:
        durable_unlink(guard_path)
        print(f"[rank 0] released HuBERT write guard: {guard_path}", flush=True)
    distributed_barrier(context)


def prepare_write_state(
    args: argparse.Namespace,
    context: DistributedContext,
    cache_root: Path,
    selection: Mapping[str, Any],
) -> None:
    complete_path = cache_root / STATE_RELATIVE_ROOT / "complete.json"
    if context.is_main and (complete_path.exists() or complete_path.is_symlink()):
        if selection["mode"] != "full":
            raise RuntimeError("A completed HuBERT cache cannot be reopened by a partial selection")
        if not complete_path.is_file() or complete_path.is_symlink():
            raise RuntimeError(f"HuBERT completion marker is not regular: {complete_path}")
        quarantine_root = cache_root / "quarantine/hubert_40hz/completion_markers" / args.attempt_id
        quarantine_file(complete_path, quarantine_root, "complete.json")
    distributed_barrier(context)


def parse_progress(record: Mapping[str, Any], source: Path) -> ProgressEntry:
    required = {
        "feature",
        "feature_dim",
        "native_frames",
        "relative_path",
        "sha256",
        "size_bytes",
        "target_frames",
        "utterance_key",
    }
    if set(record) != required or record.get("feature") != EXTRACTION_PROTOCOL:
        raise ValueError(f"Unexpected HuBERT progress record in {source}: {record}")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.fullmatch(record["sha256"]):
        raise ValueError(f"Invalid HuBERT progress SHA in {source}")
    entry = ProgressEntry(
        utterance_key=str(record["utterance_key"]),
        relative_path=str(record["relative_path"]),
        sha256=record["sha256"],
        size_bytes=int(record["size_bytes"]),
        native_frames=int(record["native_frames"]),
        target_frames=int(record["target_frames"]),
        feature_dim=int(record["feature_dim"]),
    )
    path = Path(entry.relative_path)
    if (
        entry.size_bytes <= 0
        or entry.native_frames <= 0
        or entry.target_frames <= 0
        or entry.feature_dim != HUBERT_HIDDEN_DIM
        or path.is_absolute()
        or path.as_posix() != entry.relative_path
        or path.parts[0] != FEATURE_RELATIVE_ROOT
        or path.suffix != ".npy"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Invalid HuBERT progress contract in {source}: {entry}")
    return entry


def load_progress_index(cache_root: Path) -> dict[str, ProgressEntry]:
    state_root = cache_root / STATE_RELATIVE_ROOT
    index: dict[str, ProgressEntry] = {}
    consolidated_path = state_root / "index.jsonl"
    sources: list[tuple[Path, Any]] = []
    if consolidated_path.exists() or consolidated_path.is_symlink():
        if not consolidated_path.is_file() or consolidated_path.is_symlink():
            raise RuntimeError(f"HuBERT index is not a regular file: {consolidated_path}")
        sources.append((consolidated_path, read_jsonl(consolidated_path)))
    progress_root = state_root / "progress"
    if progress_root.exists():
        if not progress_root.is_dir() or progress_root.is_symlink():
            raise RuntimeError(f"HuBERT progress root is not a regular directory: {progress_root}")
        for path in sorted(progress_root.glob("*.jsonl")):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"HuBERT progress log is not regular: {path}")
            sources.append((path, read_append_only_jsonl(path)))
    for source, records in sources:
        for record in records:
            entry = parse_progress(record, source)
            previous = index.get(entry.utterance_key)
            if previous is not None and previous != entry:
                raise RuntimeError(f"Conflicting HuBERT progress for {entry.utterance_key}: {source}")
            index[entry.utterance_key] = entry
    return index


def scan_feature_tree(cache_root: Path) -> dict[str, Path]:
    root = cache_root / FEATURE_RELATIVE_ROOT
    if not root.exists():
        return {}
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"HuBERT feature root is not a regular directory: {root}")
    files: dict[str, Path] = {}
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise RuntimeError(f"Symlink directory in HuBERT cache: {directory_path / name}")
        for name in filenames:
            path = directory_path / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Non-regular file in HuBERT cache: {path}")
            relative = path.relative_to(cache_root).as_posix()
            if relative in files:
                raise RuntimeError(f"Duplicate HuBERT cache path: {relative}")
            files[relative] = path
    return files


def load_model(spec: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.nn.Module]:
    hubert_root = Path(spec["hubert"]["root"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        hubert_root, local_files_only=True, trust_remote_code=False
    )
    if (
        feature_extractor.sampling_rate != SAMPLE_RATE
        or not feature_extractor.do_normalize
        or feature_extractor.return_attention_mask is not True
    ):
        raise RuntimeError("Loaded HuBERT feature extractor differs from the pinned normalization contract")
    model = AutoModel.from_pretrained(
        hubert_root,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    if model.__class__.__name__ != "HubertModel" or model.config.hidden_size != HUBERT_HIDDEN_DIM:
        raise RuntimeError(f"Unexpected HuBERT model class/config: {model.__class__.__name__}")
    model.eval().requires_grad_(False).to(device)
    run_golden_self_test(feature_extractor, model, Path(spec["dataset_root"]), device)
    return feature_extractor, model


def load_waveform(record: Mapping[str, Any], dataset_root: Path) -> np.ndarray:
    path = safe_join(dataset_root, record["audio_relative_path"])
    waveform, sample_rate = torchaudio.load(path)
    expected_shape = (int(record["num_channels"]), int(record["original_num_samples"]))
    if sample_rate != SAMPLE_RATE or sample_rate != record["sample_rate"] or tuple(waveform.shape) != expected_shape:
        raise RuntimeError(
            f"Decoded waveform contract mismatch for {record['utterance_key']}: "
            f"sample_rate={sample_rate}, shape={tuple(waveform.shape)}, expected={expected_shape}"
        )
    if waveform.shape[0] != 1:
        raise RuntimeError(f"HuBERT extraction requires mono audio: {path}")
    return waveform.squeeze(0).contiguous().numpy()


@torch.inference_mode()
def extract_feature(
    feature_extractor: Any,
    model: torch.nn.Module,
    waveform: np.ndarray,
    expected_native_frames: int,
    target_frames: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(device=device, dtype=torch.float32, non_blocking=True)
    native = model(input_values=input_values).last_hidden_state
    if native.shape != (1, expected_native_frames, HUBERT_HIDDEN_DIM):
        raise RuntimeError(
            f"HuBERT native shape mismatch: {tuple(native.shape)} != {(1, expected_native_frames, HUBERT_HIDDEN_DIM)}"
        )
    aligned = F.interpolate(native.transpose(1, 2), size=target_frames, mode="linear", align_corners=False).transpose(
        1, 2
    )
    native_array = native.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    aligned_array = aligned.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    if aligned_array.shape != (target_frames, HUBERT_HIDDEN_DIM) or not np.isfinite(aligned_array).all():
        raise RuntimeError(f"Invalid aligned HuBERT feature: shape={aligned_array.shape}")
    return native_array, aligned_array


def run_golden_self_test(
    feature_extractor: Any,
    model: torch.nn.Module,
    dataset_root: Path,
    device: torch.device,
) -> None:
    record = {
        "audio_relative_path": "train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac",
        "num_channels": 1,
        "original_num_samples": GOLDEN_NUM_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "utterance_key": GOLDEN_UTTERANCE_KEY,
    }
    waveform = load_waveform(record, dataset_root)
    native, aligned = extract_feature(
        feature_extractor,
        model,
        waveform,
        GOLDEN_NATIVE_FRAMES,
        GOLDEN_TARGET_FRAMES,
        device,
    )
    hashes = (
        hashlib.sha256(native.tobytes(order="C")).hexdigest(),
        hashlib.sha256(aligned.tobytes(order="C")).hexdigest(),
    )
    expected = (GOLDEN_NATIVE_RAW_SHA256, GOLDEN_ALIGNED_RAW_SHA256)
    if hashes != expected:
        raise RuntimeError(f"HuBERT golden self-test failed: actual={hashes}, expected={expected}")


def entry_from_validation(record: Mapping[str, Any], native_frames: int, sha256: str, size_bytes: int) -> ProgressEntry:
    return ProgressEntry(
        utterance_key=record["utterance_key"],
        relative_path=feature_relative_path(record),
        sha256=sha256,
        size_bytes=size_bytes,
        native_frames=native_frames,
        target_frames=int(record["latent_frames"]),
        feature_dim=HUBERT_HIDDEN_DIM,
    )


def process_record(
    record: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    dataset_root: Path,
    cache_root: Path,
    progress: ProgressEntry | None,
    model_holder: list[tuple[Any, torch.nn.Module]],
    spec: Mapping[str, Any],
) -> tuple[ProgressEntry, str]:
    native_frames = hubert_native_frames(int(record["original_num_samples"]))
    target_frames = int(record["latent_frames"])
    output_path = safe_join(cache_root, feature_relative_path(record))
    if output_path.is_file() and not output_path.is_symlink():
        try:
            validation = validate_npy(
                output_path,
                expected_shape=(target_frames, HUBERT_HIDDEN_DIM),
                expected_dtype=np.float32,
            )
            existing = entry_from_validation(record, native_frames, validation.sha256, validation.size_bytes)
            if progress is not None and existing != progress:
                raise RuntimeError(f"HuBERT file/progress mismatch: {existing} != {progress}")
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            if not args.repair:
                raise RuntimeError(f"Invalid HuBERT cache file {output_path}; use offline --repair: {error}") from error
            quarantine_root = cache_root / "quarantine/hubert_40hz/features" / args.attempt_id
            quarantine_file(output_path, quarantine_root, feature_relative_path(record))
        else:
            if progress is not None or args.validate_only:
                return existing, "validated" if args.validate_only else "resumed"
    elif output_path.exists() or output_path.is_symlink():
        raise RuntimeError(f"HuBERT output path is not a regular file: {output_path}")
    elif args.validate_only:
        raise FileNotFoundError(f"Missing HuBERT feature: {output_path}")

    if args.validate_only:
        raise AssertionError("validate-only unexpectedly reached extraction")
    if not model_holder:
        model_holder.append(load_model(spec, context.device))
    feature_extractor, model = model_holder[0]
    waveform = load_waveform(record, dataset_root)
    _, aligned = extract_feature(
        feature_extractor,
        model,
        waveform,
        native_frames,
        target_frames,
        context.device,
    )
    result = atomic_write_npy(output_path, aligned)
    entry = entry_from_validation(record, native_frames, result.sha256, result.size_bytes)
    if progress is not None and entry != progress:
        raise RuntimeError(f"Regenerated HuBERT feature differs from prior progress: {entry} != {progress}")
    return entry, "created" if result.created else "recovered"


def quarantine_orphans(args: argparse.Namespace, cache_root: Path, expected_paths: set[str]) -> int:
    count = 0
    for relative, path in scan_feature_tree(cache_root).items():
        if relative in expected_paths:
            continue
        quarantine_root = cache_root / "quarantine/hubert_40hz/orphans" / args.attempt_id
        quarantine_file(path, quarantine_root, relative)
        count += 1
    return count


def validate_and_publish_final(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    selection: Mapping[str, Any],
    selected_count: int,
    progress_index: Mapping[str, ProgressEntry],
) -> dict[str, Any]:
    cache_root = args.cache_root.resolve(strict=True)
    manifest = Path(spec["manifest"]["path"])
    expected_paths: set[str] = set()
    expected_keys: set[str] = set()
    index_records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    index_size = 0
    total_bytes = 0
    total_target_frames = 0
    total_native_frames = 0
    for record in iter_selected_records(manifest, selection):
        key = record["utterance_key"]
        entry = progress_index.get(key)
        if entry is None:
            raise RuntimeError(f"Missing HuBERT progress after extraction: {key}")
        expected = entry_from_validation(
            record, hubert_native_frames(record["original_num_samples"]), entry.sha256, entry.size_bytes
        )
        if entry != expected:
            raise RuntimeError(f"HuBERT progress contract mismatch for {key}: {entry} != {expected}")
        encoded = f"{canonical_json(entry.as_record())}\n".encode()
        digest.update(encoded)
        index_size += len(encoded)
        index_records.append(entry.as_record())
        expected_paths.add(entry.relative_path)
        expected_keys.add(key)
        total_bytes += entry.size_bytes
        total_target_frames += entry.target_frames
        total_native_frames += entry.native_frames
    if len(index_records) != selected_count:
        raise RuntimeError(f"HuBERT final count mismatch: {len(index_records)} != {selected_count}")
    actual_paths = set(scan_feature_tree(cache_root))
    missing = expected_paths - actual_paths
    if missing:
        raise RuntimeError(f"HuBERT cache is missing files: {sorted(missing)[:20]}")
    is_full = selection["mode"] == "full"
    if is_full:
        unexpected_progress = set(progress_index) - expected_keys
        unexpected_paths = actual_paths - expected_paths
        if unexpected_progress or unexpected_paths:
            raise RuntimeError(
                f"Unexpected HuBERT state: progress={sorted(unexpected_progress)[:20]}, "
                f"paths={sorted(unexpected_paths)[:20]}"
            )
    summary = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "count": selected_count,
        "feature": EXTRACTION_PROTOCOL,
        "manifest_sha256": spec["manifest"]["sha256"],
        "ordered_index_sha256": digest.hexdigest(),
        "selection": selection,
        "spec_sha256": sha256_file(cache_root / STATE_RELATIVE_ROOT / "spec.json"),
        "total_feature_size_bytes": total_bytes,
        "total_native_frames": total_native_frames,
        "total_target_frames": total_target_frames,
    }
    if not is_full:
        return summary
    index_path = cache_root / STATE_RELATIVE_ROOT / "index.jsonl"
    if args.validate_only:
        if not index_path.is_file() or index_path.is_symlink():
            raise FileNotFoundError(f"Full HuBERT validation requires a regular index: {index_path}")
        if index_path.stat().st_size != index_size or sha256_file(index_path) != digest.hexdigest():
            raise RuntimeError("HuBERT consolidated index differs from the manifest-order reconstruction")
    else:
        result = atomic_write_jsonl(index_path, index_records)
        if result.size_bytes != index_size or result.sha256 != digest.hexdigest():
            raise AssertionError("Published HuBERT index differs from the deterministic reconstruction")
    summary["consolidated_index"] = {
        "count": selected_count,
        "path": str(STATE_RELATIVE_ROOT / "index.jsonl"),
        "sha256": digest.hexdigest(),
        "size_bytes": index_size,
    }
    return summary


def main() -> None:
    args = get_args()
    selection = selection_spec(args)
    if args.attempt_id is None and not args.validate_only:
        raise ValueError("--attempt-id or HUBERT40_ATTEMPT_ID is required for writing")
    if args.attempt_id is not None:
        args.attempt_id = validate_attempt_id(args.attempt_id)
    if args.acknowledge_stale_write_attempt is not None:
        args.acknowledge_stale_write_attempt = validate_attempt_id(args.acknowledge_stale_write_attempt)
    if not SHA256_PATTERN.fullmatch(args.expected_manifest_sha256):
        raise ValueError("Invalid expected manifest SHA256")
    if not SHA256_PATTERN.fullmatch(args.expected_model_sha256):
        raise ValueError("Invalid expected model SHA256")
    if args.validate_only and args.repair:
        raise ValueError("--validate-only and --repair are mutually exclusive")
    if args.validate_only and args.acknowledge_stale_write_attempt:
        raise ValueError("Read-only validation cannot replace a write guard")
    if args.repair and selection["mode"] != "full":
        raise ValueError("HuBERT repair requires the full manifest")
    if args.distributed_timeout_minutes <= 0:
        raise ValueError("Distributed timeout must be positive")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic HuBERT extraction")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    context = initialize_distributed(
        require_cuda=not args.validate_only,
        timeout_minutes=args.distributed_timeout_minutes,
    )
    guard_path: Path | None = None
    try:
        if args.repair and context.world_size != 1:
            raise ValueError("HuBERT --repair is restricted to an offline single-GPU run")
        cache_root = args.cache_root.resolve(strict=True)
        complete_path = cache_root / STATE_RELATIVE_ROOT / "complete.json"
        active_guard = cache_root / STATE_RELATIVE_ROOT / "WRITE_ACTIVE.json"
        if args.validate_only and (active_guard.exists() or active_guard.is_symlink()):
            raise RuntimeError(f"Read-only HuBERT validation refuses to race a writer: {active_guard}")
        if (
            not args.validate_only
            and selection["mode"] != "full"
            and (complete_path.exists() or complete_path.is_symlink())
        ):
            raise RuntimeError("A completed HuBERT cache cannot be reopened by a partial selection")
        spec = publish_or_validate_spec(args, context, cache_root)
        guard_path = acquire_write_guard(args, context, cache_root)
        if not args.validate_only:
            prepare_write_state(args, context, cache_root, selection)
        manifest = Path(spec["manifest"]["path"])
        manifest_meta = read_json_object(manifest.parent / "inventory_meta.json")
        local_records, selected_count = load_local_records(
            manifest,
            selection,
            Path(spec["dataset_root"]),
            int(manifest_meta["base_posterior_seed"]),
            context,
        )
        if selection["mode"] == "full" and selected_count != int(spec["manifest"]["count"]):
            raise RuntimeError("Full HuBERT selection count differs from the immutable spec")
        progress_index = load_progress_index(cache_root)
        counters = {
            "created": 0,
            "recovered": 0,
            "resumed": 0,
            "validated": 0,
            "orphans_quarantined": 0,
        }
        if args.repair:
            expected_paths = {feature_relative_path(record) for record in local_records}
            counters["orphans_quarantined"] = quarantine_orphans(args, cache_root, expected_paths)
        progress_root = cache_root / STATE_RELATIVE_ROOT / "progress"
        if args.validate_only:
            writer_context: Any = NullProgressWriter()
        else:
            progress_path = progress_root / (
                f"{args.attempt_id}.rank-{context.rank:05d}-of-{context.world_size:05d}.jsonl"
            )
            writer_context = JsonlProgressWriter(progress_path, args.progress_fsync_interval)
        model_holder: list[tuple[Any, torch.nn.Module]] = []
        with writer_context as writer:
            iterator = tqdm(
                local_records, desc=f"HuBERT40 rank {context.rank}", disable=not context.is_main, dynamic_ncols=True
            )
            for record in iterator:
                entry, status = process_record(
                    record,
                    args=args,
                    context=context,
                    dataset_root=Path(spec["dataset_root"]),
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
            merged = load_progress_index(cache_root)
            completion = validate_and_publish_final(args, spec, selection, selected_count, merged)
            if args.validate_only:
                if selection["mode"] == "full":
                    if not complete_path.is_file() or complete_path.is_symlink():
                        raise RuntimeError(f"HuBERT completion marker is not regular: {complete_path}")
                    stored = read_json_object(complete_path)
                    if stored != completion:
                        raise RuntimeError("HuBERT completion marker differs from read-only validation")
                print(f"[rank 0] HuBERT read-only validation complete: {completion}", flush=True)
            elif selection["mode"] == "full":
                result = atomic_write_json(complete_path, completion)
                print(f"[rank 0] HuBERT completion published: {result.path} sha256={result.sha256}", flush=True)
            else:
                print(f"[rank 0] partial HuBERT selection validated: {completion}", flush=True)
        distributed_barrier(context)
        release_write_guard(guard_path, context)
        guard_path = None
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
