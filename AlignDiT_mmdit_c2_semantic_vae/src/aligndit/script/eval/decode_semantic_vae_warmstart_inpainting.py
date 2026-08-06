"""Decode generated warm-start latents with the pinned Semantic-VAE 1000k EMA.

Run this second-stage evaluator in the isolated Semantic-VAE environment.  It
strictly validates the generation manifest and decoder checkpoint, reverses
the train-only per-channel normalization, decodes generated and oracle cached
latents, crops both to the manifest's exact waveform length, and reports
masked-region latent and waveform metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from aligndit.script.misc.svae_cache_utils import (
    DEV_SUBSETS,
    SEMANTIC_VAE_LATENT_DIM,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    safe_join,
    sha256_file,
)


SAMPLE_RATE = 16_000
HOP_LENGTH = 400
SCHEMA_VERSION = 1
PROTOCOL_NAME = "librispeech-svae40-fixed-span-inpainting-v1"


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def configure_runtime(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def validate_decoder_dataset(
    dataset_info: dict[str, Any], *, selected_keys: set[str]
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    cache_root = Path(dataset_info["cache_root"]).resolve(strict=True)
    manifest_path = Path(dataset_info["manifest"]).resolve(strict=True)
    normalization_path = Path(dataset_info["normalization"]).resolve(strict=True)
    if manifest_path != safe_join(cache_root, "manifests/dev.jsonl"):
        raise RuntimeError("Generation metadata does not bind the authoritative development manifest")
    if normalization_path != safe_join(cache_root, "state/latents/train_normalization.json"):
        raise RuntimeError("Generation metadata does not bind the authoritative train-only normalization")
    latent_completion_path = safe_join(cache_root, "state/latents/complete.json")
    hash_contract = {
        manifest_path: dataset_info["manifest_sha256"],
        normalization_path: dataset_info["normalization_sha256"],
        latent_completion_path: dataset_info["latent_completion_sha256"],
        safe_join(cache_root, "state/hubert_40hz/complete.json"): dataset_info["hubert_completion_sha256"],
    }
    for path, expected_sha256 in hash_contract.items():
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Current cache resource differs from generation metadata: {path}")

    records = list(read_jsonl(manifest_path))
    if len(records) != dataset_info["manifest_count"] or not records:
        raise RuntimeError("Development manifest count differs from generation metadata")
    keys = [record.get("utterance_key") for record in records]
    if len(set(keys)) != len(keys) or any(record.get("subset") not in DEV_SUBSETS for record in records):
        raise RuntimeError("Development manifest has invalid keys or subsets")
    subset_counts = {subset: sum(record["subset"] == subset for record in records) for subset in DEV_SUBSETS}
    current_metadata = {
        "hubert_completion_sha256": dataset_info["hubert_completion_sha256"],
        "latent_completion_sha256": dataset_info["latent_completion_sha256"],
        "manifest_count": len(records),
        "manifest_sha256": sha256_file(manifest_path),
        "normalization_sha256": sha256_file(normalization_path),
        "selected_count": len(records),
        "selected_counts": subset_counts,
        "selected_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "subset_counts": subset_counts,
    }
    expected_metadata = {
        key: value for key, value in dataset_info.items() if key not in {"cache_root", "manifest", "normalization"}
    }
    if current_metadata != expected_metadata:
        raise RuntimeError("Current development cache metadata differs from generation completion metadata")
    records_by_key = {record["utterance_key"]: record for record in records}
    missing_records = sorted(selected_keys - set(records_by_key))
    if not selected_keys or missing_records:
        raise RuntimeError(f"Generation selection is absent from the development manifest: {missing_records[:3]}")

    normalization = read_json_object(normalization_path)
    if (
        normalization.get("scope") != "train"
        or normalization.get("channel_count") != SEMANTIC_VAE_LATENT_DIM
        or normalization.get("feature") != "semantic_vae_posterior_sample_v1"
        or normalization.get("latent_complete_sha256") != dataset_info["latent_completion_sha256"]
    ):
        raise RuntimeError("Invalid train-only normalization contract")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if (
        mean.shape != (SEMANTIC_VAE_LATENT_DIM,)
        or std.shape != (SEMANTIC_VAE_LATENT_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise RuntimeError("Invalid train-only normalization values")

    latent_completion = read_json_object(latent_completion_path)
    spec_path = safe_join(cache_root, "state/latents/spec.json")
    if sha256_file(spec_path) != latent_completion.get("spec_sha256"):
        raise RuntimeError("Semantic-VAE latent spec differs from its completion marker")
    index_info = latent_completion.get("consolidated_index")
    if not isinstance(index_info, dict) or index_info.get("path") != "state/latents/index.jsonl":
        raise RuntimeError("Latent completion marker has an invalid consolidated index binding")
    index_path = safe_join(cache_root, index_info["path"])
    if index_path.stat().st_size != index_info.get("size_bytes") or sha256_file(index_path) != index_info.get("sha256"):
        raise RuntimeError("Latent consolidated index differs from its completion marker")
    selected_index: dict[str, dict[str, Any]] = {}
    index_count = 0
    for entry in read_jsonl(index_path):
        index_count += 1
        key = entry.get("utterance_key")
        if key in selected_keys:
            if key in selected_index:
                raise RuntimeError(f"Duplicate selected key in latent consolidated index: {key}")
            selected_index[key] = entry
    if index_count != index_info.get("count"):
        raise RuntimeError(f"Latent consolidated index count mismatch: {index_count} != {index_info.get('count')}")
    missing_index = sorted(selected_keys - set(selected_index))
    if missing_index:
        raise RuntimeError(f"Selected latents are absent from the consolidated index: {missing_index[:3]}")
    for key, entry in selected_index.items():
        record = records_by_key[key]
        expected = {
            "feature": "semantic_vae_posterior_sample_v1",
            "latent_dim": SEMANTIC_VAE_LATENT_DIM,
            "latent_frames": int(record["latent_frames"]),
            "relative_path": record["latent_relative_path"],
            "utterance_key": key,
        }
        for field, value in expected.items():
            if entry.get(field) != value:
                raise RuntimeError(f"Latent index mismatch for {key}/{field}: {entry.get(field)!r} != {value!r}")
        if not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("size_bytes"), int):
            raise TypeError(f"Latent index has invalid integrity metadata for {key}")
    return records, mean, std, selected_index


def validate_exact_length(frames: int, original_samples: int, padded_samples: int) -> int:
    if frames <= 0 or original_samples <= 0 or padded_samples <= 0:
        raise ValueError("Latent and waveform lengths must be positive")
    expected_padded = frames * HOP_LENGTH
    if padded_samples != expected_padded:
        raise ValueError(f"Expected padded_samples={expected_padded} for {frames} frames, got {padded_samples}")
    if not expected_padded - HOP_LENGTH < original_samples <= expected_padded:
        raise ValueError(f"original_samples={original_samples} is inconsistent with ceil(length/{HOP_LENGTH})={frames}")
    return expected_padded


def stable_seed(eval_seed: int, purpose: str, utterance_key: str) -> int:
    payload = f"{PROTOCOL_NAME}:{eval_seed}:{purpose}:0:{utterance_key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def expected_inpainting_span(
    *, utterance_key: str, frames: int, mask_fraction: float, eval_seed: int
) -> tuple[int, int]:
    if frames < 2 or not 0 < mask_fraction < 1:
        raise ValueError(f"Invalid fixed-span protocol for {utterance_key}: frames={frames}, fraction={mask_fraction}")
    masked_frames = max(1, min(frames - 1, math.floor(mask_fraction * frames)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(eval_seed, "inpainting_mask_start", utterance_key))
    start_u = float(torch.rand((), generator=generator, dtype=torch.float32).item())
    max_start = frames - masked_frames
    start = min(max_start, int((max_start + 1) * start_u))
    return start, start + masked_frames


def inverse_normalize(normalized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if normalized.ndim != 2 or normalized.shape[1] != SEMANTIC_VAE_LATENT_DIM:
        raise ValueError(f"Expected normalized latent [T,64], got {normalized.shape}")
    if mean.shape != (SEMANTIC_VAE_LATENT_DIM,) or std.shape != (SEMANTIC_VAE_LATENT_DIM,):
        raise ValueError(f"Expected mean/std [64], got {mean.shape}/{std.shape}")
    raw = normalized.astype(np.float32, copy=False) * std + mean
    raw = raw.astype(np.float32, copy=False)
    if not np.isfinite(raw).all():
        raise FloatingPointError("Inverse-normalized latent contains non-finite values")
    return raw


def latent_error_metrics(generated: np.ndarray, target: np.ndarray, start: int, end: int) -> dict[str, float | int]:
    if generated.shape != target.shape or generated.ndim != 2:
        raise ValueError(f"Generated/target latent shape mismatch: {generated.shape}/{target.shape}")
    if not 0 <= start < end <= generated.shape[0]:
        raise ValueError(f"Invalid latent metric span [{start}, {end}) for length {generated.shape[0]}")
    generated_span = torch.from_numpy(generated[start:end]).float()
    target_span = torch.from_numpy(target[start:end]).float()
    difference = generated_span - target_span
    cosine = F.cosine_similarity(generated_span, target_span, dim=-1)
    absolute = difference.abs().double()
    squared = difference.square().double()
    metrics = {
        "latent_cosine": float(cosine.double().mean().item()),
        "latent_cosine_sum": float(cosine.double().sum().item()),
        "latent_element_count": int(difference.numel()),
        "latent_frame_count": int(end - start),
        "latent_mae": float(absolute.mean().item()),
        "latent_absolute_error_sum": float(absolute.sum().item()),
        "latent_mse": float(squared.mean().item()),
        "latent_squared_error_sum": float(squared.sum().item()),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite latent metrics: {metrics}")
    return metrics


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    if reference.shape != estimate.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError(f"SI-SDR expects equal non-empty vectors, got {reference.shape}/{estimate.shape}")
    reference64 = reference.astype(np.float64, copy=False)
    estimate64 = estimate.astype(np.float64, copy=False)
    reference64 = reference64 - reference64.mean()
    estimate64 = estimate64 - estimate64.mean()
    reference_energy = float(np.dot(reference64, reference64))
    if reference_energy <= eps:
        raise ValueError("SI-SDR reference segment has negligible energy")
    scale = float(np.dot(estimate64, reference64)) / (reference_energy + eps)
    target = scale * reference64
    noise = estimate64 - target
    ratio = (float(np.dot(target, target)) + eps) / (float(np.dot(noise, noise)) + eps)
    result = 10.0 * math.log10(ratio)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite SI-SDR: {result}")
    return result


def atomic_save_waveform(path: Path, waveform: torch.Tensor) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing decoded waveform: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.wav")
    try:
        torchaudio.save(str(temporary), waveform, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_semantic_vae_decoder(
    *, repo: Path, checkpoint_root: Path, cache_spec: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    repo = repo.resolve(strict=True)
    checkpoint_root = checkpoint_root.resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dac.model.dac import DAC

    metainfo_path = checkpoint_root / "metainfo.json"
    checkpoint_path = checkpoint_root / "dac/ema_state_dict.pth"
    metainfo = read_json_object(metainfo_path)
    config = dict(metainfo.get("DAC", {}))
    bigvgan_config = Path(config["bigvgan_conf"])
    if not bigvgan_config.is_absolute():
        bigvgan_config = repo / bigvgan_config
    config["bigvgan_conf"] = str(bigvgan_config.resolve(strict=True))

    checkpoint_contract = cache_spec.get("checkpoint", {})
    source_contract = cache_spec.get("semantic_vae_source", {})
    expected_hashes = {
        metainfo_path: checkpoint_contract.get("metainfo_sha256"),
        checkpoint_path: checkpoint_contract.get("ema_sha256"),
        bigvgan_config: source_contract.get("bigvgan_config_sha256"),
    }
    for path, expected_sha256 in expected_hashes.items():
        if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Semantic-VAE decoder resource differs from latent cache contract: {path}")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != source_contract.get("commit") or status:
        raise RuntimeError(
            f"Semantic-VAE source differs from latent cache contract: commit={commit}, status={status!r}"
        )

    model = DAC(**config)
    del model.projectors
    target_state = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Semantic-VAE EMA checkpoint root must be a mapping")
    if not bool(checkpoint.get("initted")):
        raise RuntimeError("Semantic-VAE EMA checkpoint is not initialized")
    ema_step = checkpoint.get("step")
    if isinstance(ema_step, torch.Tensor):
        ema_step = ema_step.item()
    if not isinstance(ema_step, int) or isinstance(ema_step, bool) or ema_step != checkpoint_contract.get("ema_step"):
        raise RuntimeError(f"Unexpected Semantic-VAE 1000k EMA step: {ema_step!r}")

    selected: dict[str, torch.Tensor] = {}
    ignored_legacy_decoder_projection: list[str] = []
    for raw_key, value in checkpoint.items():
        key = raw_key.removeprefix("ema_model.")
        if key in {"initted", "step"} or key.startswith("projectors."):
            continue
        if key not in target_state:
            if key.startswith("decoder_proj."):
                ignored_legacy_decoder_projection.append(key)
                continue
            raise KeyError(f"Unexpected Semantic-VAE EMA key: {raw_key}")
        if target_state[key].shape != value.shape or target_state[key].dtype != value.dtype:
            raise ValueError(
                f"Semantic-VAE EMA tensor mismatch for {key}: "
                f"target={target_state[key].shape}/{target_state[key].dtype}, "
                f"checkpoint={value.shape}/{value.dtype}"
            )
        selected[key] = value
    missing = sorted(set(target_state) - set(selected))
    if missing:
        raise RuntimeError(f"Semantic-VAE EMA is missing current model keys: {missing}")
    model.load_state_dict(selected, strict=True)
    if int(model.sample_rate) != SAMPLE_RATE or int(model.hop_length) != HOP_LENGTH or int(model.vae_dim) != 64:
        raise RuntimeError(
            f"Unexpected Semantic-VAE geometry: sample_rate={model.sample_rate}, "
            f"hop={model.hop_length}, vae_dim={model.vae_dim}"
        )
    decoder = model.decoder.eval().requires_grad_(False).to(device)
    del model, checkpoint, selected, target_state
    return decoder, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ema_step": ema_step,
        "git_commit": commit,
        "ignored_legacy_decoder_projection_keys": len(ignored_legacy_decoder_projection),
        "metainfo": str(metainfo_path),
        "metainfo_sha256": sha256_file(metainfo_path),
        "repo": str(repo),
    }


def load_generation(generation_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_dir = generation_dir.resolve(strict=True)
    complete_path = generation_dir / "generation_complete.json"
    complete = read_json_object(complete_path)
    if complete.get("schema_version") != SCHEMA_VERSION or complete.get("protocol", {}).get("name") != PROTOCOL_NAME:
        raise RuntimeError(f"Unsupported inpainting generation protocol: {complete_path}")
    manifest_info = complete.get("generation_manifest")
    if not isinstance(manifest_info, dict) or manifest_info.get("path") != "generation_manifest.jsonl":
        raise RuntimeError("Generation completion marker has an invalid manifest binding")
    manifest_path = generation_dir / manifest_info["path"]
    if sha256_file(manifest_path) != manifest_info.get("sha256"):
        raise RuntimeError("Generation manifest differs from its completion marker")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest_info.get("count") or not rows:
        raise RuntimeError("Generation manifest count differs from its completion marker")
    keys = [row.get("utterance_key") for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Generation manifest contains duplicate utterance keys")
    selected_digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    if selected_digest != complete.get("selected_keys_sha256"):
        raise RuntimeError("Generation selection differs from its completion marker")
    return rows, complete


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("latent_mse", "latent_mae", "latent_cosine", "masked_stoi", "masked_si_sdr_db")
    groups = {
        "overall": rows,
        "dev-clean": [row for row in rows if row["subset"] == "dev-clean"],
        "dev-other": [row for row in rows if row["subset"] == "dev-other"],
    }
    result: dict[str, Any] = {}
    for name, group in groups.items():
        if not group:
            raise RuntimeError(f"Decoded result omitted required group {name}")
        result[name] = {
            "aggregation": "utterance_macro; population_std_ddof_0",
            "masked_audio_samples": sum(row["masked_audio_samples"] for row in group),
            "masked_latent_frames": sum(row["latent_frame_count"] for row in group),
            "utterances": len(group),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in group], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite {metric_name} values in {name}")
            result[name][f"{metric_name}_mean"] = float(values.mean())
            result[name][f"{metric_name}_std"] = float(values.std())
        element_count = sum(row["latent_element_count"] for row in group)
        frame_count = sum(row["latent_frame_count"] for row in group)
        result[name]["latent_mae_micro"] = math.fsum(row["latent_absolute_error_sum"] for row in group) / element_count
        result[name]["latent_mse_micro"] = math.fsum(row["latent_squared_error_sum"] for row in group) / element_count
        result[name]["latent_cosine_micro"] = math.fsum(row["latent_cosine_sum"] for row in group) / frame_count
        audio_weights = np.asarray([row["masked_audio_samples"] for row in group], dtype=np.float64)
        result[name]["masked_stoi_duration_weighted"] = float(
            np.average([row["masked_stoi"] for row in group], weights=audio_weights)
        )
        result[name]["masked_si_sdr_db_duration_weighted"] = float(
            np.average([row["masked_si_sdr_db"] for row in group], weights=audio_weights)
        )
    return result


def decode(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from pystoi import stoi
    except ImportError as error:
        raise RuntimeError("The Semantic-VAE decoder environment must provide pystoi") from error

    started_at = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    configure_runtime(device)
    generation_dir = args.generation_dir.resolve(strict=True)
    generation_rows, generation_complete = load_generation(generation_dir)
    if generation_complete.get("label") != args.label:
        raise RuntimeError(f"Decode label must match generation label: {args.label!r}")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / args.label
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing decode directory: {final_dir}")

    dataset_info = generation_complete["dataset"]
    cache_root = Path(dataset_info["cache_root"])
    selected_keys = {row["utterance_key"] for row in generation_rows}
    records, mean, std, latent_index = validate_decoder_dataset(dataset_info, selected_keys=selected_keys)
    records_by_key = {record["utterance_key"]: record for record in records}
    latent_spec_path = safe_join(cache_root, "state/latents/spec.json")
    latent_spec = read_json_object(latent_spec_path)
    decoder, decoder_metadata = load_semantic_vae_decoder(
        repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint,
        cache_spec=latent_spec,
        device=device,
    )
    decoder_metadata["latent_cache_spec"] = str(latent_spec_path)
    decoder_metadata["latent_cache_spec_sha256"] = sha256_file(latent_spec_path)

    protocol = generation_complete["protocol"]
    eval_seed = protocol.get("eval_seed")
    mask_fraction = protocol.get("mask_fraction")
    limit_per_subset = protocol.get("limit_per_subset")
    if not isinstance(eval_seed, int) or isinstance(eval_seed, bool):
        raise TypeError(f"Invalid generation eval seed: {eval_seed!r}")
    if not isinstance(mask_fraction, (int, float)) or isinstance(mask_fraction, bool) or not 0 < mask_fraction < 1:
        raise RuntimeError(f"Invalid generation mask fraction: {mask_fraction!r}")
    if (
        not isinstance(limit_per_subset, int)
        or isinstance(limit_per_subset, bool)
        or limit_per_subset <= 0
        or len(generation_rows) != 2 * limit_per_subset
    ):
        raise RuntimeError(
            f"Invalid balanced generation count: rows={len(generation_rows)}, limit={limit_per_subset!r}"
        )

    attempt_dir = Path(tempfile.mkdtemp(prefix=f".{args.label}.", suffix=".tmp", dir=output_root))
    try:
        decoded_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, row in enumerate(generation_rows, start=1):
                key = row["utterance_key"]
                record = records_by_key[key]
                index_entry = latent_index[key]
                bound_fields = {
                    "audio_relative_path": record["audio_relative_path"],
                    "latent_frames": int(record["latent_frames"]),
                    "original_num_samples": int(record["original_num_samples"]),
                    "padded_num_samples": int(record["padded_num_samples"]),
                    "speaker_id": str(record["speaker_id"]),
                    "subset": record["subset"],
                    "target_latent_relative_path": record["latent_relative_path"],
                    "target_latent_sha256": index_entry["sha256"],
                    "target_latent_size_bytes": index_entry["size_bytes"],
                    "text": record["text"],
                }
                for field, expected in bound_fields.items():
                    if row.get(field) != expected:
                        raise RuntimeError(f"Generation manifest field mismatch for {key}/{field}: {row.get(field)!r}")

                frames = bound_fields["latent_frames"]
                original_samples = bound_fields["original_num_samples"]
                padded_samples = bound_fields["padded_num_samples"]
                try:
                    validate_exact_length(frames, original_samples, padded_samples)
                except ValueError as error:
                    raise RuntimeError(f"Invalid exact length metadata for {key}: {error}") from error
                mask_start = row.get("mask_start")
                mask_end = row.get("mask_end")
                if not isinstance(mask_start, int) or isinstance(mask_start, bool):
                    raise TypeError(f"Invalid integer mask_start for {key}: {mask_start!r}")
                if not isinstance(mask_end, int) or isinstance(mask_end, bool):
                    raise TypeError(f"Invalid integer mask_end for {key}: {mask_end!r}")
                expected_start, expected_end = expected_inpainting_span(
                    utterance_key=key,
                    frames=frames,
                    mask_fraction=float(mask_fraction),
                    eval_seed=eval_seed,
                )
                if (mask_start, mask_end) != (expected_start, expected_end):
                    raise RuntimeError(
                        f"Generation mask differs from the keyed protocol for {key}: "
                        f"{(mask_start, mask_end)} != {(expected_start, expected_end)}"
                    )
                expected_realized = (mask_end - mask_start) / frames
                if row.get("mask_fraction_requested") != mask_fraction or not math.isclose(
                    float(row.get("mask_fraction_realized", math.nan)), expected_realized, rel_tol=0, abs_tol=1e-12
                ):
                    raise RuntimeError(f"Generation mask fraction metadata mismatch for {key}")
                if row.get("ode_seed") != stable_seed(eval_seed, "inpainting_ode_noise", key):
                    raise RuntimeError(f"Generation ODE seed differs from the keyed protocol for {key}")

                generated_path = safe_join(generation_dir, row["generated_latent_relative_path"])
                if sha256_file(generated_path) != row["generated_latent_sha256"]:
                    raise RuntimeError(f"Generated latent differs from its manifest digest: {key}")
                target_path = safe_join(cache_root, row["target_latent_relative_path"])
                if (
                    target_path.stat().st_size != index_entry["size_bytes"]
                    or sha256_file(target_path) != index_entry["sha256"]
                ):
                    raise RuntimeError(f"Target latent differs from the consolidated index: {key}")
                generated_normalized = np.load(generated_path, allow_pickle=False)
                target_raw = np.load(target_path, allow_pickle=False)
                expected_shape = (frames, SEMANTIC_VAE_LATENT_DIM)
                for name, latent in (("generated", generated_normalized), ("target", target_raw)):
                    if latent.shape != expected_shape or latent.dtype != np.float32 or not np.isfinite(latent).all():
                        raise ValueError(f"Invalid {name} latent for {key}: {latent.shape}/{latent.dtype}")
                target_normalized = ((target_raw - mean) / std).astype(np.float32, copy=False)
                keep = np.ones(frames, dtype=bool)
                keep[mask_start:mask_end] = False
                if not np.array_equal(generated_normalized[keep], target_normalized[keep]):
                    raise RuntimeError(f"Generated latent changed observed frames for {key}")
                metrics = latent_error_metrics(generated_normalized, target_normalized, mask_start, mask_end)
                generated_raw = inverse_normalize(generated_normalized, mean, std)

                generated_wave = decoder(torch.from_numpy(generated_raw).unsqueeze(0).transpose(1, 2).to(device))
                oracle_wave = decoder(torch.from_numpy(target_raw).unsqueeze(0).transpose(1, 2).to(device))
                generated_wave = generated_wave.squeeze(0).float().cpu()
                oracle_wave = oracle_wave.squeeze(0).float().cpu()
                if (
                    generated_wave.shape != (1, padded_samples)
                    or oracle_wave.shape != (1, padded_samples)
                    or not torch.isfinite(generated_wave).all()
                    or not torch.isfinite(oracle_wave).all()
                ):
                    raise RuntimeError(
                        f"Semantic-VAE decoder length/value mismatch for {key}: "
                        f"generated={tuple(generated_wave.shape)}, oracle={tuple(oracle_wave.shape)}, "
                        f"expected={(1, padded_samples)}"
                    )
                generated_wave = generated_wave[:, :original_samples]
                oracle_wave = oracle_wave[:, :original_samples]
                sample_start = mask_start * HOP_LENGTH
                sample_end = min(mask_end * HOP_LENGTH, original_samples)
                if sample_start >= sample_end:
                    raise RuntimeError(f"Empty masked waveform interval for {key}")
                generated_segment = generated_wave[0, sample_start:sample_end].numpy()
                oracle_segment = oracle_wave[0, sample_start:sample_end].numpy()
                metrics["masked_stoi"] = float(stoi(oracle_segment, generated_segment, SAMPLE_RATE, extended=False))
                metrics["masked_si_sdr_db"] = si_sdr(oracle_segment, generated_segment)
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError(f"Non-finite decoded metrics for {key}: {metrics}")

                generated_relative = Path("generated") / f"{key}.wav"
                oracle_relative = Path("oracle") / f"{key}.wav"
                generated_output = safe_join(attempt_dir, generated_relative.as_posix())
                oracle_output = safe_join(attempt_dir, oracle_relative.as_posix())
                atomic_save_waveform(generated_output, generated_wave)
                atomic_save_waveform(oracle_output, oracle_wave)
                decoded_row = {
                    **row,
                    **metrics,
                    "generated_wave_relative_path": generated_relative.as_posix(),
                    "generated_wave_sha256": sha256_file(generated_output),
                    "masked_audio_end_sample": sample_end,
                    "masked_audio_samples": sample_end - sample_start,
                    "masked_audio_start_sample": sample_start,
                    "native_decoded_samples": padded_samples,
                    "oracle_wave_relative_path": oracle_relative.as_posix(),
                    "oracle_wave_sha256": sha256_file(oracle_output),
                    "saved_samples": original_samples,
                }
                decoded_rows.append(decoded_row)
                print(f"{args.label}: decoded {index}/{len(generation_rows)} {key}", flush=True)

        rows_path = attempt_dir / "decoded_metrics.jsonl"
        atomic_write_jsonl(rows_path, decoded_rows)
        summary = {
            "decoder": decoder_metadata,
            "generation": {
                "complete": str(generation_dir / "generation_complete.json"),
                "complete_sha256": sha256_file(generation_dir / "generation_complete.json"),
                "label": generation_complete["label"],
                "manifest_sha256": generation_complete["generation_manifest"]["sha256"],
            },
            "label": args.label,
            "metrics": {
                "audio_reference": "oracle waveform decoded from the cached fixed posterior latent",
                "audio_scope": "nominal frame-aligned masked interval; decoder receptive field crosses boundaries",
                "latent_reference": "cached normalized fixed posterior latent",
                "latent_scope": "masked latent frames only",
            },
            "results": summarize(decoded_rows),
            "rows": {"count": len(decoded_rows), "path": rows_path.name, "sha256": sha256_file(rows_path)},
            "runtime": {
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "elapsed_seconds": time.time() - started_at,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
            },
            "schema_version": SCHEMA_VERSION,
            "source_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(attempt_dir / "decode_summary.json", summary)
        os.replace(attempt_dir, final_dir)
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return summary
    finally:
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)


def parse_args() -> argparse.Namespace:
    root_prefix = os.environ.get("ROOT_PREFIX", "")
    workspace = Path(f"{root_prefix}/zjw524/projects/alignDiT_idea6")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-vae-repo", type=Path, default=workspace / "papers_codes/Semantic-VAE")
    parser.add_argument(
        "--semantic-vae-checkpoint",
        type=Path,
        default=workspace / "Semantic-VAE/Semantic-VAE/semantic_vae_1000k",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in args.label
    ):
        parser.error("--label must contain only letters, digits, period, underscore, and hyphen")
    return args


def main() -> None:
    decode(parse_args())


if __name__ == "__main__":
    main()
