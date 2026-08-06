"""Generate deterministic Semantic-VAE latent inpainting samples.

This is the first half of the small warm-start generation check.  It runs in
the AlignDiT environment, loads one checkpoint's EMA weights through the
strict held-out evaluator, and writes normalized 64-D/40-Hz latent arrays.
Waveform decoding is deliberately a separate process so it can run in the
isolated Semantic-VAE environment without changing the training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aligndit.script.eval.eval_semantic_vae_warmstart_dev import (
    atomic_write_json,
    atomic_write_jsonl,
    build_and_load_ema,
    configure_runtime,
    read_json_object,
    validate_and_load_dev,
)
from aligndit.script.misc.svae_cache_utils import SEMANTIC_VAE_LATENT_DIM, read_jsonl, safe_join, sha256_file


SCHEMA_VERSION = 1
PROTOCOL_NAME = "librispeech-svae40-fixed-span-inpainting-v1"
SUBSETS = ("dev-clean", "dev-other")
CANONICAL_DEV_MANIFEST_SHA256 = "a7e4fa1298cc2b6f3f0489c702dbc0b5f1d47a56ab279e54a7ab865c224f8c57"
CANONICAL_SELECTION_KEYS_SHA256 = "f69e3f4bcc573f814d07cdb35113c95cd1e28093955f86c898c087a4efff798a"


def stable_seed(eval_seed: int, purpose: str, utterance_key: str) -> int:
    """Derive an RNG seed owned by this protocol rather than another evaluator."""

    payload = f"{PROTOCOL_NAME}:{eval_seed}:{purpose}:0:{utterance_key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def stable_uniform(eval_seed: int, purpose: str, utterance_key: str) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(eval_seed, purpose, utterance_key))
    return float(torch.rand((), generator=generator, dtype=torch.float32).item())


def selection_digest(eval_seed: int, utterance_key: str) -> bytes:
    payload = f"{PROTOCOL_NAME}:{eval_seed}:selection:{utterance_key}".encode()
    return hashlib.sha256(payload).digest()


def select_inpainting_records(
    records: list[dict[str, Any]],
    *,
    eval_seed: int,
    limit_per_subset: int,
    min_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    """Choose a balanced, deterministic set with at most one item per speaker."""

    if limit_per_subset <= 0:
        raise ValueError("limit_per_subset must be positive")
    if not 0 < min_duration <= max_duration:
        raise ValueError("Duration bounds must satisfy 0 < min <= max")

    selected: list[dict[str, Any]] = []
    for subset in SUBSETS:
        candidates = [
            record
            for record in records
            if record["subset"] == subset and min_duration <= float(record["duration_seconds"]) <= max_duration
        ]
        candidates.sort(key=lambda record: selection_digest(eval_seed, record["utterance_key"]))
        seen_speakers: set[str] = set()
        subset_records: list[dict[str, Any]] = []
        for record in candidates:
            speaker = str(record["speaker_id"])
            if speaker in seen_speakers:
                continue
            seen_speakers.add(speaker)
            subset_records.append(record)
            if len(subset_records) == limit_per_subset:
                break
        if len(subset_records) != limit_per_subset:
            raise RuntimeError(
                f"Only {len(subset_records)} eligible unique speakers in {subset}; "
                f"need {limit_per_subset} within [{min_duration}, {max_duration}] seconds"
            )
        selected.extend(subset_records)
    return selected


def fixed_inpainting_span(*, utterance_key: str, frames: int, mask_fraction: float, eval_seed: int) -> tuple[int, int]:
    if frames < 2:
        raise ValueError(f"frames must be at least two so conditioning remains, got {frames}")
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must satisfy 0 < value < 1")
    masked_frames = max(1, min(frames - 1, math.floor(mask_fraction * frames)))
    max_start = frames - masked_frames
    start_u = stable_uniform(eval_seed, "inpainting_mask_start", utterance_key)
    start = min(max_start, int((max_start + 1) * start_u))
    return start, start + masked_frames


def build_keep_mask(*, frames: int, mask_start: int, mask_end: int, device: torch.device) -> torch.Tensor:
    """Return sampler semantics: True is observed; False is generated."""

    if frames < 2 or not 0 <= mask_start < mask_end <= frames:
        raise ValueError(f"Invalid inpainting span [{mask_start}, {mask_end}) for {frames} frames")
    keep_mask = torch.ones((1, frames), dtype=torch.bool, device=device)
    keep_mask[:, mask_start:mask_end] = False
    return keep_mask


def validate_generation_length(*, frames: int, original_samples: int, padded_samples: int) -> None:
    if frames <= 0 or original_samples <= 0 or padded_samples != frames * 400:
        raise ValueError(
            f"Invalid 40-Hz length contract: frames={frames}, original={original_samples}, padded={padded_samples}"
        )
    if not padded_samples - 400 < original_samples <= padded_samples:
        raise ValueError(f"original_samples={original_samples} is inconsistent with ceil(length/400)={frames}")


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing generated latent: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_normalized_latent(
    *,
    cache_root: Path,
    record: dict[str, Any],
    index_entry: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    frames = int(record["latent_frames"])
    latent_path = safe_join(cache_root, record["latent_relative_path"])
    if latent_path.stat().st_size != index_entry["size_bytes"] or sha256_file(latent_path) != index_entry["sha256"]:
        raise RuntimeError(f"Cached latent differs from the consolidated index: {record['utterance_key']}")
    latent = np.load(latent_path, allow_pickle=False)
    if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
        raise ValueError(f"Invalid cached latent for {record['utterance_key']}: {latent.shape}/{latent.dtype}")
    normalized = ((latent - mean) / std).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise FloatingPointError(f"Non-finite normalized latent for {record['utterance_key']}")
    return normalized


def load_selected_latent_index(*, cache_root: Path, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate the consolidated index and retain only the selected entries."""

    completion = read_json_object(safe_join(cache_root, "state/latents/complete.json"))
    index_info = completion.get("consolidated_index")
    if not isinstance(index_info, dict) or index_info.get("path") != "state/latents/index.jsonl":
        raise RuntimeError("Latent completion marker has an invalid consolidated index binding")
    index_path = safe_join(cache_root, index_info["path"])
    if index_path.stat().st_size != index_info.get("size_bytes") or sha256_file(index_path) != index_info.get("sha256"):
        raise RuntimeError("Latent consolidated index differs from its completion marker")
    selected_keys = {record["utterance_key"] for record in records}
    entries: dict[str, dict[str, Any]] = {}
    count = 0
    for entry in read_jsonl(index_path):
        count += 1
        key = entry.get("utterance_key")
        if key in selected_keys:
            if key in entries:
                raise RuntimeError(f"Duplicate selected key in latent consolidated index: {key}")
            entries[key] = entry
    if count != index_info.get("count"):
        raise RuntimeError(f"Latent consolidated index count mismatch: {count} != {index_info.get('count')}")
    missing = sorted(selected_keys - set(entries))
    if missing:
        raise RuntimeError(f"Selected latents are absent from the consolidated index: {missing[:3]}")
    for record in records:
        key = record["utterance_key"]
        entry = entries[key]
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
    return entries


def generate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    configure_runtime(device)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / args.label
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing generation directory: {final_dir}")

    records, mean, std, dataset_metadata = validate_and_load_dev(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        normalization_path=args.normalization,
        eval_seed=args.eval_seed,
        limit_per_subset=None,
    )
    selected = select_inpainting_records(
        records,
        eval_seed=args.eval_seed,
        limit_per_subset=args.limit_per_subset,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    selected_keys_sha256 = hashlib.sha256(
        "\n".join(record["utterance_key"] for record in selected).encode()
    ).hexdigest()
    canonical_fixture_verified = (
        dataset_metadata["manifest_sha256"] == CANONICAL_DEV_MANIFEST_SHA256
        and args.eval_seed == 666
        and args.limit_per_subset == 25
        and args.min_duration == 4.0
        and args.max_duration == 10.0
    )
    if canonical_fixture_verified and selected_keys_sha256 != CANONICAL_SELECTION_KEYS_SHA256:
        raise RuntimeError(
            "Canonical 50-utterance inpainting fixture changed: "
            f"{selected_keys_sha256} != {CANONICAL_SELECTION_KEYS_SHA256}"
        )
    for record in selected:
        frames = int(record["latent_frames"])
        if frames > args.max_frames:
            raise ValueError(f"{record['utterance_key']} has {frames} frames, exceeding --max-frames={args.max_frames}")
        validate_generation_length(
            frames=frames,
            original_samples=int(record["original_num_samples"]),
            padded_samples=int(record["padded_num_samples"]),
        )
    latent_index = load_selected_latent_index(cache_root=args.cache_root, records=selected)

    model, checkpoint_metadata = build_and_load_ema(
        checkpoint_path=args.checkpoint,
        contract_path=args.contract,
        device=device,
    )
    expected_cache_hashes = {
        "hubert_40hz": dataset_metadata["hubert_completion_sha256"],
        "latents": dataset_metadata["latent_completion_sha256"],
        "normalization": dataset_metadata["normalization_sha256"],
    }
    actual_cache_hashes = checkpoint_metadata["contract"].get("cache_completion", {})
    if actual_cache_hashes != expected_cache_hashes:
        raise RuntimeError(
            "Inpainting cache does not match checkpoint training contract: "
            f"checkpoint={actual_cache_hashes}, evaluation={expected_cache_hashes}"
        )
    odeint_kwargs = dict(model.odeint_kwargs)
    if odeint_kwargs != {"method": "euler"}:
        raise RuntimeError(f"Expected the pinned Euler ODE solver, got {odeint_kwargs}")

    attempt_dir = Path(tempfile.mkdtemp(prefix=f".{args.label}.", suffix=".tmp", dir=output_root))
    try:
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, record in enumerate(selected, start=1):
                utterance_key = record["utterance_key"]
                frames = int(record["latent_frames"])
                index_entry = latent_index[utterance_key]
                normalized = load_normalized_latent(
                    cache_root=args.cache_root,
                    record=record,
                    index_entry=index_entry,
                    mean=mean,
                    std=std,
                )
                mask_start, mask_end = fixed_inpainting_span(
                    utterance_key=utterance_key,
                    frames=frames,
                    mask_fraction=args.mask_fraction,
                    eval_seed=args.eval_seed,
                )
                keep_mask = build_keep_mask(
                    frames=frames,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    device=device,
                )
                condition = torch.from_numpy(normalized).unsqueeze(0).to(device)
                length = torch.tensor([frames], dtype=torch.long, device=device)
                ode_seed = stable_seed(args.eval_seed, "inpainting_ode_noise", utterance_key)
                generated, _ = model.sample(
                    cond=condition,
                    duration=length,
                    lens=length,
                    steps=args.steps,
                    cfg_strength=args.cfg_strength,
                    seed=ode_seed,
                    max_duration=args.max_frames,
                    use_epss=True,
                    edit_mask=keep_mask,
                )
                generated = generated.squeeze(0).float().cpu()
                if generated.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or not torch.isfinite(generated).all():
                    raise RuntimeError(f"Invalid generated latent for {utterance_key}: {tuple(generated.shape)}")
                target = torch.from_numpy(normalized)
                keep_cpu = keep_mask.squeeze(0).cpu()
                if not torch.equal(generated[keep_cpu], target[keep_cpu]):
                    raise RuntimeError(f"Conditioned latent frames changed for {utterance_key}")

                relative_path = Path("latents") / f"{utterance_key}.npy"
                output_path = safe_join(attempt_dir, relative_path.as_posix())
                generated_array = generated.numpy().astype(np.float32, copy=False)
                atomic_save_npy(output_path, generated_array)
                row = {
                    "audio_relative_path": record["audio_relative_path"],
                    "generated_latent_relative_path": relative_path.as_posix(),
                    "generated_latent_sha256": sha256_file(output_path),
                    "latent_frames": frames,
                    "mask_end": mask_end,
                    "mask_fraction_realized": (mask_end - mask_start) / frames,
                    "mask_fraction_requested": args.mask_fraction,
                    "mask_start": mask_start,
                    "ode_seed": ode_seed,
                    "original_num_samples": int(record["original_num_samples"]),
                    "padded_num_samples": int(record["padded_num_samples"]),
                    "speaker_id": str(record["speaker_id"]),
                    "subset": record["subset"],
                    "target_latent_relative_path": record["latent_relative_path"],
                    "target_latent_sha256": index_entry["sha256"],
                    "target_latent_size_bytes": index_entry["size_bytes"],
                    "text": record["text"],
                    "utterance_key": utterance_key,
                }
                rows.append(row)
                print(f"{args.label}: generated {index}/{len(selected)} {utterance_key}", flush=True)

        manifest_path = attempt_dir / "generation_manifest.jsonl"
        atomic_write_jsonl(manifest_path, rows)
        validation = checkpoint_metadata["validation"]
        complete = {
            "checkpoint": {
                **validation,
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint),
                "weights": "ema",
            },
            "dataset": {
                **dataset_metadata,
                "cache_root": str(args.cache_root.resolve()),
                "manifest": str(args.manifest.resolve()),
                "normalization": str(args.normalization.resolve()),
            },
            "generation_manifest": {
                "count": len(rows),
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
            },
            "label": args.label,
            "protocol": {
                "cfg_strength": args.cfg_strength,
                "cfg_strength_semantics": "pred + (pred - null_pred) * cfg_strength",
                "conventional_guidance_scale": 1.0 + args.cfg_strength,
                "duration_seconds": [args.min_duration, args.max_duration],
                "eval_seed": args.eval_seed,
                "limit_per_subset": args.limit_per_subset,
                "mask_fraction": args.mask_fraction,
                "max_frames": args.max_frames,
                "name": PROTOCOL_NAME,
                "odeint_kwargs": odeint_kwargs,
                "steps": args.steps,
                "use_epss": True,
            },
            "runtime": {
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "elapsed_seconds": time.time() - started_at,
                "model_dtype": str(next(model.parameters()).dtype),
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "schema_version": SCHEMA_VERSION,
            "selected_fixture_verified": canonical_fixture_verified,
            "selected_keys_sha256": selected_keys_sha256,
            "source_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(attempt_dir / "generation_complete.json", complete)
        os.replace(attempt_dir, final_dir)
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return complete
    finally:
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)


def parse_args() -> argparse.Namespace:
    root_prefix = os.environ.get("ROOT_PREFIX", "")
    default_cache = Path(f"{root_prefix}/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--cache-root", type=Path, default=default_cache)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-seed", type=int, default=666)
    parser.add_argument("--limit-per-subset", type=int, default=25)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--mask-fraction", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=4096)
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label) is None:
        parser.error("--label must contain only letters, digits, period, underscore, and hyphen")
    if args.contract is None:
        args.contract = args.checkpoint.parent / "training_contract.json"
    if args.manifest is None:
        args.manifest = args.cache_root / "manifests/dev.jsonl"
    if args.normalization is None:
        args.normalization = args.cache_root / "state/latents/train_normalization.json"
    if args.steps <= 0 or args.max_frames <= 0:
        parser.error("--steps and --max-frames must be positive")
    if args.cfg_strength <= 1e-5:
        parser.error("--cfg-strength must be greater than 1e-5 for the pinned CFM_notext sampler")
    if args.limit_per_subset <= 0:
        parser.error("--limit-per-subset must be positive")
    if not 0 < args.mask_fraction < 1:
        parser.error("--mask-fraction must satisfy 0 < value < 1")
    if not 0 < args.min_duration <= args.max_duration:
        parser.error("duration bounds must satisfy 0 < min <= max")
    return args


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
