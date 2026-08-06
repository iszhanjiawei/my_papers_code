"""Deterministic held-out evaluation for Semantic-VAE warm-start checkpoints.

The training objective samples its mask, diffusion time, Gaussian source, and
conditioning-drop decision inside ``CFM_notext.forward``.  Reusing that method
for checkpoint comparison would make the result depend on global RNG state and
batch composition.  This evaluator constructs every stochastic input from an
utterance-keyed SHA256 seed and calls the transformer directly, so every EMA
checkpoint sees exactly the same LibriSpeech development examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from aligndit.model.cfm_notext import CFM_notext
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.script.misc.svae_cache_utils import (
    HUBERT_HIDDEN_DIM,
    SEMANTIC_VAE_LATENT_DIM,
    read_jsonl,
    safe_join,
    sha256_file,
)
from aligndit.script.misc.validate_semantic_vae_warmstart_checkpoint import (
    validate_completed_stage_checkpoint,
)


PROJECT_ROOT = Path(str(files("aligndit").joinpath("../.."))).resolve()
PROTOCOL_NAME = "librispeech-svae40-dev-keyed-rng-v1"
HUBERT_40HZ_FEATURE = "hubert_large_ll60k_last_hidden_40hz_linear_v1"
SOURCE_PATHS = {
    "cfm_notext": PROJECT_ROOT / "src/aligndit/model/cfm_notext.py",
    "dit_notext": PROJECT_ROOT / "src/aligndit/model/backbone/dit_notext.py",
    "modules": PROJECT_ROOT / "src/aligndit/model/modules.py",
    "f5_cfm": PROJECT_ROOT / "src/f5_tts/model/cfm.py",
    "f5_dit": PROJECT_ROOT / "src/f5_tts/model/backbones/dit.py",
    "f5_modules": PROJECT_ROOT / "src/f5_tts/model/modules.py",
}


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def stable_seed(eval_seed: int, purpose: str, utterance_key: str, repeat: int = 0) -> int:
    payload = f"{PROTOCOL_NAME}:{eval_seed}:{purpose}:{repeat}:{utterance_key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def stable_uniform(eval_seed: int, purpose: str, utterance_key: str, repeat: int = 0) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(eval_seed, purpose, utterance_key, repeat))
    return float(torch.rand((), generator=generator, dtype=torch.float32).item())


def deterministic_draw(
    *,
    utterance_key: str,
    frames: int,
    channels: int,
    eval_seed: int,
    repeat: int,
    mask_fraction_min: float,
    mask_fraction_max: float,
) -> dict[str, Any]:
    """Construct one training-distribution draw without using global RNG state."""

    if frames <= 0 or channels <= 0:
        raise ValueError(f"frames/channels must be positive, got {frames}/{channels}")
    if not 0 < mask_fraction_min <= mask_fraction_max <= 1:
        raise ValueError("Mask fraction bounds must satisfy 0 < min <= max <= 1")
    fraction_u = stable_uniform(eval_seed, "mask_fraction", utterance_key, repeat)
    mask_fraction = mask_fraction_min + (mask_fraction_max - mask_fraction_min) * fraction_u
    masked_frames = max(1, int(mask_fraction * frames))
    max_start = frames - masked_frames
    start_u = stable_uniform(eval_seed, "mask_start", utterance_key, repeat)
    mask_start = max(0, int(max_start * start_u))
    mask_end = mask_start + masked_frames
    diffusion_time = stable_uniform(eval_seed, "diffusion_time", utterance_key, repeat)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(eval_seed, "x0_noise", utterance_key, repeat))
    x0 = torch.randn((frames, channels), generator=generator, dtype=torch.float32)
    return {
        "diffusion_time": diffusion_time,
        "mask_end": mask_end,
        "mask_fraction_sampled": mask_fraction,
        "mask_start": mask_start,
        "masked_frames": masked_frames,
        "x0": x0,
    }


def select_records(
    records: list[dict[str, Any]], *, eval_seed: int, limit_per_subset: int | None
) -> list[dict[str, Any]]:
    if limit_per_subset is None:
        return records
    if limit_per_subset <= 0:
        raise ValueError("limit_per_subset must be positive")
    selected_keys: set[str] = set()
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subset[record["subset"]].append(record)
    for subset in ("dev-clean", "dev-other"):
        candidates = by_subset[subset]
        candidates.sort(
            key=lambda record: hashlib.sha256(
                f"{PROTOCOL_NAME}:{eval_seed}:selection:{record['utterance_key']}".encode()
            ).digest()
        )
        selected_keys.update(record["utterance_key"] for record in candidates[:limit_per_subset])
    return [record for record in records if record["utterance_key"] in selected_keys]


def make_batch_plan(
    records: list[dict[str, Any]], *, padded_frame_budget: int, max_samples: int
) -> list[list[dict[str, Any]]]:
    if padded_frame_budget <= 0 or max_samples <= 0:
        raise ValueError("padded_frame_budget and max_samples must be positive")
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    max_frames = 0
    for record in records:
        frames = int(record["latent_frames"])
        if frames > padded_frame_budget:
            raise ValueError(f"One utterance has {frames} frames, exceeding padded_frame_budget={padded_frame_budget}")
        prospective_max = max(max_frames, frames)
        prospective_size = len(batch) + 1
        if batch and (prospective_size > max_samples or prospective_max * prospective_size > padded_frame_budget):
            batches.append(batch)
            batch = []
            max_frames = 0
        batch.append(record)
        max_frames = max(max_frames, frames)
    if batch:
        batches.append(batch)
    return batches


def validate_and_load_dev(
    *, cache_root: Path, manifest_path: Path, normalization_path: Path, eval_seed: int, limit_per_subset: int | None
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    cache_root = cache_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    normalization_path = normalization_path.resolve(strict=True)
    expected_manifest_path = safe_join(cache_root, "manifests/dev.jsonl")
    expected_normalization_path = safe_join(cache_root, "state/latents/train_normalization.json")
    if manifest_path != expected_manifest_path:
        raise ValueError(f"Development manifest must belong to cache_root: {manifest_path} != {expected_manifest_path}")
    if normalization_path != expected_normalization_path:
        raise ValueError(
            f"Normalization must belong to cache_root: {normalization_path} != {expected_normalization_path}"
        )
    inventory_meta = read_json_object(manifest_path.parent / "inventory_meta.json")
    manifest_entry = inventory_meta.get("manifests", {}).get(manifest_path.name)
    if not isinstance(manifest_entry, dict):
        raise TypeError(f"Manifest is not registered in inventory metadata: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_entry.get("sha256") != manifest_sha256:
        raise RuntimeError("Development manifest differs from immutable inventory metadata")

    inventory_entry = inventory_meta.get("manifests", {}).get("inventory.jsonl")
    if not isinstance(inventory_entry, dict):
        raise TypeError("Inventory metadata does not bind the full inventory manifest")
    latent_completion_path = safe_join(cache_root, "state/latents/complete.json")
    hubert_completion_path = safe_join(cache_root, "state/hubert_40hz/complete.json")
    latent_completion = read_json_object(latent_completion_path)
    hubert_completion = read_json_object(hubert_completion_path)
    expected_completion = {
        "count": int(inventory_entry["count"]),
        "manifest_sha256": inventory_entry["sha256"],
        "selection": {"mode": "full"},
    }
    for name, completion, feature in (
        ("Semantic-VAE", latent_completion, "semantic_vae_posterior_sample_v1"),
        ("HuBERT", hubert_completion, HUBERT_40HZ_FEATURE),
    ):
        if completion.get("cache_schema_version") != 1 or completion.get("feature") != feature:
            raise RuntimeError(f"Invalid {name} cache completion protocol")
        for key, expected in expected_completion.items():
            if completion.get(key) != expected:
                raise RuntimeError(
                    f"{name} cache completion mismatch for {key}: {completion.get(key)!r} != {expected!r}"
                )

    records = list(read_jsonl(manifest_path))
    if len(records) != int(manifest_entry["count"]):
        raise RuntimeError("Development manifest count differs from immutable inventory metadata")
    seen: set[str] = set()
    subset_counts: dict[str, int] = defaultdict(int)
    for record in records:
        key = record.get("utterance_key")
        subset = record.get("subset")
        frames = record.get("latent_frames")
        if record.get("split") != "dev" or subset not in {"dev-clean", "dev-other"}:
            raise ValueError(f"Invalid development record split/subset: {key}")
        if not isinstance(key, str) or key in seen:
            raise ValueError(f"Invalid or duplicate development key: {key!r}")
        if not isinstance(frames, int) or frames <= 0 or record.get("latent_dim") != SEMANTIC_VAE_LATENT_DIM:
            raise ValueError(f"Invalid latent metadata for development key: {key}")
        seen.add(key)
        subset_counts[subset] += 1

    normalization = read_json_object(normalization_path)
    if (
        normalization.get("scope") != "train"
        or normalization.get("channel_count") != SEMANTIC_VAE_LATENT_DIM
        or normalization.get("feature") != "semantic_vae_posterior_sample_v1"
        or normalization.get("latent_complete_sha256") != sha256_file(latent_completion_path)
    ):
        raise ValueError("Evaluation must use the authoritative train-only Semantic-VAE normalization")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if (
        mean.shape != (SEMANTIC_VAE_LATENT_DIM,)
        or std.shape != (SEMANTIC_VAE_LATENT_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid train-only normalization statistics")

    selected = select_records(records, eval_seed=eval_seed, limit_per_subset=limit_per_subset)
    selected_counts = {subset: sum(record["subset"] == subset for record in selected) for subset in subset_counts}
    if any(count == 0 for count in selected_counts.values()):
        raise RuntimeError(f"Evaluation selection omitted a development subset: {selected_counts}")
    metadata = {
        "hubert_completion_sha256": sha256_file(hubert_completion_path),
        "latent_completion_sha256": sha256_file(latent_completion_path),
        "manifest_count": len(records),
        "manifest_sha256": manifest_sha256,
        "normalization_sha256": sha256_file(normalization_path),
        "selected_count": len(selected),
        "selected_counts": selected_counts,
        "selected_keys_sha256": hashlib.sha256(
            "\n".join(record["utterance_key"] for record in selected).encode()
        ).hexdigest(),
        "subset_counts": dict(subset_counts),
    }
    return selected, mean, std, metadata


def verify_contract_sources(contract: dict[str, Any]) -> dict[str, str]:
    recorded = contract.get("source_sha256")
    if not isinstance(recorded, dict):
        raise TypeError("Training contract has no source_sha256 mapping")
    actual: dict[str, str] = {}
    for name, path in SOURCE_PATHS.items():
        expected = recorded.get(name)
        if not isinstance(expected, str):
            raise TypeError(f"Training contract does not bind required source {name!r}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"Source drift for {name}: checkpoint={expected}, current={digest}")
        actual[name] = digest
    return actual


def build_and_load_ema(
    *, checkpoint_path: Path, contract_path: Path, device: torch.device
) -> tuple[CFM_notext, dict[str, Any]]:
    contract = read_json_object(contract_path)
    verify_contract_sources(contract)
    stage = contract.get("policy", {}).get("stage")
    horizon = contract.get("config", {}).get("optim", {}).get("max_updates")
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint_header, dict):
        raise TypeError("Checkpoint root must be a dict")
    update = checkpoint_header.get("update")
    if isinstance(update, torch.Tensor):
        update = update.item()
    if not isinstance(update, int) or isinstance(update, bool):
        raise TypeError(f"Invalid checkpoint update: {update!r}")
    validation = validate_completed_stage_checkpoint(
        checkpoint_path,
        contract_path,
        expected_stage=stage,
        expected_update=update,
        expected_horizon=horizon,
    )

    config = contract["config"]
    model_config = config["model"]
    representation = model_config["audio_representation"]
    channels = int(representation["channels"])
    model_cls = hydra.utils.get_class(f"aligndit.model.{model_config['backbone']}")
    model = CFM_notext(
        transformer=model_cls(**model_config["arch"], mel_dim=channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=int(representation["sample_rate"]),
            hop_length=int(representation["hop_length"]),
        ),
        num_channels=channels,
        proj_lambda=0.0,
    )
    ema = checkpoint_header["ema_model_state_dict"]
    ema_model_state = {
        key.removeprefix("ema_model."): value for key, value in ema.items() if key not in {"initted", "step"}
    }
    model.load_state_dict(ema_model_state, strict=True)
    del checkpoint_header, ema, ema_model_state
    model.eval().requires_grad_(False).to(device)
    layer_indices = list(model_config["arch"].get("layer_indices", []))
    if layer_indices != [12] or model.transformer.layer_map != {12: 0}:
        raise RuntimeError(f"Expected one projector after zero-based block 12, got {layer_indices}")
    return model, {"contract": contract, "validation": validation}


def load_batch(
    records: list[dict[str, Any]], *, cache_root: Path, mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(record["latent_frames"]) for record in records], dtype=torch.long)
    max_frames = int(lengths.max())
    latents = torch.zeros((len(records), max_frames, SEMANTIC_VAE_LATENT_DIM), dtype=torch.float32)
    features = torch.zeros((len(records), max_frames, HUBERT_HIDDEN_DIM), dtype=torch.float32)
    for index, record in enumerate(records):
        frames = int(record["latent_frames"])
        latent_path = safe_join(cache_root, record["latent_relative_path"])
        feature_path = safe_join(cache_root, record["hubert_relative_path"])
        latent = np.load(latent_path, allow_pickle=False)
        feature = np.load(feature_path, allow_pickle=False)
        if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(f"Invalid latent array for {record['utterance_key']}: {latent.shape}/{latent.dtype}")
        if feature.shape != (frames, HUBERT_HIDDEN_DIM) or feature.dtype != np.float32:
            raise ValueError(f"Invalid HuBERT array for {record['utterance_key']}: {feature.shape}/{feature.dtype}")
        normalized = (latent - mean) / std
        if not np.isfinite(normalized).all() or not np.isfinite(feature).all():
            raise FloatingPointError(f"Non-finite cached features for {record['utterance_key']}")
        latents[index, :frames] = torch.from_numpy(normalized)
        features[index, :frames] = torch.from_numpy(feature.copy())
    return latents, features, lengths


def summarize_rows(rows: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty evaluation rows")
    results: dict[str, Any] = {}
    groups = {
        "overall": rows,
        "dev-clean": [row for row in rows if row["subset"] == "dev-clean"],
        "dev-other": [row for row in rows if row["subset"] == "dev-other"],
    }
    for group_index, (name, group_rows) in enumerate(groups.items()):
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_key[row["utterance_key"]].append(row)
        utterance_flow = np.asarray(
            [np.mean([draw["flow_mse"] for draw in draws]) for draws in by_key.values()], dtype=np.float64
        )
        utterance_cosine = np.asarray(
            [np.mean([draw["hubert_cosine"] for draw in draws]) for draws in by_key.values()], dtype=np.float64
        )
        flow_sum = math.fsum(row["flow_squared_error_sum"] for row in group_rows)
        flow_count = sum(row["flow_element_count"] for row in group_rows)
        cosine_sum = math.fsum(row["hubert_cosine_sum"] for row in group_rows)
        cosine_count = sum(row["hubert_frame_count"] for row in group_rows)

        def confidence_interval(values: np.ndarray, metric_seed: int) -> list[float] | None:
            if bootstrap_samples <= 0 or len(values) < 2:
                return None
            generator = np.random.default_rng(metric_seed)
            means = np.empty(bootstrap_samples, dtype=np.float64)
            for index in range(bootstrap_samples):
                means[index] = generator.choice(values, size=len(values), replace=True).mean()
            return [float(value) for value in np.quantile(means, [0.025, 0.975])]

        flow_micro = flow_sum / flow_count
        results[name] = {
            "draws": len(group_rows),
            "flow_element_count": flow_count,
            "flow_mse_macro": float(utterance_flow.mean()),
            "flow_mse_macro_ci95": confidence_interval(utterance_flow, bootstrap_seed + group_index * 2),
            "flow_mse_micro": flow_micro,
            "flow_rmse_micro": math.sqrt(flow_micro),
            "hubert_cosine_macro": float(utterance_cosine.mean()),
            "hubert_cosine_macro_ci95": confidence_interval(utterance_cosine, bootstrap_seed + group_index * 2 + 1),
            "hubert_cosine_micro": cosine_sum / cosine_count,
            "hubert_frame_count": cosine_count,
            "utterances": len(by_key),
        }
    return results


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def configure_runtime(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    configure_runtime(device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.label}.summary.json"
    rows_path = output_dir / f"{args.label}.per_utterance.jsonl"
    if not args.overwrite and (summary_path.exists() or rows_path.exists()):
        raise FileExistsError(f"Evaluation output already exists for label {args.label!r} in {output_dir}")

    records, mean, std, dataset_metadata = validate_and_load_dev(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        normalization_path=args.normalization,
        eval_seed=args.eval_seed,
        limit_per_subset=args.limit_per_subset,
    )
    batches = make_batch_plan(records, padded_frame_budget=args.padded_frame_budget, max_samples=args.max_samples)
    started_at = time.time()
    model, checkpoint_metadata = build_and_load_ema(
        checkpoint_path=args.checkpoint, contract_path=args.contract, device=device
    )
    cache_contract = checkpoint_metadata["contract"].get("cache_completion", {})
    expected_cache_hashes = {
        "hubert_40hz": dataset_metadata["hubert_completion_sha256"],
        "latents": dataset_metadata["latent_completion_sha256"],
        "normalization": dataset_metadata["normalization_sha256"],
    }
    if cache_contract != expected_cache_hashes:
        raise RuntimeError(
            f"Evaluation cache does not match the checkpoint training contract: "
            f"checkpoint={cache_contract}, evaluation={expected_cache_hashes}"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    rows: list[dict[str, Any]] = []
    autocast_context = (
        (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        if args.precision == "bf16"
        else nullcontext
    )

    with torch.inference_mode():
        for batch_index, batch_records in enumerate(batches, start=1):
            latent_cpu, feature_cpu, lengths_cpu = load_batch(
                batch_records, cache_root=args.cache_root, mean=mean, std=std
            )
            valid_cpu = torch.arange(latent_cpu.shape[1])[None, :] < lengths_cpu[:, None]
            for repeat in range(args.repeats):
                x0_cpu = torch.zeros_like(latent_cpu)
                target_cpu = torch.zeros_like(valid_cpu)
                times_cpu = torch.empty((len(batch_records),), dtype=torch.float32)
                draws: list[dict[str, Any]] = []
                for index, record in enumerate(batch_records):
                    frames = int(record["latent_frames"])
                    draw = deterministic_draw(
                        utterance_key=record["utterance_key"],
                        frames=frames,
                        channels=SEMANTIC_VAE_LATENT_DIM,
                        eval_seed=args.eval_seed,
                        repeat=repeat,
                        mask_fraction_min=args.mask_fraction_min,
                        mask_fraction_max=args.mask_fraction_max,
                    )
                    x0_cpu[index, :frames] = draw["x0"]
                    target_cpu[index, draw["mask_start"] : draw["mask_end"]] = True
                    times_cpu[index] = draw["diffusion_time"]
                    draws.append(draw)

                latent = latent_cpu.to(device, non_blocking=True)
                feature = feature_cpu.to(device, non_blocking=True)
                valid = valid_cpu.to(device, non_blocking=True)
                target = target_cpu.to(device, non_blocking=True) & valid
                x0 = x0_cpu.to(device, non_blocking=True)
                times = times_cpu.to(device, non_blocking=True)
                xt = (1 - times[:, None, None]) * x0 + times[:, None, None] * latent
                flow = latent - x0
                condition = torch.where(target[..., None], torch.zeros_like(latent), latent)
                with autocast_context():
                    prediction, intermediates = model.transformer(
                        x=xt,
                        cond=condition,
                        time=times,
                        mask=valid,
                        drop_audio_cond=False,
                        cfg_infer=False,
                        cache=False,
                    )
                projected = intermediates[12]["z_tilde"].float()
                projected_lengths = intermediates[12]["z_lens"].cpu()
                if not torch.equal(projected_lengths, lengths_cpu):
                    raise RuntimeError(
                        f"Projector length mismatch in batch {batch_index}: {projected_lengths} != {lengths_cpu}"
                    )
                prediction = prediction.float()
                cosine = F.cosine_similarity(projected, feature.float(), dim=-1)
                if not torch.isfinite(prediction).all() or not torch.isfinite(cosine).all():
                    raise FloatingPointError(f"Non-finite model output in batch {batch_index}")

                for index, (record, draw) in enumerate(zip(batch_records, draws)):
                    frames = int(record["latent_frames"])
                    sample_target = target[index, :frames]
                    squared = (prediction[index, :frames] - flow[index, :frames]).square()[sample_target]
                    sample_cosine = cosine[index, :frames]
                    flow_sum = float(squared.double().sum().item())
                    flow_count = int(squared.numel())
                    cosine_sum = float(sample_cosine.double().sum().item())
                    row = {
                        "diffusion_time": draw["diffusion_time"],
                        "flow_element_count": flow_count,
                        "flow_mse": flow_sum / flow_count,
                        "flow_squared_error_sum": flow_sum,
                        "frames": frames,
                        "hubert_cosine": cosine_sum / frames,
                        "hubert_cosine_sum": cosine_sum,
                        "hubert_frame_count": frames,
                        "mask_end": draw["mask_end"],
                        "mask_fraction_realized": draw["masked_frames"] / frames,
                        "mask_fraction_sampled": draw["mask_fraction_sampled"],
                        "mask_start": draw["mask_start"],
                        "masked_frames": draw["masked_frames"],
                        "repeat": repeat,
                        "speaker_id": record["speaker_id"],
                        "subset": record["subset"],
                        "utterance_key": record["utterance_key"],
                    }
                    rows.append(row)
            print(
                f"{args.label}: batch {batch_index}/{len(batches)}; rows={len(rows)}/{len(records) * args.repeats}",
                flush=True,
            )

    results = summarize_rows(rows, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    elapsed = time.time() - started_at
    validation = checkpoint_metadata["validation"]
    summary = {
        "checkpoint": {
            **validation,
            "path": str(args.checkpoint.resolve()),
            "projector_trained": validation["stage"] == "s2c",
            "sha256": checkpoint_sha256,
            "weights": "ema",
        },
        "dataset": {
            **dataset_metadata,
            "cache_root": str(args.cache_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "normalization": str(args.normalization.resolve()),
        },
        "label": args.label,
        "protocol": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "condition_mode": "masked_audio",
            "eval_seed": args.eval_seed,
            "mask_fraction": [args.mask_fraction_min, args.mask_fraction_max],
            "name": PROTOCOL_NAME,
            "padded_frame_budget": args.padded_frame_budget,
            "repeats": args.repeats,
        },
        "results": results,
        "runtime": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "elapsed_seconds": elapsed,
            "numpy": np.__version__,
            "precision": args.precision,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "schema_version": 1,
    }
    atomic_write_jsonl(rows_path, rows)
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


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
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--eval-seed", type=int, default=666)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mask-fraction-min", type=float, default=0.7)
    parser.add_argument("--mask-fraction-max", type=float, default=1.0)
    parser.add_argument("--limit-per-subset", type=int)
    parser.add_argument("--padded-frame-budget", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label) is None:
        parser.error("--label must contain only letters, digits, period, underscore, and hyphen")
    if args.contract is None:
        args.contract = args.checkpoint.parent / "training_contract.json"
    if args.manifest is None:
        args.manifest = args.cache_root / "manifests/dev.jsonl"
    if args.normalization is None:
        args.normalization = args.cache_root / "state/latents/train_normalization.json"
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
