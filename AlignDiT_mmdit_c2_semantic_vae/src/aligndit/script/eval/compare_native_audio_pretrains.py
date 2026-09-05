"""Compare native mel500k and Semantic-VAE S2c70k audio-only inpainting.

Run prepare in AlignDiT, encode-svae-context in the isolated Semantic-VAE
environment, then mel/svae in AlignDiT and decode-svae in Semantic-VAE.
Each mode publishes a new immutable
directory. Physical masks and utterances are shared, but native representation
errors are deliberately not compared across codecs. No text, video or HuBERT
feature is passed to either sampler. A --limit canary has separate outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchaudio

from aligndit.script.misc.svae_cache_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    safe_join,
    sha256_file,
)


PROTOCOL = "librispeech-native-mel500k-svae70k-waveform-masked-inpainting-v2"
SAMPLE_RATE = 16000
GRID_SAMPLES = 800  # LCM(160,400): exactly five mel frames and two VAE frames.
SEEDS = (666, 667, 668)
MEL_SHA256 = "4a9fc0e526ce47745aee839348406ca99597d32f5ed028bda42a3de3ec900fcd"
SVAE_SHA256 = "02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4"
NORM_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
COMMON_COUNT = 50


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def checked_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def keyed_seed(seed: int, purpose: str, key: str) -> int:
    payload = f"{PROTOCOL}:{seed}:{purpose}:{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def physical_span(num_samples: int, key: str) -> tuple[int, int]:
    """Quantize once in waveform time; never round independent branch masks."""
    units = num_samples // GRID_SAMPLES
    hidden = math.floor(0.7 * num_samples / GRID_SAMPLES)
    if not 0 < hidden < units:
        raise ValueError(f"Insufficient waveform length: {num_samples}")
    generator = torch.Generator().manual_seed(keyed_seed(666, "physical-mask", key))
    start = int(torch.randint(units - hidden + 1, (), generator=generator).item())
    return start * GRID_SAMPLES, (start + hidden) * GRID_SAMPLES


def protocol_contract() -> dict[str, Any]:
    return {
        "name": PROTOCOL,
        "sample_rate": SAMPLE_RATE,
        "selection_seed": 666,
        "sampling_seeds": list(SEEDS),
        "utterances_per_subset": 25,
        "duration_seconds": [4.0, 10.0],
        "mask_grid_samples": GRID_SAMPLES,
        "mask_fraction_requested": 0.7,
        "mask_rounding": "floor(0.7 * original_samples / 800), fixed for all seeds",
        "conditioning": "same source waveform zeroed inside physical mask BEFORE native encoding; no text/video/HuBERT",
        "context_latent": "fixed keyed posterior sample of zero-masked waveform; clean cached latents are oracle only",
        "steps": 32,
        "ode_solver": "euler",
        "use_epss": True,
        "cfg_strength": 1.0,
        "conventional_guidance_scale": 2.0,
        "batch_size": 1,
        "dtype": "float32",
        "tf32": False,
        "waveform_encoding": "32-bit IEEE float PCM; no clipping or gain normalization",
        "waveform_known_context": "native representation copied by sampler; waveform is full native decoder output",
        "boundary_caveat": "masked waveform removes hidden-content leakage; unequal codec receptive fields remain",
        "paired_seed_caveat": "same keyed seed, different native shapes; Gaussian tensors are not identical",
        "scope": "audio-only completion capability, not text-conditioned intelligibility or video dubbing",
    }


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if device.type == "cuda":
        # Initialize the selected CUDA device before querying allocator counters.
        # A fresh process has not necessarily created a CUDA context yet.
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)


def runtime_metadata(device: torch.device, started: float) -> dict[str, Any]:
    return {
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "source_sha256": sha256_file(Path(__file__)),
    }


def create_attempt(root: Path, name: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Refusing to overwrite completed/partial output: {final}")
    return Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root)), final


def artifact(path: Path, relative: Path) -> dict[str, Any]:
    return {"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def validate_artifact(root: Path, info: dict[str, Any]) -> Path:
    path = safe_join(root, info["path"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size != info["size_bytes"]:
        raise ValueError(f"Invalid bound artifact: {path}")
    checked_hash(path, info["sha256"])
    return path


def save_wave(path: Path, waveform: torch.Tensor) -> None:
    waveform = waveform.detach().cpu().float().reshape(1, -1)
    if path.exists() or waveform.shape[1] == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid/new waveform output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, SAMPLE_RATE, encoding="PCM_F", bits_per_sample=32)
    loaded, rate = torchaudio.load(str(path))
    if rate != SAMPLE_RATE or not torch.equal(loaded, waveform):
        raise RuntimeError(f"Float WAV failed exact roundtrip: {path}")


def save_array(path: Path, array: np.ndarray) -> None:
    if path.exists() or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid/new float32 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def load_source(record: dict[str, Any]) -> torch.Tensor:
    path = Path(record["source_audio_path"])
    checked_hash(path, record["source_audio_sha256"])
    waveform, rate = torchaudio.load(str(path))
    if rate != SAMPLE_RATE or waveform.shape != (1, record["original_num_samples"]):
        raise ValueError(f"Source audio shape/rate differs: {path}, {waveform.shape}, {rate}")
    if not torch.isfinite(waveform).all():
        raise FloatingPointError(f"Non-finite source waveform: {path}")
    return waveform


def prepare(args: argparse.Namespace) -> None:
    from aligndit.script.eval.eval_semantic_vae_warmstart_dev import validate_and_load_dev
    from aligndit.script.eval.generate_semantic_vae_warmstart_inpainting import (
        CANONICAL_DEV_MANIFEST_SHA256,
        CANONICAL_SELECTION_KEYS_SHA256,
        load_selected_latent_index,
        select_inpainting_records,
    )

    if args.limit is not None:
        raise ValueError("prepare always freezes the full 50 examples; use --limit only for branch canaries")
    checked_hash(args.cache_root / "manifests/dev.jsonl", CANONICAL_DEV_MANIFEST_SHA256)
    checked_hash(args.cache_root / "state/latents/train_normalization.json", NORM_SHA256)
    records, _, _, dataset_metadata = validate_and_load_dev(
        cache_root=args.cache_root,
        manifest_path=args.cache_root / "manifests/dev.jsonl",
        normalization_path=args.cache_root / "state/latents/train_normalization.json",
        eval_seed=666,
        limit_per_subset=None,
    )
    selected = select_inpainting_records(
        records, eval_seed=666, limit_per_subset=25, min_duration=4.0, max_duration=10.0
    )
    digest = hashlib.sha256("\n".join(row["utterance_key"] for row in selected).encode()).hexdigest()
    if digest != CANONICAL_SELECTION_KEYS_SHA256 or len(selected) != COMMON_COUNT:
        raise RuntimeError("Canonical previous 50-utterance selection changed")
    if len({str(row["speaker_id"]) for row in selected}) != COMMON_COUNT:
        raise RuntimeError("Paired evaluation requires globally distinct speakers")
    latent_index = load_selected_latent_index(cache_root=args.cache_root, records=selected)
    attempt, final = create_attempt(args.output_dir, "common")
    try:
        rows = []
        for source in selected:
            row = dict(source)
            key = row["utterance_key"]
            audio = safe_join(args.audio_root, row["audio_relative_path"])
            row["source_audio_path"] = str(audio)
            row["source_audio_sha256"] = sha256_file(audio)
            row["latent_index_entry"] = latent_index[key]
            wave = load_source(row)
            start, end = physical_span(wave.shape[-1], key)
            row.update(mask_start_sample=start, mask_end_sample=end)
            row["mask_fraction_realized"] = (end - start) / wave.shape[-1]
            references = {
                "full": wave,
                "masked": wave[:, start:end],
                "context": torch.cat((wave[:, :start], wave[:, end:]), dim=-1),
            }
            observed_input = wave.clone()
            observed_input[:, start:end] = 0
            references["input"] = observed_input
            for kind, value in references.items():
                relative = Path("references") / kind / f"{key}.wav"
                path = attempt / relative
                save_wave(path, value)
                row[f"reference_{kind}"] = artifact(path, Path("common") / relative)
            rows.append(row)
        manifest = attempt / "manifest.jsonl"
        atomic_write_jsonl(manifest, rows)
        complete = {
            "schema_version": 1,
            "protocol": protocol_contract(),
            "manifest": artifact(manifest, Path("common/manifest.jsonl")),
            "count": len(rows),
            "selected_keys_sha256": digest,
            "dataset": dataset_metadata,
            "cache_root": str(args.cache_root.resolve()),
            "audio_root": str(args.audio_root.resolve()),
            "latent_spec_sha256": sha256_file(args.cache_root / "state/latents/spec.json"),
            "checkpoints": {"mel": MEL_SHA256, "svae": SVAE_SHA256},
            "source_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(attempt / "complete.json", complete)
        os.rename(attempt, final)
        print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)
    finally:
        if attempt.exists():
            shutil.rmtree(attempt)


def load_common(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = read_object(args.output_dir / "common/complete.json")
    if complete.get("schema_version") != 1 or complete.get("protocol") != protocol_contract():
        raise RuntimeError("Common protocol contract differs from this evaluator")
    if Path(complete["cache_root"]).resolve() != args.cache_root.resolve():
        raise RuntimeError("Different cache root supplied after preparing common fixture")
    checked_hash(args.cache_root / "manifests/dev.jsonl", complete["dataset"]["manifest_sha256"])
    checked_hash(args.cache_root / "state/latents/train_normalization.json", NORM_SHA256)
    checked_hash(args.cache_root / "state/latents/spec.json", complete["latent_spec_sha256"])
    rows = list(read_jsonl(validate_artifact(args.output_dir, complete["manifest"])))
    if len(rows) != COMMON_COUNT or len({row["utterance_key"] for row in rows}) != COMMON_COUNT:
        raise RuntimeError("Common fixture must contain exactly 50 unique utterances")
    digest = hashlib.sha256("\n".join(row["utterance_key"] for row in rows).encode()).hexdigest()
    if digest != complete["selected_keys_sha256"]:
        raise RuntimeError("Common fixture order/selection differs")
    for row in rows:
        span = physical_span(row["original_num_samples"], row["utterance_key"])
        if span != (row["mask_start_sample"], row["mask_end_sample"]):
            raise RuntimeError("Common physical mask differs")
        for kind in ("full", "masked", "context", "input"):
            validate_artifact(args.output_dir, row[f"reference_{kind}"])
    return rows[: args.limit] if args.limit is not None else rows, complete


def branch_name(branch: str, limit: int | None) -> str:
    return branch if limit is None else f"{branch}_canary{limit}"


def load_observed_input(root: Path, record: dict[str, Any]) -> torch.Tensor:
    path = validate_artifact(root, record["reference_input"])
    observed, rate = torchaudio.load(str(path))
    source = load_source(record)
    expected = source.clone()
    expected[:, record["mask_start_sample"] : record["mask_end_sample"]] = 0
    if rate != SAMPLE_RATE or not torch.equal(observed, expected):
        raise RuntimeError("Conditioning waveform must equal source outside mask and exact zeros inside")
    return observed


def encode_svae_context(args: argparse.Namespace) -> None:
    import subprocess

    import torch.nn.functional as F

    from aligndit.script.misc.extract_librispeech_svae_latents import extract_latent, load_posterior_model

    started = time.time()
    device = torch.device(args.device)
    configure_runtime(device)
    if device.type != "cuda":
        raise ValueError("Pinned Semantic-VAE posterior golden self-test requires CUDA")
    records, common = load_common(args)
    name = branch_name("svae_context", args.limit)
    if (args.output_dir / name).exists():
        raise FileExistsError(args.output_dir / name)
    spec = read_object(args.cache_root / "state/latents/spec.json")
    source = spec["semantic_vae_source"]
    repo = args.semantic_vae_repo.resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if commit != source["commit"] or status:
        raise RuntimeError("Semantic-VAE encoder source differs from cached oracle source")
    ckpt_root = args.semantic_vae_checkpoint.resolve(strict=True)
    for relative, field in (
        ("dac/ema_state_dict.pth", "ema"),
        ("metainfo.json", "metainfo"),
        ("config.json", "config"),
    ):
        checked_hash(ckpt_root / relative, spec["checkpoint"][f"{field}_sha256"])
    spec["semantic_vae_source"]["repo"] = str(repo)
    spec["checkpoint"]["ema_path"] = str(ckpt_root / "dac/ema_state_dict.pth")
    # The existing clean-wave golden hash binds the original cache numerics.
    # Reproduce that loader self-test, then restore this comparison's strict
    # FP32/math attention policy before encoding ANY masked-context input.
    numerics = spec["extraction"]["cuda_numerics"]
    torch.use_deterministic_algorithms(numerics["deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = numerics["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = numerics["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = numerics["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = numerics["cudnn_deterministic"]
    torch.backends.cuda.enable_flash_sdp(numerics["flash_sdp"])
    torch.backends.cuda.enable_mem_efficient_sdp(numerics["mem_efficient_sdp"])
    torch.backends.cuda.enable_math_sdp(numerics["math_sdp"])
    model = load_posterior_model(spec, device, args.audio_root.resolve(strict=True))
    configure_runtime(device)
    normalization = read_object(args.cache_root / "state/latents/train_normalization.json")
    mean, std = (np.asarray(normalization[key], dtype=np.float32) for key in ("mean", "std"))
    attempt, final = create_attempt(args.output_dir, name)
    try:
        rows = []
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                key = record["utterance_key"]
                observed = load_observed_input(args.output_dir, record).unsqueeze(0).to(device)
                right_pad = record["padded_num_samples"] - record["original_num_samples"]
                if not 0 <= right_pad < 400:
                    raise RuntimeError("Unexpected Semantic-VAE right-padding geometry")
                observed = F.pad(observed, (0, right_pad))
                seed = keyed_seed(666, "context-posterior", key)
                raw = extract_latent(model, observed, seed, record["latent_frames"], device)
                normalized = ((raw - mean) / std).astype(np.float32)
                relative = Path("latents") / f"{key}.npy"
                save_array(attempt / relative, normalized)
                rows.append(
                    {
                        "utterance_key": key,
                        "reference_input": record["reference_input"],
                        "context_posterior_seed": seed,
                        "context_latent": artifact(attempt / relative, Path(name) / relative),
                    }
                )
                print(f"encode-svae-context: {index}/{len(records)}, {key}", flush=True)
        manifest = attempt / "context_manifest.jsonl"
        atomic_write_jsonl(manifest, rows)
        complete = common_generation_metadata(args, common, manifest, name, len(rows), device, started)
        complete.update(
            context_manifest=complete.pop("generation_manifest"),
            checkpoint_sha256=spec["checkpoint"]["ema_sha256"],
            context_encoding="zero-masked waveform BEFORE encoding; fixed posterior sample normalized with Libri train stats",
            encoder_golden_passed=True,
        )
        atomic_write_json(attempt / "context_complete.json", complete)
        os.rename(attempt, final)
        print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)
    finally:
        if attempt.exists():
            shutil.rmtree(attempt)


def load_context_rows(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    name = branch_name("svae_context", args.limit)
    complete = read_object(args.output_dir / name / "context_complete.json")
    if (
        complete.get("protocol") != protocol_contract()
        or complete.get("common_complete_sha256") != sha256_file(args.output_dir / "common/complete.json")
        or complete.get("canary_limit") != args.limit
    ):
        raise RuntimeError("Semantic-VAE context contract differs from the paired fixture")
    rows = list(read_jsonl(validate_artifact(args.output_dir, complete["context_manifest"])))
    mapped = {row["utterance_key"]: row for row in rows}
    if len(mapped) != len(rows) or len(rows) != (args.limit or COMMON_COUNT):
        raise RuntimeError("Missing/duplicated Semantic-VAE context records")
    return mapped


def load_mel_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from aligndit.model.backbone.dit_notext import DiT_noText
    from aligndit.model.cfm_notext import CFM_notext
    from aligndit.model.modules import PrecomputedAudioRepresentation

    checked_hash(checkpoint, MEL_SHA256)
    source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if int(source.get("update", -1)) != 500000:
        raise RuntimeError("Expected original mel500k checkpoint update")
    ema = source["ema_model_state_dict"]
    if not bool(ema["initted"]) or int(ema["step"]) != 500000:
        raise RuntimeError("Original mel500k EMA counter is invalid")
    arch = {
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "ff_mult": 2,
        "pe_attn_head": 1,
        "attn_mask_enabled": True,
        "checkpoint_activations": False,
        "layer_indices": [12],
        "projector_dim": 2048,
        "z_dim": 1024,
        "qk_norm": None,
        "projector_strides": (2, 1),
        "mel_dim": 80,
    }
    model = CFM_notext(
        transformer=DiT_noText(**arch),
        mel_spec_module=PrecomputedAudioRepresentation(80, SAMPLE_RATE, 160),
        num_channels=80,
        proj_lambda=0.0,
    )
    target = model.state_dict()
    weights = {}
    for key, tensor in ema.items():
        if key in {"initted", "step"}:
            continue
        if not key.startswith("ema_model."):
            raise KeyError(f"Unexpected original EMA entry: {key}")
        name = key.removeprefix("ema_model.")
        if name not in target or tensor.shape != target[name].shape or tensor.dtype != target[name].dtype:
            raise ValueError(f"Original mel EMA schema mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Original mel EMA non-finite tensor: {name}")
        weights[name] = tensor
    if set(weights) != set(target) or len(weights) != 277:
        raise RuntimeError(f"Original mel EMA key set differs: {len(weights)} vs {len(target)}")
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {
        "path": str(checkpoint.resolve()),
        "sha256": MEL_SHA256,
        "weights": "ema",
        "update": 500000,
        "ema_step": 500000,
        "state_keys": len(weights),
        "arch": arch,
    }


def output_pair(
    attempt: Path, name: str, key: str, prefix: str, wave: torch.Tensor, row: dict[str, Any]
) -> dict[str, Any]:
    wave = wave.reshape(1, -1).float().cpu()
    n = row["original_num_samples"]
    if wave.shape[-1] < n or not torch.isfinite(wave).all():
        raise ValueError(f"Invalid decoded waveform/length: {key}, {wave.shape[-1]}, expected >= {n}")
    wave = wave[:, :n]
    start, end = row["mask_start_sample"], row["mask_end_sample"]
    result = {}
    for kind, value in (("full", wave), ("masked", wave[:, start:end])):
        relative = Path("waves") / prefix / kind / f"{key}.wav"
        path = attempt / relative
        save_wave(path, value)
        result[kind] = artifact(path, Path(name) / relative)
    return result


def common_generation_metadata(
    args: argparse.Namespace,
    common: dict[str, Any],
    manifest: Path,
    name: str,
    count: int,
    device: torch.device,
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": common["protocol"],
        "common_complete_sha256": sha256_file(args.output_dir / "common/complete.json"),
        "common_manifest_sha256": common["manifest"]["sha256"],
        "generation_manifest": artifact(manifest, Path(name) / manifest.name),
        "count": count,
        "canary_limit": args.limit,
        "runtime": runtime_metadata(device, started),
    }


def generate(args: argparse.Namespace) -> None:
    started = time.time()
    device = torch.device(args.device)
    configure_runtime(device)
    records, common = load_common(args)
    branch = args.mode
    name = branch_name("svae_latents" if branch == "svae" else branch, args.limit)
    if (args.output_dir / name).exists():
        raise FileExistsError(args.output_dir / name)
    if branch == "mel":
        from aligndit.model.modules import MelSpec_tacotron
        from aligndit.script.eval.utils import load_vocoder

        model, checkpoint = load_mel_model(args.mel_checkpoint, device)
        frontend = MelSpec_tacotron().to(device)
        decoder = load_vocoder("hifigan_16k", is_local=True, local_path=str(args.hifigan_checkpoint), device=device)
        decoder.eval().requires_grad_(False)
        codec = {
            "path": str(args.hifigan_checkpoint.resolve()),
            "sha256": sha256_file(args.hifigan_checkpoint),
            "config_sha256": sha256_file(args.hifigan_checkpoint.parent / "config.json"),
            "name": "native_mel80_100hz_hifigan",
            "frontend": {"n_fft": 640, "win_length": 640, "hop_length": 160, "n_mels": 80},
        }
        channels, hop = 80, 160
    else:
        from aligndit.script.eval.eval_semantic_vae_warmstart_dev import build_and_load_ema

        checked_hash(args.svae_checkpoint, SVAE_SHA256)
        model, metadata = build_and_load_ema(
            checkpoint_path=args.svae_checkpoint,
            contract_path=args.svae_checkpoint.parent / "training_contract.json",
            device=device,
        )
        expected_cache = {
            "hubert_40hz": common["dataset"]["hubert_completion_sha256"],
            "latents": common["dataset"]["latent_completion_sha256"],
            "normalization": common["dataset"]["normalization_sha256"],
        }
        if metadata["contract"]["cache_completion"] != expected_cache:
            raise RuntimeError("S2c training cache differs from prepared evaluation cache")
        checkpoint = {
            "path": str(args.svae_checkpoint.resolve()),
            "sha256": SVAE_SHA256,
            "weights": "ema",
            "validation": metadata["validation"],
            "contract_sha256": sha256_file(args.svae_checkpoint.parent / "training_contract.json"),
        }
        context_rows = load_context_rows(args)
        codec = {"name": "native_svae64_40hz", "normalization_sha256": NORM_SHA256}
        channels, hop = 64, 400
    if dict(model.odeint_kwargs) != {"method": "euler"}:
        raise RuntimeError("Sampler must use the pinned Euler solver")
    attempt, final = create_attempt(args.output_dir, name)
    try:
        rows = []
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                key = record["utterance_key"]
                observed = load_observed_input(args.output_dir, record)
                if branch == "mel":
                    target = frontend(observed.to(device)).unsqueeze(0)
                else:
                    context_info = context_rows[key]
                    if context_info["reference_input"] != record["reference_input"] or context_info[
                        "context_posterior_seed"
                    ] != keyed_seed(666, "context-posterior", key):
                        raise RuntimeError("Semantic-VAE context was not encoded from the common masked waveform")
                    array = np.load(
                        validate_artifact(args.output_dir, context_info["context_latent"]), allow_pickle=False
                    )
                    if (
                        array.shape != (record["latent_frames"], 64)
                        or array.dtype != np.float32
                        or not np.isfinite(array).all()
                    ):
                        raise RuntimeError("Invalid Semantic-VAE observed context array")
                    target = torch.from_numpy(array).unsqueeze(0).to(device)
                if target.ndim != 3 or target.shape[0] != 1 or target.shape[2] != channels:
                    raise RuntimeError(f"Invalid native target shape: {target.shape}")
                frames = target.shape[1]
                start, end = record["mask_start_sample"] // hop, record["mask_end_sample"] // hop
                if not 0 <= start < end <= frames or record["mask_end_sample"] % hop:
                    raise RuntimeError("Physical mask cannot map exactly to native frames")
                keep = torch.ones((1, frames), dtype=torch.bool, device=device)
                keep[:, start:end] = False
                lengths = torch.tensor([frames], dtype=torch.long, device=device)
                oracle = None
                if branch == "mel":
                    clean_target = frontend(load_source(record).to(device)).unsqueeze(0)
                    oracle = output_pair(attempt, name, key, "oracle", decoder(clean_target.transpose(1, 2)), record)
                for seed in SEEDS:
                    ode_seed = keyed_seed(seed, "ode-noise", key)
                    generated, trajectory = model.sample(
                        cond=target,
                        duration=lengths,
                        lens=lengths,
                        steps=32,
                        cfg_strength=1.0,
                        seed=ode_seed,
                        max_duration=4096,
                        use_epss=True,
                        edit_mask=keep,
                    )
                    del trajectory
                    if generated.shape != target.shape or not torch.isfinite(generated).all():
                        raise RuntimeError(f"Invalid native generated tensor: {key}/{seed}")
                    if not torch.equal(generated[keep], target[keep]):
                        raise RuntimeError("Sampler changed observed native representation")
                    row = {
                        **record,
                        "branch": branch,
                        "sampling_seed": seed,
                        "ode_seed": ode_seed,
                        "representation_frames": frames,
                        "hop_length": hop,
                    }
                    if branch == "mel":
                        decoded = decoder(generated.transpose(1, 2))
                        pair = output_pair(attempt, name, key, f"seed{seed}", decoded, record)
                        row.update(
                            generated_full=pair["full"],
                            generated_masked=pair["masked"],
                            oracle_full=oracle["full"],
                            oracle_masked=oracle["masked"],
                        )
                    else:
                        row.update(context_info)
                        relative = Path("latents") / f"seed{seed}" / f"{key}.npy"
                        save_array(attempt / relative, generated[0].float().cpu().numpy())
                        row["generated_latent"] = artifact(attempt / relative, Path(name) / relative)
                    rows.append(row)
                    print(f"{branch}: {index}/{len(records)}, seed={seed}, {key}", flush=True)
        manifest = attempt / "generation_manifest.jsonl"
        atomic_write_jsonl(manifest, rows)
        complete = common_generation_metadata(args, common, manifest, name, len(rows), device, started)
        complete.update(branch=branch, checkpoint=checkpoint, codec=codec, waveform_complete=branch == "mel")
        marker = "latent_generation_complete.json" if branch == "svae" else "generation_complete.json"
        atomic_write_json(attempt / marker, complete)
        os.rename(attempt, final)
        print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)
    finally:
        if attempt.exists():
            shutil.rmtree(attempt)


def decode_svae(args: argparse.Namespace) -> None:
    from aligndit.script.eval.decode_semantic_vae_warmstart_inpainting import (
        inverse_normalize,
        load_semantic_vae_decoder,
    )

    started = time.time()
    device = torch.device(args.device)
    configure_runtime(device)
    records, common = load_common(args)
    by_key = {record["utterance_key"]: record for record in records}
    latent_name = branch_name("svae_latents", args.limit)
    name = branch_name("svae", args.limit)
    generation = read_object(args.output_dir / latent_name / "latent_generation_complete.json")
    if (
        generation.get("protocol") != common["protocol"]
        or generation.get("common_complete_sha256") != sha256_file(args.output_dir / "common/complete.json")
        or generation.get("checkpoint", {}).get("sha256") != SVAE_SHA256
        or generation.get("canary_limit") != args.limit
    ):
        raise RuntimeError("Semantic-VAE latent generation contract differs")
    rows = list(read_jsonl(validate_artifact(args.output_dir, generation["generation_manifest"])))
    expected_pairs = {(key, seed) for key in by_key for seed in SEEDS}
    pairs = [(row["utterance_key"], row["sampling_seed"]) for row in rows]
    if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
        raise RuntimeError("Incomplete/duplicated latent generation selection or seeds")
    checked_hash(args.cache_root / "state/latents/train_normalization.json", NORM_SHA256)
    normalization = read_object(args.cache_root / "state/latents/train_normalization.json")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    context_rows = load_context_rows(args)
    spec = read_object(args.cache_root / "state/latents/spec.json")
    decoder, codec = load_semantic_vae_decoder(
        repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint,
        cache_spec=spec,
        device=device,
    )
    attempt, final = create_attempt(args.output_dir, name)
    try:
        completed_rows = []
        oracle_by_key = {}
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                key, seed = row["utterance_key"], row["sampling_seed"]
                record = by_key[key]
                for field, value in record.items():
                    if row.get(field) != value:
                        raise RuntimeError(f"Latent row differs from common manifest: {key}/{field}")
                if row.get("ode_seed") != keyed_seed(seed, "ode-noise", key):
                    raise RuntimeError("Latent sampling seed differs")
                if row.get("hop_length") != 400 or row.get("representation_frames") != record["latent_frames"]:
                    raise RuntimeError("Latent native geometry differs")
                target_path = safe_join(args.cache_root, record["latent_relative_path"])
                checked_hash(target_path, record["latent_index_entry"]["sha256"])
                target = np.load(target_path, allow_pickle=False)
                generated = np.load(validate_artifact(args.output_dir, row["generated_latent"]), allow_pickle=False)
                shape = (record["latent_frames"], 64)
                for value in (target, generated):
                    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise RuntimeError("Native latent shape/dtype/value mismatch")
                start, end = record["mask_start_sample"] // 400, record["mask_end_sample"] // 400
                keep = np.ones(shape[0], dtype=bool)
                keep[start:end] = False
                context_info = context_rows[key]
                for field, value in context_info.items():
                    if row.get(field) != value:
                        raise RuntimeError(f"Generated latent context binding differs: {key}/{field}")
                observed_context = np.load(
                    validate_artifact(args.output_dir, context_info["context_latent"]), allow_pickle=False
                )
                if not np.array_equal(generated[keep], observed_context[keep]):
                    raise RuntimeError("Observed Semantic-VAE frames differ")
                if key not in oracle_by_key:
                    oracle_wave = decoder(torch.from_numpy(target).unsqueeze(0).transpose(1, 2).to(device))
                    if oracle_wave.shape[-1] != record["padded_num_samples"]:
                        raise RuntimeError("Semantic-VAE oracle decode length differs from exact hop length")
                    oracle_by_key[key] = output_pair(attempt, name, key, "oracle", oracle_wave, record)
                raw = inverse_normalize(generated, mean, std)
                decoded = decoder(torch.from_numpy(raw).unsqueeze(0).transpose(1, 2).to(device))
                if decoded.shape[-1] != record["padded_num_samples"]:
                    raise RuntimeError("Semantic-VAE generated decode length differs from exact hop length")
                pair = output_pair(attempt, name, key, f"seed{seed}", decoded, record)
                oracle = oracle_by_key[key]
                completed_rows.append(
                    {
                        **row,
                        "generated_full": pair["full"],
                        "generated_masked": pair["masked"],
                        "oracle_full": oracle["full"],
                        "oracle_masked": oracle["masked"],
                    }
                )
                print(f"decode-svae: {index}/{len(rows)}, {key}, seed={seed}", flush=True)
        manifest = attempt / "generation_manifest.jsonl"
        atomic_write_jsonl(manifest, completed_rows)
        complete = common_generation_metadata(args, common, manifest, name, len(rows), device, started)
        complete.update(
            branch="svae",
            checkpoint=generation["checkpoint"],
            codec=codec,
            waveform_complete=True,
            latent_generation_complete_sha256=sha256_file(
                args.output_dir / latent_name / "latent_generation_complete.json"
            ),
            generation_runtime=generation["runtime"],
        )
        atomic_write_json(attempt / "generation_complete.json", complete)
        os.rename(attempt, final)
        print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)
    finally:
        if attempt.exists():
            shutil.rmtree(attempt)


def parse_args() -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    user_root = Path(f"{prefix}/zjw524")
    workspace = user_root / "projects/alignDiT_idea6"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "mel", "encode-svae-context", "svae", "decode-svae"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Separate canary output; common fixture always contains 50 records")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=user_root / "projects/data/LibriSpeech_svae1000k_sample_seed666_fp32",
    )
    parser.add_argument("--audio-root", type=Path, default=user_root / "datasets/LibriSpeech")
    parser.add_argument(
        "--mel-checkpoint", type=Path, default=user_root / "datasets/AlignDiT_pretrain_LibriSpeech_500000.pt"
    )
    parser.add_argument(
        "--svae-checkpoint",
        type=Path,
        default=user_root
        / "projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt",
    )
    parser.add_argument(
        "--hifigan-checkpoint", type=Path, default=workspace / "my_papers_code/hifigan_16k_LRS3/g_01000000"
    )
    parser.add_argument("--semantic-vae-repo", type=Path, default=workspace / "papers_codes/Semantic-VAE")
    parser.add_argument(
        "--semantic-vae-checkpoint", type=Path, default=workspace / "Semantic-VAE/Semantic-VAE/semantic_vae_1000k"
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit < COMMON_COUNT:
        parser.error("--limit must be 1..49; omit it for the formal 50-utterance run")
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "encode-svae-context":
        encode_svae_context(args)
    elif args.mode == "decode-svae":
        decode_svae(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
