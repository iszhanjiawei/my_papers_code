"""Cache pinned WavLM-Base+ frame targets for the Direct-C2 REPA run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torchaudio
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_DIM,
    WAVLM_BASE_PLUS_LAYER,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
    WAVLM_FRAME_STRIDE_SAMPLES,
    validate_repa_feature_array,
)


SAMPLE_RATE = 16_000
DEFAULT_MANIFEST_SHA256 = "0d16d5c8f00eb25ee51c7de604299a37cace1bc0e65b7127a45420c433b4d395"


@dataclass(frozen=True)
class ExtractionItem:
    utterance_key: str
    audio_path: Path
    cache_path: Path
    duration_seconds: float


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.save(file, array, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def safe_cache_path(cache_root: Path, audio_relative_path: str) -> Path:
    relative = Path(audio_relative_path)
    if relative.is_absolute() or relative.suffix.lower() != ".wav" or relative.parts[0] != "train":
        raise ValueError(f"invalid train audio relative path: {audio_relative_path!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe train audio relative path: {audio_relative_path!r}")
    result = (cache_root / relative.with_suffix(".npy")).resolve(strict=False)
    result.relative_to(cache_root)
    return result


def read_inventory(args) -> list[ExtractionItem]:
    if sha256_file(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("train manifest SHA256 does not match the pinned Direct-C2 inventory")
    items = []
    seen: set[str] = set()
    with args.manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            relative_path = record.get("audio_relative_path")
            utterance_key = record.get("utterance_key")
            duration = record.get("duration_seconds")
            if not isinstance(relative_path, str) or not isinstance(utterance_key, str):
                raise TypeError(f"manifest line {line_number} has no string audio path/utterance key")
            if utterance_key in seen or not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError(f"manifest line {line_number} has a duplicate key or invalid duration")
            seen.add(utterance_key)
            audio_path = (args.audio_root / relative_path).resolve(strict=False)
            audio_path.relative_to(args.audio_root)
            items.append(
                ExtractionItem(
                    utterance_key=utterance_key,
                    audio_path=audio_path,
                    cache_path=safe_cache_path(args.cache_dir, relative_path),
                    duration_seconds=float(duration),
                )
            )
    if len(items) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} train records, found {len(items)}")
    return items


def iter_batches(items: list[ExtractionItem], max_items: int, max_seconds: float):
    batch: list[ExtractionItem] = []
    max_duration = 0.0
    for item in items:
        next_max_duration = max(max_duration, item.duration_seconds)
        next_padded_seconds = next_max_duration * (len(batch) + 1)
        if batch and (len(batch) >= max_items or next_padded_seconds > max_seconds):
            yield batch
            batch, max_duration = [], 0.0
        batch.append(item)
        max_duration = max(max_duration, item.duration_seconds)
    if batch:
        yield batch


def cache_is_valid(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        validate_repa_feature_array(np.load(path, allow_pickle=False), source=path)
    except Exception:  # noqa: BLE001 - any malformed target is recomputed
        return False
    return True


def load_waveform(path: Path, max_duration_seconds: float) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"audio must be a regular file: {path}")
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError("empty waveform")
    waveform = torch.from_numpy(waveform.mean(axis=1))
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() > int(max_duration_seconds * SAMPLE_RATE):
        raise ValueError(f"duration exceeds extraction limit {max_duration_seconds}s")
    return waveform


def load_teacher(device: torch.device):
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=WAVLM_BASE_PLUS_MODEL_ID,
            filename="pytorch_model.bin",
            revision=WAVLM_BASE_PLUS_REVISION,
        )
    )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != WAVLM_BASE_PLUS_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"WavLM checkpoint SHA256 mismatch: expected={WAVLM_BASE_PLUS_CHECKPOINT_SHA256}, "
            f"got={actual_sha256}"
        )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        WAVLM_BASE_PLUS_MODEL_ID,
        revision=WAVLM_BASE_PLUS_REVISION,
    )
    model = WavLMModel.from_pretrained(
        WAVLM_BASE_PLUS_MODEL_ID,
        revision=WAVLM_BASE_PLUS_REVISION,
        torch_dtype=torch.float32,
    )
    if model.config.hidden_size != WAVLM_BASE_PLUS_DIM or model.config.num_hidden_layers != WAVLM_BASE_PLUS_LAYER:
        raise RuntimeError("downloaded WavLM architecture is not the pinned 12-layer 768-D Base+ model")
    model.eval().requires_grad_(False)
    return feature_extractor, model.to(device)


@torch.inference_mode()
def extract_batch(items, feature_extractor, model, device, max_duration_seconds) -> list[tuple[ExtractionItem, int]]:
    waveforms = [load_waveform(item.audio_path, max_duration_seconds).numpy() for item in items]
    inputs = feature_extractor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model(input_values=input_values, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    feature_lens = model._get_feat_extract_output_lengths(attention_mask.sum(dim=1)).long()
    extracted = []
    for index, item in enumerate(items):
        frame_count = int(feature_lens[index].item())
        feature = hidden[index, :frame_count].float().cpu().numpy().astype(np.float16)
        validate_repa_feature_array(feature, source=item.audio_path)
        atomic_save_npy(item.cache_path, feature)
        extracted.append((item, frame_count))
    return extracted


def verify_cache(items: list[ExtractionItem], cache_dir: Path) -> dict:
    expected_paths = {item.cache_path for item in items}
    missing, invalid = [], []
    total_frames = 0
    for path in tqdm(sorted(expected_paths), desc="Verifying WavLM REPA cache", unit="file"):
        if not path.is_file() or path.is_symlink():
            missing.append(str(path))
            continue
        try:
            feature = np.load(path, allow_pickle=False)
            validate_repa_feature_array(feature, source=path)
            total_frames += feature.shape[0]
        except Exception as error:  # noqa: BLE001 - report every invalid artifact
            invalid.append({"path": str(path), "error": str(error)})
    actual_paths = set(cache_dir.rglob("*.npy"))
    extra = sorted(str(path) for path in actual_paths - expected_paths)
    return {
        "complete": not missing and not invalid and not extra,
        "expected": len(items),
        "valid": len(items) - len(missing) - len(invalid),
        "split_counts": {"train": len(items)},
        "total_frames": total_frames,
        "missing": missing,
        "invalid": invalid,
        "extra": extra,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--expected-count", type=int, default=79_613)
    parser.add_argument("--max-batch-items", type=int, default=8)
    parser.add_argument("--max-batch-seconds", type=float, default=60.0)
    parser.add_argument("--max-duration-seconds", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve(strict=True)
    args.audio_root = args.audio_root.expanduser().resolve(strict=True)
    args.cache_dir = args.cache_dir.expanduser().absolute()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir.resolve(strict=True)
    if args.max_batch_items <= 0 or args.max_batch_seconds <= 0 or args.max_duration_seconds <= 0:
        raise ValueError("batch and duration limits must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("WavLM cache extraction requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    inventory = read_inventory(args)
    shard = inventory[rank::world_size]
    feature_extractor, model = load_teacher(device)
    failures = []
    processed = 0
    shard_batches = list(iter_batches(shard, args.max_batch_items, args.max_batch_seconds))
    for batch in tqdm(shard_batches, desc=f"rank {rank} WavLM", unit="batch", disable=rank != 0):
        pending = [item for item in batch if not cache_is_valid(item.cache_path)]
        if not pending:
            continue
        try:
            processed += len(extract_batch(pending, feature_extractor, model, device, args.max_duration_seconds))
        except Exception as error:  # noqa: BLE001 - isolate failures without discarding a whole batch
            for item in pending:
                try:
                    processed += len(
                        extract_batch([item], feature_extractor, model, device, args.max_duration_seconds)
                    )
                except Exception as item_error:  # noqa: BLE001 - preserve the full failure inventory
                    failures.append(
                        {
                            "utterance_key": item.utterance_key,
                            "audio_path": str(item.audio_path),
                            "error": f"{type(item_error).__name__}: {item_error}",
                            "batch_error": f"{type(error).__name__}: {error}",
                        }
                    )

    atomic_save_json(
        args.cache_dir / "manifests" / f"rank_{rank:03d}.json",
        {"rank": rank, "world_size": world_size, "processed": processed, "failures": failures},
    )
    failure_count = torch.tensor([len(failures)], device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(failure_count)
        dist.barrier()
    if failure_count.item():
        raise RuntimeError(f"WavLM extraction failed for {failure_count.item()} utterances; inspect rank manifests")

    if rank == 0:
        coverage = verify_cache(inventory, args.cache_dir)
        atomic_save_json(args.cache_dir / "coverage_report.json", coverage)
        if not coverage["complete"]:
            raise RuntimeError("WavLM cache coverage verification failed; inspect coverage_report.json")
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "feature": "wavlm_hidden_state",
            "model_id": WAVLM_BASE_PLUS_MODEL_ID,
            "model_revision": WAVLM_BASE_PLUS_REVISION,
            "checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
            "teacher_layer": WAVLM_BASE_PLUS_LAYER,
            "sample_rate": SAMPLE_RATE,
            "frame_stride_samples": WAVLM_FRAME_STRIDE_SAMPLES,
            "frame_rate_hz": SAMPLE_RATE / WAVLM_FRAME_STRIDE_SAMPLES,
            "output_dim": WAVLM_BASE_PLUS_DIM,
            "output_dtype": "float16",
            "source_audio": "complete_unmasked_waveform",
            "manifest_sha256": args.expected_manifest_sha256,
            "expected_count": args.expected_count,
            "coverage_report": "coverage_report.json",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save_json(args.cache_dir / "metadata.json", metadata)
        print(
            f"Complete WavLM-Base+ cache: files={coverage['valid']}, frames={coverage['total_frames']}, "
            f"root={args.cache_dir}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
