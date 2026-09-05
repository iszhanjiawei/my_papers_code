"""Extract frozen CAM++ speaker embeddings from complete, unmasked waveforms.

The training process never instantiates CAM++. This one-time preprocessing job
mirrors ``audio/<split>/...wav`` under the configured cache directory and writes
one L2-normalized FP32 vector per utterance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from datasets import Dataset
from torch.utils.data import DataLoader
from torchaudio.compliance.kaldi import fbank
from tqdm import tqdm

from aligndit.model.speaker_embedding import (
    DEFAULT_SPEAKER_EMBEDDING_DIM,
    SpeakerEmbeddingError,
    speaker_embedding_path,
    validate_speaker_embedding_array,
)


MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
MODEL_REVISION = "v1.0.0"
MODEL_SOURCE_COMMIT = "065629c313eaf1a01c65c640c46d77e61e9607b4"
EXPECTED_CHECKPOINT_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
MAX_SECONDS = 90
FBANK_DIM = 80


@dataclass(frozen=True)
class ExtractionItem:
    audio_path: str
    cache_path: str
    split: str


class AudioFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[ExtractionItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        try:
            waveform, sample_rate = sf.read(item.audio_path, dtype="float32", always_2d=True)
            if waveform.shape[0] == 0:
                raise ValueError("empty waveform")
            waveform = torch.from_numpy(waveform.mean(axis=1))
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            chunks = circle_pad_chunks(waveform)
            features = []
            for chunk in chunks:
                feature = fbank(
                    chunk.unsqueeze(0),
                    num_mel_bins=FBANK_DIM,
                    sample_frequency=SAMPLE_RATE,
                    dither=0.0,
                )
                feature = feature - feature.mean(dim=0, keepdim=True)
                features.append(feature)
            return item, torch.stack(features), None
        except Exception as error:  # noqa: BLE001 - record the bad path and keep the shard auditable
            return item, None, f"{type(error).__name__}: {error}"


def circle_pad_chunks(waveform: torch.Tensor) -> torch.Tensor:
    """Apply the official 10 s chunking and whole-utterance circular padding."""
    waveform = waveform.flatten()[: SAMPLE_RATE * MAX_SECONDS]
    if waveform.numel() == 0:
        raise ValueError("empty waveform")
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS
    num_chunks = math.ceil(waveform.numel() / chunk_samples)
    target_samples = num_chunks * chunk_samples
    repeats = math.ceil(target_samples / waveform.numel())
    padded = waveform.repeat(repeats)[:target_samples]
    return padded.reshape(num_chunks, chunk_samples)


def collate_features(batch):
    items = []
    counts = []
    feature_batches = []
    errors = []
    for item, features, error in batch:
        if error is not None:
            errors.append((item, error))
            continue
        items.append(item)
        counts.append(features.shape[0])
        feature_batches.append(features)
    return {
        "items": items,
        "counts": counts,
        "features": torch.cat(feature_batches, dim=0) if feature_batches else None,
        "errors": errors,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
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


def resolve_arrow_audio_path(audio_path: str, data_dir: Path) -> Path:
    path = Path(audio_path)
    if not path.is_absolute():
        # Match CustomDataset_mel exactly: dirname(data_dir) + Arrow path.
        path = data_dir.parent / path
    return path


def build_inventory(args) -> list[ExtractionItem]:
    audio_root = args.audio_root
    cache_dir = args.cache_dir
    train_dataset = Dataset.from_file(str(args.dataset_arrow))
    items = []
    for row in train_dataset:
        audio_path = resolve_arrow_audio_path(row["audio_path"], args.data_dir)
        items.append(
            ExtractionItem(
                audio_path=str(audio_path),
                cache_path=str(speaker_embedding_path(audio_path, cache_dir, audio_root=audio_root)),
                split="train",
            )
        )

    if args.test_list is not None:
        for line in args.test_list.read_text(encoding="utf-8").splitlines():
            relative_path = line.strip()
            if not relative_path:
                continue
            audio_path = audio_root / "test" / f"{relative_path}.wav"
            items.append(
                ExtractionItem(
                    audio_path=str(audio_path),
                    cache_path=str(speaker_embedding_path(audio_path, cache_dir, audio_root=audio_root)),
                    split="test",
                )
            )

    paths = [item.audio_path for item in items]
    if len(paths) != len(set(paths)):
        raise RuntimeError("the extraction inventory contains duplicate audio paths")
    if args.limit is not None:
        items = items[: args.limit]
    return items


def cache_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        embedding = np.load(path, allow_pickle=False)
        validate_speaker_embedding_array(embedding, source=path)
    except Exception:  # noqa: BLE001 - any malformed cache must be recomputed
        return False
    return True


def load_campplus(checkpoint_path: Path, device: torch.device):
    # FunASR vendors the official CAM++ architecture and is already an eval
    # dependency of this repository. Strict loading protects architecture drift.
    from funasr.models.campplus.model import CAMPPlus

    model = CAMPPlus(
        feat_dim=FBANK_DIM,
        embedding_size=DEFAULT_SPEAKER_EMBEDDING_DIM,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=torch.float32)


def write_manifest_record(file, item: ExtractionItem, status: str, **kwargs) -> None:
    record = {
        "audio_path": item.audio_path,
        "cache_path": item.cache_path,
        "split": item.split,
        "status": status,
        **kwargs,
    }
    file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def verify_inventory(items: list[ExtractionItem], cache_dir: Path) -> dict:
    expected_paths = {Path(item.cache_path) for item in items}
    missing = []
    invalid = []
    for path in tqdm(sorted(expected_paths), desc="Verifying CAM++ cache", unit="file"):
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            embedding = np.load(path, allow_pickle=False)
            validate_speaker_embedding_array(embedding, source=path)
        except Exception as error:  # noqa: BLE001 - report every cache-contract failure
            invalid.append({"path": str(path), "error": str(error)})

    actual_paths = set(cache_dir.rglob("*.npy"))
    extra = sorted(str(path) for path in actual_paths - expected_paths)
    split_counts = {
        split: sum(item.split == split for item in items) for split in sorted({item.split for item in items})
    }
    return {
        "expected": len(expected_paths),
        "valid": len(expected_paths) - len(missing) - len(invalid),
        "missing": missing,
        "invalid": invalid,
        "extra": extra,
        "split_counts": split_counts,
        "complete": not missing and not invalid and not extra,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--batch-utterances", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_utterances <= 0 or args.num_workers < 0:
        raise ValueError("batch-utterances must be positive and num-workers must be non-negative")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="gloo")

    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"CAM++ checkpoint SHA256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_hash}"
        )

    inventory = build_inventory(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "status": "extracting" if args.limit is None else "partial_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "sample_rate": SAMPLE_RATE,
        "fbank_dim": FBANK_DIM,
        "fbank_mean_normalization": "per_chunk_per_mel_over_time",
        "chunk_seconds": CHUNK_SECONDS,
        "max_seconds": MAX_SECONDS,
        "padding": "repeat_complete_utterance_to_chunk_multiple",
        "chunk_aggregation": "arithmetic_mean_then_l2_normalize",
        "output_shape": [DEFAULT_SPEAKER_EMBEDDING_DIM],
        "output_dtype": "float32",
        "source_audio": "complete_unmasked_waveform",
        "expected_count": len(inventory),
        "world_size": world_size,
    }
    if rank == 0:
        atomic_save_json(args.cache_dir / "metadata.json", metadata)
    if distributed:
        dist.barrier()

    assigned = [item for index, item in enumerate(inventory) if index % world_size == rank]
    pending = []
    cached = []
    existing_cache_paths = set(args.cache_dir.rglob("*.npy")) if not args.overwrite else set()
    for item in assigned:
        cache_path = Path(item.cache_path)
        if cache_path in existing_cache_paths and cache_is_valid(cache_path):
            cached.append(item)
        else:
            pending.append(item)

    manifest_path = args.cache_dir / f"manifest.rank{rank:02d}.jsonl"
    error_count = 0
    created_count = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for item in cached:
            write_manifest_record(manifest, item, "cached")

        if pending:
            model = load_campplus(args.checkpoint, device)
            dataset = AudioFeatureDataset(pending)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_utterances,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                multiprocessing_context="spawn" if args.num_workers > 0 else None,
                collate_fn=collate_features,
            )
            for batch in tqdm(loader, desc=f"CAM++ rank {rank}", unit="batch"):
                for item, error in batch["errors"]:
                    error_count += 1
                    write_manifest_record(manifest, item, "error", error=error)
                if batch["features"] is None:
                    continue
                features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.inference_mode():
                    chunk_embeddings = model(features).float().cpu()
                offset = 0
                for item, chunk_count in zip(batch["items"], batch["counts"]):
                    embedding = chunk_embeddings[offset : offset + chunk_count].mean(dim=0)
                    embedding = F.normalize(embedding, dim=0).numpy().astype(np.float32, copy=False)
                    offset += chunk_count
                    try:
                        validate_speaker_embedding_array(embedding, source=item.audio_path)
                        atomic_save_npy(Path(item.cache_path), embedding)
                        created_count += 1
                        write_manifest_record(manifest, item, "created", chunks=chunk_count)
                    except Exception as error:  # noqa: BLE001 - record the exact failed utterance
                        error_count += 1
                        write_manifest_record(
                            manifest,
                            item,
                            "error",
                            error=f"{type(error).__name__}: {error}",
                        )
                manifest.flush()

    rank_summary = {
        "rank": rank,
        "world_size": world_size,
        "assigned": len(assigned),
        "cached": len(cached),
        "created": created_count,
        "errors": error_count,
        "device": str(device),
    }
    atomic_save_json(args.cache_dir / f"summary.rank{rank:02d}.json", rank_summary)

    failure_tensor = torch.tensor([int(error_count > 0)], dtype=torch.int64)
    if distributed:
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
        dist.barrier()

    verification_ok = False
    if rank == 0 and failure_tensor.item() == 0:
        report = verify_inventory(inventory, args.cache_dir)
        report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_save_json(args.cache_dir / "coverage_report.json", report)
        verification_ok = report["complete"]
        metadata["status"] = "complete" if verification_ok and args.limit is None else "partial_smoke"
        metadata["coverage_report"] = "coverage_report.json"
        atomic_save_json(args.cache_dir / "metadata.json", metadata)

    verification_tensor = torch.tensor([int(verification_ok)], dtype=torch.int64)
    if distributed:
        dist.broadcast(verification_tensor, src=0)
        dist.destroy_process_group()
    if failure_tensor.item() != 0 or verification_tensor.item() != 1:
        raise SpeakerEmbeddingError(
            f"CAM++ extraction failed validation; inspect {args.cache_dir}/summary.rank*.json and coverage_report.json"
        )
    if rank == 0:
        print(f"CAM++ cache complete: {len(inventory)} embeddings in {args.cache_dir}")


if __name__ == "__main__":
    main()
