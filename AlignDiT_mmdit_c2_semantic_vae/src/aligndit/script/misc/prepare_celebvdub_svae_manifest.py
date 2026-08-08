"""Build an immutable CelebV-Dub inventory for deterministic Semantic-VAE caching."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Dataset
from tqdm import tqdm

from aligndit.script.misc.svae_cache_utils import (
    BASE_POSTERIOR_SEED,
    CACHE_SCHEMA_VERSION,
    SAMPLE_RATE,
    SEMANTIC_VAE_HOP_LENGTH,
    SEMANTIC_VAE_LATENT_DIM,
    atomic_write_json,
    atomic_write_jsonl,
    expected_latent_frames,
    prefixed_path,
    sha256_file,
    stable_utterance_seed,
)


VIDEO_DIM = 1024
EXPECTED_TRAIN_COUNT = 79_613
EXPECTED_TEST_COUNT = 213
EXPECTED_CTC40_VALID_TRAIN_COUNT = 79_508


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=prefixed_path("datasets/CelebV-Dub"))
    parser.add_argument(
        "--metadata-arrow",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub_char/raw.arrow"),
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub/avhubert_video_feat"),
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub_char/vocab.txt"),
        help="Exact character vocabulary used by the CelebV-Dub C2 training entrypoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=prefixed_path("projects/data/CelebVDub_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument("--base-seed", type=int, default=BASE_POSTERIOR_SEED)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--expected-train-count", type=int, default=EXPECTED_TRAIN_COUNT)
    parser.add_argument("--expected-test-count", type=int, default=EXPECTED_TEST_COUNT)
    parser.add_argument(
        "--expected-ctc40-valid-train-count",
        type=int,
        default=EXPECTED_CTC40_VALID_TRAIN_COUNT,
    )
    parser.add_argument(
        "--include-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the official 213-item test set in the immutable cache inventory.",
    )
    return parser.parse_args()


def _clean_text(raw: str) -> str:
    text = raw.strip()
    if text.lower().startswith("text:"):
        text = text.partition(":")[2]
    return text.strip().lower()


def _load_char_vocab(vocab_path: Path) -> dict[str, int]:
    # This deliberately mirrors f5_tts.model.utils.get_tokenizer(..., "custom") so
    # the offline CTC feasibility check cannot silently diverge from training.
    characters = vocab_path.read_text(encoding="utf-8").splitlines()
    if not characters or len(characters) != len(set(characters)):
        raise ValueError(f"Invalid or duplicate character vocabulary: {vocab_path}")
    vocab = {character: index for index, character in enumerate(characters)}
    if vocab.get(" ") != 0:
        raise ValueError(f"CelebV-Dub vocabulary must assign the space/unknown fallback token to id 0: {vocab_path}")
    return vocab


def _add_ctc_target_contract(record: dict[str, Any], vocab: dict[str, int]) -> None:
    text = record["text"]
    token_ids = [vocab.get(character, 0) for character in text]
    adjacent_repeats = sum(left == right for left, right in pairwise(token_ids))
    record.update(
        {
            "ctc_adjacent_repeats": adjacent_repeats,
            "ctc_min_input_frames": len(token_ids) + adjacent_repeats,
            "ctc_target_length": len(token_ids),
        }
    )


def _canonical_source(split: str, video_id: str, filename: str) -> str:
    relative = Path(split, video_id, filename)
    if (
        relative.is_absolute()
        or relative.as_posix() != f"{split}/{video_id}/{filename}"
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative.as_posix()
    ):
        raise ValueError(f"Non-canonical CelebV-Dub source path: {relative}")
    return relative.as_posix()


def _parse_arrow_audio_path(value: str) -> tuple[str, str, str]:
    path = Path(value)
    parts = path.parts
    expected_prefix = ("data", "CelebVDub", "audio")
    if path.is_absolute() or len(parts) != 6 or parts[:3] != expected_prefix or parts[3] != "train":
        raise ValueError(f"Unexpected CelebV-Dub Arrow audio path: {value!r}")
    video_id, filename = parts[4], parts[5]
    if Path(filename).suffix.lower() != ".wav":
        raise ValueError(f"CelebV-Dub Arrow source is not WAV: {value!r}")
    return "train", video_id, filename


def _base_record(
    *,
    split: str,
    video_id: str,
    filename: str,
    text: str,
    source_index: int,
    base_seed: int,
) -> dict[str, Any]:
    stem = Path(filename).stem
    utterance_key = f"celebvdub/{split}/{video_id}/{stem}"
    return {
        "audio_relative_path": _canonical_source(split, video_id, filename),
        "latent_dim": SEMANTIC_VAE_LATENT_DIM,
        "latent_relative_path": f"latents/{split}/{video_id}/{stem}.npy",
        "posterior_seed": stable_utterance_seed(base_seed, utterance_key),
        "source_index": source_index,
        "split": split,
        "subset": split,
        "text": text,
        "utterance_key": utterance_key,
        "video_id": video_id,
        "video_40hz_relative_path": f"video_40hz/{split}/{video_id}/{stem}.npy",
        "video_relative_path": f"{split}/{video_id}/{stem}.npy",
    }


def _load_train_records(metadata_arrow: Path, base_seed: int) -> list[dict[str, Any]]:
    dataset = Dataset.from_file(str(metadata_arrow))
    expected_columns = {"audio_path", "duration", "text"}
    if set(dataset.column_names) != expected_columns:
        raise ValueError(f"Unexpected CelebV-Dub Arrow columns: {dataset.column_names}")
    records: list[dict[str, Any]] = []
    for source_index, row in enumerate(dataset):
        split, video_id, filename = _parse_arrow_audio_path(row["audio_path"])
        text = str(row["text"]).strip().lower()
        if not text:
            raise ValueError(f"Empty transcript in Arrow row {source_index}")
        record = _base_record(
            split=split,
            video_id=video_id,
            filename=filename,
            text=text,
            source_index=source_index,
            base_seed=base_seed,
        )
        record["arrow_duration_seconds"] = float(row["duration"])
        records.append(record)
    return records


def _load_test_records(dataset_root: Path, base_seed: int, start_index: int) -> list[dict[str, Any]]:
    test_root = dataset_root / "test"
    audio_paths = sorted(test_root.glob("*/*.wav"), key=lambda path: (path.parent.name, path.name))
    records: list[dict[str, Any]] = []
    for offset, audio_path in enumerate(audio_paths):
        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file() or text_path.is_symlink():
            raise FileNotFoundError(f"Missing regular test transcript: {text_path}")
        text = _clean_text(text_path.read_text(encoding="utf-8", errors="strict"))
        if not text:
            raise ValueError(f"Empty test transcript: {text_path}")
        records.append(
            _base_record(
                split="test",
                video_id=audio_path.parent.name,
                filename=audio_path.name,
                text=text,
                source_index=start_index + offset,
                base_seed=base_seed,
            )
        )
    return records


def _inspect_record(
    record: dict[str, Any],
    dataset_root: str,
    video_root: str,
) -> dict[str, Any]:
    audio_path = Path(dataset_root, record["audio_relative_path"])
    video_path = Path(video_root, record["video_relative_path"])
    if not audio_path.is_file() or audio_path.is_symlink():
        raise FileNotFoundError(f"Expected a regular source WAV: {audio_path}")
    if not video_path.is_file() or video_path.is_symlink():
        raise FileNotFoundError(f"Expected a regular AV-HuBERT feature: {video_path}")

    info = sf.info(audio_path)
    source_sample_rate = int(info.samplerate)
    source_channels = int(info.channels)
    source_samples = int(info.frames)
    if source_sample_rate <= 0 or source_channels <= 0 or source_samples <= 0:
        raise ValueError(
            f"Invalid source audio {audio_path}: sample_rate={source_sample_rate}, "
            f"channels={source_channels}, frames={source_samples}"
        )
    if record["split"] == "train":
        if source_sample_rate != SAMPLE_RATE or source_channels != 1:
            raise ValueError(
                f"Training audio must already be {SAMPLE_RATE} Hz mono: {audio_path}: "
                f"sample_rate={source_sample_rate}, channels={source_channels}"
            )
        target_samples = source_samples
    else:
        waveform, decoded_sample_rate = torchaudio.load(audio_path)
        if decoded_sample_rate != source_sample_rate or waveform.shape != (source_channels, source_samples):
            raise ValueError(
                f"SoundFile/torchaudio decode mismatch for {audio_path}: "
                f"info={(source_channels, source_samples, source_sample_rate)}, "
                f"decoded={(tuple(waveform.shape), decoded_sample_rate)}"
            )
        waveform = waveform.to(torch.float32).mean(dim=0, keepdim=True)
        if source_sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, source_sample_rate, SAMPLE_RATE)
        target_samples = int(waveform.shape[-1])
        if target_samples <= 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"Invalid 16-kHz test waveform after preprocessing: {audio_path}")

    frames = expected_latent_frames(target_samples)
    video = np.load(video_path, allow_pickle=False, mmap_mode="r")
    if video.ndim != 2 or video.shape[0] <= 0 or video.shape[1] != VIDEO_DIM or video.dtype != np.float32:
        raise ValueError(f"Invalid AV-HuBERT feature {video_path}: shape={video.shape}, dtype={video.dtype}")

    duration = target_samples / SAMPLE_RATE
    arrow_duration = record.pop("arrow_duration_seconds", None)
    if arrow_duration is not None and abs(float(arrow_duration) - duration) > 0.5 / SAMPLE_RATE:
        raise ValueError(
            f"Arrow/source duration mismatch for {record['utterance_key']}: {arrow_duration} != {duration}"
        )
    record.update(
        {
            "duration_seconds": duration,
            "ctc_feasible_40hz": frames >= record["ctc_min_input_frames"],
            "latent_frames": frames,
            "num_channels": 1,
            "original_num_samples": target_samples,
            "padded_num_samples": frames * SEMANTIC_VAE_HOP_LENGTH,
            "sample_rate": SAMPLE_RATE,
            "source_num_channels": source_channels,
            "source_num_samples": source_samples,
            "source_sample_rate": source_sample_rate,
            "video_dim": VIDEO_DIM,
            "video_frames_25hz": int(video.shape[0]),
        }
    )
    return record


def _line_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def main() -> None:
    args = get_args()
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    metadata_arrow = args.metadata_arrow.expanduser().resolve(strict=True)
    video_root = args.video_root.expanduser().resolve(strict=True)
    vocab_path = args.vocab_path.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().absolute()
    if args.workers <= 0:
        raise ValueError(f"workers must be positive, got {args.workers}")
    if args.expected_train_count <= 0 or args.expected_test_count <= 0 or args.expected_ctc40_valid_train_count <= 0:
        raise ValueError("Expected split counts must be positive")

    vocab = _load_char_vocab(vocab_path)

    records = _load_train_records(metadata_arrow, args.base_seed)
    if len(records) != args.expected_train_count:
        raise RuntimeError(f"Expected {args.expected_train_count} train rows, found {len(records)}")
    if args.include_test:
        test_records = _load_test_records(dataset_root, args.base_seed, len(records))
        if len(test_records) != args.expected_test_count:
            raise RuntimeError(f"Expected {args.expected_test_count} test rows, found {len(test_records)}")
        records.extend(test_records)
    for record in records:
        _add_ctc_target_contract(record, vocab)

    keys = [record["utterance_key"] for record in records]
    audio_paths = [record["audio_relative_path"] for record in records]
    if len(keys) != len(set(keys)) or len(audio_paths) != len(set(audio_paths)):
        raise RuntimeError("Duplicate CelebV-Dub utterance key or audio path")

    inspected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(
            _inspect_record,
            records,
            [str(dataset_root)] * len(records),
            [str(video_root)] * len(records),
            chunksize=128,
        )
        inspected = list(tqdm(iterator, total=len(records), desc="Inspecting CelebV-Dub"))

    train_records = [record for record in inspected if record["split"] == "train"]
    test_records = [record for record in inspected if record["split"] == "test"]
    train_ctc40_valid = [record for record in train_records if record["ctc_feasible_40hz"]]
    ctc40_excluded = [record for record in train_records if not record["ctc_feasible_40hz"]]
    if len(train_ctc40_valid) != args.expected_ctc40_valid_train_count:
        raise RuntimeError(
            "Unexpected number of 40-Hz CTC-feasible training items: "
            f"expected {args.expected_ctc40_valid_train_count}, found {len(train_ctc40_valid)}"
        )
    manifests_dir = output_root / "manifests"
    outputs = {
        "ctc40_excluded.jsonl": ctc40_excluded,
        "inventory.jsonl": inspected,
        "test.jsonl": test_records,
        "train.jsonl": train_records,
        "train_ctc40_valid.jsonl": train_ctc40_valid,
    }
    write_results = {name: atomic_write_jsonl(manifests_dir / name, value) for name, value in outputs.items()}
    manifest_entries = {
        name: {
            "count": _line_count(result.path),
            "path": result.path.relative_to(output_root).as_posix(),
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
        }
        for name, result in write_results.items()
    }
    ctc40_report = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "criterion": "input_frames >= target_length + adjacent_repeated_target_tokens",
        "input_frame_rate_hz": SAMPLE_RATE / SEMANTIC_VAE_HOP_LENGTH,
        "input_stride": 1,
        "record_fields": [
            "ctc_target_length",
            "ctc_adjacent_repeats",
            "ctc_min_input_frames",
            "ctc_feasible_40hz",
        ],
        "test": {
            "excluded": sum(not record["ctc_feasible_40hz"] for record in test_records),
            "total": len(test_records),
            "valid": sum(record["ctc_feasible_40hz"] for record in test_records),
        },
        "tokenizer": {
            # TextEmbedding reserves one filler embedding above the raw 159-entry
            # vocabulary; CFM then reserves the following class for CTC blank.
            "blank_index": len(vocab) + 1,
            "encoding": "one_python_character_per_token_unknown_to_id_0",
            "output_classes": len(vocab) + 2,
            "path": str(vocab_path),
            "sha256": sha256_file(vocab_path),
            "size": len(vocab),
            "unknown_id": 0,
            "unused_class_index": len(vocab),
        },
        "train": {
            "excluded": len(ctc40_excluded),
            "total": len(train_records),
            "valid": len(train_ctc40_valid),
        },
    }
    ctc40_report_result = atomic_write_json(manifests_dir / "ctc40_report.json", ctc40_report)
    source_path = Path(__file__).resolve(strict=True)
    utility_path = source_path.with_name("svae_cache_utils.py").resolve(strict=True)
    source_code = {
        "cache_utility": {
            "path": str(utility_path),
            "sha256": sha256_file(utility_path),
        },
        "manifest_builder": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
    }
    metadata = {
        "base_posterior_seed": args.base_seed,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": "CelebV-Dub",
        "dataset_root": str(dataset_root),
        "latent_spec": {
            "dimension": SEMANTIC_VAE_LATENT_DIM,
            "dtype": "float32",
            "frame_rate_hz": SAMPLE_RATE / SEMANTIC_VAE_HOP_LENGTH,
            "hop_length_samples": SEMANTIC_VAE_HOP_LENGTH,
            "mode": "fixed_posterior_sample",
            "sample_rate": SAMPLE_RATE,
        },
        "manifests": manifest_entries,
        "metadata_arrow": {
            "path": str(metadata_arrow),
            "sha256": sha256_file(metadata_arrow),
            "size_bytes": metadata_arrow.stat().st_size,
        },
        "split_counts": {"test": len(test_records), "train": len(train_records)},
        "ctc40_preflight": {
            "report": {
                "path": ctc40_report_result.path.relative_to(output_root).as_posix(),
                "sha256": ctc40_report_result.sha256,
                "size_bytes": ctc40_report_result.size_bytes,
            },
            "test_excluded": ctc40_report["test"]["excluded"],
            "train_excluded": len(ctc40_excluded),
            "train_valid": len(train_ctc40_valid),
            "vocab_sha256": ctc40_report["tokenizer"]["sha256"],
        },
        "source_code": source_code,
        "total_latent_frames": sum(record["latent_frames"] for record in inspected),
        "total_original_samples": sum(record["original_num_samples"] for record in inspected),
        "video_feature_spec": {
            "dimension": VIDEO_DIM,
            "dtype": "float32",
            "frame_rate_hz": 25,
            "root": str(video_root),
        },
    }
    metadata_result = atomic_write_json(manifests_dir / "inventory_meta.json", metadata)
    spec = {
        "base_posterior_seed": args.base_seed,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "include_test": args.include_test,
        "metadata_arrow_sha256": metadata["metadata_arrow"]["sha256"],
        "source_code": source_code,
        "train_ctc40_valid_count": len(train_ctc40_valid),
        "vocab_sha256": ctc40_report["tokenizer"]["sha256"],
        "video_root": str(video_root),
    }
    spec_result = atomic_write_json(manifests_dir / "manifest_spec.json", spec)
    print(
        f"CelebV-Dub inventory complete: train={len(train_records)}, test={len(test_records)}, "
        f"ctc40_valid={len(train_ctc40_valid)}, ctc40_excluded={len(ctc40_excluded)}, "
        f"inventory_sha256={manifest_entries['inventory.jsonl']['sha256']}, "
        f"metadata_sha256={metadata_result.sha256}, spec_sha256={spec_result.sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
