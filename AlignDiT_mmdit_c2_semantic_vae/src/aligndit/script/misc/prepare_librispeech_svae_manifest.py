"""Build a deterministic, auditable LibriSpeech inventory for Semantic-VAE caching."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from aligndit.script.misc.svae_cache_utils import (
    BASE_POSTERIOR_SEED,
    CACHE_SCHEMA_VERSION,
    DEFAULT_SUBSETS,
    DEV_SUBSETS,
    OFFICIAL_UTTERANCE_COUNTS,
    SAMPLE_RATE,
    SEMANTIC_VAE_HOP_LENGTH,
    SEMANTIC_VAE_LATENT_DIM,
    TRAIN_SUBSETS,
    atomic_write_json,
    atomic_write_jsonl,
    expected_latent_frames,
    prefixed_path,
    stable_utterance_seed,
)


@dataclass(frozen=True)
class AudioIdentity:
    subset: str
    speaker_id: str
    chapter_id: str
    utterance_id: str
    audio_path: Path
    transcript: str

    @property
    def utterance_key(self) -> str:
        return f"{self.subset}/{self.speaker_id}/{self.chapter_id}/{self.utterance_id}"


def is_ascii_decimal(value: str) -> bool:
    return value.isascii() and value.isdecimal()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=prefixed_path("datasets/LibriSpeech"),
        help="Root containing LibriSpeech subsets. Standard and subset/LibriSpeech/subset layouts are supported.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=prefixed_path("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--min-duration", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--base-seed", type=int, default=BASE_POSTERIOR_SEED)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--limit-per-subset",
        type=int,
        default=None,
        help="Deterministic development-only limit applied after full subset discovery and count checks.",
    )
    parser.add_argument(
        "--verify-official-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every known subset to contain the official number of FLAC files before selection.",
    )
    return parser.parse_args()


def resolve_subset_root(dataset_root: Path, subset: str) -> Path:
    standard = dataset_root / subset
    nested = standard / "LibriSpeech" / subset
    candidates = [nested, standard]
    valid = [candidate for candidate in candidates if candidate.is_dir() and next(candidate.glob("*/*/*.flac"), None)]
    if len(valid) == 0:
        raise FileNotFoundError(
            f"Could not find FLAC files for {subset!r} below {dataset_root}; checked {nested} and {standard}"
        )
    if len(valid) > 1:
        raise RuntimeError(f"Ambiguous roots for {subset!r}: {[str(path) for path in valid]}")
    return valid[0].resolve()


def parse_transcripts(subset_root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    transcript_files = sorted(subset_root.rglob("*.trans.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found below {subset_root}")

    for transcript_path in transcript_files:
        relative_parts = transcript_path.relative_to(subset_root).parts
        if len(relative_parts) != 3:
            raise ValueError(f"Transcript must be stored as <speaker>/<chapter>/<file>, got {transcript_path}")
        speaker_id, chapter_id, filename = relative_parts
        if not is_ascii_decimal(speaker_id) or not is_ascii_decimal(chapter_id):
            raise ValueError(f"Non-numeric speaker/chapter in {transcript_path}")
        if filename != f"{speaker_id}-{chapter_id}.trans.txt":
            raise ValueError(f"Transcript filename does not match its hierarchy: {transcript_path}")
        with transcript_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                utterance_id, separator, text = line.rstrip("\r\n").partition(" ")
                if not separator or not utterance_id or not text.strip():
                    raise ValueError(f"Malformed transcript at {transcript_path}:{line_number}: {line!r}")
                utterance_parts = utterance_id.split("-")
                if (
                    len(utterance_parts) != 3
                    or not all(is_ascii_decimal(part) for part in utterance_parts)
                    or len(utterance_parts[2]) != 4
                ):
                    raise ValueError(f"Malformed utterance ID at {transcript_path}:{line_number}: {utterance_id!r}")
                if utterance_parts[:2] != [speaker_id, chapter_id]:
                    raise ValueError(
                        f"Transcript utterance ID does not match its hierarchy at {transcript_path}:{line_number}"
                    )
                if utterance_id in transcripts:
                    raise ValueError(f"Duplicate transcript for {utterance_id} in {transcript_path}:{line_number}")
                transcripts[utterance_id] = text.strip()
    return transcripts


def identify_audio(subset: str, subset_root: Path, audio_path: Path, transcripts: dict[str, str]) -> AudioIdentity:
    relative_parts = audio_path.relative_to(subset_root).parts
    if len(relative_parts) != 3:
        raise ValueError(f"Audio must be stored as <speaker>/<chapter>/<file>, got {audio_path}")
    utterance_id = audio_path.stem
    parts = utterance_id.split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected <speaker>-<chapter>-<utterance>.flac, got {audio_path}")
    speaker_id, chapter_id, sequence_id = parts
    if not all(is_ascii_decimal(part) for part in (speaker_id, chapter_id, sequence_id)) or len(sequence_id) != 4:
        raise ValueError(f"Non-numeric LibriSpeech utterance ID: {utterance_id}")
    if relative_parts != (speaker_id, chapter_id, f"{utterance_id}.flac"):
        raise ValueError(f"Audio relative path does not match utterance ID for {audio_path}")
    if audio_path.parent.name != chapter_id or audio_path.parent.parent.name != speaker_id:
        raise ValueError(f"Audio hierarchy does not match utterance ID for {audio_path}")
    if utterance_id not in transcripts:
        raise KeyError(f"Missing transcript for {utterance_id} ({audio_path})")
    return AudioIdentity(subset, speaker_id, chapter_id, utterance_id, audio_path, transcripts[utterance_id])


def inspect_audio(identity: AudioIdentity) -> tuple[AudioIdentity, int, int, int]:
    info = sf.info(identity.audio_path)
    return identity, int(info.samplerate), int(info.channels), int(info.frames)


def split_name(subset: str) -> str:
    if subset in TRAIN_SUBSETS:
        return "train"
    if subset in DEV_SUBSETS:
        return "dev"
    if subset.startswith("test-"):
        return "test"
    raise ValueError(f"Unknown LibriSpeech split semantics for subset {subset!r}")


def build_record(
    identity: AudioIdentity,
    dataset_root: Path,
    sample_rate: int,
    channels: int,
    num_samples: int,
    base_seed: int,
) -> dict[str, Any]:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} for {identity.audio_path}")
    if channels != 1:
        raise ValueError(f"Expected mono audio, got {channels} channels for {identity.audio_path}")
    if num_samples <= 0:
        raise ValueError(f"Empty audio file: {identity.audio_path}")

    latent_frames = expected_latent_frames(num_samples)
    cache_stem = f"{identity.subset}/{identity.speaker_id}/{identity.chapter_id}/{identity.utterance_id}.npy"
    return {
        "audio_relative_path": identity.audio_path.relative_to(dataset_root).as_posix(),
        "chapter_id": identity.chapter_id,
        "duration_seconds": num_samples / SAMPLE_RATE,
        "hubert_relative_path": f"hubert_40hz/{cache_stem}",
        "latent_dim": SEMANTIC_VAE_LATENT_DIM,
        "latent_frames": latent_frames,
        "latent_relative_path": f"latents/{cache_stem}",
        "num_channels": channels,
        "original_num_samples": num_samples,
        "padded_num_samples": latent_frames * SEMANTIC_VAE_HOP_LENGTH,
        "posterior_seed": stable_utterance_seed(base_seed, identity.utterance_key),
        "sample_rate": sample_rate,
        "speaker_id": identity.speaker_id,
        "split": split_name(identity.subset),
        "subset": identity.subset,
        "text": identity.transcript,
        "utterance_id": identity.utterance_id,
        "utterance_key": identity.utterance_key,
    }


def manifest_line_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def numeric_record_sort_key(record: dict[str, Any], subset_order: dict[str, int]) -> tuple[int, int, int, int]:
    sequence_id = record["utterance_id"].rsplit("-", maxsplit=1)[-1]
    return (
        subset_order[record["subset"]],
        int(record["speaker_id"]),
        int(record["chapter_id"]),
        int(sequence_id),
    )


def main() -> None:
    args = get_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if args.min_duration < 0 or args.max_duration <= args.min_duration:
        raise ValueError(f"Invalid duration interval: min={args.min_duration}, max={args.max_duration}")
    if args.workers < 1:
        raise ValueError(f"workers must be positive, got {args.workers}")
    if args.limit_per_subset is not None and args.limit_per_subset < 1:
        raise ValueError(f"limit-per-subset must be positive, got {args.limit_per_subset}")
    if len(set(args.subsets)) != len(args.subsets):
        raise ValueError(f"Duplicate subset in {args.subsets}")

    identities: list[AudioIdentity] = []
    subset_roots: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    transcript_counts: dict[str, int] = {}

    for subset in args.subsets:
        subset_root = resolve_subset_root(dataset_root, subset)
        subset_roots[subset] = subset_root.relative_to(dataset_root).as_posix()
        audio_paths = sorted(subset_root.rglob("*.flac"))
        transcripts = parse_transcripts(subset_root)
        source_counts[subset] = len(audio_paths)
        transcript_counts[subset] = len(transcripts)
        if len(audio_paths) != len(transcripts):
            raise ValueError(
                f"Audio/transcript count mismatch for {subset}: {len(audio_paths)} FLAC vs {len(transcripts)} transcripts"
            )
        audio_ids = {path.stem for path in audio_paths}
        transcript_ids = set(transcripts)
        if audio_ids != transcript_ids:
            missing_text = sorted(audio_ids - transcript_ids)[:10]
            missing_audio = sorted(transcript_ids - audio_ids)[:10]
            raise ValueError(
                f"Audio/transcript ID mismatch for {subset}: missing_text={missing_text}, missing_audio={missing_audio}"
            )
        if args.verify_official_counts and subset in OFFICIAL_UTTERANCE_COUNTS:
            expected_count = OFFICIAL_UTTERANCE_COUNTS[subset]
            if len(audio_paths) != expected_count:
                raise ValueError(f"Incomplete {subset}: expected {expected_count} FLAC files, found {len(audio_paths)}")
        if args.limit_per_subset is not None:
            audio_paths = audio_paths[: args.limit_per_subset]
        selected_counts[subset] = len(audio_paths)
        identities.extend(identify_audio(subset, subset_root, path, transcripts) for path in audio_paths)

    subset_order = {subset: index for index, subset in enumerate(args.subsets)}
    identities.sort(
        key=lambda item: (
            subset_order[item.subset],
            int(item.speaker_id),
            int(item.chapter_id),
            int(item.utterance_id.rsplit("-", maxsplit=1)[-1]),
        )
    )
    keys = [identity.utterance_key for identity in identities]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate utterance keys detected across selected subsets")
    utterance_ids = [identity.utterance_id for identity in identities]
    if len(utterance_ids) != len(set(utterance_ids)):
        raise ValueError("Duplicate utterance IDs detected across selected subsets")

    manifests_dir = output_root / "manifests"
    manifest_spec = {
        "base_posterior_seed": args.base_seed,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "duration_filter_seconds": {"maximum": args.max_duration, "minimum": args.min_duration},
        "limit_per_subset": args.limit_per_subset,
        "official_count_check": args.verify_official_counts,
        "selected_counts_before_duration_filter": selected_counts,
        "source_audio_counts": source_counts,
        "subset_roots": subset_roots,
        "subsets": args.subsets,
        "transcript_counts": transcript_counts,
    }
    spec_result = atomic_write_json(manifests_dir / "manifest_spec.json", manifest_spec)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        inspected = executor.map(inspect_audio, identities, chunksize=128)
        for identity, sample_rate, channels, num_samples in tqdm(
            inspected, total=len(identities), desc="Inspecting LibriSpeech"
        ):
            record = build_record(identity, dataset_root, sample_rate, channels, num_samples, args.base_seed)
            duration = record["duration_seconds"]
            if duration < args.min_duration or duration > args.max_duration:
                rejected.append(
                    {
                        "duration_seconds": duration,
                        "reason": "duration_out_of_range",
                        "subset": identity.subset,
                        "utterance_key": identity.utterance_key,
                    }
                )
            else:
                records.append(record)

    records.sort(key=lambda record: numeric_record_sort_key(record, subset_order))
    rejected.sort(key=lambda record: (subset_order[record["subset"]], record["utterance_key"]))
    train_records = [record for record in records if record["split"] == "train"]
    dev_records = [record for record in records if record["split"] == "dev"]

    outputs = {
        "inventory.jsonl": records,
        "train.jsonl": train_records,
        "dev.jsonl": dev_records,
        "rejected.jsonl": rejected,
    }
    write_results = {name: atomic_write_jsonl(manifests_dir / name, values) for name, values in outputs.items()}
    manifest_metadata = {
        name: {
            "count": manifest_line_count(result.path),
            "path": result.path.relative_to(output_root).as_posix(),
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
        }
        for name, result in write_results.items()
    }
    metadata = {
        "base_posterior_seed": args.base_seed,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "duration_filter_seconds": {"maximum": args.max_duration, "minimum": args.min_duration},
        "accepted_counts": {subset: sum(record["subset"] == subset for record in records) for subset in args.subsets},
        "latent_spec": {
            "dimension": SEMANTIC_VAE_LATENT_DIM,
            "dtype": "float32",
            "frame_rate_hz": SAMPLE_RATE / SEMANTIC_VAE_HOP_LENGTH,
            "hop_length_samples": SEMANTIC_VAE_HOP_LENGTH,
            "mode": "fixed_posterior_sample",
            "sample_rate": SAMPLE_RATE,
        },
        "limit_per_subset": args.limit_per_subset,
        "manifests": manifest_metadata,
        "manifest_spec": {
            "path": spec_result.path.relative_to(output_root).as_posix(),
            "sha256": spec_result.sha256,
            "size_bytes": spec_result.size_bytes,
        },
        "official_count_check": args.verify_official_counts,
        "rejected_counts": {subset: sum(record["subset"] == subset for record in rejected) for subset in args.subsets},
        "selected_counts_before_duration_filter": selected_counts,
        "source_audio_counts": source_counts,
        "subset_roots": subset_roots,
        "subsets": args.subsets,
        "total_accepted_duration_hours": sum(record["duration_seconds"] for record in records) / 3600,
        "transcript_counts": transcript_counts,
    }
    metadata_result = atomic_write_json(manifests_dir / "inventory_meta.json", metadata)

    summary = {
        "accepted": len(records),
        "created": {
            name: result.created
            for name, result in {
                **write_results,
                "inventory_meta.json": metadata_result,
                "manifest_spec.json": spec_result,
            }.items()
        },
        "dev": len(dev_records),
        "output_root": str(output_root),
        "rejected": len(rejected),
        "train": len(train_records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
