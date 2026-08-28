"""Extract deterministic raw MingTok acoustic latents for CelebV-Dub.

Cache layout::

    <cache_dir>/latents/<split>/<speaker>/<clip>.npy  # FP32 [T,64]
    <cache_dir>/shards/<split>.rank-xxxxx-of-xxxxx.jsonl
    <cache_dir>/manifest.jsonl
    <cache_dir>/contract.json

Each posterior sample has a stable path-derived seed, so resuming on a
different rank or with a different process order produces the same target.
Writes use ``fsync`` plus ``os.replace``; an interrupted job can simply be run
again and already-valid arrays are reused.

Formal runs select their exact ordered 79,613-item train set from the C2
``CelebVDub_char/raw.arrow`` ``audio_path`` column. Recursive filesystem
discovery is available only as an explicit option for isolated smoke fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from aligndit.model.mingtok_codec import (
    MINGTOK_HOP_SIZE,
    MINGTOK_LATENT_DIM,
    MINGTOK_LATENT_FPS,
    MINGTOK_SAMPLE_RATE,
    MingTokAcousticCodec,
    checkpoint_contract,
    stable_sample_seed,
)


LOGGER = logging.getLogger("extract_mingtok_latents")
SCHEMA_VERSION = 1


def _rooted(path: str) -> str:
    prefix = os.environ.get("ROOT_PREFIX", "")
    return prefix.rstrip("/") + path if prefix else path


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_audio_from_filesystem(audio_root: Path, split: str, extension: str) -> list[Path]:
    split_root = audio_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"CelebV-Dub split not found: {split_root}")
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    paths = sorted(path for path in split_root.rglob(f"*{normalized_extension}") if path.is_file())
    if not paths:
        raise RuntimeError(f"No {normalized_extension} files found under {split_root}")
    return paths


def _metadata_relative_audio(raw_audio_path: str, split: str, extension: str) -> Path:
    """Map a metadata path to its logical path below CelebVDub/audio/.

    The returned path is not resolved, so the cache mirrors the C2-visible
    symlink path instead of the underlying ``/zjw524/datasets`` target.
    """

    if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
        raise RuntimeError(f"Invalid metadata audio_path: {raw_audio_path!r}")
    parts = PurePosixPath(raw_audio_path.replace("\\", "/")).parts
    marker_index = None
    for index in range(len(parts) - 1):
        if parts[index] == "CelebVDub" and parts[index + 1] == "audio":
            marker_index = index + 2
            break
    if marker_index is None:
        raise RuntimeError(f"audio_path does not contain CelebVDub/audio/: {raw_audio_path}")
    relative_parts = parts[marker_index:]
    if len(relative_parts) < 3 or relative_parts[0] != split:
        raise RuntimeError(
            f"Metadata audio_path must select split={split!r} and include speaker/file: {raw_audio_path}"
        )
    if any(part in {"", ".", ".."} for part in relative_parts):
        raise RuntimeError(f"Unsafe metadata audio_path: {raw_audio_path}")
    relative = Path(*relative_parts)
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    if relative.suffix.lower() != normalized_extension.lower():
        raise RuntimeError(
            f"Metadata audio suffix mismatch: expected {normalized_extension}, got {relative.suffix}: {raw_audio_path}"
        )
    return relative


def _discover_audio_from_metadata(
    audio_root: Path,
    metadata_path: Path,
    split: str,
    extension: str,
    check_source_files: bool,
) -> tuple[list[Path], dict]:
    if split != "train":
        raise RuntimeError(f"The C2 CelebVDub_char/raw.arrow selection is train-only, got split={split!r}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"C2 raw.arrow not found: {metadata_path}")

    from datasets import Dataset

    dataset = Dataset.from_file(str(metadata_path))
    if "audio_path" not in dataset.column_names:
        raise RuntimeError(f"raw.arrow has no audio_path column: {metadata_path}")
    paths: list[Path] = []
    logical_keys: set[str] = set()
    duplicates: list[str] = []
    missing: list[str] = []
    for raw_audio_path in dataset["audio_path"]:
        relative = _metadata_relative_audio(raw_audio_path, split, extension)
        logical_key = relative.as_posix()
        if logical_key in logical_keys:
            duplicates.append(logical_key)
            continue
        logical_keys.add(logical_key)
        # Deliberately do not call resolve(): this remains the CelebVDub/audio
        # symlink path recorded by C2 metadata.
        audio_path = audio_root / relative
        if check_source_files and not audio_path.is_file():
            missing.append(str(audio_path))
        paths.append(audio_path)
    if duplicates:
        raise RuntimeError(f"Duplicate logical audio_path values in raw.arrow: {duplicates[:8]}")
    if missing:
        raise RuntimeError(f"Missing/broken metadata-selected audio files: {missing[:8]} (total={len(missing)})")
    selection = {
        "type": "raw_arrow",
        "metadata_sha256": _sha256_file(metadata_path),
        "audio_path_field": "audio_path",
        "ordering": "metadata_row_order",
    }
    return paths, selection


def _select_audio(args: argparse.Namespace, audio_root: Path) -> tuple[list[Path], dict]:
    if args.selection == "metadata":
        paths, selection = _discover_audio_from_metadata(
            audio_root=audio_root,
            metadata_path=Path(args.metadata_path).expanduser().absolute(),
            split=args.split,
            extension=args.extension,
            check_source_files=not args.skip_source_file_check,
        )
    elif args.selection == "filesystem":
        paths = _discover_audio_from_filesystem(audio_root, args.split, args.extension)
        selection = {"type": "filesystem", "ordering": "sorted_logical_path"}
    else:  # argparse prevents this; keep library calls defensive.
        raise ValueError(f"Unknown selection mode: {args.selection!r}")
    _validate_count(paths, args.expected_count)
    return paths, selection


def _validate_count(paths: list[Path], expected_count: int) -> None:
    if expected_count >= 0 and len(paths) != expected_count:
        raise RuntimeError(
            f"CelebV-Dub count mismatch: expected {expected_count:,}, found {len(paths):,}. "
            "Refusing to create a partial cache."
        )


def _relative_audio_key(audio_path: Path, audio_root: Path) -> str:
    try:
        return audio_path.relative_to(audio_root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Audio path is outside audio_root: {audio_path} vs {audio_root}") from error


def _relative_latent_key(relative_audio: str) -> str:
    return (Path("latents") / Path(relative_audio).with_suffix(".npy")).as_posix()


def _load_mono_16k(path: Path) -> tuple[torch.Tensor, int]:
    waveform, source_sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise RuntimeError(f"Invalid audio tensor {tuple(waveform.shape)}: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != MINGTOK_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, MINGTOK_SAMPLE_RATE)
    waveform = waveform.to(dtype=torch.float32).contiguous()
    if waveform.shape[0] != 1 or waveform.shape[-1] <= 0:
        raise RuntimeError(f"Failed to obtain non-empty mono audio: {path}")
    return waveform, int(source_sample_rate)


def _validate_latent_file(path: Path, frames: int, check_values: bool = True) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(np.float32):
            return False
        if tuple(array.shape) != (frames, MINGTOK_LATENT_DIM):
            return False
        return not check_values or bool(np.isfinite(array).all())
    except (OSError, ValueError, EOFError):
        return False


def _make_contract(
    args: argparse.Namespace,
    item_count: int,
    checkpoint: dict[str, str],
    selection: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "codec": "MingTok-Audio",
        "latent_dim": MINGTOK_LATENT_DIM,
        "dtype": "float32",
        "layout": "T,D",
        "latent_fps": MINGTOK_LATENT_FPS,
        "sample_rate": MINGTOK_SAMPLE_RATE,
        "hop_size": MINGTOK_HOP_SIZE,
        "audio_video_ratio": 2,
        "normalization": "none",
        "posterior_mode": args.posterior_mode,
        "base_seed": int(args.base_seed),
        "split": args.split,
        "num_items": item_count,
        "selection": selection,
        "checkpoint": checkpoint,
    }


def _manifest_path(cache_dir: Path, split: str, rank: int, nshard: int) -> Path:
    return cache_dir / "shards" / f"{split}.rank-{rank:05d}-of-{nshard:05d}.jsonl"


def _process_shard(args: argparse.Namespace) -> None:
    audio_root = Path(args.audio_root).expanduser().absolute()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    all_paths, selection = _select_audio(args, audio_root)
    if args.nshard <= 0 or not 0 <= args.rank < args.nshard:
        raise ValueError(f"rank must satisfy 0 <= rank < nshard, got {args.rank}/{args.nshard}")
    shard_paths = all_paths[args.rank :: args.nshard]
    if not shard_paths:
        raise RuntimeError(f"Shard {args.rank}/{args.nshard} is empty")

    codec = MingTokAcousticCodec(
        repo_path=args.repo_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        load_encoder=True,
        load_decoder=False,
    )
    contract = _make_contract(args, len(all_paths), codec.checkpoint, selection)
    final_manifest = _manifest_path(cache_dir, args.split, args.rank, args.nshard)
    partial_manifest = final_manifest.with_suffix(final_manifest.suffix + ".partial")
    partial_manifest.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "rank %d/%d: %d of %d files, mode=%s, device=%s, backend=%s",
        args.rank,
        args.nshard,
        len(shard_paths),
        len(all_paths),
        args.posterior_mode,
        args.device,
        args.backend,
    )
    failures = 0
    written = 0
    reused = 0
    with partial_manifest.open("w", encoding="utf-8") as manifest:
        header = {
            "type": "header",
            "schema_version": SCHEMA_VERSION,
            "rank": args.rank,
            "nshard": args.nshard,
            "contract": contract,
        }
        manifest.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        manifest.flush()

        iterator = tqdm(shard_paths, desc=f"MingTok rank {args.rank}", dynamic_ncols=True)
        for item_index, audio_path in enumerate(iterator):
            relative_audio = _relative_audio_key(audio_path, audio_root)
            relative_latent = _relative_latent_key(relative_audio)
            latent_path = cache_dir / relative_latent
            seed = stable_sample_seed(relative_audio, args.base_seed)
            try:
                waveform, source_sample_rate = _load_mono_16k(audio_path)
                num_samples = int(waveform.shape[-1])
                frames = (num_samples + MINGTOK_HOP_SIZE - 1) // MINGTOK_HOP_SIZE
                if not args.overwrite and latent_path.is_file() and _validate_latent_file(latent_path, frames):
                    status = "existing"
                    reused += 1
                else:
                    latent, frame_lengths = codec.encode(
                        waveform,
                        torch.tensor([num_samples], dtype=torch.long),
                        mode=args.posterior_mode,
                        seeds=[seed] if args.posterior_mode == "sample" else None,
                    )
                    returned_frames = int(frame_lengths[0].item())
                    if returned_frames != frames or latent.shape[1] < frames:
                        raise RuntimeError(
                            f"Frame mismatch for {relative_audio}: expected {frames}, "
                            f"returned {returned_frames}/{latent.shape[1]}"
                        )
                    array = latent[0, :frames].float().cpu().numpy()
                    if tuple(array.shape) != (frames, MINGTOK_LATENT_DIM):
                        raise RuntimeError(f"Invalid latent shape {array.shape} for {relative_audio}")
                    if not bool(np.isfinite(array).all()):
                        raise RuntimeError(f"Non-finite latent values for {relative_audio}")
                    _atomic_npy(latent_path, array)
                    status = "written"
                    written += 1

                record = {
                    "type": "item",
                    "relative_audio": relative_audio,
                    "relative_latent": relative_latent,
                    "num_samples": num_samples,
                    "frames": frames,
                    "latent_dim": MINGTOK_LATENT_DIM,
                    "dtype": "float32",
                    "seed": seed if args.posterior_mode == "sample" else None,
                    "source_sample_rate": source_sample_rate,
                    "status": status,
                }
            except Exception as error:  # keep the other 79k items recoverable
                failures += 1
                LOGGER.exception("Failed to extract %s", audio_path)
                record = {
                    "type": "error",
                    "relative_audio": relative_audio,
                    "relative_latent": relative_latent,
                    "error": f"{type(error).__name__}: {error}",
                }
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            if (item_index + 1) % args.manifest_flush_every == 0:
                manifest.flush()

        summary = {
            "type": "summary",
            "rank": args.rank,
            "nshard": args.nshard,
            "items": len(shard_paths),
            "written": written,
            "existing": reused,
            "errors": failures,
        }
        manifest.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
        manifest.flush()
        os.fsync(manifest.fileno())

    if failures:
        raise RuntimeError(
            f"Shard {args.rank}/{args.nshard} finished with {failures} error(s); "
            f"partial manifest retained at {partial_manifest}. Re-run the same command to resume."
        )
    os.replace(partial_manifest, final_manifest)
    LOGGER.info("Shard complete: %s (written=%d, existing=%d)", final_manifest, written, reused)


def _read_shard_manifest(path: Path) -> tuple[dict, list[dict], dict]:
    header = None
    summary = None
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSON at {path}:{line_number}") from error
            record_type = record.get("type")
            if record_type == "header":
                if header is not None:
                    raise RuntimeError(f"Duplicate header in {path}")
                header = record
            elif record_type == "item":
                items.append(record)
            elif record_type == "summary":
                summary = record
            elif record_type == "error":
                raise RuntimeError(f"Error record remains in completed shard manifest {path}: {record}")
            else:
                raise RuntimeError(f"Unknown record type in {path}:{line_number}: {record_type!r}")
    if header is None or summary is None:
        raise RuntimeError(f"Incomplete shard manifest: {path}")
    if summary.get("errors") != 0 or summary.get("items") != len(items):
        raise RuntimeError(f"Invalid shard summary in {path}: {summary}, records={len(items)}")
    return header, items, summary


def _merge_and_validate(args: argparse.Namespace) -> None:
    audio_root = Path(args.audio_root).expanduser().absolute()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    all_paths, selection = _select_audio(args, audio_root)
    checkpoint = checkpoint_contract(args.checkpoint_dir)
    expected_contract = _make_contract(args, len(all_paths), checkpoint, selection)

    records_by_audio: dict[str, dict] = {}
    for rank in range(args.nshard):
        path = _manifest_path(cache_dir, args.split, rank, args.nshard)
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed shard manifest: {path}")
        header, items, _ = _read_shard_manifest(path)
        if header.get("rank") != rank or header.get("nshard") != args.nshard:
            raise RuntimeError(f"Shard identity mismatch in {path}: {header}")
        if header.get("contract") != expected_contract:
            raise RuntimeError(f"Cache contract mismatch in {path}")
        for record in items:
            key = record["relative_audio"]
            if key in records_by_audio:
                raise RuntimeError(f"Duplicate audio in shard manifests: {key}")
            records_by_audio[key] = record

    expected_keys = [_relative_audio_key(path, audio_root) for path in all_paths]
    missing = sorted(set(expected_keys) - set(records_by_audio))
    extra = sorted(set(records_by_audio) - set(expected_keys))
    if missing or extra:
        raise RuntimeError(f"Manifest coverage mismatch: missing={missing[:8]}, extra={extra[:8]}")

    merged_records = []
    iterator = tqdm(expected_keys, desc="Validate MingTok cache", dynamic_ncols=True)
    for relative_audio in iterator:
        record = dict(records_by_audio[relative_audio])
        expected_relative_latent = _relative_latent_key(relative_audio)
        if record.get("relative_latent") != expected_relative_latent:
            raise RuntimeError(f"Latent path mismatch for {relative_audio}: {record.get('relative_latent')}")
        latent_path = cache_dir / expected_relative_latent
        if not _validate_latent_file(
            latent_path,
            int(record["frames"]),
            check_values=not args.skip_value_validation,
        ):
            raise RuntimeError(f"Invalid cached latent: {latent_path}")
        record.pop("type", None)
        record.pop("status", None)
        merged_records.append(record)

    existing_contract_path = cache_dir / "contract.json"
    if existing_contract_path.is_file():
        with existing_contract_path.open("r", encoding="utf-8") as handle:
            existing_contract = json.load(handle)
        if existing_contract != expected_contract:
            raise RuntimeError(
                f"Existing cache contract differs from this extraction: {existing_contract_path}. "
                "Use a separate cache directory for a different posterior/checkpoint."
            )

    _write_jsonl_atomic(cache_dir / "manifest.jsonl", merged_records)
    _atomic_json(existing_contract_path, expected_contract)
    LOGGER.info(
        "Validated and merged %d raw FP32 [T,64] latents: %s",
        len(merged_records),
        cache_dir / "manifest.jsonl",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-root",
        default=_rooted("/zjw524/projects/data/CelebVDub/audio"),
        help="CelebV-Dub audio root containing <split>/<speaker>/*.wav.",
    )
    parser.add_argument(
        "--selection",
        choices=("metadata", "filesystem"),
        default="metadata",
        help="Formal extraction must use metadata; filesystem is only for isolated smoke fixtures.",
    )
    parser.add_argument(
        "--metadata-path",
        default=_rooted("/zjw524/projects/data/CelebVDub_char/raw.arrow"),
        help="C2 raw.arrow whose audio_path column defines the exact 79,613-item train set.",
    )
    parser.add_argument(
        "--cache-dir",
        default=_rooted("/zjw524/projects/data/CelebVDub_mingtok_acoustic_64d_sample_seed666_fp32"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--extension", default=".wav")
    parser.add_argument("--expected-count", type=int, default=79_613)
    parser.add_argument(
        "--repo-path",
        default=_rooted("/zjw524/projects/alignDiT_idea6/MingTok-VAE/paper_code/MingTok-Audio"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=_rooted("/zjw524/projects/alignDiT_idea6/MingTok-VAE/checkpoint/MingTok-Audio"),
    )
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--posterior-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--base-seed", type=int, default=666)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even when a valid cache file exists.")
    parser.add_argument("--merge-only", action="store_true", help="Validate shards and write manifest/contract only.")
    parser.add_argument(
        "--validate-selection-only",
        action="store_true",
        help="Validate metadata count, split, suffix, uniqueness and file existence without loading MingTok.",
    )
    parser.add_argument(
        "--skip-source-file-check",
        action="store_true",
        help="Skip the expensive per-file stat after a launcher-level metadata preflight.",
    )
    parser.add_argument(
        "--skip-value-validation",
        action="store_true",
        help="During merge, validate dtype/shape but do not scan every value for NaN/Inf.",
    )
    parser.add_argument("--manifest-flush-every", type=int, default=32)
    return parser


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = _build_parser().parse_args()
    if args.manifest_flush_every <= 0:
        raise ValueError("--manifest-flush-every must be positive")
    if args.validate_selection_only:
        # A validation-only invocation is the one authoritative full source
        # preflight, even if a caller accidentally passes the rank fast-path.
        args.skip_source_file_check = False
        audio_root = Path(args.audio_root).expanduser().absolute()
        paths, selection = _select_audio(args, audio_root)
        LOGGER.info(
            "Validated %d C2-selected files for split=%s via %s; first=%s; last=%s",
            len(paths),
            args.split,
            selection["type"],
            paths[0],
            paths[-1],
        )
    elif args.merge_only:
        _merge_and_validate(args)
    else:
        _process_shard(args)


if __name__ == "__main__":
    main()
