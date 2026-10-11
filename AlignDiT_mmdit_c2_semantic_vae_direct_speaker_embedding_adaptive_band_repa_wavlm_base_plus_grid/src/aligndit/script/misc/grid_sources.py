"""Audit the local GRID split and recover its original, synchronized audio.

The distributed audio_25k and 22050-Hz WAVs have silence removed.  Training must
use the complete MPG audio track on the 25-Hz video timeline instead.  Legacy
512-dimensional mouth embeddings are not AV-HuBERT features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

SAMPLE_RATE = 16000
VIDEO_FPS = 25
MAX_ALIGNMENT_ADJUSTMENT_SECONDS = 0.12
SPLIT_SUBDIR = "0_Grid_Wav_22050_Abs_Feature/Grid_Wav_22050_Abs_Feature"
GRAMMAR = (
    {"b": "bin", "l": "lay", "p": "place", "s": "set"},
    {"b": "blue", "g": "green", "r": "red", "w": "white"},
    {"a": "at", "b": "by", "i": "in", "w": "with"},
    dict(zip("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz")),
    dict(zip("z123456789", "zero one two three four five six seven eight nine".split())),
    {"a": "again", "n": "now", "p": "please", "s": "soon"},
)


class SourceValidationError(ValueError):
    """A source item cannot meet the documented GRID input contract."""


def executable(name: str, override: str | None = None) -> str:
    if override:
        candidate = Path(override)
        result = str(candidate) if candidate.is_file() else shutil.which(override)
    else:
        candidate = Path(sys.executable).resolve().parent / name
        result = str(candidate) if candidate.is_file() else shutil.which(name)
    if not result:
        raise FileNotFoundError(f"{name} executable is unavailable")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str, utterance_id: str | None = None) -> str:
    words = [w for w in text.lower().split() if w not in {"sp", "sil", "<sil>"}]
    if len(words) != 6 or any(re.fullmatch("[a-z]+", w) is None for w in words):
        raise SourceValidationError(f"Expected six spoken GRID words, got {words!r}")
    if utterance_id is not None:
        try:
            if len(utterance_id) != 6:
                raise KeyError(utterance_id)
            expected = [mapping[c] for mapping, c in zip(GRAMMAR, utterance_id)]
        except KeyError as exc:
            raise SourceValidationError(f"Invalid GRID utterance ID: {utterance_id}") from exc
        if words != expected:
            raise SourceValidationError(f"Transcript/ID mismatch: {words!r} != {expected!r}")
    return " ".join(words)


def mouth_frame_paths(mouth_dir: str | Path) -> list[Path]:
    directory = Path(mouth_dir)
    if not directory.is_dir():
        raise SourceValidationError(f"Missing mouth directory: {directory}")
    paths = [p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if any(not p.stem.isdigit() for p in paths):
        raise SourceValidationError(f"Non-numeric mouth frame name in {directory}")
    paths.sort(key=lambda p: int(p.stem))
    if len(paths) not in {74, 75}:
        raise SourceValidationError(f"Expected 74 or 75 mouth frames, found {len(paths)}")
    if [int(p.stem) for p in paths] != list(range(1, len(paths) + 1)):
        raise SourceValidationError("Mouth frame numbers must be unique and contiguous from 1")
    return paths


def _mouth_digest(paths: list[Path]) -> str:
    from PIL import Image

    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        # Bind ordering, names and contents without ambiguous concatenation.
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(payload).digest())
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if image.size != (96, 96) or image.mode != "L":
                    raise SourceValidationError(
                        f"Mouth frame must be grayscale 96x96: {path}, {image.mode}/{image.size}"
                    )
        except (OSError, ValueError) as exc:
            raise SourceValidationError(f"Invalid mouth frame {path}: {exc}") from exc
    return digest.hexdigest()


def decoder_diagnostic(message: str) -> str:
    """Remove process addresses so repeated source audits have stable reasons."""
    return re.sub(r"0x[0-9a-fA-F]+", "<address>", message.strip())[:1500]


def probe_video(path: str | Path, *, ffprobe: str | None = None, validate_video: bool = True) -> dict:
    command = [executable("ffprobe", ffprobe), "-v", "error"]
    if validate_video:
        command.append("-count_frames")
    command += ["-show_streams", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode or result.stderr.strip():
        raise SourceValidationError(f"MPG probe/decode failed: {decoder_diagnostic(result.stderr)}")
    try:
        streams = json.loads(result.stdout)["streams"]
        videos = [s for s in streams if s["codec_type"] == "video"]
        audios = [s for s in streams if s["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise SourceValidationError("Expected exactly one video and one audio stream")
        video, audio = videos[0], audios[0]
        fps = float(Fraction(video["avg_frame_rate"]))
        if not math.isclose(fps, VIDEO_FPS, abs_tol=0.001):
            raise SourceValidationError(f"Expected 25-Hz video, got {fps}")
        result = {
            "fps": VIDEO_FPS,
            "video_start_seconds": float(video["start_time"]),
            "audio_start_seconds": float(audio["start_time"]),
            "audio_duration_seconds": float(audio["duration"]),
            "video_duration_seconds": float(video["duration"]),
            "video_width": int(video["width"]),
            "video_height": int(video["height"]),
            "video_validation": "decoded_frame_count" if validate_video else "metadata_only",
        }
        if validate_video:
            result["decoded_video_frames"] = int(video["nb_read_frames"])
        if not all(math.isfinite(result[k]) for k in (
            "video_start_seconds", "audio_start_seconds", "audio_duration_seconds", "video_duration_seconds"
        )):
            raise SourceValidationError("Non-finite stream timing")
        return result
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, SourceValidationError):
            raise
        raise SourceValidationError(f"Incomplete or invalid MPG stream metadata: {exc}") from exc


def _audit_record(record: dict, ffprobe: str, validate_video: bool) -> dict:
    path = Path(record["video_path"])
    if not path.is_file():
        raise SourceValidationError(f"Missing MPG: {path}")
    frames = mouth_frame_paths(record["mouth_dir"])
    metadata = probe_video(path, ffprobe=ffprobe, validate_video=validate_video)
    count = len(frames)
    if validate_video and metadata["decoded_video_frames"] != count:
        raise SourceValidationError(
            f"Decoded video/mouth count mismatch: {metadata['decoded_video_frames']} != {count}"
        )
    duration = count / VIDEO_FPS
    offset = metadata["audio_start_seconds"] - metadata["video_start_seconds"]
    audio_end = offset + metadata["audio_duration_seconds"]
    bound = MAX_ALIGNMENT_ADJUSTMENT_SECONDS
    if abs(offset) > bound or abs(audio_end - duration) > bound:
        raise SourceValidationError(
            f"Audio/video timing mismatch exceeds {bound}s: offset={offset}, "
            f"audio_end={audio_end}, video_duration={duration}"
        )
    # MPEG-PS duration estimates can end at the last timestamped GOP.
    # The decoded frame count, rather than that estimate, defines video length.
    return {
        **record, **metadata, "frame_count": count, "duration": duration,
        "sample_rate": SAMPLE_RATE, "num_samples": count * (SAMPLE_RATE // VIDEO_FPS),
        "audio_offset_seconds": offset, "video_sha256": file_sha256(path),
        "mouth_sha256": _mouth_digest(frames), "mouth_frame_names": [p.name for p in frames],
        "audio_policy": "complete_mpg_audio_aligned_to_first_video_timestamp",
        "max_alignment_adjustment_seconds": bound,
    }


def build_source_records(
    dataset_root: str | Path,
    splits: tuple[str, ...] = ("train", "val"),
    *,
    limit: int | None = None,
    ffprobe: str | None = None,
    ffmpeg: str | None = None,
    workers: int = 8,
    validate_video: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """Return accepted records, explicit exclusions and split/protocol provenance.

    ``limit`` selects the first N input rows *per split* before auditing.  Full
    runs preserve every original row's split; they never reshuffle failed clips.
    All retained mouth images are decoded and hashed.  FFprobe's count_frames
    decodes the full video by default.  ``validate_video=False`` is diagnostic
    only and is clearly recorded in the manifest.
    """
    root = Path(dataset_root).resolve()
    implementation_sha256 = file_sha256(__file__)
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    if len(set(splits)) != len(splits) or any(s not in {"train", "val"} for s in splits):
        raise ValueError("splits must contain distinct train/val names")
    probe = executable("ffprobe", ffprobe)
    executable("ffmpeg", ffmpeg)  # Fail preflight before expensive corpus auditing.
    candidates, exclusions, split_info, seen = [], [], {}, set()
    for split in splits:
        source = root / SPLIT_SUBDIR / f"{split}.txt"
        lines = source.read_text(encoding="utf-8").splitlines()
        source_hash = file_sha256(source)
        split_info[split] = {
            "path": str(source), "sha256": source_hash, "original_rows": len(lines),
            "selected_rows": min(limit, len(lines)) if limit is not None else len(lines),
        }
        for line_number, line in enumerate(lines, start=1):
            fields = line.split("|")
            source_id = fields[0]
            if source_id in seen:
                raise SourceValidationError(f"Duplicate or split-leaked GRID ID: {source_id}")
            seen.add(source_id)
            if limit is not None and line_number > limit:
                continue
            record = {"source_id": source_id, "split": split, "source_line": line_number,
                      "source_split_sha256": source_hash}
            try:
                if len(fields) != 4:
                    raise SourceValidationError("Split row must contain id|speaker|phonemes|text")
                match = re.fullmatch(r"(s(?:[1-9]|[12][0-9]|3[0-4]))-([a-z0-9]{6})", source_id)
                if match is None or fields[1] != match[1]:
                    raise SourceValidationError(f"Invalid/mismatched GRID speaker and ID: {source_id}")
                speaker, utterance = match.groups()
                record.update({
                    "id": f"{speaker}/{utterance}", "speaker": speaker, "utterance_id": utterance,
                    "text": normalize_text(fields[3], utterance),
                    "video_path": str(root / "Grid_dataset_Raw" / "Grid_video" / speaker / f"{utterance}.mpg"),
                    "mouth_dir": str(root / "Lip_Grid_Gray" / "Face_my_Grid_Gray" / f"{speaker}-face-{utterance}"),
                })
                candidates.append(record)
            except (SourceValidationError, OSError) as exc:
                exclusions.append({**record, "reason": str(exc), "stage": "split_validation"})

    def audit(record: dict) -> tuple[dict | None, dict | None]:
        try:
            return _audit_record(record, probe, validate_video), None
        except (SourceValidationError, OSError, subprocess.TimeoutExpired) as exc:
            return None, {**record, "reason": str(exc), "stage": "source_audit"}

    accepted = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (good, bad) in enumerate(executor.map(audit, candidates), start=1):
            if good is not None:
                accepted.append(good)
            else:
                exclusions.append(bad)
            if index % 1000 == 0:
                print(f"GRID source audit: {index}/{len(candidates)}, accepted={len(accepted)}, excluded={len(exclusions)}", flush=True)
    summary = {
        "schema_version": 1, "dataset_root": str(root), "splits": split_info,
        "source_audit_implementation_sha256": implementation_sha256,
        "ffprobe_executable": probe,
        "accepted": dict(Counter(r["split"] for r in accepted)),
        "excluded": dict(Counter(r["split"] for r in exclusions)),
        "frame_counts": {str(k): v for k, v in Counter(r["frame_count"] for r in accepted).items()},
        "speakers": sorted({r["speaker"] for r in accepted}),
        "accepted_hours": sum(r["duration"] for r in accepted) / 3600,
        "limit_per_split": limit, "video_decode_validated": validate_video,
        "split_policy": "existing_local_train_val_files_without_reassignment",
        "audio_policy": "complete_mpg_audio_aligned_to_first_video_timestamp",
        "mouth_policy": "existing_96x96_grayscale_frames; original_crop_method_not_supplied",
        "legacy_wav_and_512d_features_used": False,
        "has_independent_test_split": False,
    }
    return accepted, exclusions, summary


def decode_aligned_audio(record: dict[str, Any], *, ffmpeg: str | None = None):
    """Decode complete MPG audio and align it to the first video timestamp."""
    import numpy as np

    result = subprocess.run(
        [executable("ffmpeg", ffmpeg), "-v", "error", "-nostdin", "-threads", "1",
         "-i", record["video_path"], "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=60, check=False,
    )
    if result.returncode or result.stderr.strip():
        raise SourceValidationError(f"MPG audio decode failed: {decoder_diagnostic(result.stderr.decode(errors='replace'))}")
    if len(result.stdout) % 4:
        raise SourceValidationError("Decoded float audio has an incomplete sample")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise SourceValidationError("Decoded audio is empty or non-finite")
    target = int(record["num_samples"])
    offset = round(float(record["audio_offset_seconds"]) * SAMPLE_RATE)
    bound = round(float(record.get("max_alignment_adjustment_seconds", MAX_ALIGNMENT_ADJUSTMENT_SECONDS)) * SAMPLE_RATE)
    if abs(offset) > bound or abs(len(audio) + offset - target) > bound:
        raise SourceValidationError(
            f"Decoded audio requires excessive alignment correction: samples={len(audio)}, "
            f"offset={offset}, expected={target}, bound={bound}"
        )
    if offset < 0:
        audio = audio[-offset:]
    elif offset > 0:
        audio = np.pad(audio, (offset, 0))
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return np.ascontiguousarray(audio[:target], dtype=np.float32)


def load_mouth_frames(record: dict[str, Any]):
    """Load the provided grayscale mouth frames in numeric time order."""
    import numpy as np
    from PIL import Image

    paths = mouth_frame_paths(record["mouth_dir"])
    if len(paths) != int(record["frame_count"]):
        raise SourceValidationError("Mouth frame count changed after source audit")
    images = []
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (96, 96):
                raise SourceValidationError(f"Mouth frame must be grayscale 96x96: {path}")
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser.add_argument("--dataset-root", type=Path, default=Path(f"{prefix}/zjw524/datasets/Grid_Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="First N original rows per split; smoke audit only")
    parser.add_argument("--ffprobe")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--no-decode-video", action="store_true", help="Diagnostic metadata audit, not a complete training audit")
    args = parser.parse_args()
    audit_files = [args.output_dir / name for name in (
        "source_records.jsonl", "source_exclusions.jsonl", "source_summary.json"
    )]
    if any(path.exists() for path in audit_files):
        parser.error("Audit outputs already exist; choose a new --output-dir to preserve their provenance")
    records, exclusions, summary = build_source_records(
        args.dataset_root, workers=args.workers, limit=args.limit,
        ffprobe=args.ffprobe, ffmpeg=args.ffmpeg, validate_video=not args.no_decode_video,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in (
        ("source_records.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)),
        ("source_exclusions.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in exclusions)),
        ("source_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"),
    ):
        destination = args.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    empty_splits = [split for split, info in summary["splits"].items()
                    if info["selected_rows"] and not summary["accepted"].get(split)]
    if empty_splits:
        parser.exit(1, f"No usable records in splits {empty_splits}; exclusions were written to {args.output_dir}\n")


if __name__ == "__main__":
    main()
