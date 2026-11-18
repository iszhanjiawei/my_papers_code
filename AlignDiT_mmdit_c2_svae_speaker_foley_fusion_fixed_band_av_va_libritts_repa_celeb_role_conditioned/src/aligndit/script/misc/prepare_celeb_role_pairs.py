"""Build reproducible cross-utterance training pairs from unlabeled CelebV-Dub clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from aligndit.model.semantic_vae_dataset import CELEBVDUB_TRAIN_COUNT, sha256_file
from aligndit.model.speaker_embedding import validate_speaker_cache_metadata, validate_speaker_embedding_array


TRAIN_SHA256 = "0d16d5c8f00eb25ee51c7de604299a37cace1bc0e65b7127a45420c433b4d395"
CAMPLUS_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
CAMPLUS_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
SEED = 666
MIN_SECONDS, MAX_SECONDS = 2.0, 10.0
MIN_COSINE, MAX_TEXT_MATCH = 0.65, 0.8
SETTINGS = {
    "group": "video_id and clip stem before final underscore; candidate pool only",
    "seed": SEED,
    "selection": "lowest SHA256 of seed NUL target_key NUL reference_key among eligible references",
    "reference_min_seconds": MIN_SECONDS,
    "reference_max_seconds": MAX_SECONDS,
    "min_cosine": MIN_COSINE,
    "reject_identical_embeddings": True,
    "text_normalization": "NFKC, casefold, remove apostrophes, tokenize Unicode letters/digits",
    "reject_text_overlap": "identical, contiguous containment, or longest common contiguous run covering >=0.8 "
    "of the shorter (minimum 3 words), or token SequenceMatcher ratio >=0.8",
    "maximum_text_match": MAX_TEXT_MATCH,
}


@dataclass(frozen=True)
class Clip:
    key: str
    group: tuple[str, str]
    seconds: float
    words: tuple[str, ...]


def parse_clip(record):
    key = record["utterance_key"]
    parts = key.split("/")
    if (
        record.get("split") != "train"
        or len(parts) != 4
        or parts[:2] != ["celebvdub", "train"]
        or parts[2] != record["video_id"]
        or any(part in {"", ".", ".."} for part in parts)
        or record["audio_relative_path"] != "/".join(parts[1:]) + ".wav"
    ):
        raise ValueError(f"Invalid training source identity for {key!r}")
    group_prefix, sep, suffix = parts[3].rpartition("_")
    if not sep or not group_prefix or not suffix:
        raise ValueError(f"Missing clip pairing prefix: {key}")
    seconds = record["duration_seconds"]
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid duration for {key!r}")
    text = unicodedata.normalize("NFKC", record["text"]).casefold().replace("’", "").replace("'", "")
    words = tuple(re.findall(r"[^\W_]+", text))
    return Clip(key, (parts[2], group_prefix), seconds, words)


def overlapping_text(a, b):
    if not a or not b:
        return True
    if len(a) > len(b):
        a, b = b, a
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    longest = matcher.find_longest_match().size
    return (
        longest == len(a) or (longest >= 3 and longest >= MAX_TEXT_MATCH * len(a)) or matcher.ratio() >= MAX_TEXT_MATCH
    )


def reference_priority(target_key, ref_key):
    return hashlib.sha256(f"{SEED}\0{target_key}\0{ref_key}".encode()).digest()


def pair_group(clips, embeddings):
    """Select independently for each target. A reference may serve multiple targets."""
    matrix = np.stack([embeddings[c.key] for c in clips]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cosines = (matrix @ matrix.T).clip(-1.0, 1.0)
    pairs, exclusions = [], []
    for i, target in enumerate(clips):
        candidates = [j for j in range(len(clips)) if j != i]
        reason = "no_distinct_clip_in_prefix"
        if candidates:
            candidates = [j for j in candidates if MIN_SECONDS <= clips[j].seconds <= MAX_SECONDS]
            reason = "no_reference_in_duration_range"
        if candidates:
            candidates = [j for j in candidates if cosines[i, j] >= MIN_COSINE]
            reason = "no_reference_passing_speaker_cosine"
        if candidates:
            candidates = [j for j in candidates if not np.array_equal(matrix[i], matrix[j])]
            reason = "only_identical_speaker_vectors"
        if candidates:
            reason = "overlapping_or_empty_transcript"
        for j in sorted(candidates, key=lambda j: reference_priority(target.key, clips[j].key)):
            ref = clips[j]
            if not overlapping_text(target.words, ref.words):
                pairs.append(
                    {
                        "target_key": target.key,
                        "reference_key": ref.key,
                        "cosine": float(cosines[i, j]),
                        "reference_seconds": ref.seconds,
                    }
                )
                break
        else:
            exclusions.append({"target_key": target.key, "reason": reason})
    return pairs, exclusions


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda x: x["target_key"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def distribution(values):
    quantiles = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(zip(("min", "p05", "p50", "p95", "max"), map(float, quantiles)))


def write_bundle(data_root, temp_dir, workers):
    if workers <= 0:
        raise ValueError("--workers must be positive")
    manifest = data_root / "CelebVDub_svae1000k_sample_seed666_fp32/manifests/train.jsonl"
    cache = data_root / "CelebVDub/campplus_spk_emb_zh_en_16k"
    manifest_sha = sha256_file(manifest)
    if manifest_sha != TRAIN_SHA256:
        raise ValueError("Expected the pinned full CelebV-Dub train manifest")
    meta = validate_speaker_cache_metadata(cache, model_id=CAMPLUS_ID, checkpoint_sha256=CAMPLUS_SHA256)
    coverage = json.loads((cache / "coverage_report.json").read_text())
    if coverage["split_counts"] != {"train": CELEBVDUB_TRAIN_COUNT, "test": 213}:
        raise ValueError("Unexpected CAM++ cache split counts")
    with manifest.open(encoding="utf-8") as stream:
        clips = sorted((parse_clip(json.loads(line)) for line in stream), key=lambda c: c.key)
    if len(clips) != CELEBVDUB_TRAIN_COUNT or len({c.key for c in clips}) != len(clips):
        raise ValueError("Invalid training inventory coverage")
    groups = defaultdict(list)
    for clip in clips:
        groups[clip.group].append(clip)

    def load_embedding(clip):
        relative = clip.key.removeprefix("celebvdub/") + ".npy"
        vector = np.load(cache / relative, allow_pickle=False)
        validate_speaker_embedding_array(vector, source=relative)
        return vector

    cache_digest = hashlib.sha256()
    embeddings = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound in-flight filesystem operations and memory on shared storage.
        for start in range(0, len(clips), 1024):
            chunk = clips[start : start + 1024]
            for clip, vector in zip(chunk, pool.map(load_embedding, chunk)):
                embeddings[clip.key] = vector
                cache_digest.update(clip.key.encode("utf-8") + b"\0")
                cache_digest.update(vector.astype("<f4", copy=False).tobytes())

    pairs, exclusions = [], []
    for members in groups.values():
        accepted, rejected = pair_group(members, embeddings)
        pairs.extend(accepted)
        exclusions.extend(rejected)
    if not pairs or len(pairs) + len(exclusions) != len(clips):
        raise RuntimeError("Invalid train pair coverage")
    write_rows(temp_dir / "pairs.jsonl", pairs)
    write_rows(temp_dir / "exclusions.jsonl", exclusions)
    summary = {
        "schema_version": 1,
        "train_count": len(clips),
        "train_manifest_sha256": manifest_sha,
        "pairs_sha256": sha256_file(temp_dir / "pairs.jsonl"),
        "exclusions_sha256": sha256_file(temp_dir / "exclusions.jsonl"),
        "accepted": len(pairs),
        "excluded": len(exclusions),
        "unique_references": len({p["reference_key"] for p in pairs}),
        "source_videos": len({c.group[0] for c in clips}),
        "candidate_prefixes": len(groups),
        "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
        "cosine": distribution([p["cosine"] for p in pairs]),
        "reference_seconds": distribution([p["reference_seconds"] for p in pairs]),
        "identity_status": "pairwise CAM++ screening, no clustering or human-verified speaker IDs",
        "overlap_status": "distinct clips and screened transcripts; no source-time or waveform-overlap proof",
        "settings": SETTINGS,
        "provenance": {
            "manifest": str(manifest),
            "camplus_model_id": meta["model_id"],
            "camplus_checkpoint_sha256": meta["checkpoint_sha256"],
            "cam_metadata_sha256": sha256_file(cache / "metadata.json"),
            "camplus_coverage_sha256": sha256_file(cache / "coverage_report.json"),
            "camplus_train_vectors_sha256": cache_digest.hexdigest(),
            "camplus_vector_hash_format": "sorted training key UTF8, NUL, 192 little-endian float32 values",
        },
    }
    (temp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def prepare(data_root, destination, workers):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        summary = write_bundle(data_root, staging, workers)
        try:
            staging.rename(destination)
        except OSError:
            # Concurrent/repeated build is valid only if all three results match.
            if not destination.is_dir() or any(
                not (destination / path.name).is_file() or sha256_file(destination / path.name) != sha256_file(path)
                for path in staging.iterdir()
            ):
                raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects/data")
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    destination = args.output_dir or args.data_root / "CelebVDub_role_pairs_v1"
    summary = prepare(args.data_root, destination, args.workers)
    print(
        json.dumps(
            {
                "directory": str(destination),
                "accepted": summary["accepted"],
                "excluded": summary["excluded"],
                "pairs_sha256": summary["pairs_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
