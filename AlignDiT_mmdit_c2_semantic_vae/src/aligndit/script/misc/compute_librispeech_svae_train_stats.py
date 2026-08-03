"""Compute immutable train-only per-channel statistics for raw Semantic-VAE latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from aligndit.script.misc.svae_cache_utils import (
    CACHE_SCHEMA_VERSION,
    SEMANTIC_VAE_LATENT_DIM,
    atomic_write_json,
    prefixed_path,
    read_jsonl,
    safe_join,
    sha256_file,
)


METHOD = "per_channel_population_mean_std_float64_welford_v1"
FEATURE = "semantic_vae_posterior_sample_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=prefixed_path("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_regular_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge one [time,channel] chunk into a float64 Welford accumulator."""

    chunk_count = int(chunk.shape[0])
    chunk64 = chunk.astype(np.float64)
    chunk_mean = chunk64.mean(axis=0, dtype=np.float64)
    centered = chunk64 - chunk_mean
    chunk_m2 = np.einsum("tc,tc->c", centered, centered, dtype=np.float64)
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    combined_count = count + chunk_count
    delta = chunk_mean - mean
    combined_mean = mean + delta * (chunk_count / combined_count)
    combined_m2 = m2 + chunk_m2 + np.square(delta) * (count * chunk_count / combined_count)
    return combined_count, combined_mean, combined_m2


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve(strict=True)
    manifest = (args.manifest or cache_root / "manifests/train.jsonl").resolve(strict=True)
    output = args.output or cache_root / "state/latents/train_normalization.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"Training manifest must be a regular file: {manifest}")

    inventory_meta = load_regular_json(manifest.parent / "inventory_meta.json")
    manifest_entry = inventory_meta.get("manifests", {}).get(manifest.name)
    if not isinstance(manifest_entry, dict):
        raise TypeError(f"Training manifest is not registered in inventory metadata: {manifest}")
    manifest_sha256 = sha256_file(manifest)
    if manifest_entry.get("sha256") != manifest_sha256:
        raise RuntimeError("Training manifest SHA256 differs from inventory metadata")

    latent_complete_path = safe_join(cache_root, "state/latents/complete.json")
    latent_complete = load_regular_json(latent_complete_path)
    inventory_count = inventory_meta.get("manifests", {}).get("inventory.jsonl", {}).get("count")
    if (
        latent_complete.get("feature") != FEATURE
        or latent_complete.get("selection") != {"mode": "full"}
        or latent_complete.get("count") != inventory_count
    ):
        raise RuntimeError("Latent completion marker is not the authoritative full cache")
    latent_complete_sha256 = sha256_file(latent_complete_path)

    count = 0
    record_count = 0
    mean = np.zeros(SEMANTIC_VAE_LATENT_DIM, dtype=np.float64)
    m2 = np.zeros(SEMANTIC_VAE_LATENT_DIM, dtype=np.float64)
    seen: set[str] = set()
    records = read_jsonl(manifest)
    for record in tqdm(records, total=int(manifest_entry["count"]), desc="Semantic-VAE train statistics"):
        key = record.get("utterance_key")
        if record.get("split") != "train" or not isinstance(key, str) or key in seen:
            raise ValueError(f"Invalid or duplicate training record: {key!r}")
        seen.add(key)
        frames = int(record["latent_frames"])
        latent_path = safe_join(cache_root, record["latent_relative_path"])
        if not latent_path.is_file() or latent_path.is_symlink():
            raise FileNotFoundError(f"Missing regular latent file: {latent_path}")
        latent = np.load(latent_path, allow_pickle=False)
        if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(f"Invalid latent for {key}: shape={latent.shape}, dtype={latent.dtype}")
        if not np.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite latent for {key}")
        count, mean, m2 = merge_moments(count, mean, m2, latent)
        record_count += 1

    if record_count != int(manifest_entry["count"]):
        raise RuntimeError(f"Processed {record_count} records, expected {manifest_entry['count']}")
    if count <= 0 or not np.isfinite(mean).all() or not np.isfinite(m2).all() or (m2 <= 0).any():
        raise FloatingPointError("Invalid final Semantic-VAE latent moments")
    std = np.sqrt(m2 / count)
    result = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "channel_count": SEMANTIC_VAE_LATENT_DIM,
        "count": record_count,
        "feature": FEATURE,
        "frame_count": count,
        "latent_complete_sha256": latent_complete_sha256,
        "mean": mean.tolist(),
        "method": METHOD,
        "scope": "train",
        "std": std.tolist(),
        "train_manifest_sha256": manifest_sha256,
    }
    published = atomic_write_json(output, result)
    print(
        f"Semantic-VAE train statistics {'created' if published.created else 'verified'}: "
        f"{published.path} sha256={published.sha256} frames={count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
