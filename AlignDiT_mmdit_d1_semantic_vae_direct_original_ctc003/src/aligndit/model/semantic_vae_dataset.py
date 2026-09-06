"""CelebV-Dub training data backed by fixed Semantic-VAE and 40 Hz video caches."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SEMANTIC_VAE_LATENT_DIM = 64
CELEBVDUB_VIDEO_DIM = 1024
CELEBVDUB_TRAIN_COUNT = 79_613
CELEBVDUB_INVENTORY_COUNT = 79_826
CELEBVDUB_CTC40_FEASIBLE_COUNT = 79_508
CELEBVDUB_CTC40_INFEASIBLE_COUNT = 105
SEMANTIC_VAE_FEATURE = "semantic_vae_posterior_sample_v1"
VIDEO_40HZ_FEATURE = "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    return candidate.resolve(strict=True)


def _read_json(path: Path, *, label: str) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record {line_number} is not an object: {path}")
            records.append(value)
    return records


def _safe_join(root: Path, relative_path: str, *, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Invalid {label} relative path: {relative_path!r}")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes cache root: {relative_path!r}") from error
    return candidate


def _load_vocab(path: Path) -> tuple[dict[str, int], str]:
    characters = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not characters or len(characters) != len(set(characters)):
        raise ValueError(f"Vocabulary is empty or contains duplicate entries: {path}")
    vocabulary = {character: index for index, character in enumerate(characters)}
    if vocabulary.get(" ") != 0:
        raise ValueError(f"CelebV-Dub vocabulary must map space/unknown to id 0: {path}")
    return vocabulary, sha256_file(path)


def _ctc_lengths(text: str, vocabulary: dict[str, int]) -> tuple[int, int, int]:
    token_ids = [vocabulary.get(character, 0) for character in text]
    adjacent_repeats = sum(left == right for left, right in pairwise(token_ids))
    target_length = len(token_ids)
    return target_length, adjacent_repeats, target_length + adjacent_repeats


class SemanticVaeCelebVDubDataset(Dataset):
    """The complete 79,613-record CelebV-Dub train split represented at 64D/40 Hz.

    CTC-infeasible 40 Hz examples are intentionally retained. This exactly
    preserves the original D1 diffusion training set; its existing
    ``zero_infinity=True`` CTC behavior makes only those examples' CTC term
    zero while their diffusion loss remains active.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        normalization_path: str | Path,
        vocab_path: str | Path,
        *,
        expected_manifest_sha256: str,
        expected_inventory_sha256: str,
        expected_normalization_sha256: str,
        expected_vocab_sha256: str,
        expected_record_count: int = CELEBVDUB_TRAIN_COUNT,
    ):
        self.manifest_path = _regular_file(manifest_path, label="CelebV-Dub train manifest")
        if self.manifest_path.name != "train.jsonl":
            raise ValueError(
                "Direct-D1 must use the complete train.jsonl split rather than the CTC-filtered subset: "
                f"{self.manifest_path}"
            )
        actual_manifest_sha256 = sha256_file(self.manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                "CelebV-Dub train manifest SHA256 mismatch: "
                f"expected={expected_manifest_sha256}, got={actual_manifest_sha256}"
            )

        cache_root_path = Path(cache_root).expanduser().absolute()
        if cache_root_path.is_symlink() or not cache_root_path.is_dir():
            raise FileNotFoundError(f"Semantic-VAE cache root must be a regular directory: {cache_root_path}")
        self.cache_root = cache_root_path.resolve(strict=True)

        inventory_meta_path = _regular_file(
            self.manifest_path.parent / "inventory_meta.json", label="CelebV-Dub inventory metadata"
        )
        inventory_meta = _read_json(inventory_meta_path, label="CelebV-Dub inventory metadata")
        manifests = inventory_meta.get("manifests")
        if not isinstance(manifests, dict):
            raise TypeError("CelebV-Dub inventory metadata has no manifests mapping")
        train_entry = manifests.get("train.jsonl")
        inventory_entry = manifests.get("inventory.jsonl")
        if not isinstance(train_entry, dict) or not isinstance(inventory_entry, dict):
            raise TypeError("CelebV-Dub inventory metadata is missing train/full entries")
        if (
            train_entry.get("count") != expected_record_count
            or train_entry.get("sha256") != actual_manifest_sha256
            or train_entry.get("size_bytes") != self.manifest_path.stat().st_size
            or inventory_entry.get("count") != CELEBVDUB_INVENTORY_COUNT
            or inventory_entry.get("sha256") != expected_inventory_sha256
        ):
            raise RuntimeError("CelebV-Dub train manifest differs from its immutable inventory metadata")
        if inventory_meta.get("latent_spec") != {
            "dimension": SEMANTIC_VAE_LATENT_DIM,
            "dtype": "float32",
            "frame_rate_hz": 40.0,
            "hop_length_samples": 400,
            "mode": "fixed_posterior_sample",
            "sample_rate": 16_000,
        }:
            raise RuntimeError("CelebV-Dub inventory does not describe the fixed 64D/40 Hz latent cache")

        latent_completion = _read_json(
            _regular_file(self.cache_root / "state/latents/complete.json", label="latent completion marker"),
            label="latent completion marker",
        )
        video_completion = _read_json(
            _regular_file(self.cache_root / "state/video_40hz/complete.json", label="video completion marker"),
            label="video completion marker",
        )
        for completion, feature, frame_key in (
            (latent_completion, SEMANTIC_VAE_FEATURE, "total_latent_frames"),
            (video_completion, VIDEO_40HZ_FEATURE, "total_target_frames"),
        ):
            if (
                completion.get("cache_schema_version") != 1
                or completion.get("feature") != feature
                or completion.get("selection") != {"mode": "full"}
                or completion.get("count") != CELEBVDUB_INVENTORY_COUNT
                or completion.get("manifest_sha256") != expected_inventory_sha256
                or not isinstance(completion.get(frame_key), int)
            ):
                raise RuntimeError(f"Invalid authoritative cache completion marker for {feature}")
        if latent_completion["total_latent_frames"] != video_completion["total_target_frames"]:
            raise RuntimeError("Semantic-VAE latent and 40 Hz video cache frame totals differ")
        for state_name in ("latents", "video_40hz"):
            write_guard = self.cache_root / f"state/{state_name}/WRITE_ACTIVE.json"
            if write_guard.exists() or write_guard.is_symlink():
                raise RuntimeError(f"Refusing to train while cache writer guard exists: {write_guard}")

        normalization_file = _regular_file(normalization_path, label="LibriSpeech latent normalization")
        actual_normalization_sha256 = sha256_file(normalization_file)
        if actual_normalization_sha256 != expected_normalization_sha256:
            raise RuntimeError(
                "LibriSpeech latent normalization SHA256 mismatch: "
                f"expected={expected_normalization_sha256}, got={actual_normalization_sha256}"
            )
        normalization = _read_json(normalization_file, label="LibriSpeech latent normalization")
        if (
            normalization.get("channel_count") != SEMANTIC_VAE_LATENT_DIM
            or normalization.get("feature") != SEMANTIC_VAE_FEATURE
            or normalization.get("method") != "per_channel_population_mean_std_float64_welford_v1"
            or normalization.get("scope") != "train"
        ):
            raise RuntimeError("Normalization is not the fixed LibriSpeech-train Semantic-VAE artifact")
        mean = np.asarray(normalization.get("mean"), dtype=np.float64)
        std = np.asarray(normalization.get("std"), dtype=np.float64)
        if (
            mean.shape != (SEMANTIC_VAE_LATENT_DIM,)
            or std.shape != (SEMANTIC_VAE_LATENT_DIM,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError("Invalid per-channel Semantic-VAE normalization statistics")
        self.latent_mean = mean.astype(np.float32)
        self.latent_std = std.astype(np.float32)

        vocab_file = _regular_file(vocab_path, label="CelebV-Dub vocabulary")
        vocabulary, actual_vocab_sha256 = _load_vocab(vocab_file)
        if actual_vocab_sha256 != expected_vocab_sha256:
            raise RuntimeError(
                f"CelebV-Dub vocabulary SHA256 mismatch: expected={expected_vocab_sha256}, got={actual_vocab_sha256}"
            )

        records = _read_jsonl(self.manifest_path)
        if len(records) != expected_record_count:
            raise RuntimeError(f"Expected {expected_record_count} train records, found {len(records)}")
        seen: set[str] = set()
        feasible_count = 0
        infeasible_count = 0
        for record in records:
            key = record.get("utterance_key")
            text = record.get("text")
            frames = record.get("latent_frames")
            if record.get("split") != "train" or not isinstance(key, str) or key in seen:
                raise ValueError(f"Invalid or duplicate CelebV-Dub train record: {key!r}")
            seen.add(key)
            if not isinstance(text, str) or not text or type(frames) is not int or frames <= 0:
                raise ValueError(f"Invalid text/latent length for {key}")
            if record.get("latent_dim") != SEMANTIC_VAE_LATENT_DIM or record.get("video_dim") != CELEBVDUB_VIDEO_DIM:
                raise ValueError(f"Invalid cached feature dimensions for {key}")
            if not isinstance(record.get("latent_relative_path"), str) or not isinstance(
                record.get("video_40hz_relative_path"), str
            ):
                raise TypeError(f"Missing cached feature path for {key}")
            target_length, adjacent_repeats, minimum_frames = _ctc_lengths(text, vocabulary)
            feasible = frames >= minimum_frames
            if (
                record.get("ctc_target_length") != target_length
                or record.get("ctc_adjacent_repeats") != adjacent_repeats
                or record.get("ctc_min_input_frames") != minimum_frames
                or record.get("ctc_feasible_40hz") is not feasible
            ):
                raise ValueError(f"Invalid 40 Hz CTC metadata for {key}")
            feasible_count += int(feasible)
            infeasible_count += int(not feasible)
        if feasible_count != CELEBVDUB_CTC40_FEASIBLE_COUNT or infeasible_count != CELEBVDUB_CTC40_INFEASIBLE_COUNT:
            raise RuntimeError(
                f"Unexpected 40 Hz CTC feasibility split: feasible={feasible_count}, infeasible={infeasible_count}"
            )

        self.records = records
        self.ctc_feasible_count = feasible_count
        self.ctc_infeasible_count = infeasible_count

    def __len__(self) -> int:
        return len(self.records)

    def get_frame_len(self, index: int) -> int:
        return int(self.records[index]["latent_frames"])

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        frames = int(record["latent_frames"])
        latent_path = _safe_join(self.cache_root, record["latent_relative_path"], label="Semantic-VAE latent")
        video_path = _safe_join(self.cache_root, record["video_40hz_relative_path"], label="40 Hz video")
        for label, path in (("Semantic-VAE latent", latent_path), ("40 Hz video", video_path)):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"{label} cache is missing for {record['utterance_key']}: {path}")

        latent = np.load(latent_path, allow_pickle=False)
        video = np.load(video_path, allow_pickle=False)
        if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(
                f"Invalid latent for {record['utterance_key']}: shape={latent.shape}, dtype={latent.dtype}"
            )
        if video.shape != (frames, CELEBVDUB_VIDEO_DIM) or video.dtype != np.float32:
            raise ValueError(f"Invalid video for {record['utterance_key']}: shape={video.shape}, dtype={video.dtype}")
        if not np.isfinite(latent).all() or not np.isfinite(video).all():
            raise FloatingPointError(f"Non-finite cached feature for {record['utterance_key']}")
        normalized_latent = (latent - self.latent_mean) / self.latent_std
        if not np.isfinite(normalized_latent).all():
            raise FloatingPointError(f"Non-finite normalized latent for {record['utterance_key']}")
        return {
            "ctc_feasible": bool(record["ctc_feasible_40hz"]),
            "ctc_target_length": int(record["ctc_target_length"]),
            "mel_spec": torch.from_numpy(normalized_latent).transpose(0, 1),
            "text": record["text"],
            "utterance_key": record["utterance_key"],
            "video": torch.from_numpy(video.copy()),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty Semantic-VAE CelebV-Dub batch")
        lengths = torch.tensor([item["mel_spec"].shape[1] for item in batch], dtype=torch.long)
        video_lengths = torch.tensor([item["video"].shape[0] for item in batch], dtype=torch.long)
        if not torch.equal(lengths, video_lengths):
            raise ValueError(f"Audio/video lengths differ: {lengths.tolist()} != {video_lengths.tolist()}")
        max_length = int(lengths.max())
        latent = torch.stack([F.pad(item["mel_spec"], (0, max_length - item["mel_spec"].shape[1])) for item in batch])
        video = torch.stack([F.pad(item["video"], (0, 0, 0, max_length - item["video"].shape[0])) for item in batch])
        return {
            "ctc_feasible": torch.tensor([item["ctc_feasible"] for item in batch], dtype=torch.bool),
            "mel": latent,
            "mel_lengths": lengths,
            "text": [item["text"] for item in batch],
            "text_lengths": torch.tensor([item["ctc_target_length"] for item in batch], dtype=torch.long),
            "utterance_keys": [item["utterance_key"] for item in batch],
            "video": video,
            "video_lengths": video_lengths,
        }
