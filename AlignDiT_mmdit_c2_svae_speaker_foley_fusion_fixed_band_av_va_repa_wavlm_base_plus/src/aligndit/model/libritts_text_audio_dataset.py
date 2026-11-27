"""Audited LibriTTS text/audio caches and lossless, length-budget batching."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


CACHE_SCHEMA = 1
LATENT_DIM = 64
SPEAKER_DIM = 192
SAMPLE_RATE = 16_000
HOP_LENGTH = 400
NORMALIZATION_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(relative: str) -> Path:
    path = Path(relative)
    if "\0" in str(path) or path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe cache/source relative path: {relative!r}")
    return path


def safe_path(root: Path, relative: str) -> Path:
    path = _relative_path(relative)
    result = root / path
    result.resolve().relative_to(root.resolve())
    if result.is_symlink():
        raise ValueError(f"Cache/source file must not be a symlink: {result}")
    return result


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_vocab(path: Path) -> dict[str, int]:
    chars = path.read_text(encoding="utf-8").splitlines()
    if not chars or chars[0] != " " or any(len(char) != 1 for char in chars) or len(set(chars)) != len(chars):
        raise ValueError(f"Invalid character vocabulary (space must retain ID 0): {path}")
    return {char: index for index, char in enumerate(chars)}


def ctc_min_frames(text: str, vocab: dict[str, int]) -> int:
    ids = [vocab[char] for char in text]  # An unknown character is an error, never silently a space.
    return len(ids) + sum(left == right for left, right in zip(ids, ids[1:]))


def load_manifest(cache_root: str | Path) -> tuple[dict, list[dict], dict[str, int]]:
    root = Path(cache_root).resolve(strict=True)
    metadata = read_json(root / "metadata.json")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("Unsupported LibriTTS cache schema")
    for name, expected in metadata["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Immutable LibriTTS inventory file changed: {path}")
    vocab = read_vocab(root / "vocab.txt")
    base_chars = metadata["base_vocabulary"]
    if list(vocab)[: len(base_chars)] != base_chars:
        raise RuntimeError("LibriTTS vocabulary reordered the existing vocabulary IDs")
    records = []
    with (root / "inventory.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != metadata["counts"]["accepted"] or len(records) != metadata["counts"]["discovered"]:
        raise RuntimeError("LibriTTS inventory coverage does not match metadata")
    if metadata["counts"]["rejected"] != 0:
        raise RuntimeError("This cache contains rejected source records")
    seen = set()
    speakers = {"train": set(), "val": set()}
    counts = {"train": 0, "val": 0}
    for record in records:
        key = record["utterance_key"]
        if key in seen or record["split"] not in counts:
            raise ValueError(f"Duplicate or invalid LibriTTS record: {key}")
        seen.add(key)
        for field in ("audio_relative_path", "latent_relative_path", "speaker_relative_path", "receipt_relative_path"):
            # This stage checks manifest syntax only. Every actual read/write uses
            # safe_path below or in the extractor, including symlink containment.
            # Avoid millions of redundant NFS stat calls at each DDP startup.
            _relative_path(record[field])
        if not record["text"] or record["ctc_min_frames"] != ctc_min_frames(record["text"], vocab):
            raise ValueError(f"Invalid text/CTC metadata for {key}")
        expected_samples = math.ceil(record["source_num_samples"] * SAMPLE_RATE / record["source_sample_rate"])
        if record["resampled_num_samples"] != expected_samples or record["latent_frames"] != math.ceil(expected_samples / HOP_LENGTH):
            raise ValueError(f"Invalid resampling/frame metadata for {key}")
        if record["latent_frames"] <= 0 or record["ctc_feasible"] != (record["latent_frames"] >= record["ctc_min_frames"]):
            raise ValueError(f"Invalid latent/CTC feasibility for {key}")
        counts[record["split"]] += 1
        speakers[record["split"]].add(record["speaker_id"])
    if speakers["train"] & speakers["val"] or counts != metadata["split_counts"]:
        raise RuntimeError("Speaker split leaks or split counts changed")
    return metadata, records, vocab


def validate_arrays(latent: np.ndarray, speaker: np.ndarray, frames: int) -> None:
    if latent.shape != (frames, LATENT_DIM) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError(f"Invalid latent cache: shape={latent.shape}, dtype={latent.dtype}")
    if speaker.shape != (SPEAKER_DIM,) or speaker.dtype != np.float32 or not np.isfinite(speaker).all():
        raise ValueError(f"Invalid speaker cache: shape={speaker.shape}, dtype={speaker.dtype}")
    if abs(float(np.linalg.norm(speaker)) - 1.0) > 1e-4:
        raise ValueError("Speaker embedding is not L2 normalized")


def load_cached_record(root: Path, record: dict, spec_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    receipt = read_json(safe_path(root, record["receipt_relative_path"]))
    if receipt.get("utterance_key") != record["utterance_key"] or receipt.get("extraction_spec_sha256") != spec_sha256:
        raise ValueError(f"Stale/mismatched extraction receipt for {record['utterance_key']}")
    arrays = []
    for kind in ("latent", "speaker"):
        path = safe_path(root, record[f"{kind}_relative_path"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt[f"{kind}_sha256"]:
            raise ValueError(f"Corrupt {kind} cache: {path}")
        arrays.append(np.load(io.BytesIO(payload), allow_pickle=False))
    validate_arrays(arrays[0], arrays[1], record["latent_frames"])
    return arrays[0], arrays[1]


class LibriTTSTextAudioDataset(Dataset):
    """A speaker-disjoint split; both disk hashes and runtime tensors are checked."""

    def __init__(self, cache_root, split="train", normalization_path=None, require_complete=True):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.cache_root = Path(cache_root).resolve(strict=True)
        self.metadata, records, self.vocab_char_map = load_manifest(self.cache_root)
        self.vocab_path = self.cache_root / "vocab.txt"
        self.vocab_size = len(self.vocab_char_map)
        spec_path = self.cache_root / "extraction_spec.json"
        self.spec_sha256 = sha256_file(spec_path)
        spec = read_json(spec_path)
        if spec.get("metadata_sha256") != sha256_file(self.cache_root / "metadata.json"):
            raise RuntimeError("Extraction spec refers to a different inventory")
        if require_complete:
            complete = read_json(self.cache_root / "complete.json")
            if (
                complete.get("complete") is not True
                or complete.get("expected") != len(records)
                or complete.get("valid") != len(records)
                or complete.get("metadata_sha256") != sha256_file(self.cache_root / "metadata.json")
                or complete.get("extraction_spec_sha256") != self.spec_sha256
                or complete.get("failures") != []
            ):
                raise RuntimeError("LibriTTS cache has not passed a complete audit")
        self.normalization_path = Path(normalization_path) if normalization_path else self.cache_root / "normalization.json"
        if sha256_file(self.normalization_path) != self.metadata["files"]["normalization.json"]:
            raise RuntimeError("Pretraining normalization differs from the cached/dubbing normalization")
        normalization = read_json(self.normalization_path)
        self.latent_mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.latent_std = np.asarray(normalization["std"], dtype=np.float32)
        if (
            self.latent_mean.shape != (LATENT_DIM,)
            or self.latent_std.shape != (LATENT_DIM,)
            or not np.isfinite(self.latent_mean).all()
            or not np.isfinite(self.latent_std).all()
            or np.any(self.latent_std <= 0)
        ):
            raise ValueError("Invalid 64-channel latent normalization")
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"LibriTTS {split} split is empty")
        self.contract = {
            "schema_version": CACHE_SCHEMA,
            "metadata_sha256": sha256_file(self.cache_root / "metadata.json"),
            "extraction_spec_sha256": self.spec_sha256,
            "vocab_sha256": self.metadata["files"]["vocab.txt"],
            "normalization_sha256": self.metadata["files"]["normalization.json"],
            "codec_sha256": spec["semantic_vae_checkpoint_sha256"],
            "semantic_vae_checkpoint_sha256": spec["semantic_vae_checkpoint_sha256"],
            "speaker_checkpoint_sha256": spec["speaker_checkpoint_sha256"],
            "posterior_protocol": spec["posterior_protocol"],
            "selection": self.metadata["selection"],
            "split": split,
            "count": len(self.records),
        }

    def __len__(self):
        return len(self.records)

    def get_frame_len(self, index):
        return self.records[index]["latent_frames"]

    def __getitem__(self, index):
        record = self.records[index]
        latent, speaker = load_cached_record(self.cache_root, record, self.spec_sha256)
        latent = (latent - self.latent_mean) / self.latent_std
        if not np.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite normalized latent: {record['utterance_key']}")
        return {
            "audio": torch.from_numpy(latent),
            "text": record["text"],
            "speaker_embedding": torch.from_numpy(speaker),
            "utterance_key": record["utterance_key"],
            "ctc_feasible": record["ctc_feasible"],
        }

    @staticmethod
    def collate_fn(batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        return {
            "audio": pad_sequence([item["audio"] for item in batch], batch_first=True),
            "lens": torch.tensor([item["audio"].shape[0] for item in batch], dtype=torch.long),
            "text": [item["text"] for item in batch],
            "speaker_embedding": torch.stack([item["speaker_embedding"] for item in batch]),
            "utterance_keys": [item["utterance_key"] for item in batch],
            "ctc_feasible": torch.tensor([item["ctc_feasible"] for item in batch], dtype=torch.bool),
        }


class LengthBudgetBatchSampler(Sampler):
    """Bound padded frames, retaining *every* sample including over-budget utterances.

    DDP equalization/sharding is deliberately left to the trainer. Length-sorted
    groups limit padding; whole-batch shuffling is deterministic per epoch.
    """

    def __init__(self, dataset, max_frames=3600, max_samples=32, seed=666, shuffle=True, overflow="singleton"):
        if max_frames <= 0 or max_samples <= 0 or overflow not in {"singleton", "error"}:
            raise ValueError("Invalid frame budget, sample limit, or overflow policy")
        self.dataset, self.max_frames, self.max_samples = dataset, max_frames, max_samples
        self.seed, self.shuffle, self.overflow, self.epoch = seed, shuffle, overflow, 0
        ordered = sorted(range(len(dataset)), key=lambda i: (dataset.get_frame_len(i), i))
        batches, batch = [], []
        for index in ordered:
            frames = dataset.get_frame_len(index)
            if frames <= 0:
                raise ValueError(f"Nonpositive frame length at index {index}")
            if frames > max_frames and overflow == "error":
                raise ValueError(f"Utterance {index} has {frames} frames, exceeding budget {max_frames}; nothing was dropped")
            if batch and (len(batch) >= max_samples or frames * (len(batch) + 1) > max_frames):
                batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
        self._batches = batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def batches_for_epoch(self, epoch=None):
        batches = [batch.copy() for batch in self._batches]
        if self.shuffle:
            random.Random(self.seed + (self.epoch if epoch is None else int(epoch))).shuffle(batches)
        return batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self.batches_for_epoch())
