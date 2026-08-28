"""CelebV-Dub dataset backed by cached MingTok latents.

The cache contract is intentionally small and explicit:

* every ``.npy`` file is a raw FP32 array with shape ``[T, 64]``;
* latents run at 50 Hz and AV-HuBERT video features at 25 Hz;
* cache paths mirror the path below the CelebV-Dub ``audio`` directory under
  ``<cache_dir>/latents``.

No latent normalization is applied here.  Each sample is only trimmed or
replicate-padded at its right boundary to preserve the exact 2:1 audio/video
length contract used by C2.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as Dataset_
from torch.utils.data import Dataset

from aligndit.model.dataset import cut_or_pad
from aligndit.model.mingtok_codec import EXPECTED_CONFIG_SHA256, EXPECTED_MODEL_SHA256


CELEBVDUB_RAW_ARROW_SHA256 = "99da14538f85eca3a039282d1cb5126f2a5598dd3c513422fe58b454af9437ef"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_raw_dataset(path: str):
    metadata_path = Path(path) / "raw.arrow"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Strict C2 metadata file not found: {metadata_path}")
    actual_sha256 = _sha256_file(metadata_path)
    if actual_sha256 != CELEBVDUB_RAW_ARROW_SHA256:
        raise RuntimeError(
            "CelebVDub raw.arrow SHA256 mismatch: "
            f"expected {CELEBVDUB_RAW_ARROW_SHA256}, got {actual_sha256} ({metadata_path})"
        )
    return Dataset_.from_file(str(metadata_path))


def _resolve_audio_path(audio_path: str, data_dir: str | None) -> str:
    if os.path.isabs(audio_path):
        return audio_path
    if data_dir is None:
        return os.path.abspath(audio_path)

    # CelebV-Dub metadata stores paths such as
    # ``data/CelebVDub/audio/train/...`` while data_dir points at the absolute
    # ``.../data`` directory.  Resolve that representation without baking a
    # project-specific prefix into the dataset.
    path = Path(audio_path)
    data_root = Path(data_dir)
    if path.parts and path.parts[0] == data_root.name:
        return str(data_root.parent.joinpath(path))
    return str(data_root.joinpath(path))


def _relative_below_audio(audio_path: str) -> Path:
    parts = Path(audio_path).parts
    try:
        audio_i = len(parts) - 1 - tuple(reversed(parts)).index("audio")
    except ValueError as error:
        raise ValueError(f"audio_path must contain an 'audio' component: {audio_path}") from error
    relative = Path(*parts[audio_i + 1 :]).with_suffix(".npy")
    if not relative.parts:
        raise ValueError(f"audio_path has no path below its 'audio' component: {audio_path}")
    return relative


def _replace_audio_component(audio_path: str, replacement: str) -> str:
    path = Path(audio_path)
    parts = list(path.parts)
    try:
        audio_i = len(parts) - 1 - list(reversed(parts)).index("audio")
    except ValueError as error:
        raise ValueError(f"audio_path must contain an 'audio' component: {audio_path}") from error
    parts[audio_i] = replacement
    return str(Path(*parts).with_suffix(".npy"))


def _validate_cache_contract(
    cache_dir: str,
    *,
    latent_dim: int,
    latent_fps: int,
    audio_video_ratio: int,
    metadata_count: int,
) -> None:
    contract_path = os.path.join(cache_dir, "contract.json")
    if not os.path.isfile(contract_path):
        raise FileNotFoundError(f"MingTok cache contract not found: {contract_path}")

    with open(contract_path, "r", encoding="utf-8") as file:
        contract = json.load(file)

    expected = {
        "schema_version": 1,
        "codec": "MingTok-Audio",
        "latent_dim": latent_dim,
        "dtype": "float32",
        "layout": "T,D",
        "latent_fps": latent_fps,
        "sample_rate": 16_000,
        "hop_size": 320,
        "audio_video_ratio": audio_video_ratio,
        "normalization": "none",
        "posterior_mode": "sample",
        "base_seed": 666,
        "split": "train",
        "num_items": metadata_count,
    }
    missing = sorted(set(expected) - set(contract))
    if missing:
        raise ValueError(f"MingTok cache contract is missing required fields {missing}: {contract_path}")
    for key, expected_value in expected.items():
        if contract[key] != expected_value:
            raise ValueError(
                f"MingTok cache contract mismatch for {key}: expected {expected_value!r}, "
                f"got {contract[key]!r} in {contract_path}"
            )

    nested_expected = {
        "selection": {
            "type": "raw_arrow",
            "metadata_sha256": CELEBVDUB_RAW_ARROW_SHA256,
            "audio_path_field": "audio_path",
            "ordering": "metadata_row_order",
        },
        "checkpoint": {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
        },
    }
    for section, section_expected in nested_expected.items():
        actual_section = contract.get(section)
        if not isinstance(actual_section, dict):
            raise TypeError(f"MingTok cache contract requires object field {section!r}: {contract_path}")
        for key, expected_value in section_expected.items():
            if actual_section.get(key) != expected_value:
                raise ValueError(
                    f"MingTok cache contract mismatch for {section}.{key}: "
                    f"expected {expected_value!r}, got {actual_section.get(key)!r} in {contract_path}"
                )


class CustomDatasetMingTokVideo(Dataset):
    """Cached 50 Hz MingTok latent plus native 25 Hz AV-HuBERT video."""

    def __init__(
        self,
        custom_dataset,
        *,
        data_dir: str | None,
        cache_dir: str,
        latent_dim: int = 64,
        latent_fps: int = 50,
        video_fps: int = 25,
        audio_video_ratio: int = 2,
        video_dim: int = 1024,
        video_feature_dirname: str = "avhubert_video_feat",
        min_duration: float = 0.3,
        max_duration: float = 30.0,
        validate_contract: bool = True,
    ):
        if cache_dir is None:
            raise ValueError("cache_dir is required for cached MingTok latents")
        if latent_dim != 64:
            raise ValueError(f"MingTok C2 expects latent_dim=64, got {latent_dim}")
        if latent_fps % video_fps != 0:
            raise ValueError(f"latent_fps={latent_fps} must be divisible by video_fps={video_fps}")
        derived_ratio = latent_fps // video_fps
        if audio_video_ratio != derived_ratio or audio_video_ratio != 2:
            raise ValueError(
                "MingTok C2 requires latent_fps/video_fps=audio_video_ratio=2, got "
                f"{latent_fps}/{video_fps} and ratio={audio_video_ratio}"
            )

        self.data = custom_dataset.filter(
            lambda example: min_duration <= float(example["duration"]) <= max_duration,
            # Every DDP rank constructs its dataset independently.  Avoid the
            # Hugging Face default of racing on one shared cache-*.arrow file;
            # this preserves the original C2 predicate and row order.
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        self.durations = self.data["duration"]
        self.data_dir = data_dir
        self.cache_dir = os.path.abspath(cache_dir)
        self.latent_root = os.path.join(self.cache_dir, "latents")
        self.latent_dim = latent_dim
        self.latent_fps = latent_fps
        self.video_fps = video_fps
        self.audio_video_ratio = audio_video_ratio
        self.video_dim = video_dim
        self.video_feature_dirname = video_feature_dirname

        if validate_contract:
            _validate_cache_contract(
                self.cache_dir,
                latent_dim=latent_dim,
                latent_fps=latent_fps,
                audio_video_ratio=audio_video_ratio,
                metadata_count=len(custom_dataset),
            )

    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        """Estimated number of 50 Hz latent frames for dynamic batching."""
        return float(self.durations[index]) * self.latent_fps

    def _paths(self, row):
        audio_path = _resolve_audio_path(row["audio_path"], self.data_dir)
        relative = _relative_below_audio(audio_path)
        latent_path = os.path.join(self.latent_root, relative)
        video_path = _replace_audio_component(audio_path, self.video_feature_dirname)
        return audio_path, latent_path, video_path

    def getitem(self, index):
        row = self.data[index]
        audio_path, latent_path, video_path = self._paths(row)

        latent_array = np.load(latent_path, allow_pickle=False)
        if latent_array.dtype != np.float32:
            raise TypeError(f"MingTok cache must be FP32, got {latent_array.dtype} at {latent_path}")
        if latent_array.ndim != 2 or latent_array.shape[1] != self.latent_dim:
            raise ValueError(
                f"MingTok cache must have shape [T,{self.latent_dim}], "
                f"got {latent_array.shape} at {latent_path}"
            )
        if latent_array.shape[0] == 0 or not np.isfinite(latent_array).all():
            raise ValueError(f"MingTok cache is empty or non-finite: {latent_path}")

        video_array = np.load(video_path, allow_pickle=False)
        if video_array.ndim != 2 or video_array.shape[1] != self.video_dim:
            raise ValueError(
                f"video feature must have shape [T,{self.video_dim}], got {video_array.shape} at {video_path}"
            )
        if video_array.shape[0] == 0 or not np.isfinite(video_array).all():
            raise ValueError(f"video feature is empty or non-finite: {video_path}")

        # Cache is [T, 64]; the C2 dataset/collate convention is [64, T].
        audio_latent = torch.from_numpy(latent_array).transpose(0, 1).contiguous()
        video = torch.from_numpy(video_array.astype(np.float32, copy=False))
        target_audio_len = len(video) * self.audio_video_ratio
        audio_latent = cut_or_pad(audio_latent, target_audio_len, dim=1, mode="replicate")

        return {
            "audio_latent": audio_latent,
            "video": video,
            "text": row["text"],
            "audio_path": audio_path,
            "latent_path": latent_path,
        }

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as error:  # noqa: BLE001 - corrupt/missing samples are filtered by collate_fn
            print(f"Error in loading MingTok data index {index}: {error}. Return None.")
            return None

    @staticmethod
    def collate_fn(batch):
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return {}

        audio_latents = [item["audio_latent"] for item in valid_batch]
        videos = [item["video"] for item in valid_batch]
        audio_lengths = torch.tensor([item.shape[-1] for item in audio_latents], dtype=torch.long)
        video_lengths = torch.tensor([item.shape[0] for item in videos], dtype=torch.long)

        if not torch.equal(audio_lengths, video_lengths * 2):
            raise ValueError(
                "collate received a non-2:1 MingTok/video pair: "
                f"audio={audio_lengths.tolist()}, video={video_lengths.tolist()}"
            )

        max_audio_len = int(audio_lengths.max().item())
        max_video_len = int(video_lengths.max().item())
        padded_audio = [F.pad(item, (0, max_audio_len - item.shape[-1]), value=0.0) for item in audio_latents]
        padded_video = [
            F.pad(item, (0, 0, 0, max_video_len - item.shape[0]), value=0.0) for item in videos
        ]

        text = [item["text"] for item in valid_batch]
        text_lengths = torch.tensor([len(item) for item in text], dtype=torch.long)

        return {
            "audio_latent": torch.stack(padded_audio),
            "audio_latent_lengths": audio_lengths,
            "video": torch.stack(padded_video),
            "video_lengths": video_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "audio_paths": [item["audio_path"] for item in valid_batch],
            "latent_paths": [item["latent_path"] for item in valid_batch],
        }


# Compatibility with the naming style of the original dataset module.
CustomDataset_mingtok_video = CustomDatasetMingTokVideo


def load_dataset_mingtok(
    dataset_name: str,
    tokenizer: str = "char",
    data_dir: str | None = None,
    cache_dir: str | None = None,
    latent_dim: int = 64,
    latent_fps: int = 50,
    video_fps: int = 25,
    audio_video_ratio: int = 2,
    **dataset_kwargs,
) -> CustomDatasetMingTokVideo:
    """Load CelebV-Dub metadata and attach cached MingTok/video features."""
    if data_dir:
        metadata_path = os.path.join(data_dir, f"{dataset_name}_{tokenizer}")
    else:
        metadata_path = str(files("aligndit").joinpath(f"../../data/{dataset_name}_{tokenizer}"))

    print(f"Loading MingTok dataset metadata from {metadata_path} ...")
    raw_dataset = _load_raw_dataset(metadata_path)
    if cache_dir is None:
        raise ValueError("cache_dir must point to the MingTok cache root containing contract.json and latents/")

    return CustomDatasetMingTokVideo(
        raw_dataset,
        data_dir=data_dir,
        cache_dir=cache_dir,
        latent_dim=latent_dim,
        latent_fps=latent_fps,
        video_fps=video_fps,
        audio_video_ratio=audio_video_ratio,
        **dataset_kwargs,
    )


__all__ = [
    "CELEBVDUB_RAW_ARROW_SHA256",
    "CustomDatasetMingTokVideo",
    "CustomDataset_mingtok_video",
    "load_dataset_mingtok",
]
