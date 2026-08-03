import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset

from aligndit.script.misc.svae_cache_utils import (
    HUBERT_HIDDEN_DIM,
    SEMANTIC_VAE_LATENT_DIM,
    read_jsonl,
    safe_join,
    sha256_file,
)
from f5_tts.model.dataset import CustomDataset


HUBERT_40HZ_FEATURE = "hubert_large_ll60k_last_hidden_40hz_linear_v1"


def _load_regular_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def cut_or_pad(data, size, dim=0, mode="constant", value=None):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(
            data.unsqueeze(0), [0] * (2 * (data.dim() - dim) - 1) + [padding], mode=mode, value=value
        ).squeeze(0)
        size = data.size(dim)
    elif data.size(dim) > size:
        if dim == 0:
            data = data[:size]
        elif dim == 1:
            data = data[:, :size]
        else:
            assert False
    assert data.size(dim) == size
    return data


class CustomDataset_mel(CustomDataset):
    def __init__(self, *args, data_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir  # if set, relative audio_path in arrow will be resolved to abs path
        self.data = self.data.filter(lambda example: 0.3 <= example["duration"] <= 30)
        self.durations = self.data["duration"]

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print(f"Error in loading data index {index}: {e}. Return None.")
            return None

    def getitem(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        if self.data_dir and not os.path.isabs(audio_path):
            audio_path = os.path.join(os.path.dirname(self.data_dir), audio_path)
        mel_path = os.path.splitext(audio_path.replace("/audio/", "/mel_tacotron/"))[0] + ".npy"
        mel_spec = torch.from_numpy(np.load(mel_path).T)
        return {
            "mel_spec": mel_spec,
            "text": text,
            "audio_path": audio_path,
        }

    @staticmethod
    def collate_fn(batch):
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch if item is not None]

        if len(mel_specs) == 0:
            return {}

        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:  # TODO. maybe records mask for attention here
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])

        return dict(
            mel=mel_specs,
            mel_lengths=mel_lengths,
            text=text,
            text_lengths=text_lengths,
        )


class CustomDataset_mel_rep(CustomDataset_mel):
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if ret is not None:
            audio_path = ret["audio_path"]
            rep_path = os.path.splitext(audio_path.replace("/audio/", "/hubert_large_ll60k/"))[0] + ".npy"
            ret["rep"] = torch.from_numpy(np.load(rep_path))
        return ret

    @staticmethod
    def collate_fn(batch):
        ret = CustomDataset_mel.collate_fn(batch)
        if ret == {}:
            return {}

        # rep: [T x D]
        reps = [item["rep"] for item in batch]
        rep_lengths = torch.LongTensor([len(rep) for rep in reps])
        max_rep_length = rep_lengths.amax()

        padded_reps = []
        for rep in reps:  # TODO. maybe records mask for attention here
            padding = (0, 0, 0, max_rep_length - len(rep))
            padded_rep = F.pad(rep, padding, value=0)
            padded_reps.append(padded_rep)

        reps = torch.stack(padded_reps)

        ret["rep"] = reps
        ret["rep_lengths"] = rep_lengths

        return ret


class CustomDataset_mel_video(CustomDataset_mel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_fps = 25
        self.mel_spec_fps = self.target_sample_rate // self.hop_length
        assert self.mel_spec_fps == 100
        self.audio_video_ratio = self.mel_spec_fps // self.video_fps
        assert self.audio_video_ratio == 4

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if ret is not None:
            mel_spec = ret["mel_spec"]
            audio_path = ret["audio_path"]
            video_path = os.path.splitext(audio_path.replace("/audio/", "/avhubert_video_feat/"))[0] + ".npy"
            video = torch.from_numpy(np.load(video_path))

            mel_spec = cut_or_pad(mel_spec, len(video) * self.audio_video_ratio, dim=1, mode="replicate")
            ret["mel_spec"] = mel_spec
            ret["video"] = video

        return ret

    @staticmethod
    def collate_fn(batch):
        ret = CustomDataset_mel.collate_fn(batch)
        if ret == {}:
            return {}

        video_feats = [item["video"] for item in batch]
        video_lengths = torch.LongTensor([len(feat) for feat in video_feats])
        max_video_length = video_lengths.amax()

        padded_video_feats = []
        for feat in video_feats:  # TODO. maybe records mask for attention here
            video_padding = (0, 0, 0, max_video_length - len(feat))
            padded_feat = F.pad(feat, video_padding, value=0)
            padded_video_feats.append(padded_feat)
        video = torch.stack(padded_video_feats)

        ret["video"] = video
        ret["video_lengths"] = video_lengths

        return ret


class SemanticVaePretrainDataset(Dataset):
    """Manifest-backed, exactly aligned Semantic-VAE/HuBERT training pairs."""

    def __init__(self, manifest_path: str, cache_root: str, normalization_path: str):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        if not self.manifest_path.is_file() or self.manifest_path.is_symlink():
            raise FileNotFoundError(f"Training manifest must be a regular file: {self.manifest_path}")
        self.cache_root = Path(cache_root).resolve(strict=True)
        inventory_meta = _load_regular_json(self.manifest_path.parent / "inventory_meta.json")
        manifests = inventory_meta.get("manifests")
        if not isinstance(manifests, dict):
            raise TypeError("Inventory metadata does not contain a manifests mapping")
        train_manifest_entry = manifests.get(self.manifest_path.name)
        inventory_entry = manifests.get("inventory.jsonl")
        if not isinstance(train_manifest_entry, dict) or not isinstance(inventory_entry, dict):
            raise TypeError("Inventory metadata is missing the train/full manifest entries")
        train_manifest_sha256 = sha256_file(self.manifest_path)
        if train_manifest_entry.get("sha256") != train_manifest_sha256:
            raise RuntimeError("Training manifest differs from its immutable inventory metadata")
        inventory_count = int(inventory_entry["count"])
        inventory_sha256 = inventory_entry.get("sha256")

        latent_complete_path = safe_join(self.cache_root, "state/latents/complete.json")
        hubert_complete_path = safe_join(self.cache_root, "state/hubert_40hz/complete.json")
        latent_complete = _load_regular_json(latent_complete_path)
        hubert_complete = _load_regular_json(hubert_complete_path)
        if (
            latent_complete.get("cache_schema_version") != 1
            or latent_complete.get("feature") != "semantic_vae_posterior_sample_v1"
            or latent_complete.get("selection") != {"mode": "full"}
            or latent_complete.get("count") != inventory_count
            or latent_complete.get("manifest_sha256") != inventory_sha256
        ):
            raise RuntimeError("Semantic-VAE completion marker is not the authoritative full inventory cache")
        if (
            hubert_complete.get("cache_schema_version") != 1
            or hubert_complete.get("feature") != HUBERT_40HZ_FEATURE
            or hubert_complete.get("selection") != {"mode": "full"}
            or hubert_complete.get("count") != inventory_count
            or hubert_complete.get("manifest_sha256") != inventory_sha256
            or hubert_complete.get("total_target_frames") != latent_complete.get("total_latent_frames")
        ):
            raise RuntimeError("HuBERT completion marker is not the authoritative full 40 Hz inventory cache")
        for name, completion in (("latents", latent_complete), ("hubert_40hz", hubert_complete)):
            index = completion.get("consolidated_index")
            if (
                not isinstance(index, dict)
                or index.get("count") != inventory_count
                or index.get("sha256") != completion.get("ordered_index_sha256")
            ):
                raise RuntimeError(f"Invalid {name} consolidated-index contract")
            index_path = safe_join(self.cache_root, index["path"])
            if (
                not index_path.is_file()
                or index_path.is_symlink()
                or index_path.stat().st_size != index.get("size_bytes")
                or sha256_file(index_path) != index["sha256"]
            ):
                raise RuntimeError(f"{name} consolidated index differs from its completion marker")
        normalization_path = Path(normalization_path).resolve(strict=True)
        if not normalization_path.is_file() or normalization_path.is_symlink():
            raise FileNotFoundError(f"Normalization statistics must be a regular file: {normalization_path}")
        normalization = _load_regular_json(normalization_path)
        required = {
            "cache_schema_version",
            "channel_count",
            "count",
            "feature",
            "frame_count",
            "latent_complete_sha256",
            "mean",
            "method",
            "scope",
            "std",
            "train_manifest_sha256",
        }
        if set(normalization) != required:
            raise ValueError(f"Unexpected normalization-statistics keys: {sorted(normalization)}")
        if (
            normalization["cache_schema_version"] != 1
            or normalization["channel_count"] != SEMANTIC_VAE_LATENT_DIM
            or normalization["feature"] != "semantic_vae_posterior_sample_v1"
            or normalization["method"] != "per_channel_population_mean_std_float64_welford_v1"
            or normalization["scope"] != "train"
            or normalization["train_manifest_sha256"] != train_manifest_sha256
            or normalization["latent_complete_sha256"] != sha256_file(latent_complete_path)
        ):
            raise ValueError("Normalization statistics do not match the immutable train/latent cache contract")
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        std = np.asarray(normalization["std"], dtype=np.float64)
        if (
            mean.shape != (SEMANTIC_VAE_LATENT_DIM,)
            or std.shape != (SEMANTIC_VAE_LATENT_DIM,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Invalid per-channel Semantic-VAE normalization statistics")
        self.normalization_path = normalization_path
        self.normalization_sha256 = sha256_file(normalization_path)
        self.latent_mean = mean.astype(np.float32)
        self.latent_std = std.astype(np.float32)
        self.records = list(read_jsonl(self.manifest_path))
        if not self.records:
            raise ValueError(f"Empty Semantic-VAE pretraining manifest: {self.manifest_path}")
        seen = set()
        for record in self.records:
            key = record["utterance_key"]
            if record["split"] != "train" or key in seen:
                raise ValueError(f"Invalid/duplicate training record: {key}")
            seen.add(key)
        if len(self.records) != normalization["count"]:
            raise ValueError("Normalization record count differs from the training manifest")
        if sum(int(record["latent_frames"]) for record in self.records) != normalization["frame_count"]:
            raise ValueError("Normalization frame count differs from the training manifest")

    def __len__(self):
        return len(self.records)

    def get_frame_len(self, index):
        return int(self.records[index]["latent_frames"])

    def __getitem__(self, index):
        record = self.records[index]
        latent_path = safe_join(self.cache_root, record["latent_relative_path"])
        feature_path = safe_join(self.cache_root, f"hubert_40hz/{record['utterance_key']}.npy")
        latent = np.load(latent_path, allow_pickle=False)
        feature = np.load(feature_path, allow_pickle=False)
        frames = int(record["latent_frames"])
        if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(
                f"Invalid Semantic-VAE latent for {record['utterance_key']}: {latent.shape}/{latent.dtype}"
            )
        if feature.shape != (frames, HUBERT_HIDDEN_DIM) or feature.dtype != np.float32:
            raise ValueError(f"Invalid HuBERT feature for {record['utterance_key']}: {feature.shape}/{feature.dtype}")
        if not np.isfinite(feature).all():
            raise FloatingPointError(f"Non-finite HuBERT feature for {record['utterance_key']}")
        latent = (latent - self.latent_mean) / self.latent_std
        if not np.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite normalized latent for {record['utterance_key']}")
        return {"mel_spec": torch.from_numpy(latent).transpose(0, 1), "rep": torch.from_numpy(feature.copy())}

    @staticmethod
    def collate_fn(batch):
        lengths = torch.tensor([item["rep"].shape[0] for item in batch], dtype=torch.long)
        max_length = int(lengths.max())
        latent = torch.stack([F.pad(item["mel_spec"], (0, max_length - item["mel_spec"].shape[1])) for item in batch])
        feature = torch.stack([F.pad(item["rep"], (0, 0, 0, max_length - item["rep"].shape[0])) for item in batch])
        return {"mel": latent, "mel_lengths": lengths, "rep": feature, "rep_lengths": lengths.clone()}


# Load dataset


def load_dataset_mel(
    dataset_name: str,
    tokenizer: str = "char",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    data_dir: str | None = None,
) -> CustomDataset | CustomDataset_mel | CustomDataset_mel_rep | CustomDataset_mel_video:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type in ["CustomDataset", "CustomDataset_mel", "CustomDataset_mel_rep", "CustomDataset_mel_video"]:
        if data_dir:
            rel_data_path = (
                os.path.join(data_dir, f"{dataset_name}_{tokenizer}")
                if tokenizer
                else os.path.join(data_dir, dataset_name)
            )
        elif tokenizer:
            rel_data_path = str(files("aligndit").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        else:
            rel_data_path = str(files("aligndit").joinpath(f"../../data/{dataset_name}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]

        if dataset_type == "CustomDataset":
            dataset_cls = CustomDataset
        elif dataset_type == "CustomDataset_mel":
            dataset_cls = CustomDataset_mel
        elif dataset_type == "CustomDataset_mel_rep":
            dataset_cls = CustomDataset_mel_rep
        elif dataset_type == "CustomDataset_mel_video":
            dataset_cls = CustomDataset_mel_video

        train_dataset = dataset_cls(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            data_dir=data_dir,
            **mel_spec_kwargs,
        )

    return train_dataset
