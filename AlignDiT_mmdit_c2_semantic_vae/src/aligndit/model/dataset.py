import json
import os
from collections.abc import Mapping
from importlib.resources import files
from itertools import pairwise, zip_longest
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
SEMANTIC_VAE_FEATURE = "semantic_vae_posterior_sample_v1"
VIDEO_40HZ_FEATURE = "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1"
LIBRISPEECH_SVAE_TRAIN_NORMALIZATION_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
CELEBVDUB_CTC40_TRAIN_COUNT = 79_508
CELEBVDUB_VIDEO_DIM = 1024
CELEBVDUB_INVENTORY_COUNT = 79_826
SEMANTIC_VAE_EMA_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
SEMANTIC_VAE_GOLDEN_LATENT_SHA256 = "7e3bd4e044b7f0a4f1d0295ece831e639a1e7abd78f13fd3ce85e3e1b9feccce"
SEMANTIC_VAE_SOURCE_COMMIT = "5bcca91fe8b65c0e52c5ee141968f98662dc4792"
SEMANTIC_VAE_CONFIG_SHA256 = "c12ba35b4035f97808dabaac4f254bd4e32b1dc5fba0840168ae0c41859d0235"
SEMANTIC_VAE_METAINFO_SHA256 = "24b8ff09360cdfe8a38e61862bf185c2130ef45a15e6f235bbbae8af8065c851"
SEMANTIC_VAE_BIGVGAN_CONFIG_SHA256 = "a11e013f623eedc55b2410d48cbd810322df03658377806d16ab396369525618"


def _load_regular_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def interpolate_video_to_latent_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Apply the one canonical AV-HuBERT 25 Hz -> latent 40 Hz interpolation."""

    if video.ndim != 2 or video.shape[0] <= 0 or video.shape[1] != CELEBVDUB_VIDEO_DIM:
        raise ValueError(f"video must have shape [Tv, {CELEBVDUB_VIDEO_DIM}] with Tv > 0, got {tuple(video.shape)}")
    if video.dtype != torch.float32 or not torch.isfinite(video).all():
        raise ValueError(f"video must be finite float32, got {video.dtype}")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    return (
        F.interpolate(
            video.transpose(0, 1).unsqueeze(0),
            size=target_frames,
            mode="linear",
            align_corners=False,
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )


def _load_vocab(vocab_path: Path) -> tuple[dict[str, int], str]:
    if not vocab_path.is_file() or vocab_path.is_symlink():
        raise FileNotFoundError(f"Vocabulary must be a regular file: {vocab_path}")
    characters = vocab_path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not characters or len(characters) != len(set(characters)):
        raise ValueError(f"Empty or duplicate vocabulary: {vocab_path}")
    vocabulary = {character: index for index, character in enumerate(characters)}
    if vocabulary.get(" ") != 0:
        raise ValueError(f"CelebV-Dub vocabulary must use space/unknown as id 0: {vocab_path}")
    return vocabulary, sha256_file(vocab_path)


def _ctc_record_fields(text: str, vocabulary: Mapping[str, int]) -> tuple[int, int, int]:
    token_ids = [vocabulary.get(character, 0) for character in text]
    target_length = len(token_ids)
    adjacent_repeats = sum(left == right for left, right in pairwise(token_ids))
    return target_length, adjacent_repeats, target_length + adjacent_repeats


def _validate_consolidated_index(cache_root: Path, completion: Mapping, expected_count: int) -> None:
    index = completion.get("consolidated_index")
    if (
        not isinstance(index, dict)
        or index.get("count") != expected_count
        or index.get("sha256") != completion.get("ordered_index_sha256")
    ):
        raise RuntimeError("Invalid consolidated-index contract")
    index_path = safe_join(cache_root, index["path"])
    if (
        not index_path.is_file()
        or index_path.is_symlink()
        or index_path.stat().st_size != index.get("size_bytes")
        or sha256_file(index_path) != index["sha256"]
    ):
        raise RuntimeError(f"Consolidated index differs from its completion marker: {index_path}")


def _validate_completion_contract(completion: Mapping, *, state_name: str, expected_count: int) -> None:
    common = {
        "cache_schema_version",
        "consolidated_index",
        "count",
        "feature",
        "manifest_sha256",
        "ordered_index_sha256",
        "selection",
        "spec_sha256",
        "total_npy_size_bytes",
    }
    expected_keys = (
        common | {"total_latent_frames"}
        if state_name == "latents"
        else common | {"total_source_frames", "total_target_frames"}
    )
    expected_feature = SEMANTIC_VAE_FEATURE if state_name == "latents" else VIDEO_40HZ_FEATURE
    index = completion.get("consolidated_index")
    if (
        set(completion) != expected_keys
        or completion.get("cache_schema_version") != 1
        or completion.get("feature") != expected_feature
        or completion.get("selection") != {"mode": "full"}
        or completion.get("count") != expected_count
        or not isinstance(index, dict)
        or set(index) != {"count", "path", "sha256", "size_bytes"}
    ):
        raise RuntimeError(f"Invalid authoritative {state_name} completion contract")


def _validate_cache_spec(
    cache_root: Path,
    *,
    state_name: str,
    completion: Mapping,
    inventory_meta_path: Path,
    inventory_meta_sha256: str,
    inventory_manifest_path: Path,
    inventory_manifest_sha256: str,
    inventory_count: int,
    total_frames: int,
) -> tuple[Path, str]:
    """Bind a completion marker to the exact codec/interpolation provenance."""

    spec_path = safe_join(cache_root, f"state/{state_name}/spec.json")
    spec = _load_regular_json(spec_path)
    spec_sha256 = sha256_file(spec_path)
    if completion.get("spec_sha256") != spec_sha256:
        raise RuntimeError(f"{state_name} completion marker does not authenticate its immutable spec")
    manifest_contract = spec.get("manifest")
    if not isinstance(manifest_contract, dict):
        raise TypeError(f"{state_name} spec has no manifest contract")
    try:
        spec_manifest_path = Path(manifest_contract["path"]).resolve(strict=True)
        spec_inventory_meta_path = Path(manifest_contract["inventory_metadata_path"]).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise RuntimeError(f"{state_name} spec contains invalid manifest paths") from error
    if (
        spec.get("cache_schema_version") != 1
        or manifest_contract.get("count") != inventory_count
        or manifest_contract.get("sha256") != inventory_manifest_sha256
        or manifest_contract.get("inventory_metadata_sha256") != inventory_meta_sha256
        or spec_manifest_path != inventory_manifest_path
        or spec_inventory_meta_path != inventory_meta_path
    ):
        raise RuntimeError(f"{state_name} spec is not bound to the authoritative CelebVDub inventory")

    if state_name == "latents":
        checkpoint = spec.get("checkpoint")
        extraction = spec.get("extraction")
        source = spec.get("semantic_vae_source")
        golden = extraction.get("golden_self_test") if isinstance(extraction, dict) else None
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("ema_sha256") != SEMANTIC_VAE_EMA_SHA256
            or checkpoint.get("ema_step") != 1_000_014
            or checkpoint.get("config_sha256") != SEMANTIC_VAE_CONFIG_SHA256
            or checkpoint.get("metainfo_sha256") != SEMANTIC_VAE_METAINFO_SHA256
            or not isinstance(extraction, dict)
            or extraction.get("protocol") != SEMANTIC_VAE_FEATURE
            or extraction.get("dtype") != "float32"
            or extraction.get("latent_dim") != SEMANTIC_VAE_LATENT_DIM
            or extraction.get("latent_layout") != "[time,channel]"
            or extraction.get("posterior_noise") != "torch.randn with per-utterance CUDA Generator"
            or extraction.get("sample_rate") != 16_000
            or extraction.get("vae_hop_length") != 400
            or not isinstance(golden, dict)
            or golden.get("utterance_key") != "celebvdub/train/saP2eLOlPAc/0_0_0"
            or golden.get("raw_latent_sha256") != SEMANTIC_VAE_GOLDEN_LATENT_SHA256
            or not isinstance(source, dict)
            or source.get("commit") != SEMANTIC_VAE_SOURCE_COMMIT
            or source.get("working_tree_clean") is not True
            or source.get("bigvgan_config_sha256") != SEMANTIC_VAE_BIGVGAN_CONFIG_SHA256
        ):
            raise RuntimeError("Semantic-VAE latent spec uses the wrong checkpoint, protocol, or numerics")
    elif state_name == "video_40hz":
        expected_interpolation = {
            "align_corners": False,
            "input_dtype": "float32",
            "input_frame_rate_hz": 25,
            "input_layout": "[time,channel]",
            "mode": "linear",
            "output_dtype": "float32",
            "output_frame_rate_hz": 40,
            "output_layout": "[time,channel]",
            "size": "record.latent_frames",
            "video_dim": CELEBVDUB_VIDEO_DIM,
        }
        if (
            spec.get("feature") != VIDEO_40HZ_FEATURE
            or spec.get("interpolation") != expected_interpolation
            or manifest_contract.get("total_target_frames") != total_frames
        ):
            raise RuntimeError("40 Hz video spec uses the wrong feature or interpolation protocol")
    else:
        raise ValueError(f"Unknown cache spec state: {state_name}")
    return spec_path, spec_sha256


def _bind_selected_records_to_index(
    cache_root: Path,
    completion: Mapping,
    inventory_manifest_path: Path,
    selected_records: Mapping[str, Mapping],
    *,
    modality: str,
) -> dict[str, tuple[int, str]]:
    """Authenticate every selected manifest path/length through the full cache index."""

    index_contract = completion["consolidated_index"]
    index_path = safe_join(cache_root, index_contract["path"])
    expected_schema = (
        {"feature", "latent_dim", "latent_frames", "relative_path", "sha256", "size_bytes", "utterance_key"}
        if modality == "latent"
        else {
            "feature",
            "relative_path",
            "sha256",
            "size_bytes",
            "source_frames",
            "source_sha256",
            "source_size_bytes",
            "target_frames",
            "utterance_key",
            "video_dim",
        }
    )
    bound: dict[str, tuple[int, str]] = {}
    seen: set[str] = set()
    seen_paths: set[str] = set()
    count = 0
    total_frames = 0
    total_source_frames = 0
    total_size_bytes = 0
    sentinel = object()
    paired_records = zip_longest(read_jsonl(index_path), read_jsonl(inventory_manifest_path), fillvalue=sentinel)
    for entry, inventory_record in paired_records:
        if entry is sentinel or inventory_record is sentinel:
            raise RuntimeError(f"{modality} index cardinality differs from the full inventory manifest")
        if set(entry) != expected_schema:
            raise ValueError(f"Unexpected {modality} index schema for {entry.get('utterance_key')!r}")
        key = entry.get("utterance_key")
        relative_path = entry.get("relative_path")
        relative = Path(relative_path) if isinstance(relative_path, str) else Path()
        expected_root = "latents" if modality == "latent" else "video_40hz"
        if (
            not isinstance(key, str)
            or key in seen
            or not isinstance(relative_path, str)
            or relative_path in seen_paths
            or relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != expected_root
            or relative.suffix != ".npy"
            or relative.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"Invalid/duplicate {modality} index key: {key!r}")
        seen.add(key)
        seen_paths.add(relative_path)
        count += 1
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        frames = entry.get("latent_frames" if modality == "latent" else "target_frames")
        if (
            type(size_bytes) is not int
            or size_bytes <= 0
            or type(frames) is not int
            or frames <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError(f"Invalid {modality} index values for {key}")
        try:
            int(sha256, 16)
        except ValueError as error:
            raise ValueError(f"Invalid {modality} SHA256 for {key}: {sha256!r}") from error
        total_frames += frames
        total_size_bytes += size_bytes
        if not isinstance(inventory_record, dict) or inventory_record.get("utterance_key") != key:
            raise RuntimeError(f"{modality} index order/key differs from the full inventory at record {count}")
        if modality == "latent":
            full_record_valid = (
                entry.get("feature") == SEMANTIC_VAE_FEATURE
                and entry.get("latent_dim") == SEMANTIC_VAE_LATENT_DIM
                and relative_path == inventory_record.get("latent_relative_path")
                and frames == inventory_record.get("latent_frames")
            )
        else:
            source_sha256 = entry.get("source_sha256")
            source_size_bytes = entry.get("source_size_bytes")
            source_frames = entry.get("source_frames")
            try:
                source_sha_valid = (
                    isinstance(source_sha256, str) and len(source_sha256) == 64 and int(source_sha256, 16) >= 0
                )
            except ValueError:
                source_sha_valid = False
            full_record_valid = (
                entry.get("feature") == VIDEO_40HZ_FEATURE
                and entry.get("video_dim") == CELEBVDUB_VIDEO_DIM
                and relative_path == inventory_record.get("video_40hz_relative_path")
                and source_frames == inventory_record.get("video_frames_25hz")
                and frames == inventory_record.get("latent_frames")
                and source_sha_valid
                and type(source_size_bytes) is int
                and source_size_bytes > 0
            )
            total_source_frames += source_frames if type(source_frames) is int else 0
        if not full_record_valid:
            raise RuntimeError(f"{modality} cache index disagrees with full manifest record {key}")
        selected = selected_records.get(key)
        if selected is None:
            continue
        if modality == "latent":
            valid = (
                entry.get("feature") == SEMANTIC_VAE_FEATURE
                and entry.get("latent_dim") == SEMANTIC_VAE_LATENT_DIM
                and entry.get("relative_path") == selected.get("latent_relative_path")
                and frames == selected.get("latent_frames")
            )
        else:
            valid = (
                entry.get("feature") == VIDEO_40HZ_FEATURE
                and entry.get("video_dim") == CELEBVDUB_VIDEO_DIM
                and entry.get("relative_path") == selected.get("video_40hz_relative_path")
                and entry.get("source_frames") == selected.get("video_frames_25hz")
                and frames == selected.get("latent_frames")
            )
        if not valid:
            raise RuntimeError(f"{modality} cache index disagrees with selected manifest record {key}")
        bound[key] = (size_bytes, sha256)

    expected_count = int(completion["count"])
    expected_frames = int(completion["total_latent_frames" if modality == "latent" else "total_target_frames"])
    if (
        count != expected_count
        or len(seen) != expected_count
        or total_frames != expected_frames
        or total_size_bytes != int(completion["total_npy_size_bytes"])
    ):
        raise RuntimeError(f"{modality} consolidated index totals disagree with its completion marker")
    if modality == "video" and total_source_frames != int(completion["total_source_frames"]):
        raise RuntimeError("video consolidated index source-frame total disagrees with its completion marker")
    missing = set(selected_records) - set(bound)
    if missing:
        raise RuntimeError(f"{modality} index is missing selected records: {sorted(missing)[:10]}")
    return bound


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


class SemanticVaeCelebVDubDataset(Dataset):
    """Strict C2 training view over fixed 40 Hz audio and video caches.

    The selected manifest is the immutable CTC-feasible subset of the full
    CelebV-Dub inventory. Semantic-VAE values are normalized with the fixed
    LibriSpeech-train statistics on valid frames only; collation then pads both
    modalities with exact zeros.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        normalization_path: str | Path,
        vocab_path: str | Path,
        *,
        expected_normalization_sha256: str = LIBRISPEECH_SVAE_TRAIN_NORMALIZATION_SHA256,
        expected_record_count: int | None = CELEBVDUB_CTC40_TRAIN_COUNT,
        expected_inventory_count: int = CELEBVDUB_INVENTORY_COUNT,
    ):
        manifest_path = Path(manifest_path).expanduser().absolute()
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(f"Training manifest must be a regular file: {manifest_path}")
        self.manifest_path = manifest_path.resolve(strict=True)
        if self.manifest_path.name != "train_ctc40_valid.jsonl":
            raise ValueError(
                "S3 requires the immutable CTC-feasible manifest train_ctc40_valid.jsonl, "
                f"got {self.manifest_path.name}"
            )

        cache_root = Path(cache_root).expanduser().absolute()
        if not cache_root.is_dir() or cache_root.is_symlink():
            raise FileNotFoundError(f"Cache root must be a regular directory: {cache_root}")
        self.cache_root = cache_root.resolve(strict=True)

        inventory_meta_path = self.manifest_path.parent / "inventory_meta.json"
        inventory_meta = _load_regular_json(inventory_meta_path)
        manifests = inventory_meta.get("manifests")
        if not isinstance(manifests, dict):
            raise TypeError("Inventory metadata does not contain a manifests mapping")
        selected_entry = manifests.get(self.manifest_path.name)
        inventory_entry = manifests.get("inventory.jsonl")
        if not isinstance(selected_entry, dict) or not isinstance(inventory_entry, dict):
            raise TypeError("Inventory metadata is missing the CTC-valid/full manifest entries")
        records = list(read_jsonl(self.manifest_path))
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if selected_entry.get("sha256") != self.manifest_sha256 or selected_entry.get("count") != len(records):
            raise RuntimeError("CTC-valid manifest differs from its immutable inventory metadata")
        selected_count = int(selected_entry["count"])
        if expected_record_count is not None and selected_count != expected_record_count:
            raise RuntimeError(f"Expected {expected_record_count} CTC-feasible records, found {selected_count}")
        inventory_count = int(inventory_entry["count"])
        inventory_sha256 = inventory_entry.get("sha256")
        total_inventory_frames = int(inventory_meta["total_latent_frames"])
        inventory_manifest_path = safe_join(self.manifest_path.parent.parent, inventory_entry["path"])
        if (
            inventory_count != expected_inventory_count
            or not inventory_manifest_path.is_file()
            or inventory_manifest_path.is_symlink()
            or inventory_manifest_path.stat().st_size != inventory_entry.get("size_bytes")
            or sha256_file(inventory_manifest_path) != inventory_sha256
            or inventory_meta.get("base_posterior_seed") != 666
            or inventory_meta.get("latent_spec")
            != {
                "dimension": SEMANTIC_VAE_LATENT_DIM,
                "dtype": "float32",
                "frame_rate_hz": 40.0,
                "hop_length_samples": 400,
                "mode": "fixed_posterior_sample",
                "sample_rate": 16_000,
            }
        ):
            raise RuntimeError("CelebVDub inventory is not the authoritative fixed Semantic-VAE/seed-666 contract")
        self.inventory_meta_path = inventory_meta_path
        self.inventory_meta_sha256 = sha256_file(inventory_meta_path)

        selected_records: dict[str, Mapping] = {}
        for record in records:
            key = record.get("utterance_key")
            if not isinstance(key, str) or key in selected_records:
                raise ValueError(f"Invalid/duplicate CTC-valid manifest key: {key!r}")
            selected_records[key] = record

        latent_complete_path = safe_join(self.cache_root, "state/latents/complete.json")
        video_complete_path = safe_join(self.cache_root, "state/video_40hz/complete.json")
        for state_name in ("latents", "video_40hz"):
            write_guard = safe_join(self.cache_root, f"state/{state_name}/WRITE_ACTIVE.json")
            if write_guard.exists() or write_guard.is_symlink():
                raise RuntimeError(f"Refusing to train while the {state_name} cache has an active/stale writer")
        latent_complete = _load_regular_json(latent_complete_path)
        video_complete = _load_regular_json(video_complete_path)
        _validate_completion_contract(latent_complete, state_name="latents", expected_count=inventory_count)
        _validate_completion_contract(video_complete, state_name="video_40hz", expected_count=inventory_count)
        if (
            latent_complete.get("cache_schema_version") != 1
            or latent_complete.get("feature") != SEMANTIC_VAE_FEATURE
            or latent_complete.get("selection") != {"mode": "full"}
            or latent_complete.get("count") != inventory_count
            or latent_complete.get("manifest_sha256") != inventory_sha256
            or latent_complete.get("total_latent_frames") != total_inventory_frames
        ):
            raise RuntimeError("Semantic-VAE completion marker is not the authoritative full inventory cache")
        if (
            video_complete.get("cache_schema_version") != 1
            or video_complete.get("feature") != VIDEO_40HZ_FEATURE
            or video_complete.get("selection") != {"mode": "full"}
            or video_complete.get("count") != inventory_count
            or video_complete.get("manifest_sha256") != inventory_sha256
            or video_complete.get("total_target_frames") != total_inventory_frames
        ):
            raise RuntimeError("40 Hz video completion marker is not the authoritative full inventory cache")
        _validate_consolidated_index(self.cache_root, latent_complete, inventory_count)
        _validate_consolidated_index(self.cache_root, video_complete, inventory_count)
        self.latent_complete_sha256 = sha256_file(latent_complete_path)
        self.video_complete_sha256 = sha256_file(video_complete_path)
        self.latent_spec_path, self.latent_spec_sha256 = _validate_cache_spec(
            self.cache_root,
            state_name="latents",
            completion=latent_complete,
            inventory_meta_path=self.inventory_meta_path,
            inventory_meta_sha256=self.inventory_meta_sha256,
            inventory_manifest_path=inventory_manifest_path,
            inventory_manifest_sha256=inventory_sha256,
            inventory_count=inventory_count,
            total_frames=total_inventory_frames,
        )
        self.video_spec_path, self.video_spec_sha256 = _validate_cache_spec(
            self.cache_root,
            state_name="video_40hz",
            completion=video_complete,
            inventory_meta_path=self.inventory_meta_path,
            inventory_meta_sha256=self.inventory_meta_sha256,
            inventory_manifest_path=inventory_manifest_path,
            inventory_manifest_sha256=inventory_sha256,
            inventory_count=inventory_count,
            total_frames=total_inventory_frames,
        )
        latent_index = _bind_selected_records_to_index(
            self.cache_root,
            latent_complete,
            inventory_manifest_path,
            selected_records,
            modality="latent",
        )
        video_index = _bind_selected_records_to_index(
            self.cache_root,
            video_complete,
            inventory_manifest_path,
            selected_records,
            modality="video",
        )
        for key, record in selected_records.items():
            record["_latent_index_size_bytes"], record["_latent_index_sha256"] = latent_index[key]
            record["_video_index_size_bytes"], record["_video_index_sha256"] = video_index[key]

        normalization_path = Path(normalization_path).expanduser().absolute()
        if not normalization_path.is_file() or normalization_path.is_symlink():
            raise FileNotFoundError(f"Normalization statistics must be a regular file: {normalization_path}")
        self.normalization_path = normalization_path.resolve(strict=True)
        self.normalization_sha256 = sha256_file(self.normalization_path)
        if self.normalization_sha256 != expected_normalization_sha256:
            raise RuntimeError(
                "Normalization statistics are not the fixed LibriSpeech-train artifact: "
                f"expected {expected_normalization_sha256}, got {self.normalization_sha256}"
            )
        normalization = _load_regular_json(self.normalization_path)
        required_normalization_keys = {
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
        if set(normalization) != required_normalization_keys:
            raise ValueError(f"Unexpected normalization-statistics keys: {sorted(normalization)}")
        if (
            normalization["cache_schema_version"] != 1
            or normalization["channel_count"] != SEMANTIC_VAE_LATENT_DIM
            or normalization["feature"] != SEMANTIC_VAE_FEATURE
            or normalization["method"] != "per_channel_population_mean_std_float64_welford_v1"
            or normalization["scope"] != "train"
        ):
            raise ValueError("Invalid LibriSpeech Semantic-VAE normalization contract")
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
        self.latent_mean = mean.astype(np.float32)
        self.latent_std = std.astype(np.float32)

        vocab_path = Path(vocab_path).expanduser().absolute()
        self.vocabulary, self.vocab_sha256 = _load_vocab(vocab_path)
        self.vocab_path = vocab_path.resolve(strict=True)
        ctc40_preflight = inventory_meta.get("ctc40_preflight")
        if (
            not isinstance(ctc40_preflight, dict)
            or ctc40_preflight.get("vocab_sha256") != self.vocab_sha256
            or ctc40_preflight.get("train_valid") != selected_count
            or ctc40_preflight.get("train_excluded") != 105
        ):
            raise RuntimeError("CTC-valid manifest is not bound to the configured CelebV-Dub vocabulary")

        self.records = records
        if len(self.records) != selected_count:
            raise RuntimeError("Loaded CTC-valid record count changed after manifest verification")
        seen: set[str] = set()
        for record in self.records:
            key = record.get("utterance_key")
            if record.get("split") != "train" or not isinstance(key, str) or key in seen:
                raise ValueError(f"Invalid/duplicate CTC-valid training record: {key}")
            seen.add(key)
            text = record.get("text")
            if not isinstance(text, str) or not text:
                raise TypeError(f"Invalid transcript for {key}")
            target_length, adjacent_repeats, minimum_frames = _ctc_record_fields(text, self.vocabulary)
            if (
                record.get("ctc_target_length") != target_length
                or record.get("ctc_adjacent_repeats") != adjacent_repeats
                or record.get("ctc_min_input_frames") != minimum_frames
                or record.get("ctc_feasible_40hz") is not True
                or int(record.get("latent_frames", 0)) < minimum_frames
            ):
                raise ValueError(f"Invalid 40 Hz CTC feasibility contract for {key}")
            video_relative_path = record.get("video_40hz_relative_path")
            if not isinstance(video_relative_path, str):
                raise TypeError(f"Missing 40 Hz video cache path for {key}")

    def __len__(self):
        return len(self.records)

    def get_frame_len(self, index):
        return int(self.records[index]["latent_frames"])

    def __getitem__(self, index):
        record = self.records[index]
        frames = int(record["latent_frames"])
        latent_path = safe_join(self.cache_root, record["latent_relative_path"])
        video_path = safe_join(self.cache_root, record["video_40hz_relative_path"])
        for modality, path, expected_size in (
            ("Semantic-VAE latent", latent_path, record["_latent_index_size_bytes"]),
            ("40 Hz video", video_path, record["_video_index_size_bytes"]),
        ):
            if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size:
                raise FileNotFoundError(
                    f"{modality} file differs from its authenticated index for {record['utterance_key']}: {path}"
                )
        latent = np.load(latent_path, allow_pickle=False)
        video = np.load(video_path, allow_pickle=False)
        if latent.shape != (frames, SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(
                f"Invalid Semantic-VAE latent for {record['utterance_key']}: {latent.shape}/{latent.dtype}"
            )
        if video.shape != (frames, CELEBVDUB_VIDEO_DIM) or video.dtype != np.float32:
            raise ValueError(f"Invalid 40 Hz video for {record['utterance_key']}: {video.shape}/{video.dtype}")
        if not np.isfinite(latent).all() or not np.isfinite(video).all():
            raise FloatingPointError(f"Non-finite cached feature for {record['utterance_key']}")
        normalized_latent = (latent - self.latent_mean) / self.latent_std
        if not np.isfinite(normalized_latent).all():
            raise FloatingPointError(f"Non-finite normalized latent for {record['utterance_key']}")
        return {
            "ctc_min_input_frames": int(record["ctc_min_input_frames"]),
            "ctc_target_length": int(record["ctc_target_length"]),
            "mel_spec": torch.from_numpy(normalized_latent).transpose(0, 1),
            "text": record["text"],
            "utterance_key": record["utterance_key"],
            "video": torch.from_numpy(video.copy()),
        }

    @staticmethod
    def collate_fn(batch):
        if not batch:
            raise ValueError("Cannot collate an empty Semantic-VAE CelebV-Dub batch")
        lengths = torch.tensor([item["mel_spec"].shape[1] for item in batch], dtype=torch.long)
        video_lengths = torch.tensor([item["video"].shape[0] for item in batch], dtype=torch.long)
        if not torch.equal(lengths, video_lengths):
            raise ValueError(f"Audio/video length mismatch in batch: {lengths.tolist()} != {video_lengths.tolist()}")
        target_lengths = torch.tensor([item["ctc_target_length"] for item in batch], dtype=torch.long)
        minimum_input_frames = torch.tensor([item["ctc_min_input_frames"] for item in batch], dtype=torch.long)
        if torch.any(lengths < minimum_input_frames):
            raise ValueError("CTC-infeasible record reached Semantic-VAE CelebV-Dub collation")

        max_length = int(lengths.max())
        latent = torch.stack(
            [F.pad(item["mel_spec"], (0, max_length - item["mel_spec"].shape[1]), value=0.0) for item in batch]
        )
        video = torch.stack(
            [F.pad(item["video"], (0, 0, 0, max_length - item["video"].shape[0]), value=0.0) for item in batch]
        )
        positions = torch.arange(max_length).unsqueeze(0)
        audio_mask = positions < lengths.unsqueeze(1)
        video_mask = positions < video_lengths.unsqueeze(1)
        if not torch.equal(audio_mask, video_mask):
            raise RuntimeError("Aligned 40 Hz audio/video masks unexpectedly differ")
        return {
            "audio_mask": audio_mask,
            "ctc_min_input_frames": minimum_input_frames,
            "mel": latent,
            "mel_lengths": lengths,
            "text": [item["text"] for item in batch],
            "text_lengths": target_lengths,
            "utterance_keys": [item["utterance_key"] for item in batch],
            "video": video,
            "video_lengths": video_lengths,
            "video_mask": video_mask,
        }


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
