"""Independently audit all Chem caches and exercise the real training loader."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from aligndit.model.repa import validate_repa_feature_array
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset, sha256_file
from aligndit.model.speaker_embedding import validate_speaker_embedding_array


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = "finetune_chem_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus"


def audit(config, workers: int) -> dict:
    data = OmegaConf.to_container(config.datasets, resolve=True)
    parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset = SemanticVaeCelebVDubDataset(**{key: value for key, value in data.items() if key in parameters})
    cache = Path(data["cache_root"]).resolve()
    inventory_path = cache / "manifests/inventory.jsonl"
    assert sha256_file(inventory_path) == data["expected_inventory_sha256"], "Inventory digest mismatch"
    records = [json.loads(line) for line in inventory_path.read_text().splitlines() if line.strip()]
    assert Counter(row["split"] for row in records) == data["expected_split_counts"], "Split counts changed"
    assert len(records) == len({row["utterance_key"] for row in records}), "Duplicate utterances"
    vocabulary = set(Path(data["vocab_path"]).read_text().splitlines())
    groups = {
        split: {row["video_id"] for row in records if row["split"] == split} for split in ("train", "val", "test")
    }
    assert not groups["val"] & (groups["train"] | groups["test"]), "Validation video leakage"

    def check(row):
        try:
            key, frames = row["utterance_key"], row["latent_frames"]
            assert row["text"] and not set(row["text"]) - vocabulary, "Empty/unknown transcript"
            source = sf.info(row["audio_path"])
            assert source.samplerate == 16000 and source.channels == 1, "Invalid source audio format"
            assert source.frames == row["original_num_samples"], "Changed source audio length"
            assert sha256_file(row["audio_path"]) == row["audio_sha256"], "Changed source audio content"
            assert frames == (source.frames + 399) // 400, "Incorrect 40 Hz latent duration"
            assert 0 <= frames * 400 - source.frames < 400, "Incorrect encoder right padding"
            latent = np.load(cache / row["latent_relative_path"], allow_pickle=False)
            video = np.load(cache / row["video_40hz_relative_path"], allow_pickle=False)
            assert latent.dtype == video.dtype == np.float32, "Latent/video must be float32"
            assert latent.shape == (frames, 64), f"Invalid latent shape: {latent.shape}"
            assert video.shape == (frames, 1024), f"Invalid video shape: {video.shape}"
            assert np.isfinite(latent).all() and np.isfinite(video).all(), "Nonfinite latent/video"
            normalized = (latent - dataset.latent_mean) / dataset.latent_std
            assert np.isfinite(normalized).all(), "Nonfinite normalized latent"
            assert np.std(video) > 1e-6, "Constant video feature"
            relative = Path(row["audio_relative_path"]).with_suffix(".npy")
            speaker_path = Path(data["speaker_embedding_cache_dir"]) / relative
            speaker = np.load(speaker_path, allow_pickle=False)
            validate_speaker_embedding_array(speaker, source=speaker_path)
            repa_frames = 0
            if row["split"] == "train":
                repa_path = Path(data["repa_feature_cache_dir"]) / relative
                repa = np.load(repa_path, allow_pickle=False)
                validate_repa_feature_array(repa, source=repa_path)
                # Pinned WavLM-Base+ seven convolutional strides/kernels.
                expected = source.frames
                for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
                    expected = (expected - kernel) // stride + 1
                assert len(repa) == expected, "WavLM cache does not cover the complete waveform"
                repa_frames = len(repa)
            return {"key": key, "split": row["split"], "latent_frames": frames, "repa_frames": repa_frames}
        except (AssertionError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            return {"key": row["utterance_key"], "split": row["split"], "error": str(error)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(check, records), total=len(records), desc="Chem multimodal audit"))
    errors = [row for row in results if "error" in row]
    good = [row for row in results if "error" not in row]
    if not errors:
        indices = sorted(range(len(dataset)), key=dataset.get_frame_len)
        selected = list(dict.fromkeys((indices[0], indices[-1], indices[len(indices) // 2])))
        batch = dataset.collate_fn([dataset[index] for index in selected])
        assert torch.equal(batch["mel_lengths"], batch["video_lengths"]), "Collation alignment failed"
        for field in ("mel", "video", "speaker_embedding", "repa_features"):
            assert torch.isfinite(batch[field]).all(), f"Nonfinite collated {field}"
    report = {
        "passed": not errors,
        "errors": errors,
        "cache_root": str(cache),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "latent_video_speaker_checked": len(good),
        "repa_training_checked": sum(row["split"] == "train" for row in good),
        "latent_frames": sum(row["latent_frames"] for row in good),
        "train_ctc_feasible": dataset.ctc_feasible_count,
        "train_ctc_infeasible_retained": dataset.ctc_infeasible_count,
        "normalization_sha256": data["expected_normalization_sha256"],
        "normalization_fitted_on": "LibriSpeech train; unchanged from source model",
        "manifest_sha256": data["expected_manifest_sha256"],
        "official_train_test_video_overlap": sorted(groups["train"] & groups["test"]),
        "alignment_scope": "Reuses baseline Chem global offset 0 estimate; not a per-clip synchronization proof",
        "real_loader_short_long_collation_passed": not errors,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=CONFIG)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=PROJECT / "output/chem_multimodal_audit.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.chdir(PROJECT)
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(PROJECT / "src/aligndit/config"), version_base="1.3"):
        config = compose(config_name=args.config_name)
    report = audit(config, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    if not report["passed"]:
        raise SystemExit(f"Chem multimodal audit failed: {len(report['errors'])} errors")


if __name__ == "__main__":
    main()
