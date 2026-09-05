"""Read-only CPU audit of the 213 Setting 1 speaker-conditioned references.

Validate speaker cache coverage/identity, prompt provenance, normalization and
failure guards without loading generation checkpoints or creating any outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from aligndit.model.speaker_embedding import SpeakerEmbeddingError
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    load_composed_config,
    load_normalization,
    load_setting1_speaker_embeddings,
    load_test_records,
    sha256_file,
    validate_record_arrays,
)


def expect_failure(function, error_type):
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"Expected guard to raise {error_type.__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parents[2]
        / "config/finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup.yaml",
    )
    parser.add_argument(
        "--test-list", type=Path,
        default=Path(f"{os.environ.get('ROOT_PREFIX', '')}/zjw524/projects/data/celebvdub_test_s1.lst"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_composed_config(args.config.resolve(strict=True))
    assert int(config.model.arch.speaker_dim) == 192
    assert int(config.model.arch.speaker_condition_start_layer) == 6
    cache_root = Path(config.datasets.cache_root)
    records = load_test_records(cache_root, args.test_list)
    assert sha256_file(config.datasets.normalization_path) == config.datasets.expected_normalization_sha256
    assert sha256_file(config.datasets.vocab_path) == config.datasets.expected_vocab_sha256
    mean, std, _ = load_normalization(Path(config.datasets.normalization_path))
    embeddings, metadata = load_setting1_speaker_embeddings(config, records)
    assert len(embeddings) == len(records) == len(metadata["sources"]) == 213
    speaker_batch = torch.stack(embeddings)
    assert speaker_batch.shape == (213, 192) and speaker_batch.dtype == torch.float32
    assert not speaker_batch.requires_grad
    assert torch.allclose(speaker_batch.norm(dim=1), torch.ones(213), atol=1e-4, rtol=0)
    assert metadata["conditioned_blocks_zero_based"] == list(range(6, 18))
    assert "same GT clip" in metadata["reference_protocol"]
    for row, source in zip(records, metadata["sources"]):
        latent, video = validate_record_arrays(row, cache_root, mean, std)
        assert latent.shape == (row["latent_frames"], 64)
        assert video.shape == (row["latent_frames"], 1024)
        assert source["utterance_key"] == row["utterance_key"]
        assert Path(source["prompt_audio"]).is_file()
        assert Path(source["speaker_cache"]).is_file()

    mismatched_record = copy.deepcopy(records[0])
    mismatched_record["audio_relative_path"] = records[1]["audio_relative_path"]
    expect_failure(lambda: load_setting1_speaker_embeddings(config, [mismatched_record]), ValueError)
    unsafe_record = copy.deepcopy(records[0])
    unsafe_record["audio_relative_path"] = "../train/unrelated.wav"
    expect_failure(lambda: load_setting1_speaker_embeddings(config, [unsafe_record]), ValueError)
    wrong_dim = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    wrong_dim.datasets.speaker_embedding_dim = 64
    expect_failure(lambda: load_setting1_speaker_embeddings(wrong_dim, records[:1]), ValueError)
    wrong_model = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    wrong_model.datasets.speaker_embedding_checkpoint_sha256 = "0" * 64
    expect_failure(lambda: load_setting1_speaker_embeddings(wrong_model, records[:1]), SpeakerEmbeddingError)
    missing_cache = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    missing_cache.datasets.speaker_embedding_cache_dir = None
    expect_failure(lambda: load_setting1_speaker_embeddings(missing_cache, records[:1]), ValueError)
    plain = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    del plain.model.arch.speaker_dim
    assert load_setting1_speaker_embeddings(plain, records) == ([None] * 213, None)

    print(json.dumps({
        "result": "PASS",
        "test_records": len(records),
        "speaker_shape": list(speaker_batch.shape),
        "reference_protocol": metadata["reference_protocol"],
        "speaker_cache": metadata["cache_dir"],
        "metadata_sha256": metadata["metadata_sha256"],
        "conditioned_blocks_zero_based": metadata["conditioned_blocks_zero_based"],
        "latent_video_shapes_and_lengths_valid": True,
        "guard_checks": ["prompt_mismatch", "unsafe_path", "wrong_dimension", "wrong_extractor", "missing_cache"],
        "no_speaker_backwards_compatibility": True,
        "training_updates": 0,
        "output_files_written": 0,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
