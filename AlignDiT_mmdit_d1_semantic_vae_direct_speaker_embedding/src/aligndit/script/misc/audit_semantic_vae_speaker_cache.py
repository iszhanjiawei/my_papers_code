"""CPU-only checks for the Semantic-VAE + frozen CAM++ training data path.

Run once with ``--full-audit`` before starting the distributed training job.
This never extracts embeddings, modifies the caches, or loads a model.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup",
    )
    parser.add_argument("--full-audit", action="store_true", help="Read and validate all 79,613 training vectors")
    args = parser.parse_args()
    started = time.monotonic()
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=args.config_name)
    dataset_config = OmegaConf.to_container(config.datasets, resolve=True)
    parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset_kwargs = {key: value for key, value in dataset_config.items() if key in parameters}
    dataset = SemanticVaeCelebVDubDataset(**dataset_kwargs)
    if dataset.speaker_embedding_cache_dir is None:
        raise AssertionError("This audit requires a configured speaker embedding cache")
    if config.model.arch.speaker_dim != dataset.speaker_embedding_dim:
        raise AssertionError("Model and cache speaker dimensions differ")
    examples = [dataset[index] for index in (0, len(dataset) // 2, len(dataset) - 1)]
    batch = dataset.collate_fn(examples)
    assert batch["speaker_embedding"].shape == (3, 192)
    assert batch["speaker_embedding"].dtype == torch.float32
    assert batch["mel"].shape[1] == 64
    assert batch["video"].shape[2] == 1024
    assert torch.equal(batch["mel_lengths"], batch["video_lengths"])
    assert torch.allclose(batch["speaker_embedding"].norm(dim=1), torch.ones(3), atol=1e-4, rtol=0)
    assert not batch["speaker_embedding"].requires_grad

    # Optional speaker conditioning must not change existing feature values,
    # frame lengths, padding or CTC metadata. Reuse the same dataset caches.
    speaker_cache_dir = dataset.speaker_embedding_cache_dir
    dataset.speaker_embedding_cache_dir = None
    plain_examples = [dataset[index] for index in (0, len(dataset) // 2, len(dataset) - 1)]
    dataset.speaker_embedding_cache_dir = speaker_cache_dir
    plain_batch = dataset.collate_fn(plain_examples)
    assert "speaker_embedding" not in plain_batch
    for key, value in plain_batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, batch[key]), key
        else:
            assert value == batch[key], key
    try:
        dataset.collate_fn([examples[0], plain_examples[1]])
    except RuntimeError as error:
        assert "speaker_embedding must be present" in str(error)
    else:
        raise AssertionError("Collation accepted mixed presence of speaker embeddings")

    report = {
        "config_name": args.config_name,
        "dataset_count": len(dataset),
        "ctc_feasible_count": dataset.ctc_feasible_count,
        "ctc_infeasible_count": dataset.ctc_infeasible_count,
        "batch_shape": {key: list(batch[key].shape) for key in ("mel", "video", "speaker_embedding")},
        "baseline_features_identical": True,
        "mixed_speaker_batch_rejected": True,
        "contract": dataset.speaker_embedding_contract,
    }
    if args.full_audit:
        print("Validating all 79,613 cached training speaker embeddings...", flush=True)
        audit_started = time.monotonic()
        report["full_train_audit"] = dataset.audit_speaker_embedding_cache()
        report["full_audit_seconds"] = round(time.monotonic() - audit_started, 3)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
