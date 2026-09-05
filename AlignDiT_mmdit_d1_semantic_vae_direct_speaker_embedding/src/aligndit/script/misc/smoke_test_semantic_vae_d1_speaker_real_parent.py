"""Validate the actual S2c EMA parent and real speaker/latent batches on one GPU.

This is a forward/backward integration test only: it does not update weights,
write checkpoints, or start a training job. Select the GPU with
CUDA_VISIBLE_DEVICES and retain stdout as the validation report.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectD1SpeakerTrainer
from aligndit.script.train.finetune_semantic_vae_d1_direct_speaker import build_model, validate_experiment_config
from f5_tts.model.utils import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup"
    )
    parser.add_argument("--max-frames", type=int, default=160)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The real-parent integration test requires a CUDA device")
    device = torch.device("cuda:0")
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=args.config_name)
    validate_experiment_config(config)
    rejected_configurations = {
        "model.arch.speaker_dim": 64,
        "datasets.speaker_embedding_dim": 64,
        "model.arch.speaker_condition_start_layer": 12,
        "model.arch.layer_indices_ctc": [5],
        "model.arch.ctc_sampling_ratios": [2, 1],
        "model.ctc_lambda": 0.1,
        "model.ctc_warmup_end": 20000,
        "model.audio_representation.frame_rate": 25,
        "optim.run_until_update": 0,
        "ckpts.logger": "wandb",
    }
    for config_key, invalid_value in rejected_configurations.items():
        invalid = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        OmegaConf.update(invalid, config_key, invalid_value)
        try:
            validate_experiment_config(invalid)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"Invalid experiment configuration accepted: {config_key}={invalid_value}")
    print(f"[OK] {len(rejected_configurations)} incompatible configuration overrides rejected", flush=True)
    torch.manual_seed(int(config.seed))
    ckpts = config.ckpts
    print("Validating real S2c 70k parent artifact hashes...", flush=True)
    validate_parent_artifacts(
        ckpts.pretrained_path,
        ckpts.parent_contract_path,
        expected_checkpoint_sha256=ckpts.expected_parent_sha256,
        expected_checkpoint_size=ckpts.expected_parent_size,
        expected_contract_sha256=ckpts.expected_parent_contract_sha256,
    )

    dataset_config = OmegaConf.to_container(config.datasets, resolve=True)
    dataset_parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset = SemanticVaeCelebVDubDataset(
        **{key: value for key, value in dataset_config.items() if key in dataset_parameters}
    )
    selected = [
        index for index, record in enumerate(dataset.records)
        if record["ctc_feasible_40hz"] and 64 <= record["latent_frames"] <= args.max_frames
    ][:2]
    if len(selected) != 2:
        raise RuntimeError("Need two real CTC-feasible examples between 64 and --max-frames frames")
    batch = dataset.collate_fn([dataset[index] for index in selected])
    print(json.dumps({
        "selected_utterances": batch["utterance_keys"],
        "latent_lengths": batch["mel_lengths"].tolist(),
        "text_lengths": batch["text_lengths"].tolist(),
        "speaker_norms": batch["speaker_embedding"].norm(dim=1).tolist(),
    }), flush=True)
    assert batch["mel"].shape[1] == 64
    assert batch["speaker_embedding"].shape == (2, 192)
    assert torch.equal(batch["mel_lengths"], batch["video_lengths"])

    vocab_char_map, vocab_size = get_tokenizer(config.datasets.vocab_path, "custom")
    model = build_model(config, vocab_char_map, vocab_size)
    source_state, ema_step = load_s2c_ema_state(
        ckpts.pretrained_path,
        expected_parent_contract_sha256=ckpts.expected_parent_contract_sha256,
        expected_parent_update=ckpts.expected_parent_update,
    )
    migration_kwargs = {
        "parent_path": ckpts.pretrained_path,
        "parent_sha256": ckpts.expected_parent_sha256,
        "parent_size": ckpts.expected_parent_size,
        "parent_contract_sha256": ckpts.expected_parent_contract_sha256,
        "parent_ema_step": ema_step,
    }
    invalid_source = dict(source_state)
    invalid_source.pop(next(iter(invalid_source)))
    try:
        migrate_s2c_ema_into_model(model, invalid_source, **migration_kwargs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Migration accepted an incomplete parent state")
    with torch.no_grad():
        model.transformer.speaker_proj.weight[0, 0] = 1.0
    try:
        migrate_s2c_ema_into_model(model, source_state, **migration_kwargs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Migration accepted a nonzero speaker projection")
    finally:
        with torch.no_grad():
            model.transformer.speaker_proj.weight.zero_()
    del invalid_source
    migration = migrate_s2c_ema_into_model(
        model,
        source_state,
        **migration_kwargs,
    )
    assert migration.source_key_count == 313
    assert migration.target_key_count == 560
    assert migration.loaded_key_count == 303
    assert len(migration.ignored_source_keys) == 10
    assert len(migration.new_target_keys) == 257
    assert "transformer.speaker_proj.weight" in migration.new_target_keys
    assert not torch.count_nonzero(model.transformer.speaker_proj.weight)
    for key, value in model.state_dict().items():
        if key in source_state:
            assert torch.equal(value, source_state[key]), f"Migration changed parent tensor {key}"
    print("[OK] strict migration: source=313, target=560, loaded=303, ignored=10, new=257; loaded tensors exact", flush=True)
    del source_state
    gc.collect()

    trainer = SemanticVaeDirectD1SpeakerTrainer.__new__(SemanticVaeDirectD1SpeakerTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    trainer.ctc_target_lambda = 0.03
    trainer.ctc_warmup_start = 10000
    trainer.ctc_warmup_end = 30000
    for completed, expected_weight in ((0, 0.0), (9999, 0.0), (10000, 0.0000015), (19999, 0.015), (29999, 0.03)):
        trainer._before_update(completed)
        assert trainer.completed_updates == completed
        assert math.isclose(model.ctc_lambda, expected_weight, rel_tol=1e-7, abs_tol=1e-10)
    trainer.run_until_update = 200000
    assert not trainer._reached_run_limit(199999) and trainer._reached_run_limit(200000)
    diagnostics = trainer._forward_diagnostics(torch.tensor(1.03), {"diff_loss": 1.0, "ctc_loss": 1.0})
    assert math.isclose(diagnostics["ctc_weighted_loss"], 0.03)
    assert math.isclose(diagnostics["ctc_fraction_of_total"], 0.03 / 1.03, rel_tol=1e-6)
    try:
        trainer._forward_diagnostics(torch.tensor(float("nan")), {"diff_loss": 1.0})
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Speaker trainer accepted a non-finite loss")
    print("[OK] child-update CTC boundaries, speaker diagnostics, non-finite guard and 200k run limit", flush=True)

    model.to(device).train()
    forward_kwargs = {
        "inp": batch["mel"].permute(0, 2, 1).to(device),
        "text": batch["text"],
        "lens": batch["mel_lengths"].to(device),
        "text_lens": batch["text_lengths"].to(device),
        "video": batch["video"].to(device),
        "video_lens": batch["video_lengths"].to(device),
        "speaker_embedding": batch["speaker_embedding"].to(device),
    }
    assert not forward_kwargs["speaker_embedding"].requires_grad
    results = []
    for weight in (0.0, 0.015, 0.03):
        torch.manual_seed(int(config.seed) + 1)
        torch.cuda.manual_seed_all(int(config.seed) + 1)
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = weight
        torch.cuda.reset_peak_memory_stats(device)
        # Preserve configured dropout probabilities but force this validation
        # batch to take the full-conditioning branch so identity has a gradient.
        with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(**forward_kwargs)
        assert torch.isfinite(loss)
        assert all(math.isfinite(float(value)) for value in components.values())
        assert prediction.shape == forward_kwargs["inp"].shape
        assert torch.isfinite(prediction).all()
        if weight:
            assert components["ctc_loss"] > 0
        else:
            assert "ctc_loss" not in components
        assert math.isclose(
            float(loss), components["diff_loss"] + weight * components.get("ctc_loss", 0.0), rel_tol=1e-6
        )
        loss.backward()
        for projector in model.transformer.projectors_ctc:
            ctc_gradient = projector.model[-1].weight.grad
            if weight:
                assert ctc_gradient is not None and torch.isfinite(ctc_gradient).all()
                assert torch.count_nonzero(ctc_gradient), "Both D1 CTC heads must receive a gradient"
            else:
                assert ctc_gradient is None, "CTC heads must be inactive during warmup"
        speaker_grad = model.transformer.speaker_proj.weight.grad
        assert speaker_grad is not None and torch.isfinite(speaker_grad).all()
        speaker_grad_norm = float(speaker_grad.float().norm())
        assert speaker_grad_norm > 0
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        assert math.isfinite(global_norm) and global_norm > 0
        assert not torch.count_nonzero(model.transformer.speaker_proj.weight), "test must not update weights"
        result = {
            "ctc_lambda": weight,
            "total_loss": float(loss),
            **components,
            "speaker_grad_norm_pre_clip": speaker_grad_norm,
            "global_grad_norm_pre_clip": global_norm,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del loss, prediction
    del model, forward_kwargs, batch
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "result": "PASS",
        "config_name": args.config_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parent_path": str(ckpts.pretrained_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "updates_performed": 0,
        "checks": results,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
