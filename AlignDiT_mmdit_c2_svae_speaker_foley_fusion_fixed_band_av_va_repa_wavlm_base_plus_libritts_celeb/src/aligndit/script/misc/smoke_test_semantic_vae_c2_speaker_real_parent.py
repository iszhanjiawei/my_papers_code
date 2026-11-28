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
from unittest.mock import patch

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from f5_tts.model.utils import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup"
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
    model_cls = hydra.utils.get_class(f"aligndit.model.{config.model.backbone}")
    audio = config.model.audio_representation
    model = CFM_VT(
        transformer=model_cls(**config.model.arch, text_num_embeds=vocab_size, mel_dim=audio.channels),
        mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
        num_channels=audio.channels,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=config.model.arch.audio_video_ratio,
        ctc_lambda=config.model.ctc_lambda,
    )
    source_state, ema_step = load_s2c_ema_state(
        ckpts.pretrained_path,
        expected_parent_contract_sha256=ckpts.expected_parent_contract_sha256,
        expected_parent_update=ckpts.expected_parent_update,
    )
    migration = migrate_s2c_ema_into_model(
        model,
        source_state,
        parent_path=ckpts.pretrained_path,
        parent_sha256=ckpts.expected_parent_sha256,
        parent_size=ckpts.expected_parent_size,
        parent_contract_sha256=ckpts.expected_parent_contract_sha256,
        parent_ema_step=ema_step,
    )
    assert migration.source_key_count == 313
    assert migration.target_key_count == 704
    assert migration.loaded_key_count == 303
    assert len(migration.ignored_source_keys) == 10
    assert len(migration.new_target_keys) == 401
    assert "transformer.speaker_proj.weight" in migration.new_target_keys
    assert not torch.count_nonzero(model.transformer.speaker_proj.weight)
    for key, value in model.state_dict().items():
        if key in source_state:
            assert torch.equal(value, source_state[key]), f"Migration changed parent tensor {key}"
    print("[OK] strict migration: source=313, target=704, loaded=303, ignored=10, new=401; loaded tensors exact", flush=True)
    del source_state
    gc.collect()

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
    for weight in (0.0, 0.03):
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
        loss.backward()
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
