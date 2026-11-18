"""Validate real GRID features, S2c migration and adaptive-band + REPA gradients.

This is a forward/backward integration test only: it does not update weights,
write checkpoints, or start a training job. Select the GPU with
CUDA_VISIBLE_DEVICES and retain stdout as the validation report.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import hydra
import torch
from hydra import compose, initialize_config_dir

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.script.train.finetune_grid import dataset_from_config
from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from f5_tts.model.utils import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="finetune_grid"
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Capacity diagnostic: repeat the two real clips to this batch size; no updates")
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    started = time.monotonic()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The real-parent integration test requires a CUDA device")
    device = torch.device("cuda:0")
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=args.config_name, overrides=[f"datasets.cache_root={args.cache_root.resolve()}"])
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

    dataset = dataset_from_config(config, allow_partial=True)
    eligible = [i for i, row in enumerate(dataset.records)
                if row["ctc_feasible_40hz"] and 64 <= row["latent_frames"] <= args.max_frames]
    # Prefer different reference lengths to exercise per-example P/T boundaries.
    selected = eligible[:1]
    if selected:
        first_prompt = dataset.records[dataset.reference_indices[selected[0]]]["latent_frames"]
        second = next((i for i in eligible[1:] if
                       dataset.records[dataset.reference_indices[i]]["latent_frames"] != first_prompt), None)
        selected += [second] if second is not None else eligible[1:2]
    if len(selected) != 2:
        raise RuntimeError("Need two real CTC-feasible examples between 64 and --max-frames frames")
    selected = [selected[index % len(selected)] for index in range(args.batch_size)]
    batch = dataset.collate_fn([dataset[index] for index in selected])
    print(json.dumps({
        "selected_utterances": batch["utterance_keys"],
        "reference_ids": batch["reference_utterance_keys"],
        "prompt_lens": batch["prompt_lens"].tolist(),
        "latent_lengths": batch["mel_lengths"].tolist(),
        "text_lengths": batch["text_lengths"].tolist(),
        "speaker_norms": batch["speaker_embedding"].norm(dim=1).tolist(),
        "batch_size": args.batch_size,
        "repeated_clips_for_capacity_only": args.batch_size > 2,
    }), flush=True)
    assert batch["mel"].shape[1] == 64
    assert batch["speaker_embedding"].shape == (args.batch_size, 192)
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
        repa_lambda=config.model.repa_lambda,
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
    role_keys = set("transformer.role_conditioner." + k for k in model.transformer.role_conditioner.state_dict())
    assert role_keys and role_keys.issubset(migration.new_target_keys)
    assert migration.target_key_count == 714 + len(role_keys)
    assert migration.loaded_key_count == 303
    assert len(migration.ignored_source_keys) == 10
    assert len(migration.new_target_keys) == 411 + len(role_keys)
    assert "transformer.speaker_proj.weight" in migration.new_target_keys
    assert not torch.count_nonzero(model.transformer.speaker_proj.weight)
    assert model.transformer.temporal_band is not None and model.transformer.repa_projector is not None
    assert len([k for k in migration.new_target_keys if k.startswith("transformer.temporal_band.")]) == 4
    assert len([k for k in migration.new_target_keys if k.startswith("transformer.repa_projector.")]) == 6
    for key, value in model.state_dict().items():
        if key in source_state:
            assert torch.equal(value, source_state[key]), f"Migration changed parent tensor {key}"
    print(f"[OK] strict parent migration, {migration.loaded_key_count} shared tensors exact, {len(role_keys)} new role tensors", flush=True)
    del source_state
    gc.collect()

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model.to(device).train()
    forward_kwargs = {
        "inp": batch["mel"].permute(0, 2, 1).to(device),
        "text": batch["text"],
        "reference_text": batch["reference_text"],
        "prompt_lens": batch["prompt_lens"].to(device),
        "lens": batch["mel_lengths"].to(device),
        "text_lens": batch["text_lengths"].to(device),
        "video": batch["video"].to(device),
        "video_lens": batch["video_lengths"].to(device),
        "speaker_embedding": batch["speaker_embedding"].to(device),
        "repa_features": batch["repa_features"].to(device),
        "repa_feature_lens": batch["repa_feature_lengths"].to(device),
    }
    assert not forward_kwargs["repa_features"].requires_grad
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
        assert "repa_loss" in components and 0 <= components["repa_loss"] <= 2
        expected_loss = components["diff_loss"] + config.model.repa_lambda * components["repa_loss"]
        expected_loss += weight * components.get("ctc_loss", 0.0)
        assert math.isclose(float(loss), expected_loss, rel_tol=1e-5, abs_tol=1e-6)
        assert all(math.isfinite(float(value)) for value in components.values())
        assert prediction.shape == forward_kwargs["inp"].shape
        assert torch.isfinite(prediction).all()
        if weight:
            assert components["ctc_loss"] > 0
        else:
            assert "ctc_loss" not in components
        loss.backward()
        role_grads = [p.grad for p in model.transformer.role_conditioner.parameters() if p.grad is not None]
        assert role_grads and all(torch.isfinite(g).all() for g in role_grads)
        role_norm = float(torch.stack([g.float().norm() for g in role_grads]).norm())
        assert role_norm > 0
        speaker_grad = model.transformer.speaker_proj.weight.grad
        assert speaker_grad is not None and torch.isfinite(speaker_grad).all()
        speaker_grad_norm = float(speaker_grad.float().norm())
        assert speaker_grad_norm > 0
        band_gradient = model.transformer.temporal_band.net[2].weight.grad
        assert band_gradient is not None and torch.isfinite(band_gradient).all()
        assert band_gradient[0].float().norm() > 0 and band_gradient[1].float().norm() > 0
        repa_gradients = [p.grad for p in model.transformer.repa_projector.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in repa_gradients)
        repa_norm = float(torch.stack([g.float().norm() for g in repa_gradients]).norm())
        assert repa_norm > 0
        band_norm = float(band_gradient.float().norm())
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        assert math.isfinite(global_norm) and global_norm > 0
        assert not torch.count_nonzero(model.transformer.speaker_proj.weight), "test must not update weights"
        result = {
            "ctc_lambda": weight,
            "total_loss": float(loss),
            **components,
            "role_grad_norm_pre_clip": role_norm,
            "speaker_grad_norm_pre_clip": speaker_grad_norm,
            "temporal_band_output_grad_norm_pre_clip": band_norm,
            "repa_projector_grad_norm_pre_clip": repa_norm,
            "global_grad_norm_pre_clip": global_norm,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del loss, prediction
    # Smoke-test actual ODE/CFG inference with both transcript roles and a real pair.
    model.eval()
    prompt_len = int(batch["prompt_lens"][0])
    duration = int(batch["mel_lengths"][0])
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=forward_kwargs["inp"][:1, :prompt_len], lens=batch["prompt_lens"][:1].to(device),
            text=batch["text"][:1], reference_text=batch["reference_text"][:1],
            duration=duration, video=forward_kwargs["video"][:1, :duration],
            speaker_embedding=forward_kwargs["speaker_embedding"][:1],
            seed=0, steps=4, use_epss=False, cfg_strength=5.0, cfg_strength_v=2.0,
        )
        target = generated[:, prompt_len:duration]
        assert target.shape == (1, duration - prompt_len, audio.channels) and torch.isfinite(target).all()
    print(f"[OK] structured two-text CFG sample, P={prompt_len}, T={target.size(1)}", flush=True)
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
        "batch_size": args.batch_size,
        "parameter_gib": parameter_bytes / 1024**3,
        "estimated_additional_ema_adam_ddp_gib": 4 * parameter_bytes / 1024**3,
        "capacity_note": "Additional estimate covers one FP32 EMA copy, two Adam moments and one DDP bucket copy; not an actual optimizer/DDP measurement.",
        "checks": results,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
