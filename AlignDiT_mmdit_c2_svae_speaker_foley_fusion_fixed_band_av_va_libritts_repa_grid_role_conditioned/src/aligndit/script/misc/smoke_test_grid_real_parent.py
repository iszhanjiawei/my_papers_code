"""Validate real GRID features, LibriTTS migration and fixed-band Foley fusion + REPA gradients.

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
from aligndit.model.libritts_pretrained_migration import load_libritts_ema_into_model
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
    migration = load_libritts_ema_into_model(model, ckpts.pretrained_path, config)
    assert migration["loaded_key_count"] == 608 and len(migration["row_remappings"]) == 5
    assert migration["new_target_key_count"] == 223 and len(migration["ignored_source_keys"]) == 194
    assert "transformer.speaker_proj.weight" in migration["loaded_keys"]
    parent_speaker_projection = model.transformer.speaker_proj.weight.detach().clone()
    assert parent_speaker_projection.count_nonzero() > 0
    assert model.transformer.temporal_band is not None and not list(model.transformer.temporal_band.parameters())
    print("[OK] strict LibriTTS EMA migration, including semantic mapping of the five vocabulary tensors", flush=True)
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
        fusion_grads = [module.weight.grad for module in (
            model.transformer.transformer_blocks[0].cross_attn_ada,
            model.transformer.transformer_blocks[0].v_cross_attn_ada,
        )]
        assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in fusion_grads)
        repa_gradients = [p.grad for p in model.transformer.repa_projector.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in repa_gradients)
        repa_norm = float(torch.stack([g.float().norm() for g in repa_gradients]).norm())
        assert repa_norm > 0
        fusion_norm = float(torch.stack([g.float().norm() for g in fusion_grads]).norm())
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        assert math.isfinite(global_norm) and global_norm > 0
        assert torch.equal(model.transformer.speaker_proj.weight.detach().cpu(), parent_speaker_projection)
        result = {
            "ctc_lambda": weight,
            "total_loss": float(loss),
            **components,
            "role_grad_norm_pre_clip": role_norm,
            "speaker_grad_norm_pre_clip": speaker_grad_norm,
            "joint_text_gate_grad_norm_pre_clip": fusion_norm,
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
