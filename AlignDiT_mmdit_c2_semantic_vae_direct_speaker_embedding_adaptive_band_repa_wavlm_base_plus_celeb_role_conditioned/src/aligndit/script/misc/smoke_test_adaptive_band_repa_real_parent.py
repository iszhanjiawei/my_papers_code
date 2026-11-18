"""Validate S2c migration and real adaptive-band + REPA batches on one GPU.

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
from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from aligndit.script.train.finetune_semantic_vae_c2_direct_speaker import dataset_from_config
from f5_tts.model.utils import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus"
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

    dataset = dataset_from_config(config)
    roles = bool(config.model.arch.get("role_conditioned_text", False))
    selected = []
    used_prompt_lengths = set()
    base = dataset.base_dataset if roles else dataset
    for index in range(len(dataset)):
        target_index = dataset.indices[index][0] if roles else index
        target = base.records[target_index]
        if not target["ctc_feasible_40hz"] or not 64 <= target["latent_frames"] <= args.max_frames:
            continue
        if roles:
            ref_length = base.get_frame_len(dataset.indices[index][1])
            if ref_length in used_prompt_lengths or ref_length > args.max_frames:
                continue
            used_prompt_lengths.add(ref_length)
        selected.append(index)
        if len(selected) == 2:
            break
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

    if roles:
        for n, index in enumerate(selected):
            t, r = (base.records[i] for i in dataset.indices[index])
            p, duration = int(batch["prompt_lens"][n]), int(batch["mel_lengths"][n])
            assert p == r["latent_frames"] and duration - p == t["latent_frames"]
            assert batch["reference_text"][n] == r["text"] and batch["text"][n] == t["text"]
            assert torch.equal(batch["speaker_embedding"][n], torch.from_numpy(base._load_speaker_embedding_array(r)))
            assert torch.equal(batch["mel"][n, :, :p], base._normalized_latent(r))
            assert torch.equal(batch["mel"][n, :, p:duration], base._normalized_latent(t))
            teacher = torch.from_numpy(base._load_repa_feature_array(t))
            assert torch.equal(batch["repa_features"][n, :teacher.shape[0]], teacher)
            assert not batch["video"][n, :p].any()
            assert torch.equal(batch["video"][n, p:duration], base._target_item(t)["video"])
            print(json.dumps({"target": t["utterance_key"], "target_text": t["text"], "T": duration - p,
                              "reference": r["utterance_key"], "reference_text": r["text"], "P": p}), flush=True)

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
    role_keys = ({f"transformer.role_conditioner.{k}" for k in model.transformer.role_conditioner.state_dict()}
                 if roles else set())
    assert role_keys.issubset(migration.new_target_keys)
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
    print(f"[OK] strict migration, 303 shared tensors exact, {len(role_keys)} new role tensors", flush=True)
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
        "repa_features": batch["repa_features"].to(device),
        "repa_feature_lens": batch["repa_feature_lengths"].to(device),
    }
    if roles:
        forward_kwargs.update(reference_text=batch["reference_text"], prompt_lens=batch["prompt_lens"].to(device))
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
        if roles:
            style_grad = model.transformer.role_conditioner.style_out.weight.grad
            assert style_grad is not None and torch.isfinite(style_grad).all() and style_grad.norm() > 0
            components["style_projection_grad_norm"] = float(style_grad.norm())
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
            "speaker_grad_norm_pre_clip": speaker_grad_norm,
            "temporal_band_output_grad_norm_pre_clip": band_norm,
            "repa_projector_grad_norm_pre_clip": repa_norm,
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
