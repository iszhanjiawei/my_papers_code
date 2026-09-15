"""Validate the actual S2c EMA parent and real Synchformer/speaker/latent batches on one GPU.

This is a forward/backward integration test only: it does not update weights,
write checkpoints, or start a training job. Select the GPU with
CUDA_VISIBLE_DEVICES and retain stdout as the validation report.

For a conservative capacity probe, add --batch-frames 3600 and
--reserve-training-memory. The extra byte buffers estimate resident AdamW,
EMA and DDP bucket storage; they do not execute those components and cannot
establish that a complete distributed optimizer update will fit.
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
from aligndit.model.synchformer_features import cache_path, load_synchformer_feature
from f5_tts.model.utils import get_tokenizer


def select_examples(dataset, sync_cache_dir, *, max_frames, batch_frames):
    if not batch_frames:
        selected = [
            index for index, record in enumerate(dataset.records)
            if record["ctc_feasible_40hz"] and 64 <= record["latent_frames"] <= max_frames
            and cache_path(sync_cache_dir, dataset._synchformer_clip_key(record)).is_file()
        ][:2]
        if len(selected) != 2:
            raise RuntimeError("Need two real CTC-feasible examples between 64 and --max-frames frames")
        return selected

    candidates = sorted(
        ((int(record["latent_frames"]), index) for index, record in enumerate(dataset.records)
         if record["ctc_feasible_40hz"] and 64 <= record["latent_frames"] <= batch_frames),
        reverse=True,
    )
    selected, valid_frames, longest = [], 0, 0
    for length, index in candidates:
        if valid_frames + length > batch_frames or max(longest, length) * (len(selected) + 1) > batch_frames:
            continue
        record = dataset.records[index]
        if not cache_path(sync_cache_dir, dataset._synchformer_clip_key(record)).is_file():
            continue
        selected.append(index)
        valid_frames += length
        longest = max(longest, length)
        if valid_frames >= batch_frames * 0.99:
            break
    if not selected or longest <= 1000 or valid_frames < batch_frames * 0.9:
        raise RuntimeError(
            f"Need available CTC-feasible caches totaling >=90% of {batch_frames} frames, "
            f"including a >1000-frame clip; selected {valid_frames} frames, longest={longest}"
        )
    return selected


def reserve_training_storage(model, device):
    """Keep explicit CUDA byte buffers alive, without optimizer/EMA execution."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fp32_parameter_bytes = sum(parameter.numel() for parameter in parameters) * 4
    # EMA copies floating parameters/buffers as FP32; integral buffers retain
    # their original width. Count registered tensors, not a second live model.
    ema_bytes = sum(
        value.numel() * (4 if value.is_floating_point() else value.element_size())
        for value in list(model.parameters()) + list(model.buffers())
    )
    byte_counts = {
        "adamw_exp_avg_fp32": fp32_parameter_bytes,
        "adamw_exp_avg_sq_fp32": fp32_parameter_bytes,
        "ema_parameters_and_buffers": ema_bytes,
        "ddp_gradient_bucket_estimate_fp32": fp32_parameter_bytes,
    }
    report = {
        "kind": "conservative_resident_storage_estimate_not_full_DDP_validation",
        "bytes": byte_counts,
        "total_bytes": sum(byte_counts.values()),
        "total_gib": sum(byte_counts.values()) / 1024**3,
        "exclusions": ["NCCL workspace", "optimizer.step temporary tensors", "DDP bucket rebuild transients"],
    }
    print(json.dumps({"training_memory_reservation_plan": report}, sort_keys=True), flush=True)
    buffers = [torch.empty(count, dtype=torch.uint8, device=device) for count in byte_counts.values()]
    torch.cuda.synchronize(device)
    return buffers, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_synchformer"
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--partial-cache", action="store_true", help="Validate selected available real caches before full extraction completes; production training still requires full coverage")
    parser.add_argument("--batch-frames", type=int, default=0, help="Use a nearly full long-clip batch, limiting both valid and padded frames; 0 retains the two short examples")
    parser.add_argument("--reserve-training-memory", action="store_true", help="Reserve AdamW/EMA/DDP-sized CUDA byte buffers; capacity estimate only, no optimizer update")
    parser.add_argument("--checkpoint-activations", action="store_true", help="Override activation checkpointing to true for an OOM fallback probe")
    args = parser.parse_args()
    if args.batch_frames < 0:
        parser.error("--batch-frames must be nonnegative")
    started = time.monotonic()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The real-parent integration test requires a CUDA device")
    device = torch.device("cuda:0")
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=args.config_name)
    if args.checkpoint_activations:
        config.model.arch.checkpoint_activations = True
    print(json.dumps({
        "checkpoint_activations": bool(config.model.arch.checkpoint_activations),
        "requested_batch_frames": args.batch_frames,
        "reserve_training_memory": args.reserve_training_memory,
    }), flush=True)
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
    sync_cache_dir = dataset_config["synchformer_cache_dir"]
    if args.partial_cache:
        # This read-only integration test loads/validates each selected cache
        # directly. The production dataset's full-coverage gate is unchanged.
        dataset_config = {key: value for key, value in dataset_config.items() if not key.startswith("synchformer_")}
    dataset_parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset = SemanticVaeCelebVDubDataset(
        **{key: value for key, value in dataset_config.items() if key in dataset_parameters}
    )
    selected = select_examples(dataset, sync_cache_dir, max_frames=args.max_frames, batch_frames=args.batch_frames)
    examples = [dataset[index] for index in selected]
    if args.partial_cache:
        for example, index in zip(examples, selected):
            example["sync_feat"] = load_synchformer_feature(
                sync_cache_dir, dataset._synchformer_clip_key(dataset.records[index]),
                expected_checkpoint_sha256=config.datasets.synchformer_checkpoint_sha256,
            )
    batch = dataset.collate_fn(examples)
    batch_report = {
        "selected_utterances": batch["utterance_keys"],
        "latent_lengths": batch["mel_lengths"].tolist(),
        "text_lengths": batch["text_lengths"].tolist(),
        "speaker_norms": batch["speaker_embedding"].norm(dim=1).tolist(),
        "sync_lengths": batch["sync_lens"].tolist(),
        "partial_cache_test": args.partial_cache,
        "total_valid_frames": int(batch["mel_lengths"].sum()),
        "total_padded_frames": batch["mel"].shape[0] * batch["mel"].shape[2],
    }
    print(json.dumps(batch_report), flush=True)
    assert batch["mel"].shape[1] == 64
    assert batch["speaker_embedding"].shape == (len(selected), 192)
    assert batch["sync_feat"].shape[0] == len(selected) and batch["sync_feat"].shape[2] == 768
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
    # New sync keys must not weaken the original parent checks: malformed
    # unrelated audio weights and unknown source keys still fail before loading.
    migration_identity = dict(
        parent_path=ckpts.pretrained_path,
        parent_sha256=ckpts.expected_parent_sha256,
        parent_size=ckpts.expected_parent_size,
        parent_contract_sha256=ckpts.expected_parent_contract_sha256,
        parent_ema_step=ema_step,
    )
    common_key = next(key for key in source_state if key in model.state_dict() and source_state[key].numel() > 1)
    for corruption in ("shape", "unknown_key"):
        malformed = dict(source_state)
        if corruption == "shape":
            malformed[common_key] = malformed[common_key].flatten()[:1]
        else:
            malformed["transformer.unexpected_audio_weight"] = malformed.pop(common_key)
        try:
            migrate_s2c_ema_into_model(model, malformed, **migration_identity)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Strict migration accepted {corruption} corruption")
    del malformed
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
    assert migration.target_key_count == 710
    assert migration.loaded_key_count == 303
    assert len(migration.ignored_source_keys) == 10
    assert len(migration.new_target_keys) == 407
    assert "transformer.speaker_proj.weight" in migration.new_target_keys
    assert not torch.count_nonzero(model.transformer.speaker_proj.weight)
    assert not torch.count_nonzero(model.transformer.sync_in[2].w2.weight)
    for key, value in model.state_dict().items():
        if key in source_state:
            assert torch.equal(value, source_state[key]), f"Migration changed parent tensor {key}"
    print("[OK] strict migration: source=313, target=710, loaded=303, ignored=10, new=407; loaded tensors exact", flush=True)
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
        "sync_feat": batch["sync_feat"].to(device),
        "sync_lens": batch["sync_lens"].to(device),
    }
    assert not forward_kwargs["speaker_embedding"].requires_grad
    assert not forward_kwargs["sync_feat"].requires_grad
    reservation_buffers, reservation_report = [], None
    if args.reserve_training_memory:
        reservation_buffers, reservation_report = reserve_training_storage(model, device)
    print(json.dumps({
        "resident_allocated_gib_before_forward": torch.cuda.memory_allocated(device) / 1024**3,
        "resident_reserved_gib_before_forward": torch.cuda.memory_reserved(device) / 1024**3,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }), flush=True)
    results = []
    for weight in (0.0, 0.03):
        torch.manual_seed(int(config.seed) + 1)
        torch.cuda.manual_seed_all(int(config.seed) + 1)
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = weight
        torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({"stage": "forward_backward", "ctc_lambda": weight}), flush=True)
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
        sync_grad = model.transformer.sync_in[2].w2.weight.grad
        assert sync_grad is not None and torch.isfinite(sync_grad).all()
        sync_grad_norm = float(sync_grad.float().norm())
        assert sync_grad_norm > 0
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        assert math.isfinite(global_norm) and global_norm > 0
        assert not torch.count_nonzero(model.transformer.speaker_proj.weight), "test must not update weights"
        assert not torch.count_nonzero(model.transformer.sync_in[2].w2.weight), "test must not update weights"
        torch.cuda.synchronize(device)
        result = {
            "ctc_lambda": weight,
            "total_loss": float(loss),
            **components,
            "speaker_grad_norm_pre_clip": speaker_grad_norm,
            "sync_output_projection_grad_norm_pre_clip": sync_grad_norm,
            "global_grad_norm_pre_clip": global_norm,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del loss, prediction
    del model, forward_kwargs, batch, reservation_buffers
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "result": "PASS",
        "config_name": args.config_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parent_path": str(ckpts.pretrained_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "updates_performed": 0,
        "partial_cache_test": args.partial_cache,
        "checkpoint_activations": bool(config.model.arch.checkpoint_activations),
        "batch": batch_report,
        "training_memory_reservation": reservation_report,
        "checks": results,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
