"""Validate a paired AT-then-AV GRID BF16 batch from the real LibriTTS EMA.

This is a forward/backward capacity check, with no optimizer steps or saved
weights. Select a physical GPU through CUDA_VISIBLE_DEVICES and run from this
snapshot with PYTHONPATH=src. All asset paths come from the ROOT_PREFIX-aware
training configuration. Runtime results are written under logs/ by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.libritts_pretrained_migration import load_libritts_ema_into_model
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.repa import masked_repa_cosine_loss
from aligndit.script.train.finetune_grid import dataset_from_config
from f5_tts.model.utils import get_tokenizer


PROJECT = Path(__file__).resolve().parents[1]


def check(args, report):
    torch.set_num_threads(4)
    torch.manual_seed(666)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This capacity test requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        cfg = compose(config_name="finetune_grid")
    report.update(
        project=str(PROJECT), device=str(device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(device), total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        torch_version=torch.__version__, config=OmegaConf.to_container(cfg, resolve=True),
        optimizer_updates=0, saved_weights=False,
    )
    print("Validating real GRID data and loading the strict LibriTTS parent", flush=True)
    dataset = dataset_from_config(cfg)
    indices = sorted(range(len(dataset)), key=lambda i: (-dataset.get_frame_len(i), i))[:args.samples]
    cpu_batch = dataset.collate_fn([dataset[i] for i in indices])
    frames = int(cpu_batch["mel_lengths"].sum())
    if frames > int(cfg.datasets.batch_size_per_gpu):
        raise ValueError(f"Selected batch has {frames} frames, above the unchanged frame budget")
    report["batch"] = {
        "samples": len(indices), "frames": frames, "latent_lengths": cpu_batch["mel_lengths"].tolist(),
        "teacher_lengths": cpu_batch["repa_feature_lengths"].tolist(),
        "ids": [dataset.records[i]["utterance_key"] for i in indices],
        "selection": "longest training utterances, stable index tie-break; no held-out samples",
        "manifest_sha256": dataset.cache_contract["manifest_sha256"],
    }
    vocab, size = get_tokenizer(cfg.datasets.vocab_path, "custom")
    audio = cfg.model.audio_representation
    model = CFM_VT(
        transformer=hydra.utils.get_class(f"aligndit.model.{cfg.model.backbone}")(
            **cfg.model.arch, text_num_embeds=size, mel_dim=audio.channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=audio.channels, target_sample_rate=audio.sample_rate, hop_length=audio.hop_length),
        num_channels=audio.channels, vocab_char_map=vocab,
        audio_video_ratio=cfg.model.arch.audio_video_ratio,
        ctc_lambda=cfg.model.ctc_lambda, repa_lambda=cfg.model.repa_lambda,
    )
    if model.transformer.paired_joint_order != "at_then_av" or model.transformer.repa_layer != 9:
        raise AssertionError("Expected the REPA tap after the tenth paired AT-then-AV block")
    migration = load_libritts_ema_into_model(model, cfg.ckpts.pretrained_path, cfg)
    if (migration["loaded_key_count"], migration["new_target_key_count"]) != (802, 6):
        raise AssertionError("The initialization must retain all 802 parent tensors and six fresh projector tensors")
    report["initialization"] = {key: migration[key] for key in (
        "parent_path", "parent_sha256", "parent_update", "loaded_key_count", "new_target_key_count",
        "new_target_numel", "optimizer_restored", "downstream_initial_update")}
    model = model.to(device).train()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Rank zero keeps an EMA model, AdamW keeps two FP32 moments, and ordinary
    # DDP may retain a full gradient bucket alongside parameter gradients.
    # Reserve these persistent allocations without running an optimizer step.
    reserved_training_state = [torch.empty(parameter_bytes, dtype=torch.uint8, device=device) for _ in range(4)]
    report["capacity_reservation"] = {
        "parameter_bytes": parameter_bytes, "reserved_bytes": sum(t.numel() for t in reserved_training_state),
        "components": ["rank-zero EMA", "AdamW first moment", "AdamW second moment", "DDP gradient bucket"],
        "optimizer_step_temporaries_included": False,
    }
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
    teacher = batch["repa_features"].detach().requires_grad_(True)
    if teacher.dtype != torch.float16 or not torch.isfinite(teacher).all():
        raise AssertionError("The real cached teacher must be finite FP16")
    data = dict(
        inp=batch["mel"].permute(0, 2, 1), text=batch["text"], lens=batch["mel_lengths"],
        text_lens=batch["text_lengths"], video=batch["video"], video_lens=batch["video_lengths"],
        speaker_embedding=batch["speaker_embedding"], repa_features=teacher,
        repa_feature_lens=batch["repa_feature_lengths"],
    )
    del cpu_batch, dataset
    gc.collect()
    cases = []
    report["cases"] = cases
    for ctc_lambda in (0.0, float(cfg.model.ctc_lambda)):
        model.ctc_lambda = ctc_lambda
        model.zero_grad(set_to_none=True)
        teacher.grad = None
        torch.manual_seed(916)
        torch.cuda.manual_seed_all(916)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        checks = {}

        def audited_loss(student, target, target_lens, student_lens, mask):
            actual = masked_repa_cosine_loss(student, target, target_lens, student_lens, mask)
            # Deliberately poison new padded frames; valid lengths and generation
            # mask semantics must make these entirely irrelevant to the loss.
            with torch.no_grad():
                padded_student = F.pad(student.detach(), (0, 0, 0, 2), value=20000.0)
                padded_target = F.pad(target.detach(), (0, 0, 0, 3), value=-20000.0)
                padded_mask = F.pad(mask, (0, 2), value=True)
                altered = masked_repa_cosine_loss(padded_student, padded_target, target_lens,
                                                  student_lens, padded_mask)
                if not torch.equal(actual.detach(), altered):
                    raise AssertionError("Padded student/teacher frames changed the REPA loss")
            checks.update(teacher_padding_invariant=True, student_padding_invariant=True,
                          generated_frames=int(mask.sum()), student_dtype=str(student.dtype))
            return actual

        print(f"BF16 forward/backward: CTC lambda={ctc_lambda}, {args.samples} samples, {frames} frames", flush=True)
        # Keep all conditions for this diagnostic so the speaker-gradient check
        # is meaningful. Production dropout probabilities remain unchanged.
        with patch("aligndit.model.cfm_vt.random", return_value=0.99), \
             patch("aligndit.model.cfm_vt.masked_repa_cosine_loss", side_effect=audited_loss), \
             torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(**data)
        if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite BF16 loss or generated flow")
        expected = components["diff_loss"] + model.repa_lambda * components["repa_loss"]
        expected += ctc_lambda * components.get("ctc_loss", 0.0)
        if abs(loss.item() - expected) > 2e-6 * max(1.0, abs(expected)):
            raise AssertionError("Total loss differs from diffusion + 0.1 REPA + scheduled CTC")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        if teacher.grad is not None:
            raise AssertionError("Gradient leaked into frozen cached teacher targets")
        gradients = {}
        required = [
            "transformer.input_embed.proj.weight",
            "transformer.transformer_blocks.9.attn.to_q.weight",
            "transformer.transformer_blocks.17.attn.to_q.weight",
            "transformer.speaker_proj.weight",
            *[f"transformer.repa_projector.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")],
        ]
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite parameter gradient: {name}")
            if name in required:
                if parameter.grad is None or not torch.count_nonzero(parameter.grad):
                    raise AssertionError(f"Missing or zero required gradient: {name}")
                gradients[name] = parameter.grad.float().norm().item()
        if set(gradients) != set(required):
            raise AssertionError("A required gradient target was not present in the real model")
        case = dict(
            ctc_lambda=ctc_lambda, repa_lambda=model.repa_lambda, total_loss=loss.item(),
            components=components, recomputed_total=expected, loss_combination_error=abs(loss.item() - expected),
            elapsed_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), gradient_norms=gradients,
            teacher_gradient_is_none=teacher.grad is None, all_present_gradients_finite=True,
            conditions="all kept for gradient diagnostic via RNG patch only", **checks,
        )
        cases.append(case)
        print(json.dumps(case, sort_keys=True), flush=True)
        del loss, prediction
    report["status"] = "PASS"
    report["complete"] = True
    report["max_peak_allocated_bytes"] = max(c["peak_allocated_bytes"] for c in cases)
    report["max_peak_reserved_bytes"] = max(c["peak_reserved_bytes"] for c in cases)
    report["allocated_headroom_bytes"] = report["total_memory_bytes"] - report["max_peak_allocated_bytes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PROJECT / "logs/grid_repa_gpu_validation.json")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    report = {"status": "RUNNING", "complete": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        check(args, report)
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"GPU validation {report['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
