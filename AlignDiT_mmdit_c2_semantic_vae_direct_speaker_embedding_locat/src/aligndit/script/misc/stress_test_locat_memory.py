"""Bounded longest-record bf16 forward/backward memory test; NO optimizer step.

Use CUDA_VISIBLE_DEVICES to select one otherwise idle GPU. This loads the
three longest CTC-feasible real records that fit the unchanged 3600-frame
budget. A live byte tensor approximates additional steady-state rank-0
Adam/EMA/DDP storage; this is deliberately not claimed as exact DDP memory.
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

import torch
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import build_model, load_composed_config


def run(args, report):
    config_path = args.config.resolve(strict=True)
    config = load_composed_config(config_path)
    dataset_config = OmegaConf.to_container(config.datasets, resolve=True)
    dataset_parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset = SemanticVaeCelebVDubDataset(**{
        key: value for key, value in dataset_config.items() if key in dataset_parameters
    })
    records = dataset.records
    eligible = sorted(
        (i for i, row in enumerate(records) if row["ctc_feasible_40hz"]),
        key=lambda i: records[i]["latent_frames"], reverse=True,
    )
    indices = eligible[:3]
    if len(indices) != 3 or sum(dataset.records[i]["latent_frames"] for i in indices) > args.frame_budget:
        raise RuntimeError("The three longest records do not fit the requested frame budget; do not silently shrink it")
    batch = dataset.collate_fn([dataset[i] for i in indices])
    report.update({
        "record_count": len(dataset.records),
        "indices": indices,
        "records": batch["utterance_keys"],
        "audio_frames": batch["mel_lengths"].tolist(),
        "video_frames": batch["video_lengths"].tolist(),
        "text_lengths": batch["text_lengths"].tolist(),
        "actual_frame_sum": int(batch["mel_lengths"].sum()),
        "padded_batch_shape": list(batch["mel"].shape),
    })
    print(json.dumps({"phase": "real_data_ready", **report}, ensure_ascii=False), flush=True)
    # Resolve/check both EMA and online checkpoint contracts on CPU first.
    checkpoint_path = args.checkpoint.resolve(strict=True)
    model = build_model(config_path, checkpoint_path, args.step, torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint, dataset
    gc.collect()
    model.requires_grad_(True).train()
    model.ctc_lambda = 0.03
    report["locat_config"] = model.transformer.locat_config
    report["strict_ema_load"] = True
    report["strict_online_load"] = True
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    reserve_bytes = args.reserve_parameter_copies * parameter_bytes
    report.update({
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": parameter_bytes,
        "additional_reserve_bytes": reserve_bytes,
        "additional_reserve_MiB": reserve_bytes / 2**20,
        "reserve_interpretation": "Approximate Adam first+second moments, EMA model, and DDP gradient bucket; not exact DDP peak",
    })
    device = torch.device("cuda:0")
    model.to(device)
    kwargs = {
        "inp": batch["mel"].permute(0, 2, 1).to(device),
        "text": batch["text"],
        "lens": batch["mel_lengths"].to(device),
        "text_lens": batch["text_lengths"].to(device),
        "video": batch["video"].to(device),
        "video_lens": batch["video_lengths"].to(device),
        "speaker_embedding": batch["speaker_embedding"].to(device),
    }
    # Keep a real live allocation throughout both forward and backward.
    reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
    reserve.zero_()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report["before_forward_allocated_MiB"] = torch.cuda.memory_allocated() / 2**20
    report["device_total_MiB"] = torch.cuda.get_device_properties(device).total_memory / 2**20
    print(json.dumps({"phase": "before_forward", **report}, ensure_ascii=False), flush=True)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    started = time.monotonic()
    # Keep the model's prompt/complementary-mask policy, but force the visible
    # video CFG branch and full CTC weight even though smoke update is only 3.
    with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, components, _, prediction = model(**kwargs)
    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
    assert all(math.isfinite(float(value)) for value in components.values())
    assert float(components["ctc_loss"]) > 0
    report["forward_allocated_MiB"] = torch.cuda.memory_allocated() / 2**20
    print(json.dumps({"phase": "forward_done", "loss": float(loss), "allocated_MiB": report["forward_allocated_MiB"]}), flush=True)
    loss.backward()
    torch.cuda.synchronize()
    norms = []
    locat_norms = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            if ".locat_" in name:
                raise AssertionError(f"Missing LocAt gradient: {name}")
            continue
        norm = gradient.detach().float().norm()
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient: {name}")
        norms.append(norm)
        if ".locat_" in name:
            if not norm > 0:
                raise AssertionError(f"Zero LocAt gradient: {name}")
            locat_norms[name] = float(norm)
    global_norm = torch.stack(norms).norm()
    assert torch.isfinite(global_norm) and global_norm > 0
    report.update({
        "result": "PASS",
        "loss": float(loss),
        "components": {key: float(value) for key, value in components.items()},
        "global_gradient_norm": float(global_norm),
        "locat_gradient_norms": locat_norms,
        "locat_diagnostics": {key: float(value) for key, value in model.transformer.locat_diagnostics().items()},
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
        "forward_backward_seconds": time.monotonic() - started,
        "live_reserve_bytes_after_backward": reserve.numel(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-budget", type=int, default=3600)
    parser.add_argument("--reserve-parameter-copies", type=int, default=4)
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[2] / "config/finetune_celebvdub_mm_c2_svae_speaker_locat_av.yaml"
    ))
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    if args.reserve_parameter_copies < 0:
        raise ValueError("reserve_parameter_copies must be nonnegative")
    torch.set_num_threads(2)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "step": args.step,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frame_budget": args.frame_budget, "ctc_lambda": 0.03,
        "force_full_video_cfg_branch": True, "optimizer_updates_performed": 0,
        "additional_parameter_copies": args.reserve_parameter_copies,
    }
    try:
        run(args, report)
    except torch.cuda.OutOfMemoryError as error:
        report.update({"result": "OOM", "error": str(error),
                       "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                       "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20})
    except Exception as error:  # noqa: BLE001 -- Persist the diagnostic, then exit nonzero below.
        report.update({"result": "FAIL", "error_type": type(error).__name__, "error": str(error)})
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if report["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
