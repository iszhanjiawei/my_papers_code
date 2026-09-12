"""Validate a LocAt checkpoint on real data without optimizer updates.

Strict-load EMA and online weights; test bf16 CFM/CTC backwards and confirm
the new predictors receive finite, nonzero aggregate gradients. Runtime
reports belong in ignored logs, never in the source commit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import build_model, load_composed_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[2]
        / "config/finetune_celebvdub_mm_c2_svae_speaker_locat_av.yaml"
    ))
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    config = load_composed_config(args.config.resolve(strict=True))
    model = build_model(args.config.resolve(strict=True), args.checkpoint.resolve(strict=True), args.step, device)
    if not model.transformer.locat_config["locat_enabled"]:
        raise RuntimeError("This validation requires an enabled LocAt model")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    online = checkpoint["model_state_dict"]
    for name, value in online.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite checkpoint tensor: {name}")
    model.load_state_dict(online, strict=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": args.step,
        "strict_ema_load": True,
        "strict_online_load": True,
        "locat_config": model.transformer.locat_config,
        "model_keys": len(online),
        "new_parameters": sum(p.numel() for name, p in model.named_parameters() if ".locat_" in name),
        "optimizer_updates_performed": 0,
        "regimes": [],
    }
    del online, checkpoint
    model.requires_grad_(True).train()
    dataset_config = OmegaConf.to_container(config.datasets, resolve=True)
    dataset_parameters = inspect.signature(SemanticVaeCelebVDubDataset).parameters
    dataset = SemanticVaeCelebVDubDataset(**{
        key: value for key, value in dataset_config.items() if key in dataset_parameters
    })
    indices = [i for i, row in enumerate(dataset.records)
               if row["ctc_feasible_40hz"] and 100 <= row["latent_frames"] <= 350][:2]
    if len(indices) != 2:
        raise RuntimeError("No suitable real examples for the validation batch")
    batch = dataset.collate_fn([dataset[i] for i in indices])
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    report["records"] = batch["utterance_keys"]
    report["frames"] = batch["mel_lengths"].tolist()
    model_kwargs = {
        "inp": batch["mel"].permute(0, 2, 1),
        "text": batch["text"], "lens": batch["mel_lengths"], "text_lens": batch["text_lengths"],
        "video": batch["video"], "video_lens": batch["video_lengths"],
        "speaker_embedding": batch["speaker_embedding"],
    }
    for ctc_lambda in (0.0, 0.03):
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = ctc_lambda
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666)
        torch.cuda.reset_peak_memory_stats()
        with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(**model_kwargs)
        assert torch.isfinite(loss) and torch.isfinite(prediction).all()
        assert all(math.isfinite(float(value)) for value in components.values())
        if ctc_lambda:
            assert float(components["ctc_loss"]) > 0
        loss.backward()
        gradient_groups = {}
        for direction in ("av", "va"):
            for predictor in ("log_sigma", "log_alpha"):
                marker = f".locat_{direction}.{predictor}."
                parameters = [p for name, p in model.named_parameters() if marker in name]
                if not parameters:
                    continue
                gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
                if len(gradients) != len(parameters):
                    raise AssertionError(f"Missing gradients in {marker}")
                norm = torch.stack(gradients).norm()
                assert torch.isfinite(norm)
                # Uniform control intentionally does not depend on sigma.
                if predictor == "log_alpha" or model.transformer.locat_config["locat_bias_mode"] == "gaussian":
                    assert norm > 0, f"No signal reaches {marker}"
                gradient_groups[f"{direction}/{predictor}"] = norm.item()
        global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(global_norm) and global_norm > 0
        diagnostics = {key: float(value) for key, value in model.transformer.locat_diagnostics().items()}
        assert all(math.isfinite(value) for value in diagnostics.values())
        regime = {
            "ctc_lambda": ctc_lambda, "loss": loss.item(), "components": components,
            "locat_gradient_norms": gradient_groups, "locat_diagnostics": diagnostics,
            "global_grad_norm": global_norm.item(),
            "peak_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
        }
        report["regimes"].append(regime)
        print(json.dumps(regime, sort_keys=True), flush=True)
    report["result"] = "PASS"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"result": "PASS", "report": str(args.output_json.resolve())}), flush=True)


if __name__ == "__main__":
    main()
