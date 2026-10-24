"""Validate a short real DDP checkpoint, EMA loading, and both CTC regimes.

This performs diagnostics only (no optimizer updates); it never modifies the
checkpoint or shared dataset. Run before launching the long training job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import build_model, load_composed_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[4]
    config_path = project / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band.yaml"
    config = load_composed_config(config_path)
    device = torch.device("cuda:0")
    # This is the same strict EMA construction path used by real S1 inference.
    model = build_model(config_path, args.checkpoint.resolve(strict=True), args.step, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    online = state["model_state_dict"]
    band_keys = [key for key in online if key.startswith("transformer.temporal_band.")]
    assert len(band_keys) == 4 and len(online) == 708
    assert all(torch.isfinite(online[key]).all() for key in band_keys)
    final_weight = online["transformer.temporal_band.net.2.weight"]
    assert torch.count_nonzero(final_weight), "The new predictor did not update in the real DDP smoke run"
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "step": args.step,
        "strict_ema_load": True, "model_keys": len(online),
        "band_parameters": sum(online[key].numel() for key in band_keys),
        "learned_output_weight_norm": final_weight.float().norm().item(),
    }
    model.load_state_dict(online, strict=True)
    del online, state
    model.requires_grad_(True).train()
    for name in ("audio_drop_prob", "cond_drop_prob", "text_drop_prob", "video_drop_prob"):
        setattr(model, name, 0.0)
    dataset_keys = (
        "manifest_path", "cache_root", "normalization_path", "vocab_path",
        "expected_manifest_sha256", "expected_inventory_sha256",
        "expected_normalization_sha256", "expected_vocab_sha256", "expected_record_count",
        "speaker_embedding_cache_dir", "speaker_embedding_dim", "speaker_embedding_model_id",
        "speaker_embedding_checkpoint_sha256",
    )
    dataset = SemanticVaeCelebVDubDataset(**{key: config.datasets[key] for key in dataset_keys})
    indices = [i for i, row in enumerate(dataset.records)
               if row["ctc_feasible_40hz"] and 100 <= row["latent_frames"] <= 350][:2]
    if len(indices) != 2:
        raise RuntimeError("No suitable real examples for the diagnostic batch")
    batch = dataset.collate_fn([dataset[i] for i in indices])
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    report["records"] = batch["utterance_keys"]
    report["frames"] = batch["mel_lengths"].tolist()
    report["regimes"] = []
    for ctc_lambda in (0.0, 0.03):
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = ctc_lambda
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(
                batch["mel"].permute(0, 2, 1), text=batch["text"], lens=batch["mel_lengths"],
                text_lens=batch["text_lengths"], video=batch["video"], video_lens=batch["video_lengths"],
                speaker_embedding=batch["speaker_embedding"],
            )
        assert torch.isfinite(loss) and torch.isfinite(prediction).all()
        loss.backward()
        band = model.transformer.temporal_band
        head_gradient = band.net[2].weight.grad.float()
        assert torch.isfinite(head_gradient).all() and head_gradient[0].norm() > 0 and head_gradient[1].norm() > 0
        gradients = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
        total_norm = torch.stack(gradients).norm()
        assert torch.isfinite(total_norm)
        report["regimes"].append({
            "ctc_lambda": ctc_lambda, "loss": loss.item(), "components": components,
            "delta_head_grad_norm": head_gradient[0].norm().item(),
            "sigma_head_grad_norm": head_gradient[1].norm().item(),
            "global_grad_norm": total_norm.item(),
            "peak_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
        })
        print(json.dumps(report["regimes"][-1]), flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("Real checkpoint/EMA and CTC=0/0.03 gradient validation passed.", flush=True)


if __name__ == "__main__":
    main()
