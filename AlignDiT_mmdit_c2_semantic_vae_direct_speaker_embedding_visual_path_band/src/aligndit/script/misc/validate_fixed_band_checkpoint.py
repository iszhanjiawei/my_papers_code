"""Read-only EMA/online, fixed-prior and real-data CTC gradient validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aligndit.model.fixed_temporal_band import FixedTemporalBand
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
    config_path = project / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_fixed_band.yaml"
    config = load_composed_config(config_path)
    device = torch.device("cuda:0")
    model = build_model(config_path, args.checkpoint.resolve(strict=True), args.step, device)
    assert isinstance(model.transformer.temporal_band, FixedTemporalBand)
    assert not list(model.transformer.temporal_band.parameters())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    online = state["model_state_dict"]
    assert len(online) == 704 and not any(k.startswith("transformer.temporal_band.") for k in online)
    assert all(torch.isfinite(value).all() for value in online.values())
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "step": args.step,
        "strict_ema_load": True, "model_keys": len(online), "band_parameters": 0,
        "fixed_offset_seconds": 0.0, "fixed_sigma_seconds": 0.100,
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
        offset = model.transformer.last_temporal_band_offset_seconds
        sigma = model.transformer.last_temporal_band_sigma_seconds
        assert torch.equal(offset, torch.zeros_like(offset))
        assert torch.equal(sigma, torch.full_like(sigma, 0.100))
        loss.backward()
        gradients = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
        total_norm = torch.stack(gradients).norm()
        assert torch.isfinite(total_norm) and total_norm > 0
        assert model.transformer.temporal_band.state_dict() == {}
        report["regimes"].append({
            "ctc_lambda": ctc_lambda, "loss": loss.item(), "components": components,
            "global_grad_norm": total_norm.item(),
            "peak_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
        })
        print(json.dumps(report["regimes"][-1]), flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("Fixed-band checkpoint/EMA and CTC=0/0.03 validation passed.", flush=True)


if __name__ == "__main__":
    main()
