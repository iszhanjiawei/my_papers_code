"""Check real GRID 74/75-frame examples against the pinned S2c EMA parent.

Run from this independent project after sourcing env.sh and setting PYTHONPATH=src.
The diagnostic runs forward/backward at CTC weights 0 and 0.03. It never creates
an optimizer, updates weights, or writes a training checkpoint.
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
from hydra import compose, initialize_config_dir

from aligndit.model.grid_semantic_vae_dataset import GridSemanticVaeDataset
from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from aligndit.script.train.finetune_grid_semantic_vae import build_dataset, build_model


def _select_boundary_batch(config):
    selected = {}
    contract_sha256 = None
    for split in ("train", "val"):
        dataset = build_dataset(config, split=split)
        contract_sha256 = dataset.contract_sha256
        for index, record in enumerate(dataset.records):
            frames = record.get("source_video_frames")
            if frames in (74, 75) and frames not in selected and record["ctc_feasible_40hz"]:
                selected[frames] = (dataset[index], record)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError(
            "The diagnostic cache must include CTC-feasible real 74- and 75-frame GRID clips; "
            f"found source frame lengths {sorted(selected)}"
        )
    items = [selected[frames][0] for frames in (74, 75)]
    records = [selected[frames][1] for frames in (74, 75)]
    batch = GridSemanticVaeDataset.collate_fn(items)
    if not torch.equal(batch["mel_lengths"], batch["video_lengths"]):
        raise AssertionError("GRID latent and interpolated video lengths differ")
    if len(set(batch["mel_lengths"].tolist())) != 2:
        raise AssertionError("74/75-frame diagnostic clips must exercise different valid latent lengths")
    if batch["mel"].shape[1] != 64 or batch["speaker_embedding"].shape != (2, 192):
        raise AssertionError("Wrong GRID latent/speaker dimensions")
    selection = {
        "utterances": batch["utterance_keys"],
        "source_video_frames": [record["source_video_frames"] for record in records],
        "latent_lengths": batch["mel_lengths"].tolist(),
        "repa_lengths": batch["repa_feature_lengths"].tolist(),
        "text_lengths": batch["text_lengths"].tolist(),
        "speaker_norms": batch["speaker_embedding"].norm(dim=1).tolist(),
        "contract_sha256": contract_sha256,
    }
    print(json.dumps({"batch": selection}, sort_keys=True), flush=True)
    return batch, selection


def _gradient_norm(parameter, label):
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError(f"Missing/non-finite {label} gradient")
    norm = float(gradient.float().norm())
    if not math.isfinite(norm) or norm <= 0:
        raise AssertionError(f"Expected a finite nonzero {label} gradient, got {norm}")
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-name", default="finetune_grid_mmdit")
    args = parser.parse_args()
    started = time.monotonic()
    project_root = Path(__file__).resolve().parents[1]
    imported_source = Path(inspect.getfile(GridSemanticVaeDataset)).resolve()
    if not imported_source.is_relative_to(project_root / "src"):
        raise RuntimeError(f"Wrong aligndit snapshot imported: {imported_source}; set PYTHONPATH=src")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This real-parent BF16 diagnostic requires an available CUDA device")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device must support BF16")
    torch.set_num_threads(4)
    with initialize_config_dir(version_base="1.3", config_dir=str(project_root / "src/aligndit/config")):
        config = compose(config_name=args.config_name)
    config.datasets.cache_root = str(args.cache_root.expanduser().resolve(strict=True))
    config.datasets.contract_path = str(Path(config.datasets.cache_root) / "data_contract.json")
    config.datasets.allow_subset = True
    torch.manual_seed(int(config.seed))
    batch, selection = _select_boundary_batch(config)

    parent = config.ckpts
    print("Validating real S2c 70k parent artifact hashes...", flush=True)
    validate_parent_artifacts(
        parent.pretrained_path,
        parent.parent_contract_path,
        expected_checkpoint_sha256=parent.expected_parent_sha256,
        expected_checkpoint_size=parent.expected_parent_size,
        expected_contract_sha256=parent.expected_parent_contract_sha256,
    )
    model = build_model(config)
    source, ema_step = load_s2c_ema_state(
        parent.pretrained_path,
        expected_parent_contract_sha256=parent.expected_parent_contract_sha256,
        expected_parent_update=parent.expected_parent_update,
    )
    migration = migrate_s2c_ema_into_model(
        model,
        source,
        parent_path=parent.pretrained_path,
        parent_sha256=parent.expected_parent_sha256,
        parent_size=parent.expected_parent_size,
        parent_contract_sha256=parent.expected_parent_contract_sha256,
        parent_ema_step=ema_step,
    )
    expected_counts = (313, 714, 303, 10, 411)
    actual_counts = (
        migration.source_key_count,
        migration.target_key_count,
        migration.loaded_key_count,
        len(migration.ignored_source_keys),
        len(migration.new_target_keys),
    )
    if actual_counts != expected_counts:
        raise AssertionError(f"S2c migration counts changed: {actual_counts} != {expected_counts}")
    for key, value in model.state_dict().items():
        if key in source and not torch.equal(value, source[key]):
            raise AssertionError(f"Migration changed parent tensor {key}")
    if torch.count_nonzero(model.transformer.speaker_proj.weight):
        raise AssertionError("Speaker projection must start at zero")
    if model.transformer.temporal_band is None or model.transformer.repa_projector is None:
        raise AssertionError("Both adaptive temporal band and REPA must be enabled")
    print(
        json.dumps(
            {
                "migration": {
                    "source_tensors": actual_counts[0],
                    "target_tensors": actual_counts[1],
                    "loaded_tensors": actual_counts[2],
                    "ignored_source_tensors": actual_counts[3],
                    "new_target_tensors": actual_counts[4],
                    "loaded_fraction": migration.loaded_fraction,
                    "parent_sha256": parent.expected_parent_sha256,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del source
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
    if forward_kwargs["speaker_embedding"].requires_grad or forward_kwargs["repa_features"].requires_grad:
        raise AssertionError("Frozen teacher features must not require gradients")
    results = []
    for ctc_weight in (0.0, float(config.model.ctc_lambda)):
        torch.manual_seed(int(config.seed) + 1)
        torch.cuda.manual_seed_all(int(config.seed) + 1)
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = ctc_weight
        torch.cuda.reset_peak_memory_stats(device)
        # Take the full-conditioning branch for the gradient check without
        # modifying the actual training dropout probabilities.
        with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(**forward_kwargs)
        if not torch.isfinite(loss) or not all(math.isfinite(float(value)) for value in components.values()):
            raise AssertionError(f"Non-finite GRID loss: {components}")
        if "repa_loss" not in components or not 0 <= components["repa_loss"] <= 2:
            raise AssertionError(f"Invalid REPA cosine loss: {components}")
        expected_loss = components["diff_loss"] + float(config.model.repa_lambda) * components["repa_loss"]
        expected_loss += ctc_weight * components.get("ctc_loss", 0.0)
        if not math.isclose(float(loss), expected_loss, rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError("Total loss differs from diff + REPA + CTC")
        if prediction.shape != forward_kwargs["inp"].shape or not torch.isfinite(prediction).all():
            raise AssertionError("GRID prediction has an invalid shape or non-finite values")
        if ctc_weight and components.get("ctc_loss", 0) <= 0:
            raise AssertionError("Both diagnostic GRID examples must contribute positive CTC loss")
        if not ctc_weight and "ctc_loss" in components:
            raise AssertionError("CTC must be disabled during its zero-weight warmup")
        loss.backward()
        speaker_norm = _gradient_norm(model.transformer.speaker_proj.weight, "speaker")
        band_weight = model.transformer.temporal_band.net[2].weight
        band_norm = _gradient_norm(band_weight, "temporal band output")
        for index, label in enumerate(("offset", "width")):
            if float(band_weight.grad[index].float().norm()) <= 0:
                raise AssertionError(f"Temporal band {label} receives no gradient")
        repa_norms = [
            _gradient_norm(parameter, f"REPA {name}")
            for name, parameter in model.transformer.repa_projector.named_parameters()
        ]
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        if not math.isfinite(global_norm) or global_norm <= 0:
            raise AssertionError(f"Invalid global pre-clip gradient norm: {global_norm}")
        if torch.count_nonzero(model.transformer.speaker_proj.weight):
            raise AssertionError("The diagnostic must never update model weights")
        result = {
            "ctc_lambda": ctc_weight,
            "total_loss": float(loss),
            **{name: float(value) for name, value in components.items()},
            "speaker_grad_norm_pre_clip": speaker_norm,
            "temporal_band_output_grad_norm_pre_clip": band_norm,
            "repa_projector_grad_norm_pre_clip": math.sqrt(sum(value**2 for value in repa_norms)),
            "global_grad_norm_pre_clip": global_norm,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del loss, prediction
    print(
        json.dumps(
            {
                "result": "PASS",
                "config_name": args.config_name,
                "project_root": str(project_root),
                "cache_root": config.datasets.cache_root,
                "batch": selection,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "parent_path": str(parent.pretrained_path),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "updates_performed": 0,
                "checks": results,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
