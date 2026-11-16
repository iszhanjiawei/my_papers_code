"""Audit a pilot cache, train one update, resume for one update, and check CTC.

Runs the real LRS3 trainer and pinned LibriTTS parent, with TensorBoard logging.
The tiny manifest is a validated subset, certified as ``pilot`` only. No full
training starts here. CUDA_VISIBLE_DEVICES selects the assigned GPU.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from aligndit.model.lrs3_semantic_vae_dataset import LATENT_SPEC, SemanticVaeLRS3Dataset
from aligndit.model.semantic_vae_dataset import SEMANTIC_VAE_FEATURE, VIDEO_40HZ_FEATURE, sha256_file


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "finetune_lrs3_svae_speaker_window13_libritts"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        return compose(config_name=CONFIG_NAME, overrides=overrides)


def load_dataset(config):
    parameters = inspect.signature(SemanticVaeLRS3Dataset).parameters
    values = OmegaConf.to_container(config.datasets, resolve=True)
    return SemanticVaeLRS3Dataset(**{key: value for key, value in values.items() if key in parameters})


def freeze_smoke_subset(data, destination, max_frames):
    selected = [index for index, row in enumerate(data.records)
                if row["ctc_feasible_40hz"] and 40 <= row["latent_frames"] <= max_frames][:4]
    if len(selected) < 2:
        raise RuntimeError(f"Need at least two CTC-feasible pilot samples of 40..{max_frames} frames")
    rows = [data.records[index] for index in selected]
    for index in selected:
        data[index]  # Validate every cache array before certifying the subset.
    destination.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in ("train", "valid", "test", "inventory"):
        chosen = rows if name in {"train", "inventory"} else []
        path = destination / f"{name}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in chosen)
        if path.exists() and path.read_text() != payload:
            raise RuntimeError(f"Refusing to replace another smoke inventory: {path}")
        path.write_text(payload)
        entry = {"count": len(chosen), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if name != "inventory":
            entry.update(ctc_feasible_count=len(chosen), ctc_infeasible_count=0)
        entries[path.name] = entry
    write_json(destination / "inventory_meta.json", {
        "schema_version": 1, "dataset": "LRS3", "status": "complete", "selection": {"mode": "pilot"},
        "latent_spec": LATENT_SPEC, "manifests": entries,
        "source_inventory_sha256": data.data_contract["inventory_sha256"],
    })
    source_receipt = json.loads((data.manifest_path.parent / "cache_completion.json").read_text())
    source_speaker = Path(source_receipt.get("speaker_metadata_dir", data.manifest_path.parent / "speaker_contract"))
    metadata = json.loads((source_speaker / "metadata.json").read_text())
    metadata["expected_count"] = len(rows)
    speaker_dir = destination / "speaker_contract"
    write_json(speaker_dir / "metadata.json", metadata)
    write_json(speaker_dir / "coverage_report.json", {
        "complete": True, "expected": len(rows), "valid": len(rows), "missing": [], "invalid": [], "extra": [],
        "split_counts": {"train": len(rows), "valid": 0, "test": 0},
        "source_coverage_sha256": source_receipt["speaker_coverage_sha256"],
    })
    frames = sum(row["latent_frames"] for row in rows)
    write_json(destination / "cache_completion.json", {
        "schema_version": 1, "status": "complete", "selection": {"mode": "pilot"},
        "inventory_sha256": entries["inventory.jsonl"]["sha256"], "count": len(rows),
        "features": {
            "latents": {"feature": SEMANTIC_VAE_FEATURE, "count": len(rows), "total_frames": frames},
            "video_40hz": {"feature": VIDEO_40HZ_FEATURE, "count": len(rows), "total_frames": frames},
            "speaker_embeddings": {"count": len(rows), "dimension": 192},
        },
        "speaker_metadata_dir": str(speaker_dir.resolve()),
        "speaker_metadata_sha256": sha256_file(speaker_dir / "metadata.json"),
        "speaker_coverage_sha256": sha256_file(speaker_dir / "coverage_report.json"),
    })
    return rows


def checkpoint_summary(path, update):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    required = {"model_state_dict", "optimizer_state_dict", "ema_model_state_dict", "scheduler_state_dict", "update"}
    if not required.issubset(checkpoint) or checkpoint["update"] != update:
        raise RuntimeError(f"Incomplete or unexpected smoke checkpoint at update {update}")
    states = checkpoint["optimizer_state_dict"]["state"]
    if not states:
        raise RuntimeError("Smoke checkpoint has no AdamW state")
    steps = [int(state["step"]) for state in states.values()]
    if max(steps) != update or min(steps) <= 0:
        raise RuntimeError(f"AdamW did not resume correctly: optimizer steps {min(steps)}..{max(steps)}")
    for key, tensor in checkpoint["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite saved model tensor: {key}")
    for state in states.values():
        if any(not torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)):
            raise FloatingPointError("Non-finite saved AdamW state")
    ema_step = int(checkpoint["ema_model_state_dict"]["step"])
    if ema_step != 50000 + update:
        raise RuntimeError(f"Parent EMA clock was not preserved: {ema_step}")
    summary = {
        "update": update, "optimizer_state_count": len(states), "optimizer_min_step": min(steps),
        "optimizer_max_step": max(steps), "ema_step": ema_step,
        "scheduler_last_epoch": checkpoint["scheduler_state_dict"]["last_epoch"],
        "size_bytes": path.stat().st_size,
    }
    del checkpoint
    gc.collect()
    return summary


def run_training(overrides, update, output):
    command = [sys.executable, "-m", "aligndit.script.train.finetune_lrs3_svae_speaker",
               "--config-name", CONFIG_NAME, *overrides, f"optim.run_until_update={update}"]
    environment = dict(os.environ)
    environment.update(ACCELERATE_MIXED_PRECISION="bf16", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    environment["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    with (output / f"train_update_{update}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"LRS3 smoke training phase {update} exited with {code}")


def ctc_backward_check(config, dataset, checkpoint_path, logdir):
    from aligndit.model.cfm_vt import CFM_VT
    from aligndit.model.modules import PrecomputedAudioRepresentation
    from f5_tts.model.utils import get_tokenizer
    from torch.utils.tensorboard import SummaryWriter

    vocab, count = get_tokenizer(config.datasets.vocab_path, "custom")
    cls = hydra.utils.get_class(f"aligndit.model.{config.model.backbone}")
    audio = config.model.audio_representation
    model = CFM_VT(
        transformer=cls(**config.model.arch, text_num_embeds=count, mel_dim=audio.channels),
        mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
        num_channels=audio.channels, vocab_char_map=vocab, audio_video_ratio=1, ctc_lambda=0.03,
    )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    model.cuda().train()
    batch = dataset.collate_fn([dataset[0]])
    torch.manual_seed(int(config.seed))
    with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, components, _, prediction = model(
            inp=batch["mel"].permute(0, 2, 1).cuda(), text=batch["text"], lens=batch["mel_lengths"].cuda(),
            text_lens=batch["text_lengths"].cuda(), video=batch["video"].cuda(),
            video_lens=batch["video_lengths"].cuda(), speaker_embedding=batch["speaker_embedding"].cuda(),
        )
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all() or float(components.get("ctc_loss", 0)) <= 0:
        raise FloatingPointError("CTC-enabled smoke loss/prediction is invalid")
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    speaker_grad = model.transformer.speaker_proj.weight.grad
    if not math.isfinite(norm) or norm <= 0 or speaker_grad is None or not torch.isfinite(speaker_grad).all():
        raise FloatingPointError("CTC-enabled smoke backward produced invalid gradients")
    result = {"ctc_lambda": 0.03, "loss": float(loss), "global_grad_norm": norm,
              "speaker_grad_norm": float(speaker_grad.float().norm()),
              **{key: float(value) for key, value in components.items()}}
    with SummaryWriter(str(logdir)) as writer:
        for key, value in result.items():
            writer.add_scalar(f"ctc_smoke/{key}", value, 2)
    del model, batch, loss, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1"))
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--audit-only", action="store_true", help="Validate the complete pilot cache on CPU; no training")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = (args.manifest_path or args.cache_root / "manifests/pilot/train.jsonl").resolve()
    base = [f"datasets.cache_root={args.cache_root.resolve()}", f"datasets.manifest_path={source_manifest}",
            "datasets.expected_selection=pilot"]
    data = load_dataset(load_config(base))
    audit = data.audit_cache()
    write_json(output / "cache_audit.json", audit)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Smoke validation requires one assigned CUDA process")
    checkpoint_dir = output / "avt_joint_fixed_band_window13_av_va_checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        raise RuntimeError("Use a fresh smoke output directory; existing checkpoints must not be overwritten")
    rows = freeze_smoke_subset(data, output / "manifests", args.max_frames)
    name = "LRS3_pilot_smoke_avt_joint_fixed_band_window13_av_va_" + output.name
    overrides = [f"datasets.cache_root={args.cache_root.resolve()}",
                 f"datasets.manifest_path={output / 'manifests/train.jsonl'}", "datasets.expected_selection=pilot",
                 "datasets.batch_size_type=sample", "datasets.batch_size_per_gpu=1", "datasets.max_samples=1",
                 "datasets.num_workers=0", "optim.epochs=4", "optim.num_warmup_updates=1",
                 "ckpts.save_per_updates=100", "ckpts.last_per_updates=100", "ckpts.keep_last_n_checkpoints=0",
                 f"ckpts.save_dir={checkpoint_dir}", f"model.name={name}"]
    stages = []
    for update in (1, 2):
        run_training(overrides, update, output)
        stages.append(checkpoint_summary(checkpoint_dir / "model_last.pt", update))
        write_json(output / f"checkpoint_update_{update}_audit.json", stages[-1])
    if stages[1]["scheduler_last_epoch"] != stages[0]["scheduler_last_epoch"] + 1:
        raise RuntimeError("Learning-rate scheduler did not resume for exactly one update")
    config = load_config(overrides)
    subset = load_dataset(config)
    exp_name = f"{name}_{config.model.audio_representation.name}_LRS3_char"
    logdir = PROJECT / "runs" / exp_name
    ctc = ctc_backward_check(config, subset, checkpoint_dir / "model_last.pt", logdir)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(logdir)).Reload()
    losses = events.Scalars("loss")
    if {event.step for event in losses} != {1, 2} or any(not math.isfinite(event.value) for event in losses):
        raise RuntimeError("TensorBoard does not contain both finite training losses")
    result = {
        "status": "passed", "selection": "pilot", "updates_performed": 2,
        "selected_utterances": [row["utterance_key"] for row in rows], "source_data_contract": data.data_contract,
        "checkpoint_audits": stages, "ctc_backward": ctc, "tensorboard_logdir": str(logdir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.monotonic() - started,
        "checks": ["real_LibriTTS_50k_parent", "AdamW_update", "checkpoint_save_restore",
                   "optimizer_scheduler_EMA_resume", "CTC_backward", "TensorBoard_loss"],
    }
    write_json(output / "smoke_report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
