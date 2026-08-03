"""Measure worst-batch memory for the real Semantic-VAE pretraining stack without writing checkpoints."""

from __future__ import annotations

import argparse
import gc
from importlib.resources import files
from pathlib import Path

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from aligndit.model.cfm_notext import CFM_notext
from aligndit.model.dataset import SemanticVaePretrainDataset
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.trainer_notext import Trainer_notext
from f5_tts.model.dataset import DynamicBatchSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=files("aligndit").joinpath("config/pretrain_semantic_vae.yaml"))
    parser.add_argument("--frame-budget", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=1)
    return parser.parse_args()


def batch_shape(dataset: SemanticVaePretrainDataset, indices: list[int]) -> dict[str, int]:
    lengths = [dataset.get_frame_len(index) for index in indices]
    batch_size = len(lengths)
    max_length = max(lengths)
    return {
        "attention_proxy": batch_size * max_length * max_length,
        "batch_size": batch_size,
        "effective_frames": sum(lengths),
        "max_length": max_length,
        "padded_frames": batch_size * max_length,
    }


def select_worst_batches(
    dataset: SemanticVaePretrainDataset, batches: list[list[int]], max_samples: int
) -> list[tuple[str, list[int], dict[str, int]]]:
    shaped = [(batch, batch_shape(dataset, batch)) for batch in batches]
    selectors = {
        "max_attention": lambda item: item[1]["attention_proxy"],
        "max_length": lambda item: item[1]["max_length"],
        "max_padded": lambda item: item[1]["padded_frames"],
    }
    full_batches = [item for item in shaped if item[1]["batch_size"] == max_samples]
    if full_batches:
        selectors["longest_full_batch"] = lambda item: item[1]["max_length"]

    selected: list[tuple[str, list[int], dict[str, int]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, score in selectors.items():
        candidates = full_batches if name == "longest_full_batch" else shaped
        batch, shape = max(candidates, key=score)
        identity = tuple(batch)
        if identity not in seen:
            selected.append((name, batch, shape))
            seen.add(identity)
    selected.sort(key=lambda item: item[2]["attention_proxy"])
    return selected


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def run_step(trainer: Trainer_notext, batch: dict[str, torch.Tensor], update: int, seed: int) -> dict[str, float]:
    accelerator = trainer.accelerator
    set_seed(seed * 1_000_003 + update * accelerator.num_processes + accelerator.process_index)
    trainer.optimizer.zero_grad()
    with accelerator.autocast():
        loss, components, _, _ = trainer.model(
            batch["mel"].permute(0, 2, 1),
            lens=batch["mel_lengths"],
            feature=batch["rep"],
            feature_lens=batch["rep_lengths"],
        )
    accelerator.backward(loss)
    grad_norm = accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite benchmark loss/gradient: loss={loss}, grad_norm={grad_norm}")
    if any(not torch.isfinite(torch.tensor(value)) for value in components.values()):
        raise FloatingPointError(f"Non-finite benchmark loss component: {components}")
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    if trainer.is_main:
        trainer.ema_model.update()
    torch.cuda.synchronize(accelerator.device)
    return {"grad_norm": float(grad_norm), "loss": float(loss), **components}


def main() -> None:
    args = parse_args()
    if args.frame_budget <= 0 or args.max_samples <= 0 or args.repeats < 2 or args.expected_world_size <= 0:
        raise ValueError("frame-budget/max-samples/world-size must be positive and repeats must be at least 2")

    cfg = OmegaConf.load(args.config)
    cfg.datasets.batch_size_per_gpu = args.frame_budget
    cfg.datasets.max_samples = args.max_samples
    channels = int(cfg.model.audio_representation.channels)
    model_cls = hydra.utils.get_class(f"aligndit.model.{cfg.model.backbone}")
    model = CFM_notext(
        transformer=model_cls(**cfg.model.arch, mel_dim=channels),
        mel_spec_module=PrecomputedAudioRepresentation(
            n_channels=channels,
            target_sample_rate=int(cfg.model.audio_representation.sample_rate),
            hop_length=int(cfg.model.audio_representation.hop_length),
        ),
        num_channels=channels,
        proj_lambda=float(cfg.model.proj_lambda),
    )
    trainer = Trainer_notext(
        model,
        epochs=1,
        learning_rate=float(cfg.optim.learning_rate),
        num_warmup_updates=int(cfg.optim.num_warmup_updates),
        checkpoint_path=None,
        batch_size_per_gpu=args.frame_budget,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=1,
        max_grad_norm=float(cfg.optim.max_grad_norm),
        logger=None,
        log_samples=False,
        bnb_optimizer=False,
        ema_kwargs=cfg.ema,
    )
    accelerator = trainer.accelerator
    if (
        accelerator.num_processes != args.expected_world_size
        or accelerator.mixed_precision != "bf16"
        or accelerator.device.type != "cuda"
    ):
        raise RuntimeError(
            "Memory benchmark runtime differs from its requested CUDA/BF16 contract; got "
            f"world={accelerator.num_processes}, precision={accelerator.mixed_precision}, device={accelerator.device}"
        )

    dataset = SemanticVaePretrainDataset(
        manifest_path=cfg.datasets.manifest_path,
        cache_root=cfg.datasets.cache_root,
        normalization_path=cfg.datasets.normalization_path,
    )
    sampler = DynamicBatchSampler(
        SequentialSampler(dataset),
        args.frame_budget,
        max_samples=args.max_samples,
        random_seed=int(cfg.seed),
        drop_residual=False,
    )
    selected = select_worst_batches(dataset, sampler.batches, args.max_samples)
    print(
        f"rank={accelerator.process_index} benchmark contract: world={accelerator.num_processes}, "
        f"frame_budget={args.frame_budget}, max_samples={args.max_samples}, "
        f"batches={len(sampler.batches)}, candidates={[(name, shape) for name, _, shape in selected]}",
        flush=True,
    )

    update = 0
    for name, indices, shape in selected:
        cpu_batch = dataset.collate_fn([dataset[index] for index in indices])
        batch = to_device(cpu_batch, accelerator.device)
        del cpu_batch
        for repeat in range(args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(accelerator.device)
            metrics = run_step(trainer, batch, update, int(cfg.seed))
            update += 1
            allocated = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(accelerator.device) / 1024**3
            print(
                f"rank={accelerator.process_index} candidate={name} repeat={repeat + 1}/{args.repeats} shape={shape} "
                f"peak_allocated_gib={allocated:.3f} peak_reserved_gib={reserved:.3f} metrics={metrics}",
                flush=True,
            )
        del batch

    accelerator.wait_for_everyone()
    print(
        f"rank={accelerator.process_index} Semantic-VAE pretraining memory benchmark completed without checkpoint writes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
