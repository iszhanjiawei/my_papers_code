"""Check a real TPCA D1 batch without a Trainer, checkpoint, or event writer.

Run from the experiment root with PYTHONPATH=src. Examples::

    python -u src/aligndit/script/misc/validate_tpca_training.py --device cuda:0
    python -u src/aligndit/script/misc/validate_tpca_training.py \
        --selection dynamic --batches 3 --batch-size 32 --pretrained

``mixed`` deliberately combines real short/long utterances to stress padding;
other selections use the actual DynamicBatchSampler. Frames always means the
sum of valid mel frames, not the padded tensor size. This is a single-process
validation, so it does not establish DDP or EMA memory feasibility.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.dataset import load_dataset_mel
from aligndit.model.modules import MelSpec_tacotron
from aligndit.model.tpca import OccurrenceCTCAligner
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.utils import get_tokenizer


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="finetune_celebvdub_mm_d1_hunyuan_tpca")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--frames", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum samples; at least four are required.")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--selection", choices=("mixed", "dynamic", "long", "short"), default="mixed")
    parser.add_argument("--max-candidates", type=int, default=1000, help="Maximum sampler batches inspected.")
    parser.add_argument("--pretrained", nargs="?", const="config", default=None,
                        help="Optional EMA warm start; no argument uses ckpts.pretrained_path.")
    parser.add_argument("--override", action="append", default=[], help="Hydra override, repeatable.")
    args = parser.parse_args()
    if not 1 <= args.frames <= 9000:
        parser.error("--frames must be between 1 and 9000")
    if args.batch_size < 4 or args.batches < 1 or args.max_candidates < 1:
        parser.error("--batch-size >= 4, --batches >= 1 and --max-candidates >= 1 are required")
    return args


def load_warm_start(model, path):
    """Copy only matching EMA names/shapes; do not initialize a Trainer."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    source = checkpoint.get("ema_model_state_dict")
    if source is None:
        raise ValueError("Expected ema_model_state_dict in the optional pretrained checkpoint")
    target = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in source.items():
        key = key.removeprefix("ema_model.")
        if key in target and torch.is_tensor(value) and value.shape == target[key].shape:
            compatible[key] = value
        else:
            skipped.append(key)
    if not compatible:
        raise RuntimeError("No pretrained EMA parameters match this model")
    result = model.load_state_dict(compatible, strict=False)
    emit(stage="pretrained", path=str(path), matched=len(compatible),
         missing=len(result.missing_keys), skipped=len(skipped), skipped_examples=skipped[:8])
    del checkpoint, source, compatible, target
    gc.collect()


def load_item(dataset, index):
    item = dataset[index]
    if item is None:
        raise RuntimeError(f"Dataset row {index} failed to load; validation does not silently replace bad samples")
    return item


def actual_frames(item):
    return int(item["mel_spec"].shape[-1])


def selected_batches(dataset, args):
    if args.selection == "mixed":
        ordered = sorted((float(dataset.get_frame_len(i)), i) for i in range(len(dataset)))
        lengths = [x[0] for x in ordered]
        used = set()
        for batch_number in range(args.batches):
            items, indices, remaining = [], [], args.frames
            # Unequal target lengths keep complete transcripts and actual
            # utterances intact while exercising substantial batch padding.
            targets = np.linspace(1.5, 0.5, args.batch_size) * args.frames / args.batch_size
            for slot, target in enumerate(targets):
                slots_left = args.batch_size - slot - 1
                ceiling = min(float(target), remaining - (lengths[0] + 8) * slots_left)
                position = bisect.bisect_right(lengths, ceiling) - 1
                found = False
                while position >= 0:
                    index = ordered[position][1]
                    position -= 1
                    if index in used:
                        continue
                    item = load_item(dataset, index)
                    count = actual_frames(item)
                    if count > ceiling or count > remaining:
                        continue
                    items.append(item)
                    indices.append(index)
                    used.add(index)
                    remaining -= count
                    found = True
                    break
                if not found:
                    raise RuntimeError("Cannot fill a mixed batch of complete utterances within --frames; increase the budget")
            yield indices, items
        return

    sampler = DynamicBatchSampler(SequentialSampler(dataset), args.frames,
                                  max_samples=args.batch_size, random_seed=args.seed, drop_residual=False)
    candidates = list(sampler)
    if args.selection in ("long", "short"):
        candidates.sort(key=lambda ids: max(dataset.get_frame_len(i) for i in ids),
                        reverse=args.selection == "long")
    accepted = 0
    for indices in candidates[:args.max_candidates]:
        if len(indices) < 4:
            continue
        items = [load_item(dataset, i) for i in indices]
        counts = [actual_frames(item) for item in items]
        # The sampler estimates frames from duration; check actual AV-grid
        # lengths rather than silently exceeding the requested stress budget.
        if sum(counts) > args.frames or len(set(counts)) < 2:
            continue
        yield indices, items
        accepted += 1
        if accepted == args.batches:
            return
    raise RuntimeError(f"Only {accepted} qualifying padded batches found; increase --max-candidates or --frames")


def gradient_summary(named_parameters):
    with_grad = 0
    max_abs = 0.0
    bad = []
    for name, parameter in named_parameters:
        grad = parameter.grad
        if grad is None:
            continue
        with_grad += 1
        if not bool(torch.isfinite(grad).all()):
            bad.append(name)
        else:
            max_abs = max(max_abs, float(grad.detach().abs().max()))
    if bad:
        raise RuntimeError(f"Nonfinite gradients: {bad[:20]}")
    return {"parameters_with_grad": with_grad, "max_gradient_abs": max_abs}


def main():
    args = arguments()
    root = Path(__file__).resolve().parents[4]
    os.chdir(root)
    with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base="1.3"):
        cfg = compose(config_name=args.config, overrides=args.override)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but unavailable on the selected device")

    data_dir = cfg.datasets.data_dir
    tokenizer = cfg.model.tokenizer
    if tokenizer in ("custom", "byte"):
        raise ValueError("This D1 validation expects the real CelebVDub character tokenizer")
    vocab_path = str(Path(data_dir) / f"{cfg.datasets.name}_{tokenizer}" / "vocab.txt")
    vocab, vocab_size = get_tokenizer(vocab_path, "custom")
    mel_kwargs = OmegaConf.to_container(cfg.model.mel_spec, resolve=True)
    model_cls = get_class(f"aligndit.model.{cfg.model.backbone}")
    model = CFM_VT(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size,
                              mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_module=MelSpec_tacotron(**mel_kwargs),
        mel_spec_kwargs={k: v for k, v in mel_kwargs.items() if k != "mel_spec_type"},
        vocab_char_map=vocab,
        ctc_lambda=cfg.model.ctc_lambda,
        tpca_visual_ctc_lambda=cfg.model.tpca_visual_ctc_lambda,
        tpca_path_lambda=cfg.model.tpca_path_lambda,
    )
    if args.pretrained:
        load_warm_start(model, cfg.ckpts.pretrained_path if args.pretrained == "config" else args.pretrained)
    for name in ("audio_drop_prob", "text_drop_prob", "video_drop_prob", "cond_drop_prob"):
        setattr(model, name, 0.0)
    if not bool(getattr(model.transformer, "tpca_enabled", False)):
        raise RuntimeError("This check requires tpca_enabled=true")
    enabled_step = int(cfg.model.arch.tpca_warmup_steps + cfg.model.arch.tpca_ramp_steps)
    model.transformer.set_tpca_step(enabled_step)
    model.to(device).train()

    dataset = load_dataset_mel(
        cfg.datasets.name, tokenizer, dataset_type="CustomDataset_mel_video", data_dir=data_dir,
        mel_spec_module=MelSpec_tacotron(**mel_kwargs),
        mel_spec_kwargs={k: v for k, v in mel_kwargs.items() if k != "mel_spec_type"},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.learning_rate))
    layers = list(cfg.model.arch.tpca_layers)
    query_parameters = [model.transformer.transformer_blocks[i].cross_attn.audio_cross_q.weight for i in layers]
    aligners = [(name, module) for name, module in model.named_modules() if isinstance(module, OccurrenceCTCAligner)]
    if not aligners:
        raise RuntimeError("No OccurrenceCTCAligner found in the real model")
    captured = {}

    def capture_auxiliary(_module, _inputs, output):
        aux = output[1].get("__tpca__")
        if aux is None:
            raise RuntimeError("Transformer returned no TPCA auxiliary losses")
        # Save tensors before CFM removes the special entry and converts its
        # public logging dictionary to Python floats.
        captured["path_loss"] = aux["path_loss"]
        captured["visual_ctc_loss"] = aux["ctc_loss"]

    handle = model.transformer.register_forward_hook(capture_auxiliary)
    emit(stage="configuration", project=str(root), config=args.config, device=str(device), dtype=args.dtype,
         seed=args.seed, frame_budget=args.frames, max_samples=args.batch_size, selection=args.selection,
         parameters=sum(p.numel() for p in model.parameters()), tpca_step=enabled_step,
         checkpoint_activations=cfg.model.arch.checkpoint_activations,
         initialization="pretrained" if args.pretrained else "random", aligners=[n for n, _ in aligners])
    completed = 0
    try:
        for batch_index, (indices, items) in enumerate(selected_batches(dataset, args)):
            batch = dataset.collate_fn(items)
            lengths = batch["mel_lengths"].tolist()
            if len(lengths) < 4 or len(set(lengths)) < 2:
                raise RuntimeError("Validation must include at least four samples with actual padding")
            if sum(lengths) > args.frames:
                raise RuntimeError("Actual batch exceeds the specified valid-frame budget")
            emit(stage="batch", batch=batch_index, dataset_indices=indices, mel_lengths=lengths,
                 video_lengths=batch["video_lengths"].tolist(), text_lengths=batch["text_lengths"].tolist(),
                 valid_frames=sum(lengths), padded_frames=len(lengths) * max(lengths),
                 audio_paths=[item["audio_path"] for item in items])
            tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            captured.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
                loss, components, _cond, pred = model(
                    tensors["mel"].permute(0, 2, 1), text=tensors["text"],
                    lens=tensors["mel_lengths"], text_lens=tensors["text_lengths"],
                    video=tensors["video"], video_lens=tensors["video_lengths"],
                )
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(pred).all()):
                raise RuntimeError("Nonfinite loss or generated flow prediction")
            if any(not math.isfinite(float(value)) for value in components.values()):
                raise RuntimeError("A logged loss or TPCA diagnostic is nonfinite")
            if components.get("tpca_active") != 1.0 or components.get("tpca_path_scale", 0.0) <= 0:
                raise RuntimeError("TPCA prior/path loss is not active despite full conditioning and completed ramp")
            path_loss = captured["path_loss"]
            if not path_loss.requires_grad or float(path_loss.detach()) <= 0:
                raise RuntimeError("Path loss has no live positive objective")
            path_gradients = torch.autograd.grad(path_loss, query_parameters, retain_graph=True, allow_unused=True)
            path_norms = {}
            for layer, grad in zip(layers, path_gradients):
                if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.abs().max() > 0):
                    raise RuntimeError(f"Selected layer {layer} audio-to-text query has no finite nonzero path gradient")
                path_norms[str(layer)] = float(grad.detach().float().norm())
            del path_gradients
            loss.backward()
            gradient_info = gradient_summary(model.named_parameters())
            aligner_norms = {}
            for name, aligner in aligners:
                norm = sum(float(p.grad.detach().float().norm()) for p in aligner.parameters() if p.grad is not None)
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"Visual alignment head {name} has no finite nonzero total-loss gradient")
                aligner_norms[name] = norm
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.optim.max_grad_norm), error_if_nonfinite=True)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(f"Nonfinite parameter after optimizer step: {name}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            emit(stage="passed", batch=batch_index, loss=float(loss.detach()), components=components,
                 path_query_gradient_norms=path_norms, aligner_gradient_norms=aligner_norms,
                 total_gradient_norm=float(total_norm), **gradient_info,
                 elapsed_seconds=time.monotonic() - start,
                 peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                 peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None)
            completed += 1
            captured.clear()
            del loss, pred, _cond, tensors, batch, path_loss
    finally:
        handle.remove()
    emit(stage="complete", batches=completed, checkpoint_written=False, tensorboard_started=False,
         scope="single-process forward/backward/path-gradient/AdamW validation; not a DDP training run")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit(stage="failed", error_type=type(error).__name__, error=str(error))
        raise
