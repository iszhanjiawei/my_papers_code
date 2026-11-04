"""Measure full-model BF16 memory, including optimizer and rank-zero EMA.

Synthetic batches only: no pretrained checkpoint is published by this diagnostic.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_text_audio import CFMTextAudio
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.text_audio_checkpoint import initialize_from_s2c
from aligndit.model.trainer_text_audio import create_text_audio_ema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes', nargs='+', default=['4x300x200', '1x4000x2000'],
                        help='Batch x audio frames x text tokens; all must fit without dropping examples')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-activations', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).parents[2] / 'config/pretrain_libritts_text_audio.yaml')
    del cfg['hydra']  # This standalone diagnostic does not initialize Hydra's time resolvers.
    cfg.model.arch.checkpoint_activations = args.checkpoint_activations
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    if world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    torch.manual_seed(666)
    vocab = {' ': 0, 'a': 1, 'b': 2}
    model = CFMTextAudio(
        transformer=DiT_VT_MMDiT(**cfg.model.arch, text_num_embeds=180, mel_dim=64),
        num_channels=64, mel_spec_module=PrecomputedAudioRepresentation(64, 16000, 400),
        vocab_char_map=vocab, ctc_lambda=0, audio_drop_prob=0, cond_drop_prob=0,
    )
    initialize_from_s2c(model, cfg)
    print(f'rank={rank} S2c initialization verified', flush=True)
    model = model.cuda().train()
    ema = create_text_audio_ema(model, beta=0.999) if rank == 0 else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8)
    if world_size > 1:
        print(f'rank={rank} preparing DDP', flush=True)
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        print(f'rank={rank} DDP ready', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (SummaryWriter(str(args.output)) if rank == 0 else nullcontext()) as writer:
        for index, shape in enumerate(args.shapes, 1):
            batch, frames, tokens = map(int, shape.split('x'))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            audio = torch.randn(batch, frames, 64, device='cuda')
            speaker = torch.nn.functional.normalize(torch.randn(batch, 192, device='cuda'), dim=-1)
            text = [('ab ' * ((tokens + 2) // 3))[:tokens] for _ in range(batch)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, components, _, _ = model(audio, text, speaker_embedding=speaker)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite loss or gradient in full-model memory probe')
            optimizer.step()
            if ema is not None:
                ema.update()
            torch.cuda.synchronize()
            result = dict(shape=shape, rank=rank, world_size=world_size,
                          loss=float(loss.detach()), grad_norm=float(norm),
                          checkpoint_activations=args.checkpoint_activations,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          seconds=time.monotonic() - started)
            results.append(result)
            if writer is not None:
                writer.add_scalar('diagnostic/total', result['loss'], index)
                writer.add_scalar('diagnostic/flow', components['diff_loss'], index)
                writer.add_scalar('diagnostic/peak_allocated_gib', result['peak_allocated_gib'], index)
                writer.flush()
            print(json.dumps(result), flush=True)
            (args.output / f'measurements_rank_{rank}.json').write_text(json.dumps(results, indent=2) + '\n')
            del audio, speaker, loss
    if world_size > 1:
        dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
