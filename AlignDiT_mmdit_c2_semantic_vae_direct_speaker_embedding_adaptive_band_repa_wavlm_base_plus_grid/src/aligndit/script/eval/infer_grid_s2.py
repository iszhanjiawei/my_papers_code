"""GRID Setting2 GTText dubbing: independent same-speaker reference prompt; target video and supplied dialogue."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from aligndit.script.eval.grid_s1_common import identity, load_rows, output_record, publish, read, sha
from aligndit.script.eval.grid_s2_common import PROTOCOL, load_pairs, SELECTION
from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs, reference_target_text
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    atomic_save_waveform, build_model, load_composed_config, load_normalization,
)
from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
from aligndit.script.misc.prepare_grid import artifact_path, validate_receipt
from aligndit.model.grid_dataset import NORMALIZATION_SHA256, VOCAB_SHA256


def run(a):
    rows, done = load_rows(a.cache_root)
    pairs, references = load_pairs(a.cache_root, a.output_dir, rows)
    if a.limit is not None:
        if not 0 < a.limit <= len(rows):
            raise ValueError('Invalid smoke limit')
        rows = rows[:a.limit]
    if not 0 <= a.shard < a.nshard <= len(rows):
        raise ValueError('Invalid shard')
    config = load_composed_config(a.config.resolve())
    normalization = Path(config.datasets.normalization_path)
    if sha(normalization) != NORMALIZATION_SHA256 or sha(config.datasets.vocab_path) != VOCAB_SHA256:
        raise RuntimeError('Pretrained normalization or tokenizer changed')
    mean, std, _ = load_normalization(normalization)
    contract = {'protocol': PROTOCOL, 'requested_count': len(rows), 'partial_smoke_test': a.limit is not None,
                'checkpoint': {'path': str(a.checkpoint.resolve()), 'sha256': sha(a.checkpoint), 'update': a.step, 'weights': 'EMA'},
                'manifest_sha256': sha(a.cache_root / 'manifests/val.jsonl'),
                'cache_completion_sha256': sha(a.cache_root / 'complete.json'),
                'reference_pairs_sha256': sha(a.output_dir/'reference_pairs.json'), 'reference_selection': SELECTION,
                'text_source': 'supplied_target_GT_dialogue; no VSR', 'target_audio_read_for_generation': False,
                'target_latent_or_speaker_read_for_generation': False,
                'normalization_sha256': sha(normalization), 'config': OmegaConf.to_container(config, resolve=True),
                'generation': {'seed': 0, 'nfe': 32, 'ode': 'euler', 'epss': True, 'sway': -1, 'cfg_text': 5, 'cfg_video': 2, 'batch_size': 1},
                'reference': 'same-speaker different-text validation clip canonical audio; latent and CAM++ both from this independent reference',
                'output': 'only target continuation; prompt removed; exact original duration; 16k mono PCM16'}
    protocol_path = a.output_dir / 'protocol.json'
    if protocol_path.exists() and read(protocol_path) != contract:
        raise RuntimeError('Output directory belongs to another experiment')
    publish(protocol_path, contract)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    model = build_model(a.config.resolve(), a.checkpoint.resolve(), a.step, device)
    decoder, decoder_meta = load_semantic_vae_decoder(repo=a.semantic_vae_repo, checkpoint_root=a.semantic_vae_checkpoint,
                                                     cache_spec=read(a.cache_root / 'state/latents/spec.json'), device=device)
    publish(a.output_dir / f'decoder_shard{a.shard}.json', decoder_meta)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    with torch.inference_mode():
        for row in tqdm(rows[a.shard::a.nshard], desc=f'GRID S2 shard {a.shard}', mininterval=5):
            out = a.output_dir / row['audio_relative_path']
            receipt = a.output_dir / 'receipts' / (identity(row) + '.json')
            if out.exists() or receipt.exists():
                output_record(a.output_dir, row, PROTOCOL)
                continue
            ref = references[identity(row)]
            if not validate_receipt(a.cache_root, row, 'video', done['extraction_sha256']):
                raise RuntimeError('Missing target video receipt')
            for stage in ('audio', 'latent', 'speaker'):
                if not validate_receipt(a.cache_root, ref, stage, done['extraction_sha256']):
                    raise RuntimeError(f'Missing reference {stage} receipt: {identity(ref)}')
            for stage, field in (('audio','reference_audio_sha256'),('latent','reference_latent_sha256'),('speaker','reference_speaker_sha256')):
                if sha(artifact_path(a.cache_root, ref, stage)) != pairs[identity(row)][field]:
                    raise RuntimeError('Reference artifact differs from fixed pairing')
            latent = np.load(artifact_path(a.cache_root, ref, 'latent'))
            video = np.load(artifact_path(a.cache_root, row, 'video'))
            speaker = torch.from_numpy(np.load(artifact_path(a.cache_root, ref, 'speaker'))).to(device)
            frames = row['latent_frames']
            text = reference_target_text(ref['text'], row['text'])
            cond, combined_video, prompt_frames, target_frames, duration = make_sampling_inputs(
                latent, video, text, mean, std)
            if frames != target_frames or row['original_num_samples'] != row['source_video_frames']*640:
                raise RuntimeError('Target duration differs from video timebase')
            generated, _ = model.sample(cond=torch.from_numpy(cond)[None].to(device),
                text=[text], duration=torch.tensor([duration], device=device),
                video=torch.from_numpy(combined_video)[None].to(device), lens=torch.tensor([prompt_frames], device=device),
                steps=32, cfg_strength=5, cfg_strength_v=2, sway_sampling_coef=-1, seed=0, use_epss=True,
                speaker_embedding=speaker[None])
            target = generated[:, prompt_frames:prompt_frames+target_frames].float()
            if target.shape != (1, frames, 64) or not torch.isfinite(target).all():
                raise RuntimeError('Invalid generated latent')
            waveform = decoder((target*std_t+mean_t).transpose(1, 2)).squeeze(0).float().cpu()
            if waveform.shape != (1, frames*400) or not torch.isfinite(waveform).all():
                raise RuntimeError('Invalid decoded waveform')
            atomic_save_waveform(out, waveform[:, :row['original_num_samples']])
            publish(receipt, {'id': identity(row), 'relative_path': row['audio_relative_path'], 'sha256': sha(out),
                              'samples': row['original_num_samples'], 'reference_id': identity(ref), 'prompt_frames': prompt_frames,
                              'target_frames': target_frames, 'sampler_frames': duration, 'protocol_sha256': sha(protocol_path)})
    publish(a.output_dir / f'shard{a.shard}_complete.json', {'count': len(rows[a.shard::a.nshard]), 'shard': a.shard, 'nshard': a.nshard})


def args():
    base = Path(os.environ.get('ROOT_PREFIX', '') + '/zjw524/projects')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--step', type=int, default=100000)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=base/'data/GRID_svae1000k_sample_seed666_fp32')
    p.add_argument('--config', type=Path, default=Path(__file__).parents[2]/'config/finetune_grid.yaml')
    p.add_argument('--semantic-vae-repo', type=Path, default=base/'alignDiT_idea6/papers_codes/Semantic-VAE')
    p.add_argument('--semantic-vae-checkpoint', type=Path, default=base/'alignDiT_idea6/Semantic-VAE/semantic_vae_1000k')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int)
    return p.parse_args()


if __name__ == '__main__':
    run(args())
