"""Versioned LRS3 posterior/CAM++/40-Hz caches using the existing codec protocol."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

from lrs3_build_inventory import digest, write_json
from aligndit.lrs3_media_quality import TERMINAL, excluded_map, load_terminal
from aligndit.script.misc.extract_campplus_speaker_embeddings import (
    AudioFeatureDataset, ExtractionItem, EXPECTED_CHECKPOINT_SHA256,
    atomic_save_npy, load_campplus, validate_speaker_embedding_array,
)

CODEC_SHA = '7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516'
POSTERIOR_PROTOCOL = 'semantic_vae_posterior_sample_v1'
VIDEO_PROTOCOL = 'avhubert_video_25hz_to_40hz_linear_align_corners_false_v1'


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


class Posterior(nn.Module):
    # Exact architecture and operation ordering of the audited CelebV-Dub extractor.
    def __init__(self, repo):
        super().__init__()
        sys.path.insert(0, str(repo))
        from dac.model.dac import Encoder
        from dac.model.attn_proj import AttnProjection
        self.encoder = Encoder(d_model=64, strides=[4,4,5,5], d_latent=1024)
        self.pre_block = AttnProjection(1024,64,num_heads=8)
        self.fc_mu = nn.Linear(64,64)
        self.fc_var = nn.Linear(64,64)

    def stats(self, waveform):
        h = self.pre_block(self.encoder(waveform).transpose(1,2))
        return self.fc_mu(h), torch.clamp(self.fc_var(h), min=-12, max=12)


def checked_array(path, shape, *, unit_norm=False):
    a = np.load(path, allow_pickle=False)
    if a.shape != tuple(shape) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError(f'Invalid cache {path}: {a.shape}/{a.dtype}, expected {shape}/float32')
    if unit_norm:
        validate_speaker_embedding_array(a, source=path)
    return a


def receipt_matches(path, identity, outputs):
    """Resume only outputs bound to the same source, protocol, and file hashes."""
    try:
        receipt = json.loads(path.read_text())
        return (receipt['identity'] == identity and
                all(receipt['outputs'][name] == digest(output)
                    for name, output in outputs.items()))
    except (OSError, ValueError, KeyError):
        return False


@torch.inference_mode()
def encode(model, waveform, frames, seed, device):
    x = torch.from_numpy(waveform).reshape(1,1,-1).to(device)
    x = F.pad(x, (0, frames * 400 - x.shape[-1]))
    mu, log_var = model.stats(x)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
    z = mu + torch.exp(0.5 * log_var) * noise
    if z.shape != (1,frames,64):
        raise RuntimeError(f'Unexpected latent shape {z.shape}')
    a = z[0].contiguous().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite codec output')
    return a


def load_encoder(spec, device, prefix):
    checkpoint = Path(spec['checkpoint']['ema_path'])
    if digest(checkpoint) != CODEC_SHA:
        raise RuntimeError('Semantic-VAE checkpoint hash mismatch')
    model = Posterior(Path(spec['semantic_vae_source']['repo']))
    state = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if int(state['step']) != 1000014 or not bool(state['initted']):
        raise RuntimeError('Unexpected codec EMA identity')
    selected = {k.removeprefix('ema_model.'):v for k,v in state.items()
                if k.startswith(('ema_model.encoder.','ema_model.pre_block.','ema_model.fc_mu.','ema_model.fc_var.'))}
    assert len(selected) == 145
    model.load_state_dict(selected,strict=True)
    model.eval().requires_grad_(False).to(device)
    del state, selected
    # Compare against a real legacy cache before writing any LRS3 posterior.
    with torch.inference_mode():
        model.stats(torch.zeros(1,1,400,device=device))
    golden_audio = Path(prefix + '/zjw524/projects/data/CelebVDub/audio/train/saP2eLOlPAc/0_0_0.wav')
    wave,sr = sf.read(golden_audio,dtype='float32')
    assert sr == 16000 and wave.shape == (378579,)
    actual = encode(model,wave,947,3721164253472998262,device)
    expected = np.load(prefix + '/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32/latents/train/saP2eLOlPAc/0_0_0.npy',allow_pickle=False)
    error = float(np.max(np.abs(actual-expected)))
    # Same numerical protocol on a different GPU/runtime can differ slightly.
    if not np.allclose(actual,expected,rtol=2e-5,atol=2e-5):
        raise RuntimeError(f'Legacy posterior regression failed: max_abs_error={error}')
    print(json.dumps({'golden_posterior_max_abs_error':error,'sha256':hashlib.sha256(actual.tobytes()).hexdigest()}),flush=True)
    return model


def init_spec(args):
    reference = json.loads(Path(args.prefix + '/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32/state/latents/spec.json').read_text())
    spec = {k:reference[k] for k in ['cache_schema_version','checkpoint','semantic_vae_source']}
    spec['dataset_root'] = str(args.root)
    spec['extraction'] = {'protocol':POSTERIOR_PROTOCOL,'latent_dim':64,'dtype':'float32',
                          'sample_rate':16000,'vae_hop_length':400,'base_posterior_seed':666,
                          'posterior_noise':'torch.randn with per-utterance CUDA Generator',
                          'latent_layout':'[time,channel]',
                          'source_code_sha256':digest(__file__)}
    path = args.root/'state/latents/spec.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['checkpoint'] != spec['checkpoint'] or old['semantic_vae_source'] != spec['semantic_vae_source']:
            raise RuntimeError('Cache codec identity changed')
        if any(old['extraction'].get(key) != value for key, value in spec['extraction'].items()
               if key != 'source_code_sha256'):
            raise RuntimeError('Cache posterior extraction protocol changed')
    else:
        write_json(path,spec)
    return json.loads(path.read_text())


def main():
    prefix = os.environ.get('ROOT_PREFIX','')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(prefix+'/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1'))
    p.add_argument('--manifest',type=Path)
    p.add_argument('--stage',choices=['init','audio','video40','decode'],required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--media-quality-terminal', type=Path)
    p.add_argument('--media-quality-terminal-sha256')
    args = p.parse_args(); args.prefix=prefix
    assert 0 <= args.rank < args.world_size
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    spec = init_spec(args)
    if args.stage == 'init':
        print('Codec contract initialized',flush=True); return
    rows = read_rows(args.manifest)[args.rank::args.world_size]
    if args.limit is not None:rows=rows[:args.limit]
    quality = load_terminal(args.root) if args.stage == 'video40' else None
    if quality is not None or args.media_quality_terminal or args.media_quality_terminal_sha256:
        if (args.stage != 'video40' or quality is None or args.limit is not None
                or args.media_quality_terminal is None
                or args.media_quality_terminal.resolve() != (args.root / TERMINAL).resolve()
                or args.media_quality_terminal_sha256 != quality['_sha256']):
            raise ValueError('Quality exclusions require an explicit hash-pinned canonical video40 pass')
    quality_excluded = excluded_map(quality)
    device = torch.device(args.device)
    model=cam=decoder=None
    if args.stage=='audio':
        model=load_encoder(spec,device,prefix)
        ckpt=Path(prefix+'/zjw524/projects/data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt')
        if digest(ckpt)!=EXPECTED_CHECKPOINT_SHA256:raise RuntimeError('CAM++ checkpoint hash mismatch')
        cam=load_campplus(ckpt,device)
    elif args.stage=='decode':
        from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
        decoder,_=load_semantic_vae_decoder(repo=Path(spec['semantic_vae_source']['repo']),
                         checkpoint_root=Path(spec['checkpoint']['ema_path']).parent.parent,
                         cache_spec=spec,device=device)
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    progress=args.root/f'reports/{args.stage}.{stamp}.rank{args.rank}.jsonl'
    progress.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); failed=0; excluded_count=0
    with progress.open('w',buffering=1) as log:
        for i,r in enumerate(rows):
            result={'sample_id':r['sample_id'],'utterance_key':r['utterance_key'],'stage':args.stage}
            if r['sample_id'] in quality_excluded:
                exclusion = quality_excluded[r['sample_id']]
                if (r['relative_stem'] != exclusion['relative_stem']
                        or r['original_clip_id'] != exclusion['original_clip_id']):
                    raise ValueError('Quality-excluded source identity differs')
                excluded_count += 1
                result.update(status='excluded', reason='media_quality:' + exclusion['reason'],
                              media_quality_terminal_sha256=quality['_sha256'])
                log.write(json.dumps(result,ensure_ascii=False)+'\n')
                continue
            try:
                frames=r['latent_frames']
                if args.stage=='audio':
                    a,sr=sf.read(r['audio_path'],dtype='float32')
                    if sr!=16000 or a.ndim!=1 or len(a)!=r['original_num_samples'] or not np.isfinite(a).all():
                        raise ValueError('Source audio does not match inventory')
                    dst=args.root/r['audio_relative_path'];dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():dst.symlink_to(r['audio_path'])
                    if dst.resolve()!=Path(r['audio_path']).resolve():raise ValueError('Wrong audio link')
                    lp=args.root/r['latent_relative_path']
                    sp=args.root/r['speaker_relative_path']
                    receipt_path=args.root/'state/audio_receipts'/f'{r["relative_stem"]}.json'
                    identity={'audio_sha256':digest(r['audio_path']),
                              'utterance_key':r['utterance_key'],'posterior_seed':r['posterior_seed'],
                              'posterior_protocol':POSTERIOR_PROTOCOL,'codec_sha256':CODEC_SHA,
                              'campplus_sha256':EXPECTED_CHECKPOINT_SHA256,
                              'speaker_protocol':'full_waveform_10s_repeat_fbank_mean_average_l2_v1'}
                    reuse=receipt_matches(receipt_path,identity,{'latent':lp,'speaker':sp})
                    if not reuse:atomic_save_npy(lp,encode(model,a,frames,r['posterior_seed'],device))
                    checked_array(lp,(frames,64))
                    if not reuse:
                        item=ExtractionItem(r['audio_path'],str(sp),r['split'])
                        _,features,error=AudioFeatureDataset([item])[0]
                        if error:raise ValueError(error)
                        with torch.inference_mode():
                            embedding=cam(features.to(device)).float().cpu().mean(dim=0)
                        embedding=F.normalize(embedding,dim=0).numpy().astype(np.float32,copy=False)
                        validate_speaker_embedding_array(embedding,source=sp)
                        atomic_save_npy(sp,embedding)
                    checked_array(sp,(192,),unit_norm=True)
                    outputs={'latent':digest(lp),'speaker':digest(sp)}
                    write_json(receipt_path,{'identity':identity,'outputs':outputs})
                    result.update(latent_sha256=outputs['latent'],speaker_sha256=outputs['speaker'],reused=reuse)
                elif args.stage=='video40':
                    src=args.root/r['video_25hz_relative_path']
                    a=checked_array(src,(r['video_frames'],1024))
                    dst=args.root/r['video_40hz_relative_path']
                    receipt_path=args.root/'state/video40_receipts'/f'{r["relative_stem"]}.json'
                    identity={'source_sha256':digest(src),'protocol':VIDEO_PROTOCOL,
                              'utterance_key':r['utterance_key'],'target_frames':frames}
                    reuse=receipt_matches(receipt_path,identity,{'video_40hz':dst})
                    if not reuse:
                        t=torch.from_numpy(a).T.unsqueeze(0)
                        aligned=F.interpolate(t,size=frames,mode='linear',align_corners=False)[0].T.contiguous().numpy()
                        atomic_save_npy(dst,aligned)
                    checked_array(dst,(frames,1024))
                    outputs={'video_40hz':digest(dst)}
                    write_json(receipt_path,{'identity':identity,'outputs':outputs})
                    result.update(source_sha256=identity['source_sha256'],video_40hz_sha256=outputs['video_40hz'],reused=reuse)
                else:
                    a=checked_array(args.root/r['latent_relative_path'],(frames,64))
                    with torch.inference_mode():
                        wav=decoder(torch.from_numpy(a).T.unsqueeze(0).to(device)).float().cpu().numpy().reshape(-1)
                    if not np.isfinite(wav).all() or len(wav)!=frames*400:raise ValueError('Invalid decoder waveform')
                    dst=args.root/'reports/reconstructions'/f'{r["sample_id"]}.wav'
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(dst,wav[:r['original_num_samples']],16000,subtype='PCM_16')
                    result.update(path=str(dst),decoded_samples=len(wav),peak=float(np.abs(wav).max()))
                result['status']='ok'
            except Exception as error:
                failed+=1;result.update(status='error',error=f'{type(error).__name__}: {error}')
                print(json.dumps(result),flush=True)
            log.write(json.dumps(result,ensure_ascii=False)+'\n')
            if (i+1)%20==0 or i+1==len(rows):
                print(json.dumps({'stage':args.stage,'rank':args.rank,'done':i+1,'total':len(rows),'failed':failed,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    summary={'stage':args.stage,'rank':args.rank,'requested':len(rows),'succeeded':len(rows)-failed-excluded_count,'failed':failed,
             'quality_excluded':excluded_count,'processing_complete':failed==0,
             'complete':failed==0 and excluded_count==0,
             'media_quality_terminal_sha256':quality['_sha256'] if quality else None,
             'manifest_sha256':digest(args.manifest),'progress_path':str(progress),
             'elapsed_seconds':time.monotonic()-started,'worker_sha256':digest(__file__)}
    write_json(progress.with_suffix('.summary.json'),summary)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
