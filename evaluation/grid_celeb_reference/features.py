"""Extract input-only visual features and independent reference caches."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from protocol import (DEFAULT_RUN, PROJECT, DATA, REPA, TOOLS, WEIGHTS,
                      rows, add_repa, load_module, save_json, sha)


def npy(path, a):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite feature: ' + str(path))
    with path.with_suffix('.tmp').open('wb') as f:
        np.save(f, np.asarray(a, dtype=np.float32), allow_pickle=False)
    path.with_suffix('.tmp').replace(path)


def references(run):
    add_repa()
    from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae
    from aligndit.script.eval.prepare_grid_reference import posterior_stats
    from aligndit.script.misc.extract_campplus_speaker_embeddings import load_campplus, AudioFeatureDataset, ExtractionItem
    unique = {r['ref_id']: r for r in rows(run)}
    spec = json.loads((DATA / 'CelebVDub_svae1000k_sample_seed666_fp32/state/latents/spec.json').read_text())
    vae, meta = load_semantic_vae(repo=PROJECT / 'papers_codes/Semantic-VAE',
             checkpoint_root=PROJECT / 'Semantic-VAE/semantic_vae_1000k', cache_spec=spec, device=torch.device('cpu'))
    del vae.decoder
    vae.cuda()
    camp_path = DATA / 'pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt'
    camp = load_campplus(camp_path, torch.device('cuda:0'))
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with torch.inference_mode():
        for i, (ident, row) in enumerate(unique.items(), 1):
            audio, sr = torchaudio.load(row['ref_audio'])
            if sr != 16000 or audio.shape != (1, row['ref_num_samples']) or sha(row['ref_audio']) != row['ref_audio_sha256']:
                raise ValueError('Reference waveform changed')
            padded = F.pad(audio.unsqueeze(0), (0, row['ref_frames'] * 400 - audio.shape[-1])).cuda()
            mu, logvar = posterior_stats(vae, padded)
            seed = int.from_bytes(hashlib.sha256(('666:' + ident).encode()).digest()[:8], 'little') % (2**63-1)
            g = torch.Generator(device='cuda:0').manual_seed(seed)
            noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=g)
            latent = (mu + torch.exp(.5 * logvar) * noise).squeeze(0).cpu().numpy()
            if latent.shape != (row['ref_frames'], 64):
                raise ValueError('Invalid reference latent shape')
            item, feature, err = AudioFeatureDataset([ExtractionItem(row['ref_audio'], row['ref_speaker_path'], 'reference')])[0]
            if err:
                raise ValueError(err)
            emb = F.normalize(camp(feature.cuda()).float().cpu().mean(0), dim=0).numpy()
            npy(row['ref_latent_path'], latent)
            npy(row['ref_speaker_path'], emb)
            records.append({'ref_id': ident, 'posterior_seed': seed, 'audio_sha256': sha(row['ref_audio']),
                            'latent_sha256': sha(row['ref_latent_path']), 'speaker_sha256': sha(row['ref_speaker_path'])})
            print(f'References {i}/66: {ident}', flush=True)
    save_json(run / 'references/features.json', {'count': 66, 'vae': meta, 'campplus_sha256': sha(camp_path),
              'cache_spec': spec, 'records': records})


def visual(run, shard, nshard, limit):
    add_repa()
    fairseq_root = DATA / 'av_hubert/fairseq/fairseq'
    sys.path.insert(0, str(fairseq_root))
    sys.path.insert(0, str(DATA / 'av_hubert/avhubert/avhubert'))
    import fairseq
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=str(DATA / 'av_hubert/avhubert/avhubert')))
    from aligndit.script.misc.extract_avhubert_from_only_video import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    avreader = AVHubertFeatureReader(str(PROJECT.parent.parent / 'alignDiT_pretrain_models/large_vox_iter5.pt'), custom_utils=avhubert_utils)
    native = load_module('grid_native_preprocessing', TOOLS / 'stage45_video_embed.py')
    lip_model = native.load_lipreading(torch.device('cuda:0'))
    face_model = native.load_emonet(torch.device('cuda:0'))
    # The uploaded helper predates face_alignment's compile=True default.
    # Eager inference uses the identical FAN/S3FD weights and preprocessing,
    # without expensive compiler warm-up for every independent GPU worker.
    if 'compile' in inspect.signature(native.face_alignment.FaceAlignment).parameters:
        lm_type = getattr(native.face_alignment.LandmarksType, 'TWO_D', None)
        if lm_type is None:
            lm_type = native.face_alignment.LandmarksType._2D
        fa = native.face_alignment.FaceAlignment(lm_type, flip_input=False, device='cuda:0', compile=False)
    else:
        fa = native.make_face_aligner(0)
    items = rows(run)
    if limit:
        items = items[:limit]
    items = items[shard::nshard]
    records = []
    ffmpeg = Path(sys.executable).resolve().parent / 'ffmpeg'
    for i, row in enumerate(items, 1):
        record_path = run / 'assets/records' / (row['target_id'] + '.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            for key, expected in record['artifact_sha256'].items():
                if sha(record['artifact_paths'][key]) != expected:
                    raise ValueError('Changed visual artifact: ' + row['target_id'])
            records.append(record)
            print(f'Visual {i}/{len(items)} cached: {row["target_id"]}', flush=True)
            continue
        frames = native.read_frames(row['source_video'])
        if len(frames) != row['native_frames']:
            raise ValueError(f'Raw video / source frame count mismatch: {row["target_id"]}, {len(frames)} vs {row["native_frames"]}')
        landmarks = native.detect_landmarks(fa, frames)
        mouth = native.crop_mouth_sequence(frames, landmarks)
        if mouth is None or mouth.shape != (len(frames), 96, 96):
            raise ValueError('Native mouth cropping failed: ' + row['target_id'])
        # Lossless mouth video shared by visual conditioning and AV scoring.
        dest = Path(row['mouth_video'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(ffmpeg), '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', '96x96',
               '-r', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv420p', '-threads', '1', str(dest)]
        subprocess.run(cmd, input=mouth.tobytes(), check=True, timeout=60)
        with torch.inference_mode():
            lip = lip_model(torch.from_numpy((mouth.astype(np.float32) / 255 - native.LIP_MEAN) / native.LIP_STD)[None,None].cuda(), lengths=[len(frames)])
            face = []
            for b in range(0, len(frames), native.FACE_BATCH):
                crops = np.stack([native.face_crop_from_lm(f, lm) for f,lm in zip(frames[b:b+native.FACE_BATCH], landmarks[b:b+native.FACE_BATCH])])
                face.append(native.emonet_256(face_model, torch.from_numpy(crops.astype(np.float32)/255).permute(0,3,1,2).cuda()))
            video = avreader.get_feats((str(dest), None)).cpu().float()
        if tuple(video.shape) != (len(frames), 1024):
            raise ValueError('Invalid AV-HuBERT visual output')
        npy(row['lip_feature'], lip.squeeze(0).cpu().numpy())
        npy(row['face_feature'], np.concatenate(face, axis=0))
        npy(row['video_25hz_path'], video.numpy())
        aligned = F.interpolate(video.T[None], size=row['target_frames'], mode='linear', align_corners=False)[0].T
        npy(row['target_video_path'], aligned.numpy())
        # Evaluation-only GT: decode the original video's synchronous audio,
        # not GRID's separately distributed endpoint-trimmed waveform.
        raw = subprocess.check_output([str(ffmpeg), '-v', 'error', '-nostdin', '-i', row['source_video'],
                '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'], timeout=60)
        gt = np.frombuffer(raw, dtype='<i2').copy()
        tail = row['target_num_samples'] - len(gt)
        if abs(tail) > 1600:
            raise ValueError('Source audio/video discrepancy exceeds 100 ms')
        gt = np.pad(gt, (0, max(tail,0)))[:row['target_num_samples']]
        gt_path = run / 'assets/gt' / (row['target_id'] + '.wav')
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(gt_path, gt, 16000, subtype='PCM_16')
        paths = {k: row[k] for k in ('mouth_video','lip_feature','face_feature','video_25hz_path','target_video_path')}
        paths['gt_audio'] = str(gt_path)
        record = {'target_id': row['target_id'], 'frames': len(frames), 'landmark_valid_frames': sum(x is not None for x in landmarks),
                  'gt_audio_tail_adjustment_samples': tail, 'artifact_paths': paths,
                  'artifact_sha256': {k: sha(p) for k,p in paths.items()},
                  'preprocessing_source_sha256': sha(TOOLS / 'stage45_video_embed.py')}
        save_json(record_path, record)
        records.append(record)
        print(f'Visual {i}/{len(items)} complete: {row["target_id"]}', flush=True)
    save_json(run / f'assets/shard{shard}.json', {'count': len(records), 'shard': shard, 'nshard': nshard,
              'native_weight_sha256': {str(p.relative_to(WEIGHTS)): sha(p) for p in [WEIGHTS/'lrw_resnet18_dctcn_video.pth',WEIGHTS/'emonet/pretrained/emonet_8.pth']},
              'avhubert_checkpoint_sha256': sha(PROJECT.parent.parent / 'alignDiT_pretrain_models/large_vox_iter5.pt')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--stage', choices=['references','visual'], required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.stage == 'references':
        references(args.run)
    else:
        visual(args.run, args.shard, args.nshard, args.limit)
