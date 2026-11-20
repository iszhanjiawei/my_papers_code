"""Recover frozen external references and extract this model's VAE/CAM++ inputs."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import wave
from pathlib import Path

from aligndit.script.eval.grid_s1_common import ensure_contract, identity, load_rows, publish, read, sha
from aligndit.script.eval.grid_zeroshot_common import frozen_document, jsonl_bytes, normalize_wer


def prepare(a):
    from aligndit.script.eval.prepare_grid_reference import (
        atomic_bytes, extract_reference, validate_cached_record, write_json,
    )
    from aligndit.script.misc.prepare_grid import artifact_path, validate_receipt

    a.shared_dir.mkdir(parents=True, exist_ok=True)
    with (a.shared_dir/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        portable, blob, refs, assignments, frozen = frozen_document(a.document)
        rows, done = load_rows(a.cache_root)
        by_id = {identity(r).removeprefix('val/'): r for r in rows}
        assert set(by_id) <= {r['id'] for r in portable}
        source_meta = read(a.cache_root/'manifests/inventory_meta.json')
        excluded = {r['id']: r for r in source_meta['exclusions'] if r['split'] == 'val'}
        omitted = [r for r in portable if r['id'] not in by_id]
        if len(omitted) != 7 or any(r['id'] not in excluded for r in omitted):
            raise RuntimeError('Unexpected difference between frozen targets and verified GRID cache')
        omissions = [dict(id=r['id'], global_index=r['index'], reason=excluded[r['id']]['reason']) for r in omitted]
        reference_plan = []
        for ref in refs:
            source = a.celeb_root/ref['audio_relative']
            if sha(source) != ref['audio_sha256']:
                raise RuntimeError(f'Frozen reference WAV differs: {source}')
            with wave.open(str(source)) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getnframes()) != (ref['sample_rate'], ref['channels'], ref['frames']):
                    raise RuntimeError('Reference header differs from frozen document')
            transcript = a.celeb_root/ref['text_relative']
            if sha(transcript) != ref['text_sha256'] or ' '.join(transcript.read_text().lower().split()) != ref['text']:
                raise RuntimeError(f'Frozen prepared reference TXT differs: {transcript}')
            samples = math.ceil(ref['frames']*16000/ref['sample_rate'])
            seed = int.from_bytes(hashlib.sha256(('666:'+ref['id']).encode()).digest()[:8], 'little') % (2**63-1)
            relative = Path(ref['source_id'])
            reference_plan.append(dict(ref_id=ref['id'], ref_source_audio=str(source.resolve()),
                ref_source_audio_sha256=ref['audio_sha256'], ref_source_sample_rate=ref['sample_rate'],
                ref_source_channels=ref['channels'], ref_source_num_samples=ref['frames'],
                ref_transcript_path=str(transcript.resolve()), ref_transcript_sha256=ref['text_sha256'],
                ref_text=ref['text'], ref_num_samples=samples, ref_frames=math.ceil(samples/400), posterior_seed=seed,
                ref_audio=str((a.shared_dir/'references/audio'/relative.with_suffix('.wav')).resolve()),
                ref_latent_path=str((a.shared_dir/'references/latents'/relative.with_suffix('.npy')).resolve()),
                ref_speaker_path=str((a.shared_dir/'references/speakers'/relative.with_suffix('.npy')).resolve())))
        spec = read(a.cache_root/'state/latents/spec.json')
        from aligndit.script.misc.extract_campplus_speaker_embeddings import EXPECTED_CHECKPOINT_SHA256
        if sha(a.speaker_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError('CAM++ checkpoint differs from this model')
        if sha(a.semantic_vae_checkpoint/'dac/ema_state_dict.pth') != spec['checkpoint']['ema_sha256']:
            raise RuntimeError('Semantic-VAE checkpoint differs from the GRID training cache')
        plan = dict(document=str(a.document.resolve()), document_sha256=sha(a.document),
                    frozen_portable_sha256=hashlib.sha256(blob).hexdigest(), frozen_metadata=frozen,
                    assignments=assignments, count=3274, reference_count=33, document_count=3281,
                    cache_root=str(a.cache_root.resolve()), cache_manifest_sha256=sha(a.cache_root/'manifests/val.jsonl'),
                    cache_completion_sha256=sha(a.cache_root/'complete.json'), omissions=omissions,
                    speaker_checkpoint=str(a.speaker_checkpoint.resolve()), speaker_checkpoint_sha256=sha(a.speaker_checkpoint),
                    vae_checkpoint_sha256=spec['checkpoint']['ema_sha256'],
                    reference_plan=reference_plan,
                    reference_frontend='complete WAV; arithmetic channel mean; torchaudio 16k; PCM16; no RMS normalization',
                    latent='64-D/40Hz posterior sample; base seed 666 + full external ID SHA256; zero pad to 400; logvar clamp [-12,12]',
                    speaker='CAM++ 192-D; full utterance circular 10s padding; mean and L2 normalization')
        ensure_contract(a.shared_dir/'plan.json', plan)
        atomic_bytes(a.shared_dir/'document_portable_pairs.jsonl', blob)
        print(f'Frozen {len(rows)} targets, {len(refs)} references; {len(omissions)} prior source exclusions', flush=True)
        if a.plan_only:
            return
        completed, pending = {}, []
        for ref in reference_plan:
            receipt = a.shared_dir/'references/records'/Path(ref['ref_id']).relative_to('celebvdub/test').with_suffix('.json')
            if receipt.exists():
                record = read(receipt)
                validate_cached_record(record, ref)
                completed[ref['ref_id']] = record
            else:
                pending.append((ref, receipt))
        if pending:
            import torch
            from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae
            from aligndit.script.misc.extract_campplus_speaker_embeddings import load_campplus
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            device = torch.device(a.device)
            vae, vae_meta = load_semantic_vae(repo=a.semantic_vae_repo, checkpoint_root=a.semantic_vae_checkpoint,
                                            cache_spec=spec, device=torch.device('cpu'))
            del vae.decoder
            vae.to(device=device, dtype=torch.float32)
            campplus = load_campplus(a.speaker_checkpoint, device)
            for i, (ref, receipt) in enumerate(pending, 1):
                record = extract_reference(ref, vae, campplus, device)
                write_json(receipt, record)
                completed[ref['ref_id']] = record
                print(f'Prepared external VAE/CAM++ reference {i}/{len(pending)}: {ref["ref_id"]}', flush=True)
            ensure_contract(a.shared_dir/'extraction_runtime.json', dict(torch=torch.__version__, device=str(device),
                            device_name=torch.cuda.get_device_name(device), vae_loader=vae_meta))
        result = []
        for pair in portable:
            if pair['id'] not in by_id:
                continue
            row = by_id[pair['id']]
            if normalize_wer(pair['text'], 'digits_silence') != row['text']:
                raise RuntimeError('Frozen raw scoring truth and generation text describe different targets')
            if not validate_receipt(a.cache_root, row, 'video', done['extraction_sha256']):
                raise RuntimeError('Target video cache receipt failed')
            result.append(dict(row, global_index=pair['index'], raw_truth=pair['text'],
                               reference_id=pair['reference_id'], reference=completed[pair['reference_id']],
                               target_gt_sha256=sha(artifact_path(a.cache_root, row, 'audio')),
                               target_video_sha256=sha(artifact_path(a.cache_root, row, 'video'))))
        if len(result) != 3274 or len({r['reference_id'] for r in result}) != 33:
            raise RuntimeError('Incomplete external-reference coverage')
        atomic_bytes(a.shared_dir/'manifest.jsonl', jsonl_bytes(result))
        publish(a.shared_dir/'metadata.json', dict(status='complete', count=3274, reference_count=33,
                plan_sha256=sha(a.shared_dir/'plan.json'), manifest_sha256=sha(a.shared_dir/'manifest.jsonl'),
                source_cache_manifest_sha256=plan['cache_manifest_sha256'],
                document_sha256=plan['document_sha256'], frozen_portable_sha256=plan['frozen_portable_sha256']))
        print('External-reference preparation complete: 3274 targets / 33 references', flush=True)


def args():
    base = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/projects')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--document', type=Path, default=base/'alignDiT_idea6/zero-shot-Grid/GRID目标_CelebVDub参考_zero-shot复现说明.md')
    p.add_argument('--celeb-root', type=Path, default=base/'data/CelebVDub')
    p.add_argument('--cache-root', type=Path, default=base/'data/GRID_svae1000k_sample_seed666_fp32')
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--semantic-vae-repo', type=Path, default=base/'alignDiT_idea6/papers_codes/Semantic-VAE')
    p.add_argument('--semantic-vae-checkpoint', type=Path, default=base/'alignDiT_idea6/Semantic-VAE/semantic_vae_1000k')
    p.add_argument('--speaker-checkpoint', type=Path, default=base/'data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--plan-only', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    prepare(args())
