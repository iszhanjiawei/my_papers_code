"""Model-isolated AlignDiT preprocessing; never imports StyleDubber/ProDubber.

The 660/66 identities come from the frozen v2 manifest, not a new selection.
Each model calls its own unmodified crop and visual extractor. Runtime artifacts
live outside the source tree, in separate baseline/repa child runs.
"""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np
import torch

from protocol import BASELINE, DATA, PROJECT, REPA, DEFAULT_RUN, rows, sha, save_json, load_module

NATIVE_RUN = DATA / 'evaluations/grid660_celeb66_seed0_aligndit_native_v3'
CACHE_TREE = PROJECT / 'my_papers_code/AlignDiT_mmdit_c2_semantic_vae'
MEAN_FACE_URL = ('https://raw.githubusercontent.com/mpc001/'
                 'Lipreading_using_Temporal_Convolutional_Networks/'
                 'master/preprocessing/20words_mean_face.npy')
MEAN_FACE_SHA = 'dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422'
VISUAL_FILES = ('script/misc/crop_mouth_celebvdub.py',
                'script/misc/extract_avhubert_from_only_video.py',
                'script/misc/avhubert_utils.py')


def tree_for(method):
    if method not in ('baseline', 'repa'):
        raise ValueError('This frontend is restricted to Baseline and REPA')
    return BASELINE if method == 'baseline' else REPA


def save_npy(path, value):
    path = Path(path)
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite feature')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        np.save(f, value, allow_pickle=False)
    path.with_suffix('.tmp').replace(path)


def verified_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError('Changed immutable artifact: ' + str(destination))
    else:
        shutil.copyfile(source, destination)
    return sha(destination)


def initialize(source, run):
    import torchaudio
    source_rows = rows(source, evaluation=True)
    if run.exists():
        raise FileExistsError('Use a new run directory: ' + str(run))
    specroot = DATA / 'CelebVDub_svae1000k_sample_seed666_fp32'
    manifest_path = specroot / 'manifests/test.jsonl'
    manifest = {r['utterance_key']: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    latent_index = {r['utterance_key']: r for r in map(json.loads, (specroot/'state/latents/index.jsonl').read_text().splitlines())}
    blob = urllib.request.urlopen(MEAN_FACE_URL, timeout=30).read()
    if hashlib.sha256(blob).hexdigest() != MEAN_FACE_SHA or np.load(io.BytesIO(blob)).shape != (68, 2):
        raise ValueError('Canonical mean-face template failed validation')
    run.mkdir(parents=True)
    mean_face = run/'native_resources/20words_mean_face.npy'
    mean_face.parent.mkdir(parents=True)
    mean_face.write_bytes(blob)
    ref_records = []
    unique = {r['ref_id']: r for r in source_rows}
    for ident, row in unique.items():
        key = ident.replace('celebvdub/', 'celebvdub/test/', 1)
        native = manifest[key]
        latent = specroot / native['latent_relative_path']
        speaker = DATA/'CelebVDub/campplus_spk_emb_zh_en_16k'/Path(native['audio_relative_path']).with_suffix('.npy')
        if sha(latent) != latent_index[key]['sha256']:
            raise ValueError('Native latent cache hash mismatch: ' + key)
        if (np.load(latent).shape != (row['ref_frames'], 64) or
                np.load(speaker).shape != (192,) or native['text'] != row['ref_text'] or
                native['original_num_samples'] != row['ref_num_samples']):
            raise ValueError('Native reference metadata mismatch: ' + key)
        wav, sr = torchaudio.load(row['ref_source_audio'])
        wav = wav.float().mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        frozen, frozen_sr = torchaudio.load(row['ref_audio'])
        if frozen_sr != 16000 or frozen.shape != wav.shape:
            raise ValueError('Frozen reference waveform shape differs from native decoding')
        delta = float((wav-frozen).abs().max())
        if delta > 1/32768 + 1e-7:
            raise ValueError('Frozen reference differs from native decode by more than PCM16 precision')
        ref_records.append({'ref_id': ident, 'utterance_key': key,
            'posterior_seed': native['posterior_seed'], 'latent_source': str(latent),
            'latent_sha256': sha(latent), 'speaker_source': str(speaker), 'speaker_sha256': sha(speaker),
            'audio_sha256': sha(row['ref_audio']), 'native_decode_max_abs_difference': delta,
            'old_posterior_latent_equal': np.array_equal(np.load(latent), np.load(row['ref_latent_path']))})
    for method in ('baseline', 'repa'):
        child = run / method
        child.mkdir()
        inference, evaluation = [], []
        for old in source_rows:
            row = {k: v for k, v in old.items() if k not in (
                'target_gt_audio', 'source_lip_feature', 'source_face_feature', 'lip_feature', 'face_feature')}
            for key in ('mouth_video', 'video_25hz_path', 'target_video_path', 'ref_audio', 'ref_latent_path', 'ref_speaker_path'):
                row[key] = str(child/Path(old[key]).relative_to(source))
            verified_copy(old['ref_audio'], row['ref_audio'])
            if sha(row['ref_audio']) != row['ref_audio_sha256']:
                raise ValueError('Changed reference audio')
            key = row['ref_id'].replace('celebvdub/', 'celebvdub/test/', 1)
            native = manifest[key]
            if method == 'repa':
                verified_copy(specroot/native['latent_relative_path'], row['ref_latent_path'])
                verified_copy(DATA/'CelebVDub/campplus_spk_emb_zh_en_16k'/Path(native['audio_relative_path']).with_suffix('.npy'), row['ref_speaker_path'])
            row['frontend_method'] = method
            row['frontend_protocol'] = 'model_isolated_aligndit_native_v1'
            inference.append(row)
            gt = child/'assets/gt'/(row['target_id']+'.wav')
            verified_copy(old['target_gt_audio'], gt)
            evaluation.append({**row, 'target_gt_audio': str(gt)})
        for name, items in [('inference', inference), ('pairs', evaluation)]:
            (child/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in items))
        selection = json.loads((source/'selection.json').read_text())
        selection.update(pairs_sha256=sha(child/'pairs.jsonl'), inference_sha256=sha(child/'inference.jsonl'),
            frozen_source_run=str(source), frozen_source_pairs_sha256=sha(source/'pairs.jsonl'),
            frontend_method=method, frontend_protocol='model_isolated_aligndit_native_v1')
        save_json(child/'selection.json', selection)
        tree = tree_for(method)
        hashes = {str(tree/'src/aligndit'/p): sha(tree/'src/aligndit'/p) for p in VISUAL_FILES}
        if method == 'repa':
            interp = CACHE_TREE/'src/aligndit/script/misc/cache_celebvdub_video_40hz.py'
            hashes[str(interp)] = sha(interp)
        save_json(child/'native_frontend.json', {
            'method': method, 'model_tree': str(tree), 'source_sha256': hashes,
            'mean_face': str(mean_face), 'mean_face_sha256': MEAN_FACE_SHA,
            'mean_face_source': MEAN_FACE_URL,
            'historical_mean_face_file_present': False,
            'mean_face_provenance_note': 'Native default file is absent; canonical 20words asset restored from its upstream author, not a different model frontend. Historical file hash cannot be independently verified.',
            'crop': 'unmodified native process_video, 5 stable landmarks, queue=12, cubic/replicate warp, grayscale after warp, OpenCV mp4v',
            'visual': 'own native AVHubertFeatureReader; audio=None, mask=False, checkpoint-defined center crop and normalization',
            'avhubert_checkpoint': str(PROJECT.parent.parent/'alignDiT_pretrain_models/large_vox_iter5.pt'),
            'avhubert_checkpoint_sha256': sha(PROJECT.parent.parent/'alignDiT_pretrain_models/large_vox_iter5.pt'),
            'video_rate': 25 if method == 'baseline' else 40,
            'references': ref_records, 'native_reference_manifest_sha256': sha(manifest_path),
            'runtime_script_sha256': sha(__file__),
            'reference_processing': 'native TacotronSTFT, native inference RMS rule' if method == 'baseline' else 'verified original fp32 Semantic-VAE posterior caches and CAM++ caches; training normalization',
            'target_acoustic_inputs_used': False})
        rows(child)
    save_json(run/'native_run.json', {'source': str(source), 'source_pairs_sha256': sha(source/'pairs.jsonl'),
        'target_count': 660, 'reference_count': 66, 'methods': ['baseline','repa'],
        'pairing_unchanged': True, 'mean_face_sha256': MEAN_FACE_SHA})
    print('Initialized two isolated native runs, identical 660/66 pairing', flush=True)


def verify_audit(run, method):
    audit = json.loads((run/'native_frontend.json').read_text())
    if audit['method'] != method or audit['model_tree'] != str(tree_for(method)):
        raise ValueError('Wrong model frontend')
    for path, expected in audit['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Native preprocessing source changed: ' + path)
    if sha(audit['mean_face']) != audit['mean_face_sha256']:
        raise ValueError('Changed canonical template')
    return audit


def visual(run, method, shard, nshard):
    audit = verify_audit(run, method)
    tree = tree_for(method)
    sys.path.insert(0, str(tree/'src'))
    sys.path.insert(0, str(DATA/'av_hubert/fairseq/fairseq'))
    sys.path.insert(0, str(DATA/'av_hubert/avhubert/avhubert'))
    import fairseq
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=str(DATA/'av_hubert/avhubert/avhubert')))
    from aligndit.script.misc.extract_avhubert_from_only_video import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    crop = load_module('own_native_mouth_crop', tree/'src/aligndit/script/misc/crop_mouth_celebvdub.py')
    fa_kwargs = {'flip_input': False, 'device': 'cuda'}
    if 'compile' in inspect.signature(crop.face_alignment.FaceAlignment).parameters:
        fa_kwargs['compile'] = False  # execution-only; same FAN/S3FD weights
    fa = crop.face_alignment.FaceAlignment(crop.face_alignment.LandmarksType.TWO_D, **fa_kwargs)
    reader = AVHubertFeatureReader(audit['avhubert_checkpoint'], custom_utils=avhubert_utils)
    interpolate = None
    if method == 'repa':
        # This is the actual source bound by the REPA training video cache spec.
        sys.path.insert(0, str(CACHE_TREE/'src/aligndit/script/misc'))
        load_module('aligndit.script.misc.svae_cache_utils', CACHE_TREE/'src/aligndit/script/misc/svae_cache_utils.py')
        interp_module = load_module('training_video40_cache', CACHE_TREE/'src/aligndit/script/misc/cache_celebvdub_video_40hz.py')
        interpolate = interp_module.interpolate_video
    mean_face = np.load(audit['mean_face'], allow_pickle=False)
    items = rows(run)[shard::nshard]
    records = []
    for i, row in enumerate(items, 1):
        if row['frontend_method'] != method or sha(row['source_video']) != row['source_video_sha256']:
            raise ValueError('Changed target/native frontend identity')
        record_path = run/'assets/records'/(row['target_id']+'.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            for k, value in record['artifact_sha256'].items():
                if sha(record['artifact_paths'][k]) != value:
                    raise ValueError('Changed resumed native feature')
        else:
            mouth = Path(row['mouth_video'])
            mouth.parent.mkdir(parents=True, exist_ok=True)
            if mouth.exists():
                raise FileExistsError('Unverified interrupted mouth artifact: '+str(mouth))
            if not crop.process_video(fa, row['source_video'], str(mouth), mean_face, 'cuda'):
                raise ValueError('Native mouth cropping failed')
            decoded = avhubert_utils.load_video(str(mouth))
            if decoded.shape != (row['native_frames'],96,96):
                raise ValueError('Native mouth frame count changed')
            with torch.inference_mode():
                feat = reader.get_feats((str(mouth), None)).cpu().float().numpy()
            if feat.shape != (75,1024):
                raise ValueError('Invalid native AV-HuBERT shape')
            save_npy(row['video_25hz_path'], feat)
            paths = {k:row[k] for k in ('mouth_video','video_25hz_path')}
            if interpolate is not None:
                aligned = interpolate(feat, row['target_frames'], torch.device('cuda:0'))
                save_npy(row['target_video_path'], aligned)
                paths['target_video_path'] = row['target_video_path']
            record = {'target_id': row['target_id'], 'frontend_method': method, 'frames':len(decoded),
                'artifact_paths':paths, 'artifact_sha256':{k:sha(v) for k,v in paths.items()},
                'native_crop_sha256':sha(crop.__file__), 'audio_used_for_visual_condition':False}
            save_json(record_path, record)
        records.append(record)
        print(f'Native visual {method} shard {shard}: {i}/{len(items)} {row["target_id"]}', flush=True)
        if i % 50 == 0:
            torch.cuda.empty_cache()
    save_json(run/f'assets/native_shard{shard}.json', {'count':len(records),'nshard':nshard,'method':method,'records':records})


def infer(run, method, limit):
    audit = verify_audit(run, method)
    from infer import aligndit, loop
    items = rows(run)
    if limit:
        items = items[:limit]
    for row in items:
        if sha(row['ref_audio']) != row['ref_audio_sha256']:
            raise ValueError('Changed reference waveform')
        if method == 'repa':
            reference = next(r for r in audit['references'] if r['ref_id'] == row['ref_id'])
            if (sha(row['ref_latent_path']) != reference['latent_sha256'] or
                    sha(row['ref_speaker_path']) != reference['speaker_sha256']):
                raise ValueError('Reference cache differs from native training cache')
        record = json.loads((run/'assets/records'/(row['target_id']+'.json')).read_text())
        if record['frontend_method'] != method:
            raise ValueError('Cross-model visual cache is forbidden')
        for k, value in record['artifact_sha256'].items():
            if sha(record['artifact_paths'][k]) != value:
                raise ValueError('Changed native visual artifact')
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('high')
    args = argparse.Namespace(run=run, method=method, limit=limit)
    loop(args, items, aligndit(args, items))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=NATIVE_RUN)
    p.add_argument('--source', type=Path, default=DEFAULT_RUN)
    p.add_argument('--stage', choices=['initialize','visual','infer'], required=True)
    p.add_argument('--method', choices=['baseline','repa'], default='baseline')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, choices=[0,1,33], default=0)
    a = p.parse_args()
    if a.stage == 'initialize':
        initialize(a.source, a.run)
    elif a.stage == 'visual':
        visual(a.run, a.method, a.shard, a.nshard)
    else:
        infer(a.run, a.method, a.limit)
