#!/usr/bin/env python3
"""Baseline Setting 2 dubbing: supplied true dialogue and cross-utterance audio.

Derived from the audited baseline VTS adapter edce32f78cd1d1e65994c419c6c0070788eb96d14d49bc4b5253100c9f7b7661.
Original EMA loading, prompt mel, sampler, vocoder and cropping are preserved.
No target waveform or joint AV feature is opened during inference.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3] / 'Video-to-Speech-benchmark'
PACKAGE = ROOT.parent / 'my_papers_code/Setting2-Dubbing-test'
DEFAULT_CONFIG = ROOT / 'shared/aligndit_baseline/ckpts/AlignDiT_finetune_hifigan_16k_CelebVDub_char/outputs/2026-06-10/11-08-47/.hydra/config.yaml'
FORBIDDEN_FIELDS = {'gt_audio', 'gt_audio_path', 'gt_text', 'target_audio',
                    'target_audio_path', 'target_text', 'target_text_path'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def safe_id(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 3 or path.parts[0] != 'test':
        raise ValueError('Expected safe test/<folder>/<clip> id: ' + str(value))
    return str(path)


def manifest_rows(args):
    rows, seen = [], set()
    for number, line in enumerate(args.manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        source = json.loads(line)
        if FORBIDDEN_FIELDS.intersection(source):
            raise ValueError('Target GT fields are forbidden in inference manifest at line %d' % number)
        ident, reference_id = safe_id(source['id']), safe_id(source['reference_id'])
        if ident in seen or ((ident == reference_id) != (args.reference_policy == 'same-utterance-reference')):
            raise ValueError('Duplicate target or same-clip reference: ' + ident)
        seen.add(ident)
        if source['speaker_id'] != source['reference_speaker_id']:
            raise ValueError('Speaker IDs differ: ' + ident)
        row = {key: source[key] for key in ('id', 'reference_id', 'reference_text',
                   'speaker_id', 'identity_status', 'protocol')}
        for field in ('reference_audio', 'reference_video_feature', 'video_feature'):
            path = Path(source[field]).expanduser()
            if not path.is_absolute():
                path = args.manifest.parent / path
            path = path.resolve(strict=True)
            if field.endswith('video_feature'):
                relative = path.relative_to(args.manifest.parent).as_posix()
                inventory = json.loads((args.manifest.parent / 'checksums.json').read_text())['files']
                if not relative.startswith('features/video/') or inventory[relative]['sha256'] != digest(path):
                    raise ValueError('Require verified video-only package feature: ' + str(path))
            expected_id = reference_id if field.startswith('reference_') else ident
            if '/'.join(path.with_suffix('').parts[-3:]) != expected_id:
                raise ValueError('Asset path does not match explicit ID: ' + str(path))
            row[field] = str(path)
        row['reference_text'] = row['reference_text'].strip().lower()
        if not row['reference_text']:
            raise ValueError('Empty reference text: ' + ident)
        transcript = args.target_text_dir / (ident + '.txt')
        relative = transcript.relative_to(args.manifest.parent).as_posix()
        inventory = json.loads((args.manifest.parent / 'checksums.json').read_text())['files']
        if inventory[relative]['sha256'] != digest(transcript):
            raise ValueError('GT dialogue checksum mismatch: ' + ident)
        row['target_text_path'] = str(transcript)
        row['target_text'] = transcript.read_text(encoding='utf-8').strip().lower()
        if row['target_text'] is not None and not row['target_text'] and args.reference_policy != 'same-utterance-reference':
            raise ValueError('Empty supplied GT dialogue: ' + ident)
        rows.append(row)
    if args.ids:
        requested = set(args.ids)
        absent = requested - seen
        if absent:
            raise ValueError('Unknown selected IDs: ' + ', '.join(sorted(absent)))
        rows = [row for row in rows if row['id'] in requested]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No selected manifest rows')
    return rows


def install_import_paths(repo):
    sys.path.insert(0, str(repo / 'src'))
    sys.path.insert(1, '/zjw524/projects/data/av_hubert/fairseq/fairseq')


def load_components(args, device):
    import torch
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from aligndit.model import CFM_VT
    from aligndit.model.modules import MelSpec_tacotron
    from aligndit.script.eval.utils import load_vocoder
    from f5_tts.model.utils import get_tokenizer

    cfg = OmegaConf.load(args.config)
    if cfg.model.tokenizer != 'char' or cfg.model.mel_spec.mel_spec_type != 'hifigan_16k':
        raise ValueError('This adapter supports the audited char / hifigan_16k configuration')
    vocab, count = get_tokenizer(str(args.vocab), 'custom')
    if vocab.get(' ') != 0:
        raise ValueError('Expected space at vocabulary index 0')
    mel = MelSpec_tacotron(**cfg.model.mel_spec)
    model = CFM_VT(
        transformer=get_class('aligndit.model.' + cfg.model.backbone)(
            **cfg.model.arch, text_num_embeds=count, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_module=MelSpec_tacotron(**cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in cfg.model.mel_spec.items() if k != 'mel_spec_type'},
        odeint_kwargs={'method': args.ode_method}, vocab_char_map=vocab,
    )
    # Keep optimizer tensors on CPU and load exactly the upstream EMA weights.
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True, mmap=True)
    weights = {k.removeprefix('ema_model.'): v for k, v in checkpoint['ema_model_state_dict'].items()
               if k not in ('initted', 'step')}
    for key in ('mel_spec.mel_stft.mel_scale.fb', 'mel_spec.mel_stft.spectrogram.window'):
        weights.pop(key, None)
    model.load_state_dict(weights, strict=True)
    update = checkpoint.get('update')
    if update != args.expected_update:
        raise ValueError('Checkpoint update %r does not match requested update %d' % (update, args.expected_update))
    del weights, checkpoint
    model = model.eval().to(device=device, dtype=torch.float32)
    vocoder = load_vocoder('hifigan_16k', is_local=True, local_path=str(args.vocoder), device=device)
    return model, vocoder, mel, vocab, cfg, update


def prepare_prompt(row, mel, cfg, ratio):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio
    audio, sample_rate = torchaudio.load(row['reference_audio'])
    audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != cfg.model.mel_spec.target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, cfg.model.mel_spec.target_sample_rate)
    rms = float(audio.square().mean().sqrt())
    if not torch.isfinite(audio).all() or rms <= 0 or audio.shape[-1] <= 5000:
        raise ValueError('Invalid reference waveform: ' + row['id'])
    # All original benchmark references had precomputed, unnormalized Tacotron
    # mels. Recompute the same representation directly from the explicit reference
    # waveform; never derive any path to target mel/audio.
    reference_mel = mel(audio).float()  # upstream MelSpec_tacotron returns [T, 80]
    video = np.load(row['video_feature'], allow_pickle=False)
    ref_video = np.load(row['reference_video_feature'], allow_pickle=False, mmap_mode='r')
    for feature in (video, ref_video):
        if feature.ndim != 2 or feature.shape[1] != 1024 or len(feature) == 0 or not np.isfinite(feature).all():
            raise ValueError('Invalid video-only feature shape/content: ' + row['id'])
    ref_len, target_len = len(ref_video) * ratio, len(video) * ratio
    total_len = ref_len + target_len
    if total_len > 4096:
        raise ValueError('Prompt exceeds original model max_duration=4096: ' + row['id'])
    if reference_mel.ndim != 2 or reference_mel.shape[1] != 80:
        raise ValueError('Unexpected reference mel layout: ' + str(reference_mel.shape))
    reference_mel = reference_mel[:ref_len]
    if len(reference_mel) < ref_len:
        reference_mel = F.pad(reference_mel.T.unsqueeze(0), (0, ref_len - len(reference_mel)), mode='replicate')[0].T
    full_video = torch.cat((torch.zeros(len(ref_video), 1024), torch.from_numpy(video).float()), dim=0)
    # Preserve the two separating spaces used by original get_inference_prompt_vt
    # for ASCII references and its leading-space target transcript convention.
    reference_text = row['reference_text']
    if len(reference_text[-1].encode('utf-8')) == 1:
        reference_text += ' '
    text = reference_text + ' ' + row['target_text'] if row['target_text'] is not None else None
    if text is not None and len(text) + 1 > total_len:
        raise ValueError('Text would override video duration in upstream sampler: ' + row['id'])
    return reference_mel.unsqueeze(0), full_video.unsqueeze(0), ref_len, target_len, rms, text


@contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        print('Waiting for GPU lock: ' + str(path), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('Acquired GPU lock', flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(args, rows, device):
    import torch
    import torchaudio
    model, vocoder, mel, vocab, cfg, update = load_components(args, device)
    print('Strict EMA checkpoint load passed, update=%s, vocab=%d' % (update, len(vocab)), flush=True)
    if args.check_model:
        checks = []
        for row in rows:
            cond, video, ref_len, target_len, rms, text = prepare_prompt(row, mel, cfg, model.audio_video_ratio)
            checks.append({'id': row['id'], 'reference_mel_frames': ref_len, 'target_mel_frames': target_len,
                           'reference_rms': rms, 'target_text_available': text is not None})
        report = {'mode': 'CPU strict checkpoint/vocoder load and explicit-input preprocessing; no synthesis',
                  'checkpoint_update': update, 'samples': len(checks), 'checks': checks}
        write_json(args.output_dir / 'model_check.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2), flush=True)
        return
    run_provenance = {
        'method': f'AlignDiT_GTText_Dubbing_{args.expected_update // 1000}k',
        'training': 'User-designated CelebVDub finetune checkpoint; original config records LibriSpeech no-text initialization',
        'training_provenance_limit': 'Training config and colocated logs; checkpoint does not embed training dataset metadata',
        'checkpoint_update': update,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest(args.checkpoint),
        'config_sha256': digest(args.config), 'vocab_sha256': digest(args.vocab),
        'vocoder_sha256': digest(args.vocoder), 'manifest_sha256': digest(args.manifest),
        'adapter_sha256': digest(__file__), 'baseline_repo': str(args.repo),
        'source_sha256': {str(p.relative_to(args.repo)): digest(p) for p in
                          sorted((args.repo / 'src/aligndit/model').rglob('*.py'))},
        'seed': args.seed, 'steps': args.steps, 'ode_method': args.ode_method, 'sway': args.sway,
        'cfg_t': args.cfg_t, 'cfg_v': args.cfg_v, 'dtype': 'float32', 'use_ema': True,
        'duration_source': 'target video-only feature frames * 4 mel frames * 160 samples',
        'reference_mel': 'recomputed from raw explicit reference waveform, matching upstream mel cache',
        'reference_video': 'only frame count used; all reference visual conditioning zeroed',
        'reference_policy': args.reference_policy,
        'task': 'Setting2-GTText-Dubbing',
        'target_gt_audio_or_text_read': True, 'target_gt_text_read': True,
        'target_gt_audio_read': False, 'target_acoustic_cache_read': False, 'vsr_used': False,
        'target_gt_text_as_target_condition': True,
        'oracle_reference': args.reference_policy == 'same-utterance-reference',
        'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
    }
    write_json(args.output_dir / 'run_config.json', run_provenance)
    errors = []
    complete = 0
    for index, row in enumerate(rows, 1):
        target = args.output_dir / (row['id'] + '.wav')
        meta_path = args.output_dir / (row['id'] + '.json')
        started = time.monotonic()
        try:
            sample = {'run': run_provenance, 'id': row['id'], 'reference_id': row['reference_id'],
                      'target_text': row['target_text'], 'reference_text': row['reference_text'],
                      'input_sha256': {k: digest(row[k]) for k in ('reference_audio', 'reference_video_feature',
                                                                  'video_feature', 'target_text_path')}}
            if target.is_file() and meta_path.is_file():
                previous = json.loads(meta_path.read_text())
                if previous.get('inputs') != sample or previous.get('audio_sha256') != digest(target):
                    raise ValueError('Existing output provenance differs; use a new output directory')
                print('[%d/%d] verified existing %s' % (index, len(rows), row['id']), flush=True)
                complete += 1
                continue
            cond, video, ref_len, target_len, rms, text = prepare_prompt(row, mel, cfg, model.audio_video_ratio)
            oov = sorted(set(text) - set(vocab))
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=cond.to(device), text=[text], duration=torch.tensor([ref_len + target_len], device=device),
                    video=video.to(device), lens=torch.tensor([ref_len], device=device), steps=args.steps,
                    cfg_strength=args.cfg_t, cfg_strength_v=args.cfg_v, sway_sampling_coef=args.sway,
                    seed=args.seed, no_ref_audio=False, ignore_modality=None)
                generated = generated[:, ref_len:ref_len + target_len, :].transpose(1, 2).float()
                wave = vocoder(generated).squeeze(1).cpu()
            if rms < 0.1:
                wave = wave * (rms / 0.1)
            expected_samples = target_len * cfg.model.mel_spec.hop_length
            if wave.shape != (1, expected_samples) or not torch.isfinite(wave).all() or not wave.square().mean() > 0:
                raise ValueError('Invalid generated waveform shape or values: ' + str(wave.shape))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp.wav')
            torchaudio.save(str(temporary), wave, cfg.model.mel_spec.target_sample_rate)
            temporary.replace(target)
            write_json(meta_path, {'inputs': sample, 'audio_sha256': digest(target),
                                  'seconds': expected_samples / cfg.model.mel_spec.target_sample_rate,
                                  'generation_seconds': time.monotonic() - started, 'oov_characters': oov,
                                  'peak': float(wave.abs().max()), 'rms': float(wave.square().mean().sqrt())})
            complete += 1
            print('[%d/%d] generated %s in %.1fs' % (index, len(rows), row['id'], time.monotonic() - started), flush=True)
        except Exception as exc:
            errors.append({'id': row['id'], 'error': repr(exc)})
            print('[%d/%d] FAILED %s: %r' % (index, len(rows), row['id'], exc), flush=True)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        write_json(args.output_dir / 'progress.json', {'requested': len(rows), 'complete': complete, 'failures': errors})
    write_json(args.output_dir / 'progress.json', {'requested': len(rows), 'complete': complete, 'failures': errors})
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-policy', choices=['cross-reference'], default='cross-reference')
    parser.add_argument('--manifest', type=Path, default=PACKAGE / 'aligndit_inference.jsonl')
    parser.add_argument('--repo', type=Path, default=ROOT / 'shared/aligndit_baseline')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'shared/AlignDiT_CelebVDub_finetune_100k.pt')
    parser.add_argument('--expected-update', type=int, default=100000)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--vocab', type=Path, default=Path('/zjw524/projects/data/CelebVDub_char/vocab.txt'))
    parser.add_argument('--vocoder', type=Path, default=ROOT / 'shared/hifigan_16k_LRS3/g_01000000')
    parser.add_argument('--target-text-dir', type=Path, default=PACKAGE / 'text')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/setting2_gttext_dubbing/AlignDiT_100k')
    parser.add_argument('--gpu-lock', type=Path, default=Path('/tmp/alignDiT_idea6_vts_gpu0.lock'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--ode-method', default='euler')
    parser.add_argument('--sway', type=float, default=-1)
    parser.add_argument('--cfg-t', type=float, default=5)
    parser.add_argument('--cfg-v', type=float, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--id', dest='ids', action='append')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and report GT dialogue availability, without loading torch')
    parser.add_argument('--check-model', action='store_true', help='Strict CPU load and preprocessing only; allows missing GT dialogue')
    args = parser.parse_args()
    if args.limit < 0 or args.steps < 1:
        parser.error('--limit must be nonnegative and --steps must be positive')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    for path in (args.manifest, args.checkpoint, args.config, args.vocab, args.vocoder):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.repo / 'src/aligndit/model/cfm_vt.py').is_file():
        raise FileNotFoundError('Expected baseline repo at ' + str(args.repo))
    rows = manifest_rows(args)
    missing = [row['id'] for row in rows if row['target_text'] is None]
    report = {'selected': len(rows), 'target_text_available': len(rows) - len(missing), 'missing_target_text_ids': missing,
              'repo': str(args.repo), 'checkpoint': str(args.checkpoint), 'config': str(args.config),
              'target_input': 'video-only features and supplied true target dialogue',
              'reference_input': args.reference_policy + ' audio and transcript',
              'reference_policy': args.reference_policy,
        'task': 'Setting2-GTText-Dubbing',
        'target_gt_audio_or_text_read': True, 'target_gt_text_read': True,
        'target_gt_audio_read': False, 'target_acoustic_cache_read': False, 'vsr_used': False,
        'target_gt_text_as_target_condition': True,
        'oracle_reference': args.reference_policy == 'same-utterance-reference', 'output_dir': str(args.output_dir)}
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return 2 if missing else 0
    if missing and not args.check_model:
        raise SystemExit('GT dialogue incomplete; no synthesis started')
    install_import_paths(args.repo)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if args.check_model:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        run(args, rows, 'cpu')
    else:
        with gpu_lock(args.gpu_lock):
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            run(args, rows, 'cuda:0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
