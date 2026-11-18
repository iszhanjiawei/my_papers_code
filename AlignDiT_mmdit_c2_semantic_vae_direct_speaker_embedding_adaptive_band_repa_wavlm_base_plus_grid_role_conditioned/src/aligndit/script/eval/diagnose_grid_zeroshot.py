"""Matched-condition GRID speaker diagnosis using frozen external references.

prepare freezes a balanced subset and copies hash-verified prior target-only WAVs.
infer changes one or two conditioning inputs without reading target audio, latent,
or CAM++ caches. reconstruct is an independent reference autoencoder control.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import wave
from pathlib import Path

from aligndit.script.eval.grid_s1_common import ensure_contract, identity, read, sha
from aligndit.script.eval.grid_zeroshot_common import check_output, jsonl_bytes, load_prepared

PROTOCOL = 'grid_external_reference_matched_speaker_diagnosis_v1'
VARIANTS = ('target_only', 'ref_text', 'cam_zero', 'cam_swapped', 'prompt_zero',
            'reference_disabled', 'video_disabled', 'ref_text_video_disabled')
ALL_VARIANTS = (*VARIANTS, 'baseline_regenerated')
GENERATION = dict(seed=0, nfe=32, ode='euler', epss=True, sway=-1, cfg_text=5,
                  cfg_video=2, batch_size=1, dtype='float32', matmul_precision='high')


def frozen_bytes(path, data):
    """Publish once under a lock; refuse an unchecked replacement on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError(f'Existing artifact differs: {path}')
        else:
            temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
            try:
                with temporary.open('xb') as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def select_rows(rows, per_speaker):
    if per_speaker < 1:
        raise ValueError('--per-speaker must be positive')
    selected, counts, assignments = [], collections.Counter(), {}
    for row in rows:
        speaker = Path(identity(row)).parts[1]
        if assignments.setdefault(speaker, row['reference_id']) != row['reference_id']:
            raise RuntimeError('One GRID speaker has multiple frozen references')
        if counts[speaker] < per_speaker:
            selected.append(row)
            counts[speaker] += 1
    if len(assignments) != 33 or len(set(assignments.values())) != 33:
        raise RuntimeError('Expected 33 one-to-one GRID/external reference assignments')
    if set(counts.values()) != {per_speaker}:
        raise ValueError('Requested more samples per speaker than the manifest contains')
    speakers = sorted(assignments)
    swapped = {s: assignments[speakers[(i+1) % len(speakers)]] for i, s in enumerate(speakers)}
    return selected, swapped


def load_selection(output_root):
    shared = Path(output_root)/'shared'
    protocol = read(shared/'protocol.json')
    if protocol['protocol'] != PROTOCOL or sha(shared/'manifest.jsonl') != protocol['manifest_sha256']:
        raise RuntimeError('Diagnosis selection changed')
    rows = [json.loads(line) for line in (shared/'manifest.jsonl').read_text().splitlines()]
    if len(rows) != protocol['count'] or len({identity(r) for r in rows}) != len(rows):
        raise RuntimeError('Incomplete or duplicate selected manifest')
    return rows, protocol


def source_snapshot(config):
    project = Path(__file__).parents[4]
    files = set((project/'src/aligndit/model').rglob('*.py'))
    files.update((project/'src/aligndit/config').rglob('*.yaml'))
    files.update(Path(__file__).with_name(name) for name in (
        'diagnose_grid_zeroshot.py', 'grid_s1_common.py', 'grid_zeroshot_common.py',
        'infer_celebvdub_grid_reference.py', 'infer_celebvdub_semantic_vae_s1.py',
        'semantic_vae_decoder.py', 'prepare_grid_reference.py'))
    files.add(config)
    return [dict(path=str(p.resolve()), sha256=sha(p)) for p in sorted(files)]


def validate_sources(protocol):
    for item in protocol['source_files']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError(f"Generation source changed: {item['path']}")
    if sha(protocol['checkpoint']['path']) != protocol['checkpoint']['sha256']:
        raise RuntimeError('Checkpoint changed')


def validate_reference(ref):
    from aligndit.script.eval.prepare_grid_reference import validate_cached_record
    for path_key, hash_key in (('ref_source_audio', 'ref_source_audio_sha256'),
                              ('ref_transcript_path', 'ref_transcript_sha256')):
        if sha(ref[path_key]) != ref[hash_key]:
            raise RuntimeError(f'Frozen reference artifact changed: {ref[path_key]}')
    validate_cached_record(ref, {k: v for k, v in ref.items() if not k.endswith('_sha256')})


def condition_record(row, variant, selection, references):
    from aligndit.script.eval.infer_celebvdub_grid_reference import reference_target_text
    if variant not in ALL_VARIANTS:
        raise ValueError(variant)
    ref_text = variant in ('ref_text', 'ref_text_video_disabled')
    speaker_id = selection['cam_swapped_reference_ids'][Path(identity(row)).parts[1]] if variant == 'cam_swapped' else row['reference_id']
    speaker_ref = references[speaker_id]
    return dict(id=identity(row), variant=variant, reference_id=row['reference_id'],
        reference_source_audio_sha256=row['reference']['ref_source_audio_sha256'],
        reference_latent_sha256=row['reference']['ref_latent_sha256'],
        reference_speaker_sha256=row['reference']['ref_speaker_sha256'],
        speaker_condition_reference_id=None if variant in ('cam_zero', 'reference_disabled') else speaker_id,
        speaker_condition_sha256=None if variant in ('cam_zero', 'reference_disabled') else speaker_ref['ref_speaker_sha256'],
        speaker_condition='model_speaker_dropout' if variant == 'reference_disabled' else 'zero_vector' if variant == 'cam_zero' else 'cyclic_other_reference' if variant == 'cam_swapped' else 'original_reference',
        prompt_condition='zero_normalized_latent' if variant in ('prompt_zero', 'reference_disabled') else 'original_reference',
        no_ref_audio=variant == 'reference_disabled',
        ignore_modality='video' if variant in ('video_disabled', 'ref_text_video_disabled') else None,
        video_condition='model_learned_video_dropout' if variant in ('video_disabled', 'ref_text_video_disabled') else 'target_video',
        target_video_sha256=row['target_video_sha256'],
        model_text=reference_target_text(row['reference']['ref_text'], row['text']) if ref_text else row['text'],
        text_conditioning='reference-target' if ref_text else 'target-only',
        prompt_frames=row['reference']['ref_frames'], target_frames=row['latent_frames'],
        sampler_frames=row['reference']['ref_frames']+row['latent_frames'],
        samples=row['original_num_samples'], seed=0,
        target_audio_read_for_generation=False, target_latent_or_speaker_read_for_generation=False)


def validate_wav(path, samples):
    with wave.open(str(path)) as handle:
        actual = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes())
    if actual != (16000, 1, 2, samples):
        raise RuntimeError(f'Incorrect output WAV geometry: {path}')


def check_diagnostic_output(root, step, variant, row, selection):
    directory = Path(root)/f'ema_{step}'/variant
    rows, _ = load_selection(root)
    references = {r['reference_id']: r['reference'] for r in rows}
    expected = condition_record(row, variant, selection, references)
    receipt = read(directory/'receipts'/(identity(row)+'.json'))
    expected['protocol_sha256'] = sha(directory/'protocol.json')
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Output conditioning receipt changed: {variant}/{identity(row)}')
    wav = directory/row['audio_relative_path']
    if receipt['sha256'] != sha(wav):
        raise RuntimeError(f'Output WAV changed: {wav}')
    validate_wav(wav, row['original_num_samples'])
    return receipt


def prepare(a):
    from omegaconf import OmegaConf
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import load_composed_config
    from aligndit.model.grid_dataset import NORMALIZATION_SHA256, VOCAB_SHA256
    prior_rows, metadata = load_prepared(a.prior_root/'shared')
    rows, swapped = select_rows(prior_rows, a.per_speaker)
    shared = a.output_root/'shared'
    blob = jsonl_bytes(rows)
    selection = dict(protocol=PROTOCOL, count=len(rows), per_speaker=a.per_speaker,
        selection='first N rows per GRID speaker in frozen original manifest order',
        prior_root=str(a.prior_root.resolve()), prior_manifest_sha256=metadata['manifest_sha256'],
        manifest_sha256=hashlib.sha256(blob).hexdigest(),
        cam_swapped_reference_ids=swapped,
        cam_swap_order='lexically sorted GRID speaker IDs, cyclic successor reference',
        selected_ids=[identity(r) for r in rows])
    ensure_contract(shared/'protocol.json', selection)
    frozen_bytes(shared/'manifest.jsonl', blob)
    prior_dir = a.prior_root/f'ema_{a.step}'
    prior_protocol = read(prior_dir/'protocol.json')
    checkpoint = a.checkpoint or Path(prior_protocol['checkpoint']['path'])
    if prior_protocol['checkpoint']['update'] != a.step or sha(checkpoint) != prior_protocol['checkpoint']['sha256']:
        raise RuntimeError('Checkpoint differs from the prior evaluated model')
    config = load_composed_config(a.config.resolve())
    resolved_config = OmegaConf.to_container(config, resolve=True)
    if resolved_config != prior_protocol['config']:
        raise RuntimeError('Composed model configuration differs from the prior run')
    if sha(config.datasets.normalization_path) != NORMALIZATION_SHA256 or sha(config.datasets.vocab_path) != VOCAB_SHA256:
        raise RuntimeError('Normalization or vocabulary differs from GRID training')
    references = {r['reference_id']: r['reference'] for r in rows}
    for ref in references.values():
        validate_reference(ref)
    for row in rows:
        # Guard against text-induced duration extension: every intervention must
        # have identical P+T, hence identical seed-0 initial target noise.
        combined = condition_record(row, 'ref_text', selection, references)['model_text']
        total = row['reference']['ref_frames']+row['latent_frames']
        if not 0 < row['reference']['ref_frames'] < total <= 4096 or len(combined)+1 > total:
            raise RuntimeError('Reference text would change sampling duration')
        if sha(a.cache_root/row['video_40hz_relative_path']) != row['target_video_sha256']:
            raise RuntimeError('Target video cache changed')
    contract = dict(protocol=PROTOCOL, selection_sha256=sha(shared/'protocol.json'),
        manifest_sha256=selection['manifest_sha256'], requested_count=len(rows),
        prior_protocol_sha256=sha(prior_dir/'protocol.json'),
        checkpoint=dict(path=str(checkpoint.resolve()), sha256=sha(checkpoint), update=a.step, weights='EMA'),
        config_path=str(a.config.resolve()), config_file_sha256=sha(a.config), config=resolved_config,
        source_files=source_snapshot(a.config.resolve()), generation=GENERATION,
        cache_root=str(a.cache_root.resolve()), cache_spec_sha256=sha(a.cache_root/'state/latents/spec.json'),
        semantic_vae_repo=str(a.semantic_vae_repo.resolve()), semantic_vae_checkpoint=str(a.semantic_vae_checkpoint.resolve()),
        target_audio_read_for_generation=False, target_latent_or_speaker_read_for_generation=False,
        crop='all variants preserve original P+T; decode only [P:P+T]; exact original target samples',
        target_only='copied from checked prior WAV and receipt',
        baseline_regenerated='identical baseline regenerated; reports byte identity versus checked prior output',
        video_disabled='sample(ignore_modality=video): learned video null; CFG=6*tts-5*null, mathematically baseline with full=tts',
        prompt_zero='zero normalized reference latent only; retain prompt length and original CAM++',
        reference_disabled='sample(no_ref_audio=True): zero latent condition and model speaker dropout; retain prompt length')
    step_root = a.output_root/f'ema_{a.step}'
    ensure_contract(step_root/'protocol.json', contract)
    for variant in ALL_VARIANTS:
        ensure_contract(step_root/variant/'protocol.json', dict(contract, variant=variant))
    for row in rows:
        prior_receipt = check_output(prior_dir, row)
        output = step_root/'target_only'/row['audio_relative_path']
        receipt_path = step_root/'target_only'/'receipts'/(identity(row)+'.json')
        if output.exists() or receipt_path.exists():
            check_diagnostic_output(a.output_root, a.step, 'target_only', row, selection)
            continue
        frozen_bytes(output, (prior_dir/row['audio_relative_path']).read_bytes())
        if sha(output) != prior_receipt['sha256']:
            raise RuntimeError('Copied baseline hash differs')
        receipt = condition_record(row, 'target_only', selection, references)
        receipt.update(sha256=sha(output), protocol_sha256=sha(step_root/'target_only/protocol.json'),
            reused_from=str(prior_dir/row['audio_relative_path']),
            prior_receipt_sha256=sha(prior_dir/'receipts'/(identity(row)+'.json')),
            target_noise_sha256=None, note='prior baseline is verified; no fresh noise was sampled')
        ensure_contract(receipt_path, receipt)
    print(f'Prepared {len(rows)} paired targets and reused verified EMA {a.step} baselines', flush=True)


def validate_run(a):
    rows, selection = load_selection(a.output_root)
    protocol = read(a.output_root/f'ema_{a.step}'/'protocol.json')
    if (selection['prior_root'] != str(a.prior_root.resolve()) or selection['per_speaker'] != a.per_speaker
            or protocol['selection_sha256'] != sha(a.output_root/'shared/protocol.json')
            or protocol['manifest_sha256'] != selection['manifest_sha256']
            or protocol['cache_root'] != str(a.cache_root.resolve())
            or protocol['config_path'] != str(a.config.resolve())
            or protocol['semantic_vae_repo'] != str(a.semantic_vae_repo.resolve())
            or protocol['semantic_vae_checkpoint'] != str(a.semantic_vae_checkpoint.resolve())):
        raise RuntimeError('Arguments differ from prepared diagnosis')
    if a.checkpoint is not None and a.checkpoint.resolve() != Path(protocol['checkpoint']['path']):
        raise RuntimeError('Requested checkpoint path differs')
    if sha(a.prior_root/'shared/manifest.jsonl') != selection['prior_manifest_sha256']:
        raise RuntimeError('Prior manifest changed')
    if sha(a.cache_root/'state/latents/spec.json') != protocol['cache_spec_sha256']:
        raise RuntimeError('Decoder cache specification changed')
    validate_sources(protocol)
    references = {r['reference_id']: r['reference'] for r in rows}
    for ref in references.values():
        validate_reference(ref)
    invocation = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    signature = hashlib.sha256(json.dumps(invocation, sort_keys=True).encode()).hexdigest()[:16]
    ensure_contract(a.output_root/f'ema_{a.step}'/'invocations'/f'{a.task}_{signature}.json', invocation)
    return rows, selection, protocol, references


def infer(a):
    import numpy as np
    import torch
    from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import atomic_save_waveform, build_model, load_normalization
    from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
    rows, selection, protocol, references = validate_run(a)
    selected = rows if a.limit is None else rows[:a.limit]
    if not selected or a.limit is not None and not 0 < a.limit <= len(rows) or not 0 <= a.shard < a.nshard <= len(selected):
        raise ValueError('Invalid limit or shard')
    selected = selected[a.shard::a.nshard]
    variants = [v for v in a.variants if v != 'target_only']
    for row in selected:
        check_diagnostic_output(a.output_root, a.step, 'target_only', row, selection)
    if not variants:
        return
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    mean, std, _ = load_normalization(Path(protocol['config']['datasets']['normalization_path']))
    model = build_model(a.config.resolve(), Path(protocol['checkpoint']['path']), a.step, device)
    decoder, decoder_meta = load_semantic_vae_decoder(repo=a.semantic_vae_repo,
        checkpoint_root=a.semantic_vae_checkpoint, cache_spec=read(a.cache_root/'state/latents/spec.json'), device=device)
    ensure_contract(a.output_root/f'ema_{a.step}'/f'decoder_shard{a.shard}.json', decoder_meta)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    arrays = {key: (np.load(ref['ref_latent_path'], allow_pickle=False),
                   np.load(ref['ref_speaker_path'], allow_pickle=False)) for key, ref in references.items()}
    with torch.inference_mode():
        for index, row in enumerate(selected):
            video_path = a.cache_root/row['video_40hz_relative_path']
            if sha(video_path) != row['target_video_sha256']:
                raise RuntimeError('Target video cache changed')
            video = np.load(video_path, allow_pickle=False)
            latent, original_speaker = arrays[row['reference_id']]
            for variant in variants:
                directory = a.output_root/f'ema_{a.step}'/variant
                out = directory/row['audio_relative_path']
                receipt_path = directory/'receipts'/(identity(row)+'.json')
                if out.exists() or receipt_path.exists():
                    check_diagnostic_output(a.output_root, a.step, variant, row, selection)
                    continue
                receipt = condition_record(row, variant, selection, references)
                speaker = np.zeros_like(original_speaker) if variant == 'cam_zero' else original_speaker if variant == 'reference_disabled' else arrays[receipt['speaker_condition_reference_id']][1]
                input_video = video
                cond, combined_video, prompt, target_frames, duration = make_sampling_inputs(
                    latent, input_video, receipt['model_text'], mean, std)
                if variant == 'prompt_zero':
                    cond = np.zeros_like(cond)
                if (prompt, target_frames, duration) != (receipt['prompt_frames'], receipt['target_frames'], receipt['sampler_frames']):
                    raise RuntimeError('Condition intervention changed P/T/noise geometry')
                # Match CFM_VT.sample exactly; sample() resets seed again below.
                torch.manual_seed(0)
                initial = torch.randn(duration, 64, device=device, dtype=torch.float32)
                noise_sha = hashlib.sha256(initial[prompt:prompt+target_frames].cpu().numpy().tobytes()).hexdigest()
                del initial
                generated, _ = model.sample(cond=torch.from_numpy(cond)[None].to(device),
                    text=[receipt['model_text']], duration=torch.tensor([duration], device=device),
                    video=torch.from_numpy(combined_video)[None].to(device), lens=torch.tensor([prompt], device=device),
                    steps=32, cfg_strength=5, cfg_strength_v=2, sway_sampling_coef=-1, seed=0,
                    use_epss=True, speaker_embedding=torch.from_numpy(speaker)[None].to(device),
                    ignore_modality=receipt['ignore_modality'], no_ref_audio=receipt['no_ref_audio'])
                target = generated[:, prompt:prompt+target_frames].float()
                if target.shape != (1, target_frames, 64) or not torch.isfinite(target).all():
                    raise RuntimeError('Invalid generated target latent')
                waveform = decoder((target*std_t+mean_t).transpose(1, 2)).squeeze(0).float().cpu()
                if waveform.shape != (1, target_frames*400) or not torch.isfinite(waveform).all():
                    raise RuntimeError('Invalid decoded target waveform')
                atomic_save_waveform(out, waveform[:, :row['original_num_samples']])
                receipt.update(sha256=sha(out), protocol_sha256=sha(directory/'protocol.json'),
                    target_noise_sha256=noise_sha,
                    supplied_speaker_array_sha256=hashlib.sha256(speaker.tobytes()).hexdigest(),
                    supplied_video_array_sha256=hashlib.sha256(combined_video.tobytes()).hexdigest(),
                    supplied_normalized_prompt_sha256=hashlib.sha256(cond.tobytes()).hexdigest(),
                    effective_normalized_prompt_sha256=hashlib.sha256(
                        (np.zeros_like(cond) if receipt['no_ref_audio'] else cond).tobytes()).hexdigest())
                if variant == 'baseline_regenerated':
                    prior = a.output_root/f'ema_{a.step}'/'target_only'/row['audio_relative_path']
                    receipt.update(prior_generated_sha256=sha(prior), byte_identical_to_prior=sha(prior) == sha(out))
                ensure_contract(receipt_path, receipt)
                check_diagnostic_output(a.output_root, a.step, variant, row, selection)
                print(f'EMA{a.step} shard{a.shard} {index+1}/{len(selected)} {variant} {identity(row)}', flush=True)
    for variant in variants:
        receipts = [check_diagnostic_output(a.output_root, a.step, variant, row, selection) for row in selected]
        ensure_contract(a.output_root/f'ema_{a.step}'/variant/f'shard{a.shard}_limit{a.limit}_complete.json',
            dict(count=len(selected), ids=[identity(r) for r in selected], shard=a.shard, nshard=a.nshard,
                 limit=a.limit, byte_identical_to_prior=all(r['byte_identical_to_prior'] for r in receipts)
                 if variant == 'baseline_regenerated' else None))


def reconstruct(a):
    import numpy as np
    import torch
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import atomic_save_waveform
    from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
    _, _, protocol, references = validate_run(a)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    decoder, metadata = load_semantic_vae_decoder(repo=a.semantic_vae_repo,
        checkpoint_root=a.semantic_vae_checkpoint, cache_spec=read(a.cache_root/'state/latents/spec.json'), device=device)
    directory = a.output_root/'controls/ref_reconstruction'
    ensure_contract(directory/'protocol.json', dict(protocol=PROTOCOL, control='reference_autoencoder_reconstruction',
        decoder=metadata, cache_spec_sha256=protocol['cache_spec_sha256'], dtype='float32',
        source_files=protocol['source_files'], count=len(references),
        method='decode raw cached reference posterior sample; crop to exact canonical reference samples'))
    with torch.inference_mode():
        for key, ref in references.items():
            relative = key.removeprefix('celebvdub/test/')
            if relative == key or len(Path(relative).parts) != 2 or '..' in Path(relative).parts:
                raise ValueError('Unsafe reference identity')
            out = directory/(relative+'.wav')
            receipt_path = directory/'receipts'/(relative+'.json')
            binding = dict(reference_id=key, reference_source_audio_sha256=ref['ref_source_audio_sha256'],
                reference_latent_sha256=ref['ref_latent_sha256'], samples=ref['ref_num_samples'],
                protocol_sha256=sha(directory/'protocol.json'))
            if out.exists() or receipt_path.exists():
                receipt = read(receipt_path)
                if any(receipt.get(k) != v for k, v in binding.items()) or receipt['sha256'] != sha(out):
                    raise RuntimeError('Reference reconstruction receipt differs')
                validate_wav(out, ref['ref_num_samples'])
                continue
            latent = torch.from_numpy(np.load(ref['ref_latent_path'], allow_pickle=False))[None].to(device)
            waveform = decoder(latent.transpose(1, 2)).squeeze(0).float().cpu()
            if waveform.shape != (1, ref['ref_frames']*400) or not torch.isfinite(waveform).all():
                raise RuntimeError('Invalid reference reconstruction')
            atomic_save_waveform(out, waveform[:, :ref['ref_num_samples']])
            validate_wav(out, ref['ref_num_samples'])
            ensure_contract(receipt_path, dict(binding, sha256=sha(out)))
            print(f'Reconstructed {key}', flush=True)


def args():
    base = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/projects')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['prepare', 'infer', 'reconstruct'], required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=[100000, 150000], default=150000)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--per-speaker', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Inference-only first N selected rows, then shard; use 3 for baseline regeneration')
    parser.add_argument('--variants', nargs='+', choices=[*ALL_VARIANTS, 'all'], default=list(VARIANTS))
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[2]/'config/finetune_grid.yaml')
    parser.add_argument('--cache-root', type=Path, default=base/'data/GRID_svae1000k_sample_seed666_fp32')
    parser.add_argument('--semantic-vae-repo', type=Path, default=base/'alignDiT_idea6/papers_codes/Semantic-VAE')
    parser.add_argument('--semantic-vae-checkpoint', type=Path, default=base/'alignDiT_idea6/Semantic-VAE/semantic_vae_1000k')
    result = parser.parse_args()
    if 'all' in result.variants:
        result.variants = list(VARIANTS)
    if len(set(result.variants)) != len(result.variants):
        raise ValueError('Duplicate variants')
    if result.output_root.resolve() == result.prior_root.resolve():
        raise ValueError('Diagnosis must use a separate output directory')
    return result


def main():
    a = args()
    {'prepare': prepare, 'infer': infer, 'reconstruct': reconstruct}[a.task](a)


if __name__ == '__main__':
    main()
