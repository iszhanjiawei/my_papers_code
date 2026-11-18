"""Inference and metric workers for the current Semantic-VAE/CAM++ GRID model."""
import argparse
import json
import os
from pathlib import Path

from aligndit.script.eval.grid_s1_common import ensure_contract, identity, publish, read, sha
from aligndit.script.eval.grid_zeroshot_common import (
    PROTOCOL, TASKS, WER_PROTOCOLS, check_output, jsonl_bytes, load_prepared, summarize, wer_variants,
)


def infer(a, rows, metadata):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
        atomic_save_waveform, build_model, load_composed_config, load_normalization,
    )
    from aligndit.script.eval.prepare_grid_reference import validate_cached_record
    from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
    from aligndit.model.grid_dataset import NORMALIZATION_SHA256, VOCAB_SHA256

    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    config = load_composed_config(a.config.resolve())
    if sha(config.datasets.normalization_path) != NORMALIZATION_SHA256 or sha(config.datasets.vocab_path) != VOCAB_SHA256:
        raise RuntimeError('Normalization or tokenizer differs from GRID training')
    mean, std, _ = load_normalization(Path(config.datasets.normalization_path))
    contract = dict(protocol=PROTOCOL, model_family='current GRID Semantic-VAE 64-D / CAM++ 192-D C2 snapshot',
        requested_count=len(rows), partial_smoke_test=a.smoke, text_conditioning='target-only',
        model_text='normalized supplied target GT dialogue only', reference_transcript_used_for_generation=False,
        target_audio_read_for_generation=False, target_latent_or_speaker_read_for_generation=False,
        shared_manifest_sha256=metadata['manifest_sha256'], shared_plan_sha256=metadata['plan_sha256'],
        document_sha256=metadata['document_sha256'], frozen_portable_sha256=metadata['frozen_portable_sha256'],
        checkpoint=dict(path=str(a.checkpoint.resolve()), sha256=sha(a.checkpoint), update=a.step, weights='EMA'),
        config=OmegaConf.to_container(config, resolve=True), normalization_sha256=NORMALIZATION_SHA256,
        vocab_sha256=VOCAB_SHA256,
        generation=dict(seed=0, nfe=32, ode='euler', epss=True, sway=-1, cfg_text=5, cfg_video=2, batch_size=1),
        reference='frozen CelebV-Dub test clip; reference latent and CAM++ from its complete canonical audio',
        target_duration='exact full GRID MPG audio/video duration; original audited mouth frames',
        output='only target continuation; remove P prompt frames; 16k mono PCM16; exact target samples')
    ensure_contract(a.output_dir/'protocol.json', contract)
    references = {r['reference_id']: r['reference'] for r in rows}
    for ref in references.values():
        if sha(ref['ref_source_audio']) != ref['ref_source_audio_sha256']:
            raise RuntimeError('Frozen external reference source changed')
        validate_cached_record(ref, {k: v for k, v in ref.items() if not k.endswith('_sha256')})
    model = build_model(a.config.resolve(), a.checkpoint.resolve(), a.step, device)
    decoder, decoder_meta = load_semantic_vae_decoder(repo=a.semantic_vae_repo,
        checkpoint_root=a.semantic_vae_checkpoint, cache_spec=read(a.cache_root/'state/latents/spec.json'), device=device)
    publish(a.output_dir/f'decoder_shard{a.shard}.json', decoder_meta)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    reference_arrays = {key: (np.load(ref['ref_latent_path'], allow_pickle=False),
                             torch.from_numpy(np.load(ref['ref_speaker_path'], allow_pickle=False)).to(device))
                        for key, ref in references.items()}
    selected = rows[a.shard::a.nshard]
    with torch.inference_mode():
        for row in tqdm(selected, desc=f'GRID zero-shot EMA{a.step} shard{a.shard}', mininterval=5):
            out = a.output_dir/row['audio_relative_path']
            receipt = a.output_dir/'receipts'/(identity(row)+'.json')
            if out.exists() or receipt.exists():
                check_output(a.output_dir, row)
                continue
            video_path = a.cache_root/row['video_40hz_relative_path']
            if sha(video_path) != row['target_video_sha256']:
                raise RuntimeError('Target video differs from prepared manifest')
            latent, speaker = reference_arrays[row['reference_id']]
            video = np.load(video_path, allow_pickle=False)
            cond, combined_video, prompt, target_frames, duration = make_sampling_inputs(latent, video, row['text'], mean, std)
            if target_frames != row['latent_frames'] or duration != prompt+target_frames:
                raise RuntimeError('Unexpected zero-shot prompt/target geometry')
            generated, _ = model.sample(cond=torch.from_numpy(cond)[None].to(device), text=[row['text']],
                duration=torch.tensor([duration], device=device), video=torch.from_numpy(combined_video)[None].to(device),
                lens=torch.tensor([prompt], device=device), steps=32, cfg_strength=5, cfg_strength_v=2,
                sway_sampling_coef=-1, seed=0, use_epss=True, speaker_embedding=speaker[None])
            target = generated[:, prompt:prompt+target_frames].float()
            if target.shape != (1, target_frames, 64) or not torch.isfinite(target).all():
                raise RuntimeError('Invalid generated target latent')
            waveform = decoder((target*std_t+mean_t).transpose(1, 2)).squeeze(0).float().cpu()
            if waveform.shape != (1, target_frames*400) or not torch.isfinite(waveform).all():
                raise RuntimeError('Invalid decoded target waveform')
            atomic_save_waveform(out, waveform[:, :row['original_num_samples']])
            publish(receipt, dict(id=identity(row), global_index=row['global_index'], sha256=sha(out),
                samples=row['original_num_samples'], reference_id=row['reference_id'],
                reference_source_audio_sha256=row['reference']['ref_source_audio_sha256'],
                reference_latent_sha256=row['reference']['ref_latent_sha256'],
                reference_speaker_sha256=row['reference']['ref_speaker_sha256'],
                prompt_frames=prompt, target_frames=target_frames, sampler_frames=duration,
                model_text=row['text'], text_conditioning='target-only', protocol_sha256=sha(a.output_dir/'protocol.json')))
    publish(a.output_dir/f'shard{a.shard}_complete.json', dict(count=len(selected), shard=a.shard, nshard=a.nshard))


def verify(a, rows, metadata):
    contract = read(a.output_dir/'protocol.json')
    if (contract['checkpoint']['update'] != a.step or contract['checkpoint']['sha256'] != sha(a.checkpoint)
            or contract['shared_manifest_sha256'] != metadata['manifest_sha256']
            or contract['requested_count'] != len(rows) or contract['partial_smoke_test'] != a.smoke):
        raise RuntimeError('Inference contract differs from requested model/data selection')
    outputs = [check_output(a.output_dir, r) for r in rows]
    actual = {p.relative_to(a.output_dir).as_posix() for p in a.output_dir.glob('val/*/*.wav')}
    if actual != {r['audio_relative_path'] for r in rows}:
        raise RuntimeError('Incomplete or extra generated WAV coverage')
    summary = dict(protocol=PROTOCOL, count=len(rows), complete=not a.smoke, partial_smoke_test=a.smoke,
                   all_requested_succeeded=True, outputs=outputs, protocol_sha256=sha(a.output_dir/'protocol.json'))
    publish(a.output_dir/'inference_summary.json', summary)
    print(f'Verified {len(rows)} zero-shot WAVs and external-reference bindings', flush=True)


def audit_assets(a):
    from aligndit.script.eval.eval_grid_s1 import audit_assets as original_audit
    a.setting = 's2'
    original_audit(a)
    record = read(a.output_dir/'metric_assets.json')
    extra = [Path(__file__).with_name(name) for name in (
        'grid_zeroshot_common.py', 'prepare_grid_zeroshot.py', 'eval_grid_zeroshot.py',
        'run_grid_zeroshot.py', 'verify_grid_zeroshot.py',
        'prepare_grid_reference.py', 'infer_celebvdub_semantic_vae_s1.py', 'semantic_vae_decoder.py')]
    project = Path(__file__).parents[4]
    extra += [project/'src/aligndit/script/misc/extract_campplus_speaker_embeddings.py',
              project/'src/aligndit/model/cfm_vt.py', project/'src/aligndit/model/backbone/dit_vt_mm.py',
              project/'src/f5_tts/eval/ecapa_tdnn.py', a.config]
    record['source_files'] += [dict(path=str(p.resolve()), sha256=sha(p)) for p in extra]
    publish(a.output_dir/'metric_assets.json', record)


def speaker_similarity(a, rows):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from tqdm import tqdm
    from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None,
                             wavlm_ckpt_path=str(a.wavlm_base_ckpt))
    state = torch.load(a.wavlm_ckpt, weights_only=True, map_location='cpu')['model'].copy()
    state.pop('loss_calculator.projection.weight')
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    reference_embeddings = {}

    def embed(path, reference=False):
        wav, rate = torchaudio.load(str(path))
        if reference:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[0] != 1 or not torch.isfinite(wav).all():
            raise RuntimeError('Invalid speaker-scoring audio')
        wav = torchaudio.functional.resample(wav, rate, 16000)
        result = model(wav.cuda())
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite speaker embedding')
        return result

    results = []
    with torch.inference_mode():
        for row in tqdm(rows, desc='SPKSIM_GT and SPKSIM_REF', mininterval=5):
            generated = embed(a.output_dir/row['audio_relative_path'])
            gt = embed(a.cache_root/'audio'/row['audio_relative_path'])
            key = row['reference_id']
            if key not in reference_embeddings:
                reference_embeddings[key] = embed(row['reference']['ref_source_audio'], reference=True)
            reference = reference_embeddings[key]
            results.append(dict(wav=Path(row['audio_relative_path']).stem,
                sim=F.cosine_similarity(generated, gt)[0].item(),
                sim_ref=F.cosine_similarity(generated, reference)[0].item()))
    return results


def metric(a, rows):
    import torch
    from aligndit.script.eval.eval_grid_s1 import run_avsync, run_wer
    torch.set_num_threads(4)
    summary = read(a.output_dir/'inference_summary.json')
    if summary['count'] != len(rows) or summary['partial_smoke_test'] != a.smoke or summary['protocol_sha256'] != sha(a.output_dir/'protocol.json'):
        raise RuntimeError('Metrics require verified inference for this exact selection')
    rows = rows[a.shard::a.nshard]
    receipts = [check_output(a.output_dir, r) for r in rows]
    for row in rows:
        if sha(a.cache_root/'audio'/row['audio_relative_path']) != row['target_gt_sha256']:
            raise RuntimeError('Evaluation-only target GT audio changed')
        if sha(row['reference']['ref_source_audio']) != row['reference']['ref_source_audio_sha256']:
            raise RuntimeError('External reference source changed')
    tests = [(str(a.output_dir/r['audio_relative_path']), str(a.cache_root/'audio'/r['audio_relative_path']), r['raw_truth']) for r in rows]
    if a.task == 'sim':
        results = speaker_similarity(a, rows)
    elif a.task == 'wer':
        results = run_wer(a, rows, tests)
        for result in results:
            variants = wer_variants(result['raw_truth'], result['raw_hypo'])
            result.update(wer_variants=variants, **variants['digits_silence'])
    elif a.task == 'emosim':
        from aligndit.script.eval.utils import run_emosim
        results = run_emosim((0, tests, str(a.emo_ckpt)))
    else:
        results = run_avsync(a, rows, tests)
    if len(results) != len(rows):
        raise RuntimeError('Incomplete worker results')
    for row, receipt, result in zip(rows, receipts, results):
        if result['wav'] != Path(row['audio_relative_path']).stem:
            raise RuntimeError('Metric output order mismatch')
        result.update(id=identity(row), wav=identity(row), reference_id=row['reference_id'],
                      generated_audio_sha256=receipt['sha256'], target_gt_sha256=row['target_gt_sha256'],
                      reference_source_audio_sha256=row['reference']['ref_source_audio_sha256'])
    from aligndit.script.eval.prepare_grid_reference import atomic_bytes
    atomic_bytes(a.output_dir/f'{a.task}_shard{a.shard}.jsonl', jsonl_bytes(results))
    score = summarize(results, a.task)
    publish(a.output_dir/f'{a.task}_shard{a.shard}_summary.json', dict(score, nshard=a.nshard,
            inference_summary_sha256=sha(a.output_dir/'inference_summary.json')))
    print(json.dumps(score), flush=True)


def merge(a, rows):
    from aligndit.script.eval.prepare_grid_reference import atomic_bytes
    expected = [identity(r) for r in rows]
    report = dict(protocol=PROTOCOL, count=len(rows), complete=not a.smoke, partial_smoke_test=a.smoke,
                  all_requested_succeeded=True, checkpoint=read(a.output_dir/'protocol.json')['checkpoint'],
                  text_conditioning='target-only', metrics={}, metric_summaries={},
                  metric_assets_sha256=sha(a.output_dir/'metric_assets.json'),
                  definitions=dict(sim='WavLM/ECAPA cosine vs target GRID GT',
                      sim_ref='WavLM/ECAPA cosine vs actual frozen CelebV-Dub reference; channel mean before 16k resampling',
                      wer='digits_silence corpus WER; three variants reuse each raw ASR hypothesis',
                      emosim='emotion2vec classification-score cosine vs target GT',
                      avsync='mean frame joint AV-HuBERT cosine; same target mouth video + generated or target GT audio'))
    for task in TASKS:
        collected = []
        for shard in range(a.nshard):
            part = [json.loads(line) for line in (a.output_dir/f'{task}_shard{shard}.jsonl').read_text().splitlines()]
            stored = read(a.output_dir/f'{task}_shard{shard}_summary.json')
            summary = summarize(part, task)
            if ([r['id'] for r in part] != expected[shard::a.nshard] or stored['nshard'] != a.nshard
                    or stored['inference_summary_sha256'] != sha(a.output_dir/'inference_summary.json')
                    or any(stored.get(k) != v for k, v in summary.items())):
                raise RuntimeError('Metric shard coverage, input or summary differs')
            collected.extend(part)
        by_id = {r['id']: r for r in collected}
        if len(collected) != len(rows) or set(by_id) != set(expected):
            raise RuntimeError('Incomplete/duplicate metric ID coverage')
        ordered = [by_id[k] for k in expected]
        for row, result in zip(rows, ordered):
            receipt = check_output(a.output_dir, row)
            if (result['reference_id'] != row['reference_id'] or result['generated_audio_sha256'] != receipt['sha256']
                    or result['target_gt_sha256'] != row['target_gt_sha256']
                    or result['reference_source_audio_sha256'] != row['reference']['ref_source_audio_sha256']):
                raise RuntimeError('Metric/reference/audio binding differs')
            if task == 'wer' and (result['raw_truth'] != row['raw_truth'] or result['wer_variants'] != wer_variants(row['raw_truth'], result['raw_hypo'])):
                raise RuntimeError('Three-protocol WER independent recomputation differs')
        atomic_bytes(a.output_dir/f'{task}_results.jsonl', jsonl_bytes(ordered))
        summary = summarize(ordered, task)
        report['metric_summaries'][task] = summary
        if task == 'wer':
            for variant in WER_PROTOCOLS:
                report['metrics']['wer_'+variant] = summary['wer_variants'][variant]['wer']
            report['metrics']['wer'] = summary['wer']
            if summary['wer_variants']['digits_silence']['reference_word_count'] != 6*len(rows):
                raise RuntimeError('Main corpus denominator differs from six GRID words per target')
        else:
            for key in (('sim', 'sim_ref') if task == 'sim' else (task,)):
                report['metrics'][key] = summary[key]
    report['display_values'] = {k: f'{v:.5f}' for k, v in report['metrics'].items()}
    publish(a.output_dir/'evaluation_summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def args():
    base = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/projects')
    pretrain = Path(os.environ.get('ROOT_PREFIX', '')+'/zjw524/alignDiT_pretrain_models')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['infer', 'verify', 'audit', 'merge', *TASKS], required=True)
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--cache-root', type=Path, default=base/'data/GRID_svae1000k_sample_seed666_fp32')
    p.add_argument('--config', type=Path, default=Path(__file__).parents[2]/'config/finetune_grid.yaml')
    p.add_argument('--semantic-vae-repo', type=Path, default=base/'alignDiT_idea6/papers_codes/Semantic-VAE')
    p.add_argument('--semantic-vae-checkpoint', type=Path, default=base/'alignDiT_idea6/Semantic-VAE/semantic_vae_1000k')
    p.add_argument('--wavlm-ckpt', type=Path, default=pretrain/'wavlm_large_finetune.pth')
    p.add_argument('--wavlm-base-ckpt', type=Path, default=pretrain/'wavlm_large_s3prl.pt')
    p.add_argument('--asr-ckpt', type=Path, default=base/'data/faster-whisper-large-v3')
    p.add_argument('--emo-ckpt', type=Path, default=base/'data/emotion2vec_plus_large')
    p.add_argument('--avhubert-checkpoint', type=Path, default=base/'data/large_vox_iter5.pt')
    p.add_argument('--fairseq-repo', type=Path, default=base/'data/av_hubert/fairseq/fairseq')
    p.add_argument('--avhubert-user-dir', type=Path, default=base/'data/av_hubert/avhubert/avhubert')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='One target per frozen reference; separate output directory')
    return p.parse_args()


def main():
    a = args()
    rows, metadata = load_prepared(a.shared_dir, a.smoke)
    if not 0 <= a.shard < a.nshard <= len(rows):
        raise ValueError('Invalid shard')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.task == 'infer':
        infer(a, rows, metadata)
    elif a.task == 'verify':
        verify(a, rows, metadata)
    elif a.task == 'audit':
        audit_assets(a)
    elif a.task == 'merge':
        verify(a, rows, metadata)
        merge(a, rows)
    else:
        metric(a, rows)


if __name__ == '__main__':
    main()
