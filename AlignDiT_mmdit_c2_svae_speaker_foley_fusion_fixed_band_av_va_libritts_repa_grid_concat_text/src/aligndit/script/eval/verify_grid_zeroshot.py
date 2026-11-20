"""Independent full-ID/source audit; deliberately does not import the evaluator."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import re
import string
import wave
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(text, variant):
    if variant == 'original':
        from zhon.hanzi import punctuation
        for character in punctuation+string.punctuation:
            text = text.replace(character, '')
        text = text.replace('  ', ' ').lower()
    else:
        text = re.sub(r'[^\w\s]', '', text).lower()
        words = 'zero one two three four five six seven eight nine'.split()
        for digit, word in enumerate(words):
            text = text.replace(str(digit), ' '+word)
    return ' '.join(w for w in text.split() if variant != 'digits_silence' or w not in {'sp', 'sil'})


def distance(a, b):
    grid = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        grid[i][0] = i
    for j in range(len(b)+1):
        grid[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            grid[i][j] = min(grid[i-1][j]+1, grid[i][j-1]+1, grid[i-1][j-1]+(a[i-1] != b[j-1]))
    return grid[-1][-1]


def verify(a):
    import jiwer
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    torch.set_num_threads(4)
    root, shared = a.output_dir, a.shared_dir
    metadata, plan = read(shared/'metadata.json'), read(shared/'plan.json')
    assert metadata['status'] == 'complete' and metadata['count'] == 3274 and metadata['reference_count'] == 33
    assert metadata['plan_sha256'] == sha(shared/'plan.json') and metadata['manifest_sha256'] == sha(shared/'manifest.jsonl')
    assert sha(plan['document']) == metadata['document_sha256'] == plan['document_sha256']
    parser_file = Path(plan['document']).with_name('rebuild_from_document.py')
    spec = importlib.util.spec_from_file_location('independent_frozen_document_parser', parser_file)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    portable, blob, frozen_protocol, _ = parser.load_document(Path(plan['document']))
    assert hashlib.sha256(blob).hexdigest() == metadata['frozen_portable_sha256'] == sha(shared/'document_portable_pairs.jsonl')
    full = [json.loads(line) for line in (shared/'manifest.jsonl').read_text().splitlines()]
    cache = Path(plan['cache_root'])
    cached = [json.loads(line) for line in (cache/'manifests/val.jsonl').read_text().splitlines()]
    cache_by_id = {r['audio_relative_path'].removeprefix('val/').removesuffix('.wav'): r for r in cached}
    assert len(cache_by_id) == 3274 and plan['cache_manifest_sha256'] == sha(cache/'manifests/val.jsonl')
    assert plan['cache_completion_sha256'] == sha(cache/'complete.json')
    assert read(cache/'complete.json')['complete'] is True
    source_meta = read(cache/'manifests/inventory_meta.json')
    exclusions = {r['id']: r['reason'] for r in source_meta['exclusions'] if r['split'] == 'val'}
    frozen_by_id = {r['id']: r for r in portable}
    assert set(cache_by_id) < set(frozen_by_id)
    omitted = [r for r in portable if r['id'] not in cache_by_id]
    assert len(omitted) == 7 and plan['omissions'] == [dict(id=r['id'], global_index=r['index'], reason=exclusions[r['id']]) for r in omitted]
    assert [r['global_index'] for r in full] == [r['index'] for r in portable if r['id'] in cache_by_id]
    for row in full:
        key = row['audio_relative_path'].removeprefix('val/').removesuffix('.wav')
        frozen = frozen_by_id[key]
        assert row['raw_truth'] == frozen['text'] and row['text'] == normalize(frozen['text'], 'digits_silence')
        assert row['reference_id'] == frozen['reference_id'] == plan['assignments'][row['speaker']]
        assert row['reference']['ref_source_audio_sha256'] == frozen['reference_audio_sha256']
        for name, value in cache_by_id[key].items():
            assert row[name] == value
    rows = full
    if a.smoke:
        rows, seen = [], set()
        for row in full:
            if row['reference_id'] not in seen:
                rows.append(row)
                seen.add(row['reference_id'])
    expected_count = 33 if a.smoke else 3274
    assert len(rows) == expected_count
    ids = [Path(r['audio_relative_path']).with_suffix('').as_posix() for r in rows]
    assert len(ids) == len(set(ids))
    protocol = read(root/'protocol.json')
    assert protocol['protocol'] == 'grid_svae_campplus_celebvdub_reference_target_text_only_v1'
    assert protocol['checkpoint']['update'] in (100000, 150000) and protocol['checkpoint']['weights'] == 'EMA'
    assert protocol['partial_smoke_test'] == a.smoke and protocol['requested_count'] == expected_count
    assert protocol['shared_manifest_sha256'] == metadata['manifest_sha256']
    assert protocol['shared_plan_sha256'] == metadata['plan_sha256']
    assert protocol['text_conditioning'] == 'target-only' and protocol['reference_transcript_used_for_generation'] is False
    assert protocol['target_audio_read_for_generation'] is False and protocol['target_latent_or_speaker_read_for_generation'] is False
    assert protocol['generation'] == dict(seed=0, nfe=32, ode='euler', epss=True, sway=-1, cfg_text=5, cfg_video=2, batch_size=1)
    assert sha(protocol['config']['datasets']['normalization_path']) == protocol['normalization_sha256']
    assert sha(protocol['config']['datasets']['vocab_path']) == protocol['vocab_sha256']
    references = {r['reference_id']: r['reference'] for r in rows}
    frozen_refs = {r['id']: r for r in frozen_protocol['references']}
    max_channel_mean_error = 0.
    for key, ref in references.items():
        frozen = frozen_refs[key]
        assert sha(ref['ref_source_audio']) == ref['ref_source_audio_sha256'] == frozen['audio_sha256']
        assert sha(ref['ref_transcript_path']) == ref['ref_transcript_sha256'] == frozen['text_sha256']
        for kind in ('ref_audio', 'ref_latent', 'ref_speaker'):
            path = ref[kind] if kind == 'ref_audio' else ref[kind+'_path']
            assert sha(path) == ref[kind+'_sha256']
        source, rate = sf.read(ref['ref_source_audio'], dtype='float32', always_2d=True)
        canonical, canonical_rate = sf.read(ref['ref_audio'], dtype='float32', always_2d=True)
        assert rate == frozen['sample_rate'] and canonical_rate == 16000
        assert source.shape == (frozen['frames'], frozen['channels'])
        assert canonical.shape == (ref['ref_num_samples'], 1)
        assert ref['ref_num_samples'] == math.ceil(frozen['frames']*16000/rate)
        expected_mono = torchaudio.functional.resample(torch.from_numpy(source.mean(axis=1)), rate, 16000).numpy()
        expected_mono = np.clip(expected_mono, -1, 32767/32768)
        assert len(expected_mono) == len(canonical)
        error = float(np.max(np.abs(canonical[:, 0]-expected_mono)))
        assert error <= 1/32768+1e-7
        max_channel_mean_error = max(max_channel_mean_error, error)
        latent = np.load(ref['ref_latent_path'], allow_pickle=False)
        embedding = np.load(ref['ref_speaker_path'], allow_pickle=False)
        assert latent.shape == (math.ceil(len(canonical)/400), 64) and latent.dtype == np.float32 and np.isfinite(latent).all()
        assert ref['ref_frames'] == len(latent)
        assert embedding.shape == (192,) and embedding.dtype == np.float32 and np.isfinite(embedding).all()
        assert abs(float(np.linalg.norm(embedding))-1) < 1e-5
        assert ref['posterior_seed'] == int.from_bytes(hashlib.sha256(('666:'+key).encode()).digest()[:8], 'little') % (2**63-1)
    assert len(references) == 33 and len({k.split('/')[2] for k in references}) == 33
    print('Verified all 33 frozen reference WAV/TXT, mono frontend, latents, CAM++ and seeds', flush=True)
    inference = read(root/'inference_summary.json')
    assert inference['count'] == expected_count and inference['all_requested_succeeded'] is True
    assert inference['complete'] is (not a.smoke) and inference['protocol_sha256'] == sha(root/'protocol.json')
    generated_hashes, gt_hashes = {}, {}
    for index, (row, key) in enumerate(zip(rows, ids)):
        ref = row['reference']
        receipt = read(root/'receipts'/f'{key}.json')
        assert receipt == inference['outputs'][index]
        assert receipt['id'] == key and receipt['global_index'] == row['global_index']
        assert receipt['reference_id'] == row['reference_id']
        assert receipt['reference_source_audio_sha256'] == ref['ref_source_audio_sha256']
        assert receipt['reference_latent_sha256'] == ref['ref_latent_sha256'] and receipt['reference_speaker_sha256'] == ref['ref_speaker_sha256']
        assert receipt['model_text'] == row['text'] and receipt['text_conditioning'] == 'target-only'
        assert receipt['protocol_sha256'] == sha(root/'protocol.json')
        assert receipt['prompt_frames'] == ref['ref_frames'] and receipt['target_frames'] == row['latent_frames']
        assert receipt['sampler_frames'] == ref['ref_frames']+row['latent_frames']
        assert receipt['samples'] == row['original_num_samples'] == 640*row['source_video_frames']
        generated_hashes[key] = sha(root/row['audio_relative_path'])
        gt_hashes[key] = sha(cache/'audio'/row['audio_relative_path'])
        assert generated_hashes[key] == receipt['sha256'] and gt_hashes[key] == row['target_gt_sha256']
        assert sha(cache/row['video_40hz_relative_path']) == row['target_video_sha256']
        with wave.open(str(root/row['audio_relative_path'])) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, row['original_num_samples'])
        if (index+1)%500 == 0:
            print('Verified target/reference bindings and output WAVs', index+1, flush=True)
    assert {p.relative_to(root).with_suffix('').as_posix() for p in root.glob('val/*/*.wav')} == set(ids)
    report = read(root/'evaluation_summary.json')
    assert report['count'] == expected_count and report['all_requested_succeeded'] is True and report['complete'] is (not a.smoke)
    assert report['partial_smoke_test'] == a.smoke and report['checkpoint'] == protocol['checkpoint']
    all_results, scores, wer_totals = {}, {}, {}
    for task in ('sim', 'wer', 'emosim', 'avsync'):
        results = [json.loads(line) for line in (root/f'{task}_results.jsonl').read_text().splitlines()]
        assert [r['id'] for r in results] == ids and all(r['wav'] == r['id'] for r in results)
        all_results[task] = results
        for shard in range(a.nshard):
            part = [json.loads(line) for line in (root/f'{task}_shard{shard}.jsonl').read_text().splitlines()]
            assert part == results[shard::a.nshard]
            stored = read(root/f'{task}_shard{shard}_summary.json')
            assert stored['count'] == len(part) and stored['nshard'] == a.nshard
            assert stored['inference_summary_sha256'] == sha(root/'inference_summary.json')
        for row, result, key in zip(rows, results, ids):
            assert result['reference_id'] == row['reference_id'] and result['generated_audio_sha256'] == generated_hashes[key]
            assert result['target_gt_sha256'] == gt_hashes[key]
            assert result['reference_source_audio_sha256'] == row['reference']['ref_source_audio_sha256']
            if task == 'wer':
                assert result['raw_truth'] == row['raw_truth'] and isinstance(result['raw_hypo'], str)
                assert set(result['wer_variants']) == {'original', 'digits_only', 'digits_silence'}
                for variant in result['wer_variants']:
                    item = result['wer_variants'][variant]
                    truth, hypo = normalize(result['raw_truth'], variant), normalize(result['raw_hypo'], variant)
                    assert (truth, hypo) == (item['truth'], item['hypo'])
                    counts = jiwer.process_words(truth, hypo)
                    edits = distance(truth.split(), hypo.split())
                    assert edits == item['word_edit_distance'] == counts.substitutions+counts.deletions+counts.insertions
                    assert [item[k] for k in ('hits', 'substitutions', 'deletions', 'insertions')] == [counts.hits, counts.substitutions, counts.deletions, counts.insertions]
                    assert item['reference_word_count'] == len(truth.split()) and item['wer'] == edits/len(truth.split())
            else:
                for metric in (('sim', 'sim_ref') if task == 'sim' else (task,)):
                    assert math.isfinite(result[metric]) and -1.000001 <= result[metric] <= 1.000001
        if task == 'wer':
            for variant in ('original', 'digits_only', 'digits_silence'):
                counts = {k: sum(r['wer_variants'][variant][k] for r in results) for k in ('hits', 'substitutions', 'deletions', 'insertions')}
                errors = counts['substitutions']+counts['deletions']+counts['insertions']
                words = counts['hits']+counts['substitutions']+counts['deletions']
                value = errors/words
                expected = report['metric_summaries']['wer']['wer_variants'][variant]
                assert expected['word_edit_distance'] == errors and expected['reference_word_count'] == words
                assert all(expected[k] == v for k, v in counts.items()) and expected['wer'] == value
                assert report['metrics']['wer_'+variant] == value
                scores['wer_'+variant] = value
                wer_totals[variant] = dict(**counts, word_edit_distance=errors, reference_word_count=words, wer=value)
            assert wer_totals['digits_silence']['reference_word_count'] == 6*expected_count
            assert report['metrics']['wer'] == wer_totals['digits_silence']['wer']
            scores['wer'] = wer_totals['digits_silence']['wer']
        else:
            for metric in (('sim', 'sim_ref') if task == 'sim' else (task,)):
                value = math.fsum(r[metric] for r in results)/expected_count
                assert value == report['metrics'][metric]
                scores[metric] = value
        print('Independent metric/coverage/word-count recomputation passed:', task, flush=True)
    assert report['display_values'] == {k: f'{v:.5f}' for k, v in scores.items()}
    max_av_difference = 0.
    for index, (row, key, scored) in enumerate(zip(rows, ids, all_results['avsync'])):
        arrays = []
        for role, audio_hash in (('gen', generated_hashes[key]), ('gt', gt_hashes[key])):
            path = root/'avhubert_joint'/role/f'{key}.npy'
            receipt = read(path.with_suffix('.json'))
            assert receipt['id'] == key and receipt['sha256'] == sha(path) and receipt['audio_sha256'] == audio_hash
            assert receipt['mouth_sha256'] == row['source']['mouth_sha256']
            assert receipt['checkpoint_sha256'] == '343c9dbaa29847b8c4d0d9503c43f2877fd2efde9988d41abf14e74f28232b75'
            assert receipt['feature'] == 'joint_audio_video_25hz_finetune_1024'
            array = np.load(path, allow_pickle=False)
            assert array.shape == (row['source_video_frames'], 1024) and array.dtype == np.float32 and np.isfinite(array).all()
            arrays.append(array.astype(np.float64))
        x, y = arrays
        value = float(np.mean(np.sum(x*y, axis=1)/(np.maximum(np.linalg.norm(x, axis=1), 1e-8)*np.maximum(np.linalg.norm(y, axis=1), 1e-8))))
        difference = abs(value-scored['avsync'])
        assert difference < 1e-6
        max_av_difference = max(max_av_difference, difference)
        if (index+1)%500 == 0:
            print('Verified joint feature pairs', index+1, flush=True)
    for role in ('gen', 'gt'):
        assert {p.relative_to(root/'avhubert_joint'/role).with_suffix('').as_posix() for p in (root/'avhubert_joint'/role).glob('val/*/*.npy')} == set(ids)
    assets = read(root/'metric_assets.json')
    assert report['metric_assets_sha256'] == sha(root/'metric_assets.json')
    for asset in assets['assets']:
        assert Path(asset['path']).stat().st_size == asset['size_bytes'] and sha(asset['path']) == asset['sha256']
    assert sha(protocol['checkpoint']['path']) == protocol['checkpoint']['sha256']
    verifier_updates = []
    for record in assets['source_files']+read(root/'launch_source_snapshot.json')['source_files']:
        actual_sha = sha(record['path'])
        if actual_sha != record['sha256']:
            # Only the independent verifier may be updated while auditing an
            # existing smoke run. Inference/metric inputs and source stay strict.
            assert a.allow_verifier_update and a.smoke and Path(record['path']).resolve() == Path(__file__).resolve(), record['path']
            verifier_updates.append(dict(path=record['path'], previous_sha256=record['sha256'], actual_sha256=actual_sha))
    assert sha(plan['speaker_checkpoint']) == plan['speaker_checkpoint_sha256']
    result = dict(requested=expected_count, succeeded=expected_count, failed=0, complete=not a.smoke,
        partial_smoke_test=a.smoke, all_requested_succeeded=True, checkpoint=protocol['checkpoint'],
        protocol=protocol['protocol'], text_conditioning='target-only', metrics=scores, wer_variants=wer_totals,
        reference_count=33, cross_dataset_pairing_verified=True, omitted_document_targets=plan['omissions'],
        joint_feature_counts=dict(gen=expected_count, gt=expected_count),
        max_numpy_avsync_recompute_difference=max_av_difference,
        max_reference_channel_mean_PCM16_difference=max_channel_mean_error,
        model_assets_rehashed=len(assets['assets']), checkpoint_rehashed=True,
        source_hashes_unchanged=not verifier_updates, inference_metric_sources_unchanged=True,
        smoke_verifier_source_updates=verifier_updates,
        frozen_document_parser_sha256=sha(parser_file), verification_script_sha256=sha(__file__),
        evaluation_summary_sha256=sha(root/'evaluation_summary.json'), verified_at_utc=datetime.now(timezone.utc).isoformat())
    (root/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--nshard', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-verifier-update', action='store_true',
                   help='Smoke audits only: record this verifier changing, still reject every inference/metric source change')
    verify(p.parse_args())


if __name__ == '__main__':
    main()
