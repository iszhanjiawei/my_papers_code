"""Persist speaker embeddings and score paired, small GRID zero-shot ablations.

The WavLM path matches eval_grid_zeroshot; CAM++ is a separate diagnostic
using the conditioning extractor. GRID centroids exclude every selected target
and contain two other validation utterances per speaker, so matching content
cannot by itself explain identification against these centroids.
"""
import argparse
from collections import defaultdict
import fcntl
import gc
import io
import json
import os
from pathlib import Path

import numpy as np

from aligndit.script.eval.grid_s1_common import ensure_contract, identity, publish, read, sha
from aligndit.script.eval.grid_zeroshot_common import jsonl_bytes, load_prepared, summarize, wer_variants
from aligndit.script.eval.prepare_grid_reference import atomic_bytes


VARIANTS = ('target_only', 'ref_text', 'cam_zero', 'cam_swapped', 'prompt_zero',
            'reference_disabled', 'video_disabled', 'ref_text_video_disabled')
FAMILIES = ('wavlm', 'campplus')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_controls(all_rows, selected, per_speaker=2):
    excluded = {identity(row) for row in selected}
    speakers = {row['speaker'] for row in selected}
    controls = defaultdict(list)
    for row in all_rows:
        if row['speaker'] in speakers and identity(row) not in excluded and len(controls[row['speaker']]) < per_speaker:
            controls[row['speaker']].append(row)
    if set(controls) != speakers or any(len(rows) != per_speaker for rows in controls.values()):
        raise RuntimeError('Need two other validation utterances per selected GRID speaker')
    return dict(controls)


def generated_path(a, variant, row):
    if variant == 'target_only':
        return a.prior_root / f'ema_{a.step}' / row['audio_relative_path']
    return a.output_root / f'ema_{a.step}' / variant / row['audio_relative_path']


def prepare_inventory(a):
    all_rows, metadata = load_prepared(a.prior_root / 'shared')
    selected_path = a.output_root / 'shared/manifest.jsonl'
    selected = read_jsonl(selected_path)
    by_id = {identity(row): row for row in all_rows}
    if not selected or len({identity(row) for row in selected}) != len(selected):
        raise RuntimeError('Empty or duplicated diagnosis selection')
    for row in selected:
        if row != by_id.get(identity(row)):
            raise RuntimeError('Diagnosis selection changed a frozen prior row')
    protocol = read(a.output_root / 'shared/protocol.json')
    swap = protocol['cam_swapped_reference_ids']
    controls = select_controls(all_rows, selected)
    refs = {row['reference_id']: row['reference'] for row in selected}
    if len(refs) != 33 or len(controls) != 33 or set(swap) != set(controls):
        raise RuntimeError('Diagnosis must cover all 33 speaker/reference groups')
    if set(swap.values()) != set(refs):
        raise RuntimeError('Swapped CAM mapping must permute the selected references')
    inventory = []

    def add(key, path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError(f'Changed audio: {path}')
        inventory.append(dict(key=key, path=str(path.resolve()), sha256=digest))

    for ref_id, ref in refs.items():
        add('reference/' + ref_id, ref['ref_source_audio'], ref['ref_source_audio_sha256'])
        add('reference_canonical/' + ref_id, ref['ref_audio'], ref['ref_audio_sha256'])
        source_id = ref_id.removeprefix('celebvdub/test/')
        reconstruction_root = a.output_root / 'controls/ref_reconstruction'
        reconstruction_receipt = read(reconstruction_root / 'receipts' / (source_id + '.json'))
        add('reference_reconstruction/' + ref_id, reconstruction_root / (source_id + '.wav'), reconstruction_receipt['sha256'])
        if sha(ref['ref_speaker_path']) != ref['ref_speaker_sha256']:
            raise RuntimeError('Changed cached conditioning embedding')
    for row in selected:
        item_id = identity(row)
        add('gt/' + item_id, a.cache_root / 'audio' / row['audio_relative_path'], row['target_gt_sha256'])
        for variant in a.variants:
            path = generated_path(a, variant, row)
            receipt_root = a.prior_root / f'ema_{a.step}' if variant == 'target_only' else a.output_root / f'ema_{a.step}' / variant
            receipt = read(receipt_root / 'receipts' / (item_id + '.json'))
            if receipt.get('id') != item_id or receipt.get('reference_id') != row['reference_id']:
                raise RuntimeError('Generated audio receipt has a different target/reference')
            if variant == 'cam_swapped' and receipt.get('speaker_condition_reference_id') != swap[row['speaker']]:
                raise RuntimeError('CAM swap receipt differs from diagnosis protocol')
            add(f'generated/{variant}/{item_id}', path, receipt['sha256'])
    for speaker, rows in controls.items():
        for row in rows:
            add('control/' + identity(row), a.cache_root / 'audio' / row['audio_relative_path'], row['target_gt_sha256'])
    if len({item['key'] for item in inventory}) != len(inventory):
        raise RuntimeError('Duplicate logical embedding key')
    plan = dict(step=a.step, variants=a.variants, selected_ids=[identity(row) for row in selected],
                selected_manifest_sha256=sha(selected_path), prior_manifest_sha256=metadata['manifest_sha256'],
                generation_protocol_sha256=sha(a.output_root / 'shared/protocol.json'),
                grid_controls={speaker: [identity(row) for row in rows] for speaker, rows in controls.items()},
                control_rule='First two other validation utterances in frozen manifest order; exclude all selected targets',
                inventory=inventory)
    score_root = a.output_root / f'ema_{a.step}' / 'diagnostic_scores'
    score_root.mkdir(parents=True, exist_ok=True)
    # Different invocations may intentionally score a different completed subset of variants.
    plan_path = score_root / ('inventory_' + '_'.join(a.variants) + '.json')
    ensure_contract(plan_path, plan)
    return selected, controls, refs, swap, inventory, score_root


def metric_contract(a, family):
    import importlib.metadata
    prior_assets = read(a.prior_root / f'ema_{a.step}/metric_assets.json')['assets']
    prior_by_path = {item['path']: item['sha256'] for item in prior_assets}
    project = Path(__file__).parents[4]
    if family == 'wavlm':
        assets = [a.wavlm_ckpt, a.wavlm_base_ckpt]
        sources = [project / 'src/f5_tts/eval/ecapa_tdnn.py']
        frontend = 'Complete waveform, arithmetic channel mean, torchaudio 16k resample; no trimming or RMS change'
    elif family == 'campplus':
        assets = [a.speaker_checkpoint]
        sources = [project / 'src/aligndit/script/misc/extract_campplus_speaker_embeddings.py']
        frontend = 'AudioFeatureDataset: channel mean, 16k, circular 10s chunks, 80-D centered fbank, mean and L2'
    else:
        assets = sorted(a.asr_ckpt.glob('*'))
        assets = [path for path in assets if str(path.resolve()) in prior_by_path]
        sources = [Path(__file__).with_name('eval_grid_s1.py'), Path(__file__).with_name('grid_zeroshot_common.py')]
        frontend = 'faster-whisper Large-v3 / float16 / English / beam 5; unchanged raw hypothesis shared by three WER variants'
    hashes = []
    for path in assets:
        digest = sha(path)
        if family in ('wavlm', 'wer') and digest != prior_by_path.get(str(path.resolve())):
            raise RuntimeError(f'Metric asset differs from prior run: {path}')
        if family == 'campplus':
            from aligndit.script.misc.extract_campplus_speaker_embeddings import EXPECTED_CHECKPOINT_SHA256
            if digest != EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError('CAM++ checkpoint differs from conditioning extractor')
        hashes.append(dict(path=str(path.resolve()), sha256=digest))
    if family == 'wer' and len(hashes) != 5:
        raise RuntimeError('Expected five frozen Whisper assets')
    sources.append(Path(__file__))
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchaudio', 'numpy', 'funasr', 'faster-whisper', 'ctranslate2')}
    contract = dict(family=family, frontend=frontend, assets=hashes, versions=versions,
                    sources=[dict(path=str(path.resolve()), sha256=sha(path)) for path in sources],
                    torch_num_threads=4, float32_matmul_precision='highest', cudnn_allow_tf32=True)
    path = a.output_root / 'diagnostic_metric_contracts' / (family + '.json')
    ensure_contract(path, contract)
    return sha(path)


def feature_paths(a, family, digest):
    base = a.output_root / 'speaker_embeddings' / family / digest
    return base.with_suffix('.npy'), base.with_suffix('.json')


def load_feature(a, family, item, contract_sha=None):
    path, meta_path = feature_paths(a, family, item['sha256'])
    meta = read(meta_path)
    if (meta['audio_sha256'] != item['sha256'] or meta['family'] != family
            or meta['embedding_sha256'] != sha(path)
            or (contract_sha is not None and meta['metric_contract_sha256'] != contract_sha)):
        raise RuntimeError('Embedding cache metadata changed')
    embedding = np.load(path, allow_pickle=False)
    expected_dim = 256 if family == 'wavlm' else 192
    if embedding.shape != (expected_dim,) or embedding.dtype != np.float32 or not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 0:
        raise RuntimeError('Invalid cached embedding')
    return embedding.astype(np.float64) / np.linalg.norm(embedding.astype(np.float64))


def embed(a, inventory):
    import torch
    import torchaudio
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = True
    unique = {item['sha256']: item for item in inventory}
    items = sorted(unique.values(), key=lambda item: item['sha256'])[a.shard::a.nshard]
    device = torch.device('cuda:0')
    for family in a.families:
        contract_sha = metric_contract(a, family)
        pending = []
        for item in items:
            path, metadata_path = feature_paths(a, family, item['sha256'])
            if path.exists() and metadata_path.exists():
                load_feature(a, family, item, contract_sha)
            else:
                pending.append(item)
        if not pending:
            print(f'{family}: all {len(items)} features already verified', flush=True)
            continue
        if family == 'wavlm':
            from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None,
                                     wavlm_ckpt_path=str(a.wavlm_base_ckpt))
            state = torch.load(a.wavlm_ckpt, weights_only=True, map_location='cpu')['model'].copy()
            state.pop('loss_calculator.projection.weight')
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device).eval()
        else:
            from aligndit.script.misc.extract_campplus_speaker_embeddings import AudioFeatureDataset, ExtractionItem, load_campplus
            model = load_campplus(a.speaker_checkpoint, device)
        with torch.inference_mode():
            for index, item in enumerate(pending, 1):
                path, metadata_path = feature_paths(a, family, item['sha256'])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    # Both checkpoint workers share controls; do not race their
                    # vector/metadata writes when they encounter the same WAV.
                    if path.exists() and metadata_path.exists():
                        load_feature(a, family, item, contract_sha)
                        continue
                    if sha(item['path']) != item['sha256']:
                        raise RuntimeError('Audio changed during extraction')
                    if family == 'wavlm':
                        waveform, rate = torchaudio.load(item['path'])
                        waveform = waveform.mean(dim=0, keepdim=True)
                        waveform = torchaudio.functional.resample(waveform, rate, 16000)
                        vector = model(waveform.to(device)).squeeze(0).float().cpu().numpy()
                    else:
                        record = ExtractionItem(item['path'], '', 'diagnosis')
                        _, features, error = AudioFeatureDataset([record])[0]
                        if error is not None:
                            raise RuntimeError(error)
                        vector_t = model(features.to(device=device, dtype=torch.float32)).float().cpu().mean(dim=0)
                        vector = torch.nn.functional.normalize(vector_t, dim=0).numpy()
                    stream = io.BytesIO()
                    np.save(stream, vector.astype(np.float32), allow_pickle=False)
                    atomic_bytes(path, stream.getvalue())
                    publish(metadata_path, dict(audio_sha256=item['sha256'], family=family,
                        embedding_sha256=sha(path), metric_contract_sha256=contract_sha,
                        source_path_at_extraction=item['path'], shape=list(vector.shape), dtype='float32'))
                    load_feature(a, family, item, contract_sha)
                print(f'{family} {index}/{len(pending)}: {item["key"]}', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def score_vector(vector, gt, refs, ref_ids, assigned_ref, swapped_ref, centroids, speaker_ids, speaker):
    ref_values = refs @ vector
    grid_values = centroids @ vector
    ref_index, speaker_index = ref_ids.index(assigned_ref), speaker_ids.index(speaker)
    other_refs = np.delete(ref_values, ref_index)
    other_grid = np.delete(grid_values, speaker_index)
    return dict(sim_gt=float(vector @ gt), sim_ref=float(ref_values[ref_index]),
                sim_swapped_reference=float(ref_values[ref_ids.index(swapped_ref)]),
                sim_mean_other_references=float(other_refs.mean()), sim_max_other_reference=float(other_refs.max()),
                assigned_reference_rank=int(1 + (ref_values > ref_values[ref_index]).sum()),
                best_reference_id=ref_ids[int(ref_values.argmax())],
                sim_same_grid_speaker_other_content=float(grid_values[speaker_index]),
                sim_mean_other_grid_speakers=float(other_grid.mean()), sim_max_other_grid_speaker=float(other_grid.max()),
                grid_speaker_rank=int(1 + (grid_values > grid_values[speaker_index]).sum()),
                best_grid_speaker=speaker_ids[int(grid_values.argmax())])


def numeric_means(records):
    numeric = [key for key, value in records[0].items() if isinstance(value, (float, int)) and not isinstance(value, bool)]
    result = {key: float(np.mean([row[key] for row in records])) for key in numeric}
    result['assigned_reference_top1_fraction'] = float(np.mean([row['assigned_reference_rank'] == 1 for row in records]))
    result['grid_speaker_top1_fraction'] = float(np.mean([row['grid_speaker_rank'] == 1 for row in records]))
    result['gt_higher_than_ref_fraction'] = float(np.mean([row['sim_gt'] > row['sim_ref'] for row in records]))
    return result


def summarize_speakers(a, selected, controls, refs, swap, inventory, output):
    by_key = {item['key']: item for item in inventory}
    ref_ids, speaker_ids = sorted(refs), sorted(controls)
    report = dict(step=a.step, count=len(selected), reference_count=len(refs), variants={}, baselines={},
                  interpretation='Paired descriptive diagnostics on the same selected targets; no significance claim',
                  centroids='Mean of two unit embeddings from other GRID validation utterances, then L2 normalized')
    all_results = {variant: [] for variant in a.variants}
    gt_results = []
    prior_results = {row['id']: row for row in read_jsonl(a.prior_root / f'ema_{a.step}/sim_results.jsonl')}
    for family in a.families:
        contract_path = a.output_root / 'diagnostic_metric_contracts' / (family + '.json')
        contract_sha = sha(contract_path)
        if read(contract_path)['sources'][-1]['sha256'] != sha(__file__):
            raise RuntimeError('Scoring implementation differs from embedding extraction')
        vectors = {key: load_feature(a, family, item, contract_sha) for key, item in by_key.items()}
        ref_matrix = np.stack([vectors['reference/' + key] for key in ref_ids])
        centers = []
        same_grid_pairs = []
        for speaker in speaker_ids:
            values = np.stack([vectors['control/' + identity(row)] for row in controls[speaker]])
            same_grid_pairs.append(float(values[0] @ values[1]))
            center = values.mean(axis=0)
            centers.append(center / np.linalg.norm(center))
        centers = np.stack(centers)
        cross_grid = centers @ centers.T
        cross_refs = ref_matrix @ ref_matrix.T
        gt_metrics, canonical_agreement, cam_condition_agreement, reconstruction_metrics = [], [], [], []
        for ref_id in ref_ids:
            canonical_agreement.append(float(vectors['reference/' + ref_id] @ vectors['reference_canonical/' + ref_id]))
            reconstruction = vectors['reference_reconstruction/' + ref_id]
            similarities = ref_matrix @ reconstruction
            ref_index = ref_ids.index(ref_id)
            reconstruction_metrics.append(dict(reference_id=ref_id, sim_ref=float(similarities[ref_index]),
                reference_rank=int(1 + (similarities > similarities[ref_index]).sum()),
                best_reference_id=ref_ids[int(similarities.argmax())],
                mean_other_references=float(np.delete(similarities, ref_index).mean())))
            if family == 'campplus':
                cached = np.load(refs[ref_id]['ref_speaker_path'], allow_pickle=False).astype(np.float64)
                cached /= np.linalg.norm(cached)
                actual = vectors['reference_canonical/' + ref_id]
                cam_condition_agreement.append(dict(reference_id=ref_id, cosine=float(actual @ cached),
                                                   max_abs_difference=float(np.abs(actual - cached).max())))
        for row in selected:
            item_id = identity(row)
            gt = vectors['gt/' + item_id]
            values = score_vector(gt, gt, ref_matrix, ref_ids, row['reference_id'], swap[row['speaker']], centers, speaker_ids, row['speaker'])
            gt_metrics.append(values)
            gt_results.append(dict(id=item_id, family=family, speaker=row['speaker'], reference_id=row['reference_id'], **values))
        reproduction_errors = []
        for variant in a.variants:
            values_per_row = []
            for row in selected:
                item_id = identity(row)
                vector = vectors[f'generated/{variant}/{item_id}']
                values = score_vector(vector, vectors['gt/' + item_id], ref_matrix, ref_ids, row['reference_id'],
                                      swap[row['speaker']], centers, speaker_ids, row['speaker'])
                if 'target_only' in a.variants:
                    values['embedding_cosine_to_target_only'] = float(vector @ vectors['generated/target_only/' + item_id])
                values_per_row.append(values)
                all_results[variant].append(dict(id=item_id, family=family, variant=variant, speaker=row['speaker'],
                    reference_id=row['reference_id'], swapped_reference_id=swap[row['speaker']], **values))
                if family == 'wavlm' and variant == 'target_only':
                    old = prior_results[item_id]
                    reproduction_errors.append(max(abs(values['sim_gt'] - old['sim']), abs(values['sim_ref'] - old['sim_ref'])))
            report['variants'].setdefault(variant, {})[family] = numeric_means(values_per_row)
        off_diagonal = ~np.eye(len(centers), dtype=bool)
        report['baselines'][family] = dict(gt=numeric_means(gt_metrics),
            same_grid_speaker_two_other_utterances=float(np.mean(same_grid_pairs)),
            different_grid_speaker_centroids=float(cross_grid[off_diagonal].mean()),
            different_references=float(cross_refs[~np.eye(len(ref_ids), dtype=bool)].mean()),
            raw_vs_canonical_reference_cosine_min=float(min(canonical_agreement)),
            raw_vs_canonical_reference_cosine_mean=float(np.mean(canonical_agreement)),
            reference_reconstruction=dict(count=len(reconstruction_metrics),
                mean_sim_ref=float(np.mean([item['sim_ref'] for item in reconstruction_metrics])),
                own_reference_top1_fraction=float(np.mean([item['reference_rank'] == 1 for item in reconstruction_metrics])),
                mean_other_references=float(np.mean([item['mean_other_references'] for item in reconstruction_metrics])),
                per_reference=reconstruction_metrics),
            campplus_condition_cache_agreement=cam_condition_agreement,
            prior_spksim_reproduction_max_abs_error=max(reproduction_errors) if reproduction_errors else None)
        if reproduction_errors and max(reproduction_errors) > 1e-6:
            raise RuntimeError('Recomputed baseline SPKSIM does not reproduce prior scores within 1e-6')
        if cam_condition_agreement and max(item['max_abs_difference'] for item in cam_condition_agreement) > 1e-3:
            raise RuntimeError('Recomputed reference CAM++ differs from actual conditioning cache')
    for variant, results in all_results.items():
        if len(results) != len(selected) * len(a.families):
            raise RuntimeError('Incomplete speaker diagnostic coverage')
        atomic_bytes(output / f'speaker_{variant}.jsonl', jsonl_bytes(results))
    atomic_bytes(output / 'speaker_gt_controls.jsonl', jsonl_bytes(gt_results))
    publish(output / 'speaker_summary.json', report)
    return report


def score_wer(a, selected, output):
    from aligndit.script.eval.eval_grid_s1 import run_wer
    contract_sha = metric_contract(a, 'wer')
    prior = {row['id']: row for row in read_jsonl(a.prior_root / f'ema_{a.step}/wer_results.jsonl')}
    tasks, pending_rows, pending_tests = [], [], []
    chosen = selected[a.shard::a.nshard]
    for variant in a.variants:
        for row in chosen:
            item_id = identity(row)
            path = output / 'wer_receipts' / variant / (item_id + '.json')
            wav = generated_path(a, variant, row)
            expected = dict(id=item_id, variant=variant, generated_audio_sha256=sha(wav), metric_contract_sha256=contract_sha)
            if path.exists():
                old = read(path)
                if any(old.get(key) != value for key, value in expected.items()) or old['wer_variants'] != wer_variants(row['raw_truth'], old['raw_hypo']):
                    raise RuntimeError('Stored WER receipt changed')
                continue
            if variant == 'target_only':
                raw = prior[item_id]
                if raw['generated_audio_sha256'] != expected['generated_audio_sha256'] or raw['raw_truth'] != row['raw_truth']:
                    raise RuntimeError('Prior ASR binding mismatch')
                publish(path, dict(expected, raw_truth=row['raw_truth'], raw_hypo=raw['raw_hypo'],
                    wer_variants=wer_variants(row['raw_truth'], raw['raw_hypo']), reused_prior_asr=True))
            else:
                tasks.append((row, path, expected))
                pending_rows.append(row)
                pending_tests.append((str(wav), str(a.cache_root / 'audio' / row['audio_relative_path']), row['raw_truth']))
    if pending_tests:
        results = run_wer(a, pending_rows, pending_tests)
        if len(results) != len(tasks):
            raise RuntimeError('Incomplete ASR coverage')
        for (row, path, expected), result in zip(tasks, results):
            if result['wav'] != Path(row['audio_relative_path']).stem or result['raw_truth'] != row['raw_truth']:
                raise RuntimeError('ASR output order changed')
            publish(path, dict(expected, raw_truth=row['raw_truth'], raw_hypo=result['raw_hypo'],
                wer_variants=wer_variants(row['raw_truth'], result['raw_hypo']), reused_prior_asr=False))
    print(f'WER shard {a.shard}/{a.nshard} complete: {len(chosen)} targets per variant', flush=True)


def summarize_wer(a, selected, output):
    scores = {}
    contract_path = a.output_root / 'diagnostic_metric_contracts/wer.json'
    contract_sha = sha(contract_path) if contract_path.exists() else None
    for variant in a.variants:
        paths = [output / 'wer_receipts' / variant / (identity(row) + '.json') for row in selected]
        if not all(path.exists() for path in paths):
            if a.require_wer:
                raise RuntimeError('Incomplete WER coverage for ' + variant)
            return None
        results = [read(path) for path in paths]
        for row, result in zip(selected, results):
            if (result['id'] != identity(row) or result['variant'] != variant
                    or result['generated_audio_sha256'] != sha(generated_path(a, variant, row))
                    or result['metric_contract_sha256'] != contract_sha):
                raise RuntimeError('WER receipt target binding changed')
            if result['wer_variants'] != wer_variants(row['raw_truth'], result['raw_hypo']):
                raise RuntimeError('Independent WER count verification failed')
        scores[variant] = summarize(results, 'wer')
        atomic_bytes(output / f'wer_{variant}.jsonl', jsonl_bytes(results))
    publish(output / 'wer_summary.json', scores)
    return scores


def args():
    prefix = Path(os.environ.get('ROOT_PREFIX', '') + '/zjw524')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, choices=(100000, 150000))
    parser.add_argument('--task', choices=('embed', 'wer', 'summarize'), required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--require-wer', action='store_true')
    parser.add_argument('--cache-root', type=Path, default=prefix / 'projects/data/GRID_svae1000k_sample_seed666_fp32')
    parser.add_argument('--wavlm-ckpt', type=Path, default=prefix / 'alignDiT_pretrain_models/wavlm_large_finetune.pth')
    parser.add_argument('--wavlm-base-ckpt', type=Path, default=prefix / 'alignDiT_pretrain_models/wavlm_large_s3prl.pt')
    parser.add_argument('--speaker-checkpoint', type=Path, default=prefix / 'projects/data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt')
    parser.add_argument('--asr-ckpt', type=Path, default=prefix / 'projects/data/faster-whisper-large-v3')
    result = parser.parse_args()
    if not 0 <= result.shard < result.nshard or len(set(result.variants)) != len(result.variants):
        parser.error('Invalid sharding or duplicate variants')
    return result


def main():
    a = args()
    selected, controls, refs, swap, inventory, output = prepare_inventory(a)
    if a.task == 'embed':
        embed(a, inventory)
    elif a.task == 'wer':
        score_wer(a, selected, output)
    else:
        report = summarize_speakers(a, selected, controls, refs, swap, inventory, output)
        report['wer'] = summarize_wer(a, selected, output)
        report['wer_complete'] = report['wer'] is not None
        publish(output / 'diagnosis_summary.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
