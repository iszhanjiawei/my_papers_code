"""Full-coverage GRID Setting1 metrics; one worker process per metric shard."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from aligndit.script.eval.grid_s1_common import (
    PROTOCOL, identity, metric_summary, normalize_text, output_record, publish, read, sha, verify_outputs,
)
from aligndit.script.eval.eval_celebvdub_grid_reference import word_edit_counts


def audit_assets(a):
    """Record local model binaries and source hashes without loading GPU models."""
    import importlib.metadata
    import subprocess
    files = [a.wavlm_ckpt, a.wavlm_base_ckpt, a.avhubert_checkpoint,
             a.asr_ckpt/'model.bin', a.asr_ckpt/'config.json', a.asr_ckpt/'tokenizer.json',
             a.asr_ckpt/'vocabulary.json', a.asr_ckpt/'preprocessor_config.json',
             a.emo_ckpt/'model.pt', a.emo_ckpt/'config.yaml', a.emo_ckpt/'configuration.json']
    project = Path(__file__).parents[4]
    sources = [Path(__file__), Path(__file__).with_name('grid_s1_common.py'),
               project/'src/f5_tts/eval/utils_eval.py', Path(__file__).with_name('utils.py'),
               project/'src/aligndit/script/misc/extract_avhubert.py']
    if a.setting == 's2':
        sources += [Path(__file__).with_name('grid_s2_common.py'), Path(__file__).with_name('infer_grid_s2.py')]
    assets = [{'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)} for path in files]
    versions = {}
    for package in ('torch', 'torchaudio', 'numpy', 'faster-whisper', 'ctranslate2', 'funasr', 'jiwer'):
        versions[package] = importlib.metadata.version(package)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    publish(a.output_dir/'metric_assets.json', {'assets': assets, 'code_revision': revision, 'versions': versions,
            'source_files': [{'path': str(path), 'sha256': sha(path)} for path in sources]})
    print('Recorded local metric assets and source hashes', flush=True)


def save_results(root, task, rows, results, shard, nshard):
    if len(results) != len(rows):
        raise RuntimeError('Incomplete metric worker results')
    identified = []
    for row, result in zip(rows, results):
        if result.get('wav') != Path(row['audio_relative_path']).stem:
            raise RuntimeError('Metric worker order mismatch')
        identified.append({**result, 'wav': identity(row), 'id': identity(row)})
    summary = metric_summary(identified, task)
    path = root / f'{task}_shard{shard}.jsonl'
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in identified))
    os.replace(tmp, path)
    publish(root / f'{task}_shard{shard}_summary.json', {**summary, 'nshard': nshard,
            'inference_summary_sha256': sha(root / 'inference_summary.json')})
    print(json.dumps(summary), flush=True)


def run_wer(a, rows, tests):
    from faster_whisper import WhisperModel
    from tqdm import tqdm
    import jiwer
    # All shards remain independent; serialize model allocation and wait for
    # enough actual free memory when other workloads share this GPU.
    import fcntl
    import subprocess
    import time
    physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    lock_path = Path('/tmp')/f'grid_whisper_gpu_{physical_gpu}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            free_mib = int(subprocess.check_output(
                ['nvidia-smi', '--id='+physical_gpu, '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'], text=True).strip())
            if free_mib >= 6000:
                break
            print(f'Waiting for Whisper GPU memory: {free_mib} MiB free (need 6000 MiB)', flush=True)
            time.sleep(5)
        model = WhisperModel(str(a.asr_ckpt), device='cuda', device_index=0, compute_type='float16', cpu_threads=4)
        # Materialize decoder buffers before another shard checks available memory.
        warmup_segments, _ = model.transcribe(tests[0][0], language='en', beam_size=5)
        list(warmup_segments)
        fcntl.flock(lock, fcntl.LOCK_UN)
    results = []
    for row, (wav, ref, text) in tqdm(list(zip(rows, tests)), mininterval=5):
        segments, _ = model.transcribe(wav, language='en', beam_size=5)
        raw = ' '.join(s.text.strip() for s in segments)
        truth, hypo = normalize_text(text), normalize_text(raw)
        counts = jiwer.process_words(truth, hypo)
        errors, words = word_edit_counts(truth, hypo)
        if errors != counts.substitutions + counts.deletions + counts.insertions:
            raise RuntimeError('Independent WER count recomputation disagrees')
        results.append({'wav': Path(wav).stem, 'raw_truth': text, 'raw_hypo': raw, 'truth': truth, 'hypo': hypo,
                        'wer': errors/words, 'word_edit_distance': errors, 'reference_word_count': words,
                        'substitutions': counts.substitutions, 'deletions': counts.deletions, 'insertions': counts.insertions})
    return results


def run_avsync(a, rows, tests):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from aligndit.script.misc.grid_sources import load_mouth_frames, mouth_frame_paths, _mouth_digest
    sys.path.insert(0, str(a.fairseq_repo))
    sys.path.insert(0, str(a.avhubert_user_dir))
    import fairseq
    fairseq.utils.import_user_module(SimpleNamespace(user_dir=str(a.avhubert_user_dir)))
    from aligndit.script.misc.extract_avhubert import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    expected = '343c9dbaa29847b8c4d0d9503c43f2877fd2efde9988d41abf14e74f28232b75'
    if sha(a.avhubert_checkpoint) != expected:
        raise RuntimeError('AV-HuBERT checkpoint differs from pinned protocol')

    class MouthReader(AVHubertFeatureReader):
        def load_image(self, path):
            return self.images

    reader = MouthReader(str(a.avhubert_checkpoint), custom_utils=avhubert_utils)
    results = []
    for row, (wav, ref, _) in tqdm(list(zip(rows, tests)), mininterval=5):
        frame_root = Path(row['source']['mouth_dir'])
        frame_names = [p.name for p in sorted(frame_root.glob('*.jpg'))]
        if frame_names != row['source']['mouth_frame_names'] or _mouth_digest(mouth_frame_paths(frame_root)) != row['source']['mouth_sha256']:
            raise RuntimeError('Mouth frame identity changed')
        reader.images = reader.transform(load_mouth_frames(row['source']))[..., None]
        features = []
        for role, audio in (('gen', wav), ('gt', ref)):
            path = a.output_dir / 'avhubert_joint' / role / (identity(row) + '.npy')
            receipt_path = path.with_suffix('.json')
            binding = {'id': identity(row), 'audio_sha256': sha(audio), 'mouth_sha256': row['source']['mouth_sha256'],
                       'checkpoint_sha256': expected, 'feature': 'joint_audio_video_25hz_finetune_1024'}
            if path.exists() or receipt_path.exists():
                receipt = read(receipt_path)
                if any(receipt.get(k) != v for k, v in binding.items()) or receipt.get('sha256') != sha(path):
                    raise RuntimeError('Stale AVSync feature cache')
                feat = np.load(path, allow_pickle=False)
            else:
                feat = reader.get_feats((str(frame_root), audio)).detach().cpu().numpy()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f:
                    np.save(f, feat, allow_pickle=False)
                os.replace(tmp, path)
                publish(receipt_path, {**binding, 'sha256': sha(path)})
            if feat.shape != (row['source_video_frames'], 1024) or not np.isfinite(feat).all():
                raise RuntimeError('Invalid AVSync joint feature')
            features.append(torch.from_numpy(feat))
        sim = F.cosine_similarity(features[0], features[1], dim=1).mean().item()
        results.append({'wav': Path(wav).stem, 'avsync': sim})
    return results


def protocol(a):
    if a.setting == 's2':
        from aligndit.script.eval.grid_s2_common import PROTOCOL as s2_protocol
        return s2_protocol
    return PROTOCOL


def merge(a, rows):
    report = {'protocol': protocol(a), 'count': len(rows), 'complete': a.limit is None,
              'partial_smoke_test': a.limit is not None, 'checkpoint': read(a.output_dir/'protocol.json')['checkpoint'],
              'metrics': {}, 'wer_normalization': 'NFKC, casefold, Unicode punctuation to spaces, whitespace collapse, isolated digits 0..9 to English words; preserve all GRID letter words',
              'references': {'sim': 'same-clip canonical GT audio', 'emosim': 'emotion2vec classifier-score cosine vs canonical GT',
                             'avsync': 'mean frame cosine of joint AV-HuBERT features: same mouth video + generated vs canonical GT audio'}}
    if a.setting == 's2':
        report['text_source'] = 'supplied_target_GT_dialogue; no VSR'
        report['references']['sim'] = 'same-speaker different-text reference audio used by generation'
        report['reference_pairs_sha256'] = sha(a.output_dir/'reference_pairs.json')
    if (a.output_dir/'metric_assets.json').exists():
        report['metric_assets'] = {'path': str(a.output_dir/'metric_assets.json'), 'sha256': sha(a.output_dir/'metric_assets.json')}
    expected = [identity(r) for r in rows]
    for task in ('sim', 'wer', 'emosim', 'avsync'):
        gathered = []
        for shard in range(a.nshard):
            summary = read(a.output_dir / f'{task}_shard{shard}_summary.json')
            if summary['nshard'] != a.nshard or summary['inference_summary_sha256'] != sha(a.output_dir/'inference_summary.json'):
                raise RuntimeError('Metric shard protocol mismatch')
            results = [json.loads(line) for line in (a.output_dir/f'{task}_shard{shard}.jsonl').read_text().splitlines()]
            if [r['id'] for r in results] != expected[shard::a.nshard] or metric_summary(results, task) != {k: summary[k] for k in metric_summary(results, task)}:
                raise RuntimeError('Metric shard coverage or summary mismatch')
            if task == 'wer':
                for result in results:
                    errors, words = word_edit_counts(normalize_text(result['raw_truth']), normalize_text(result['raw_hypo']))
                    if (errors, words) != (result['word_edit_distance'], result['reference_word_count']):
                        raise RuntimeError('Corpus WER independent recomputation failed')
            gathered.extend(results)
        by_id = {r['id']: r for r in gathered}
        if len(gathered) != len(rows) or set(by_id) != set(expected):
            raise RuntimeError('Incomplete full metric coverage')
        gathered = [by_id[key] for key in expected]
        (a.output_dir / f'{task}_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in gathered))
        summary = metric_summary(gathered, task)
        summary['display_value'] = f'{summary[task]:.5f}'
        if task == 'wer' and summary['reference_word_count'] != 6*len(rows):
            raise RuntimeError('Unexpected GRID corpus WER denominator')
        report['metrics'][task] = summary
    publish(a.output_dir/'evaluation_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run(a):
    if a.task == 'audit':
        audit_assets(a)
        return
    # Full verification is performed once before metric shards, and again at merge.
    if a.task in ('verify', 'merge'):
        rows, _ = verify_outputs(a.cache_root, a.output_dir, a.limit, protocol(a))
        if a.task == 'merge':
            merge(a, rows)
        print(f'Verified {len(rows)} generated clips', flush=True)
        return
    from aligndit.script.eval.grid_s1_common import load_rows
    rows, _ = load_rows(a.cache_root)
    pairs = None
    if a.setting == 's2':
        from aligndit.script.eval.grid_s2_common import load_pairs
        pairs, _ = load_pairs(a.cache_root, a.output_dir, rows)
        if read(a.output_dir/'protocol.json').get('reference_pairs_sha256') != sha(a.output_dir/'reference_pairs.json'):
            raise RuntimeError('Reference pairing differs from inference protocol')
    if a.limit is not None:
        rows = rows[:a.limit]
    summary = read(a.output_dir/'inference_summary.json')
    if (summary['count'] != len(rows) or summary['partial_smoke_test'] != (a.limit is not None)
            or summary['protocol'] != protocol(a)):
        raise RuntimeError('Metrics require verified inference for exactly this selection')
    if not 0 <= a.shard < a.nshard <= len(rows):
        raise ValueError('Invalid metric shard')
    rows = rows[a.shard::a.nshard]
    import torch
    torch.set_num_threads(4)
    for row in rows:
        output_record(a.output_dir, row, protocol(a))
    tests = [(str(a.output_dir/r['audio_relative_path']), str(a.cache_root/'audio'/r['audio_relative_path']), r['text']) for r in rows]
    if a.task == 'sim' and pairs is not None:
        tests = [(wav, str(a.cache_root/'audio'/pairs[identity(row)]['reference_audio_relative_path']), text)
                 for row, (wav, _, text) in zip(rows, tests)]
        for row, (_, ref, _) in zip(rows, tests):
            if sha(ref) != pairs[identity(row)]['reference_audio_sha256']:
                raise RuntimeError('Setting2 speaker reference audio changed')
    if a.task == 'sim':
        from f5_tts.eval.utils_eval import run_sim
        results = run_sim((0, tests, str(a.wavlm_ckpt), str(a.wavlm_base_ckpt)))
    elif a.task == 'wer':
        results = run_wer(a, rows, tests)
    elif a.task == 'emosim':
        from aligndit.script.eval.utils import run_emosim
        results = run_emosim((0, tests, str(a.emo_ckpt)))
    else:
        results = run_avsync(a, rows, tests)
    if a.task == 'sim' and pairs is not None:
        for row, result in zip(rows, results):
            pair = pairs[identity(row)]
            result.update(reference_id=pair['reference_id'], reference_audio_relative_path=pair['reference_audio_relative_path'],
                          reference_audio_sha256=pair['reference_audio_sha256'])
    save_results(a.output_dir, a.task, rows, results, a.shard, a.nshard)


def args():
    prefix = os.environ.get('ROOT_PREFIX', '')
    data = Path(prefix+'/zjw524/projects/data')
    pretrain = Path(prefix+'/zjw524/alignDiT_pretrain_models')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting', choices=['s1', 's2'], default='s1')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=data/'GRID_svae1000k_sample_seed666_fp32')
    p.add_argument('--task', choices=['audit', 'verify', 'merge', 'sim', 'wer', 'emosim', 'avsync'], required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', '-n', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--wavlm-ckpt', type=Path, default=pretrain/'wavlm_large_finetune.pth')
    p.add_argument('--wavlm-base-ckpt', type=Path, default=pretrain/'wavlm_large_s3prl.pt')
    p.add_argument('--asr-ckpt', type=Path, default=data/'faster-whisper-large-v3')
    p.add_argument('--emo-ckpt', type=Path, default=data/'emotion2vec_plus_large')
    p.add_argument('--avhubert-checkpoint', type=Path, default=data/'large_vox_iter5.pt')
    p.add_argument('--fairseq-repo', type=Path, default=data/'av_hubert/fairseq/fairseq')
    p.add_argument('--avhubert-user-dir', type=Path, default=data/'av_hubert/avhubert/avhubert')
    return p.parse_args()


if __name__ == '__main__':
    run(args())
