"""Auditable GRID Setting1 identities, text normalization and output coverage."""
import hashlib
import fcntl
import json
import math
import os
import re
import unicodedata
import wave
from pathlib import Path

PROTOCOL = 'grid_val_setting1_same_clip_prompt_v1'
DIGITS = dict(zip('0123456789', 'zero one two three four five six seven eight nine'.split()))


def normalize_text(text):
    """Preserve GRID letters (including 'a'); normalize formatting, not errors."""
    text = unicodedata.normalize('NFKC', text).casefold()
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    text = re.sub(r'(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])', ' ', text)
    return ' '.join(DIGITS.get(word, word) for word in text.split())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def ensure_contract(path, contract):
    """Publish once under a file lock; other GPU workers only validate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if read(path) != contract:
                raise RuntimeError('Output directory belongs to another experiment')
        else:
            publish(path, contract)
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
        fcntl.flock(lock, fcntl.LOCK_UN)


def load_rows(cache):
    cache = Path(cache)
    if (cache / 'WRITE_ACTIVE.json').exists():
        raise RuntimeError('GRID cache has an active writer')
    meta = read(cache / 'manifests/inventory_meta.json')
    done = read(cache / 'complete.json')
    if (done.get('complete') is not True or done.get('missing') != [] or done.get('invalid') != []
            or done.get('selection') != {'mode': 'full'} or done.get('split_counts', {}).get('val') != 3274
            or done.get('inventory_meta_sha256') != sha(cache / 'manifests/inventory_meta.json')):
        raise RuntimeError('A verified complete GRID cache with 3274 validation clips is required')
    partitions = {}
    for name in ('inventory.jsonl', 'train.jsonl', 'val.jsonl'):
        path = cache / 'manifests' / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        entry = meta['manifests'][name]
        if entry['sha256'] != sha(path) or entry['count'] != len(rows):
            raise RuntimeError(f'GRID manifest changed: {name}')
        partitions[name] = rows
    rows = partitions['val.jsonl']
    ids = [identity(r) for r in rows]
    if len(rows) != 3274 or len(set(ids)) != len(ids):
        raise RuntimeError('Invalid validation coverage or duplicate full identities')
    if {r['utterance_key'] for r in rows} & {r['utterance_key'] for r in partitions['train.jsonl']}:
        raise RuntimeError('Train and validation overlap')
    for r in rows:
        if len(normalize_text(r['text']).split()) != 6:
            raise RuntimeError('Expected six GRID reference words')
    return rows, done


def identity(row):
    path = Path(row['audio_relative_path'])
    if path.is_absolute() or len(path.parts) != 3 or path.parts[0] != 'val' or '..' in path.parts or path.suffix != '.wav':
        raise ValueError('Unsafe validation identity')
    if row['utterance_key'] != 'grid/' + path.with_suffix('').relative_to('val').as_posix():
        raise ValueError('GRID identity mismatch')
    return path.with_suffix('').as_posix()


def output_record(root, row, protocol=PROTOCOL):
    path = Path(root) / row['audio_relative_path']
    receipt = read(Path(root) / 'receipts' / (identity(row) + '.json'))
    contract = read(Path(root) / 'protocol.json')
    if receipt.get('protocol_sha256') != sha(Path(root) / 'protocol.json') or receipt.get('id') != identity(row):
        raise RuntimeError(f'Stale inference receipt: {path}')
    if receipt.get('sha256') != sha(path):
        raise RuntimeError(f'Generated WAV changed: {path}')
    with wave.open(str(path)) as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) != (16000, 1, 2, row['original_num_samples']):
            raise RuntimeError(f'Incorrect WAV format/duration: {path}')
    if contract.get('protocol') != protocol:
        raise RuntimeError('Wrong evaluation protocol')
    return receipt


def verify_outputs(cache, root, limit=None, protocol=PROTOCOL):
    rows, _ = load_rows(cache)
    pair_map = reference_rows = None
    if protocol == 'grid_val_setting2_same_speaker_cross_utterance_gttext_v1':
        from aligndit.script.eval.grid_s2_common import load_pairs
        pair_map, reference_rows = load_pairs(Path(cache), Path(root), rows)
        if read(Path(root)/'protocol.json').get('reference_pairs_sha256') != sha(Path(root)/'reference_pairs.json'):
            raise RuntimeError('Setting2 pairs differ from inference protocol')
    if limit is not None:
        rows = rows[:limit]
    outputs = [output_record(root, row, protocol) for row in rows]
    if pair_map is not None:
        for row, receipt in zip(rows, outputs):
            ref = reference_rows[identity(row)]
            if (receipt.get('reference_id') != pair_map[identity(row)]['reference_id']
                    or receipt.get('prompt_frames') != ref['latent_frames']
                    or receipt.get('target_frames') != row['latent_frames']
                    or receipt.get('sampler_frames') != ref['latent_frames']+row['latent_frames']):
                raise RuntimeError('Setting2 output reference or prompt/target geometry mismatch')
    actual = {p.relative_to(root).as_posix() for p in Path(root).glob('val/*/*.wav')}
    expected = {r['audio_relative_path'] for r in rows}
    if actual != expected:
        raise RuntimeError('Generated WAV coverage differs from requested validation list')
    summary = {'protocol': protocol, 'count': len(rows), 'complete': limit is None,
               'partial_smoke_test': limit is not None, 'outputs': outputs,
               'protocol_sha256': sha(Path(root) / 'protocol.json')}
    publish(Path(root) / 'inference_summary.json', summary)
    return rows, summary


def metric_summary(results, task):
    if not results or len({r['id'] for r in results}) != len(results):
        raise ValueError('Empty or duplicate metric identities')
    if not all(math.isfinite(r[task]) for r in results):
        raise ValueError('Nonfinite metric values')
    if task == 'wer':
        errors = sum(r['word_edit_distance'] for r in results)
        words = sum(r['reference_word_count'] for r in results)
        return {'count': len(results), 'wer': errors / words, 'word_edit_distance': errors,
                'reference_word_count': words, 'aggregation': 'corpus_word_edit_distance'}
    return {'count': len(results), task: math.fsum(r[task] for r in results) / len(results), 'aggregation': 'sample_mean'}
