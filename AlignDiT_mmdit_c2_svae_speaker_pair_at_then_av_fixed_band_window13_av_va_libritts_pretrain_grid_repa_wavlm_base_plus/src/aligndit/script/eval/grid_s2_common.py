"""Deterministic same-speaker different-text validation reference assignments."""
import hashlib
from collections import defaultdict
from pathlib import Path

from aligndit.script.eval.grid_s1_common import identity, load_rows, normalize_text, publish, read, sha

PROTOCOL = 'grid_val_setting2_same_speaker_cross_utterance_gttext_v1'
SELECTION = 'val_same_speaker_sha256_seed0_cycle_first_different_text_shift_v1'


def select_references(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['speaker']].append(row)
    references = {}
    for speaker, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: (hashlib.sha256(f'0:{identity(r)}'.encode()).digest(), identity(r)))
        if len(ordered) < 2:
            raise ValueError(f'No independent reference for {speaker}')
        for shift in range(1, len(ordered)):
            if all(normalize_text(r['text']) != normalize_text(ordered[(i+shift)%len(ordered)]['text'])
                   for i, r in enumerate(ordered)):
                for i, row in enumerate(ordered):
                    references[identity(row)] = ordered[(i+shift)%len(ordered)]
                break
        else:
            raise ValueError(f'Cannot form distinct-text reference permutation for {speaker}')
    return references


def pair_records(cache, rows):
    references = select_references(rows)
    pairs = []
    for row in rows:
        ref = references[identity(row)]
        pairs.append({'id': identity(row), 'reference_id': identity(ref), 'speaker': row['speaker'],
                      'reference_speaker': ref['speaker'], 'reference_text': ref['text'],
                      'reference_audio_relative_path': ref['audio_relative_path'],
                      'reference_source_sha256': ref['source_sha256'],
                      'reference_audio_sha256': read(cache/'receipts/audio'/Path(ref['audio_relative_path']).with_suffix('.json'))['artifact_sha256'],
                      'reference_latent_sha256': read(cache/'receipts/latent'/Path(ref['audio_relative_path']).with_suffix('.json'))['artifact_sha256'],
                      'reference_speaker_sha256': read(cache/'receipts/speaker'/Path(ref['audio_relative_path']).with_suffix('.json'))['artifact_sha256']})
    return pairs


def prepare_pairs(cache, root):
    rows, _ = load_rows(cache)
    pairs = pair_records(cache, rows)
    path = root/'reference_pairs.json'
    contents = {'protocol': PROTOCOL, 'selection': SELECTION, 'seed': 0, 'count': len(rows),
                'target_manifest_sha256': sha(cache/'manifests/val.jsonl'), 'pairs': pairs}
    if path.exists() and read(path) != contents:
        raise RuntimeError('Reference assignments changed; use a new output directory')
    publish(path, contents)
    return pairs


def load_pairs(cache, root, rows):
    saved = read(root/'reference_pairs.json')
    if (saved.get('protocol') != PROTOCOL or saved.get('selection') != SELECTION or saved.get('count') != 3274
            or saved.get('target_manifest_sha256') != sha(cache/'manifests/val.jsonl')):
        raise RuntimeError('Invalid Setting2 reference manifest')
    by_id = {identity(row): row for row in rows}
    if len(saved['pairs']) != len(rows) or len({p['id'] for p in saved['pairs']}) != len(rows):
        raise RuntimeError('Incomplete or duplicate reference assignment')
    chosen = select_references(rows)
    for row, pair in zip(rows, saved['pairs']):
        if pair['id'] != identity(row) or pair['reference_id'] != identity(chosen[identity(row)]):
            raise RuntimeError('Setting2 reference identity/order mismatch')
        ref = by_id[pair['reference_id']]
        if (ref['speaker'] != row['speaker'] or pair['reference_speaker'] != ref['speaker']
                or pair['reference_audio_relative_path'] != ref['audio_relative_path']
                or pair['reference_text'] != ref['text'] or pair['reference_source_sha256'] != ref['source_sha256']
                or normalize_text(ref['text']) == normalize_text(row['text'])):
            raise RuntimeError('Reference is not a verified different-text utterance of the target speaker')
    return {p['id']: p for p in saved['pairs']}, chosen
