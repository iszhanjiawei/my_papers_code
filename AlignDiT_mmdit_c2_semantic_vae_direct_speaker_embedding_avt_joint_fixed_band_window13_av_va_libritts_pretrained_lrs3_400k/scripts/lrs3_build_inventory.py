"""Freeze the audited LRS3 source inventory and a deterministic stratified pilot.

This is a source inventory, not a declaration that training caches are complete.
Original sources and candidate transcripts are never changed by this script.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(path)


def clean_text(text):
    text = re.sub(r'\{(?:lg|ns|pc|tc)\}', ' ', text, flags=re.I)
    return ' '.join(text.lower().split())


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(f'666:{key}'.encode()).digest()[:8], 'little') % (2**63 - 1)


def ordered(rows):
    return sorted(rows, key=lambda r: hashlib.sha256(('pilot666:' + r['sample_id']).encode()).digest())


def main():
    prefix = os.environ.get('ROOT_PREFIX', '')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(prefix + '/zjw524/datasets/LRS3'))
    p.add_argument('--output-root', type=Path, default=Path(prefix + '/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v1'))
    p.add_argument('--vocab', type=Path, default=Path(prefix + '/zjw524/projects/data/CelebVDub_char/vocab.txt'))
    args = p.parse_args()
    out = args.output_root
    source = args.source_root / 'DARKNSU_extracted'
    inputs = [source / 'audit/all_pairs_manifest.tsv',
              source / 'audit/pipeline_validated_label_candidates.tsv',
              source / 'audit/direct_label_candidates.tsv',
              source / 'audit/test_labels/direct_label_candidates.tsv',
              args.source_root / 'verification/lrs3-valid.id.avhubert',
              source / 'annotations/USR_TEXT_RECOVERY_CANDIDATES.jsonl', args.vocab]
    identities = {str(f): digest(f) for f in inputs}
    marker = out / 'reports/source_contract.json'
    if marker.exists() and json.loads(marker.read_text())['input_sha256'] != identities:
        raise RuntimeError('Source hashes changed; use a new versioned output root')
    vocab = {c: i for i, c in enumerate(args.vocab.read_text().splitlines())}
    assert vocab[' '] == 0 and len(vocab) == 159
    valid = set(inputs[4].read_text().splitlines())
    assert len(valid) == 1200
    labels = {}
    # Earlier sources have priority. Preserve external text provenance explicitly.
    for f in inputs[1:4]:
        with f.open() as h:
            for r in csv.DictReader(h, delimiter='\t'):
                if not r.get('text', '').strip():
                    continue
                labels.setdefault(r['video_path'], r)
    rows = []
    seen = set()
    missing_groups = collections.defaultdict(list)
    with inputs[0].open() as h:
        for s in csv.DictReader(h, delimiter='\t'):
            sample_id = Path(s['video_path']).stem
            if sample_id in seen:
                raise ValueError(f'Duplicate media identity: {sample_id}')
            seen.add(sample_id)
            original_id = s['original_clip_id']
            split = 'test' if s['split'] == 'test' else 'valid' if original_id in valid else 'train'
            relative = f'{split}/{s["split"]}/{sample_id}'
            key = f'lrs3/{relative}'
            label = labels.get(s['video_path'], {})
            raw_text = label.get('text', '')
            text = clean_text(raw_text)
            token_ids = [vocab.get(c, 0) for c in text]
            repeats = sum(a == b for a, b in zip(token_ids, token_ids[1:]))
            samples = round(float(s['audio_seconds']) * 16000)
            frames = (samples + 399) // 400
            status = 'original_pipeline' if label.get('validation_status') == 'pipeline_segment_set_frames_and_audio_samples_validated' else 'external_direct' if text else 'missing'
            r = dict(sample_id=sample_id, utterance_key=key, relative_stem=relative,
                     split=split, source_split=s['split'], original_clip_id=original_id,
                     segment_id=s['segment_id'], video_path=s['video_path'], audio_path=s['audio_path'],
                     video_frames=int(s['frames']), width=int(s['width']), height=int(s['height']),
                     video_fps=25, video_seconds=float(s['video_seconds']), audio_seconds=samples / 16000,
                     duration_seconds=samples / 16000, sample_rate=16000, num_channels=1,
                     source_sample_rate=16000, source_num_channels=1, source_num_samples=samples,
                     original_num_samples=samples, padded_num_samples=frames * 400,
                     latent_frames=frames, latent_dim=64, video_dim=1024,
                     posterior_seed=stable_seed(key), subset=s['split'],
                     text=text, original_label_text=raw_text, text_status=status,
                     label_source=label.get('label_source', label.get('text_source', '')),
                     label_validation=label.get('validation_status', label.get('label_status', '')),
                     independent_content_verified=False,
                     transcript_path=label.get('original_txt_path', label.get('original_transcript_path', '')),
                     source_segment_start_seconds=label.get('segment_start_seconds'),
                     source_segment_end_seconds=label.get('segment_end_seconds'),
                     ctc_target_length=len(token_ids), ctc_adjacent_repeats=repeats,
                     ctc_min_input_frames=len(token_ids)+repeats,
                     ctc_feasible_40hz=bool(text) and frames >= len(token_ids)+repeats,
                     oov_characters=sorted(set(text) - set(vocab)),
                     duration_eligible=0.4 <= samples / 16000 <= 30,
                     audio_relative_path=f'audio/{relative}.wav',
                     mouth_relative_path=f'video_mouth/{relative}.mp4',
                     video_25hz_relative_path=f'avhubert_video_25hz/{relative}.npy',
                     latent_relative_path=f'latents/{relative}.npy',
                     video_40hz_relative_path=f'video_40hz/{relative}.npy',
                     speaker_relative_path=f'speaker_embeddings/{relative}.npy')
            rows.append(r)
            if status == 'missing':
                missing_groups[original_id].append(r)
    assert len(rows) == 302167
    assert len({r['original_clip_id'] for r in rows}) == 151819
    assert len(missing_groups) == 3406
    assert sum(r['split'] == 'valid' for r in rows) == 1200
    assert sum(r['split'] == 'test' for r in rows) == 1321
    pilot = []
    for split, width, count in [('train',224,128),('train',96,96),('valid',96,16),('test',96,16)]:
        pool = ordered([r for r in rows if r['split']==split and r['width']==width and r['text'] and r['duration_eligible'] and not r['oov_characters']])
        # Cycle duration strata to cover both short clips and long clips.
        buckets = [[r for r in pool if lo <= r['audio_seconds'] < hi]
                   for lo,hi in [(0.4,2),(2,5),(5,12),(12,30.01)]]
        for i in range(count):
            available = [b for b in buckets if b]
            pilot.append(available[i % len(available)].pop(0))
    with inputs[5].open() as f:
        candidates = [json.loads(l) for l in f]
    selected = [r['original_clip_id'] for r in candidates if r.get('previously_no_Llama_text')]
    for source_kind, count in [('trim600',10), ('pretrain',9)]:
        pool = sorted([r for r in candidates if source_kind in r['source_splits'] and r['original_clip_id'] not in selected
                       and all(x['duration_eligible'] for x in missing_groups[r['original_clip_id']])],
                      key=lambda r: hashlib.sha256(('pilot666:'+r['original_clip_id']).encode()).digest())
        selected.extend(r['original_clip_id'] for r in pool[:count])
    for key in selected:
        pilot.extend(missing_groups[key])
    pilot.sort(key=lambda r: r['utterance_key'])
    assert len({r['sample_id'] for r in pilot}) == len(pilot)
    for r in pilot:
        dst = out / r['audio_relative_path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            assert dst.resolve() == Path(r['audio_path']).resolve()
        elif not dst.exists():
            dst.symlink_to(r['audio_path'])
        else:
            raise FileExistsError(dst)
    write_jsonl(out/'manifests/source_inventory.jsonl', rows)
    write_jsonl(out/'manifests/pilot.jsonl', pilot)
    write_json(marker, {'schema_version':1,'input_sha256':identities,'source_media_pairs':len(rows),
                        'source_inventory_sha256':digest(out/'manifests/source_inventory.jsonl'),
                        'original_clip_count':151819,'purpose':'audited source inventory; not training-ready',
                        'base_posterior_seed':666,'normalization':'pinned LibriSpeech train statistics',
                        'vocab_sha256':digest(args.vocab)})
    report = {'count':len(pilot),'split_counts':dict(collections.Counter(r['split'] for r in pilot)),
              'dimension_counts':dict(collections.Counter(str(r['width']) for r in pilot)),
              'text_status_counts':dict(collections.Counter(r['text_status'] for r in pilot)),
              'recovery_original_ids':selected, 'audio_hours':sum(r['audio_seconds'] for r in pilot)/3600,
              'source_inventory_sha256':digest(out/'manifests/source_inventory.jsonl'),
              'pilot_sha256':digest(out/'manifests/pilot.jsonl'),
              'training_ready':False}
    write_json(out/'reports/pilot_selection.json', report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
