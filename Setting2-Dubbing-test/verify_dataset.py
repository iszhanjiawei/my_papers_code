#!/usr/bin/env python3
"""Verify checksums, portable paths, GT dialogue and cross-utterance pairs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import wave

PATH_FIELDS = {
    'video', 'mouth_video', 'video_feature', 'reference_video_feature',
    'reference_audio', 'reference_text_path', 'target_text_path',
    'target_audio', 'gt_av_feature', 'original_video',
}

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def npy_shape(path):
    # Read the standard NPY header without installing numpy or loading features.
    with path.open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError(f'Invalid NPY file: {path}')
        version = tuple(stream.read(2))
        length = int.from_bytes(stream.read(2 if version == (1, 0) else 4), 'little')
        header = ast.literal_eval(stream.read(length).decode('utf-8').strip())
    return header['shape']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    root = args.root.resolve()
    inventory = json.loads((root / 'checksums.json').read_text())['files']
    for relative, record in inventory.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing file or external link: {relative}')
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'File changed: {relative}')
    manifests = {}
    for name in ('inference.jsonl', 'evaluation.jsonl', 'aligndit_inference.jsonl'):
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        ids = [row['id'] for row in rows]
        if len(ids) != len(set(ids)) or len(ids) != 115 or len({r['reference_id'] for r in rows}) != 83:
            raise ValueError(f'Incorrect target/reference coverage: {name}')
        for row in rows:
            if row['id'] == row['reference_id']:
                raise ValueError(f'Same-utterance reference: {row["id"]}')
            for field in PATH_FIELDS.intersection(row):
                relative = Path(row[field])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Nonportable or unsafe path: {row[field]}')
                if not (root / relative).is_file():
                    raise FileNotFoundError(root / relative)
        manifests[name] = rows
    infer, evaluate, compatible = [manifests[n] for n in ('inference.jsonl', 'evaluation.jsonl', 'aligndit_inference.jsonl')]
    ids = [row['id'] for row in infer]
    if any([row['id'] for row in rows] != ids for rows in (evaluate, compatible)):
        raise ValueError('Manifest ID/order mismatch')
    frames = 0
    for row, gt, compat in zip(infer, evaluate, compatible):
        if row['reference_id'] != gt['reference_id'] or row['reference_id'] != compat['reference_id']:
            raise ValueError('Reference pairing changed')
        if any(key in row for key in ('target_audio', 'gt_av_feature', 'original_video', 'target_latent', 'pseudo_text_path')):
            raise ValueError('Forbidden generation inputs')
        if any(key in compat for key in ('target_text', 'target_text_path', 'target_audio', 'gt_av_feature', 'pseudo_text_path')):
            raise ValueError('Incorrect adapter compatibility manifest')
        if row['speaker_id'] != row['reference_speaker_id'] or row['fps'] != 25:
            raise ValueError('Speaker pairing or video rate changed')
        if row['target_text'] != (root / row['target_text_path']).read_text().strip() or row['target_text'] != gt['target_text']:
            raise ValueError('Target dialogue changed')
        if row['reference_text'] != (root / row['reference_text_path']).read_text().strip():
            raise ValueError('Reference dialogue changed')
        if npy_shape(root / row['video_feature']) != (row['num_frames'], 1024):
            raise ValueError('Video feature/frame count mismatch')
        if npy_shape(root / gt['gt_av_feature']) != (row['num_frames'], 1024):
            raise ValueError('GT AV feature/frame count mismatch')
        if row['target_samples_16khz'] != row['num_frames'] * 640:
            raise ValueError('Target duration changed')
        with wave.open(str(root / row['reference_audio'])) as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getnframes() == 0:
                raise ValueError('Expected original mono 16 kHz reference')
        frames += row['num_frames']
    if frames != 13283:
        raise ValueError('Total video duration changed')
    print(json.dumps({'complete': True, 'verified_files': len(inventory), 'targets': 115,
                      'unique_references': 83, 'video_frames': frames,
                      'duration_seconds': frames / 25, 'target_text': 'GT dialogue',
                      'external_data_dependencies': 0}, ensure_ascii=False))

if __name__ == '__main__':
    main()
