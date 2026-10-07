"""Verify every packaged file and the frozen cross-reference protocol."""
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    root = Path(__file__).resolve().parent
    index = json.loads((root / 'checksums.json').read_text())
    for relative, record in index['files'].items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or linked asset: {relative}')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Asset changed: {relative}')
    id_order = None
    for name in ('manifest.jsonl', 'inference.jsonl', 'evaluation.jsonl'):
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        ids = [row['id'] for row in rows]
        if len(ids) != 115 or len(set(ids)) != 115:
            raise ValueError(f'Invalid target coverage: {name}')
        if id_order is not None and ids != id_order:
            raise ValueError(f'Target order mismatch: {name}')
        id_order = ids
        if len({row['reference_id'] for row in rows}) != 83:
            raise ValueError(f'Invalid reference coverage: {name}')
        for row in rows:
            if row['id'] == row['reference_id']:
                raise ValueError('Same-utterance reference in Setting 2')
            if name == 'inference.jsonl' and any(k in row for k in ('target_audio', 'target_text', 'target_text_path', 'gt_av_feature')):
                raise ValueError('Target GT leaked into inference manifest')
            for field, value in row.items():
                if isinstance(value, str) and field in {
                    'video', 'mouth_video', 'target_audio', 'reference_audio',
                    'original_audio_path', 'original_video', 'original_mouth_video',
                    'target_text_path', 'reference_text_path', 'pseudo_text_path',
                    'video_feature', 'reference_video_feature', 'gt_av_feature',
                }:
                    path = (root / value).resolve()
                    path.relative_to(root)
                    if not path.is_file():
                        raise FileNotFoundError(path)
    print(json.dumps({'verified_files': len(index['files']), 'targets': 115, 'unique_references': 83, 'complete': True}))

if __name__ == '__main__':
    main()
