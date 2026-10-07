#!/usr/bin/env python3
"""Resolve portable dataset manifests on the current server (standard library)."""
import argparse
import json
from pathlib import Path

PATH_FIELDS = {
    'video', 'mouth_video', 'video_feature', 'reference_video_feature',
    'reference_audio', 'reference_text_path', 'target_text_path',
    'target_audio', 'gt_av_feature', 'original_video',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--generated-root', type=Path,
                        help='Optional completed generation root containing test/<video>/<clip>.wav; also exports LSE pairs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir or root / 'resolved'
    output.mkdir(parents=True, exist_ok=True)
    evaluation = []
    for name in ('inference.jsonl', 'evaluation.jsonl', 'aligndit_inference.jsonl'):
        rows = []
        for line in (root / name).read_text().splitlines():
            row = json.loads(line)
            for field in PATH_FIELDS.intersection(row):
                path = (root / row[field]).resolve()
                path.relative_to(root)
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[field] = str(path)
            rows.append(row)
        (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
        if name == 'evaluation.jsonl':
            evaluation = rows
    if args.generated_root:
        generated = args.generated_root.resolve()
        lse = []
        for row in evaluation:
            audio = generated / (row['id'] + '.wav')
            if not audio.is_file():
                raise FileNotFoundError(audio)
            # SyncNet uses original face video, never the mouth crop. Original
            # video may contain GT audio; the evaluator must use explicit audio.
            lse.append({'id': row['id'], 'video': row['original_video'], 'audio': str(audio)})
        (output / 'lse_manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in lse))
    print(output.resolve())

if __name__ == '__main__':
    main()
