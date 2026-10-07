"""Resolve portable input manifests for existing adapters; uses only stdlib."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir or root / 'resolved'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.jsonl', 'inference.jsonl', 'evaluation.jsonl'):
        records = []
        for line in (root / name).read_text().splitlines():
            row = json.loads(line)
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
                    row[field] = str(path)
            records.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        (output / name).write_text('\n'.join(records) + '\n')
    print(output.resolve())

if __name__ == '__main__':
    main()
