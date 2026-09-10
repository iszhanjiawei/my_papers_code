"""Check complete clip identities and independently aggregate historical S1 metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from jiwer import compute_measures


def validate(root, stage, list_path=Path("data/celebvdub_test_s1.lst"), split="test", gt_feature_root=Path("data/CelebVDub/avhubert_feat")):
    expected = {split + "/" + s.strip() for s in list_path.read_text().splitlines() if s.strip()}
    assert len(expected) == 213
    if stage in ('wav', 'features'):
        folder = root if stage == 'wav' else root / 'avhubert_feat'
        suffix = '.wav' if stage == 'wav' else '.npy'
        paths = list((folder / split).rglob('*' + suffix))
        actual = {str(p.relative_to(folder).with_suffix('')) for p in paths}
        assert actual == expected, (stage, len(actual), sorted(expected-actual), sorted(actual-expected))
        for p in paths:
            if stage == 'wav':
                a, sr = sf.read(p)
                assert sr == 16000 and a.ndim == 1 and len(a) > 0, p
            else:
                a = np.load(p)
                gt = np.load(gt_feature_root / p.relative_to(folder))
                assert a.shape == gt.shape, (p, a.shape, gt.shape)
            assert a.size and np.isfinite(a).all(), p
        return {'stage': stage, 'count': len(paths)}
    lines = (root / f'_{stage}_results.jsonl').read_text().splitlines()
    rows = [json.loads(s) for s in lines if s.startswith('{')]
    ids = [r['utterance_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 213 and set(ids) == expected
    assert all(np.isfinite(r[stage]) for r in rows)
    result = {'stage': stage, 'count': len(rows)}
    if stage == 'wer':
        m = compute_measures([r['truth'] for r in rows], [r['hypo'] for r in rows])
        result.update({k: m[k] for k in ('hits', 'substitutions', 'deletions', 'insertions')})
        value = m['wer']
        result['reference_words'] = m['hits'] + m['substitutions'] + m['deletions']
    else:
        value = float(np.mean([r[stage] for r in rows]))
    trailer = float(next(s.split(':', 1)[1] for s in lines if s.startswith(stage.upper()+':')))
    assert abs(value-trailer) <= 0.00000501, (value, trailer)
    result['value'] = value
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('stage', choices=['wav','features','sim','wer','emosim','avsync','all'])
    parser.add_argument("--test-list", type=Path, default=Path("data/celebvdub_test_s1.lst"))
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--gt-feature-root", type=Path, default=Path("data/CelebVDub/avhubert_feat"))
    args = parser.parse_args()
    stages = ['wav','features','sim','wer','emosim','avsync'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print(json.dumps(validate(args.root, stage, args.test_list, args.split, args.gt_feature_root), ensure_ascii=False))
