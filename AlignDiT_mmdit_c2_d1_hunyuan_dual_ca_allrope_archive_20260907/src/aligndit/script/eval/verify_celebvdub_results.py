"""Read-only completeness and metric audit; prints JSON for archival."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import compute_measures


def verify(result_dir, clips, data_dir):
    expected_wavs = {f"test/{clip}.wav" for clip in clips}
    actual_wavs = {str(p.relative_to(result_dir)) for p in (result_dir / "test").rglob("*.wav")}
    assert actual_wavs == expected_wavs, ("WAV paths differ", result_dir)
    expected_feats = {f"test/{clip}.npy" for clip in clips}
    actual_feats = {str(p.relative_to(result_dir / "avhubert_feat"))
                    for p in (result_dir / "avhubert_feat").rglob("*.npy")}
    assert actual_feats == expected_feats, ("Feature paths differ", result_dir)
    max_duration_error = 0
    for clip in clips:
        wav, sr = sf.read(result_dir / f"test/{clip}.wav")
        assert sr == 16000 and wav.ndim == 1 and len(wav) > 0, clip
        assert np.isfinite(wav).all() and np.any(wav != 0), clip
        gt_info = sf.info(data_dir / f"audio/test/{clip}.wav")
        duration_error = abs(len(wav) / sr - gt_info.duration)
        assert duration_error < 0.05, (clip, duration_error)
        max_duration_error = max(max_duration_error, duration_error)
        feat = np.load(result_dir / f"avhubert_feat/test/{clip}.npy")
        gt = np.load(data_dir / f"avhubert_feat/test/{clip}.npy", mmap_mode="r")
        assert feat.shape == gt.shape and np.isfinite(feat).all(), clip

    metrics = {}
    counts = {}
    expected_stems = [Path(clip).name for clip in clips]
    for metric in ("sim", "wer", "emosim", "avsync"):
        path = result_dir / f"_{metric}_results.jsonl"
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        assert lines[-1].startswith(f"{metric.upper()}: "), path
        rows = [json.loads(line) for line in lines[:-1]]
        assert len(rows) == len(clips), (path, len(rows))
        # Stems are not unique: validate original single-GPU list order, not a set.
        assert [row["wav"] for row in rows] == expected_stems, path
        assert all(math.isfinite(row[metric]) for row in rows), path
        if metric == "wer":
            for clip, row in zip(clips, rows):
                text = (data_dir / f"text/test/{clip}.txt").read_text().splitlines()[0].strip().lower()
                assert row["raw_truth"] == text, clip
            measures = compute_measures([r["truth"] for r in rows], [r["hypo"] for r in rows])
            value = measures["wer"]
            counts = {key: measures[key] for key in ("hits", "substitutions", "deletions", "insertions")}
            counts["reference_words"] = sum(counts[k] for k in ("hits", "substitutions", "deletions"))
        else:
            value = float(np.mean([r[metric] for r in rows]))
        assert round(value, 5) == float(lines[-1].split(": ")[1]), (path, value)
        metrics[metric] = {"value": round(value, 5), "samples": len(rows),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"result_dir": str(result_dir.resolve()), "wav_count": len(actual_wavs),
            "feature_count": len(actual_feats), "sample_rate": 16000, "channels": 1,
            "max_duration_error_seconds": max_duration_error, "metrics": metrics,
            "wer_counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--test-list", type=Path, default=Path("data/celebvdub_test_s1.lst"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/CelebVDub"))
    args = parser.parse_args()
    clips = args.test_list.read_text().splitlines()
    assert len(clips) == len(set(clips)) == 213
    result = {"test_list_sha256": hashlib.sha256(args.test_list.read_bytes()).hexdigest(),
              "results": [verify(p, clips, args.data_dir) for p in args.result_dirs]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
