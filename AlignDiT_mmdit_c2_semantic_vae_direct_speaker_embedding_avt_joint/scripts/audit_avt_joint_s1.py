"""Audit complete CelebV-Dub S1 runs and recompute their four metrics.

The inherited single-worker evaluator writes basename-only IDs. Its ordered
records are checked against the test list, then exported with full clip IDs.
Run after every inference/evaluation pipeline has completed successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import compute_measures


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_results(path, task, clips):
    records = []
    reported = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("{"):
            records.append(json.loads(line))
        else:
            label, value = line.split(":", 1)
            require(label == task.upper() and reported is None, f"Invalid summary: {path}: {line}")
            reported = float(value)
    require(len(records) == len(clips), f"Incomplete {task}: {len(records)}/{len(clips)}")
    require(reported is not None and math.isfinite(reported), f"Missing finite summary: {path}")
    for clip, row in zip(clips, records):
        require(row["wav"] == Path(clip).name, f"Wrong ordered clip identity: {path}: {clip}: {row}")
    return records, reported


def audit(output_dir, test_list, gt_root):
    clips = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
    require(len(clips) == len(set(clips)) == 213, "Expected 213 unique test clips")
    summary = json.loads((output_dir / "inference_summary.json").read_text())
    require(summary["dataset"]["count"] == 213, "Inference used a partial dataset")
    require(summary["dataset"]["test_list_sha256"] == sha256(test_list), "Test list changed")
    require(summary["checkpoint"]["weights"] == "EMA", "Expected EMA weights")
    expected_generation = {
        "cfg_text": 5.0, "cfg_video": 2.0, "nfe": 32, "ode_method": "euler",
        "seed": 0, "sway": -1.0, "use_epss": True,
    }
    for key, expected in expected_generation.items():
        require(summary["generation"][key] == expected, f"Wrong generation setting: {key}")
    require(summary["speaker_embedding"] is not None, "Missing speaker conditioning metadata")
    expected_wavs = {f"test/{clip}.wav" for clip in clips}
    expected_features = {f"test/{clip}.npy" for clip in clips}
    actual_wavs = {str(p.relative_to(output_dir)) for p in (output_dir / "test").rglob("*.wav")}
    feature_root = output_dir / "avhubert_feat"
    actual_features = {str(p.relative_to(feature_root)) for p in feature_root.rglob("*.npy")}
    require(actual_wavs == expected_wavs, "WAV set does not exactly match test list")
    require(actual_features == expected_features, "AV-HuBERT feature set does not exactly match test list")
    receipts = {row["relative_path"]: row for row in summary["outputs"]}
    require(len(summary["outputs"]) == len(receipts) == 213, "Duplicate/missing inference receipts")
    results, reported = {}, {}
    for task in ("sim", "wer", "emosim", "avsync"):
        results[task], reported[task] = read_results(output_dir / f"_{task}_results.jsonl", task, clips)
    per_clip = []
    for index, clip in enumerate(clips):
        relative = f"test/{clip}.wav"
        path = output_dir / relative
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 16000 and audio.shape == (receipts[relative]["samples"], 1), f"Invalid WAV: {relative}")
        require(np.isfinite(audio).all() and len(audio) > 0, f"Non-finite/empty WAV: {relative}")
        require(sha256(path) == receipts[relative]["sha256"], f"WAV checksum mismatch: {relative}")
        feature_path = Path("test") / f"{clip}.npy"
        generated = np.load(feature_root / feature_path, allow_pickle=False)
        ground_truth = np.load(gt_root / "avhubert_feat" / feature_path, allow_pickle=False)
        require(generated.shape == ground_truth.shape and generated.ndim == 2, f"Feature shape mismatch: {clip}")
        require(np.isfinite(generated).all() and np.isfinite(ground_truth).all(), f"Non-finite features: {clip}")
        avsync = torch.nn.functional.cosine_similarity(
            torch.from_numpy(ground_truth), torch.from_numpy(generated), dim=1,
        ).mean().item()
        require(abs(avsync - results["avsync"][index]["avsync"]) < 1e-6, f"AVSync mismatch: {clip}")
        wer_row = results["wer"][index]
        reference = (gt_root / "text/test" / f"{clip}.txt").read_text().splitlines()[0].strip().lower()
        require(wer_row["raw_truth"] == reference, f"ASR reference mismatch: {clip}")
        measures = compute_measures(wer_row["truth"], wer_row["hypo"])
        require(abs(measures["wer"] - wer_row["wer"]) < 1e-9, f"Per-clip WER mismatch: {clip}")
        row = {
            "clip_id": f"test/{clip}", "wav": relative,
            "SPKSIM": float(results["sim"][index]["sim"]),
            "WER": float(measures["wer"]),
            "EMOSIM": float(results["emosim"][index]["emosim"]), "AVSync": avsync,
            "reference_words": measures["hits"] + measures["substitutions"] + measures["deletions"],
            "word_errors": measures["substitutions"] + measures["deletions"] + measures["insertions"],
        }
        require(all(math.isfinite(row[key]) for key in ("SPKSIM", "WER", "EMOSIM", "AVSync")), f"Invalid metric: {clip}")
        per_clip.append(row)
    reference_words = sum(row["reference_words"] for row in per_clip)
    word_errors = sum(row["word_errors"] for row in per_clip)
    require(reference_words > 0, "Empty corpus reference")
    metrics = {
        "SPKSIM": float(np.mean([row["SPKSIM"] for row in per_clip])),
        "WER": word_errors / reference_words,
        "EMOSIM": float(np.mean([row["EMOSIM"] for row in per_clip])),
        "AVSync": float(np.mean([row["AVSync"] for row in per_clip])),
    }
    corpus = compute_measures([row["truth"] for row in results["wer"]], [row["hypo"] for row in results["wer"]])
    require(abs(corpus["wer"] - metrics["WER"]) < 1e-12, "Corpus WER disagrees with summed edit counts")
    for key, task in (("SPKSIM", "sim"), ("WER", "wer"), ("EMOSIM", "emosim"), ("AVSync", "avsync")):
        require(abs(metrics[key] - reported[task]) <= 0.0000050001, f"Reported {key} disagrees with recomputation")
    report = {
        "complete": True, "count": 213, "output_dir": str(output_dir),
        "checkpoint": summary["checkpoint"], "generation": summary["generation"],
        "protocol": "CelebV-Dub Setting 1, same-clip full GT prompt and speaker conditioning",
        "identity_mapping": "single-worker result order checked against full test-list IDs and raw basenames",
        "metrics": metrics, "metrics_5dp": {key: f"{value:.5f}" for key, value in metrics.items()},
        "wer_reference_words": reference_words, "wer_word_errors": word_errors,
        "coverage": {"wavs": 213, "avhubert_features": 213, "each_metric": 213},
    }
    (output_dir / "evaluation_per_clip.jsonl").write_text("".join(json.dumps(row) + "\n" for row in per_clip))
    (output_dir / "evaluation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, action="append", required=True)
    parser.add_argument("--test-list", type=Path, default=Path(prefix + "/zjw524/projects/data/celebvdub_test_s1.lst"))
    parser.add_argument("--gt-root", type=Path, default=Path(prefix + "/zjw524/projects/data/CelebVDub"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    for output in args.output_dir:
        print(json.dumps(audit(output.resolve(), args.test_list, args.gt_root), indent=2), flush=True)
