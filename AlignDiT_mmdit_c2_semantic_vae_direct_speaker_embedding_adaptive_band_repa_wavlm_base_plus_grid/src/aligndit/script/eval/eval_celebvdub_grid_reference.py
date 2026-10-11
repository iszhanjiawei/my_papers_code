"""Evaluate 213 CelebV-Dub targets conditioned on independent GRID references.

Run one task on one visible GPU, e.g. CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src
python -m aligndit.script.eval.eval_celebvdub_grid_reference --manifest pairs.jsonl
--gen-wav-dir generated -e sim. Historical Setting 1 remains a separate protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import wave
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_COUNT = 213
TASKS = ("sim", "wer", "emosim", "emoembed", "avsync")
REQUIRED_FIELDS = (
    "pair_id", "target_id", "target_text", "target_gt_audio", "ref_id", "ref_speaker_id", "ref_audio"
)
REFERENCE_PROTOCOLS = {
    "sim": "generated audio versus actual GRID reference audio",
    "wer": "generated audio ASR versus CelebV-Dub target text",
    "emosim": "emotion2vec classifier-score cosine versus CelebV-Dub target GT audio",
    "emoembed": "emotion2vec utterance-embedding cosine versus CelebV-Dub target GT audio",
    "avsync": "AV-HuBERT joint-feature cosine: target video with generated audio versus target video with target GT audio",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inference_summary(rows: list[dict[str, Any]], manifest: Path, gen_dir: Path) -> dict[str, Any]:
    """Bind evaluation to the complete GRID run, not an old Setting 1 directory."""
    path = gen_dir / "inference_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("protocol") != "celebvdub_grid_reference_one_per_target_v1"
        or summary.get("pair_manifest_sha256") != sha256_file(manifest)
        or summary.get("count") != EXPECTED_COUNT
        or summary.get("partial_smoke_test") is not False
    ):
        raise ValueError("Inference summary is not a complete 213-pair GRID run for this manifest")
    outputs = summary.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != EXPECTED_COUNT:
        raise ValueError("Inference summary does not contain exactly 213 outputs")
    by_target = {}
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("target_id"), str):
            raise ValueError("Invalid inference output record")
        if output["target_id"] in by_target:
            raise ValueError(f"Duplicate inference output: {output['target_id']}")
        by_target[output["target_id"]] = output
    if set(by_target) != {row["target_id"] for row in rows}:
        raise ValueError("Inference outputs and pairing manifest select different targets")
    for row in rows:
        output = by_target[row["target_id"]]
        for field in ("pair_id", "target_id", "ref_id", "ref_speaker_id"):
            if output.get(field) != row[field]:
                raise ValueError(f"Inference {field} mismatch for {row['pair_id']}")
        relative_path = f"{row['target_id']}.wav"
        if output.get("relative_path") != relative_path:
            raise ValueError(f"Inference WAV path mismatch for {row['pair_id']}")
        audio = gen_dir / relative_path
        if output.get("sha256") != sha256_file(audio):
            raise ValueError(f"Generated WAV hash mismatch for {row['pair_id']}")
        with wave.open(str(audio), "rb") as handle:
            samples = handle.getnframes()
            if handle.getframerate() != 16000 or handle.getnchannels() != 1 or samples <= 0:
                raise ValueError(f"Expected nonempty 16-kHz mono generated WAV: {audio}")
        if output.get("samples") != samples:
            raise ValueError(f"Generated WAV sample count mismatch for {row['pair_id']}")
        if "target_num_samples" in row and row["target_num_samples"] != samples:
            raise ValueError(f"Generated WAV duration differs from target: {row['pair_id']}")
    return summary


def _safe_id(value: str, prefix: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute() or len(path.parts) != 3 or path.parts[0] != prefix
        or ".." in path.parts or "\\" in value or path.as_posix() != value
    ):
        raise ValueError(f"Invalid {prefix} identifier: {value!r}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Validate the complete pairing before any model is loaded; never skip rows."""
    path = path.resolve(strict=True)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Manifest line {line_number} must be an object")
        for key in REQUIRED_FIELDS:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Manifest line {line_number}: missing/non-string {key}")
        _safe_id(row["target_id"], "test")
        _safe_id(row["ref_id"], "grid")
        if PurePosixPath(row["ref_id"]).parts[1] != row["ref_speaker_id"]:
            raise ValueError(f"Reference speaker mismatch for {row['pair_id']}")
        for key in ("target_gt_audio", "ref_audio"):
            audio = Path(row[key])
            if not audio.is_absolute():
                audio = path.parent / audio
            if not audio.is_file():
                raise FileNotFoundError(f"{row['pair_id']}: missing {key}: {audio}")
            row[key] = str(audio.resolve(strict=True))
        rows.append(row)
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_COUNT} pairs, found {len(rows)}")
    for key in ("pair_id", "target_id"):
        if len({row[key] for row in rows}) != EXPECTED_COUNT:
            raise ValueError(f"Expected {EXPECTED_COUNT} unique {key} values")
    return rows


def build_test_set(rows: list[dict[str, Any]], gen_dir: Path, task: str) -> list[tuple[str, str, str]]:
    if task not in TASKS:
        raise ValueError(f"Unknown metric task: {task}")
    test_set = []
    for row in rows:
        generated = gen_dir / f"{row['target_id']}.wav"
        if not generated.is_file():
            raise FileNotFoundError(f"{row['pair_id']}: missing generated WAV: {generated}")
        # WER and AVSync ignore the second item. Emotion must keep target GT.
        reference = row["ref_audio"] if task == "sim" else row["target_gt_audio"]
        test_set.append((str(generated), reference, row["target_text"]))
    return test_set


def validate_av_features(rows: list[dict[str, Any]], gen_dir: Path, gt_dir: Path) -> None:
    import numpy as np

    for row in rows:
        gt_path = gt_dir / f"{row['target_id']}.npy"
        generated_path = gen_dir / "avhubert_feat" / f"{row['target_id']}.npy"
        gt = np.load(gt_path, allow_pickle=False)
        generated = np.load(generated_path, allow_pickle=False)
        if (
            gt.ndim != 2 or not all(gt.shape) or gt.shape != generated.shape
            or not np.isfinite(gt).all() or not np.isfinite(generated).all()
        ):
            raise ValueError(f"Missing, mismatched or non-finite AV features: {row['target_id']}")


def word_edit_counts(truth: str, hypothesis: str) -> tuple[int, int]:
    """Exact word Levenshtein numerator and reference-word denominator."""
    reference, prediction = truth.split(), hypothesis.split()
    if not reference:
        raise ValueError("WER reference contains no words")
    previous = list(range(len(prediction) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1], len(reference)


def attach_identities(
    rows: list[dict[str, Any]], results: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    """The reused single-rank workers preserve order. Never join on basename."""
    if len(results) != len(rows):
        raise ValueError(f"Incomplete {task} results: {len(results)} != {len(rows)}")
    identified = []
    for row, result in zip(rows, results):
        expected_stem = PurePosixPath(row["target_id"]).name
        if result.get("wav") != expected_stem:
            raise ValueError(f"Worker result order mismatch for {row['pair_id']}")
        value = result.get(task)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Invalid {task} result for {row['pair_id']}: {value!r}")
        item = dict(result)
        # Replace ambiguous legacy basename with the full target identifier.
        item["wav"] = row["target_id"]
        item.update({key: row[key] for key in ("pair_id", "target_id", "ref_id", "ref_speaker_id")})
        item["ref_audio"] = row["ref_audio"]
        item["target_gt_audio"] = row["target_gt_audio"]
        if task == "wer":
            if not isinstance(item.get("truth"), str) or not isinstance(item.get("hypo"), str):
                raise ValueError(f"Missing WER text for {row['pair_id']}")
            errors, words = word_edit_counts(item["truth"], item["hypo"])
            if not math.isclose(value, errors / words, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"WER count mismatch for {row['pair_id']}")
            item["word_edit_distance"] = errors
            item["reference_word_count"] = words
        identified.append(item)
    return identified


def summarize(results: list[dict[str, Any]], task: str) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize empty results")
    summary: dict[str, Any] = {"count": len(results)}
    if task == "wer":
        errors = sum(item["word_edit_distance"] for item in results)
        words = sum(item["reference_word_count"] for item in results)
        if words <= 0:
            raise ValueError("Corpus WER requires a positive reference-word count")
        metric = errors / words
        summary.update(word_edit_distance=errors, reference_word_count=words, aggregation="corpus_word_edit_distance")
    else:
        metric = math.fsum(item[task] for item in results) / len(results)
        summary["aggregation"] = "sample_mean"
    if not math.isfinite(metric):
        raise ValueError(f"Non-finite aggregate {task}")
    summary.update({task: metric, "display_value": f"{metric:.5f}"})
    return summary


def make_summary(results: list[dict[str, Any]], task: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[item["ref_speaker_id"]].append(item)
    return {
        "protocol": "CelebV-Dub targets with independent GRID references",
        "task": task,
        "metric_reference": REFERENCE_PROTOCOLS[task],
        **summarize(results, task),
        "per_reference_speaker": {speaker: summarize(items, task) for speaker, items in sorted(groups.items())},
    }


def run_worker(args: argparse.Namespace, test_set: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    if args.eval_task == "sim":
        from f5_tts.eval.utils_eval import run_sim

        return run_sim((0, test_set, str(args.wavlm_ckpt), str(args.wavlm_base_ckpt)))
    from aligndit.script.eval.utils import run_asr_wer, run_avsync, run_emoembed, run_emosim

    if args.eval_task == "wer":
        return run_asr_wer((0, "en", test_set, str(args.asr_ckpt)))
    if args.eval_task in ("emosim", "emoembed"):
        worker = run_emosim if args.eval_task == "emosim" else run_emoembed
        return worker((0, test_set, str(args.emo_ckpt)))
    return run_avsync((0, test_set, str(args.gt_av_feat), str(args.gen_wav_dir / "avhubert_feat")))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.manifest = args.manifest.resolve(strict=True)
    args.gen_wav_dir = args.gen_wav_dir.resolve(strict=True)
    rows = load_manifest(args.manifest)
    test_set = build_test_set(rows, args.gen_wav_dir, args.eval_task)
    inference_summary = validate_inference_summary(rows, args.manifest, args.gen_wav_dir)
    result_path = args.gen_wav_dir / f"_{args.eval_task}_results.jsonl"
    summary_path = args.gen_wav_dir / f"_{args.eval_task}_summary.json"
    for path in (result_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation: {path}")
    resources = {
        "sim": (args.wavlm_ckpt, args.wavlm_base_ckpt),
        "wer": (args.asr_ckpt,),
        "emosim": (args.emo_ckpt,),
        "emoembed": (args.emo_ckpt,),
        "avsync": (args.gt_av_feat,),
    }[args.eval_task]
    for resource in resources:
        if not resource.exists():
            raise FileNotFoundError(f"Local metric resource unavailable: {resource}")
    if args.eval_task == "avsync":
        validate_av_features(rows, args.gen_wav_dir, args.gt_av_feat)
    results = attach_identities(rows, run_worker(args, test_set), args.eval_task)
    summary = make_summary(results, args.eval_task)
    summary.update(
        manifest=str(args.manifest),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        gen_wav_dir=str(args.gen_wav_dir),
        metric_resources=[str(path.resolve()) for path in resources],
        inference_summary_sha256=sha256_file(args.gen_wav_dir / "inference_summary.json"),
        checkpoint=inference_summary.get("checkpoint"),
    )
    with result_path.open("x", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        handle.write(f"\n{args.eval_task.upper()}: {summary['display_value']}\n")
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Total {len(results)} samples\n{args.eval_task.upper()}: {summary['display_value']}")
    print(f"Results saved to {result_path}\nSummary saved to {summary_path}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    data_root = Path(f"{prefix}/zjw524/projects/data")
    pretrained_root = Path(f"{prefix}/zjw524/alignDiT_pretrain_models")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gen-wav-dir", "-g", type=Path, required=True)
    parser.add_argument("--eval-task", "-e", choices=TASKS, required=True)
    parser.add_argument("--wavlm-ckpt", type=Path, default=pretrained_root / "wavlm_large_finetune.pth")
    parser.add_argument("--wavlm-base-ckpt", type=Path, default=pretrained_root / "wavlm_large_s3prl.pt")
    parser.add_argument("--asr-ckpt", type=Path, default=data_root / "faster-whisper-large-v3")
    parser.add_argument("--emo-ckpt", type=Path, default=data_root / "emotion2vec_plus_large")
    parser.add_argument("--gt-av-feat", type=Path, default=data_root / "CelebVDub/avhubert_feat")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
