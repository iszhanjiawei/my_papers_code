"""Independently verify a complete GRID-reference run without loading GPU models.

Repeated verification accepts an identical existing _verified_summary.json;
different existing verification evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf


EXPECTED_COUNT = 213
REQUIRED_TASKS = ("sim", "wer", "emosim", "avsync")
IDENTITY_FIELDS = ("pair_id", "target_id", "ref_id", "ref_speaker_id")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_rows(path: Path, *, task: str | None = None) -> list[dict[str, Any]]:
    """Only an optional final legacy metric display line may be non-JSON."""
    lines = [(i, line.strip()) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if line.strip()]
    rows = []
    for position, (line_number, line) in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            require(task is not None and position == len(lines) - 1 and line.startswith(f"{task.upper()}:"),
                    f"Invalid JSON or misplaced summary at {path}:{line_number}")
            display = line.split(":", 1)[1].strip()
            require(math.isfinite(float(display)), f"Non-finite display trailer: {path}")
            continue
        require(isinstance(value, dict), f"Non-object row at {path}:{line_number}")
        rows.append(value)
    return rows


def resolved_path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def safe_target_id(value: str) -> None:
    require(isinstance(value, str), "target_id must be a string")
    path = PurePosixPath(value)
    require(not path.is_absolute() and len(path.parts) == 3 and path.parts[0] == "test"
            and ".." not in path.parts and "\\" not in value and path.as_posix() == value,
            f"Invalid full target identifier: {value!r}")


def finite_value(value: Any, label: str) -> float:
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            f"Invalid finite numeric value for {label}: {value!r}")
    return float(value)


def close_value(actual: Any, expected: float, label: str) -> None:
    value = finite_value(actual, label)
    require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Recomputed value mismatch for {label}: {value} != {expected}")


def word_edit_counts(truth: str, hypothesis: str) -> tuple[int, int]:
    """Independent word-level Levenshtein implementation, including insertions."""
    require(isinstance(truth, str) and isinstance(hypothesis, str), "WER texts must be strings")
    reference, prediction = truth.split(), hypothesis.split()
    require(bool(reference), "Empty WER reference")
    costs = list(range(len(reference) + 1))
    for predicted_index, predicted in enumerate(prediction, 1):
        next_costs = [predicted_index]
        for reference_index, expected in enumerate(reference, 1):
            next_costs.append(min(costs[reference_index] + 1, next_costs[-1] + 1,
                                  costs[reference_index - 1] + (predicted != expected)))
        costs = next_costs
    return costs[-1], len(reference)


def normalized_wer_text(text: str) -> str:
    # Match the persisted evaluator's punctuation removal, without importing it.
    from zhon.hanzi import punctuation

    for character in punctuation + string.punctuation:
        text = text.replace(character, "")
    return text.replace("  ", " ").lower()


def aggregate(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    require(bool(rows), f"Empty {task} result set")
    values = [finite_value(row.get(task), task) for row in rows]
    result: dict[str, Any] = {"count": len(rows)}
    if task == "wer":
        errors = words = 0
        for row, value in zip(rows, values):
            row_errors, row_words = word_edit_counts(row.get("truth"), row.get("hypo"))
            close_value(value, row_errors / row_words, f"per-sample WER {row.get('pair_id')}")
            require(row.get("word_edit_distance") == row_errors and row.get("reference_word_count") == row_words,
                    f"Persisted WER counts differ for {row.get('pair_id')}")
            errors += row_errors
            words += row_words
        value = errors / words
        result.update(word_edit_distance=errors, reference_word_count=words,
                      aggregation="corpus_word_edit_distance")
    else:
        value = math.fsum(values) / len(values)
        result["aggregation"] = "sample_mean"
    result.update({task: value, "display_value": f"{value:.5f}"})
    return result


def verify_aggregate(actual: dict[str, Any], expected: dict[str, Any], task: str, label: str) -> None:
    close_value(actual.get(task), expected[task], label)
    for field in ("count", "display_value", "aggregation"):
        require(actual.get(field) == expected[field], f"Wrong {label} {field}")
    if task == "wer":
        for field in ("word_edit_distance", "reference_word_count"):
            require(actual.get(field) == expected[field], f"Wrong {label} {field}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    manifest_hash = sha256_file(manifest)
    pairs = read_rows(manifest)
    require(len(pairs) == EXPECTED_COUNT, f"Expected {EXPECTED_COUNT} manifest rows")
    for field in ("pair_id", "target_id"):
        require(all(isinstance(row.get(field), str) and row[field] for row in pairs), f"Missing {field}")
        require(len({row[field] for row in pairs}) == EXPECTED_COUNT, f"Duplicate {field}")
    summary_path = output / "inference_summary.json"
    inference = read_object(summary_path)
    require(inference.get("count") == EXPECTED_COUNT and inference.get("partial_smoke_test") is False,
            "Inference must be a complete 213-pair run")
    require(inference.get("pair_manifest_sha256") == manifest_hash, "Inference manifest hash mismatch")
    require(resolved_path(inference["pair_manifest"], output) == manifest, "Inference references another manifest")
    checkpoint = inference["checkpoint"]
    require(checkpoint.get("weights") == "EMA", "Inference did not use EMA")
    require(checkpoint.get("update") == args.expected_step, "Inference checkpoint update mismatch")
    checkpoint_path = resolved_path(checkpoint["path"], output)
    require(sha256_file(checkpoint_path) == checkpoint.get("sha256"), "Checkpoint hash changed")
    generation = inference.get("generation", {})
    require(generation.get("target_acoustic_inputs") is False, "Inference does not attest external-only acoustic input")
    generated_rows = inference.get("outputs")
    require(isinstance(generated_rows, list) and len(generated_rows) == EXPECTED_COUNT,
            "Missing complete inference output records")
    journal = output / "generation_records.jsonl"
    if journal.exists():
        require(read_rows(journal) == generated_rows, "Generation journal differs from final inference summary")
    expected_wavs, expected_features = set(), set()
    for pair, generated in zip(pairs, generated_rows):
        safe_target_id(pair["target_id"])
        for field in IDENTITY_FIELDS:
            require(generated.get(field) == pair.get(field) and pair.get(field) is not None,
                    f"Inference identity mismatch: {field}, {pair['pair_id']}")
        reference_id = PurePosixPath(pair["ref_id"])
        require(len(reference_id.parts) == 3 and reference_id.parts[0] == "grid"
                and reference_id.parts[1] == pair["ref_speaker_id"], "GRID speaker identifier mismatch")
        require(resolved_path(pair["ref_audio"], manifest.parent)
                != resolved_path(pair["target_gt_audio"], manifest.parent), "GRID reference equals target GT audio")
        for path_field, hash_field in (
            ("ref_audio", "ref_audio_sha256"), ("ref_latent_path", "ref_latent_sha256"),
            ("ref_speaker_path", "ref_speaker_sha256"), ("target_video_path", "target_video_sha256"),
        ):
            path = resolved_path(pair[path_field], manifest.parent)
            require(sha256_file(path) == pair[hash_field], f"Changed artifact: {path}")
        relative = pair["target_id"] + ".wav"
        expected_wavs.add(relative)
        require(generated.get("relative_path") == relative, f"Wrong generated path: {pair['pair_id']}")
        wav_path = output / relative
        require(sha256_file(wav_path) == generated.get("sha256"), f"Generated WAV hash mismatch: {relative}")
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        samples = pair["target_num_samples"]
        require(sample_rate == 16000 and waveform.shape == (samples, 1) and np.isfinite(waveform).all(),
                f"Invalid 16-kHz mono target-length WAV: {relative}")
        require(generated.get("samples") == samples, f"Inference sample-count mismatch: {relative}")
        feature_relative = pair["target_id"] + ".npy"
        expected_features.add(feature_relative)
        feature = np.load(output / "avhubert_feat" / feature_relative, allow_pickle=False)
        require(feature.ndim == 2 and all(feature.shape) and np.isfinite(feature).all(),
                f"Invalid generated AV feature: {feature_relative}")
    actual_wavs = {path.relative_to(output).as_posix() for path in output.rglob("*")
                   if path.is_file() and path.suffix.lower() == ".wav"}
    actual_features = {path.relative_to(output / "avhubert_feat").as_posix()
                       for path in (output / "avhubert_feat").rglob("*")
                       if path.is_file() and path.suffix.lower() == ".npy"}
    require(actual_wavs == expected_wavs, "Actual WAV file set differs from all 213 expected targets")
    require(actual_features == expected_features, "Actual AV feature set differs from all 213 expected targets")
    metrics, evidence = {}, {}
    tasks = list(REQUIRED_TASKS)
    if (output / "_emoembed_results.jsonl").exists() or (output / "_emoembed_summary.json").exists():
        tasks.append("emoembed")
    for task in tasks:
        result_path = output / f"_{task}_results.jsonl"
        metric_summary_path = output / f"_{task}_summary.json"
        rows = read_rows(result_path, task=task)
        require(len(rows) == EXPECTED_COUNT, f"Expected {EXPECTED_COUNT} {task} rows")
        for pair, row in zip(pairs, rows):
            for field in IDENTITY_FIELDS:
                require(row.get(field) == pair[field], f"{task} identity/order mismatch: {field}, {pair['pair_id']}")
            require(row.get("wav") == pair["target_id"], f"{task} must use full target_id, not basename")
            for field in ("ref_audio", "target_gt_audio"):
                require(resolved_path(row[field], output) == resolved_path(pair[field], manifest.parent),
                        f"{task} audio reference mismatch: {pair['pair_id']}, {field}")
            if task == "wer":
                require(row.get("raw_truth") == pair["target_text"], f"WER raw target text differs: {pair['pair_id']}")
                require(row.get("truth") == normalized_wer_text(pair["target_text"]), "WER target normalization mismatch")
                require(isinstance(row.get("raw_hypo"), str)
                        and row.get("hypo") == normalized_wer_text(row["raw_hypo"]), "WER hypothesis normalization mismatch")
        recomputed = aggregate(rows, task)
        recorded = read_object(metric_summary_path)
        require(recorded.get("manifest_sha256") == manifest_hash, f"{task} summary manifest hash mismatch")
        require(recorded.get("task") == task, f"Wrong task in {task} summary")
        require(resolved_path(recorded["gen_wav_dir"], output) == output, f"{task} summary uses another output directory")
        verify_aggregate(recorded, recomputed, task, task)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["ref_speaker_id"]].append(row)
        per_speaker = recorded.get("per_reference_speaker")
        require(isinstance(per_speaker, dict) and set(per_speaker) == set(groups), f"{task} speaker coverage mismatch")
        for speaker, speaker_rows in groups.items():
            verify_aggregate(per_speaker[speaker], aggregate(speaker_rows, task), task, f"{task}/{speaker}")
        trailer = [line.strip() for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()][-1]
        if trailer.startswith(f"{task.upper()}:"):
            require(trailer.split(":", 1)[1].strip() == recomputed["display_value"], f"{task} display trailer mismatch")
        metrics[task] = recomputed
        evidence[task] = {"results_sha256": sha256_file(result_path), "summary_sha256": sha256_file(metric_summary_path)}
    result = {
        "status": "verified", "count": EXPECTED_COUNT,
        "protocol": inference.get("protocol"), "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "output_dir": str(output), "checkpoint": checkpoint, "generation": generation,
        "inference_summary_sha256": sha256_file(summary_path),
        "wav_count": len(actual_wavs), "av_feature_count": len(actual_features),
        "metrics": metrics, "metric_evidence": evidence,
    }
    for key in ("device", "gpu", "gpu_metadata", "inference_device", "cuda_visible_devices"):
        if key in inference:
            result[key] = inference[key]
    verification_path = output / "_verified_summary.json"
    if verification_path.exists():
        require(read_object(verification_path) == result,
                f"Existing verification evidence differs; refusing overwrite: {verification_path}")
    else:
        with verification_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verified": EXPECTED_COUNT, "metrics": metrics, "output": str(verification_path)},
                     ensure_ascii=False, allow_nan=False), flush=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=150000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    verify(parse_args())
