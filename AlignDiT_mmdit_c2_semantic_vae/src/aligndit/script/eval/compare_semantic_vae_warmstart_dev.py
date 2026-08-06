"""Compare deterministic Semantic-VAE warm-start development evaluations.

Each input summary must have the paired ``*.per_utterance.jsonl`` produced by
``eval_semantic_vae_warmstart_dev.py`` in the same directory.  The comparator
fails closed if protocol, selection, repeats, paired draw keys, or stochastic
draw metadata differ.  Metrics are first averaged across repeats within each
utterance, then compared with an utterance-paired bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_SUFFIX = ".summary.json"
ROWS_SUFFIX = ".per_utterance.jsonl"
GROUPS = ("overall", "dev-clean", "dev-other")
SELECTION_FIELDS = (
    "hubert_completion_sha256",
    "latent_completion_sha256",
    "manifest_count",
    "manifest_sha256",
    "normalization_sha256",
    "selected_count",
    "selected_counts",
    "selected_keys_sha256",
    "subset_counts",
)
PAIR_FIELDS = (
    "diffusion_time",
    "flow_element_count",
    "frames",
    "hubert_frame_count",
    "mask_end",
    "mask_fraction_realized",
    "mask_fraction_sampled",
    "mask_start",
    "masked_frames",
    "repeat",
    "speaker_id",
    "subset",
    "utterance_key",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty per-utterance evaluation file: {path}")
    return rows


def _paired_rows_path(summary_path: Path) -> Path:
    if not summary_path.name.endswith(SUMMARY_SUFFIX):
        raise ValueError(f"Summary filename must end with {SUMMARY_SUFFIX!r}: {summary_path}")
    prefix = summary_path.name[: -len(SUMMARY_SUFFIX)]
    return summary_path.with_name(f"{prefix}{ROWS_SUFFIX}")


def _require_finite_number(row: dict[str, Any], field: str, *, context: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise TypeError(f"{context} has invalid finite numeric field {field!r}: {value!r}")
    return float(value)


def _extract_selection(summary: dict[str, Any], *, label: str) -> dict[str, Any]:
    dataset = summary.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError(f"Summary {label!r} has no dataset object")
    missing = [field for field in SELECTION_FIELDS if field not in dataset]
    if missing:
        raise KeyError(f"Summary {label!r} dataset is missing selection fields: {missing}")
    return {field: dataset[field] for field in SELECTION_FIELDS}


def _group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "overall":
        return rows
    return [row for row in rows if row["subset"] == group]


def _utterance_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[row["utterance_key"]].append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for key, draws in by_key.items():
        metrics[key] = {
            "flow_mse": math.fsum(float(draw["flow_mse"]) for draw in draws) / len(draws),
            "hubert_cosine": math.fsum(float(draw["hubert_cosine"]) for draw in draws) / len(draws),
            "speaker_id": draws[0]["speaker_id"],
            "subset": draws[0]["subset"],
        }
    return metrics


def _validate_summary_aggregates(summary: dict[str, Any], rows: list[dict[str, Any]], *, label: str) -> None:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise TypeError(f"Summary {label!r} has no results object")
    for group in GROUPS:
        result = results.get(group)
        if not isinstance(result, dict):
            raise TypeError(f"Summary {label!r} has no results[{group!r}] object")
        group_rows = _group_rows(rows, group)
        metrics = _utterance_metrics(group_rows)
        expected_flow = math.fsum(value["flow_mse"] for value in metrics.values()) / len(metrics)
        expected_cosine = math.fsum(value["hubert_cosine"] for value in metrics.values()) / len(metrics)
        checks = {
            "draws": len(group_rows),
            "utterances": len(metrics),
        }
        for field, expected in checks.items():
            if result.get(field) != expected:
                raise ValueError(f"Summary {label!r} {group} {field} mismatch: {result.get(field)!r} != {expected!r}")
        for field, expected in (
            ("flow_mse_macro", expected_flow),
            ("hubert_cosine_macro", expected_cosine),
        ):
            actual = _require_finite_number(result, field, context=f"summary {label!r} results[{group!r}]")
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Summary {label!r} {group} {field} mismatch: {actual!r} != {expected!r}")


def load_evaluation(summary_path: str | Path) -> dict[str, Any]:
    """Load and strictly validate one evaluator summary/JSONL pair."""

    summary_path = Path(summary_path).resolve(strict=True)
    summary = _read_json_object(summary_path)
    if summary.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation summary schema in {summary_path}: {summary.get('schema_version')!r}")
    label = summary.get("label")
    if not isinstance(label, str) or not label:
        raise TypeError(f"Invalid evaluation label in {summary_path}: {label!r}")
    if summary_path.name != f"{label}{SUMMARY_SUFFIX}":
        raise ValueError(f"Summary filename/label mismatch: {summary_path.name!r} != {label + SUMMARY_SUFFIX!r}")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError(f"Summary {label!r} has no protocol object")
    repeats = protocol.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise TypeError(f"Summary {label!r} has invalid protocol repeats: {repeats!r}")

    rows_path = _paired_rows_path(summary_path)
    rows = _read_jsonl(rows_path)
    selection = _extract_selection(summary, label=label)
    expected_draws = int(selection["selected_count"]) * repeats
    if len(rows) != expected_draws:
        raise ValueError(f"Evaluation {label!r} row count mismatch: {len(rows)} != {expected_draws}")

    seen_pairs: set[tuple[str, int]] = set()
    repeats_by_key: dict[str, set[int]] = defaultdict(set)
    subset_by_key: dict[str, str] = {}
    ordered_keys: list[str] = []
    for index, row in enumerate(rows):
        context = f"evaluation {label!r} row {index + 1}"
        missing = [field for field in PAIR_FIELDS + ("flow_mse", "hubert_cosine") if field not in row]
        if missing:
            raise KeyError(f"{context} is missing fields: {missing}")
        key = row["utterance_key"]
        repeat = row["repeat"]
        if not isinstance(key, str) or not key:
            raise TypeError(f"{context} has invalid utterance_key: {key!r}")
        if not isinstance(repeat, int) or isinstance(repeat, bool) or not 0 <= repeat < repeats:
            raise ValueError(f"{context} has invalid repeat: {repeat!r}")
        pair = (key, repeat)
        if pair in seen_pairs:
            raise ValueError(f"Evaluation {label!r} contains duplicate paired draw: {pair!r}")
        seen_pairs.add(pair)
        if key not in repeats_by_key:
            ordered_keys.append(key)
            subset_by_key[key] = row["subset"]
        elif subset_by_key[key] != row["subset"]:
            raise ValueError(f"Evaluation {label!r} changes subset within utterance {key!r}")
        repeats_by_key[key].add(repeat)
        flow = _require_finite_number(row, "flow_mse", context=context)
        cosine = _require_finite_number(row, "hubert_cosine", context=context)
        if flow < 0:
            raise ValueError(f"{context} has negative flow_mse: {flow}")
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"{context} has out-of-range hubert_cosine: {cosine}")

    expected_repeats = set(range(repeats))
    invalid_repeats = {key: values for key, values in repeats_by_key.items() if values != expected_repeats}
    if invalid_repeats:
        example = next(iter(invalid_repeats.items()))
        raise ValueError(f"Evaluation {label!r} has incomplete repeat coverage, e.g. {example!r}")
    if len(ordered_keys) != int(selection["selected_count"]):
        raise ValueError(
            f"Evaluation {label!r} utterance count mismatch: {len(ordered_keys)} != {selection['selected_count']}"
        )
    selected_counts = {subset: sum(value == subset for value in subset_by_key.values()) for subset in GROUPS[1:]}
    if selected_counts != selection["selected_counts"]:
        raise ValueError(
            f"Evaluation {label!r} selected subset counts mismatch: {selected_counts} != {selection['selected_counts']}"
        )
    keys_sha256 = hashlib.sha256("\n".join(ordered_keys).encode()).hexdigest()
    if keys_sha256 != selection["selected_keys_sha256"]:
        raise ValueError(
            f"Evaluation {label!r} selected key order/hash mismatch: "
            f"{keys_sha256} != {selection['selected_keys_sha256']}"
        )
    _validate_summary_aggregates(summary, rows, label=label)
    return {
        "label": label,
        "metrics": _utterance_metrics(rows),
        "pair_metadata": {
            (row["utterance_key"], row["repeat"]): tuple(row[field] for field in PAIR_FIELDS) for row in rows
        },
        "protocol": protocol,
        "rows_path": str(rows_path),
        "selection": selection,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def _validate_compatible(evaluations: list[dict[str, Any]]) -> None:
    reference = evaluations[0]
    reference_pairs = set(reference["pair_metadata"])
    for evaluation in evaluations[1:]:
        label = evaluation["label"]
        if evaluation["protocol"] != reference["protocol"]:
            raise ValueError(f"Protocol mismatch between {reference['label']!r} and {label!r}")
        if evaluation["selection"] != reference["selection"]:
            raise ValueError(f"Selection mismatch between {reference['label']!r} and {label!r}")
        pairs = set(evaluation["pair_metadata"])
        if pairs != reference_pairs:
            missing = sorted(reference_pairs - pairs)[:3]
            extra = sorted(pairs - reference_pairs)[:3]
            raise ValueError(f"Paired draw keys mismatch for {label!r}: missing={missing}, extra={extra}")
        for pair in reference_pairs:
            if evaluation["pair_metadata"][pair] != reference["pair_metadata"][pair]:
                raise ValueError(f"Paired draw metadata mismatch for {label!r} at {pair!r}")


def _bootstrap_seed(seed: int, baseline: str, group: str) -> int:
    payload = f"semantic-vae-warmstart-paired-bootstrap-v1:{seed}:{baseline}:{group}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _group_comparison(
    evaluations: list[dict[str, Any]],
    *,
    baseline_label: str,
    group: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(evaluations, key=lambda evaluation: evaluation["label"])
    baseline = next(evaluation for evaluation in ordered if evaluation["label"] == baseline_label)
    keys = sorted(key for key, value in baseline["metrics"].items() if group == "overall" or value["subset"] == group)
    if not keys:
        raise ValueError(f"No paired utterances for group {group!r}")
    baseline_flow = np.asarray([baseline["metrics"][key]["flow_mse"] for key in keys], dtype=np.float64)
    baseline_cosine = np.asarray([baseline["metrics"][key]["hubert_cosine"] for key in keys], dtype=np.float64)
    flow_values = np.asarray(
        [[evaluation["metrics"][key]["flow_mse"] for key in keys] for evaluation in ordered], dtype=np.float64
    )
    cosine_values = np.asarray(
        [[evaluation["metrics"][key]["hubert_cosine"] for key in keys] for evaluation in ordered],
        dtype=np.float64,
    )
    flow_delta = flow_values - baseline_flow[None, :]
    cosine_delta = cosine_values - baseline_cosine[None, :]
    flow_bootstrap = np.empty((len(ordered), bootstrap_samples), dtype=np.float64)
    cosine_bootstrap = np.empty((len(ordered), bootstrap_samples), dtype=np.float64)
    generator = np.random.default_rng(_bootstrap_seed(bootstrap_seed, baseline_label, group))
    chunk_size = 32
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        indices = generator.integers(0, len(keys), size=(stop - start, len(keys)))
        flow_bootstrap[:, start:stop] = flow_delta[:, indices].mean(axis=2)
        cosine_bootstrap[:, start:stop] = cosine_delta[:, indices].mean(axis=2)
    flow_ci = np.quantile(flow_bootstrap, [0.025, 0.975], axis=1).T
    cosine_ci = np.quantile(cosine_bootstrap, [0.025, 0.975], axis=1).T
    flow_means = flow_values.mean(axis=1)
    cosine_means = cosine_values.mean(axis=1)
    flow_ranks = {
        index: rank + 1
        for rank, index in enumerate(sorted(range(len(ordered)), key=lambda i: (flow_means[i], ordered[i]["label"])))
    }
    cosine_ranks = {
        index: rank + 1
        for rank, index in enumerate(sorted(range(len(ordered)), key=lambda i: (-cosine_means[i], ordered[i]["label"])))
    }
    entries: list[dict[str, Any]] = []
    for index, evaluation in enumerate(ordered):
        checkpoint = evaluation["summary"].get("checkpoint", {})
        entries.append(
            {
                "checkpoint": {
                    key: checkpoint.get(key) for key in ("stage", "update", "sha256", "weights") if key in checkpoint
                },
                "flow_delta_vs_baseline": {
                    "better_direction": "negative",
                    "ci95": [float(value) for value in flow_ci[index]],
                    "estimate": float(flow_delta[index].mean()),
                },
                "flow_mse": float(flow_means[index]),
                "flow_rank": flow_ranks[index],
                "hubert_cosine": float(cosine_means[index]),
                "hubert_cosine_delta_vs_baseline": {
                    "better_direction": "positive",
                    "ci95": [float(value) for value in cosine_ci[index]],
                    "estimate": float(cosine_delta[index].mean()),
                },
                "hubert_cosine_rank": cosine_ranks[index],
                "label": evaluation["label"],
                "utterances": len(keys),
            }
        )
    return entries


def compare_evaluations(
    summary_paths: list[str | Path],
    *,
    baseline_label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Validate evaluation pairs and return paired-bootstrap comparisons."""

    if len(summary_paths) < 2:
        raise ValueError("At least two evaluation summaries are required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    evaluations = [load_evaluation(path) for path in summary_paths]
    labels = [evaluation["label"] for evaluation in evaluations]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Evaluation labels must be unique: {labels}")
    if baseline_label not in labels:
        raise ValueError(f"Unknown baseline label {baseline_label!r}; available labels: {sorted(labels)}")
    _validate_compatible(evaluations)
    groups = {
        group: _group_comparison(
            evaluations,
            baseline_label=baseline_label,
            group=group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for group in GROUPS
    }
    reference = evaluations[0]
    return {
        "baseline": baseline_label,
        "bootstrap": {
            "ci_quantiles": [0.025, 0.975],
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "utterance_paired",
        },
        "groups": groups,
        "inputs": [
            {
                "label": evaluation["label"],
                "rows_path": evaluation["rows_path"],
                "summary_path": evaluation["summary_path"],
            }
            for evaluation in sorted(evaluations, key=lambda evaluation: evaluation["label"])
        ],
        "protocol": reference["protocol"],
        "schema_version": 1,
        "selection": reference["selection"],
    }


def _format_delta(value: dict[str, Any]) -> str:
    low, high = value["ci95"]
    return f"{value['estimate']:+.6f} [{low:+.6f}, {high:+.6f}]"


def render_markdown(comparison: dict[str, Any], *, sort_by: str) -> str:
    """Render one sorted Markdown table per evaluation group."""

    if sort_by not in {"flow", "cosine"}:
        raise ValueError(f"Unknown table sort metric: {sort_by!r}")
    lines = [
        f"Baseline: `{comparison['baseline']}`",
        "",
        (
            "Flow deltas are candidate - baseline (negative is better); cosine deltas are candidate - baseline "
            "(positive is better)."
        ),
    ]
    for group in GROUPS:
        entries = comparison["groups"][group]
        if sort_by == "flow":
            entries = sorted(entries, key=lambda entry: (entry["flow_rank"], entry["label"]))
        else:
            entries = sorted(entries, key=lambda entry: (entry["hubert_cosine_rank"], entry["label"]))
        lines.extend(
            [
                "",
                f"### {group}",
                "",
                (
                    "| Flow rank | Cos rank | Label | Stage/update | Flow MSE ↓ | Δflow vs baseline (95% CI) | "
                    "HuBERT cosine ↑ | Δcos vs baseline (95% CI) |"
                ),
                "|---:|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for entry in entries:
            checkpoint = entry["checkpoint"]
            stage_update = f"{checkpoint.get('stage', '?')}/{checkpoint.get('update', '?')}"
            lines.append(
                f"| {entry['flow_rank']} | {entry['hubert_cosine_rank']} | `{entry['label']}` | "
                f"{stage_update} | {entry['flow_mse']:.6f} | {_format_delta(entry['flow_delta_vs_baseline'])} | "
                f"{entry['hubert_cosine']:.6f} | "
                f"{_format_delta(entry['hubert_cosine_delta_vs_baseline'])} |"
            )
    return "\n".join(lines) + "\n"


def _atomic_write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Comparison output already exists: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlink output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True, help="Evaluation label used as the paired baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--sort-by", choices=("flow", "cosine"), default="flow")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.summaries) < 2:
        parser.error("--summaries requires at least two files")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    comparison = compare_evaluations(
        args.summaries,
        baseline_label=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.output_json is not None:
        _atomic_write_json(args.output_json, comparison, overwrite=args.overwrite)
    print(render_markdown(comparison, sort_by=args.sort_by), end="")


if __name__ == "__main__":
    main()
