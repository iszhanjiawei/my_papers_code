"""Summarize frozen InfoNCE diagnostics without treating frames as replicates.

Noise seeds are first averaged within each clip. Interventions are paired with
the correct-video condition, and source videos (with all their clips) are the
resampling units for exploratory, pointwise percentile confidence intervals.
These teacher-forced diagnostics are not ODE generation or AVSync evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIRED_METRICS = ("top1_full", "top1_gap5", "loss_gap5", "near_margin", "flow_mse")
IDENTIFIERS = {"utterance_key", "video_id", "noise_seed", "flow_t", "condition", "latent_frames", "anchors"}


def read_run(run_dir):
    """Require a complete factorial run, including recorded zero-anchor rows."""
    protocol = json.loads((run_dir / "protocol.json").read_text())
    seeds = protocol["noise_seeds"]
    times = protocol["flow_times"]
    conditions = protocol["conditions"]
    for name, values in (("noise_seeds", seeds), ("flow_times", times), ("conditions", conditions)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Protocol {name} must be nonempty and unique")
    if "correct" not in conditions:
        raise ValueError("A correct-video reference condition is required")
    rows = []
    seen = set()
    identities = {}
    anchor_counts = {}
    metric_names = None
    with (run_dir / "per_clip.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            clip = row["utterance_key"]
            key = (clip, row["noise_seed"], row["flow_t"], row["condition"])
            if key in seen:
                raise ValueError(f"Duplicate diagnostic row at line {line_number}: {key}")
            if row["noise_seed"] not in seeds or row["flow_t"] not in times or row["condition"] not in conditions:
                raise ValueError(f"Diagnostic row is outside protocol at line {line_number}: {key}")
            identity = (row["video_id"], row["latent_frames"])
            if clip in identities and identities[clip] != identity:
                raise ValueError(f"Inconsistent clip identity: {clip}")
            identities[clip] = identity
            anchors = row["anchors"]
            if type(anchors) is not int or anchors < 0:
                raise ValueError(f"Invalid anchor count: {key}")
            if clip in anchor_counts and anchor_counts[clip] != anchors:
                raise ValueError(f"Anchors differ across paired conditions/times/seeds: {clip}")
            anchor_counts[clip] = anchors
            if anchors:
                numeric = set(row) - IDENTIFIERS
                if metric_names is None:
                    metric_names = numeric
                if numeric != metric_names or not set(PAIRED_METRICS).issubset(numeric):
                    raise ValueError(f"Metric schema differs or required metrics are missing: {key}")
                if not all(isinstance(row[m], (int, float)) and np.isfinite(row[m]) for m in numeric):
                    raise ValueError(f"Nonfinite or nonnumeric metric: {key}")
            rows.append(row)
            seen.add(key)
    if len(identities) != protocol["test_count"]:
        raise ValueError(f"Incomplete clip coverage: {len(identities)} != {protocol['test_count']}")
    expected = len(identities) * len(seeds) * len(times) * len(conditions)
    if len(seen) != expected:
        raise ValueError(f"Incomplete paired design: {len(seen)} rows != {expected}")
    if not metric_names:
        raise ValueError("No positive-anchor clips are available for analysis")
    return protocol, rows, sorted(metric_names), anchor_counts


def average_seeds(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        if row["anchors"]:
            grouped[(row["utterance_key"], row["flow_t"], row["condition"])].append(row)
    averaged = {}
    for key, values in grouped.items():
        first = values[0]
        averaged[key] = {
            "utterance_key": key[0], "flow_t": key[1], "condition": key[2],
            "video_id": first["video_id"], "anchors": first["anchors"],
            "n_noise_seeds": len(values),
            **{metric: float(np.mean([row[metric] for row in values])) for metric in metrics},
        }
    return averaged


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd_across_clips": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


class VideoBootstrap:
    def __init__(self, repetitions, seed):
        if repetitions < 1:
            raise ValueError("Bootstrap repetitions must be positive")
        self.repetitions = repetitions
        self.seed = seed
        self.draw_cache = {}

    def interval(self, values, video_ids):
        """Resample equally likely videos; retain clip weighting within draws."""
        names = tuple(sorted(set(video_ids)))
        if len(names) < 2:
            return None, None
        if names not in self.draw_cache:
            rng = np.random.default_rng(self.seed)
            self.draw_cache[names] = rng.multinomial(
                len(names), np.full(len(names), 1.0 / len(names)), size=self.repetitions
            )
        lookup = {name: index for index, name in enumerate(names)}
        indices = np.asarray([lookup[name] for name in video_ids])
        sums = np.bincount(indices, weights=values, minlength=len(names))
        counts = np.bincount(indices, minlength=len(names))
        draws = self.draw_cache[names]
        bootstrapped = (draws @ sums) / (draws @ counts)
        low, high = np.quantile(bootstrapped, [0.025, 0.975])
        return float(low), float(high)


def paired_effect(averaged, clips, flow_t, condition, metric, bootstrap, reference_metric=None):
    reference_metric = reference_metric or metric
    current = [averaged[(clip, flow_t, condition)] for clip in clips]
    reference = [averaged[(clip, flow_t, "correct")] for clip in clips]
    differences = np.asarray([a[metric] - b[reference_metric] for a, b in zip(current, reference)])
    videos = [row["video_id"] for row in current]
    low, high = bootstrap.interval(differences, videos)
    return {
        "flow_t": flow_t, "condition": condition, "metric": metric,
        "reference_condition": "correct", "reference_metric": reference_metric,
        "contrast": "intervention_minus_correct" if metric == reference_metric else "correct_full_minus_gap5",
        "n_clips": len(clips), "n_source_videos": len(set(videos)),
        "anchors_total_per_seed": sum(row["anchors"] for row in current),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "sd_difference_across_clips": float(differences.std(ddof=1)) if len(clips) > 1 else None,
        "n_positive": int((differences > 0).sum()), "n_negative": int((differences < 0).sum()),
        "n_zero": int((differences == 0).sum()),
        "ci95_low": low, "ci95_high": high,
        "ci_status": "available" if low is not None else "unavailable_fewer_than_two_source_videos",
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_dir, repetitions=10000, seed=20261002):
    protocol, rows, metrics, anchor_counts = read_run(run_dir)
    averaged = average_seeds(rows, metrics)
    clips = sorted(clip for clip, count in anchor_counts.items() if count > 0)
    bootstrap = VideoBootstrap(repetitions, seed)
    groups, flat_groups, effects = [], [], []
    for flow_t in protocol["flow_times"]:
        for condition in protocol["conditions"]:
            values = [averaged[(clip, flow_t, condition)] for clip in clips]
            group = {
                "flow_t": flow_t, "condition": condition, "n_clips": len(clips),
                "n_source_videos": len({row["video_id"] for row in values}),
                "n_noise_seeds_per_clip": len(protocol["noise_seeds"]),
                "n_clip_seed_rows": sum(row["n_noise_seeds"] for row in values),
                "n_zero_anchor_clips_excluded": len(anchor_counts) - len(clips),
                "anchors_total_per_seed": sum(row["anchors"] for row in values),
                "metrics": {metric: describe([row[metric] for row in values]) for metric in metrics},
            }
            groups.append(group)
            flat_groups.append({
                **{key: value for key, value in group.items() if key != "metrics"},
                **{f"{metric}_{statistic}": value for metric, stats in group["metrics"].items()
                   for statistic, value in stats.items()},
            })
            if condition != "correct":
                for metric in PAIRED_METRICS:
                    effects.append(paired_effect(averaged, clips, flow_t, condition, metric, bootstrap))
        effects.append(paired_effect(
            averaged, clips, flow_t, "correct", "top1_full", bootstrap, reference_metric="top1_gap5"
        ))
    result = {
        "protocol": protocol,
        "analysis": {
            "schema_version": 1, "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "estimand": "Equal-clip mean after averaging the prescribed noise seeds within each clip",
            "resampling": "Paired source-video cluster bootstrap; all clips retained for each sampled video; clip-weighted mean per draw",
            "confidence_intervals": "Exploratory pointwise 95% percentile intervals; no multiplicity correction or p values",
            "replication": "Frames and noise seeds are not independent replicates; intervals exclude training-seed/checkpoint-selection uncertainty",
            "zero_anchors": "Clips with zero evaluation anchors excluded from all reported metrics, including flow metrics",
            "anchors_total_per_seed": "Unique eligible anchors summed over clips once; not multiplied by noise seeds or conditions",
            "interpretation": "Frozen teacher-forced flow-state diagnostic, not ODE generation, AVSync, or waveform lip-sync delay",
            "source_rows": len(rows), "included_clips": len(clips),
            "excluded_zero_anchor_clips": sorted(set(anchor_counts) - set(clips)),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "groups": groups, "paired_effects": effects,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(run_dir / "summary.csv", flat_groups)
    write_csv(run_dir / "paired_effects.csv", effects)
    write_csv(run_dir / "per_clip_seed_mean.csv", [averaged[key] for key in sorted(averaged)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261002)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.bootstrap_repetitions, args.bootstrap_seed)
    print(json.dumps({
        "run_dir": str(args.run_dir.resolve()), "groups": len(result["groups"]),
        "paired_effects": len(result["paired_effects"]),
        "included_clips": result["analysis"]["included_clips"],
        "excluded_zero_anchor_clips": result["analysis"]["excluded_zero_anchor_clips"],
    }))


if __name__ == "__main__":
    main()
