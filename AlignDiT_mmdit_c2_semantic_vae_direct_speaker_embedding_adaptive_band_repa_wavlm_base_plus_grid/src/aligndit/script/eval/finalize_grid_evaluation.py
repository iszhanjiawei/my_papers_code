"""Verify all GRID generation/evaluation shards and publish the metric summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import compute_measures


def read_result_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-shards", type=int, required=True)
    parser.add_argument("--wer-shards", type=int, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if len(records) != 3_281:
        raise RuntimeError("Formal GRID result must contain 3,281 manifest records")

    summaries = [
        json.loads((args.output / f"inference_summary.rank{rank}.json").read_text())
        for rank in range(args.generation_shards)
    ]
    outputs = [row for summary in summaries for row in summary["outputs"]]
    if len(outputs) != len(records) or {row["id"] for row in outputs} != {row["id"] for row in records}:
        raise RuntimeError("Inference summaries do not cover the exact GRID validation manifest")
    for row in records:
        waveform, rate = sf.read(args.output / "test" / (row["id"] + ".wav"), always_2d=True)
        if rate != 16_000 or waveform.shape not in {(47_360, 1), (48_000, 1)} or not np.isfinite(waveform).all():
            raise RuntimeError(f"Invalid generated waveform: {row['id']}")

    metrics = {}
    for task in ("sim", "emosim", "avsync"):
        rows = read_result_rows([args.output / f"_{task}_results.jsonl"])
        if len(rows) != len(records):
            raise RuntimeError(f"Incomplete {task}: {len(rows)}")
        metrics[task] = round(float(np.mean([row[task] for row in rows])), 5)
    wer_paths = [args.output / f"_wer_results.rank{rank}.jsonl" for rank in range(args.wer_shards)]
    wer_rows = read_result_rows(wer_paths)
    if len(wer_rows) != len(records) or {row["id"] for row in wer_rows} != {row["id"] for row in records}:
        raise RuntimeError("WER shards do not cover the exact GRID validation manifest")
    metrics["wer"] = round(
        compute_measures([row["truth"] for row in wer_rows], [row["hypo"] for row in wer_rows])["wer"], 5
    )
    summary = {
        "status": "complete",
        "dataset": "GRID",
        "samples": len(records),
        "setting": summaries[0]["setting"],
        "checkpoint": summaries[0]["checkpoint"],
        "generation": summaries[0]["generation"],
        "protocol": summaries[0]["protocol"],
        "metrics": {
            "spksim": metrics["sim"],
            "wer": metrics["wer"],
            "emosim": metrics["emosim"],
            "avsync": metrics["avsync"],
        },
    }
    (args.output / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "metrics_summary.txt").write_text(
        "\n".join(f"{key.upper()}: {value:.5f}" for key, value in summary["metrics"].items()) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
