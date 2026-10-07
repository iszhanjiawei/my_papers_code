"""Evaluate a complete generated GRID validation directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from jiwer import compute_measures

from aligndit.script.eval.utils import run_asr_wer, run_avsync, run_emosim
from f5_tts.eval.utils_eval import run_sim


NUMBER_WORDS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
SILENCE_TOKENS = frozenset(("sp", "sil"))


def normalize_grid_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text).lower()
    for digit, word in NUMBER_WORDS.items():
        text = text.replace(digit, " " + word)
    return " ".join(word for word in text.split() if word not in SILENCE_TOKENS)


def normalize_wer_result(row: dict) -> dict:
    result = dict(row)
    result["legacy_truth"], result["legacy_hypo"], result["legacy_wer"] = row["truth"], row["hypo"], row["wer"]
    result["truth"] = normalize_grid_text(row.get("raw_truth", row["truth"]))
    result["hypo"] = normalize_grid_text(row.get("raw_hypo", row["hypo"]))
    measures = compute_measures(result["truth"], result["hypo"])
    result.update({key: measures[key] for key in ("wer", "hits", "substitutions", "deletions", "insertions")})
    result["wer_protocol"] = "grid_digit_expansion_no_silence_corpus_v1"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--eval-task", choices=("sim", "wer", "emosim", "avsync"), required=True)
    parser.add_argument("-g", "--gen-wav-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wavlm-ckpt", type=Path)
    parser.add_argument("--asr-ckpt", type=Path)
    parser.add_argument("--emo-ckpt", type=Path)
    parser.add_argument("--gt-av-feat", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    args = parser.parse_args()
    if args.nshard < 1 or not 0 <= args.rank < args.nshard:
        parser.error("Require nshard >= 1 and 0 <= rank < nshard")

    all_records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if len(all_records) != 3_281:
        raise RuntimeError(f"Expected 3281 GRID samples, found {len(all_records)}")
    records = all_records[args.rank :: args.nshard]
    test_set = []
    for row in records:
        generated = args.gen_wav_dir / "test" / (row["id"] + ".wav")
        ground_truth = Path(row["gt_wav"])
        if not generated.is_file() or not ground_truth.is_file():
            raise FileNotFoundError(generated if not generated.is_file() else ground_truth)
        test_set.append((str(generated), str(ground_truth), row["text"]))

    if args.eval_task == "sim":
        results = run_sim((0, test_set, str(args.wavlm_ckpt)))
        metric = float(np.mean([row["sim"] for row in results]))
    elif args.eval_task == "wer":
        results = run_asr_wer((0, "en", test_set, str(args.asr_ckpt)))
        results = [
            dict(normalize_wer_result(result), id=row["id"], generated_wav=test[0])
            for result, row, test in zip(results, records, test_set)
        ]
        metric = compute_measures([row["truth"] for row in results], [row["hypo"] for row in results])["wer"]
    elif args.eval_task == "emosim":
        results = run_emosim((0, test_set, str(args.emo_ckpt)))
        metric = float(np.mean([row["emosim"] for row in results]))
    else:
        results = run_avsync((0, test_set, str(args.gt_av_feat), str(args.gen_wav_dir / "avhubert_feat")))
        metric = float(np.mean([row["avsync"] for row in results]))

    suffix = "" if args.nshard == 1 else f".rank{args.rank}"
    output = args.gen_wav_dir / f"_{args.eval_task}_results{suffix}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(f"\n{args.eval_task.upper()}: {round(metric, 5)}\n")
    print(f"{args.eval_task} shard {args.rank}/{args.nshard}: {len(results)} samples -> {output}", flush=True)


if __name__ == "__main__":
    main()
