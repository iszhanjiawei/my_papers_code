"""Verify full original Setting2 coverage, conditioning and four metric aggregates."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import compute_measures

from aligndit.script.eval.infer_setting2_dubbing import PROTOCOL, PROJECT, model_text, source_hashes
from aligndit.script.eval.semantic_vae_decoder import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def word_distance(truth, hypo):
    a, b = truth.split(), hypo.split()
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def audit(output, benchmark):
    read = lambda path: json.loads(path.read_text())
    cfg = read(output / "run_config.json")
    inference = read(output / "inference_summary.json")
    require(inference["requested"] == inference["complete"] == 115 and not inference["failures"], "Incomplete generation")
    require(cfg["limit"] == 0 and cfg["use_ema"] and cfg["checkpoint_update"] == 150000, "Wrong checkpoint protocol")
    require(cfg["target_gt_text_read"] and not cfg["target_gt_audio_read"] and not cfg["target_acoustic_cache_read"] and not cfg["vsr_used"], "Wrong conditioning protocol")
    for key, value in {"seed": 0, "steps": 32, "cfg_t": 5, "cfg_v": 2, "sway": -1, "use_epss": True}.items():
        require(cfg[key] == value, "Wrong sampler setting: " + key)
    require(sha256_file(cfg["checkpoint"]) == cfg["checkpoint_sha256"], "Checkpoint changed")
    require(sha256_file(Path(cfg["checkpoint"]).parent / "speaker_training_contract.json") == cfg["training_contract_sha256"], "Training contract changed")
    require(source_hashes() == cfg["source_sha256"], "Inference implementation changed")
    config_hash = sha256_file(output / "run_config.json")
    require(inference["run_config_sha256"] == config_hash, "Inference summary config hash differs")
    require(sha256_file(cfg["manifest"]) == cfg["manifest_sha256"], "Manifest changed")
    rows = [json.loads(line) for line in (benchmark / "setting2/evaluation.jsonl").read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) == 115 and [row["id"] for row in inference["outputs"]] == ids, "Wrong original target coverage/order")
    hashes, samples, references, records = {}, 0, set(), []
    for row, receipt in zip(rows, inference["outputs"]):
        ident = row["id"]
        meta = read(output / (ident + ".json"))
        inputs = meta["inputs"]["row"]
        ref = meta["inputs"]["reference"]
        require(meta["inputs"]["run_config_sha256"] == config_hash, "Sample receipt config differs")
        require(inputs["id"] == ident and inputs["reference_id"] == row["reference_id"] != ident, "Different reference pairing")
        require(inputs["target_text"] == row["target_text"].strip().lower(), "Different target dialogue")
        require(inputs["model_text"] == model_text(inputs["reference_text"], inputs["target_text"]), "Reference transcript not concatenated")
        require(inputs["speaker_id"] == inputs["reference_speaker_id"] and inputs["protocol"] == PROTOCOL, "Wrong speaker identity/protocol")
        require(not {"target_audio", "target_latent", "gt_av_feature"}.intersection(inputs), "Target acoustics entered inference")
        for field, digest in inputs["input_sha256"].items():
            require(sha256_file(inputs[field]) == digest, "Inference input changed")
        require(ref["reference_audio_sha256"] == sha256_file(row["reference_audio"]), "Reference differs from original test")
        require(sha256_file(ref["cache_record_path"]) == ref["cache_record_sha256"], "Reference extraction receipt changed")
        for field in ("latent", "speaker"):
            require(sha256_file(ref[field + "_path"]) == ref[field + "_sha256"], "Reference features changed")
        references.add(ref["reference_id"])
        audio_path = output / (ident + ".wav")
        audio, sr = sf.read(audio_path, always_2d=True)
        require(sr == 16000 and audio.shape == (inputs["video_frames_25hz"] * 640, 1)
                and np.isfinite(audio).all() and np.square(audio).mean() > 0, "Invalid target-only WAV")
        hashes[ident] = sha256_file(audio_path)
        require(hashes[ident] == meta["audio_sha256"] == receipt["audio_sha256"], "WAV hash differs")
        require(meta["prompt_frames"] == ref["frames"] and meta["target_frames"] == math.ceil(len(audio) / 400), "Prompt/target frame offsets differ")
        samples += len(audio)
        records.append({"id": ident, "reference_id": row["reference_id"], "samples": len(audio), "sha256": hashes[ident]})
    require(samples == 8501120 and len(references) == 83, "Original corpus duration/reference count differs")
    metrics, counts = {}, {}
    for metric, key in (("spksim", "sim"), ("wer", "wer"), ("emosim", "emosim"), ("avsync", "avsync")):
        report = read(output / f"setting2_{metric}.json")
        require(report["n"] == 115 and [row["id"] for row in report["results"]] == ids, "Metric coverage/order differs")
        require(report["generated_audio_sha256"] == hashes, "Metric scored different audio")
        require(report["manifest_sha256"] == sha256_file(benchmark / "setting2/evaluation.jsonl"), "Metric manifest differs")
        require(report["evaluator_sha256"] == sha256_file(benchmark / "scripts/evaluate_setting2.py"), "Evaluator changed")
        for row, result in zip(rows, report["results"]):
            require(result["reference_id"] == row["reference_id"], "Metric reference differs")
        values = [result[key] for result in report["results"]]
        require(np.isfinite(values).all(), "Nonfinite score")
        if metric == "wer":
            result_rows = report["results"]
            require(all(result["raw_truth"] == row["target_text"] for row, result in zip(rows, result_rows)), "WER target text differs")
            measures = compute_measures([r["truth"] for r in result_rows], [r["hypo"] for r in result_rows])
            words = measures["hits"] + measures["substitutions"] + measures["deletions"]
            errors = sum(word_distance(r["truth"], r["hypo"]) for r in result_rows)
            require(words == 1721 and errors == sum(measures[k] for k in ("substitutions", "deletions", "insertions")), "WER independent edit count differs")
            score = errors / words * 100
            counts = {k: measures[k] for k in ("hits", "substitutions", "deletions", "insertions")}
            counts.update(errors=errors, reference_words=words)
        else:
            score = math.fsum(values) / 115
        if metric == "avsync":
            for row, result in zip(rows, report["results"]):
                gt = np.load(row["gt_av_feature"], allow_pickle=False).astype(np.float64)
                gen = np.load(output / "avhubert_feat" / (row["id"] + ".npy"), allow_pickle=False).astype(np.float64)
                require(gt.shape == gen.shape and gt.shape[1] == 1024 and np.isfinite(gen).all(), "Invalid joint AV feature")
                denom = np.maximum(np.linalg.norm(gt, axis=1), 1e-8) * np.maximum(np.linalg.norm(gen, axis=1), 1e-8)
                cosine = np.mean(np.sum(gt * gen, axis=1) / denom)
                require(abs(cosine - result[key]) < 2e-6, "Independent frame AVSync differs")
        require(abs(score - report["score"]) < 1e-10, "Aggregate metric differs")
        metrics[metric] = score
    result = {"complete": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "n": 115,
        "protocol": PROTOCOL, "checkpoint": cfg["checkpoint"], "checkpoint_sha256": cfg["checkpoint_sha256"],
        "use_ema": True, "duration_seconds": samples / 16000, "unique_reference_clips": len(references),
        "metrics": metrics, "corpus_wer_counts": counts, "run_config_sha256": config_hash,
        "audit_source_sha256": sha256_file(__file__), "peak_cuda_reserved_bytes": inference["peak_cuda_reserved_bytes"]}
    (output / "evaluation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "evaluation_per_clip.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=PROJECT.parents[1] / "Video-to-Speech-benchmark")
    args = parser.parse_args()
    audit(args.output_dir.resolve(), args.benchmark.resolve())
