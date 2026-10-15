"""Check full target-only ablation coverage, fixed conditions, and corpus WER."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def s1_imported_code_hash(source):
    """Ignore only S1 CLI/run functions, neither of which Setting2 calls.

    The baseline file is git blob 2cc463b at commit 7032540; commit a90188a
    subsequently added S1 sharding and memory reporting only to those functions.
    All module imports, globals, and imported helper definitions remain checked.
    """
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"run", "parse_args"})]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def word_edits(truth, hypothesis):
    """Independent Levenshtein distance over already normalized ASR word lists."""
    reference, predicted = truth.split(), hypothesis.split()
    previous = list(range(len(predicted) + 1))
    for i, actual in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(predicted, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (actual != candidate)))
        previous = current
    return previous[-1], len(reference)


def verify(generated, baseline, evaluator):
    new = read_json(generated / "run_config.json")
    old = read_json(baseline / "run_config.json")
    require(new["text_input_mode"] == "target-only", "Not target-only inference")
    require(new["reference_text_conditioned"] is False, "Reference text was conditioned")
    require(new["text_source"] == "supplied_target_GT_dialogue_only", "Wrong target text source")
    invariant_keys = set(old) - {"source_sha256", "text_source", "shared_implementation_sha256"}
    for key in invariant_keys:
        require(new[key] == old[key], f"Historical inference configuration changed: {key}")
    wrapper = "src/aligndit/script/eval/infer_celebvdub_setting2_dubbing.py"
    s1_helper = "src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py"
    project = Path(__file__).resolve().parents[4]
    require(new["source_sha256"].keys() == old["source_sha256"].keys(), "Source inventory changed")
    for path, digest in old["source_sha256"].items():
        if path == s1_helper:
            require(digest == "7fdc7dfddadce612016a54c377495d288a441a432e3c25939237a232a3108d95",
                    "Unexpected historical S1 helper source")
            require(sha256(project / path) == new["source_sha256"][path], "S1 helper changed after inference")
            require(s1_imported_code_hash((project / path).read_text())
                    == "23deaf10b5c672d228a9dfc4ff931a9b25e8202896295c819b0d1358622a380b",
                    "Imported S1 helper behavior differs from historical baseline")
        elif path != wrapper:
            require(new["source_sha256"][path] == digest, f"Non-text inference source changed: {path}")
        require(sha256(project / path) == new["source_sha256"][path], f"Source changed after inference: {path}")
    require(sha256(new["checkpoint"]) == new["checkpoint_sha256"], "Checkpoint hash changed")
    require(sha256(new["config"]) == new["config_sha256"], "Model configuration hash changed")
    require(sha256(new["manifest"]) == new["manifest_sha256"], "Pair manifest hash changed")
    rows = [json.loads(line) for line in Path(new["manifest"]).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) == 115, "Require all original 115 targets")
    summary = read_json(generated / "inference_summary.json")
    require(summary["requested"] == summary["complete"] == 115 and not summary["failures"],
            "Generation is incomplete")
    config_hash = sha256(generated / "run_config.json")
    require(summary["run_config_sha256"] == config_hash, "Summary provenance changed")
    outputs = {row["id"]: row for row in summary["outputs"]}
    require(len(summary["outputs"]) == len(outputs) == 115 and set(outputs) == set(ids),
            "Summary target coverage mismatch")
    new_metric = read_json(generated / "setting2_wer.json")
    old_metric = read_json(baseline / "setting2_wer.json")
    require(new_metric["evaluator_sha256"] == old_metric["evaluator_sha256"] == sha256(evaluator),
            "WER evaluator differs from baseline")
    for key in ("metric", "units", "n", "protocol", "reference_policy", "manifest_sha256", "baseline_source"):
        require(new_metric[key] == old_metric[key], f"WER protocol changed: {key}")
    results = {row["id"]: row for row in new_metric["results"]}
    old_results = {row["id"]: row for row in old_metric["results"]}
    require(len(new_metric["results"]) == len(results) == 115 and set(results) == set(ids),
            "WER target coverage mismatch")
    require(set(new_metric["generated_audio_sha256"]) == set(ids), "WER audio hash coverage mismatch")
    records, errors, words, old_errors, old_words, total_samples = [], 0, 0, 0, 0, 0
    reference_ids = set()
    for row in rows:
        ident = row["id"]
        meta = read_json(generated / (ident + ".json"))
        prior = read_json(baseline / (ident + ".json"))
        require(meta["id"] == ident and meta["text_input_mode"] == "target-only", "Wrong sample mode/ID")
        require(meta["inputs"]["row"] == prior["inputs"]["row"], f"Pair inputs changed: {ident}")
        require(meta["inputs"]["run_config_sha256"] == config_hash, "Sample config hash changed")
        target = meta["inputs"]["row"]["target_text"]
        require(meta["model_text"] == target, f"Model text includes non-target content: {ident}")
        text_file = meta["inputs"]["row"]["target_text_path"]
        require(Path(text_file).read_text().strip().lower() == target, "Target text file changed")
        for field, digest in meta["inputs"]["row"]["input_sha256"].items():
            require(sha256(meta["inputs"]["row"][field]) == digest, f"Input file changed: {ident}/{field}")
        for key in ("samples", "sample_rate", "prompt_frames", "target_frames", "sampler_frames", "text_extension_frames"):
            require(meta[key] == prior[key], f"Audio/video geometry changed: {ident}/{key}")
        reference = meta["inputs"]["reference"]
        prior_reference = prior["inputs"]["reference"]
        require(reference["reference_id"] == prior_reference["reference_id"] == row["reference_id"],
                "Reference pairing changed")
        reference_ids.add(reference["reference_id"])
        for key in ("reference_audio_sha256", "posterior_seed", "frames", "samples"):
            require(reference[key] == prior_reference[key], f"Reference extraction changed: {ident}/{key}")
        for key in ("latent", "speaker"):
            require(sha256(reference[key + "_path"]) == reference[key + "_sha256"]
                    == prior_reference[key + "_sha256"] == sha256(prior_reference[key + "_path"]),
                    f"Reference condition differs: {ident}/{key}")
        wav = generated / (ident + ".wav")
        waveform, rate = sf.read(wav, always_2d=True)
        require(rate == 16000 and waveform.shape == (meta["samples"], 1)
                and np.isfinite(waveform).all() and np.mean(waveform ** 2) > 0,
                f"Invalid generated waveform: {ident}")
        digest = sha256(wav)
        require(digest == meta["audio_sha256"] == outputs[ident]["audio_sha256"]
                == new_metric["generated_audio_sha256"][ident], f"Output WAV hash mismatch: {ident}")
        require(outputs[ident]["samples"] == meta["samples"] and outputs[ident]["sample_rate"] == rate,
                "Summary audio geometry changed")
        result, old_result = results[ident], old_results[ident]
        require(result["raw_truth"] == target and result["truth"] == old_result["truth"],
                f"Scoring target text changed: {ident}")
        require(result["reference_id"] == row["reference_id"], "WER reference ID changed")
        edits, nwords = word_edits(result["truth"], result["hypo"])
        before, before_words = word_edits(old_result["truth"], old_result["hypo"])
        require(nwords > 0 and abs(edits / nwords - result["wer"]) < 1e-12, "Per-sentence WER mismatch")
        errors += edits
        words += nwords
        old_errors += before
        old_words += before_words
        total_samples += len(waveform)
        records.append({"id": ident, "errors": edits, "reference_words": nwords,
                        "baseline_errors": before, "audio_sha256": digest})
    require(words == old_words == 1721 and len(reference_ids) == 83, "Original corpus changed")
    wer = errors / words * 100
    baseline_wer = old_errors / old_words * 100
    require(abs(wer - new_metric["score"]) < 1e-12
            and abs(baseline_wer - old_metric["score"]) < 1e-12, "Corpus WER recomputation mismatch")
    report = {"complete": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "targets": 115, "unique_reference_clips": len(reference_ids), "text_input_mode": "target-only",
              "checkpoint_update": new["checkpoint_update"], "reference_condition_hashes_match_baseline": True,
              "all_nontext_configuration_matches_baseline": True, "audio_geometry_matches_baseline": True,
              "s1_helper_source_audit": "S1-only run/parse_args changed; all imported helpers and module globals match historical AST",
              "target_audio_seconds": total_samples / 16000, "reference_words": words,
              "errors": errors, "wer_percent": wer, "baseline_errors": old_errors,
              "baseline_wer_percent": baseline_wer, "delta_percentage_points": wer - baseline_wer,
              "generated": str(generated), "baseline": str(baseline),
              "run_config_sha256": config_hash, "metric_report_sha256": sha256(generated / "setting2_wer.json"),
              "verifier_sha256": sha256(__file__), "samples": records}
    (generated / "target_only_wer_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    verify(args.generated.resolve(), args.baseline.resolve(), args.evaluator.resolve())


if __name__ == "__main__":
    main()
