"""CPU-only, independent audit of the preregistered 132-clip GRID CFG ablation.

Selection, word edits and feature cosines are independently reconstructed. This
does not treat a diagnostic subset as a full validation evaluation, and never
imports the ablation generator or its metric implementation.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import wave

import jiwer
import numpy as np

from verify_grid_evaluation import (
    AVHUBERT_SHA, NORMALIZATION_SHA, VOCAB_SHA, file_stat, identity, normalize, publish,
    read, read_lines, require, sha, verify_checkpoint, verify_decoder,
    verify_fixed_architecture, verify_launch_sources, word_distance,
)


SELECTION_SALT = "cfgablation20261116:"
SELECTION_METHOD = "per-speaker smallest SHA256(salt + full_id), retain original val manifest order"
CHECKPOINT_SHA = "624b8f370a8e48060c382534d71dcc5a35065bcb10f516a2765ba4b6fe221984"
VAL_SHA = "c86309f5c5a0b9910608e74f54f82e71f0c30340b06b7c73dad6cb2a2b4a6230"
TASKS = ("sim", "wer", "emosim", "avsync")
STUDY = "grid_s1_repa100k_cfg_ablation132_20261116_v1"
VARIANTS = {
    "cfg_t5_v2": {"cfg_text": 5, "cfg_video": 2, "ignore_modality": None},
    "cfg_t3_v2": {"cfg_text": 3, "cfg_video": 2, "ignore_modality": None},
    "cfg_t1_v2": {"cfg_text": 1, "cfg_video": 2, "ignore_modality": None},
    "cfg_t5_v1": {"cfg_text": 5, "cfg_video": 1, "ignore_modality": None},
    "cfg_t5_v0": {"cfg_text": 5, "cfg_video": 0, "ignore_modality": None},
    "cfg_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": None},
    "drop_text_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": "text"},
    "drop_video_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": "video"},
}


def close(actual, expected, message, tolerance=1e-12):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and abs(actual - expected) <= tolerance, message)


def independent_summary(records, task):
    require(bool(records), f"Empty {task} records")
    if task == "wer":
        edits = sum(record["word_edit_distance"] for record in records)
        words = sum(record["reference_word_count"] for record in records)
        return {"count": len(records), "wer": edits / words,
                "word_edit_distance": edits, "reference_word_count": words,
                "aggregation": "corpus_word_edit_distance"}
    return {"count": len(records), task: math.fsum(record[task] for record in records) / len(records),
            "aggregation": "sample_mean"}


def require_summary(record, expected, label):
    for key, value in expected.items():
        require(record.get(key) == value, f"{label}: incorrect {key}")


def verify_selection(output, cache):
    selection_path = output / "selection.json"
    selection = read(selection_path)
    require(sha(cache / "manifests/val.jsonl") == VAL_SHA == selection["manifest_sha256"],
            "Validation manifest changed")
    rows = read_lines(cache / "manifests/val.jsonl")
    require(len(rows) == len({identity(row) for row in rows}) == 3274,
            "Expected 3274 unique validation identities")
    training = read_lines(cache / "manifests/train.jsonl")
    require(not {row["utterance_key"] for row in rows} & {row["utterance_key"] for row in training},
            "Train/validation overlap")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["speaker"]].append(row)
    require(len(groups) == 33, "Expected 33 validation speakers")
    score = lambda row: hashlib.sha256((SELECTION_SALT + identity(row)).encode()).hexdigest()
    selected = {identity(row) for group in groups.values()
                for row in sorted(group, key=lambda row: (score(row), identity(row)))[:4]}
    expected = [row for row in rows if identity(row) in selected]
    ids = [identity(row) for row in expected]
    counts = dict(collections.Counter(row["speaker"] for row in expected))
    require(len(expected) == len(selected) == 132 and set(counts.values()) == {4},
            "Selection is not exactly four per speaker")
    require(selection["salt"] == SELECTION_SALT and selection["method"] == SELECTION_METHOD
            and selection["selected_count"] == 132 and selection["speaker_counts"] == counts
            and selection["ids"] == ids and selection["rows"] == expected,
            "Selection differs from score-independent SHA256 selection")
    require(selection["selection_scores"] == {identity(row): score(row) for row in expected},
            "Selection hash scores differ")
    require(all(len(normalize(row["text"]).split()) == 6 for row in expected), "GRID dialogue is not six words")
    return expected, {"count": 132, "speakers": 33, "speaker_counts": counts,
                      "selection_sha256": sha(selection_path), "manifest_sha256": VAL_SHA,
                      "selection_independent_of_results": True}


def verify_cache(rows, cache):
    require(not (cache / "WRITE_ACTIVE.json").exists(), "GRID cache has an active writer")
    done = read(cache / "complete.json")
    require(done["complete"] is True and done["missing"] == done["invalid"] == []
            and done["selection"] == {"mode": "full"}, "Incomplete GRID cache")
    inventory_path = cache / "manifests/inventory_meta.json"
    require(done["inventory_meta_sha256"] == sha(inventory_path), "Stale inventory receipt")
    for name, entry in read(inventory_path)["manifests"].items():
        path = cache / "manifests" / name
        require(sha(path) == entry["sha256"] and len(read_lines(path)) == entry["count"],
                f"Changed cache manifest: {name}")
    hashes = {}
    for row in rows:
        key = identity(row)
        for stage, path in (("audio", cache / "audio" / row["audio_relative_path"]),
                            ("latent", cache / row["latent_relative_path"]),
                            ("video", cache / row["video_40hz_relative_path"]),
                            ("speaker", cache / "speakers" / Path(row["audio_relative_path"]).with_suffix(".npy"))):
            receipt = read(cache / "receipts" / stage / Path(row["audio_relative_path"]).with_suffix(".json"))
            digest = sha(path)
            require(receipt["artifact_sha256"] == digest and receipt["stage"] == stage
                    and receipt["utterance_key"] == row["utterance_key"]
                    and receipt["source_sha256"] == row["source_sha256"]
                    and receipt["extraction_sha256"] == done["extraction_sha256"], f"Invalid {stage} input: {key}")
            hashes[key, stage] = digest
            if stage != "audio":
                value = np.load(path, allow_pickle=False)
                shape = ((row["latent_frames"], 64) if stage == "latent" else
                         (row["latent_frames"], 1024) if stage == "video" else (192,))
                require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                        f"Invalid {stage} array: {key}")
                if stage != "video":
                    require(receipt["audio_sha256"] == hashes[key, "audio"], f"Unbound {stage}: {key}")
    return hashes


def verify_metric_results(variant_dir, rows, wav_hashes, cache_hashes, contract_sha, assets_sha):
    ids = [identity(row) for row in rows]
    summaries, all_results = {}, {}
    for task in TASKS:
        records = read_lines(variant_dir / f"{task}_results.jsonl")
        require([record["id"] for record in records] == ids, f"{variant_dir.name}/{task}: missing/reordered IDs")
        for row, record in zip(rows, records):
            key = identity(row)
            require(record["wav"] == key and math.isfinite(record[task]), f"Invalid {task} value: {key}")
            require(record["generated_audio_sha256"] == wav_hashes[key]
                    and record["reference_audio_sha256"] == cache_hashes[key, "audio"],
                    f"{task} audio identity mismatch: {key}")
            if task == "wer":
                require(record["raw_truth"] == row["text"], f"Wrong WER truth: {key}")
                truth, hypo = normalize(record["raw_truth"]), normalize(record["raw_hypo"])
                require((truth, hypo) == (record["truth"], record["hypo"]), f"Normalization mismatch: {key}")
                counts = jiwer.process_words(truth, hypo)
                errors = word_distance(truth.split(), hypo.split())
                require(errors == record["word_edit_distance"] == counts.substitutions + counts.deletions + counts.insertions
                        and (counts.substitutions, counts.deletions, counts.insertions)
                        == (record["substitutions"], record["deletions"], record["insertions"])
                        and record["reference_word_count"] == len(truth.split()) == 6
                        and record["wer"] == errors / 6, f"Independent WER differs: {key}")
        expected = independent_summary(records, task)
        summary = read(variant_dir / f"{task}_summary.json")
        require_summary(summary, expected, f"{variant_dir.name}/{task}")
        require(summary["inference_summary_sha256"] == sha(variant_dir / "inference_summary.json")
                and summary["metric_assets_sha256"] == assets_sha
                and summary["ablation_contract_sha256"] == contract_sha
                and summary["results_sha256"] == sha(variant_dir / f"{task}_results.jsonl")
                and summary["input_audio_hashes"] == {record["id"]: {
                    "generated": wav_hashes[record["id"]], "reference": cache_hashes[record["id"], "audio"]}
                    for record in records}, f"Unbound {task} summary")
        summaries[task], all_results[task] = expected, records
    return summaries, all_results


def verify_joint_features(variant_dir, rows, records, wav_hashes, cache_hashes):
    max_difference = 0.0
    for row, record in zip(rows, records):
        key = identity(row)
        arrays = []
        for role in ("gen", "gt"):
            path = variant_dir / "avhubert_joint" / role / f"{key}.npy"
            receipt = read(path.with_suffix(".json"))
            require(receipt["id"] == key and receipt["sha256"] == sha(path)
                    and receipt["audio_sha256"] == (wav_hashes[key] if role == "gen" else cache_hashes[key, "audio"])
                    and receipt["mouth_sha256"] == row["source"]["mouth_sha256"]
                    and receipt["checkpoint_sha256"] == AVHUBERT_SHA
                    and receipt["feature"] == "joint_audio_video_25hz_finetune_1024", f"Unbound {role} feature: {key}")
            array = np.load(path, allow_pickle=False)
            require(array.shape == (row["source_video_frames"], 1024) and array.dtype == np.float32
                    and np.isfinite(array).all(), f"Invalid {role} feature array: {key}")
            arrays.append(array.astype(np.float64))
        first, second = arrays
        denominator = np.maximum(np.linalg.norm(first, axis=1), 1e-8) * np.maximum(np.linalg.norm(second, axis=1), 1e-8)
        value = float(np.mean(np.sum(first * second, axis=1) / denominator))
        difference = abs(value - record["avsync"])
        require(difference < 1e-6, f"Independent AVSync differs: {key}")
        max_difference = max(max_difference, difference)
    expected = {identity(row) for row in rows}
    for role in ("gen", "gt"):
        root = variant_dir / "avhubert_joint" / role
        require({path.relative_to(root).with_suffix("").as_posix() for path in root.glob("val/*/*.npy")} == expected,
                f"{role} joint feature coverage differs")
    return max_difference


def verify_assets(output, baseline):
    path = output / "metric_assets.json"
    assets = read(path)
    historical = read(baseline / "metric_assets.json")
    require(assets["assets"] == historical["assets"] and assets["versions"] == historical["versions"],
            "Metric assets/software differ from the original N100 baseline")
    require(len(assets["assets"]) == 11, "Expected eleven pinned metric asset files")
    for entry in assets["assets"]:
        require(Path(entry["path"]).stat().st_size == entry["size_bytes"]
                and sha(entry["path"]) == entry["sha256"], f"Metric asset changed: {entry['path']}")
    for entry in assets["source_files"]:
        require(sha(entry["path"]) == entry["sha256"], f"Metric source changed: {entry['path']}")
    return {"path": str(path), "sha256": sha(path), "asset_count": 11,
            "source_files_rehashed": len(assets["source_files"]),
            "same_metric_assets_and_versions_as_historical_baseline": True}


def verify_waves(variant_dir, rows, protocol):
    inference_path = variant_dir / "inference_summary.json"
    inference = read(inference_path)
    ids = [identity(row) for row in rows]
    protocol_sha = sha(variant_dir / "protocol.json")
    require(inference["count"] == 132 and inference["complete"] is False
            and inference["complete_subset"] is True and inference["partial_smoke_test"] is True
            and inference["protocol"] == protocol["protocol"] and inference["protocol_sha256"] == protocol_sha
            and [record["id"] for record in inference["outputs"]] == ids,
            f"Invalid subset inference summary: {variant_dir.name}")
    hashes = {}
    for index, row in enumerate(rows):
        key = identity(row)
        path = variant_dir / row["audio_relative_path"]
        receipt = read(variant_dir / "receipts" / f"{key}.json")
        hashes[key] = sha(path)
        require(receipt["sha256"] == hashes[key] and receipt["protocol_sha256"] == protocol_sha
                and receipt["id"] == key and receipt["relative_path"] == row["audio_relative_path"]
                and receipt["samples"] == row["original_num_samples"] == 640 * row["source_video_frames"]
                and receipt == inference["outputs"][index], f"Invalid WAV receipt: {variant_dir.name}/{key}")
        with wave.open(str(path)) as stream:
            require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes())
                    == (16000, 1, 2, row["original_num_samples"]), f"Invalid WAV format/duration: {key}")
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
            require(len(samples) > 0 and bool(samples.any()), f"Empty/silent waveform: {key}")
    require({path.relative_to(variant_dir).with_suffix("").as_posix()
             for path in variant_dir.glob("val/*/*.wav")} == set(ids), "WAV full-ID coverage differs")
    return hashes


def historical_subset(baseline, rows, current_results, current_wav_hashes):
    ids = [identity(row) for row in rows]
    by_id = {identity(row): row for row in rows}
    original = read(baseline / "evaluation_summary.json")
    require(original["complete"] is True and original["count"] == 3274
            and original["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and original["checkpoint"]["update"] == 100000, "Historical comparison is not the N100 full evaluation")
    prior_verification = read(baseline / "verification.json")
    require(prior_verification["status"] == "PASS" and prior_verification["complete"] is True
            and prior_verification["evaluation_summary_sha256"] == sha(baseline / "evaluation_summary.json"),
            "Historical N100 evaluation has no valid PASS report")
    all_results, summaries, source_hashes = {}, {}, {}
    for task in TASKS:
        path = baseline / f"{task}_results.jsonl"
        complete = read_lines(path)
        require(len(complete) == len({record["id"] for record in complete}) == 3274,
                f"Incomplete historical {task} source")
        require(sha(path) == prior_verification["metrics"][task]["results_sha256"],
                f"Historical {task} results changed after verification")
        index = {record["id"]: record for record in complete}
        selected = [index[key] for key in ids]
        require(all(math.isfinite(record[task]) for record in selected), "Nonfinite historical metric")
        if task == "wer":
            for record in selected:
                require(record["raw_truth"] == by_id[record["id"]]["text"], "Historical target text differs")
                truth, hypo = normalize(record["raw_truth"]), normalize(record["raw_hypo"])
                require(word_distance(truth.split(), hypo.split()) == record["word_edit_distance"]
                        and record["reference_word_count"] == len(truth.split()) == 6,
                        "Historical WER independent recomputation differs")
        all_results[task] = selected
        summaries[task] = independent_summary(selected, task)
        source_hashes[task] = sha(path)
    same_wavs = 0
    for row in rows:
        key = identity(row)
        digest = sha(baseline / row["audio_relative_path"])
        require(read(baseline / "receipts" / f"{key}.json")["sha256"] == digest,
                f"Historical WAV changed: {key}")
        same_wavs += digest == current_wav_hashes[key]
    differences = {task: independent_summary(current_results[task], task)[task] - summaries[task][task] for task in TASKS}
    hypotheses_changed = sum(first["hypo"] != second["hypo"]
                             for first, second in zip(current_results["wer"], all_results["wer"]))
    return {"source": str(baseline), "count": 132, "historical_subset_metrics": summaries,
            "regenerated_minus_historical_subset": differences,
            "wav_hash_matches": same_wavs, "wav_hash_differences": 132 - same_wavs,
            "normalized_asr_hypotheses_changed": hypotheses_changed,
            "metric_source_sha256": source_hashes,
            "full_validation_metrics_are_not_replaced": True}


def verify_bound_file(entry, *, rehash=True):
    path = Path(entry["path"])
    before = file_stat(path)
    require(entry["stat"] == before, f"Bound file metadata changed: {path}")
    if rehash:
        require(sha(path) == entry["sha256"], f"Bound file contents changed: {path}")
        require(file_stat(path) == before, f"Bound file changed during verification: {path}")


def verify_source_contract(output, contract):
    project = Path(__file__).resolve().parents[1]
    required = {"scripts/grid_cfg_ablation.py", "scripts/run_grid_cfg_ablation.py",
                "scripts/verify_grid_cfg_ablation.py", "scripts/verify_grid_evaluation.py",
                "src/aligndit/model/cfm_vt.py", "src/aligndit/model/backbone/dit_vt_mm.py",
                "src/aligndit/config/finetune_grid.yaml",
                "src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py",
                "src/aligndit/script/eval/eval_grid_s1.py"}
    present = set()
    for entry in contract["source_files"]:
        path = Path(entry["path"]).resolve(strict=True)
        require(path.is_relative_to(project), f"Source is outside this project: {path}")
        present.add(path.relative_to(project).as_posix())
        verify_bound_file(entry)
    require(required <= present and len(present) == len(contract["source_files"]),
            "Source snapshot omits required executable/configuration or has duplicates")
    require(contract["source_snapshot_sha256"] == sha(output / "source_snapshot.json")
            and contract["launch_source_snapshot_sha256"] == sha(output / "launch_source_snapshot.json"),
            "Source snapshot inventories changed")
    snapshot = read(output / "source_snapshot.json")["files"]
    by_source = {entry["path"]: entry for entry in contract["source_files"]}
    require(len(snapshot) == len(by_source)
            and {entry["source_path"] for entry in snapshot} == set(by_source), "Incomplete source backups")
    for entry in snapshot:
        source = Path(entry["source_path"])
        expected = output / "source_snapshot" / source.relative_to(project)
        require(Path(entry["snapshot_path"]) == expected
                and entry["sha256"] == by_source[str(source)]["sha256"] == sha(expected),
                f"Source backup differs: {source}")
    launch = verify_launch_sources(output)
    launch_entries = read(output / "launch_source_snapshot.json")["source_files"]
    require({entry["path"] for entry in launch_entries} == present, "Launch source inventory is incomplete")
    return {"source_files_rehashed": len(present), "required_executables_present": True,
            "source_backups_rehashed": len(snapshot), "launch_snapshot": launch}


def verify_variant_protocol(output, name, contract, selection):
    protocol = read(output / name / "protocol.json")
    parameters = {"name": name, **VARIANTS[name]}
    generation = {"seed": 0, "nfe": 32, "ode": "euler", "epss": True,
                  "sway": -1, **VARIANTS[name], "batch_size": 1}
    require(protocol["protocol"] == "grid_val_setting1_same_clip_prompt_v1"
            and protocol["study"] == STUDY and protocol["requested_count"] == 132
            and protocol["partial_smoke_test"] is True and protocol["full_validation_count"] == 3274
            and protocol["ablation_contract_sha256"] == sha(output / "ablation_contract.json")
            and protocol["selection_sha256"] == selection["selection_sha256"]
            and protocol["variant"] == parameters and protocol["generation"] == generation
            and protocol["checkpoint"] == contract["checkpoint"]
            and protocol["config"] == contract["config"], f"Wrong variant protocol: {name}")
    require(protocol["reference"] == "same target clip complete canonical MPG audio; GT latent prompt plus same-clip CAM++"
            and protocol["model_text"] == "same transcript + two spaces + same transcript; length retained when text is dropped"
            and protocol["output"] == "only target continuation; prompt removed; exact original duration; 16k mono PCM16"
            and protocol["condition_drop"] == "internal trained null embedding; preserve sequence lengths, prompt, speaker and initial-noise seed",
            f"Changed reference/condition/output semantics: {name}")
    return protocol


def verify_condition_receipts(folder, rows, contract, name, cache_hashes):
    for row in rows:
        key = identity(row)
        receipt = read(folder / "receipts" / f"{key}.json")
        frames = row["latent_frames"]
        require(receipt["variant"] == {"name": name, **VARIANTS[name]}
                and receipt["model_text"] == row["text"] + "  " + row["text"]
                and receipt["prompt_frames"] == receipt["target_frames"] == frames
                and receipt["sampler_frames"] == 2 * frames
                and receipt["seed"] == 0 and receipt["nfe"] == 32
                and receipt["inputs"] == contract["input_bindings"][key],
                f"Wrong model conditions/geometry: {name}/{key}")
        require(receipt["video_mode"] == "zeros_like prefix then cached target video; internal null replacement iff ignore_modality=video"
                and receipt["speaker_mode"] == "same-clip cached CAM++; retained in conditional branch",
                f"Changed video/speaker semantics: {name}/{key}")
        for stage, entry in receipt["inputs"].items():
            require(stage in ("audio", "latent", "video", "speaker")
                    and entry["sha256"] == cache_hashes[key, stage], f"Unbound model input: {name}/{key}/{stage}")


def paired_errors(records, baseline):
    changes = [record["word_edit_distance"] - previous["word_edit_distance"]
               for record, previous in zip(records, baseline)]
    return {"delta_word_errors": sum(changes), "delta_WER_percentage_points": 100 * sum(changes) / 792,
            "utterances_worse_equal_better": [sum(value > 0 for value in changes),
                                               sum(value == 0 for value in changes),
                                               sum(value < 0 for value in changes)],
            "by_id": {record["id"]: value for record, value in zip(records, changes)}}


def pcm(path):
    with wave.open(str(path)) as stream:
        require(stream.getsampwidth() == 2 and stream.getnchannels() == 1 and stream.getframerate() == 16000,
                "Historical/current waveform is not 16-kHz PCM16 mono")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def verify_historical_claims(claim, baseline, rows, current_folder, current_results, computed):
    require(claim["historical_root"] == str(baseline) and claim["subset_count"] == 132
            and claim["historical_subset_metrics"] == computed["historical_subset_metrics"]
            and claim["current_subset_metrics"] == {task: independent_summary(current_results[task], task) for task in TASKS}
            and claim["metric_deltas_current_minus_historical"] == computed["regenerated_minus_historical_subset"]
            and claim["waveform_file_identical_count"] == computed["wav_hash_matches"]
            and claim["ASR_normalized_identical_count"] == 132 - computed["normalized_asr_hypotheses_changed"],
            "Historical baseline comparison summary differs from independent audit")
    historical_index = {record["id"]: record for record in read_lines(baseline / "wer_results.jsonl")}
    historical_wer = [historical_index[identity(row)] for row in rows]
    require(claim["paired_word_errors"] == paired_errors(current_results["wer"], historical_wer),
            "Historical paired word errors differ")
    ids = [identity(row) for row in rows]
    require([record["id"] for record in claim["waveforms"]] == ids
            and [record["id"] for record in claim["ASR"]] == ids, "Historical comparison identity mismatch")
    identical_pcm = 0
    for row, waveform, asr, current, old in zip(rows, claim["waveforms"], claim["ASR"], current_results["wer"], historical_wer):
        new_path, old_path = current_folder / row["audio_relative_path"], baseline / row["audio_relative_path"]
        new_audio, old_audio = pcm(new_path), pcm(old_path)
        require(new_audio.shape == old_audio.shape, "Historical/current audio shapes differ")
        difference = new_audio - old_audio
        same = bool(np.array_equal(new_audio, old_audio))
        identical_pcm += same
        require(waveform["current_sha256"] == sha(new_path) and waveform["historical_sha256"] == sha(old_path)
                and waveform["file_identical"] == (waveform["current_sha256"] == waveform["historical_sha256"])
                and waveform["pcm_identical"] == same, "Historical waveform equality claim differs")
        close(waveform["max_abs_difference"], float(np.abs(difference).max()), "Historical max audio difference wrong")
        close(waveform["rmse"], float(np.sqrt(np.mean(difference ** 2))), "Historical audio RMSE wrong")
        require(asr["current_raw_hypo"] == current["raw_hypo"] and asr["historical_raw_hypo"] == old["raw_hypo"]
                and asr["current_hypo"] == current["hypo"] and asr["historical_hypo"] == old["hypo"]
                and asr["normalized_identical"] == (current["hypo"] == old["hypo"])
                and asr["delta_word_errors"] == current["word_edit_distance"] - old["word_edit_distance"],
                "Historical per-sample ASR comparison differs")
    require(claim["waveform_pcm_identical_count"] == identical_pcm, "Historical PCM count differs")
    computed["waveform_pcm_identical_count"] = identical_pcm


def verify(output, cache=None):
    contract = read(output / "ablation_contract.json")
    contract_sha = sha(output / "ablation_contract.json")
    require(contract["study"] == STUDY and contract["protocol"] == "grid_val_setting1_same_clip_prompt_v1"
            and contract["diagnostic_subset_only"] is True and contract["count"] == 132
            and contract["full_validation_count"] == 3274 and contract["variants"] == VARIANTS,
            "Wrong ablation contract, counts or variant definitions")
    cache = Path(cache or contract["cache_root"]).resolve(strict=True)
    require(str(cache) == contract["cache_root"], "Requested cache differs from contract")
    baseline = Path(contract["historical_root"]).resolve(strict=True)
    rows, selection = verify_selection(output, cache)
    require(contract["selection_sha256"] == selection["selection_sha256"], "Unbound selection")
    source_verification = verify_source_contract(output, contract)
    for entry in contract["resource_bindings"]:
        verify_bound_file(entry)
    for entry in contract["asset_bindings"]:
        verify_bound_file(entry, rehash=False)
    for entry in contract["historical_bindings"].values():
        verify_bound_file(entry)
    cache_hashes = verify_cache(rows, cache)
    require(set(contract["input_bindings"]) == {identity(row) for row in rows}, "Input bindings differ from selection")
    for key, entries in contract["input_bindings"].items():
        require(set(entries) == {"audio", "latent", "video", "speaker"}, "Missing model input binding")
        for stage, entry in entries.items():
            verify_bound_file(entry)
            require(entry["sha256"] == cache_hashes[key, stage], f"Input does not match source cache: {key}/{stage}")
    require(contract["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and contract["checkpoint"]["update"] == 100000
            and contract["checkpoint"]["weights"] == "EMA"
            and contract["checkpoint"]["state_tensor_count"] == 808, "Wrong checkpoint/EMA metadata")
    verify_bound_file(contract["checkpoint"], rehash=False)
    checkpoint = verify_checkpoint(contract)
    require(checkpoint["ema_state_entries"] == 810 and checkpoint["ema_step"] == 150000,
            "Expected 808 model tensors plus EMA metadata at inherited step 150000")
    architecture_protocol = {**contract, "normalization_sha256": NORMALIZATION_SHA}
    architecture = verify_fixed_architecture(architecture_protocol)
    assets = verify_assets(output, baseline)
    require(contract["metric_assets_sha256"] == assets["sha256"], "Metric asset report changed")
    print("Verified selection, input cache, sources, checkpoint and metric assets", flush=True)
    all_metrics, all_results, all_hashes = {}, {}, {}
    av_differences = {}
    decoder = None
    for name in VARIANTS:
        folder = output / name
        protocol = verify_variant_protocol(output, name, contract, selection)
        hashes = verify_waves(folder, rows, protocol)
        verify_condition_receipts(folder, rows, contract, name, cache_hashes)
        metrics, records = verify_metric_results(folder, rows, hashes, cache_hashes, contract_sha, assets["sha256"])
        av_differences[name] = verify_joint_features(folder, rows, records["avsync"], hashes, cache_hashes)
        if decoder is None:
            decoder = verify_decoder(folder, cache, 1)
        else:
            require(read(folder / "decoder_shard0.json") == read(output / "cfg_t5_v2/decoder_shard0.json"),
                    f"Decoder differs for {name}")
        all_metrics[name], all_results[name], all_hashes[name] = metrics, records, hashes
        print(f"Verified {name}: 132 WAVs, four metrics, 264 joint features; WER={metrics['wer']['wer']:.9f}", flush=True)
    summary = read(output / "ablation_summary.json")
    require(summary["study"] == STUDY and summary["complete_subset"] is True
            and summary["full_validation_complete"] is False and summary["diagnostic_subset_only"] is True
            and summary["selected_count"] == 132 and summary["full_validation_count"] == 3274
            and summary["variant_count"] == 8 and summary["generated_wavs"] == 1056
            and summary["ablation_contract_sha256"] == contract_sha
            and summary["selection_sha256"] == selection["selection_sha256"]
            and summary["checkpoint"] == contract["checkpoint"] and summary["variants"] == VARIANTS
            and summary["metrics"] == all_metrics, "Final summary differs from independently recomputed results")
    paired = {name: {reference: paired_errors(all_results[name]["wer"], all_results[reference]["wer"])
                     for reference in ("cfg_t5_v2", "cfg_t0_v0")} for name in VARIANTS}
    require(summary["paired_word_error_changes"] == paired, "Final paired WER changes differ")
    historical = historical_subset(baseline, rows, all_results["cfg_t5_v2"], all_hashes["cfg_t5_v2"])
    verify_historical_claims(summary["historical_baseline_comparison"], baseline, rows, output / "cfg_t5_v2",
                             all_results["cfg_t5_v2"], historical)
    result = {"status": "PASS", "complete_subset": True, "full_validation_complete": False,
              "study": STUDY, "selection": selection, "variant_count": 8,
              "requested_wavs": 1056, "succeeded_wavs": 1056, "failed_wavs": 0,
              "metrics": all_metrics, "paired_word_error_changes": paired,
              "joint_feature_counts": {"gen": 1056, "gt": 1056},
              "max_numpy_avsync_recompute_difference_by_variant": av_differences,
              "checkpoint_verification": checkpoint, "architecture_verification": architecture,
              "source_verification": source_verification, "metric_asset_verification": assets,
              "decoder_verification": decoder, "historical_baseline_comparison": historical,
              "ablation_contract_sha256": contract_sha, "ablation_summary_sha256": sha(output / "ablation_summary.json"),
              "verification_script_sha256": sha(__file__),
              "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    publish(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    try:
        result = verify(output, args.cache_root)
    except Exception as error:
        publish(output / "verification.json", {"status": "FAIL", "complete_subset": False,
                "full_validation_complete": False, "error": str(error),
                "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
