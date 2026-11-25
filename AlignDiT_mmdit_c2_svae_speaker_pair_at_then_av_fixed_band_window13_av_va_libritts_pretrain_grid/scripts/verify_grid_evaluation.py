"""Independently verify a complete fixed-13 GRID evaluation without GPU work.

This intentionally imports no inference/evaluation implementation. All 3,274
identities, word edit counts, feature bindings and metric means are recomputed.
The optional checkpoint receipt avoids rereading an unchanged, already audited
6-GB checkpoint; by default the actual checkpoint bytes are always rehashed.
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
import re
import subprocess
import unicodedata
import wave

import jiwer
import numpy as np


S1 = "grid_val_setting1_same_clip_prompt_v1"
S2 = "grid_val_setting2_same_speaker_cross_utterance_gttext_v1"
AVHUBERT_SHA = "343c9dbaa29847b8c4d0d9503c43f2877fd2efde9988d41abf14e74f28232b75"
NORMALIZATION_SHA = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
VOCAB_SHA = "225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d"
VAE_SHA = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
GENERATION = {"seed": 0, "nfe": 32, "ode": "euler", "epss": True,
              "sway": -1, "cfg_text": 5, "cfg_video": 2, "batch_size": 1}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    text = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", text)
    digits = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
    return " ".join(digits.get(word, word) for word in text.split())


def word_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1]


def identity(row):
    relative = Path(row["audio_relative_path"])
    require(not relative.is_absolute() and ".." not in relative.parts
            and len(relative.parts) == 3 and relative.parts[0] == "val"
            and relative.suffix == ".wav", "Unsafe validation identity")
    require(row["utterance_key"] == "grid/" + relative.with_suffix("").relative_to("val").as_posix(),
            "GRID utterance identity differs from audio path")
    return relative.with_suffix("").as_posix()


def file_stat(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def verify_checkpoint(protocol, reuse_path=None):
    checkpoint = protocol["checkpoint"]
    path = Path(checkpoint["path"]).resolve(strict=True)
    before = file_stat(path)
    require(checkpoint["weights"] == "EMA" and checkpoint["update"] > 0,
            "Expected a positive-update EMA checkpoint")
    require(path.name == f"model_{checkpoint['update']}.pt", "Checkpoint name/update mismatch")
    verification = {"path": str(path), "sha256": checkpoint["sha256"], "stat": before}
    if reuse_path:
        previous = read(reuse_path)
        prior = previous.get("checkpoint_verification", {})
        require(previous.get("status") == "PASS" and previous.get("complete") is True
                and prior.get("path") == str(path) and prior.get("sha256") == checkpoint["sha256"]
                and prior.get("stat") == before, "Reusable checkpoint verification is stale or incomplete")
        verification.update(rehashed=False, reused_verification=str(Path(reuse_path).resolve()),
                            reused_verification_sha256=sha(reuse_path))
    else:
        require(sha(path) == checkpoint["sha256"], "Checkpoint byte hash differs from generation protocol")
        verification["rehashed"] = True
    # mmap reads the actual archive's metadata without allocating optimizer/model
    # tensors on a GPU. EMA's inherited parent step may exceed downstream update.
    import torch
    checkpoint_metadata = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    require(checkpoint_metadata.get("update") == checkpoint["update"], "Actual checkpoint update mismatch")
    architecture = read(path.parent / "speaker_training_contract.json")["checkpoint_architecture"]
    require(checkpoint_metadata.get("architecture") == architecture,
            "Actual checkpoint architecture differs from the GRID training contract")
    ema = checkpoint_metadata.get("ema_model_state_dict", {})
    require(bool(ema.get("initted", False)), "Actual checkpoint EMA is uninitialized")
    verification.update(update=checkpoint_metadata["update"], ema_initialized=True,
                        ema_step=int(ema["step"]), ema_state_entries=len(ema))
    del checkpoint_metadata, ema
    require(file_stat(path) == before, "Checkpoint changed while being audited")
    return verification


def verify_fixed_architecture(protocol):
    config = protocol["config"]
    arch = config["model"]["arch"]
    contract_path = Path(protocol["checkpoint"]["path"]).parent / "speaker_training_contract.json"
    sidecar = read(contract_path)
    stored_arch = sidecar["config"]["model"]["arch"]
    # Dataset hashes are resolved during training, while generic evaluation YAML
    # leaves them null. Architecture must nevertheless match in its entirety.
    require(arch == stored_arch, "Generation architecture differs from the training sidecar")
    expected = {"depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
                "audio_video_ratio": 1, "text_joint_attention": True, "paired_joint_order": "at_then_av",
                "temporal_band_enabled": True, "temporal_band_mode": "fixed",
                "temporal_band_audio_fps": 40.0, "temporal_band_video_fps": 40.0,
                "temporal_band_fixed_offset_seconds": 0.0,
                "temporal_band_fixed_sigma_seconds": 0.15,
                "temporal_band_bidirectional": True}
    architecture = sidecar["checkpoint_architecture"]
    require(architecture.get("dataset") == "GRID"
            and architecture.get("parent_stage") == "paired_libritts_text_audio",
            "Wrong dataset or LibriTTS parent for the checkpoint")
    for key, value in expected.items():
        require(type(arch.get(key)) is type(value) and arch[key] == value, f"Fixed-13 mismatch: {key}")
        if key.startswith("temporal_band_") or key == "paired_joint_order":
            require(type(architecture.get(key)) is type(value) and architecture[key] == value,
                    f"Fixed-band sidecar contradicts {key}")
    require(sha(config["datasets"]["normalization_path"]) == protocol["normalization_sha256"] == NORMALIZATION_SHA,
            "Pretrained latent normalization changed")
    require(sha(config["datasets"]["vocab_path"]) == VOCAB_SHA, "Pretrained vocabulary changed")
    return {"path": str(contract_path), "sha256": sha(contract_path), "architecture_exact_match": True,
            "fixed_band_sigma_seconds": 0.15, "fixed_band_offset_seconds": 0.0}


def reference_assignment(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["speaker"]].append(row)
    assignment = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (hashlib.sha256(("0:" + identity(row)).encode()).digest(), identity(row)))
        for offset in range(1, len(ordered)):
            if all(normalize(row["text"]) != normalize(ordered[(i + offset) % len(ordered)]["text"])
                   for i, row in enumerate(ordered)):
                assignment.update({identity(row): identity(ordered[(i + offset) % len(ordered)])
                                   for i, row in enumerate(ordered)})
                break
        else:
            raise RuntimeError("Cannot independently reconstruct reference permutation")
    return assignment


def verify_decoder(output, cache, nshard):
    spec = read(cache / "state/latents/spec.json")
    checkpoint_contract, source = spec["checkpoint"], spec["semantic_vae_source"]
    records = [read(output / f"decoder_shard{shard}.json") for shard in range(nshard)]
    require(all(record == records[0] for record in records), "Decoder differs between inference shards")
    record = records[0]
    checkpoint = Path(record["checkpoint"])
    metadata = checkpoint.parent.parent / "metainfo.json"
    require(record["checkpoint_sha256"] == checkpoint_contract["ema_sha256"] == VAE_SHA == sha(checkpoint)
            and record["ema_step"] == checkpoint_contract["ema_step"]
            and record["metainfo_sha256"] == checkpoint_contract["metainfo_sha256"] == sha(metadata),
            "Decoder differs from the latent extraction contract")
    require(record["git_commit"] == source["commit"], "Semantic-VAE source revision differs")
    actual_revision = subprocess.check_output(["git", "-C", record["repo"], "rev-parse", "HEAD"], text=True).strip()
    source_status = subprocess.check_output(["git", "-C", record["repo"], "status", "--porcelain", "--untracked-files=normal"], text=True).strip()
    require(actual_revision == source["commit"] and not source_status, "Semantic-VAE source changed")
    config_path = Path(read(metadata)["DAC"]["bigvgan_conf"])
    if not config_path.is_absolute():
        config_path = Path(record["repo"]) / config_path
    require(sha(config_path) == source["bigvgan_config_sha256"], "Semantic-VAE BigVGAN config changed")
    return {"inference_shards": nshard, "checkpoint": str(checkpoint), "checkpoint_sha256": VAE_SHA,
            "ema_step": record["ema_step"], "git_commit": actual_revision,
            "metainfo_sha256": record["metainfo_sha256"], "source_clean": True}


def verify_launch_sources(output):
    snapshot_path = output / "launch_source_snapshot.json"
    snapshot = read(snapshot_path)
    project = Path(snapshot["project_dir"]).resolve(strict=True)
    files = {entry["path"]: entry for entry in snapshot["source_files"]}
    required = {"src/aligndit/model/cfm_vt.py", "src/aligndit/model/backbone/dit_vt_mm.py",
                "src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py",
                "src/aligndit/script/eval/infer_grid_s1.py", "src/aligndit/script/eval/infer_grid_s2.py",
                "src/aligndit/config/finetune_grid.yaml"}
    require(required <= set(files), "Launch snapshot omits inference/model/config sources")
    checked = 0
    for relative, entry in files.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe launch source path")
        # Reports may later append final results. Only executable source,
        # configuration and launch scripts must stay identical during scoring.
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".sh", ".toml", ".cfg"}:
            continue
        actual = project / path
        require(actual.stat().st_size == entry["bytes"] and sha(actual) == entry["sha256"],
                f"Source differs from launch snapshot: {relative}")
        checked += 1
    return {"path": str(snapshot_path), "sha256": sha(snapshot_path),
            "captured_revision": snapshot["git_commit"], "executable_and_config_files_rehashed": checked}


def verify(output, cache, reuse_checkpoint=None):
    status = read(output / "pipeline_status.json")
    require(status.get("complete") is True and not status.get("error"), "Evaluation pipeline is not complete")
    require(not (cache / "WRITE_ACTIVE.json").exists(), "GRID cache has an active writer")
    rows = read_lines(cache / "manifests/val.jsonl")
    ids = [identity(row) for row in rows]
    require(len(ids) == len(set(ids)) == 3274, "Expected all 3274 unique validation clips")
    by_id = dict(zip(ids, rows))
    training = read_lines(cache / "manifests/train.jsonl")
    require(not {row["utterance_key"] for row in rows} & {row["utterance_key"] for row in training},
            "Train/validation leakage")
    require(all(len(normalize(row["text"]).split()) == 6 for row in rows), "Expected six target words per clip")
    inventory = read(cache / "manifests/inventory_meta.json")
    completion = read(cache / "complete.json")
    require(completion.get("complete") is True and completion["missing"] == completion["invalid"] == []
            and completion["selection"] == {"mode": "full"} and completion["split_counts"]["val"] == 3274,
            "Incomplete or filtered GRID cache")
    require(completion["inventory_meta_sha256"] == sha(cache / "manifests/inventory_meta.json"), "Stale GRID inventory")
    for name, entry in inventory["manifests"].items():
        path = cache / "manifests" / name
        require(sha(path) == entry["sha256"] and len(read_lines(path)) == entry["count"], f"Changed manifest: {name}")
    protocol = read(output / "protocol.json")
    protocol_sha = sha(output / "protocol.json")
    require(protocol["protocol"] in (S1, S2), "Unknown GRID setting protocol")
    s2 = protocol["protocol"] == S2
    mode = protocol.get("text_conditioning", "reference-target")
    require(mode in ("reference-target", "target-only") and (s2 or mode == "reference-target"), "Unknown text conditioning")
    require(protocol["requested_count"] == 3274 and protocol["partial_smoke_test"] is False, "Partial evaluation")
    require(protocol["generation"] == GENERATION, "Generation sampling parameters differ from formal GRID protocol")
    require(protocol["manifest_sha256"] == sha(cache / "manifests/val.jsonl")
            and protocol["cache_completion_sha256"] == sha(cache / "complete.json"), "Cache/protocol binding mismatch")
    sidecar = verify_fixed_architecture(protocol)
    launch_sources = verify_launch_sources(output)
    checkpoint = verify_checkpoint(protocol, reuse_checkpoint)
    pair_map = None
    if s2:
        require(protocol["text_source"] == "supplied_target_GT_dialogue; no VSR"
                and protocol["target_audio_read_for_generation"] is False
                and protocol["target_latent_or_speaker_read_for_generation"] is False, "S2 uses forbidden target acoustic input")
        if mode == "target-only":
            require(protocol.get("reference_transcript_used_for_generation") is False, "Target-only still uses reference text")
        pairs = read(output / "reference_pairs.json")
        require(pairs["count"] == 3274 and pairs["seed"] == 0
                and pairs["protocol"] == S2
                and pairs["selection"] == "val_same_speaker_sha256_seed0_cycle_first_different_text_shift_v1"
                and pairs["target_manifest_sha256"] == protocol["manifest_sha256"], "Invalid S2 reference selection")
        require([pair["id"] for pair in pairs["pairs"]] == ids
                and {pair["reference_id"] for pair in pairs["pairs"]} == set(ids), "References are not a full permutation")
        require(protocol["reference_pairs_sha256"] == sha(output / "reference_pairs.json"), "Reference pairing hash mismatch")
        pair_map = {pair["id"]: pair for pair in pairs["pairs"]}
        expected_pairs = reference_assignment(rows)
        for key, pair in pair_map.items():
            row, ref = by_id[key], by_id[pair["reference_id"]]
            require(pair["reference_id"] == expected_pairs[key] and pair["reference_id"] != key
                    and row["speaker"] == ref["speaker"] == pair["speaker"] == pair["reference_speaker"]
                    and normalize(row["text"]) != normalize(ref["text"]), f"Wrong independent reference: {key}")
            require(pair["reference_text"] == ref["text"] and pair["reference_source_sha256"] == ref["source_sha256"]
                    and pair["reference_audio_relative_path"] == ref["audio_relative_path"], f"Reference metadata mismatch: {key}")
    inference = read(output / "inference_summary.json")
    require(inference["count"] == 3274 and inference["complete"] is True and inference["partial_smoke_test"] is False
            and inference["protocol"] == protocol["protocol"] and inference["protocol_sha256"] == protocol_sha
            and [record["id"] for record in inference["outputs"]] == ids, "Invalid inference summary")
    wav_hashes, gt_hashes, cache_hashes = {}, {}, {}
    for index, row in enumerate(rows):
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
                    and receipt["extraction_sha256"] == completion["extraction_sha256"], f"Invalid {stage} cache receipt: {key}")
            cache_hashes[(key, stage)] = digest
            if stage == "audio":
                gt_hashes[key] = digest
            else:
                feature = np.load(path, allow_pickle=False)
                expected_shape = ((row["latent_frames"], 64) if stage == "latent" else
                                  (row["latent_frames"], 1024) if stage == "video" else (192,))
                require(feature.shape == expected_shape and feature.dtype == np.float32 and np.isfinite(feature).all(),
                        f"Invalid source {stage} array: {key}")
                if stage != "video":
                    require(receipt["audio_sha256"] == cache_hashes[(key, "audio")], f"Unbound source {stage}: {key}")
        wav = output / row["audio_relative_path"]
        receipt = read(output / "receipts" / f"{key}.json")
        wav_hashes[key] = sha(wav)
        require(receipt["sha256"] == wav_hashes[key] and receipt["protocol_sha256"] == protocol_sha
                and receipt["id"] == key and receipt["relative_path"] == row["audio_relative_path"]
                and receipt["samples"] == row["original_num_samples"] == 640 * row["source_video_frames"]
                and receipt == inference["outputs"][index], f"Invalid generated WAV receipt: {key}")
        with wave.open(str(wav)) as stream:
            require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes())
                    == (16000, 1, 2, row["original_num_samples"]), f"Invalid WAV format/duration: {key}")
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
            require(bool(len(samples)) and bool(samples.any()), f"Empty or entirely zero generated audio: {key}")
        if s2:
            pair = pair_map[key]
            ref = by_id[pair["reference_id"]]
            require(receipt["reference_id"] == pair["reference_id"]
                    and receipt["prompt_frames"] == ref["latent_frames"]
                    and receipt["target_frames"] == row["latent_frames"]
                    and receipt["sampler_frames"] == ref["latent_frames"] + row["latent_frames"], f"Wrong prompt/target slices: {key}")
            if mode == "target-only":
                require(receipt["text_conditioning"] == mode and receipt["model_text"] == row["text"], f"Wrong target-only text: {key}")
        if (index + 1) % 500 == 0:
            print(f"Verified WAV/cache bindings: {index + 1}/3274", flush=True)
    if s2:
        for key, pair in pair_map.items():
            for stage in ("audio", "latent", "speaker"):
                require(pair[f"reference_{stage}_sha256"] == cache_hashes[(pair["reference_id"], stage)],
                        f"Reference {stage} hash mismatch: {key}")
    require({path.relative_to(output).with_suffix("").as_posix() for path in output.glob("val/*/*.wav")} == set(ids),
            "Generated WAV full-ID coverage differs")
    report = read(output / "evaluation_summary.json")
    require(report["count"] == 3274 and report["complete"] is True and report["partial_smoke_test"] is False
            and report["protocol"] == protocol["protocol"] and report["checkpoint"] == protocol["checkpoint"], "Invalid final summary")
    if s2:
        require(report["text_conditioning"] == mode and report["reference_pairs_sha256"] == protocol["reference_pairs_sha256"],
                "Final summary reports a different S2 protocol")
    metrics, all_results = {}, {}
    edits = words = substitutions = deletions = insertions = changed = 0
    inference_sha = sha(output / "inference_summary.json")
    shard_counts = set()
    for task in ("sim", "wer", "emosim", "avsync"):
        path = output / f"{task}_results.jsonl"
        results = read_lines(path)
        all_results[task] = results
        require([record["id"] for record in results] == ids
                and all(record["wav"] == record["id"] and math.isfinite(record[task]) for record in results), f"Incomplete/invalid {task} results")
        summaries = sorted(output.glob(f"{task}_shard*_summary.json"))
        require(bool(summaries), f"No {task} shard summaries")
        nshard = read(summaries[0])["nshard"]
        require(len(summaries) == nshard, f"Wrong number of {task} shards")
        shard_counts.add(nshard)
        for shard in range(nshard):
            part = read_lines(output / f"{task}_shard{shard}.jsonl")
            summary = read(output / f"{task}_shard{shard}_summary.json")
            require(part == results[shard::nshard] and summary["nshard"] == nshard
                    and summary["inference_summary_sha256"] == inference_sha and summary["count"] == len(part), f"Invalid {task} shard {shard}")
            value = (sum(record["word_edit_distance"] for record in part) / sum(record["reference_word_count"] for record in part)
                     if task == "wer" else math.fsum(record[task] for record in part) / len(part))
            require(value == summary[task], f"Incorrect {task} shard mean")
        if task == "wer":
            for row, record in zip(rows, results):
                require(record["raw_truth"] == row["text"], "WER truth differs from GRID dialogue")
                truth, hypo = normalize(record["raw_truth"]), normalize(record["raw_hypo"])
                require((truth, hypo) == (record["truth"], record["hypo"]), "WER normalization mismatch")
                count = jiwer.process_words(truth, hypo)
                errors = word_distance(truth.split(), hypo.split())
                require(errors == record["word_edit_distance"] == count.substitutions + count.deletions + count.insertions
                        and (count.substitutions, count.deletions, count.insertions)
                        == (record["substitutions"], record["deletions"], record["insertions"])
                        and record["reference_word_count"] == len(truth.split()) == 6
                        and record["wer"] == errors / 6, "Independent WER recomputation differs")
                edits += errors
                words += 6
                substitutions += count.substitutions
                deletions += count.deletions
                insertions += count.insertions
                changed += record["raw_hypo"] != hypo
            value = edits / words
            require(words == 19644 and report["metrics"][task]["word_edit_distance"] == edits
                    and report["metrics"][task]["reference_word_count"] == words, "Wrong corpus WER numerator/denominator")
        else:
            value = math.fsum(record[task] for record in results) / len(results)
        require(value == report["metrics"][task][task] and report["metrics"][task]["count"] == 3274
                and report["metrics"][task]["display_value"] == f"{value:.5f}", f"Incorrect final {task} summary")
        if task == "sim" and s2:
            for record in results:
                pair = pair_map[record["id"]]
                require(all(record[field] == pair[field] for field in
                            ("reference_id", "reference_audio_relative_path", "reference_audio_sha256")), "SPKSIM uses wrong reference")
        metrics[task] = {"count": 3274, "value": value, "display_value": f"{value:.5f}", "results_sha256": sha(path)}
        print(f"Independent {task}: {value:.9f}", flush=True)
    require(len(shard_counts) == 1, "Metric shard counts differ")
    decoder = verify_decoder(output, cache, next(iter(shard_counts)))
    max_av_difference = 0.0
    for index, (row, av_result) in enumerate(zip(rows, all_results["avsync"])):
        key = identity(row)
        features = []
        for role in ("gen", "gt"):
            path = output / "avhubert_joint" / role / f"{key}.npy"
            receipt = read(path.with_suffix(".json"))
            require(receipt["id"] == key and receipt["sha256"] == sha(path)
                    and receipt["audio_sha256"] == (wav_hashes[key] if role == "gen" else gt_hashes[key])
                    and receipt["mouth_sha256"] == row["source"]["mouth_sha256"]
                    and receipt["checkpoint_sha256"] == AVHUBERT_SHA
                    and receipt["feature"] == "joint_audio_video_25hz_finetune_1024", f"Wrong {role} feature binding: {key}")
            feature = np.load(path, allow_pickle=False)
            require(feature.shape == (row["source_video_frames"], 1024) and feature.dtype == np.float32
                    and np.isfinite(feature).all(), f"Invalid joint feature: {key}/{role}")
            features.append(feature.astype(np.float64))
        first, second = features
        denominator = np.maximum(np.linalg.norm(first, axis=1), 1e-8) * np.maximum(np.linalg.norm(second, axis=1), 1e-8)
        value = float(np.mean(np.sum(first * second, axis=1) / denominator))
        difference = abs(value - av_result["avsync"])
        require(difference < 1e-6, f"Independent AVSync recomputation differs: {key}")
        max_av_difference = max(max_av_difference, difference)
        if (index + 1) % 500 == 0:
            print(f"Verified paired joint features: {index + 1}/3274", flush=True)
    for role in ("gen", "gt"):
        root = output / "avhubert_joint" / role
        require({path.relative_to(root).with_suffix("").as_posix() for path in root.glob("val/*/*.npy")} == set(ids),
                f"Incomplete {role} joint-feature coverage")
    assets_path = output / "metric_assets.json"
    assets = read(assets_path)
    require(report["metric_assets"]["sha256"] == sha(assets_path), "Metric asset report changed")
    require(len(assets["assets"]) == 11, "Expected eleven frozen metric asset files")
    for asset in assets["assets"]:
        require(Path(asset["path"]).stat().st_size == asset["size_bytes"] and sha(asset["path"]) == asset["sha256"],
                f"Metric asset changed: {asset['path']}")
    require(any(asset["sha256"] == AVHUBERT_SHA for asset in assets["assets"]), "Pinned AV-HuBERT asset missing")
    sources = (assets["resume_source_history"][-1]["source_files"]
               if assets.get("resume_source_history") else assets["source_files"])
    for source in sources:
        require(sha(source["path"]) == source["sha256"], f"Evaluation source changed: {source['path']}")
    result = {"status": "PASS", "complete": True, "count": 3274, "requested": 3274,
              "succeeded": 3274, "failed": 0, "protocol": protocol["protocol"],
              "text_conditioning": mode, "metrics": metrics,
              "speaker_counts": dict(collections.Counter(row["speaker"] for row in rows)),
              "word_errors": edits, "reference_words": words, "substitutions": substitutions,
              "deletions": deletions, "insertions": insertions, "normalized_hypotheses_changed": changed,
              "wav_count": 3274, "independent_reference_pairs": 3274 if s2 else 0,
              "reference_permutation_verified": s2, "nshard": next(iter(shard_counts)),
              "joint_feature_counts": {"gen": 3274, "gt": 3274},
              "max_numpy_avsync_recompute_difference": max_av_difference,
              "model_assets_rehashed": len(assets["assets"]), "source_files_rehashed": len(sources),
              "checkpoint_rehashed": checkpoint["rehashed"], "checkpoint_verification": checkpoint,
              "training_contract": sidecar, "decoder_verification": decoder,
              "launch_source_verification": launch_sources,
              "evaluation_summary_sha256": sha(output / "evaluation_summary.json"),
              "protocol_sha256": protocol_sha, "manifest_sha256": protocol["manifest_sha256"],
              "verification_script_sha256": sha(__file__),
              "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if s2:
        result["reference_pairs_sha256"] = protocol["reference_pairs_sha256"]
        result["text_source"] = protocol["text_source"]
    publish(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser.add_argument("--cache-root", type=Path,
                        default=Path(prefix + "/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32"))
    parser.add_argument("--reuse-checkpoint-verification", type=Path,
                        help="Prior PASS verification.json for the same immutable checkpoint/stat; default rehashes actual bytes")
    args = parser.parse_args()
    try:
        result = verify(args.output_dir.resolve(strict=True), args.cache_root.resolve(strict=True),
                        args.reuse_checkpoint_verification)
    except Exception as error:
        failure = {"status": "FAIL", "complete": False, "error": str(error),
                   "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if args.output_dir.is_dir():
            publish(args.output_dir / "verification.json", failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
