#!/usr/bin/env python3
"""Recover LRS3 segment labels with source text and unprompted ASR evidence.

Only source-supported words also reproduced by an independent segment decode are
accepted. Conflicting ASR corrections remain review candidates. Whisper word
timestamps are measured on concatenated PCM audio, never on summed video times.
All ASR is local/offline; GPU IDs explicitly name physical devices.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import importlib.metadata
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import traceback
import unicodedata

PROTOCOL = "lrs3_source_segment_asr_consensus_v3"
MARKERS = {"lg", "ns", "pc", "tc"}
WORD_EQUIVALENTS = {
    "ok": ["okay"], "cannot": ["can", "not"], "can't": ["can", "not"],
    "won't": ["will", "not"], "shan't": ["shall", "not"],
    "don't": ["do", "not"], "doesn't": ["does", "not"], "didn't": ["did", "not"],
    "isn't": ["is", "not"], "aren't": ["are", "not"], "wasn't": ["was", "not"], "weren't": ["were", "not"],
    "haven't": ["have", "not"], "hasn't": ["has", "not"], "hadn't": ["had", "not"],
    "couldn't": ["could", "not"], "shouldn't": ["should", "not"], "wouldn't": ["would", "not"],
    "mustn't": ["must", "not"], "needn't": ["need", "not"],
    "i'm": ["i", "am"], "you're": ["you", "are"], "we're": ["we", "are"], "they're": ["they", "are"],
    "i've": ["i", "have"], "you've": ["you", "have"], "we've": ["we", "have"], "they've": ["they", "have"],
}


def small_number_words(number):
    low = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if number < 20:
        return [low[number]]
    if number < 100:
        return [tens[number // 10]] + (small_number_words(number % 10) if number % 10 else [])
    return [low[number // 100], "hundred"] + (small_number_words(number % 100) if number % 100 else [])


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_identity(model_path, output):
    names = ["model.bin", "config.json", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
    files = [Path(model_path) / name for name in names if (Path(model_path) / name).is_file()]
    stats = {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files}
    cache = Path(output) / "model_identity.json"
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous.get("file_stats") == stats and previous.get("model_path") == str(Path(model_path).resolve()):
            return previous
    hashes = {}
    for path in files:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                hasher.update(block)
        hashes[path.name] = hasher.hexdigest()
    identity = {"model_path": str(Path(model_path).resolve()), "file_stats": stats, "sha256": hashes,
                "libraries": {name: importlib.metadata.version(name) for name in ["faster-whisper", "ctranslate2"]}}
    atomic_json(cache, identity)
    return identity


def raw_tokens(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'").replace("‘", "'")
    return re.findall(r"\d+(?:[.,/:]\d+)+|[a-z0-9]+(?:'[a-z0-9]+)*", text)


def tokens(text):
    raw = raw_tokens(text)
    result = []
    for word in raw:
        if word in WORD_EQUIVALENTS:
            result.extend(WORD_EQUIVALENTS[word])
        elif word.isdecimal() and len(word) <= 3 and (len(word) == 1 or not word.startswith("0")):
            result.extend(small_number_words(int(word)))
        elif re.fullmatch(r"[a-z]\d{1,2}", word):
            result.extend([word[0]] + small_number_words(int(word[1:])))
        else:
            result.append(word)
    return result


def load_alternative_sources(path, candidate_ids):
    """Read the publisher's seg16s CSV, retaining its own ordering/provenance."""
    grouped = collections.defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) < 5:
                continue
            pieces = Path(row[1]).parts
            if len(pieces) < 4 or pieces[-3] not in {"pretrain", "trainval"}:
                continue
            stem = Path(pieces[-1]).stem
            match = re.fullmatch(r"(\d+)_(\d+)", stem)
            if match is None:
                continue
            original_id = "/".join([pieces[-3], pieces[-2], match.group(1)])
            if original_id in candidate_ids:
                grouped[original_id].append({"source_row_1based": line_number, "source_path": row[1],
                    "segment_index": int(match.group(2)), "frames": int(row[2]), "text": ",".join(row[4:])})
    result = {}
    for original_id, records in grouped.items():
        records.sort(key=lambda r: r["segment_index"])
        result[original_id] = {"source_name": "Llama_AVSR_seg16s", "source_path": str(Path(path).resolve()),
            "source_rows": records, "concatenated_text": " ".join(r["text"] for r in records),
            "source_indices_contiguous_from_zero": [r["segment_index"] for r in records] == list(range(len(records)))}
    return result


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def segment_index(row):
    original = row["original_clip_id"].split("/")
    base = original[-2] + "_" + original[-1]
    name = row["sample_id"]
    if name == base:
        return -1
    if name.startswith(base + "_") and name[len(base) + 1:].isdigit():
        return int(name[len(base) + 1:])
    raise ValueError(f"Cannot order segment {name!r} within {row['original_clip_id']!r}")


def word_tokens(words):
    result = []
    for word in words:
        for token in tokens(word["word"]):
            result.append(dict(word, token=token))
    return result


def transcribe(model, audio):
    segments, info = model.transcribe(
        audio, language="en", task="transcribe", beam_size=5, temperature=0.0,
        condition_on_previous_text=False, initial_prompt=None, prefix=None,
        word_timestamps=True, vad_filter=False,
    )
    result = {"text": "", "words": [], "segments": [], "language_probability": float(info.language_probability)}
    texts = []
    for segment in segments:
        texts.append(segment.text)
        result["segments"].append({
            "start": float(segment.start), "end": float(segment.end), "text": segment.text,
            "avg_logprob": float(segment.avg_logprob), "no_speech_prob": float(segment.no_speech_prob),
            "compression_ratio": float(segment.compression_ratio),
        })
        for word in segment.words or []:
            result["words"].append({"word": word.word, "start": float(word.start), "end": float(word.end), "probability": float(word.probability)})
    result["text"] = " ".join(texts).strip()
    return result


def choose_source(candidate, hypothesis, allow_marker_cleanup=True):
    if candidate.get("alternative_sources"):
        primary = dict(candidate)
        primary.pop("alternative_sources")
        options = [choose_source(option, hypothesis, allow_marker_cleanup) for option in [primary] + candidate["alternative_sources"]]
        selected = min(options, key=lambda pair: pair[0]["edit_distance"])
        selected[0]["source_comparison_options"] = [{k: v for k, v in pair[0].items() if k in ["source_name", "variant", "edit_distance", "wer"]} for pair in options]
        return selected
    raw = tokens(candidate["concatenated_text"])
    cleaned = [token for token in raw if token not in MARKERS]
    variants = [(edit_distance(raw, hypothesis), "original_decoded_units", raw, [])]
    if allow_marker_cleanup and cleaned != raw:
        variants.append((edit_distance(cleaned, hypothesis), "marker_removal_candidate", cleaned, [t for t in raw if t in MARKERS]))
    # Raw transcription wins ties: genuine spoken acronyms must not be removed.
    distance, variant, reference, removed = min(variants, key=lambda item: item[0])
    source_raw = raw_tokens(candidate["concatenated_text"])
    if variant == "marker_removal_candidate":
        source_raw = [word for word in source_raw if word not in MARKERS]
    source_spans = []
    source_offset = 0
    for word in source_raw:
        length = len(tokens(word))
        source_spans.append([source_offset, source_offset + length])
        source_offset += length
    assert [token for word in source_raw for token in tokens(word)] == reference
    mapping = {}
    changes = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False).get_opcodes():
        if tag == "equal":
            mapping.update({j: i1 + j - j1 for j in range(j1, j2)})
        else:
            changes.append({"operation": tag, "reference_range": [i1, i2], "asr_range": [j1, j2], "reference": reference[i1:i2], "asr": hypothesis[j1:j2]})
    return {"source_name": candidate.get("source_name", "USR_unigram1000"), "variant": variant, "tokens": reference,
            "raw_words": source_raw, "raw_word_spans": source_spans,
            "removed_marker_candidates": removed, "edit_distance": distance,
            "wer": distance / max(1, len(reference)), "changes": changes}, mapping


def quality_reasons(selected, segment_asr, assigned, start, end, config):
    reasons = []
    if not selected:
        reasons.append("no_assigned_words")
        return reasons
    if tokens(segment_asr["text"]) != assigned:
        reasons.append("independent_segment_asr_disagrees")
    if any(w["end"] <= w["start"] for w in selected):
        reasons.append("zero_or_negative_word_duration")
    if min(w["probability"] for w in selected) < config["min_word_probability"]:
        reasons.append("low_word_probability")
    if sum(w["probability"] for w in selected) / len(selected) < config["min_mean_probability"]:
        reasons.append("low_mean_word_probability")
    if any(min(w["end"], start) - w["start"] > config["max_boundary_crossing"] or
           w["end"] - max(w["start"], end) > config["max_boundary_crossing"] for w in selected):
        reasons.append("word_crosses_segment_boundary")
    if any(s["avg_logprob"] < -1.0 or s["compression_ratio"] > 2.4 for s in segment_asr["segments"]):
        reasons.append("unreliable_segment_decode")
    return reasons


def classify_consensus(candidate, full_asr, segment_decodes, spans, config):
    """Use known clip boundaries to resolve approximate full-recording timestamps.

    A clip transcript must be a contiguous equal block in both the source and
    the unprompted full-recording decode. The clip is decoded independently;
    source text is never supplied to either recognizer call.
    """
    per_segment = [word_tokens(decode["words"]) for decode in segment_decodes]
    combined = [word for words in per_segment for word in words]
    hypothesis = [word["token"] for word in combined]
    source, source_mapping = choose_source(candidate, hypothesis)
    group_reference = tokens(full_asr["text"])
    group_comparison, group_mapping = choose_source({"concatenated_text": " ".join(group_reference)}, hypothesis, allow_marker_cleanup=False)
    labels = []
    offset = 0
    for decode, selected, span in zip(segment_decodes, per_segment, spans):
        indices = list(range(offset, offset + len(selected)))
        offset += len(selected)
        assigned = [w["token"] for w in selected]
        duration = (span["end_sample"] - span["start_sample"]) / span["sample_rate"]
        reasons = quality_reasons(selected, decode, assigned, 0, duration, config)
        source_indices = [source_mapping.get(i) for i in indices]
        group_indices = [group_mapping.get(i) for i in indices]
        for prefix, positions, comparison in [("source", source_indices, source), ("group_asr", group_indices, group_comparison)]:
            if not positions or any(i is None for i in positions):
                reasons.append(f"segment_words_not_fully_supported_by_{prefix}")
            elif positions != list(range(positions[0], positions[-1] + 1)):
                reasons.append(f"{prefix}_contains_missing_words_inside_segment")
            if indices and any(max(indices[0] - 1, change["asr_range"][0]) <= min(indices[-1] + 1, change["asr_range"][1]) for change in comparison["changes"]):
                reasons.append(f"{prefix}_conflict_at_or_near_segment")
        raw_label_words = []
        if source_indices and all(i is not None for i in source_indices):
            begin, finish = source_indices[0], source_indices[-1] + 1
            raw_selected = [(word, bounds) for word, bounds in zip(source["raw_words"], source["raw_word_spans"]) if bounds[1] > begin and bounds[0] < finish]
            if raw_selected and raw_selected[0][1][0] == begin and raw_selected[-1][1][1] == finish:
                raw_label_words = [word for word, _ in raw_selected]
            else:
                reasons.append("partial_normalized_source_word_at_boundary")
        accepted = not reasons
        labels.append({
            "sample_id": decode["sample_id"], "original_clip_id": candidate["original_clip_id"],
            "text": " ".join(raw_label_words) if accepted else None,
            "source": source["source_name"] + "+Whisper_large_v3_unprompted_group_and_segment",
            "quality": "accepted_source_group_segment_consensus" if accepted else "review_required",
            "accepted": accepted, "reasons": reasons,
            "asr_candidate_text": " ".join(assigned),
            "candidate_source_token_range": [source_indices[0], source_indices[-1] + 1] if source_indices and all(i is not None for i in source_indices) else None,
            "group_asr_token_range": [group_indices[0], group_indices[-1] + 1] if group_indices and all(i is not None for i in group_indices) else None,
            "pcm_span": span, "duration": duration, "source_variant": source["variant"],
            "source_segment_asr_wer": source["wer"], "group_segment_asr_wer": group_comparison["wer"],
            "comparison_normalizations": [{"source_word": word, "comparison_tokens": tokens(word)} for word in raw_label_words if tokens(word) != [word]],
            "min_word_probability": min((w["probability"] for w in selected), default=None),
            "mean_word_probability": sum(w["probability"] for w in selected) / len(selected) if selected else None,
        })
    return labels, source, group_comparison


def read_bound_audio(path):
    """Decode exactly the bytes whose hash is recorded, rather than rereading."""
    import soundfile as sf
    path = Path(path)
    payload = path.read_bytes()
    audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
    identity = {"audio_path": str(path), "file_sha256": hashlib.sha256(payload).hexdigest(),
                "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "sample_count": len(audio), "sample_rate": sr,
                "hash_binding": "exact_bytes_decoded_by_asr"}
    return audio, sr, identity


def process_group(model, job, config):
    import numpy as np
    import soundfile as sf
    rows = sorted(job["rows"], key=segment_index)
    candidate = job["candidate"]
    if len(rows) != int(candidate["darknsu_processed_pairs"]):
        raise ValueError("Incomplete original group in manifest; all source segments are required")
    if sum(int(r.get("video_frames", r.get("frames", 0))) for r in rows) != int(candidate["darknsu_frame_sum"]):
        raise ValueError("Manifest frame total differs from the audited original group")
    audios, spans, identities = [], [], []
    offset = 0
    for row in rows:
        audio, sr, identity = read_bound_audio(row["audio_path"])
        if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite mono 16 kHz PCM: {row['audio_path']}")
        audios.append(audio)
        identities.append({"sample_id": row["sample_id"], **identity})
        spans.append({"sample_id": row["sample_id"], "start_sample": offset, "end_sample": offset + len(audio), "sample_rate": sr,
                      "pcm_sha256": identity["pcm_sha256"], "file_sha256": identity["file_sha256"]})
        offset += len(audio)
    full_asr = transcribe(model, np.concatenate(audios))
    segment_decodes = []
    for row, audio in zip(rows, audios):
        segment_decodes.append({"sample_id": row["sample_id"], **transcribe(model, audio)})
    labels, source, group_comparison = classify_consensus(candidate, full_asr, segment_decodes, spans, config)
    return {"original_clip_id": job["id"], "status": "complete", "duration": offset / 16000,
            "source_candidate": candidate, "source_comparison": source, "group_segment_comparison": group_comparison,
            "full_asr": full_asr, "segment_asr": segment_decodes, "labels": labels,
            "audio_source_bindings": identities,
            "segment_inputs": [{"sample_id": r["sample_id"], "audio_path": r["audio_path"], "video_path": r["video_path"]} for r in rows]}



def process_control(model, job, config):
    import soundfile as sf
    row = job["row"]
    audio, sr, identity = read_bound_audio(row["audio_path"])
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Control audio must be mono 16 kHz")
    result = transcribe(model, audio)
    reference, hypothesis = tokens(row["text"]), tokens(result["text"])
    distance = edit_distance(reference, hypothesis)
    return {"sample_id": row["sample_id"], "status": "complete", "duration": len(audio) / sr,
            "reference_text": row["text"], "reference_word_count": len(reference),
            "edit_distance": distance, "wer": distance / max(1, len(reference)), "exact_match": reference == hypothesis,
            "asr": result, "audio_source_bindings": [{"sample_id": row["sample_id"], **identity}]}


def worker(gpu, jobs, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    from faster_whisper import WhisperModel
    model = WhisperModel(config["model"], device="cuda", device_index=0, compute_type="float16", cpu_threads=config["cpu_threads"], local_files_only=True)
    print(json.dumps({"event": "model_loaded", "physical_gpu": gpu, "jobs": len(jobs)}), flush=True)
    for job in jobs:
        started = time.time()
        path = Path(config["output"]) / ("groups" if job["kind"] == "group" else "controls") / (digest(job["id"])[:24] + ".json")
        fingerprint = digest({"protocol": PROTOCOL, "job": job, "config": {k:v for k,v in config.items() if k not in ["output", "cpu_threads"]}})
        if path.exists():
            previous = json.loads(path.read_text())
            bindings = previous.get("audio_source_bindings", [])
            if (previous.get("fingerprint") == fingerprint and previous.get("status") == "complete"
                    and bindings and all(hashlib.sha256(Path(r["audio_path"]).read_bytes()).hexdigest() == r["file_sha256"] for r in bindings)):
                print(json.dumps({"event": "resume_skip", "id": job["id"], "physical_gpu": gpu}), flush=True)
                continue
        try:
            result = process_group(model, job, config) if job["kind"] == "group" else process_control(model, job, config)
        except Exception as exc:
            result = {"status": "error", "id": job["id"], "error": str(exc), "traceback": traceback.format_exc()}
        result.update({"fingerprint": fingerprint, "protocol": PROTOCOL, "physical_gpu": gpu, "elapsed_seconds": time.time() - started,
                       "asr_started_unix": started, "asr_completed_unix": time.time()})
        atomic_json(path, result)
        print(json.dumps({"event": "job_finished", "id": job["id"], "status": result["status"], "physical_gpu": gpu,
                          "accepted": sum(r["accepted"] for r in result.get("labels", [])), "labels": len(result.get("labels", [])),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def summarize(output):
    output = Path(output)
    run = json.loads((output / "run.json").read_text()) if (output / "run.json").exists() else {}
    group_ids = set(run.get("group_ids", []))
    control_ids = set(run.get("control_ids", []))
    groups = [json.loads(p.read_text()) for p in sorted((output / "groups").glob("*.json"))]
    controls = [json.loads(p.read_text()) for p in sorted((output / "controls").glob("*.json"))]
    if group_ids:
        groups = [r for r in groups if r.get("original_clip_id", r.get("id")) in group_ids]
    if control_ids:
        controls = [r for r in controls if r.get("sample_id", r.get("id")) in control_ids]
    labels = [r for group in groups for r in group.get("labels", [])]
    accepted, review = [r for r in labels if r["accepted"]], [r for r in labels if not r["accepted"]]
    atomic_jsonl(output / "labels.accepted.jsonl", accepted)
    atomic_jsonl(output / "labels.review.jsonl", review)
    valid_controls = [r for r in controls if r.get("status") == "complete"]
    summary = {
        "protocol": PROTOCOL, "groups_complete": sum(r.get("status") == "complete" for r in groups),
        "groups_error": sum(r.get("status") == "error" for r in groups),
        "labels_total": len(labels), "labels_accepted": len(accepted), "labels_review": len(review),
        "accepted_hours": sum(r["duration"] for r in accepted) / 3600,
        "review_hours": sum(r["duration"] for r in review) / 3600,
        "review_reasons": dict(collections.Counter(reason for r in review for reason in r["reasons"])),
        "controls_complete": len(valid_controls), "controls_exact_match": sum(r["exact_match"] for r in valid_controls),
        "controls_corpus_wer": sum(r["edit_distance"] for r in valid_controls) / max(1, sum(r["reference_word_count"] for r in valid_controls)),
        "groups_requested": run.get("groups_requested"), "controls_requested": run.get("controls_requested"),
        "limitations": ["Automatic source/ASR consensus is evidence, not manual or original-label verification.",
                       "Two ASR passes use the same large-v3 model and can share errors.",
                       "Review candidates must not be used as verified training labels.",
                       "Segment identity is established by source/full-decode/independent-clip word-sequence agreement, not by approximate full-recording timestamps.",
                       "Transcript recovery does not establish audio/video synchronization."],
    }
    summary["complete"] = (summary["groups_complete"] == run.get("groups_requested")
                           and summary["controls_complete"] == run.get("controls_requested")
                           and summary["groups_error"] == 0)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--additional-candidates", type=Path, help="Optional Llama-AVSR seg16s transcript CSV")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--controls", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--min-word-probability", type=float, default=0.35)
    parser.add_argument("--min-mean-probability", type=float, default=0.65)
    parser.add_argument("--max-boundary-crossing", type=float, default=0.12)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--reclassify-only", action="store_true", help="Reclassify saved ASR evidence without loading a GPU model")
    args = parser.parse_args()
    if args.reclassify_only:
        run = json.loads((args.output / "run.json").read_text())
        alternatives = load_alternative_sources(args.additional_candidates, set(run["group_ids"])) if args.additional_candidates else {}
        for path in sorted((args.output / "groups").glob("*.json")):
            result = json.loads(path.read_text())
            if result.get("status") != "complete":
                continue
            if result["original_clip_id"] in alternatives:
                result["source_candidate"]["alternative_sources"] = [alternatives[result["original_clip_id"]]]
            spans = [r["pcm_span"] for r in result["labels"]]
            labels, source, group = classify_consensus(result["source_candidate"], result["full_asr"], result["segment_asr"], spans, run["config"])
            result.update({"labels": labels, "source_comparison": source, "group_segment_comparison": group,
                           "protocol": PROTOCOL, "reclassified_from_saved_asr": True})
            atomic_json(path, result)
        run["protocol"] = PROTOCOL
        for path in sorted((args.output / "controls").glob("*.json")):
            control = json.loads(path.read_text())
            if control.get("status") != "complete":
                continue
            reference, hypothesis = tokens(control["reference_text"]), tokens(control["asr"]["text"])
            distance = edit_distance(reference, hypothesis)
            control.update({"reference_word_count": len(reference), "edit_distance": distance,
                            "wer": distance / max(1, len(reference)), "exact_match": reference == hypothesis})
            atomic_json(path, control)
        run["comparison_normalization"] = {"word_equivalents": WORD_EQUIVALENTS, "numbers": "0..999 cardinals and single-letter 0..99 identifiers; accepted labels preserve source word forms", "version": PROTOCOL}
        if args.additional_candidates:
            run["additional_candidates"] = str(args.additional_candidates.resolve())
            run["additional_candidates_sha256"] = hashlib.sha256(args.additional_candidates.read_bytes()).hexdigest()
        atomic_json(args.output / "run.json", run)
    if args.summarize_only or args.reclassify_only:
        print(json.dumps(summarize(args.output), indent=2))
        return
    if not all((args.manifest, args.candidates, args.model)):
        parser.error("--manifest, --candidates and --model are required unless --summarize-only")
    rows = read_jsonl(args.manifest)
    candidates = {r["original_clip_id"]: r for r in read_jsonl(args.candidates)}
    if args.additional_candidates:
        alternatives = load_alternative_sources(args.additional_candidates, set(candidates))
        for original_id, alternative in alternatives.items():
            candidates[original_id]["alternative_sources"] = [alternative]
    missing = collections.defaultdict(list)
    for row in rows:
        if not row.get("text") and row["original_clip_id"] in candidates:
            missing[row["original_clip_id"]].append(row)
    group_ids = sorted(missing)
    if args.limit_groups:
        group_ids = group_ids[:args.limit_groups]
    jobs = [{"kind": "group", "id": oid, "rows": missing[oid], "candidate": candidates[oid]} for oid in group_ids]
    controls = sorted((r for r in rows if r.get("text")), key=lambda r: digest(r["sample_id"]))[:args.controls]
    jobs += [{"kind": "control", "id": r["sample_id"], "row": r} for r in controls]
    args.output.mkdir(parents=True, exist_ok=True)
    identity = model_identity(args.model, args.output)
    config = {"model": str(args.model.resolve()), "output": str(args.output.resolve()), "cpu_threads": args.cpu_threads,
              "min_word_probability": args.min_word_probability, "min_mean_probability": args.min_mean_probability,
              "max_boundary_crossing": args.max_boundary_crossing,
              "model_sha256": identity["sha256"], "libraries": identity["libraries"]}
    atomic_json(args.output / "run.json", {"protocol": PROTOCOL, "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "candidates": str(args.candidates.resolve()),
        "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(), "groups_requested": len(group_ids),
        "additional_candidates": str(args.additional_candidates.resolve()) if args.additional_candidates else None,
        "additional_candidates_sha256": hashlib.sha256(args.additional_candidates.read_bytes()).hexdigest() if args.additional_candidates else None,
        "comparison_normalization": {"word_equivalents": WORD_EQUIVALENTS, "numbers": "0..999 cardinals and single-letter 0..99 identifiers; accepted labels preserve source word forms", "version": PROTOCOL},
        "group_ids": group_ids, "control_ids": [r["sample_id"] for r in controls], "controls_requested": len(controls), "config": config, "gpus": args.gpus,
        "decode": {"language": "en", "beam_size": 5, "temperature": 0, "condition_on_previous_text": False,
                   "initial_prompt": None, "word_timestamps": True, "vad_filter": False, "compute_type": "float16"}})
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(gpu, jobs[i::len(gpus)], config)) for i, gpu in enumerate(gpus) if jobs[i::len(gpus)]]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    summary = summarize(args.output)
    summary["worker_exitcodes"] = [p.exitcode for p in processes]
    summary["groups_requested"] = len(group_ids)
    summary["controls_requested"] = len(controls)
    summary["complete"] = (summary["groups_complete"] == len(group_ids) and summary["controls_complete"] == len(controls)
                           and not summary["groups_error"] and all(p.exitcode == 0 for p in processes))
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
