#!/usr/bin/env python3
"""Incrementally verify remaining source text with an independent CTC model.

Only additional source-grounded consensus labels are approved. Agreement of
recognizers without the exact bounded source phrase remains unapproved.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, read_bound_audio, read_jsonl, tokens
from lrs3_text_recovery_pipeline import process_start

PROTOCOL = "source_whisper_ctc_mfa_consensus_v1"
ALLOWED_REPLACED_FAILURES = {
    "source_conflict_at_or_near_segment", "group_asr_conflict_at_or_near_segment",
    "segment_words_not_fully_supported_by_group_asr", "group_asr_contains_missing_words_inside_segment",
    "zero_or_negative_word_duration",
}


def decide(label, group, diagnostic, ctc):
    reasons = []
    if not set(label["reasons"]).issubset(ALLOWED_REPLACED_FAILURES):
        reasons.append("existing_source_content_or_acoustic_confidence_failure")
    hypothesis = tokens(label["asr_candidate_text"])
    if tokens(ctc["text"]) != hypothesis:
        reasons.append("independent_recognizers_disagree")
    if len(hypothesis) < 3:
        reasons.append("too_few_words_for_this_consensus_rule")
    scores = [ctc.get("min_character_probability"), ctc.get("mean_character_probability")]
    if (any(s is None or not math.isfinite(s) for s in scores)
            or (scores[0] or 0) < .8 or (scores[1] or 0) < .97):
        reasons.append("weak_independent_character_evidence")
    words = diagnostic.get("words", [])
    assigned = " ".join(w["word"] for w in words)
    if tokens(assigned) != hypothesis:
        reasons.append("independently_aligned_source_phrase_differs")
    span = label["pcm_span"]
    begin, end = span["start_sample"] / span["sample_rate"], span["end_sample"] / span["sample_rate"]
    if not words or any(w["word"].startswith("<") or w["word"] in {"sil", "sp", "spn"} for w in words):
        reasons.append("unknown_or_missing_alignment_words")
    if words and (min(w["begin"] for w in words) < begin - .04 or max(w["end"] for w in words) > end + .04):
        reasons.append("mfa_word_crosses_actual_boundary")
    if words and min(w["end"] - w["begin"] for w in words) < .0199:
        reasons.append("mfa_word_duration_too_short")
    source_range = label.get("candidate_source_token_range")
    raw_words = []
    if source_range:
        source = group["source_comparison"]
        selected = [(w, bounds) for w, bounds in zip(source["raw_words"], source["raw_word_spans"])
                    if bounds[1] > source_range[0] and bounds[0] < source_range[1]]
        if selected and selected[0][1][0] == source_range[0] and selected[-1][1][1] == source_range[1]:
            raw_words = [w for w, _ in selected]
    if tokens(" ".join(raw_words)) != hypothesis:
        reasons.append("no_complete_matching_source_word_span")
    result = dict(label)
    result.update({"accepted": not reasons, "text": " ".join(raw_words) if not reasons else None,
        "source": label["source"] + "+independent_wav2vec2_CTC+MFA_bounded_source_phrase",
        "quality": "accepted_source_whisper_ctc_mfa_consensus" if not reasons else "review_required",
        "prior_review_reasons": label["reasons"], "reasons": reasons,
        "independent_ctc_evidence": {"text": ctc["text"], "min_character_probability": ctc["min_character_probability"],
            "mean_character_probability": ctc["mean_character_probability"], "source_binding": ctc["source_binding"],
            "unchanged_whisper_confidence_gates": True, "protocol": PROTOCOL},
        "mfa_evidence": {"word_sequence": assigned, "boundary_tolerance_seconds": .04, "minimum_word_seconds": .02}})
    return result


class Decoder:
    def __init__(self, model_path):
        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        torch.set_num_threads(4)
        config = Wav2Vec2Config.from_pretrained(str(model_path), local_files_only=True)
        config.mask_time_prob = config.mask_feature_prob = 0.0
        config.apply_spec_augment = False
        self.processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Wav2Vec2ForCTC(config)
        state = torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(device="cuda", dtype=torch.float16).eval()
        self.blank = config.pad_token_id
        self.delimiter = self.processor.tokenizer.word_delimiter_token_id

    def decode(self, audio, source_binding):
        torch = self.torch
        started = time.time()
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        attention = inputs.attention_mask.to("cuda") if "attention_mask" in inputs else None
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(device="cuda", dtype=torch.float16), attention_mask=attention).logits[0].float()
            probabilities = torch.softmax(logits, dim=-1)
            indices = probabilities.argmax(dim=-1)
            scores = probabilities.gather(-1, indices[:, None])[:, 0]
            ids, scores = indices.cpu().tolist(), scores.cpu().tolist()
        text = self.processor.batch_decode([ids])[0].strip().lower()
        character_scores = []
        index = 0
        while index < len(ids):
            end = index + 1
            while end < len(ids) and ids[end] == ids[index]:
                end += 1
            if ids[index] not in {self.blank, self.delimiter}:
                character_scores.append(max(scores[index:end]))
            index = end
        return {"text": text, "min_character_probability": min(character_scores, default=0),
            "mean_character_probability": sum(character_scores) / max(1, len(character_scores)),
            "source_binding": source_binding, "elapsed_seconds": time.time() - started}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--mfa-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--asr-pid", type=int)
    parser.add_argument("--mfa-pid", type=int)
    parser.add_argument("--pilot", action="store_true", help="Use the single frozen pilot MFA diagnostics, not batch receipts")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "groups").mkdir(exist_ok=True)
    (args.output / "segments").mkdir(exist_ok=True)
    with (args.output / "coordinator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = json.loads((args.evidence_dir / "run.json").read_text())
        wanted = set(run["group_ids"])
        identity = json.loads((args.model / "download_info.json").read_text())
        atomic_json(args.output / "run.json", {"protocol": PROTOCOL, "model_identity": identity,
            "source_code_sha256_at_launch": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "asr_run": str(args.evidence_dir / "run.json"), "groups_requested": sorted(wanted),
            "gpu": args.gpu, "asr_pid": args.asr_pid, "mfa_pid": args.mfa_pid,
            "rule": "Exact complete source phrase + independent clip Whisper + independent CTC + bounded MFA; original Whisper confidence gates unchanged; no ASR-only promotion."})
        followed = {pid: process_start(pid) for pid in [args.asr_pid, args.mfa_pid] if pid}
        decoder = None
        group_cache, batch_cache = {}, {}
        final_sweep = False
        while True:
            active_results = {}
            covered_ids = set()
            for path in sorted((args.evidence_dir / "groups").glob("*.json")):
                stamp = path.stat().st_mtime_ns
                if path not in group_cache or group_cache[path][0] != stamp:
                    payload = path.read_bytes()
                    group_cache[path] = (stamp, json.loads(payload), hashlib.sha256(payload).hexdigest())
                _, group, group_hash = group_cache[path]
                original_id = group.get("original_clip_id")
                if original_id not in wanted or group.get("status") != "complete":
                    continue
                if args.pilot:
                    batch = args.mfa_dir
                else:
                    receipt_path = args.mfa_dir / "receipts" / (digest(original_id)[:24] + ".json")
                    if not receipt_path.exists():
                        continue
                    receipt = json.loads(receipt_path.read_text())
                    if receipt["source_evidence_sha256"] != group_hash:
                        continue
                    if receipt["status"] == "no_review_needed":
                        covered_ids.add(original_id)
                        continue
                    if receipt["status"] != "mfa_processed":
                        continue
                    batch = Path(receipt["batch"])
                diagnostic_path = batch / "boundary_diagnostics.jsonl"
                diagnostic_stamp = diagnostic_path.stat().st_mtime_ns
                if batch not in batch_cache or batch_cache[batch][0] != diagnostic_stamp:
                    diagnostic_bytes = diagnostic_path.read_bytes()
                    all_diagnostics = {r["sample_id"]: r for r in map(json.loads, diagnostic_bytes.splitlines())}
                    previous_accepted = {r["sample_id"] for r in read_jsonl(batch / "labels.mfa_additional.accepted.jsonl")}
                    batch_cache[batch] = (diagnostic_stamp, hashlib.sha256(diagnostic_bytes).hexdigest(), all_diagnostics, previous_accepted)
                _, diagnostic_hash, diagnostics, previous_accepted = batch_cache[batch]
                fingerprint = digest({"protocol": PROTOCOL, "source": group_hash,
                    "mfa": diagnostic_hash, "ctc_model": identity})
                destination = args.output / "groups" / (digest(original_id)[:24] + ".json")
                if destination.exists():
                    saved = json.loads(destination.read_text())
                    finite_acceptance = all(math.isfinite(r["independent_ctc_evidence"][key])
                        for r in saved.get("labels", []) if r["accepted"]
                        for key in ["min_character_probability", "mean_character_probability"])
                    if saved.get("fingerprint") == fingerprint and finite_acceptance:
                        active_results[original_id] = saved
                        covered_ids.add(original_id)
                        continue
                pending = [r for r in group["labels"] if not r["accepted"] and r["sample_id"] not in previous_accepted]
                inputs = {r["sample_id"]: r for r in group.get("segment_inputs", [])}
                if pending and not inputs:
                    manifest = {r["sample_id"]: r for r in read_jsonl(Path(run["manifest"]))}
                    inputs = manifest
                labels = []
                for label in pending:
                    audio, sr, binding = read_bound_audio(inputs[label["sample_id"]]["audio_path"])
                    expected = label["pcm_span"]
                    if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) != expected["end_sample"] - expected["start_sample"]:
                        raise ValueError(f"CTC input differs from ASR span: {label['sample_id']}")
                    if expected.get("pcm_sha256") and expected["pcm_sha256"] != binding["pcm_sha256"]:
                        raise ValueError(f"CTC input PCM differs from ASR binding: {label['sample_id']}")
                    cached_path = args.output / "segments" / (digest(label["sample_id"])[:24] + ".json")
                    ctc_fingerprint = digest({"audio": binding, "model": identity})
                    ctc = json.loads(cached_path.read_text()) if cached_path.exists() else {}
                    if ctc.get("fingerprint") != ctc_fingerprint:
                        if decoder is None:
                            decoder = Decoder(args.model)
                        ctc = decoder.decode(audio, binding)
                        ctc.update({"fingerprint": ctc_fingerprint, "sample_id": label["sample_id"]})
                        atomic_json(cached_path, ctc)
                    result = decide(label, group, diagnostics.get(label["sample_id"], {}), ctc)
                    result["independent_ctc_evidence"].update({"record": str(cached_path), "model_identity_file": str(args.output / "run.json")})
                    result["mfa_evidence"]["diagnostics_file"] = str(diagnostic_path)
                    result["asr_evidence_sha256"] = group_hash
                    labels.append(result)
                if hashlib.sha256(path.read_bytes()).hexdigest() != group_hash:
                    raise ValueError(f"ASR evidence changed during CTC verification: {original_id}")
                if hashlib.sha256(diagnostic_path.read_bytes()).hexdigest() != diagnostic_hash:
                    raise ValueError(f"MFA diagnostics changed during CTC verification: {original_id}")
                result = {"original_clip_id": original_id, "fingerprint": fingerprint,
                    "asr_evidence_sha256": group_hash, "labels": labels, "status": "complete"}
                atomic_json(destination, result)
                active_results[original_id] = result
                covered_ids.add(original_id)
                print(json.dumps({"original_clip_id": original_id, "reviewed": len(labels), "additional_accepted": sum(r["accepted"] for r in labels)}), flush=True)
            results = list(active_results.values())
            labels = [r for group in results for r in group["labels"]]
            accepted = [r for r in labels if r["accepted"]]
            atomic_jsonl(args.output / "labels.independent_ctc.accepted.jsonl", accepted)
            atomic_jsonl(args.output / "labels.review.jsonl", [r for r in labels if not r["accepted"]])
            report = {"protocol": PROTOCOL, "groups_processed": len(results), "segments_processed": len(labels),
                "additional_source_grounded_accepted": len(accepted), "accepted_hours": sum(r["duration"] for r in accepted) / 3600,
                "remaining_review": sum(not r["accepted"] for r in labels),
                "rejection_reasons": dict(collections.Counter(reason for r in labels if not r["accepted"] for reason in r["reasons"])),
                "asr_only_promotions": 0, "updated_unix": time.time(),
                "processing_complete": covered_ids == wanted,
                "originals_with_ctc_completed_or_unnecessary": len(covered_ids),
                "limitations": ["Model agreement is evidence, not manual or original transcription.", "This separate layer does not overwrite earlier accepted labels or imply full recovery."]}
            atomic_json(args.output / "summary.json", report)
            print(json.dumps(report), flush=True)
            if not any(start is not None and process_start(pid) == start for pid, start in followed.items()):
                if final_sweep:
                    break
                final_sweep = True
                continue
            time.sleep(20)


if __name__ == "__main__":
    main()
