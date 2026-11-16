#!/usr/bin/env python3
"""Export source-grounded LRS3 labels and a distinct conservative pseudo tier.

The default training publication consumes labels.strict.accepted.jsonl only.
P1 pseudo text is genuine unprompted segment ASR, never an unverified source
candidate. P2, empty/no-speech, weak-confidence, and boundary failures stay out.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, raw_tokens, tokens, word_tokens
from lrs3_text_recovery_pipeline import process_start

POLICY = "lrs3_asr_pseudo_p1_v1"
THRESHOLDS = {"min_word_probability": .50, "mean_word_probability": .80,
    "min_segment_avg_logprob": -.75, "max_no_speech_probability": .30,
    "max_compression_ratio": 2.4, "ctc_min_char_probability": .80,
    "ctc_mean_char_probability": .97, "minimum_word_count": 3,
    "minimum_duration_seconds": 1.0, "clip_boundary_tolerance_seconds": .04,
    "group_boundary_tolerance_seconds": .12}
SOURCE_KINDS = {
    "accepted_source_group_segment_consensus": "source_asr_consensus",
    "accepted_source_group_segment_mfa_boundary_consensus": "source_asr_mfa_consensus",
    "accepted_source_whisper_ctc_mfa_consensus": "source_asr_ctc_consensus",
}
REQUIRED_RELEASE_FILES = frozenset({
    "labels.strict.accepted.jsonl", "labels.pseudo_p1.accepted.jsonl",
    "labels.pseudo_p2.candidates.jsonl", "labels.excluded.jsonl",
    "labels.pending.jsonl", "coverage_by_original.jsonl", "summary.json", "rules.json",
})


def frozen_output_hashes(output, expected_summary):
    """Pin actual output bytes, refusing an incomplete or inconsistent release."""
    paths = {p.name: p for p in output.iterdir()
             if p.is_file() and p.suffix in {".json", ".jsonl"} and p.name != "frozen_export.json"}
    missing = REQUIRED_RELEASE_FILES - paths.keys()
    if missing:
        raise ValueError(f"Cannot freeze text release; missing required outputs: {sorted(missing)}")
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    actual_summary = json.loads(payloads["summary.json"])
    if actual_summary != expected_summary:
        raise ValueError("Text summary changed before frozen output hashing")
    if not actual_summary.get("processing_complete") or actual_summary.get("pending_segments") != 0:
        raise ValueError("Cannot freeze incomplete text release")
    if payloads["labels.pending.jsonl"].strip():
        raise ValueError("Cannot freeze text release containing pending labels")
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}


def pseudo_candidate(row, label, group, decode, ctc):
    words = decode["words"]
    pieces = decode["segments"]
    hypothesis = tokens(decode["text"])
    literal_word_count = len(raw_tokens(decode["text"]))
    confidence = {
        "min_word_probability": min((w["probability"] for w in words), default=None),
        "mean_word_probability": sum(w["probability"] for w in words) / len(words) if words else None,
        "min_segment_avg_logprob": min((p["avg_logprob"] for p in pieces), default=None),
        "max_no_speech_probability": max((p["no_speech_prob"] for p in pieces), default=None),
        "max_compression_ratio": max((p["compression_ratio"] for p in pieces), default=None),
        "ctc_min_char_probability": ctc.get("min_character_probability"),
        "ctc_mean_char_probability": ctc.get("mean_character_probability"),
    }
    full_words = word_tokens(group["full_asr"]["words"])
    group_range = label.get("group_asr_token_range")
    supported_words = full_words[group_range[0]:group_range[1]] if group_range else []
    support = (bool(hypothesis) and bool(group_range) and group_range[1] - group_range[0] == len(hypothesis)
               and [w["token"] for w in supported_words] == hypothesis)
    span = label["pcm_span"]
    begin, end = span["start_sample"] / span["sample_rate"], span["end_sample"] / span["sample_rate"]
    duration = end - begin
    gates = {
        "ctc_text_exact": bool(hypothesis) and tokens(ctc.get("text", "")) == hypothesis,
        "group_contiguous_support": support,
        "word_text_exact": bool(words) and [w["token"] for w in word_tokens(words)] == hypothesis,
        "positive_word_durations": bool(words) and all(w["end"] > w["start"] for w in words),
        "within_pcm_bounds": bool(words) and all(w["start"] >= -.04 and w["end"] <= duration + .04 for w in words),
        "group_boundary_consistent": support and all(w["start"] >= begin - .12 and w["end"] <= end + .12 for w in supported_words),
    }
    reasons = []
    if row["split"] != "train":
        reasons.append("pseudo_labels_not_allowed_in_validation_or_test")
    if not hypothesis:
        reasons.append("empty_or_no_speech_transcript")
    if literal_word_count < 3 or duration < 1.0:
        reasons.append("insufficient_words_or_duration")
    for key in ["min_word_probability", "mean_word_probability", "min_segment_avg_logprob"]:
        if confidence[key] is None or not math.isfinite(confidence[key]) or confidence[key] < THRESHOLDS[key]:
            reasons.append("low_" + key)
    for key in ["max_no_speech_probability", "max_compression_ratio"]:
        if confidence[key] is None or not math.isfinite(confidence[key]) or confidence[key] > THRESHOLDS[key]:
            reasons.append("high_or_missing_" + key)
    for key, value in gates.items():
        if key != "ctc_text_exact" and not value:
            reasons.append(key + "_failed")
    ctc_confident = all(confidence[k] is not None and math.isfinite(confidence[k]) and confidence[k] >= THRESHOLDS[k]
                        for k in ["ctc_min_char_probability", "ctc_mean_char_probability"])
    p2 = not reasons and not (gates["ctc_text_exact"] and ctc_confident)
    if not gates["ctc_text_exact"]:
        reasons.append("independent_ctc_text_differs")
    if not ctc_confident:
        reasons.append("independent_ctc_character_confidence_insufficient")
    accepted = not reasons
    text = " ".join(raw_tokens(decode["text"]))
    if tokens(text) != hypothesis:
        raise ValueError("Pseudo text normalization changed ASR words")
    return {"sample_id": row["sample_id"], "original_clip_id": row["original_clip_id"],
        "text": text, "accepted": accepted, "training_eligible": accepted,
        "source_kind": "asr_pseudo", "text_status": "asr_pseudo" if accepted else "asr_pseudo_candidate",
        "independent_content_verified": False, "acceptance_policy_version": POLICY,
        "source": "Whisper_large_v3_unprompted_actual_segment_decode",
        "quality": "asr_pseudo_p1_multimodel_consensus" if accepted else ("asr_pseudo_p2_same_model_candidate" if p2 else "excluded_asr_candidate"),
        "duration": duration, "word_count": literal_word_count, "confidence": confidence, "quality_gates": gates,
        "asr_raw_text": decode["text"], "asr_word_spans": words, "asr_segments": pieces,
        "ctc_text": ctc.get("text"), "original_source_review_reasons": label["reasons"],
        "reasons": reasons, "split": row["split"], "pcm_span": span}


class Exporter:
    def __init__(self, args):
        self.args = args
        self.cache = {}
        self.binding_cache = {}
        self.run, self.run_ref = self.read(args.evidence_dir / "run.json")
        self.wanted = set(self.run["group_ids"])
        if len(self.wanted) != self.run["groups_requested"]:
            raise ValueError("ASR requested original IDs/count disagree")
        self.source_code_sha256_at_start = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.rows = {}
        self.by_original = collections.defaultdict(list)
        with args.manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["original_clip_id"] in self.wanted:
                    self.rows[row["sample_id"]] = row
                    self.by_original[row["original_clip_id"]].append(row)
        self.manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if self.manifest_hash != self.run["manifest_sha256"]:
            raise ValueError("Current input manifest differs from ASR launch manifest")

    def read(self, path, jsonl=False):
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if path not in self.cache or self.cache[path][0] != stamp:
            payload = path.read_bytes()
            value = [json.loads(line) for line in payload.splitlines() if line.strip()] if jsonl else json.loads(payload)
            self.cache[path] = (stamp, value, {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
        return self.cache[path][1:]

    def bind(self, row, group_ref, audit, audit_ref, ctc=None):
        source = Path(row["audio_path"])
        stat = source.stat()
        receipt_path = self.args.audio_receipts / (row["relative_stem"] + ".json")
        if not receipt_path.exists():
            raise ValueError("audio_cache_receipt_not_yet_available")
        receipt, receipt_ref = self.read(receipt_path)
        audit_segment = next((s for s in audit["segments"] if s["sample_id"] == row["sample_id"]), None)
        if audit.get("asr_evidence_sha256") != group_ref["sha256"] or not audit_segment or audit_segment["issues"]:
            raise ValueError("posthoc_source_audit_missing_or_mismatched")
        if audit_segment["source_mtime_unix"] != stat.st_mtime or not audit_segment["mtime_before_asr_start"]:
            raise ValueError("source_mtime_changed_after_posthoc_audit")
        signature = (stat.st_mtime_ns, stat.st_size, audit_ref["sha256"], receipt_ref["sha256"], group_ref["sha256"])
        cached = self.binding_cache.get(row["sample_id"])
        if not cached or cached[0] != signature:
            actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            after = source.stat()
            if (after.st_mtime_ns, after.st_size) != (stat.st_mtime_ns, stat.st_size):
                raise ValueError("source_changed_during_export_hash")
            if actual_hash != audit_segment["file_sha256_posthoc"] or actual_hash != receipt["identity"]["audio_sha256"]:
                raise ValueError("current_source_audit_and_cache_sha_disagree")
            self.binding_cache[row["sample_id"]] = (signature, actual_hash)
        audio_hash = self.binding_cache[row["sample_id"]][1]
        if ctc and ctc["source_binding"]["file_sha256"] != audio_hash:
            raise ValueError("ctc_actual_input_hash_differs_from_audio_cache")
        return {"source_audio_sha256": audio_hash,
            "source_audio_binding_kind": "asr_decode_time_and_cache" if audit_segment["asr_decode_time_hash_available"] else "posthoc_source_check_and_audio_cache",
            "legacy_asr_decode_time_hash_available": audit_segment["asr_decode_time_hash_available"],
            "evidence_refs": [self.run_ref, group_ref, audit_ref, receipt_ref]}

    def export(self):
        if self.read(self.args.evidence_dir / "run.json")[1] != self.run_ref:
            raise ValueError("ASR run contract changed; restart exporter after validating the new run")
        strict, p1, p2, excluded, pending, coverage = [], [], [], [], [], []
        ctc_run_ref = self.read(self.args.ctc_dir / "run.json")[1]
        completed = set()
        for original_id in sorted(self.wanted):
            tag = digest(original_id)[:24] + ".json"
            group_path = self.args.evidence_dir / "groups" / tag
            counts = collections.Counter()
            hours = collections.Counter()
            row_status = {}
            if not group_path.exists():
                coverage.append({"original_clip_id": original_id, "status": "asr_pending", "expected_segments": len(self.by_original[original_id])})
                continue
            group, group_ref = self.read(group_path)
            if group.get("status") != "complete":
                coverage.append({"original_clip_id": original_id, "status": "asr_error", "error": group.get("error")})
                continue
            expected_ids = {r["sample_id"] for r in self.by_original[original_id]}
            actual_ids = [r["sample_id"] for r in group["labels"]]
            if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
                raise ValueError(f"ASR segment IDs do not exactly match manifest: {original_id}")
            extra, mfa_refs = {}, []
            receipt_path = self.args.mfa_dir / "receipts" / tag
            receipt = None
            if receipt_path.exists():
                receipt, receipt_ref = self.read(receipt_path)
                if receipt.get("source_evidence_sha256") != group_ref["sha256"]:
                    receipt = None
                elif receipt["status"] == "mfa_processed":
                    batch = Path(receipt["batch"])
                    additions, add_ref = self.read(batch / "labels.mfa_additional.accepted.jsonl", jsonl=True)
                    extra.update({r["sample_id"]: r for r in additions if r["original_clip_id"] == original_id})
                    mfa_refs = [receipt_ref, add_ref]
            ctc_group_path = self.args.ctc_dir / "groups" / tag
            ctc_group, ctc_ref = None, None
            if ctc_group_path.exists():
                ctc_group, ctc_ref = self.read(ctc_group_path)
                if ctc_group.get("asr_evidence_sha256") != group_ref["sha256"]:
                    ctc_group = None
                else:
                    for label in ctc_group["labels"]:
                        if label["accepted"]:
                            if label["sample_id"] in extra:
                                raise ValueError("Strict MFA and CTC addition sets overlap")
                            extra[label["sample_id"]] = label
            audit_path = self.args.provenance_dir / "groups" / tag
            audit, audit_ref = self.read(audit_path) if audit_path.exists() else (None, None)
            decodes = {r["sample_id"]: r for r in group["segment_asr"]}
            for original_label in group["labels"]:
                sid = original_label["sample_id"]
                row = self.rows[sid]
                label = original_label if original_label["accepted"] else extra.get(sid)
                ctc_path = self.args.ctc_dir / "segments" / (digest(sid)[:24] + ".json")
                ctc, ctc_raw_ref = self.read(ctc_path) if ctc_path.exists() else (None, None)
                try:
                    if audit is None:
                        raise ValueError("posthoc_audit_pending")
                    binding = self.bind(row, group_ref, audit, audit_ref, ctc)
                except ValueError as exc:
                    pending.append({"sample_id": sid, "original_clip_id": original_id, "reason": str(exc), "duration": original_label["duration"]})
                    row_status[sid] = "provenance_pending_or_failed"
                    continue
                if label is not None:
                    exported = dict(label)
                    exported.update(binding)
                    exported.update({"source_kind": SOURCE_KINDS[label["quality"]], "text_status": "recovered_source_consensus",
                        "training_eligible": True, "independent_content_verified": False,
                        "acceptance_policy_version": "lrs3_source_grounded_consensus_export_v1", "split": row["split"]})
                    if label["quality"] == "accepted_source_group_segment_mfa_boundary_consensus":
                        exported["evidence_refs"] += mfa_refs
                    elif label["quality"] == "accepted_source_whisper_ctc_mfa_consensus":
                        exported["evidence_refs"] += [ctc_run_ref, ctc_ref, ctc_raw_ref]
                    strict.append(exported)
                    category = "strict"
                elif ctc is None or ctc_group is None:
                    pending.append({"sample_id": sid, "original_clip_id": original_id, "reason": "independent_ctc_pending", "duration": original_label["duration"]})
                    category = "ctc_pending"
                else:
                    exported = pseudo_candidate(row, original_label, group, decodes[sid], ctc)
                    exported.update(binding)
                    exported["evidence_refs"] += [ctc_run_ref, ctc_ref, ctc_raw_ref]
                    if exported["accepted"]:
                        p1.append(exported)
                        category = "pseudo_p1"
                    elif exported["quality"] == "asr_pseudo_p2_same_model_candidate":
                        p2.append(exported)
                        category = "pseudo_p2_candidate"
                    else:
                        excluded.append(exported)
                        category = "excluded"
                row_status[sid] = category
                counts[category] += 1
                hours[category] += original_label["duration"] / 3600
            stage_complete = (receipt is not None and receipt["status"] in {"mfa_processed", "no_review_needed"}
                              and (ctc_group is not None or receipt["status"] == "no_review_needed")
                              and set(row_status) == expected_ids
                              and not any("pending" in s or "failed" in s for s in row_status.values()))
            if stage_complete:
                completed.add(original_id)
            coverage.append({"original_clip_id": original_id, "status": "processed" if stage_complete else "stages_pending",
                "expected_segments": len(self.by_original[original_id]), "counts": dict(counts), "hours": dict(hours),
                "strict_covers_all_segments": counts["strict"] == len(self.by_original[original_id]),
                "strict_plus_p1_covers_all_segments": counts["strict"] + counts["pseudo_p1"] == len(self.by_original[original_id]),
                "uncovered_sample_ids": [sid for sid, state in row_status.items() if state not in {"strict", "pseudo_p1"}]})
        outputs = {"labels.strict.accepted.jsonl": strict, "labels.pseudo_p1.accepted.jsonl": p1,
            "labels.pseudo_p2.candidates.jsonl": p2, "labels.excluded.jsonl": excluded,
            "labels.pending.jsonl": pending, "coverage_by_original.jsonl": coverage}
        for name, rows in outputs.items():
            atomic_jsonl(self.args.output / name, rows)
        asr_summary_path = self.args.evidence_dir / "summary.json"
        asr_summary = self.read(asr_summary_path)[0] if asr_summary_path.exists() else {}
        summary = {"groups_requested": len(self.wanted), "groups_complete": len(completed),
            "segments_requested": len(self.rows), "strict_segments": len(strict), "pseudo_p1_segments": len(p1),
            "pseudo_p2_candidate_segments": len(p2), "excluded_segments": len(excluded), "pending_segments": len(pending),
            "strict_hours": sum(r["duration"] for r in strict) / 3600,
            "pseudo_p1_hours": sum(r["duration"] for r in p1) / 3600,
            "pseudo_p2_candidate_hours": sum(r["duration"] for r in p2) / 3600,
            "excluded_hours": sum(r["duration"] for r in excluded) / 3600,
            "excluded_reasons": dict(collections.Counter(reason for r in excluded for reason in r["reasons"])),
            "asr_and_requested_controls_complete": asr_summary.get("complete", False),
            "processing_complete": completed == self.wanted and asr_summary.get("complete", False),
            "all_requested_segments_source_grounded": len(strict) == len(self.rows),
            "strict_and_pseudo_are_separate_files": True, "updated_unix": time.time()}
        atomic_json(self.args.output / "summary.json", summary)
        atomic_json(self.args.output / "rules.json", {"pseudo_policy_version": POLICY, "pseudo_p1_thresholds": THRESHOLDS,
            "export_source_code_sha256_at_start": self.source_code_sha256_at_start,
            "required_quality_gates": ["ctc_text_exact", "group_contiguous_support", "word_text_exact", "positive_word_durations", "within_pcm_bounds", "group_boundary_consistent"],
            "pseudo_allowed_split": "train", "pseudo_requires_explicit_text_policy": "source_plus_asr_pseudo",
            "minimum_word_count_counts_literal_asr_words_before_comparison_expansion": True,
            "strict_default_excludes_pseudo": True, "source_kind_by_strict_quality": SOURCE_KINDS,
            "pseudo_p2_is_not_training_eligible": True,
            "asr_run_ref": self.run_ref, "source_manifest": {"path": str(self.args.manifest), "sha256": self.manifest_hash},
            "legacy_source_binding": "Explicit posthoc source/PCM-shape checks, unchanged mtimes, current WAV SHA and independent audio-cache SHA agree; this is not a legacy ASR decode-time hash."})
        print(json.dumps(summary), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "evidence-dir", "mfa-dir", "ctc-dir", "provenance-dir", "audio-receipts", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--follow-pids", default="")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "frozen_export.json").exists():
        raise SystemExit("Frozen release exists; choose another output path to recompute")
    with (args.output / "export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exporter = Exporter(args)
        followed = {int(p): process_start(int(p)) for p in args.follow_pids.split(",") if p.strip()}
        final_sweep = False
        while True:
            summary = exporter.export()
            if not any(start is not None and process_start(pid) == start for pid, start in followed.items()):
                if final_sweep:
                    break
                final_sweep = True
                continue
            time.sleep(30)
        if args.freeze:
            if not summary["processing_complete"]:
                raise SystemExit("Refusing to freeze incomplete text processing")
            refs = {}
            for path in args.output.glob("labels.*.jsonl"):
                with path.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        for ref in row.get("evidence_refs", []):
                            if refs.get(ref["path"], ref["sha256"]) != ref["sha256"]:
                                raise ValueError(f"Multiple exported hashes for one evidence file: {ref['path']}")
                            refs[ref["path"]] = ref["sha256"]
            refs[str(args.manifest.resolve())] = exporter.manifest_hash
            refs[exporter.run_ref["path"]] = exporter.run_ref["sha256"]
            for path, expected in refs.items():
                hasher = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != expected:
                    raise ValueError(f"Evidence changed before frozen export: {path}")
            hashes = frozen_output_hashes(args.output, summary)
            atomic_json(args.output / "frozen_export.json", {"frozen_unix": time.time(), "files": hashes,
                "source_manifest_sha256": exporter.manifest_hash, "unique_evidence_hashes_rechecked": len(refs), "summary": summary})


if __name__ == "__main__":
    main()
