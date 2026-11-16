#!/usr/bin/env python3
"""Prepare paired source/ASR hypotheses for independent MFA boundary diagnostics.

Alignment success alone never promotes a label. Extra acceptance requires source,
full-recording ASR, independent clip ASR, and bounded MFA words, while keeping
the original ASR confidence thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import collections

from lrs3_text_recovery import atomic_json, atomic_jsonl, read_jsonl, tokens


def analyze(output):
    records = json.loads((output / "corpus_manifest.json").read_text())["records"]
    database = output / "tmp" / "corpus" / "corpus.db"
    connection = sqlite3.connect("file:" + str(database.resolve()) + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    alignments = {}
    for row in connection.execute("SELECT f.name,u.id,u.alignment_log_likelihood,u.speech_log_likelihood,u.duration_deviation,u.oovs,u.ignored FROM utterance u JOIN file f ON f.id=u.file_id"):
        result = dict(row)
        result["words"] = [dict(w) for w in connection.execute(
            "SELECT wi.begin,wi.end,w.word FROM word_interval wi JOIN word w ON w.id=wi.word_id WHERE wi.utterance_id=? ORDER BY wi.begin", (row["id"],))]
        alignments[row["name"]] = result
    grouped = collections.defaultdict(dict)
    diagnostics = []
    additional = []
    promotion_reviews = []
    allowed_boundary_reasons = {"source_conflict_at_or_near_segment", "group_asr_conflict_at_or_near_segment"}
    for record in records:
        alignment = alignments.get(record["id"], {})
        grouped[record["original_clip_id"]][record["hypothesis"]] = alignment
        if record["hypothesis"] != "source":
            continue
        evidence_bytes = Path(record["asr_evidence"]).read_bytes()
        if not record.get("asr_evidence_sha256") or hashlib.sha256(evidence_bytes).hexdigest() != record["asr_evidence_sha256"]:
            raise ValueError(f"Missing or changed prepared ASR evidence: {record['original_clip_id']}")
        evidence = json.loads(evidence_bytes)
        decodes = {r["sample_id"]: r for r in evidence["segment_asr"]}
        labels = {r["sample_id"]: r for r in evidence["labels"]}
        for span in record["segments"]:
            begin, end = span["start_sample"] / span["sample_rate"], span["end_sample"] / span["sample_rate"]
            words = [w for w in alignment.get("words", []) if begin <= (w["begin"] + w["end"]) / 2 < end and w["word"] not in {"sil", "sp", "", "<eps>"}]
            asr = decodes[span["sample_id"]]
            assigned = " ".join(w["word"] for w in words)
            diagnostic = {"sample_id": span["sample_id"], "original_clip_id": record["original_clip_id"],
                "mfa_source_assigned_text": assigned, "segment_asr_text": asr["text"],
                "word_sequence_matches": tokens(assigned) == tokens(asr["text"]),
                "mfa_group_oovs": alignment.get("oovs"), "source_alignment_log_likelihood": alignment.get("alignment_log_likelihood"),
                "words": words, "pcm_span": span, "current_quality": labels[span["sample_id"]]["quality"],
                "status": "diagnostic_only_not_training_approval"}
            diagnostics.append(diagnostic)
            label = labels[span["sample_id"]]
            if label["quality"] == "review_required" and diagnostic["word_sequence_matches"]:
                reasons = []
                if not set(label["reasons"]).issubset(allowed_boundary_reasons):
                    reasons.append("existing_content_or_acoustic_confidence_failure")
                if not words or any(w["word"].startswith("<") or w["word"] in {"spn", "sil", "sp"} for w in words):
                    reasons.append("unknown_or_nonlexical_alignment_word")
                if words and (min(w["begin"] for w in words) < begin - .04 or max(w["end"] for w in words) > end + .04):
                    reasons.append("mfa_word_outside_actual_segment_by_more_than_one_video_frame")
                if words and min(w["end"] - w["begin"] for w in words) < .0199:
                    reasons.append("mfa_word_shorter_than_two_acoustic_frames")
                source_range = label.get("candidate_source_token_range")
                raw_label = []
                source = evidence["source_comparison"]
                if source_range:
                    selected = [(w, b) for w, b in zip(source["raw_words"], source["raw_word_spans"]) if b[1] > source_range[0] and b[0] < source_range[1]]
                    if selected and selected[0][1][0] == source_range[0] and selected[-1][1][1] == source_range[1]:
                        raw_label = [w for w, _ in selected]
                if tokens(" ".join(raw_label)) != tokens(assigned):
                    reasons.append("aligned_text_does_not_equal_complete_source_word_span")
                promotion_reviews.append({"sample_id": span["sample_id"], "original_reasons": label["reasons"],
                    "additional_rejection_reasons": reasons, "accepted": not reasons,
                    "rule": "Keep original ASR confidence/content tests; resolve only adjacent conflicts using independently aligned source words contained in the actual PCM segment."})
                if not reasons:
                    promoted = dict(label)
                    promoted.update({"text": " ".join(raw_label), "quality": "accepted_source_group_segment_mfa_boundary_consensus",
                        "accepted": True, "reasons": [], "prior_review_reasons": label["reasons"],
                        "mfa_evidence": {"diagnostics_file": str(output / "boundary_diagnostics.jsonl"),
                            "asr_evidence_sha256": record["asr_evidence_sha256"],
                            "source_alignment_log_likelihood": alignment.get("alignment_log_likelihood"),
                            "boundary_tolerance_seconds": .04, "minimum_word_seconds": .02,
                            "min_word_start": min(w["begin"] for w in words), "max_word_end": max(w["end"] for w in words),
                            "confidence_thresholds_unchanged": True}})
                    additional.append(promoted)
    pairs = []
    for original_id, alignments_by_kind in grouped.items():
        source, asr = alignments_by_kind.get("source", {}), alignments_by_kind.get("asr", {})
        s, a = source.get("alignment_log_likelihood"), asr.get("alignment_log_likelihood")
        pairs.append({"original_clip_id": original_id, "source_score": s, "asr_score": a,
                      "asr_minus_source": a - s if a is not None and s is not None else None,
                      "source_oovs": source.get("oovs"), "asr_oovs": asr.get("oovs")})
    atomic_jsonl(output / "boundary_diagnostics.jsonl", diagnostics)
    atomic_jsonl(output / "hypothesis_comparison.jsonl", pairs)
    atomic_jsonl(output / "boundary_promotion_review.jsonl", promotion_reviews)
    atomic_jsonl(output / "labels.mfa_additional.accepted.jsonl", additional)
    report = {"hypotheses_requested": len(records), "hypotheses_with_alignment": sum(r.get("alignment_log_likelihood") is not None for r in alignments.values()),
        "original_groups": len(grouped), "segments_with_diagnostics": len(diagnostics),
        "mfa_source_segment_asr_exact": sum(r["word_sequence_matches"] for r in diagnostics),
        "review_segments_with_mfa_source_asr_agreement": sum(r["word_sequence_matches"] and r["current_quality"] == "review_required" for r in diagnostics),
        "additional_boundary_consensus_labels": len(additional),
        "original_acceptance_files_modified": False,
        "limitations": ["A successful forced alignment is not evidence that all words were spoken.",
                        "MFA dictionary OOVs and partial words limit boundary reliability.",
                        "Paired acoustic scores are diagnostic and are not calibrated transcript confidence scores."]}
    atomic_json(output / "diagnostic_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--group-ids-file", type=Path)
    parser.add_argument("--expected-evidence-file", type=Path)
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.output), indent=2))
        return
    import numpy as np
    import soundfile as sf
    rows = None
    selected_ids = set(json.loads(args.group_ids_file.read_text())) if args.group_ids_file else None
    expected_hashes = json.loads(args.expected_evidence_file.read_text()) if args.expected_evidence_file else None
    records = []
    for path in sorted((args.evidence_dir / "groups").glob("*.json")):
        evidence_bytes = path.read_bytes()
        evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
        group = json.loads(evidence_bytes)
        if group.get("status") != "complete":
            continue
        original_id = group["original_clip_id"]
        if selected_ids is not None and original_id not in selected_ids:
            continue
        if expected_hashes is not None and evidence_hash != expected_hashes.get(original_id):
            raise ValueError(f"ASR evidence changed before MFA preparation: {original_id}")
        if args.review_only and all(label["accepted"] for label in group["labels"]):
            continue
        if group.get("segment_inputs"):
            group_rows = {r["sample_id"]: r for r in group["segment_inputs"]}
        else:
            if rows is None:
                rows = {r["sample_id"]: r for r in read_jsonl(args.manifest)}
            group_rows = rows
        speaker = original_id.split("/")[-2]
        samples = []
        for label in group["labels"]:
            audio, sr = sf.read(group_rows[label["sample_id"]]["audio_path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid MFA audio: {label['sample_id']}")
            span = label["pcm_span"]
            if len(audio) != span["end_sample"] - span["start_sample"]:
                raise ValueError(f"Audio length changed after ASR: {label['sample_id']}")
            expected_pcm = span.get("pcm_sha256")
            if expected_pcm is not None and hashlib.sha256(audio.tobytes()).hexdigest() != expected_pcm:
                raise ValueError(f"Audio PCM changed after ASR: {label['sample_id']}")
            samples.append(audio)
        combined = np.concatenate(samples)
        hypotheses = [("source", " ".join(group["source_comparison"]["tokens"]))]
        if not args.source_only:
            hypotheses.append(("asr", " ".join(tokens(group["full_asr"]["text"]))))
        for hypothesis, text in hypotheses:
            stem = hashlib.sha256(original_id.encode()).hexdigest()[:16] + "_" + hypothesis
            target = args.output / "corpus" / speaker / stem
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target.with_suffix(".wav"), combined, 16000, subtype="PCM_16")
            target.with_suffix(".lab").write_text(text + "\n")
            records.append({"id": stem, "original_clip_id": original_id, "hypothesis": hypothesis,
                            "speaker": speaker, "text": text, "wav": str(target.with_suffix(".wav")),
                            "audio_seconds": len(combined) / 16000,
                            "segments": [r["pcm_span"] for r in group["labels"]],
                            "asr_evidence": str(path.resolve()), "asr_evidence_sha256": evidence_hash,
                            "audio_pcm_verified_against_asr": all(r["pcm_span"].get("pcm_sha256") for r in group["labels"]),
                            "status": "alignment_diagnostic_only"})
    if selected_ids is not None and {r["original_clip_id"] for r in records} != selected_ids:
        raise ValueError("Prepared MFA groups do not exactly match requested batch")
    atomic_json(args.output / "corpus_manifest.json", {"records": records,
        "purpose": "Compare source and ASR alignment/boundaries; alignment success alone never validates transcript content."})
    print(json.dumps({"utterances": len(records), "source_groups": len({r["original_clip_id"] for r in records}),
                      "corpus": str((args.output / "corpus").resolve())}))


if __name__ == "__main__":
    main()
