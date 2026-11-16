#!/usr/bin/env python3
"""Build a human-review queue from a frozen text release; never approve labels."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, tokens


def verified_bytes(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Frozen evidence changed: {path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.release.resolve():
        raise ValueError("Human review output must be separate from the frozen release")
    frozen_bytes = (args.release / "frozen_export.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    if not frozen["summary"]["processing_complete"]:
        raise ValueError("Frozen text release is incomplete")
    by_partition = {}
    for partition, name in [("strict", "labels.strict.accepted.jsonl"),
                            ("pseudo_p1", "labels.pseudo_p1.accepted.jsonl"),
                            ("pseudo_p2", "labels.pseudo_p2.candidates.jsonl"),
                            ("excluded", "labels.excluded.jsonl")]:
        by_partition[partition] = [json.loads(line) for line in verified_bytes(args.release / name, frozen["files"][name]).splitlines()]
    all_labels = [r for labels in by_partition.values() for r in labels]
    ids = [r["sample_id"] for r in all_labels]
    if len(set(ids)) != len(ids) or len(ids) != frozen["summary"]["segments_requested"]:
        raise ValueError("Frozen label partitions overlap or have incorrect coverage")
    requested_ids = set(ids)
    manifest_bytes = verified_bytes(args.manifest, frozen["source_manifest_sha256"])
    manifest = {r["sample_id"]: r for line in manifest_bytes.splitlines()
                if (r := json.loads(line))["sample_id"] in requested_ids}
    if set(manifest) != requested_ids:
        raise ValueError("Frozen sample IDs are absent from the manifest")
    del manifest_bytes
    group_cache = {}
    queue = []
    for partition in ["pseudo_p2", "excluded"]:
        for row in by_partition[partition]:
            sid, oid = row["sample_id"], row["original_clip_id"]
            source = manifest[sid]
            path = (args.evidence_dir / "groups" / (digest(oid)[:24] + ".json")).resolve()
            ref = next(r for r in row["evidence_refs"] if Path(r["path"]).resolve() == path)
            if oid not in group_cache:
                group_cache[oid] = json.loads(verified_bytes(path, ref["sha256"]))
            group = group_cache[oid]
            original = next(r for r in group["labels"] if r["sample_id"] == sid)
            span = original.get("candidate_source_token_range")
            source_words = group["source_comparison"]["raw_words"]
            source_spans = group["source_comparison"]["raw_word_spans"]
            candidate_words = [word for word, bounds in zip(source_words, source_spans)
                               if span and bounds[1] > span[0] and bounds[0] < span[1]]
            dual_agree = bool(tokens(row["text"])) and tokens(row["text"]) == tokens(row.get("ctc_text") or "")
            category = ("p2_dual_agreement_low_ctc_confidence" if dual_agree else "p2_recognizer_disagreement") if partition == "pseudo_p2" else (
                "other_dual_agreement_quality_or_boundary_failure" if dual_agree else "other_recognizer_or_boundary_failure")
            queue.append({"sample_id": sid, "original_clip_id": oid, "duration": row["duration"],
                "split": source["split"], "audio_path": source["audio_path"], "video_path": source["video_path"],
                "source_audio_sha256": row["source_audio_sha256"], "partition": partition, "review_category": category,
                "source_candidate_excerpt_unverified": " ".join(candidate_words) if candidate_words else None,
                "source_candidate_group_text_unverified": " ".join(source_words),
                "whisper_segment_text_unverified": row["asr_raw_text"], "ctc_text_unverified": row.get("ctc_text"),
                "original_source_review_reasons": row["original_source_review_reasons"],
                "pseudo_rejection_reasons": row["reasons"], "confidence": row["confidence"],
                "source_evidence": ref, "accepted": False, "training_eligible": False,
                "human_annotation_status": "not_reviewed"})
    categories = ["p2_dual_agreement_low_ctc_confidence", "p2_recognizer_disagreement",
                  "other_dual_agreement_quality_or_boundary_failure", "other_recognizer_or_boundary_failure"]
    queue.sort(key=lambda r: (categories.index(r["review_category"]), -r["duration"], r["sample_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output / "segments_for_human_review.jsonl", queue)
    buffer = io.StringIO()
    fields = ["sample_id", "original_clip_id", "duration", "partition", "review_category", "audio_path", "video_path",
              "source_candidate_excerpt_unverified", "whisper_segment_text_unverified", "ctc_text_unverified", "pseudo_rejection_reasons"]
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in queue:
        writer.writerow({**row, "pseudo_rejection_reasons": ";".join(row["pseudo_rejection_reasons"])})
    temporary = args.output / "segments_for_human_review.tsv.tmp"
    temporary.write_text(buffer.getvalue())
    temporary.replace(args.output / "segments_for_human_review.tsv")
    originals = collections.defaultdict(list)
    for row in queue:
        originals[row["original_clip_id"]].append(row)
    atomic_jsonl(args.output / "originals_with_remaining_review.jsonl", [
        {"original_clip_id": oid, "review_segments": len(rows), "review_hours": sum(r["duration"] for r in rows) / 3600,
         "sample_ids": [r["sample_id"] for r in rows]} for oid, rows in sorted(originals.items())])
    summary = {"frozen_export_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "source_manifest_sha256": frozen["source_manifest_sha256"],
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partitions": {k: {"segments": len(v), "hours": sum(r["duration"] for r in v) / 3600} for k, v in by_partition.items()},
        "segments_for_review": len(queue), "originals_with_remaining_review": len(originals),
        "hours_for_review": sum(r["duration"] for r in queue) / 3600,
        "review_categories": {category: {"segments": sum(r["review_category"] == category for r in queue),
            "hours": sum(r["duration"] for r in queue if r["review_category"] == category) / 3600} for category in categories},
        "source_or_recognizer_candidates_are_not_approved_labels": True,
        "review_order_is_workflow_priority_not_a_claim_of_transcription_accuracy": True}
    if (args.release / "frozen_export.json").read_bytes() != frozen_bytes:
        raise ValueError("Frozen release changed while generating human-review queue")
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
