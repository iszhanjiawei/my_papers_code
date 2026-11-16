#!/usr/bin/env python3
"""Audit current source bytes against ASR spans without inventing past hashes.

Legacy ASR jobs did not bind source bytes at decode time. This separate audit
explicitly records posthoc checks, source modification times, and independently
available VAE source receipts. It never rewrites the original ASR evidence.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import time

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, read_jsonl
from lrs3_text_recovery_pipeline import process_start


def audit_group(path, rows, audio_receipts):
    import numpy as np
    import soundfile as sf
    payload = path.read_bytes()
    group = json.loads(payload)
    published = path.stat().st_mtime
    started = group.get("asr_started_unix", published - group["elapsed_seconds"])
    bindings = {r["sample_id"]: r for r in group.get("audio_source_bindings", [])}
    results = []
    for label in group["labels"]:
        row = rows[label["sample_id"]]
        source = Path(row["audio_path"])
        before = source.stat()
        raw = source.read_bytes()
        after = source.stat()
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        file_hash = hashlib.sha256(raw).hexdigest()
        pcm_hash = hashlib.sha256(audio.tobytes()).hexdigest()
        span = label["pcm_span"]
        receipt_path = audio_receipts / (row["relative_stem"] + ".json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        receipt_hash = receipt.get("identity", {}).get("audio_sha256") if receipt else None
        binding = bindings.get(label["sample_id"])
        issues = []
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            issues.append("source_changed_during_posthoc_read")
        if sr != span["sample_rate"] or audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) != span["end_sample"] - span["start_sample"]:
            issues.append("source_pcm_shape_differs_from_asr")
        if after.st_mtime > started:
            issues.append("source_mtime_after_estimated_asr_start")
        if receipt_hash is not None and receipt_hash != file_hash:
            issues.append("independent_audio_receipt_hash_mismatch")
        if binding and (binding["file_sha256"] != file_hash or binding["pcm_sha256"] != pcm_hash):
            issues.append("asr_decode_time_binding_mismatch")
        results.append({"sample_id": label["sample_id"], "audio_path": str(source),
            "file_sha256_posthoc": file_hash, "pcm_sha256_posthoc": pcm_hash,
            "sample_count": len(audio), "sample_rate": sr, "source_mtime_unix": after.st_mtime,
            "mtime_before_asr_start": after.st_mtime <= started,
            "asr_decode_time_hash_available": binding is not None,
            "independent_audio_receipt": str(receipt_path) if receipt else None,
            "independent_audio_receipt_sha256": receipt_hash,
            "independent_audio_receipt_hash_matches": receipt_hash == file_hash if receipt_hash else None,
            "independent_audio_receipt_predates_asr": receipt_path.stat().st_mtime <= started if receipt else None,
            "issues": issues, "status": "posthoc_source_check_pass" if not issues else "source_review_required"})
    return {"original_clip_id": group["original_clip_id"], "checked_unix": time.time(),
        "asr_evidence": str(path), "asr_evidence_sha256": hashlib.sha256(payload).hexdigest(),
        "asr_start_unix": started, "asr_start_is_estimated_from_result_mtime": "asr_started_unix" not in group,
        "status": "posthoc_source_check_pass" if all(not r["issues"] for r in results) else "source_review_required",
        "segments": results,
        "limitation": "Posthoc hashes do not prove bytes observed by legacy ASR. Source mtime and independently available audio receipts are supporting provenance only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--audio-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow-pid", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "groups").mkdir(exist_ok=True)
    run = json.loads((args.evidence_dir / "run.json").read_text())
    wanted = set(run["group_ids"])
    rows = {r["sample_id"]: r for r in read_jsonl(args.manifest) if r["original_clip_id"] in wanted}
    process_identity = process_start(args.follow_pid) if args.follow_pid else None
    seen = set()
    while True:
        for path in sorted((args.evidence_dir / "groups").glob("*.json")):
            if path in seen:
                continue
            group = json.loads(path.read_text())
            if group.get("original_clip_id") not in wanted or group.get("status") != "complete":
                continue
            destination = args.output / "groups" / (digest(group["original_clip_id"])[:24] + ".json")
            if not destination.exists() or args.refresh:
                result = audit_group(path, rows, args.audio_receipts)
                atomic_json(destination, result)
            seen.add(path)
        audits = [json.loads(p.read_text()) for p in (args.output / "groups").glob("*.json")]
        audits = [r for r in audits if r["original_clip_id"] in wanted]
        segments = [s for r in audits for s in r["segments"]]
        summary = {"originals_requested": len(wanted), "originals_checked": len(audits),
            "segments_checked": len(segments), "issues": dict(collections.Counter(issue for s in segments for issue in s["issues"])),
            "independent_audio_receipt_available": sum(s["independent_audio_receipt"] is not None for s in segments),
            "independent_audio_receipt_hash_matches": sum(s["independent_audio_receipt_hash_matches"] is True for s in segments),
            "asr_decode_time_hash_available": sum(s["asr_decode_time_hash_available"] for s in segments),
            "all_requested_originals_checked": {r["original_clip_id"] for r in audits} == wanted,
            "hashes_are_posthoc_not_legacy_asr_decode_time": True,
            "updated_unix": time.time()}
        atomic_json(args.output / "summary.json", summary)
        atomic_jsonl(args.output / "segments.jsonl", segments)
        print(json.dumps(summary), flush=True)
        if process_identity is None or process_start(args.follow_pid) != process_identity:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
