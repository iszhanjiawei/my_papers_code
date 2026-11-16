#!/usr/bin/env python3
"""Follow ASR group receipts and run resumable, source-only MFA batches.

Run this supervisor in its own setsid session. It never feeds recovered labels
back into ASR and never changes the frozen pilot or primary acceptance files.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, read_jsonl

PROTOCOL = "mfa_source_boundary_consensus_batches_v1"


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (FileNotFoundError, ProcessLookupError):
        return None


class Coordinator:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "receipts").mkdir(exist_ok=True)
        (self.output / "batches").mkdir(exist_ok=True)
        self.cache = {}
        self.group_paths = {}
        self.last_status = 0
        self.phase = "waiting_for_asr_groups"
        self.mfa_pid = None
        self.stop_requested = False
        self.asr_start = process_start(args.asr_pid)
        self.run_info = json.loads((args.evidence_dir / "run.json").read_text())
        self.requested_ids = set(self.run_info["group_ids"])
        if len(self.requested_ids) != self.run_info["groups_requested"]:
            raise ValueError("ASR requested group IDs/count disagree")

    def asr_alive(self):
        return self.asr_start is not None and process_start(self.args.asr_pid) == self.asr_start

    def groups(self):
        paths = set((self.args.evidence_dir / "groups").glob("*.json"))
        self.cache = {p: entry for p, entry in self.cache.items() if p in paths}
        for path in sorted(paths):
            stamp = path.stat().st_mtime_ns
            if path not in self.cache or self.cache[path][0] != stamp:
                payload = path.read_bytes()
                self.cache[path] = (stamp, json.loads(payload), hashlib.sha256(payload).hexdigest())
        selected = {path: entry for path, entry in self.cache.items()
                    if entry[1].get("original_clip_id", entry[1].get("id")) in self.requested_ids}
        rows = [entry[1] for entry in selected.values()]
        self.group_paths = {entry[1].get("original_clip_id", entry[1].get("id")): path for path, entry in selected.items()}
        self.group_hashes = {entry[1].get("original_clip_id", entry[1].get("id")): entry[2] for entry in selected.values()}
        if len(self.group_paths) != len(rows):
            raise ValueError("Duplicate ASR group IDs")
        return rows

    def receipts(self):
        valid = {}
        for path in (self.output / "receipts").glob("*.json"):
            receipt = json.loads(path.read_text())
            original_id = receipt["original_clip_id"]
            evidence = self.group_paths.get(original_id)
            if (evidence is not None and receipt.get("source_evidence_sha256") == self.group_hashes[original_id]
                    and receipt["source_evidence_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()):
                valid[original_id] = receipt
        return valid

    def status(self, force=False):
        if not force and time.time() - self.last_status < 20:
            return
        groups = self.groups()
        receipts = self.receipts()
        accepted = sum(label["accepted"] for group in groups for label in group.get("labels", []))
        labels = sum(len(group.get("labels", [])) for group in groups)
        mfa_extra = sum(r.get("additional_accepted", 0) for r in receipts.values())
        report = {"protocol": PROTOCOL, "updated_unix": time.time(), "phase": self.phase,
            "supervisor_pid": os.getpid(), "asr_pid": self.args.asr_pid, "asr_alive": self.asr_alive(),
            "mfa_pid": self.mfa_pid, "asr_groups_complete": sum(g.get("status") == "complete" for g in groups),
            "asr_groups_error": sum(g.get("status") == "error" for g in groups),
            "asr_labels_processed": labels, "primary_source_grounded_accepted": accepted,
            "mfa_groups_processed": sum(r["status"] == "mfa_processed" for r in receipts.values()),
            "groups_without_mfa_needed": sum(r["status"] == "no_review_needed" for r in receipts.values()),
            "mfa_groups_failed": sum(r["status"] == "mfa_failed" for r in receipts.values()),
            "additional_boundary_consensus_accepted": mfa_extra, "remaining_review_segments_so_far": labels - accepted - mfa_extra}
        atomic_json(self.output / "status.json", report)
        self.last_status = time.time()
        print(json.dumps(report), flush=True)

    def receipt(self, original_id, value):
        source = self.group_paths[original_id]
        current_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if current_hash != self.group_hashes[original_id]:
            raise ValueError(f"ASR evidence changed while recording MFA outcome: {original_id}")
        value.update({"original_clip_id": original_id, "protocol": PROTOCOL,
                      "source_evidence": str(source), "source_evidence_sha256": current_hash})
        atomic_json(self.output / "receipts" / (digest(original_id)[:24] + ".json"), value)

    def run_command(self, command, log, phase, env=None):
        self.phase = phase
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": command, "start_unix": time.time()}) + "\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                       pass_fds=(self.args.lock_fd,))
            self.mfa_pid = process.pid
            self.status(force=True)
            while process.poll() is None:
                self.status()
                time.sleep(5)
            self.mfa_pid = None
            if process.returncode:
                raise RuntimeError(f"{phase} failed with exit code {process.returncode}; see {log}")

    def batch(self, original_ids):
        evidence_hashes = {original_id: hashlib.sha256(self.group_paths[original_id].read_bytes()).hexdigest() for original_id in original_ids}
        previous = self.receipts()
        attempts = {original_id: previous.get(original_id, {}).get("attempts", 0) + 1 for original_id in original_ids}
        batch_name = "batch_" + digest({"ids": original_ids, "evidence": evidence_hashes, "attempts": attempts})[:16]
        path = self.output / "batches" / batch_name
        path.mkdir(exist_ok=True)
        atomic_json(path / "group_ids.json", original_ids)
        atomic_json(path / "evidence_hashes.json", evidence_hashes)
        helper = Path(__file__).with_name("lrs3_text_recovery_mfa.py")
        helper_command = [sys.executable, "-u", str(helper), "--evidence-dir", str(self.args.evidence_dir),
            "--manifest", str(self.args.manifest), "--output", str(path)]
        try:
            self.run_command(helper_command + ["--group-ids-file", str(path / "group_ids.json"), "--expected-evidence-file", str(path / "evidence_hashes.json"), "--review-only", "--source-only"], path / "prepare.log", "preparing_mfa_batch")
            env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
            env["PATH"] = str(self.args.mfa_python.parent) + os.pathsep + env.get("PATH", "")
            env["OMP_NUM_THREADS"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            command = [str(self.args.mfa_python), "-u", str(self.args.mfa_python.parent / "mfa"), "align",
                str(path / "corpus"), str(self.args.dictionary), str(self.args.acoustic_model), str(path / "aligned"),
                "--temporary_directory", str(path / "tmp"), "--num_jobs", str(self.args.mfa_jobs),
                "--no_use_postgres", "--output_format", "json", "--single_speaker"]
            self.run_command(command, path / "mfa.log", "aligning_mfa_batch", env=env)
            self.run_command(helper_command + ["--analyze"], path / "analyze.log", "analyzing_mfa_batch")
            additions = read_jsonl(path / "labels.mfa_additional.accepted.jsonl")
            counts = collections.Counter(r["original_clip_id"] for r in additions)
            diagnostics = json.loads((path / "diagnostic_summary.json").read_text())
            for original_id in original_ids:
                if hashlib.sha256(self.group_paths[original_id].read_bytes()).hexdigest() != evidence_hashes[original_id]:
                    raise ValueError(f"ASR evidence changed during MFA: {original_id}")
            atomic_json(path / "done.json", {"group_ids": original_ids, "evidence_hashes": evidence_hashes, "summary": diagnostics, "completed_unix": time.time()})
            for original_id in original_ids:
                self.receipt(original_id, {"status": "mfa_processed", "batch": str(path), "additional_accepted": counts[original_id], "attempts": attempts[original_id]})
        except Exception as exc:
            atomic_json(path / "failed.json", {"group_ids": original_ids, "error": str(exc), "failed_unix": time.time()})
            for original_id in original_ids:
                self.receipt(original_id, {"status": "mfa_failed", "batch": str(path), "error": str(exc), "attempts": attempts[original_id]})
            print(json.dumps({"batch_error": str(exc), "batch": str(path)}), flush=True)
        self.phase = "waiting_for_asr_groups"
        self.aggregate()
        self.status(force=True)

    def aggregate(self):
        groups = self.groups()
        receipts = self.receipts()
        extra = {}
        batches = {r["batch"] for r in receipts.values() if r["status"] == "mfa_processed"}
        for batch in sorted(batches):
            done = Path(batch) / "done.json"
            if not done.exists():
                continue
            for row in read_jsonl(done.parent / "labels.mfa_additional.accepted.jsonl"):
                receipt = receipts.get(row["original_clip_id"], {})
                if receipt.get("status") != "mfa_processed" or receipt.get("batch") != batch:
                    continue
                if row["sample_id"] in extra:
                    raise RuntimeError(f"Duplicate MFA acceptance receipt: {row['sample_id']}")
                extra[row["sample_id"]] = row
        accepted, review, pseudo, originals = [], [], [], []
        for group in sorted(groups, key=lambda g: g.get("original_clip_id", g.get("id", ""))):
            original_id = group.get("original_clip_id", group.get("id"))
            if group.get("status") != "complete":
                originals.append({"original_clip_id": original_id, "status": "asr_error", "error": group.get("error")})
                continue
            primary_count, extra_count, review_count = 0, 0, 0
            for row in group["labels"]:
                if row["accepted"]:
                    if row["sample_id"] in extra:
                        raise RuntimeError("Primary and additional label sets must be disjoint")
                    accepted.append(row)
                    primary_count += 1
                elif row["sample_id"] in extra:
                    accepted.append(extra[row["sample_id"]])
                    extra_count += 1
                else:
                    review.append(row)
                    review_count += 1
                    pseudo.append({"sample_id": row["sample_id"], "original_clip_id": original_id,
                        "text": row["asr_candidate_text"], "quality": "asr_only_unverified", "training_eligible": False,
                        "source": "Whisper_large_v3_independent_clip_decode", "review_reasons": row["reasons"]})
            originals.append({"original_clip_id": original_id, "status": "all_segments_source_grounded" if not review_count else "review_remaining",
                "processed_segments": len(group["labels"]), "duration_seconds": group["duration"],
                "primary_source_grounded": primary_count, "additional_mfa_boundary_consensus": extra_count,
                "review_segments": review_count, "mfa_status": receipts.get(original_id, {}).get("status", "pending"),
                "source_name": group["source_comparison"]["source_name"], "source_asr_wer": group["source_comparison"]["wer"]})
        atomic_jsonl(self.output / "labels.source_grounded.accepted.jsonl", accepted)
        atomic_jsonl(self.output / "labels.review.jsonl", review)
        atomic_jsonl(self.output / "labels.asr_only.unverified.jsonl", pseudo)
        atomic_jsonl(self.output / "coverage_by_original.jsonl", originals)
        run = self.run_info
        seen_ids = {g.get("original_clip_id", g.get("id")) for g in groups}
        asr_complete = seen_ids == self.requested_ids and all(g.get("status") == "complete" for g in groups)
        processing_complete = asr_complete and set(receipts) == self.requested_ids and all(r["status"] != "mfa_failed" for r in receipts.values())
        summary = {"protocol": PROTOCOL, "original_groups_requested": run.get("groups_requested"),
            "original_groups_processed": len(groups), "accepted_segments": len(accepted), "review_segments": len(review),
            "accepted_hours": sum(r["duration"] for r in accepted) / 3600,
            "review_hours": sum(r["duration"] for r in review) / 3600,
            "additional_boundary_consensus_segments": len(extra),
            "processing_complete": processing_complete,
            "asr_groups_missing": len(self.requested_ids - seen_ids),
            "asr_groups_error": sum(g.get("status") == "error" for g in groups),
            "review_reasons": dict(collections.Counter(reason for r in review for reason in r["reasons"])),
            "mfa_failed_groups": sum(r["status"] == "mfa_failed" for r in receipts.values()),
            "all_text_verified": processing_complete and not review,
            "asr_only_candidates_are_training_labels": False,
            "limitations": ["Automatic source/ASR consensus remains distinct from original/manual transcription.",
                            "ASR-only candidates require review; a completed processing pipeline does not imply all text was recovered."]}
        atomic_json(self.output / "summary.json", summary)
        return summary

    def run(self):
        final_sweep = False
        while True:
            groups = self.groups()
            receipts = self.receipts()
            pending = []
            for group in groups:
                original_id = group.get("original_clip_id", group.get("id"))
                receipt = receipts.get(original_id, {})
                retry = receipt.get("status") == "mfa_failed" and receipt.get("attempts", 1) < self.args.max_attempts
                if group.get("status") != "complete" or (receipt and not retry):
                    continue
                if any(not row["accepted"] for row in group["labels"]):
                    pending.append(original_id)
                else:
                    self.receipt(original_id, {"status": "no_review_needed", "additional_accepted": 0})
            running = self.asr_alive()
            if self.stop_requested:
                break
            if len(pending) >= self.args.batch_size or (pending and not running):
                self.batch(sorted(pending)[:self.args.batch_size])
                continue
            self.status()
            if not running:
                if not final_sweep:
                    final_sweep = True
                    continue
                break
            time.sleep(10)
        summary = self.aggregate()
        self.phase = "stopped_between_batches" if self.stop_requested else ("complete" if summary["processing_complete"] else "incomplete")
        self.status(force=True)
        return summary["processing_complete"] or self.stop_requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-pid", type=int, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mfa-python", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--acoustic-model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mfa-jobs", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "coordinator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        args.lock_fd = lock.fileno()
        coordinator = Coordinator(args)
        def stop_after_batch(_signum, _frame):
            coordinator.stop_requested = True
        signal.signal(signal.SIGTERM, stop_after_batch)
        signal.signal(signal.SIGINT, stop_after_batch)
        if not coordinator.run():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
