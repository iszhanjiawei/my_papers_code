"""Select an intermediate 40 Hz wave from append-only completed AV reports.

This only writes a fresh source subset and its selection evidence. It neither
checks every cache file nor certifies full completion; the cache worker and
the final full-source pass must still validate the actual arrays and receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


PROTOCOL = "29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f"
IDENTITY = ("relative_stem", "original_clip_id")


def snapshot_json(path):
    payload = path.read_bytes()
    return json.loads(payload), {"path": str(path.resolve()), "size_bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()}


def identity(row):
    sample = row.get("sample_id")
    fields = tuple(row.get(key) for key in IDENTITY)
    if not isinstance(sample, str) or not sample or any(not isinstance(x, str) or not x for x in fields):
        raise ValueError(f"Missing source identity: {sample!r}")
    return sample, fields


def report_processes(state):
    stages = list(state["stages"].values())
    for attempts in state.get("stage_attempt_history", {}).values():
        stages.extend(attempts)
    reports, skipped_legacy = {}, 0
    for stage in stages:
        for process in stage.get("processes", []):
            command = process.get("command", [])
            if not any(Path(arg).name == "lrs3_media_worker.py" for arg in command):
                continue
            media_stage = command[command.index("--stage") + 1] if "--stage" in command else "all"
            if media_stage != "all":
                continue
            if "report" not in process:
                skipped_legacy += 1
                continue
            path = Path(process["report"])
            reports[str(path)] = process
    return reports, skipped_legacy


def read_completed_reports(root, state):
    reports, skipped_legacy = report_processes(state)
    identities, ready, evidence, missing = {}, set(), [], []
    statuses = Counter()
    for name, process in sorted(reports.items()):
        path = Path(name)
        if not path.resolve().is_relative_to((root / "reports").resolve()):
            raise ValueError(f"Media report is outside the cache reports directory: {path}")
        try:
            with path.open("rb") as handle:
                observed_size = os.fstat(handle.fileno()).st_size
                payload = handle.read(observed_size)
                if len(payload) != observed_size or os.fstat(handle.fileno()).st_size < observed_size:
                    raise RuntimeError(f"Append-only media report was truncated: {path}")
        except FileNotFoundError:
            if "exit_code" in process:
                raise
            missing.append(str(path))
            continue
        prefix_length = payload.rfind(b"\n") + 1
        prefix = payload[:prefix_length]
        count = ok_count = 0
        for count, line in enumerate(prefix.splitlines(), 1):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Malformed complete media report line {count}: {path}") from error
            if row.get("stage") != "all":
                raise ValueError(f"Non-all record in all-stage media report: {path}:{count}")
            sample, fields = identity(row)
            if identities.setdefault(sample, fields) != fields:
                raise ValueError(f"Conflicting media report identity: {sample}")
            statuses[row.get("status", "missing")] += 1
            if row.get("status") == "ok" and row.get("protocol_sha256") == PROTOCOL:
                ready.add(sample)
                ok_count += 1
        evidence.append({"path": str(path.resolve()), "read_prefix_bytes": prefix_length,
                         "read_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                         "observed_size_bytes": observed_size, "ignored_incomplete_tail_bytes": observed_size - prefix_length,
                         "complete_records": count, "matching_ok_records": ok_count})
    return identities, ready, evidence, {"record_status_counts": dict(statuses),
        "skipped_legacy_processes_without_report": skipped_legacy, "reports_not_created_yet": missing}


def read_exclusions(paths):
    identities, evidence = {}, []
    for path in paths:
        digest, count, own = hashlib.sha256(), 0, set()
        with path.open("rb") as handle:
            for count, line in enumerate(handle, 1):
                digest.update(line)
                sample, fields = identity(json.loads(line))
                if sample in own:
                    raise ValueError(f"Duplicate sample in exclusion manifest: {path}: {sample}")
                own.add(sample)
                if identities.setdefault(sample, fields) != fields:
                    raise ValueError(f"Conflicting excluded source identity: {sample}")
        evidence.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "count": count})
    return identities, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--selection-report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root.resolve(strict=True)
    outputs = (args.output_manifest.absolute(), args.selection_report.absolute())
    if outputs[0] == outputs[1] or any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("Output manifest and selection report must be distinct fresh paths")
    contract, contract_ref = snapshot_json(root / "reports/source_contract.json")
    state, state_ref = snapshot_json(root / "reports/media_full_pipeline/state.json")
    expected_hash = contract["source_inventory_sha256"]
    canonical = root / "manifests/source_inventory.jsonl"
    if state["manifest_sha256"] != expected_hash or Path(state["manifest"]).resolve() != canonical:
        raise ValueError("Media pipeline state does not reference the canonical source contract")
    reported, ready, report_refs, report_summary = read_completed_reports(root, state)
    excluded, exclusion_refs = read_exclusions(args.exclude_manifest)
    selected = ready - excluded.keys()
    seen, originals, selected_counts = set(), set(), Counter()
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    temporary_paths = []
    try:
        for destination in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            os.close(descriptor)
            temporary_paths.append(Path(name))
        selected_count = source_bytes = output_bytes = 0
        with canonical.open("rb") as source, temporary_paths[0].open("wb") as output:
            before = os.fstat(source.fileno())
            for line in source:
                source_hash.update(line)
                source_bytes += len(line)
                row = json.loads(line)
                sample, fields = identity(row)
                if sample in seen:
                    raise ValueError(f"Duplicate canonical source sample ID: {sample}")
                seen.add(sample)
                originals.add(row["original_clip_id"])
                for label, mapping in (("media report", reported), ("exclusion manifest", excluded)):
                    if sample in mapping and mapping[sample] != fields:
                        raise ValueError(f"{label} identity differs from canonical inventory: {sample}")
                if sample in selected:
                    output.write(line)
                    output_hash.update(line)
                    output_bytes += len(line)
                    selected_count += 1
                    selected_counts[row["split"]] += 1
            after = os.fstat(source.fileno())
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or source_hash.hexdigest() != expected_hash
                or len(seen) != contract["source_media_pairs"] or len(originals) != contract["original_clip_count"]):
            raise ValueError("Canonical source inventory differs from the pinned source contract")
        if reported.keys() - seen or excluded.keys() - seen or selected_count != len(selected):
            raise ValueError("Media reports or exclusion manifests contain IDs outside the canonical inventory")
        if snapshot_json(Path(contract_ref["path"]))[1] != contract_ref:
            raise ValueError("Source contract changed during subset selection")
        report = {"schema_version": 1, "created_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
            "scope": "Intermediate completed-AV subset only; worker validation and a final full-source pass remain required.",
            "full_completion_certified": False, "media_protocol_sha256": PROTOCOL,
            "source_contract": contract_ref, "media_pipeline_state_snapshot": state_ref,
            "source_manifest": {"path": str(canonical), "sha256": source_hash.hexdigest(), "size_bytes": source_bytes},
            "requested_source_count": len(seen), "reported_ready_unique_count": len(ready),
            "excluded_unique_count": len(excluded), "already_scheduled_ready_count": len(ready & excluded.keys()),
            "ready_subset_count": selected_count, "subset_split_counts": dict(selected_counts),
            "subset_manifest": str(outputs[0]), "manifest_sha256": output_hash.hexdigest(),
            "manifest_size_bytes": output_bytes, "source_report_prefixes": report_refs,
            "excluded_manifests": exclusion_refs, **report_summary}
        temporary_paths[1].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        # Hard-link publication fails if another writer created either output;
        # no existing manifest or global completion receipt is overwritten.
        os.link(temporary_paths[0], outputs[0])
        os.link(temporary_paths[1], outputs[1])
        print(json.dumps({key: report[key] for key in ("scope", "ready_subset_count", "subset_split_counts",
              "manifest_sha256", "elapsed_seconds", "subset_manifest")}, ensure_ascii=False))
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
