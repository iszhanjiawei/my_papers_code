"""Immutable, explicit quality exclusions for an audited full LRS3 media run.

This module never removes guards or changes media. Only the separate sealing
CLI may terminate a failed writer after an operator supplies an exact list.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


TERMINAL = "reports/media_full_pipeline/quality_terminal.json"
EXCLUSIONS = "reports/media_full_pipeline/media_quality_exclusions.json"
POLICY = "lrs3_media_quality_exclusions_v1"
PROTOCOL = "29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f"
QUALITY_ERRORS = {
    "no_valid_landmarks": ("No valid landmark detections; full-face resize is prohibited",),
    "landmark_coverage_insufficient": ("Landmark coverage requires review:",),
    "landmark_geometry_invalid": ("Re-detected landmarks leave the source geometry",
                                   "Landmarks outside the 224px source geometry"),
    "mouth_geometry_invalid": ("Mouth ROI outside canonical canvas:", "Invalid landmark alignment transform",
                               "No alignment transform; resize fallback is prohibited"),
}


def matches_reason(error, reason):
    return isinstance(error, str) and any(error.startswith(prefix + message)
        for message in QUALITY_ERRORS.get(reason, ())
        for prefix in ('', 'ValueError: ', "ValueError('", 'ValueError("'))


def recovery_chain(root, state, sample_ids):
    """Read latest officially registered FAN and media retry results separately."""
    fan, retry, paths = {}, {}, []
    for name in ('full_media_landmark_repair', 'full_media_retry', 'full_media_retry_long'):
        stage = state.get('stages', {}).get(name)
        if not stage or stage.get('finished') is not True:
            raise ValueError(f'Latest media recovery stage is unfinished: {name}')
        target = fan if name == 'full_media_landmark_repair' else retry
        for item in stage.get('processes', []):
            if item.get('exit_code') not in (0, 1):
                raise ValueError('Media recovery process did not finish normally')
            path = Path(item['report']).resolve(strict=True)
            if not path.is_relative_to((root / 'reports').resolve()):
                raise ValueError('Media recovery report is outside this cache')
            if str(path) not in stage.get('reports', []):
                raise ValueError('Recovery process report is not registered by its latest stage')
            paths.append(path)
            for row in rows(path):
                if row['sample_id'] in sample_ids:
                    ref = {'path': str(path), 'record_sha256': row_hash(row),
                           'status': row.get('status'), 'error': row.get('error')}
                    target[row['sample_id']] = (row, ref)
    if fan.keys() != sample_ids or retry.keys() != sample_ids:
        raise ValueError('Every final exclusion requires its latest formal FAN and media retry result')
    return fan, retry, paths


def check_recovery_chain(fan, retry, *, reason, relative_stem, source_video_sha256):
    if fan.get('relative_stem') != relative_stem or retry.get('relative_stem') != relative_stem:
        raise ValueError('Media recovery chain differs from its canonical identity')
    if retry.get('status') != 'failed' or not matches_reason(retry.get('error'), reason):
        raise ValueError('Latest media retry failure is not the explicitly reviewed quality rejection')
    if fan.get('status') == 'failed':
        allowed_fan = ('no_valid_landmarks', 'landmark_coverage_insufficient', 'landmark_geometry_invalid')
        if not any(matches_reason(fan.get('error'), code) for code in allowed_fan):
            raise ValueError('Latest formal FAN failure is IO/OOM/unknown, not an allowed quality rejection')
    elif fan.get('status') == 'ok':
        if (fan.get('source_sha256') != source_video_sha256
                or reason not in {'landmark_geometry_invalid', 'mouth_geometry_invalid'}):
            raise ValueError('Successful formal FAN permits only a source-bound final crop geometry rejection')
    else:
        raise ValueError('Latest formal FAN did not complete with an auditable result')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    fingerprint = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Quality evidence changed while reading: {path}")
    return {"path": str(path), "sha256": fingerprint, "size_bytes": after.st_size}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scope(terminal):
    if terminal is None:
        return None
    return {"policy": POLICY, "terminal_path": TERMINAL, "terminal_sha256": terminal["_sha256"],
            "source_pairs": terminal["requested"], "verified_media_pairs": terminal["verified"],
            "quality_excluded_pairs": terminal["excluded_count"], "all_source_media_caches_ready": False}


def evidence_paths(terminal):
    if terminal is None:
        return []
    return [Path(terminal["_path"])] + [Path(ref["path"]) for ref in terminal["evidence_files"]]


def excluded_map(terminal):
    return {} if terminal is None else {row["sample_id"]: row for row in terminal["excluded"]}


def load_terminal(root, *, expected_sha256=None, source_sha256=None):
    """Validate a fixed closure and all its evidence, never mutable summaries.

    Sealing verifies the complete canonical proof and reviewed exact failure
    set. Every consumer rechecks that immutable proof's hashes and the per-ID
    exclusion schema; finalizer/loader additionally enforce membership.
    """
    root = Path(root).resolve(strict=True)
    from aligndit.lrs3_imported_source import CONTRACT, validate_imported_source
    if (root / CONTRACT).exists() or (root / CONTRACT).is_symlink():
        imported = validate_imported_source(root)
        terminal = imported['terminal']
        if (expected_sha256 is not None and terminal['_sha256'] != expected_sha256
                or source_sha256 is not None and terminal['source_manifest_sha256'] != source_sha256):
            raise ValueError('Imported media quality closure differs from explicit source/terminal pins')
        terminal['_path'] = str(root / TERMINAL)
        terminal['_imported_source'] = {key: value for key, value in imported.items() if key != 'terminal'}
        return terminal
    path = root / TERMINAL
    if not path.exists():
        if expected_sha256 is not None or (root / EXCLUSIONS).exists():
            raise ValueError("Media quality exclusion publication is absent or incomplete")
        return None
    if path.is_symlink():
        raise ValueError("Media quality terminal must be a regular file")
    payload = path.read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("Media quality terminal hash differs from its explicit pin")
    terminal = json.loads(payload)
    if (terminal.get("schema_version") != 1 or terminal.get("policy") != POLICY
            or terminal.get("status") != "closed_with_quality_exclusions"
            or terminal.get("processing_complete") is not True or terminal.get("media_complete") is not False
            or terminal.get("all_source_ids_accounted") is not True or terminal.get("protocol_sha256") != PROTOCOL):
        raise ValueError("Invalid media quality terminal semantics")
    for key in ("requested", "verified", "excluded_count", "original_ids"):
        if type(terminal.get(key)) is not int or terminal[key] <= 0:
            raise ValueError(f"Invalid media quality terminal count: {key}")
    if terminal["verified"] + terminal["excluded_count"] != terminal["requested"]:
        raise ValueError("Media quality terminal does not account for the complete source")
    refs = terminal.get("evidence_files", [])
    by_path = {}
    for ref in refs:
        current = snapshot(ref["path"])
        if current != ref or ref["path"] in by_path:
            raise ValueError(f"Changed or duplicate media quality evidence: {ref['path']}")
        by_path[ref["path"]] = ref
    canonical = root / "manifests/source_inventory.jsonl"
    contract_path = root / "reports/source_contract.json"
    required = [canonical, contract_path, root / EXCLUSIONS,
                root / "reports/media_full_pipeline/state.json", root / "reports/media_full_pipeline/summary.json",
                root / "reports/media_full_pipeline/independent_verification.jsonl",
                root / "reports/media_full_pipeline/quality_guard_before.json"]
    if any(str(p.resolve()) not in by_path for p in required):
        raise ValueError("Media quality closure omits required canonical evidence")
    reviewed = terminal.get('reviewed_exclusions', {})
    if not reviewed.get('path') or by_path.get(reviewed['path']) != reviewed:
        raise ValueError('Media quality closure omits its explicit reviewed list pin')
    contract = read_json(contract_path)
    canonical_hash = by_path[str(canonical)]["sha256"]
    if (canonical_hash != terminal.get("source_manifest_sha256")
            or canonical_hash != contract.get("source_inventory_sha256")
            or source_sha256 is not None and canonical_hash != source_sha256
            or terminal.get("source_contract_sha256") != by_path[str(contract_path)]["sha256"]
            or terminal["requested"] != contract["source_media_pairs"]
            or terminal["original_ids"] != contract["original_clip_count"]):
        raise ValueError("Media quality closure differs from the canonical source contract")
    state = read_json(root / 'reports/media_full_pipeline/state.json')
    summary = read_json(root / 'reports/media_full_pipeline/summary.json')
    if (state.get('status') != 'finished_with_failures' or state.get('summary') != summary
            or state.get('manifest_sha256') != canonical_hash
            or summary.get('complete') is not False or summary.get('requested') != terminal['requested']
            or summary.get('verified') != terminal['verified'] or summary.get('failed') != terminal['excluded_count']
            or summary.get('original_ids') != terminal['original_ids'] or summary.get('protocol_sha256') != PROTOCOL):
        raise ValueError('Media quality closure must retain the actual incomplete media state')
    exclusion_file = read_json(root / EXCLUSIONS)
    excluded = exclusion_file.get("excluded")
    if (exclusion_file.get("policy") != POLICY or not isinstance(excluded, list)
            or len(excluded) != terminal["excluded_count"]):
        raise ValueError("Media quality exclusion list differs from its terminal")
    approved = read_json(reviewed['path'])
    if (not isinstance(approved, list) or len(approved) != len(excluded)
            or {r['sample_id']:r['reason'] for r in approved} != {r['sample_id']:r['reason'] for r in excluded}):
        raise ValueError('Media quality exclusion set differs from the explicit reviewed list')
    seen = set()
    fan, retry, report_paths = recovery_chain(root, state, {row['sample_id'] for row in excluded})
    if any(str(path) not in by_path for path in report_paths):
        raise ValueError('Media quality closure omits latest formal FAN/retry evidence')
    for row in excluded:
        key, reason = row.get("sample_id"), row.get("reason")
        proof = row.get("quality_evidence", {})
        if (not isinstance(key, str) or not key or key in seen or reason not in QUALITY_ERRORS
                or not isinstance(row.get("original_clip_id"), str) or not isinstance(row.get("relative_stem"), str)
                or proof.get("path") not in by_path or proof.get("status") != "failed"
                or not matches_reason(proof.get("error"), reason)):
            raise ValueError("Invalid/duplicate media quality exclusion or unauthorized failure reason")
        seen.add(key)
        fan_row, fan_ref = fan[key]
        retry_row, retry_ref = retry[key]
        if (proof != retry_ref or row.get('fan_evidence') != fan_ref or row.get('retry_evidence') != retry_ref
                or sha256(row["source_video_path"]) != row["source_video_sha256"]):
            raise ValueError("Quality exclusion differs from its actual failure/source evidence")
        check_recovery_chain(fan_row, retry_row, reason=reason, relative_stem=row['relative_stem'],
                             source_video_sha256=row['source_video_sha256'])
    proof_status, proof_excluded = {}, {}
    for r in rows(root / "reports/media_full_pipeline/independent_verification.jsonl"):
        proof_status[r['sample_id']] = r.get('status')
        if r['sample_id'] in seen:proof_excluded[r['sample_id']] = r
    failed = {key for key, status in proof_status.items() if status == "failed"}
    if (len(proof_status) != terminal["requested"] or failed != seen
            or sum(status == "ok" for status in proof_status.values()) != terminal["verified"]):
        raise ValueError("Terminal exclusions differ from complete independent media proof")
    for row in excluded:
        proof = proof_excluded[row["sample_id"]]
        if (row_hash(proof) != row["independent_verification_record_sha256"]
                or proof.get("relative_stem") != row["relative_stem"]
                or proof.get("original_clip_id") != row["original_clip_id"]):
            raise ValueError("Excluded ID differs from its final independent verification")
    terminal.update(_path=str(path), _sha256=fingerprint, excluded=excluded)
    return terminal


def build_terminal(root, reviewed_path):
    """Read-only proposal. Caller must hold media.lock and check no live writers."""
    root, reviewed_path = Path(root).resolve(strict=True), Path(reviewed_path).resolve(strict=True)
    state_path = root / "reports/media_full_pipeline/state.json"
    summary_path = state_path.parent / "summary.json"
    proof_path = state_path.parent / "independent_verification.jsonl"
    canonical = root / "manifests/source_inventory.jsonl"
    contract_path = root / "reports/source_contract.json"
    state, summary, contract = read_json(state_path), read_json(summary_path), read_json(contract_path)
    if (state.get("status") != "finished_with_failures" or state.get("summary") != summary
            or summary.get("complete") is not False or summary.get("failed", 0) <= 0
            or summary.get("protocol_sha256") != PROTOCOL):
        raise ValueError("Only a finished, fully verified media run with explicit failures may be closed")
    source_hash = sha256(canonical)
    if (state.get("manifest_sha256") != source_hash or source_hash != contract.get("source_inventory_sha256")
            or Path(state.get('manifest', '')).resolve() != canonical):
        raise ValueError("Media quality source hash mismatch")
    canonical_rows = {}
    for row in rows(canonical):
        if row["sample_id"] in canonical_rows:
            raise ValueError("Duplicate canonical sample ID")
        canonical_rows[row["sample_id"]] = row
    if (len(canonical_rows) != contract["source_media_pairs"]
            or len({r["original_clip_id"] for r in canonical_rows.values()}) != contract["original_clip_count"]
            or summary.get('original_ids') != contract['original_clip_count']):
        raise ValueError("Canonical source counts differ from contract")
    latest = {}
    for row in rows(proof_path):
        source = canonical_rows.get(row.get("sample_id"))
        if source is None or any(row.get(k) != source[k] for k in ("relative_stem", "original_clip_id", "video_frames")):
            raise ValueError("Independent media proof contains another source identity")
        if row.get("status") not in {"ok", "failed"}:
            raise ValueError("Independent media proof has an unknown status")
        latest[row["sample_id"]] = row
    failed = {key for key, row in latest.items() if row["status"] == "failed"}
    if (latest.keys() != canonical_rows.keys() or len(latest) != summary.get("requested")
            or len(failed) != summary["failed"] or len(latest) - len(failed) != summary.get("verified")):
        raise ValueError("Independent media proof does not account for every canonical source ID")
    reviewed = read_json(reviewed_path)
    if not isinstance(reviewed, list):
        raise ValueError("Reviewed exclusions must be a JSON array of sample_id/reason records")
    approved = {row["sample_id"]: row for row in reviewed}
    if len(approved) != len(reviewed) or approved.keys() != failed:
        raise ValueError("Reviewed exclusions must equal the exact final independent-verification failure ID set")
    fan, retry, report_paths = recovery_chain(root, state, failed)
    excluded = []
    for key in sorted(failed):
        source = canonical_rows[key]
        if source.get("width") != 224:
            raise ValueError("This policy only permits failed 224-pixel face-to-mouth quality repairs")
        source_video_sha = sha256(source['video_path'])
        check_recovery_chain(fan[key][0], retry[key][0], reason=approved[key].get('reason'),
                             relative_stem=source['relative_stem'], source_video_sha256=source_video_sha)
        excluded.append({"sample_id": key, "relative_stem": source["relative_stem"],
            "original_clip_id": source["original_clip_id"], "reason": approved[key]["reason"],
            "review_note": approved[key].get("review_note", ""), "source_video_path": source["video_path"],
            "source_video_sha256": source_video_sha, "quality_evidence": retry[key][1],
            'fan_evidence': fan[key][1], 'retry_evidence': retry[key][1],
            "independent_verification_record_sha256": row_hash(latest[key]), "independent_verification_error": latest[key].get("error")})
    paths = [canonical, contract_path, state_path, summary_path, proof_path, reviewed_path] + report_paths
    refs = [snapshot(path) for path in dict.fromkeys(paths)]
    proposal = {"schema_version": 1, "policy": POLICY, "status": "closed_with_quality_exclusions",
        "processing_complete": True, "media_complete": False, "all_source_ids_accounted": True,
        "requested": len(latest), "verified": len(latest) - len(failed), "excluded_count": len(failed),
        "original_ids": contract["original_clip_count"], "protocol_sha256": PROTOCOL,
        "source_manifest_sha256": source_hash, "source_contract_sha256": sha256(contract_path),
        "reviewed_exclusions": snapshot(reviewed_path), "evidence_files": refs,
        "write_guard_meaning": "Removal means media writing terminated, not that every source has a usable cache",
        "source_and_failed_cache_files_deleted": False}
    return proposal, {"schema_version": 1, "policy": POLICY, "excluded": excluded}
