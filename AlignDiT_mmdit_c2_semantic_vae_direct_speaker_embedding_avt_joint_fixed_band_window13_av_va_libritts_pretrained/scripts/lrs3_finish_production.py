"""Finish an existing LRS3 production run without starting model training.

Run under setsid, after the explicitly listed external video40 waves exist.
This coordinator schedules intermediate CPU waves, a mandatory canonical full
pass, immutable strict/pseudo publications, and both launcher validation runs.
It does not repair failed upstream jobs, remove their guards, or alter labels.
Children inherit finish.lock so a killed parent cannot leave duplicate writers.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from aligndit.lrs3_media_quality import TERMINAL, excluded_map, load_terminal, scope as media_quality_scope
from lrs3_text_recovery_export import REQUIRED_RELEASE_FILES


SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
PROTOCOL = "29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f"


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def alive(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ValueError, TypeError):
        return False


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid run name: {value!r}")
    return value


def cache_status(summary, stage, manifest, fingerprint, count, quality=None):
    """A complete flag alone is never enough to accept a cache coordinator."""
    if (summary.get("stage") != stage or summary.get("manifest_sha256") != fingerprint
            or Path(summary.get("source_manifest", "")).resolve() != manifest.resolve()
            or summary.get("requested") != count):
        raise ValueError("Cache coordinator has another source scope")
    if quality is not None:
        if stage != 'video40' or summary.get('media_quality_terminal_sha256') != quality['_sha256']:
            raise ValueError('Canonical video40 coordinator uses another media quality closure')
        if summary.get('complete') is True:
            raise ValueError('A quality exclusion cannot certify complete source caches')
        if summary.get('processing_complete') is True:
            if (summary.get('complete') is not False or summary.get('failed') != 0
                    or summary.get('succeeded') != count - quality['excluded_count']
                    or summary.get('quality_excluded') != quality['excluded_count']
                    or not summary.get('finished_unix')):
                raise ValueError('Quality-scoped video40 completion counts are inconsistent')
            return True
    if summary.get("complete") is True:
        if (summary.get("succeeded") != count or summary.get("failed") != 0
                or not summary.get("finished_unix") or summary.get('quality_excluded', 0) != 0
                or summary.get('media_quality_terminal_sha256') is not None):
            raise ValueError("Cache coordinator falsely reports completion")
        return True
    if summary.get("finished_unix") or not alive(summary.get("pid")):
        raise RuntimeError("Cache coordinator failed or stopped before completion; inspect its reports")
    return False


class Finisher:
    def __init__(self, args, lock_fd):
        self.args, self.lock_fd = args, lock_fd
        self.root = args.root.resolve(strict=True)
        self.text_release = (getattr(args, 'text_release', None) or self.root / 'text_recovery/release').resolve()
        self.run = self.root / "reports" / args.name
        self.run.mkdir(parents=True, exist_ok=True)
        self.path = self.run / "state.json"
        self.canonical = self.root / "manifests/source_inventory.jsonl"
        contract_path = self.root / "reports/source_contract.json"
        contract = read(contract_path)
        source_hash = digest(self.canonical)
        if source_hash != contract["source_inventory_sha256"]:
            raise ValueError("Canonical manifest differs from the source contract")
        self.count = contract["source_media_pairs"]
        self.original_count = contract["original_clip_count"]
        self.pins = {str(self.canonical): source_hash, str(contract_path): digest(contract_path)}
        self.source_hash = source_hash
        self.identities, self.missing_ids = {}, set()
        self.missing_count = 0
        originals = set()
        with self.canonical.open() as handle:
            for line in handle:
                row = json.loads(line)
                sample = row["sample_id"]
                if sample in self.identities:
                    raise ValueError("Duplicate canonical sample ID")
                self.identities[sample] = (row["relative_stem"], row["original_clip_id"])
                originals.add(row["original_clip_id"])
                if row["text_status"] == "missing":
                    self.missing_ids.add(row["original_clip_id"])
                    self.missing_count += 1
        if len(self.identities) != self.count or len(originals) != self.original_count:
            raise ValueError("Canonical source counts differ from the pinned contract")
        if digest(self.canonical) != source_hash or read(contract_path) != contract:
            raise ValueError("Source inputs changed during coordinator initialization")
        settings = {"existing_waves": args.existing_wave, "audio_run": args.audio_run,
                    "workers": args.workers, "finalizer_workers": args.finalizer_workers,
                    'text_release': str(self.text_release)}
        self.state = read(self.path) if self.path.exists() else {
            "schema_version": 1, "pins": self.pins, "settings": settings,
            "stages": {}, "waves": [], "selections": [], "last_selection_unix": 0}
        if 'text_release' not in self.state.get('settings', {}):
            self.state['settings']['text_release'] = str((self.root / 'text_recovery/release').resolve())
        if self.state.get("pins") != self.pins or self.state.get("settings") != settings:
            raise ValueError("Finisher inputs or run settings changed; use a different coordinator name")
        self.stamps = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in self.pins}
        self.state.update(pid=os.getpid(), sid=os.getsid(0), status="running")
        self.quality = None
        self.save()

    def save(self):
        self.state["heartbeat_unix"] = time.time()
        write(self.path, self.state)

    def check_pins(self, full=False):
        for path, expected in self.pins.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != self.stamps[path] or (full and digest(path) != expected):
                raise ValueError(f"Pinned production input changed: {path}")
        if self.quality is not None:
            if digest(self.root / TERMINAL) != self.quality['_sha256']:
                raise ValueError('Pinned media quality closure changed')
            if full:load_terminal(self.root, expected_sha256=self.quality['_sha256'], source_sha256=self.source_hash)

    def quiet(self, kinds):
        guards = {"audio": ["latents", "speaker_embeddings"], "media": ["video_25hz"],
                  "video40": ["video_40hz"], "publication": []}
        with ExitStack() as stack:
            for kind in kinds:
                path = self.root / f"state/locks/{kind}.lock"
                handle = stack.enter_context(path.open("a"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                for feature in guards[kind]:
                    marker = self.root / f"state/{feature}/WRITE_ACTIVE.json"
                    if marker.exists() or marker.is_symlink():
                        info = read(marker)
                        if not alive(info.get("pid")):
                            if kind == 'video40' and self.retire_failed_canonical_guard(marker, info):
                                continue
                            raise RuntimeError(f"Stale writer guard requires upstream repair: {marker}")
                        return False
        return True

    def retire_failed_canonical_guard(self, marker, info):
        """Authorize a new unique final pass, retaining the previous failed guard/evidence.

        Caller holds a shared video40 lock. No file is removed here; the next
        coordinator replaces its own guard only after acquiring its exclusive
        lock. Only an exact reviewed failure set from the old canonical pass
        can use this recovery path.
        """
        if self.quality is None:
            return False
        expected_run = self.root / 'reports' / (self.args.name + '_canonical_video40')
        if info.get('stage') != 'video40' or Path(info.get('run', '')).resolve() != expected_run:
            return False
        summary_path = expected_run / 'summary.json'
        summary = read(summary_path)
        if (summary.get('complete') is not False or not summary.get('finished_unix')
                or summary.get('manifest_sha256') != self.source_hash or summary.get('requested') != self.count
                or Path(summary.get('source_manifest', '')).resolve() != self.canonical
                or alive(summary.get('pid'))):
            raise ValueError('Old canonical writer lacks a finished failed canonical summary')
        latest, pins = {}, {str(marker): digest(marker), str(summary_path): digest(summary_path)}
        for attempt in summary.get('rounds', []):
            for worker in attempt.get('workers', []):
                if alive(worker.get('pid')):
                    raise RuntimeError('Old canonical worker is still active')
            for path in attempt.get('summaries', []):
                item = read(path)
                progress = Path(item['progress_path'])
                pins[str(path)], pins[str(progress)] = digest(path), digest(progress)
                with progress.open() as handle:
                    for line in handle:
                        result = json.loads(line)
                        if result['sample_id'] not in self.identities:
                            raise ValueError('Old canonical progress contains a noncanonical sample')
                        latest[result['sample_id']] = result['status']
        failed = {key for key, status in latest.items() if status == 'error'}
        if (latest.keys() != self.identities.keys() or failed != excluded_map(self.quality).keys()
                or any(status not in {'ok','error'} for status in latest.values())
                or summary.get('failed') != len(failed)
                or summary.get('succeeded') != self.count - len(failed)):
            raise ValueError('Old canonical failures are not exactly the reviewed media exclusions')
        record = {'terminal_sha256': self.quality['_sha256'], 'old_guard': info,
                  'old_evidence_pins': pins, 'action': 'new unique canonical quality pass; old evidence retained'}
        path = self.run / ('canonical_quality_transition_' + self.quality['_sha256'][:16] + '.json')
        if path.exists() and read(path) != record:
            raise ValueError('Old canonical failure evidence changed since transition review')
        if not path.exists():write(path, record)
        return True

    def manifest_scope(self, path):
        path = Path(path).resolve(strict=True)
        seen = set()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                sample = row["sample_id"]
                if (sample in seen or self.identities.get(sample) !=
                        (row["relative_stem"], row["original_clip_id"])):
                    raise ValueError(f"Wave contains duplicate or noncanonical identities: {path}")
                seen.add(sample)
        return {"manifest": str(path), "sha256": digest(path), "count": len(seen)}

    def cache_done(self, name, manifest, fingerprint, count, stage="video40", quality=None):
        path = self.root / "reports" / name / "summary.json"
        if not path.exists():
            return False
        return cache_status(read(path), stage, Path(manifest), fingerprint, count, quality)

    def audio_done(self):
        path = self.root / "reports" / self.args.audio_run / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Required upstream audio summary is absent: {path}")
        return self.cache_done(self.args.audio_run, self.canonical, self.source_hash, self.count, "audio")

    def media_done(self):
        path = self.root / "reports/media_full_pipeline/state.json"
        state = read(path)
        if state.get("manifest_sha256") != self.source_hash or Path(state["manifest"]).resolve() != self.canonical:
            raise ValueError("Media pipeline has another canonical source scope")
        if state.get('status') == 'finished_with_failures':
            quality = load_terminal(self.root, source_sha256=self.source_hash,
                                    expected_sha256=self.state.get('media_quality_terminal_sha256'))
            if quality is None:
                raise RuntimeError('Media failures require explicit reviewed quality closure or upstream repair')
            if any(self.identities.get(key) != (row['relative_stem'], row['original_clip_id'])
                   for key, row in excluded_map(quality).items()):
                raise ValueError('Media quality closure contains noncanonical identities')
            if not self.quiet(['media']):return False
            self.quality = quality
            self.state['media_quality_terminal_sha256'] = quality['_sha256']
            self.state['media_quality'] = media_quality_scope(quality)
            self.save()
            return True
        if (self.root / TERMINAL).exists() or self.state.get('media_quality_terminal_sha256'):
            raise ValueError('Media state changed after a fixed quality closure')
        if state.get("status") == "complete":
            summary = read(path.parent / "summary.json")
            if (state.get("summary") != summary or summary.get("complete") is not True
                    or summary.get("requested") != self.count or summary.get("verified") != self.count
                    or summary.get("failed") != 0 or summary.get("original_ids") != self.original_count
                    or summary.get("complete_original_ids") != self.original_count
                    or summary.get("protocol_sha256") != PROTOCOL):
                raise ValueError("Media completion lacks canonical independent-verification coverage")
            if not self.quiet(["media"]):
                return False
            proof = path.parent / "independent_verification.jsonl"
            fingerprint = digest(proof)
            if self.state.get("media_verification_sha256") != fingerprint:
                latest = {}
                with proof.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        sample = row["sample_id"]
                        if self.identities.get(sample) != (row["relative_stem"], row["original_clip_id"]):
                            raise ValueError("Independent verification returned a noncanonical identity")
                        latest[sample] = row["status"]
                if latest.keys() != self.identities.keys() or any(status != "ok" for status in latest.values()):
                    raise ValueError("Independent media verification is incomplete")
                self.state["media_verification_sha256"] = fingerprint
                self.save()
            return True
        if state.get("status") != "running" or not alive(state.get("coordinator_pid")):
            raise RuntimeError("Media pipeline failed/stopped; root must repair it before automatic finishing")
        return False

    def text_done(self):
        release = self.text_release
        path = release / "frozen_export.json"
        if not path.exists():
            # A live export may legitimately still be waiting for ASR/MFA/CTC.
            # No mutable summary is accepted as a substitute for this marker.
            if not (release / "summary.json").exists():
                raise FileNotFoundError("Text recovery summary is absent")
            with (release / "export.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                if not path.exists():
                    raise RuntimeError("Text exporter stopped without a frozen release; inspect its log")
            return self.text_done()
        frozen = read(path)
        summary = frozen.get("summary", {})
        categories = ("strict_segments", "pseudo_p1_segments", "pseudo_p2_candidate_segments", "excluded_segments")
        if (frozen.get("source_manifest_sha256") != self.source_hash
                or summary.get("processing_complete") is not True
                or summary.get("asr_and_requested_controls_complete") is not True
                or summary.get("pending_segments") != 0
                or summary.get("groups_complete") != len(self.missing_ids)
                or summary.get("groups_requested") != len(self.missing_ids)
                or summary.get("segments_requested") != self.missing_count
                or sum(summary.get(key, -1) for key in categories) != self.missing_count):
            raise ValueError("Text recovery freeze is incomplete or belongs to another source scope")
        required = REQUIRED_RELEASE_FILES
        if not isinstance(frozen.get('files'), dict) or not required <= frozen['files'].keys():
            raise ValueError('Frozen text export must pin every required publication file')
        for name in required:
            if frozen.get("files", {}).get(name) != digest(release / name):
                raise ValueError(f"Frozen recovery file changed: {name}")
        if read(release / "summary.json") != summary:
            raise ValueError("Frozen recovery summary differs from published summary")
        if (release / 'labels.pending.jsonl').read_bytes().strip():
            raise ValueError('Frozen recovery still contains pending labels')
        fingerprint = digest(path)
        if self.state.get("frozen_text_sha256", fingerprint) != fingerprint:
            raise ValueError("Previously observed immutable text freeze changed")
        self.state["frozen_text_sha256"] = fingerprint
        return True

    def command(self, key, command, completed):
        previous = self.state["stages"].get(key)
        if completed():
            self.state["stages"][key] = {**(previous or {}), "status": "complete",
                "completion_revalidated_unix": time.time(), "recovered_from_output": previous is None or previous.get("status") != "complete"}
            self.save()
            return
        if previous:
            raise RuntimeError(f"Stage {key} has no valid completed output after an earlier attempt; inspect/repair it manually")
        self.check_pins(full=True)
        log_path = self.run / f"{key}.log"
        stage = {"status": "starting", "command": command, "log": str(log_path), "started_unix": time.time()}
        self.state["stages"][key] = stage
        self.save()
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(PROJECT / "src"), PYTHONUNBUFFERED="1")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name, None)
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(self.lock_fd,))
            stage.update(status="running", pid=process.pid)
            self.save()
            while process.poll() is None:
                self.check_pins()
                self.save()
                time.sleep(min(10, self.args.poll_seconds))
        stage.update(returncode=process.returncode, finished_unix=time.time())
        if process.returncode != 0:
            stage["status"] = "failed"
            self.save()
            raise RuntimeError(f"Stage {key} failed with {process.returncode}; see {log_path}")
        if not completed():
            stage["status"] = "failed"
            self.save()
            raise RuntimeError(f"Stage {key} exited zero without complete evidence")
        stage["status"] = "complete"
        self.save()

    def wave(self, entry):
        if digest(entry["manifest"]) != entry["sha256"]:
            raise ValueError("Previously scheduled wave manifest changed")
        command = [sys.executable, "-u", str(SCRIPTS / "lrs3_cache_parallel.py"), "--root", str(self.root),
                   "--manifest", entry["manifest"], "--stage", "video40", "--gpus", ",".join(["cpu"] * self.args.workers),
                   "--name", entry["name"]]
        quality = self.quality if entry.get('canonical_quality') else None
        if quality:
            command += ['--media-quality-terminal', str(self.root / TERMINAL),
                        '--media-quality-terminal-sha256', quality['_sha256']]
        self.command(entry["name"], command,
                     lambda: self.cache_done(entry["name"], entry["manifest"], entry["sha256"], entry["count"], quality=quality))

    def select_wave(self, exclusions):
        index = len(self.state["selections"]) + 1
        name = f"{self.args.name}_wave{index:04d}"
        manifest = self.root / "manifests" / f"{name}.jsonl"
        report = self.run / f"{name}_selection.json"
        entry = {"name": name, "manifest": str(manifest), "report": str(report)}
        self.state["selections"].append(entry)
        self.save()
        def selected():
            if not manifest.exists() or not report.exists():
                return False
            result = read(report)
            scope = self.manifest_scope(manifest)
            if (result.get("source_manifest", {}).get("sha256") != self.source_hash
                    or result.get("manifest_sha256") != scope["sha256"]
                    or result.get("ready_subset_count") != scope["count"]):
                raise ValueError("Selection output failed its source/count/hash checks")
            entry.update(scope)
            return True
        command = [sys.executable, "-u", str(SCRIPTS / "lrs3_select_video40_ready.py"), "--root", str(self.root),
                   "--output-manifest", str(manifest), "--selection-report", str(report)]
        for path in exclusions:
            command += ["--exclude-manifest", path]
        self.command(name + "_selection", command, selected)
        self.state["last_selection_unix"] = time.time()
        if entry["count"] >= self.args.min_wave:
            self.state["waves"].append(entry)
            self.save()
            self.wave(entry)
        self.save()

    def publication_done(self, name):
        directory = self.root / "manifests" / name
        marker = directory / "cache_completion.json"
        if not marker.exists():
            return False
        receipt = read(marker)
        meta = read(directory / "inventory_meta.json")
        if meta.get('media_quality') != media_quality_scope(self.quality) or receipt.get('media_quality') != meta.get('media_quality'):
            raise ValueError('Publication media quality closure differs from the canonical final pass')
        expected = "source_grounded" if name == "full" else "source_plus_asr_pseudo"
        if (receipt.get("status") != "complete" or receipt.get("selection") != {"mode": "full"}
                or meta.get("source_manifest_sha256") != self.source_hash or meta.get("source_pairs") != self.count
                or meta.get("source_original_ids") != self.original_count or meta.get("text_policy") != expected
                or receipt.get("text_policy") != expected or receipt.get("count") != meta.get("training_ready_pairs")
                or meta.get("source_contract_sha256") != self.pins[str(self.root / "reports/source_contract.json")]):
            raise ValueError(f"Existing immutable publication {name} has another scope or incomplete metadata")
        for filename, ref in meta["manifests"].items():
            if digest(directory / filename) != ref["sha256"]:
                raise ValueError(f"Existing immutable publication changed: {name}/{filename}")
        if receipt.get("inventory_sha256") != meta["manifests"]["inventory.jsonl"]["sha256"]:
            raise ValueError("Publication completion marker differs from its inventory")
        return True

    def publish(self):
        labels = self.text_release
        for name in ("full", "full_pseudo"):
            command = [sys.executable, "-u", str(SCRIPTS / "lrs3_finalize_cache.py"), "--root", str(self.root),
                "--source-manifest", str(self.canonical), "--recovered-labels", str(labels / "labels.strict.accepted.jsonl"),
                "--name", name, "--mode", "full", "--workers", str(self.args.finalizer_workers)]
            if name == "full_pseudo":
                command += ["--recovered-labels", str(labels / "labels.pseudo_p1.accepted.jsonl"),
                            "--text-policy", "source_plus_asr_pseudo", "--strict-baseline", "manifests/full"]
            self.command("publish_" + name, command, lambda name=name: self.publication_done(name))
        for variant, publication in (("strict", "full"), ("pseudo", "full_pseudo")):
            output = self.root / "reports" / f"{publication}_training_launch"
            # A complete plan is the launcher's last artifact. Validate its pins
            # on resume; no command in this coordinator includes --start.
            command = [sys.executable, "-u", str(SCRIPTS / "launch_lrs3_training.py"), "--variant", variant,
                       "--cache-root", str(self.root), "--output-dir", str(output)]
            key = "dryrun_" + variant
            def launched():
                plan = output / "launch_plan.json"
                if not plan.exists():
                    return False
                value = read(plan)
                scope = value.get("scope", {})
                if scope.get('media_quality') != media_quality_scope(self.quality):
                    raise ValueError('Dry-run plan uses another media quality closure')
                if (value.get("variant") != variant or scope.get("source_inventory_sha256") != self.source_hash
                        or scope.get("source_pairs") != self.count or scope.get("source_original_ids") != self.original_count
                        or not value.get("artifacts")):
                    raise ValueError("Dry-run launch plan has another variant/source")
                for artifact in value["artifacts"]:
                    if digest(artifact["path"]) != artifact["sha256"]:
                        raise ValueError(f"Dry-run plan artifact changed: {artifact['path']}")
                return True
            self.command(key, command, launched)

    def run_all(self):
        external = []
        for name in self.args.existing_wave:
            path = self.root / "reports" / name / "summary.json"
            if not path.exists():
                raise FileNotFoundError(f"Explicit existing wave has no coordinator summary: {path}")
            summary = read(path)
            external.append({"name": name, **self.manifest_scope(Path(summary["source_manifest"]))})
        # A parent may stop after selection publication but before scheduling its
        # wave. Recover that completed selection without overwriting its files.
        scheduled = {entry["name"] for entry in self.state["waves"]}
        for entry in self.state["selections"]:
            if entry["name"] in scheduled:
                continue
            if not Path(entry["report"]).exists() or not Path(entry["manifest"]).exists():
                raise RuntimeError("Interrupted selection has no complete output; inspect the recorded attempt")
            result = read(entry["report"])
            actual = self.manifest_scope(Path(entry["manifest"]))
            if (result.get("source_manifest", {}).get("sha256") != self.source_hash
                    or result.get("manifest_sha256") != actual["sha256"]
                    or result.get("ready_subset_count") != actual["count"]):
                raise ValueError("Recovered selection failed count/hash/source checks")
            entry.update(actual)
            stage = self.state["stages"].setdefault(entry["name"] + "_selection", {})
            stage.update(status="complete", completion_revalidated_unix=time.time(), recovered_from_output=True)
            if actual["count"] >= self.args.min_wave:
                self.state["waves"].append(entry)
                scheduled.add(entry["name"])
        self.save()
        while True:
            self.check_pins()
            audio = self.audio_done()
            media = self.media_done()
            text_ready = self.text_done()
            pending = [w for w in external if not self.cache_done(w["name"], w["manifest"], w["sha256"], w["count"])]
            self.state["waiting"] = {"audio": not audio, "media": not media, "text": not text_ready,
                                     "external_waves": [w["name"] for w in pending]}
            self.save()
            if not pending and self.quiet(["video40"]):
                for entry in self.state["waves"]:
                    self.wave(entry)
                if media:
                    suffix = '_quality_' + self.quality['_sha256'][:16] if self.quality else ''
                    self.wave({"name": self.args.name + "_canonical_video40" + suffix, "manifest": str(self.canonical),
                               "sha256": self.source_hash, "count": self.count, 'canonical_quality': self.quality is not None})
                    if audio and text_ready and self.quiet(["audio", "media", "video40", "publication"]):
                        self.publish()
                        self.check_pins(full=True)
                        self.state.update(status="complete", finished_unix=time.time())
                        self.save()
                        return
                elif time.time() - self.state["last_selection_unix"] >= self.args.wave_seconds:
                    exclusions = [w["manifest"] for w in external + self.state["waves"]]
                    self.select_wave(exclusions)
            time.sleep(self.args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", type=safe_name, default="finish_production")
    parser.add_argument("--existing-wave", type=safe_name, action="append", default=[])
    parser.add_argument("--audio-run", type=safe_name, default="full_audio_001")
    parser.add_argument('--text-release', type=Path,
                        help='Immutable frozen text release directory (default: ROOT/text_recovery/release)')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalizer-workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--wave-seconds", type=float, default=300)
    parser.add_argument("--min-wave", type=int, default=10000)
    args = parser.parse_args()
    if min(args.workers, args.finalizer_workers, args.poll_seconds, args.wave_seconds, args.min_wave) <= 0:
        parser.error("Worker counts, intervals and minimum wave size must be positive")
    if len(set(args.existing_wave)) != len(args.existing_wave):
        parser.error("Existing waves must be distinct")
    lock_path = args.root / "state/locks/finish.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finisher = Finisher(args, lock.fileno())
        try:
            finisher.run_all()
        except Exception as error:
            finisher.state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_unix=time.time())
            finisher.save()
            raise


if __name__ == "__main__":
    main()
