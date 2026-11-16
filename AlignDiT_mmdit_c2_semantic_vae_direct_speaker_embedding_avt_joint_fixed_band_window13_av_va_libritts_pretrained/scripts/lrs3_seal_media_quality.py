"""Audit explicitly reviewed terminal media failures; --seal publishes closure.

Default is read-only. --exclusions must name a JSON list of sample_id/reason
objects exactly matching the final canonical media verification failures.
Only --seal, under exclusive media.lock and with no live media writers, may
remove the media WRITE_ACTIVE marker. No source or cache arrays are removed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from aligndit.lrs3_media_quality import (
    EXCLUSIONS, TERMINAL, build_terminal, load_terminal, snapshot,
)


def command_targets_root(command, root, cwd):
    """Recognize both argparse spellings and resolve relative/symlink roots."""
    for index, token in enumerate(command):
        value = None
        if token in {'--root', '--output-root'} and index + 1 < len(command):
            value = command[index + 1]
        elif token.startswith(('--root=', '--output-root=')):
            value = token.split('=', 1)[1]
        if value:
            path = Path(value)
            if not path.is_absolute():path = cwd / path
            if path.resolve() == root.resolve():return True
    return False


def active_media_writers(root):
    names = {"lrs3_media_worker.py", "lrs3_media_worker_repair.py", "lrs3_media_worker_pipeline.py",
             "lrs3_crop_parallel.py", "lrs3_media_worker_benchmark.py", "lrs3_media_worker_decode_benchmark.py",
             "lrs3_media_memory_benchmark.py", "lrs3_media_early_fan.py"}
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg).name in names for arg in command):
            try:
                cwd = (entry / 'cwd').resolve(strict=True)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                raise RuntimeError(f'Cannot establish media process output scope: PID {entry.name}')
            if command_targets_root(command, root, cwd):result.append(int(entry.name))
    return sorted(result)


def immutable_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Immutable artifact differs from the reviewed proposal: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".quality_", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--seal", action="store_true", help="Publish reviewed terminal closure and end the media write guard")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    with (root / "state/locks/media.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_media_writers(root):
            raise RuntimeError(f"Media processes still exist: {active_media_writers(root)}")
        marker = root / "state/video_25hz/WRITE_ACTIVE.json"
        if (root / TERMINAL).exists():
            terminal = load_terminal(root)
            if terminal["reviewed_exclusions"] != snapshot(args.exclusions):
                raise ValueError("Existing immutable closure uses another explicit review list")
        else:
            terminal, exclusions = build_terminal(root, args.exclusions)
            if args.seal:
                if not marker.exists():
                    raise RuntimeError("Failed media write guard is absent before explicit sealing")
                guard_bytes = marker.read_bytes()
                guard_copy = root / "reports/media_full_pipeline/quality_guard_before.json"
                immutable_json(guard_copy, {"path": str(marker), "sha256": hashlib.sha256(guard_bytes).hexdigest(),
                                           "contents": json.loads(guard_bytes)})
                immutable_json(root / EXCLUSIONS, exclusions)
                terminal["evidence_files"] += [snapshot(guard_copy), snapshot(root / EXCLUSIONS)]
                terminal["sealed_unix"] = time.time()
                # Check every input again before the immutable completion marker.
                if any(snapshot(ref["path"]) != ref for ref in terminal["evidence_files"]):
                    raise ValueError("Quality evidence changed during sealing")
                immutable_json(root / TERMINAL, terminal)
                terminal = load_terminal(root)
        if args.seal:
            if active_media_writers(root):
                raise RuntimeError("A media writer appeared before terminal guard removal")
            marker.unlink(missing_ok=True)
        print(json.dumps({"sealed": args.seal, "requested": terminal["requested"],
            "verified": terminal["verified"], "excluded_count": terminal["excluded_count"],
            "media_complete": False, "processing_complete": True,
            "terminal": str(root / TERMINAL) if args.seal else None,
            "terminal_sha256": terminal.get("_sha256"), "source_or_cache_files_deleted": False}, indent=2))


if __name__ == "__main__":
    main()
