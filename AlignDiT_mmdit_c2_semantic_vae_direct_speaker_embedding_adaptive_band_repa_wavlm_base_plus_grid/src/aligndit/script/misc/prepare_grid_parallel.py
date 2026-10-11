"""Run the frozen GRID extractor across disjoint CPU/GPU worker processes.

Create the immutable plan with prepare_grid.py first. This coordinator changes
only scheduling: every worker calls the original run_stage, with the original
per-utterance seeds, shapes, numerical settings and artifact receipts. It owns
the cache lock and writer guard until all workers exit and the full audit passes.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aligndit.script.misc import prepare_grid as prep


GPU_STAGES = ("latent", "video", "speaker", "repa")
PATH_ARGUMENTS = (
    "output_dir", "grid_root", "source_audit_dir", "semantic_vae_repo", "semantic_vae_checkpoint",
    "semantic_vae_spec", "normalization", "vocab", "speaker_checkpoint", "wavlm_model_dir",
    "avhubert_checkpoint", "fairseq_repo", "avhubert_user_dir", "wavlm_resolved_dir",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def shard_indices(total: int, shard_index: int, shard_count: int) -> list[int]:
    if total < 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid GRID shard dimensions")
    return list(range(shard_index, total, shard_count))


@dataclass(frozen=True)
class WorkerCommand:
    name: str
    command: list[str]
    log_path: Path
    env: dict[str, str] | None = None


class WorkerFailed(RuntimeError):
    pass


def terminate_owned(processes: list[subprocess.Popen], *, grace_seconds: float = 5.0) -> None:
    """Stop only session groups created by this invocation, including descendants.

    A failed session leader may already have exited while its FFmpeg subprocess
    remains. Its recorded group still belongs to this pool and must be stopped.
    Never discover or kill workers by command substrings, GPU IDs or cache paths.
    """
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(grace_seconds, 0)
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A child can outlive an exited leader. SIGKILL closes that group as well;
    # these are fresh PGIDs from start_new_session, never arbitrary external IDs.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()


def run_pool(commands: list[WorkerCommand], *, poll_seconds: float = 0.5,
             terminate_grace: float = 5.0) -> list[dict]:
    """Wait for one owned pool; any failed child cancels all its peers."""
    if poll_seconds <= 0:
        raise ValueError("Worker poll interval must be positive")
    processes, handles = [], []
    try:
        for command in commands:
            command.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = command.log_path.open("wb")
            handles.append(handle)
            process = subprocess.Popen(
                command.command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=command.env, start_new_session=True, close_fds=True,
            )
            processes.append(process)
            print(f"Started {command.name}: pid={process.pid}, log={command.log_path}", flush=True)
        while True:
            pending = False
            for command, process in zip(commands, processes):
                code = process.poll()
                if code is None:
                    pending = True
                elif code != 0:
                    raise WorkerFailed(f"GRID worker {command.name} exited {code}; inspect {command.log_path}")
            if not pending:
                break
            time.sleep(poll_seconds)
        return [{"name": command.name, "pid": process.pid, "returncode": process.returncode,
                 "log": str(command.log_path)} for command, process in zip(commands, processes)]
    except BaseException:
        terminate_owned(processes, grace_seconds=terminate_grace)
        raise
    finally:
        for handle in handles:
            handle.close()


@contextmanager
def coordinator_signals():
    previous = {}
    def interrupted(signum, _frame):
        # Subsequent signals must not interrupt owned-child cleanup.
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        raise InterruptedError(f"GRID coordinator received signal {signum}")
    try:
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[number] = signal.signal(number, interrupted)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _serialize_arguments(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _worker_commands(kind: str, count: int, args, config: Path, run_dir: Path) -> list[WorkerCommand]:
    commands = []
    for index in range(count):
        name = f"{kind}_{index:02d}"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["OMP_NUM_THREADS"] = str(args.audio_threads if kind == "audio" else args.torch_threads)
        environment["MKL_NUM_THREADS"] = environment["OMP_NUM_THREADS"]
        # Each GPU worker sees exactly one physical GPU. Its logical cuda:0
        # maintains the same generator/device semantics as the serial extractor.
        environment["CUDA_VISIBLE_DEVICES"] = "" if kind == "audio" else args.gpus[index]
        command = [sys.executable, "-u", "-m", "aligndit.script.misc.prepare_grid_parallel",
                   "--worker-config", str(config), "--worker-kind", kind,
                   "--shard-index", str(index), "--shard-count", str(count)]
        commands.append(WorkerCommand(name, command, run_dir / f"{name}.log", environment))
    return commands


def coordinate(args) -> None:
    if args.audio_workers <= 0 or args.audio_threads <= 0 or args.torch_threads <= 0:
        raise ValueError("Worker and CPU thread counts must be positive")
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must contain distinct GPU identifiers")
    if any(not gpu or gpu.startswith("-") for gpu in args.gpus):
        raise ValueError("Invalid GPU identifier")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch one GRID parallel coordinator, outside torchrun/accelerate")
    args.output_dir = args.output_dir.expanduser().resolve(strict=True)
    if args.output_dir.is_relative_to(Path(__file__).resolve().parents[5]):
        raise ValueError("GRID runtime caches must remain outside my_papers_code")
    args.device = "cuda:0"
    prep.torch.set_num_threads(args.torch_threads)
    with (args.output_dir / ".prepare.lock").open("a") as lock, coordinator_signals():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        planned = prep.reuse_plan(args)
        if planned is None:
            raise ValueError("Create the complete frozen plan with scripts/prepare_grid.sh --plan-only first")
        rows, metadata, _spec = planned
        if not rows:
            raise ValueError("The GRID inventory is empty")
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
        run_dir = args.output_dir / "parallel_runs" / run_id
        run_dir.mkdir(parents=True)
        config = run_dir / "request.json"
        request = {
            "schema_version": 1, "run_id": run_id, "owner_pid": os.getpid(), "hostname": socket.gethostname(),
            "arguments": _serialize_arguments(args), "count": len(rows),
            "inventory_sha256": metadata["manifests"]["inventory.jsonl"]["sha256"],
            "inventory_meta_sha256": prep.sha256_file(args.output_dir / "manifests/inventory_meta.json"),
            "extraction_sha256": metadata["extraction_sha256"],
            "coordinator_sha256": prep.sha256_file(Path(__file__)),
            "started_at": timestamp(), "partition": "original_inventory_rows[worker_index::worker_count]",
        }
        prep.write_json(config, request)
        guard = args.output_dir / "WRITE_ACTIVE.json"
        prep.write_json(guard, {"pid": os.getpid(), "run_id": run_id, "stage": "parallel",
                                "request": str(config)}, replace=True)
        (args.output_dir / "complete.json").unlink(missing_ok=True)
        status = {"run_id": run_id, "state": "running", "started_at": request["started_at"], "pools": {}}
        prep.write_json(run_dir / "status.json", status, replace=True)
        print(f"GRID parallel run {run_id}: {len(rows)} utterances, GPUs={','.join(args.gpus)}, "
              f"audio_workers={args.audio_workers}, logs={run_dir}", flush=True)
        try:
            audio_count = min(args.audio_workers, len(rows))
            gpu_count = min(len(args.gpus), len(rows))
            status["pools"]["audio"] = run_pool(_worker_commands("audio", audio_count, args, config, run_dir))
            prep.write_json(run_dir / "status.json", status, replace=True)
            status["pools"]["gpu"] = run_pool(_worker_commands("gpu", gpu_count, args, config, run_dir))
            prep.write_json(run_dir / "status.json", status, replace=True)
            if metadata["extraction_protocol"]["implementation_sha256"] != prep.implementation_hashes():
                raise RuntimeError("Frozen extraction source changed while workers were running")
            # This is the only completion publisher. It checks every original
            # inventory row and all five artifact receipts, never a worker shard.
            prep.verify(rows, metadata, args, required=True)
            status.update(state="complete", finished_at=timestamp())
            prep.write_json(run_dir / "status.json", status, replace=True)
        except BaseException as error:
            (args.output_dir / "complete.json").unlink(missing_ok=True)
            status.update(state="failed", finished_at=timestamp(), error=f"{type(error).__name__}: {error}")
            prep.write_json(run_dir / "status.json", status, replace=True)
            raise
        finally:
            guard.unlink(missing_ok=True)


def _require_owner(request: dict, args) -> None:
    if os.getppid() != request["owner_pid"]:
        raise RuntimeError("GRID worker must be a direct child of its active coordinator")
    guard = prep.read_json_object(args.output_dir / "WRITE_ACTIVE.json")
    if guard.get("pid") != request["owner_pid"] or guard.get("run_id") != request["run_id"]:
        raise RuntimeError("GRID worker ownership does not match the active cache writer")


def _parent_death_signal() -> None:
    """Prevent orphan workers continuing cache writes after coordinator SIGKILL."""
    if sys.platform != "linux":
        raise RuntimeError("The GRID parallel coordinator requires Linux parent-death signals")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def worker(config: Path, kind: str, shard_index: int, shard_count: int) -> None:
    _parent_death_signal()
    request = prep.read_json_object(config)
    args = argparse.Namespace(**request["arguments"])
    for key in PATH_ARGUMENTS:
        if getattr(args, key, None) is not None:
            setattr(args, key, Path(getattr(args, key)))
    _require_owner(request, args)
    if request.get("coordinator_sha256") != prep.sha256_file(Path(__file__)):
        raise RuntimeError("Parallel driver changed after coordinator launch")
    metadata_path = args.output_dir / "manifests/inventory_meta.json"
    inventory_path = args.output_dir / "manifests/inventory.jsonl"
    if (prep.sha256_file(metadata_path) != request["inventory_meta_sha256"]
            or prep.sha256_file(inventory_path) != request["inventory_sha256"]):
        raise RuntimeError("Frozen GRID plan changed after coordinator validation")
    metadata = prep.read_json_object(metadata_path)
    if (metadata["extraction_sha256"] != request["extraction_sha256"]
            or metadata["extraction_protocol"]["implementation_sha256"] != prep.implementation_hashes()):
        raise RuntimeError("Frozen extraction code/protocol changed after coordinator validation")
    indices = set(shard_indices(request["count"], shard_index, shard_count))
    rows = []
    count = 0
    with inventory_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                rows.append(json.loads(line))
            count += 1
    if count != request["count"] or len(rows) != len(indices):
        raise RuntimeError("GRID worker shard does not cover its planned inventory indices")
    args.device = "cpu" if kind == "audio" else "cuda:0"
    prep.torch.set_num_threads(args.audio_threads if kind == "audio" else args.torch_threads)
    if kind == "gpu":
        if not prep.torch.cuda.is_available():
            raise RuntimeError("Assigned GRID extraction GPU is unavailable")
        prep.torch.cuda.set_device(0)
    prep.torch.backends.cuda.matmul.allow_tf32 = False
    prep.torch.backends.cudnn.allow_tf32 = True
    prep.torch.backends.cudnn.benchmark = False
    prep.torch.backends.cudnn.deterministic = False
    prep.torch.use_deterministic_algorithms(False)
    spec = prep.read_json_object(args.semantic_vae_spec)
    status_path = config.parent / f"{kind}_{shard_index:02d}.json"
    status = {"state": "running", "pid": os.getpid(), "run_id": request["run_id"],
              "kind": kind, "index": shard_index, "shards": shard_count, "count": len(rows),
              "device": args.device, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "started_at": timestamp(), "stages": {}}
    try:
        for stage in ("audio",) if kind == "audio" else GPU_STAGES:
            _require_owner(request, args)
            status["current_stage"] = stage
            status["stages"][stage] = {"started_at": timestamp()}
            prep.write_json(status_path, status, replace=True)
            prep.run_stage(stage, rows, metadata, spec, args)
            status["stages"][stage]["finished_at"] = timestamp()
        status.update(state="complete", finished_at=timestamp())
        prep.write_json(status_path, status, replace=True)
    except BaseException as error:
        status.update(state="failed", finished_at=timestamp(), error=f"{type(error).__name__}: {error}")
        prep.write_json(status_path, status, replace=True)
        raise


def parse_args(argv: list[str] | None = None):
    # Reuse the original path defaults without changing its frozen source file.
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__))]
        defaults = prep.parse_args()
    finally:
        sys.argv = original_argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3", help="Distinct physical GPU IDs (each worker sees logical cuda:0)")
    parser.add_argument("--audio-workers", type=int, default=12)
    parser.add_argument("--audio-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument("--limit", type=int, help="Must match the existing smoke plan; omit for full training data")
    for argument in PATH_ARGUMENTS:
        if argument != "wavlm_resolved_dir":
            parser.add_argument("--" + argument.replace("_", "-"), type=Path, default=getattr(defaults, argument))
    parser.add_argument("--ffmpeg", default=defaults.ffmpeg)
    parser.add_argument("--ffprobe", default=defaults.ffprobe)
    parsed = parser.parse_args(argv)
    for key, value in vars(defaults).items():
        if not hasattr(parsed, key):
            setattr(parsed, key, value)
    parsed.gpus = tuple(identifier.strip() for identifier in parsed.gpus.split(","))
    return parsed


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if "--worker-config" in arguments:
        parser = argparse.ArgumentParser(description="Internal worker for an active GRID coordinator")
        parser.add_argument("--worker-config", type=Path, required=True)
        parser.add_argument("--worker-kind", choices=("audio", "gpu"), required=True)
        parser.add_argument("--shard-index", type=int, required=True)
        parser.add_argument("--shard-count", type=int, required=True)
        args = parser.parse_args(arguments)
        worker(args.worker_config, args.worker_kind, args.shard_index, args.shard_count)
    else:
        coordinate(parse_args(arguments))


if __name__ == "__main__":
    main()
