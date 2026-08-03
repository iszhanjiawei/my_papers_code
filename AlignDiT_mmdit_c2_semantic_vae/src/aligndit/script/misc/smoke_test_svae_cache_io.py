"""Exercise Semantic-VAE cache invariants on the real target filesystem."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from aligndit.script.misc.svae_cache_utils import (
    JsonlProgressWriter,
    atomic_write_npy,
    exact_rank_shard,
    expected_latent_frames,
    quarantine_file,
    read_append_only_jsonl,
    safe_join,
    stable_utterance_seed,
    validate_npy,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="Existing directory on the same NFS filesystem used by the production cache.",
    )
    return parser.parse_args()


def atomic_worker(path: str, value: float, queue: Any) -> None:
    try:
        array = np.full((7, 64), value, dtype=np.float32)
        result = atomic_write_npy(Path(path), array)
        queue.put(("ok", result.created, result.sha256, result.size_bytes))
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        queue.put(("error", type(error).__name__, str(error)))


def collect_process_results(processes: list[mp.Process], queue: Any) -> list[tuple[Any, ...]]:
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError(f"Atomic writer process did not exit: pid={process.pid}")
        if process.exitcode != 0:
            raise RuntimeError(f"Atomic writer process failed: pid={process.pid}, exitcode={process.exitcode}")
    results: list[tuple[Any, ...]] = []
    for _ in processes:
        try:
            results.append(queue.get(timeout=5))
        except Empty as error:
            raise RuntimeError("Atomic writer did not report a result") from error
    return results


def test_exact_sharding() -> None:
    for item_count in (0, 1, 7, 8, 9, 17):
        items = list(range(item_count))
        for world_size in (1, 2, 3, 8):
            shards = [list(exact_rank_shard(items, rank, world_size)) for rank in range(world_size)]
            flattened = [item for shard in shards for item in shard]
            assert sorted(flattened) == items
            assert len(flattened) == len(set(flattened))
    for invalid_rank, invalid_world in ((0, 0), (-1, 1), (1, 1)):
        try:
            exact_rank_shard([1], invalid_rank, invalid_world)
        except ValueError:
            pass
        else:
            raise AssertionError((invalid_rank, invalid_world))


def test_frame_and_seed_contracts() -> None:
    expected = {1: 1, 399: 1, 400: 1, 401: 2, 800: 2, 801: 3, 60_960: 153}
    assert {samples: expected_latent_frames(samples) for samples in expected} == expected
    try:
        expected_latent_frames(0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero samples must fail")
    key = "train-clean-100/103/1240/103-1240-0015"
    assert stable_utterance_seed(666, key) == 3_920_034_511_769_737_100
    assert stable_utterance_seed(666, key) == stable_utterance_seed(666, key)
    assert stable_utterance_seed(667, key) != stable_utterance_seed(666, key)


def expect_invalid_npy(path: Path, shape: tuple[int, ...]) -> None:
    try:
        validate_npy(path, expected_shape=shape, expected_dtype=np.float32)
    except (OSError, TypeError, ValueError):
        return
    raise AssertionError(f"Invalid NPY unexpectedly passed validation: {path}")


def test_npy_validation(root: Path) -> None:
    expected_shape = (3, 64)
    valid_path = root / "valid.npy"
    array = np.arange(np.prod(expected_shape), dtype=np.float32).reshape(expected_shape)
    write = atomic_write_npy(valid_path, array)
    validation = validate_npy(valid_path, expected_shape=expected_shape, expected_dtype=np.float32)
    assert validation.sha256 == write.sha256
    assert validation.size_bytes == write.size_bytes

    invalid_arrays = {
        "float64.npy": array.astype(np.float64),
        "three_dimensional.npy": array[None],
        "transposed.npy": array.T,
        "nan.npy": np.full(expected_shape, np.nan, dtype=np.float32),
        "positive_inf.npy": np.full(expected_shape, np.inf, dtype=np.float32),
        "negative_inf.npy": np.full(expected_shape, -np.inf, dtype=np.float32),
    }
    for name, invalid in invalid_arrays.items():
        path = root / name
        with path.open("wb") as file:
            np.save(file, invalid, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        expect_invalid_npy(path, expected_shape)

    zero_path = root / "zero.npy"
    zero_path.touch()
    expect_invalid_npy(zero_path, expected_shape)
    truncated_path = root / "truncated.npy"
    truncated_path.write_bytes(valid_path.read_bytes()[:31])
    expect_invalid_npy(truncated_path, expected_shape)


def test_atomic_publication(root: Path) -> None:
    context = mp.get_context("spawn")
    same_path = root / "same.npy"
    same_queue = context.Queue()
    same_processes = [context.Process(target=atomic_worker, args=(str(same_path), 1.0, same_queue)) for _ in range(16)]
    for process in same_processes:
        process.start()
    same_results = collect_process_results(same_processes, same_queue)
    assert all(result[0] == "ok" for result in same_results), same_results
    assert sum(bool(result[1]) for result in same_results) == 1
    assert len({result[2] for result in same_results}) == 1
    validate_npy(same_path, expected_shape=(7, 64), expected_dtype=np.float32)

    conflict_path = root / "conflict.npy"
    conflict_queue = context.Queue()
    conflict_processes = [
        context.Process(target=atomic_worker, args=(str(conflict_path), float(index % 2), conflict_queue))
        for index in range(16)
    ]
    for process in conflict_processes:
        process.start()
    conflict_results = collect_process_results(conflict_processes, conflict_queue)
    successes = [result for result in conflict_results if result[0] == "ok"]
    failures = [result for result in conflict_results if result[0] == "error"]
    assert len(successes) == 8, conflict_results
    assert len(failures) == 8, conflict_results
    assert {result[1] for result in failures} == {"FileExistsError"}
    winner = np.load(conflict_path, allow_pickle=False)
    assert np.all(winner == winner.flat[0]) and winner.flat[0] in {0.0, 1.0}


def test_progress_logs(root: Path) -> None:
    progress_path = root / "attempt.rank-00000-of-00001.jsonl"
    with JsonlProgressWriter(progress_path, fsync_interval=2) as writer:
        writer.append({"key": "a", "value": 1})
        writer.append({"key": "b", "value": 2})
    assert list(read_append_only_jsonl(progress_path)) == [
        {"key": "a", "value": 1},
        {"key": "b", "value": 2},
    ]
    try:
        JsonlProgressWriter(progress_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Progress path reuse must fail")

    truncated = root / "truncated.jsonl"
    truncated.write_bytes(b'{"key":"complete"}\n{"key":"partial"')
    assert list(read_append_only_jsonl(truncated)) == [{"key": "complete"}]
    malformed = root / "malformed.jsonl"
    malformed.write_bytes(b"not-json\n")
    try:
        list(read_append_only_jsonl(malformed))
    except ValueError:
        pass
    else:
        raise AssertionError("A malformed complete progress line must fail")


def test_safe_paths_and_quarantine(root: Path) -> None:
    root.mkdir(parents=True)
    cache_root = root / "cache"
    cache_root.mkdir()
    assert safe_join(cache_root, "latents/a.npy") == cache_root.resolve() / "latents" / "a.npy"
    for invalid in ("/absolute.npy", "../escape.npy", "a/../b.npy", "a//b.npy", "a\\b.npy", "./a.npy"):
        try:
            safe_join(cache_root, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe path accepted: {invalid}")

    dangling = cache_root / "dangling"
    dangling.symlink_to(root / "missing-target")
    try:
        safe_join(cache_root, "dangling/file.npy")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Dangling parent symlink must fail")

    source = root / "source.npy"
    source.write_bytes(b"recoverable")
    quarantine_root = root / "quarantine"
    destination = safe_join(quarantine_root, "latents/source.npy")
    destination.parent.mkdir(parents=True)
    os.link(source, destination)
    recovered = quarantine_file(source, quarantine_root, "latents/source.npy")
    assert recovered == destination and destination.read_bytes() == b"recoverable" and not source.exists()

    conflicting_source = root / "conflicting-source.npy"
    conflicting_source.write_bytes(b"new")
    try:
        quarantine_file(conflicting_source, quarantine_root, "latents/source.npy")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Quarantine must not replace different content")
    assert conflicting_source.read_bytes() == b"new"


def main() -> None:
    args = get_args()
    scratch_root = args.scratch_root.resolve(strict=True)
    if not scratch_root.is_dir() or scratch_root.is_symlink():
        raise ValueError(f"--scratch-root must be a regular directory: {scratch_root}")
    work_root = Path(tempfile.mkdtemp(prefix="svae-cache-io-", dir=scratch_root))
    try:
        test_exact_sharding()
        test_frame_and_seed_contracts()
        test_npy_validation(work_root / "npy")
        test_atomic_publication(work_root / "atomic")
        test_progress_logs(work_root / "progress")
        test_safe_paths_and_quarantine(work_root / "paths")
        print(f"Semantic-VAE cache I/O smoke test passed on {work_root}")
    finally:
        shutil.rmtree(work_root)


if __name__ == "__main__":
    main()
