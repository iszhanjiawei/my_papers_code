"""Shared invariants and safe I/O helpers for the Semantic-VAE cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from typing_extensions import Self


CACHE_SCHEMA_VERSION = 1
SAMPLE_RATE = 16_000
SEMANTIC_VAE_HOP_LENGTH = 400
SEMANTIC_VAE_LATENT_DIM = 64
HUBERT_HIDDEN_DIM = 1024
BASE_POSTERIOR_SEED = 666

TRAIN_SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500")
DEV_SUBSETS = ("dev-clean", "dev-other")
DEFAULT_SUBSETS = (*TRAIN_SUBSETS, *DEV_SUBSETS)

OFFICIAL_UTTERANCE_COUNTS = {
    "train-clean-100": 28_539,
    "train-clean-360": 104_014,
    "train-other-500": 148_688,
    "dev-clean": 2_703,
    "dev-other": 2_864,
    "test-clean": 2_620,
    "test-other": 2_939,
}


@dataclass(frozen=True)
class AtomicWriteResult:
    path: Path
    sha256: str
    size_bytes: int
    created: bool


@dataclass(frozen=True)
class NpyValidation:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    size_bytes: int


_ATTEMPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def prefixed_path(relative_to_user_root: str) -> Path:
    """Resolve a `/zjw524/...` path using the repository ROOT_PREFIX convention."""

    root_prefix = os.environ.get("ROOT_PREFIX", "")
    relative = relative_to_user_root.lstrip("/")
    return Path(f"{root_prefix}/zjw524/{relative}")


def expected_latent_frames(num_samples: int) -> int:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    return (num_samples + SEMANTIC_VAE_HOP_LENGTH - 1) // SEMANTIC_VAE_HOP_LENGTH


def stable_utterance_seed(base_seed: int, utterance_key: str) -> int:
    components = utterance_key.split("/")
    if (
        len(components) != 4
        or any(component in {"", ".", ".."} for component in components)
        or utterance_key.startswith("/")
        or "\\" in utterance_key
    ):
        raise ValueError(f"utterance_key must be a canonical four-component relative key, got {utterance_key!r}")
    if not 0 <= base_seed < 2**63:
        raise ValueError(f"base_seed must be in [0, 2**63), got {base_seed}")
    digest = hashlib.sha256(f"{base_seed}:{utterance_key}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_attempt_id(attempt_id: str) -> str:
    if not _ATTEMPT_ID_PATTERN.fullmatch(attempt_id):
        raise ValueError(
            "attempt_id must be 1-80 characters, start with an alphanumeric character, "
            "and contain only alphanumerics, '.', '_' or '-'"
        )
    return attempt_id


def safe_join(root: Path, relative_path: str) -> Path:
    """Join a canonical POSIX relative path without following cache-tree symlinks."""

    candidate = Path(relative_path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or "\\" in relative_path
        or candidate.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"Expected a canonical relative path, got {relative_path!r}")
    resolved_root = root.resolve()
    destination = resolved_root.joinpath(*candidate.parts)
    current = resolved_root
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"Refusing to follow a symlink in the cache tree: {current}")
    return destination


def exact_rank_shard(items: Sequence[Any], rank: int, world_size: int) -> Sequence[Any]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return items[rank::world_size]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir() or current.is_symlink():
        raise RuntimeError(f"Cannot create a cache directory below {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir() or directory.is_symlink():
                raise RuntimeError(f"Cache directory was replaced by a non-directory: {directory}") from None
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _commit_temp_file(temp_path: Path, destination: Path, digest: str, size_bytes: int) -> AtomicWriteResult:
    try:
        # A same-filesystem hard link is an atomic create-if-absent operation.  Unlike
        # exists()+replace(), it cannot overwrite a file published concurrently by another rank.
        os.link(temp_path, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            temp_path.unlink()
            _fsync_directory(destination.parent)
            raise RuntimeError(f"Refusing to compare cache output with a non-regular file: {destination}") from None
        existing_size = destination.stat().st_size
        existing_digest = sha256_file(destination)
        temp_path.unlink()
        _fsync_directory(destination.parent)
        if existing_size == size_bytes and existing_digest == digest:
            return AtomicWriteResult(destination, digest, size_bytes, created=False)
        raise FileExistsError(
            f"Refusing to replace non-identical file {destination}: "
            f"existing sha256={existing_digest}, new sha256={digest}"
        )
    temp_path.unlink()
    _fsync_directory(destination.parent)
    return AtomicWriteResult(destination, digest, size_bytes, created=True)


def atomic_write_lines(path: Path, lines: Iterable[str]) -> AtomicWriteResult:
    """Write UTF-8 lines atomically without ever replacing different content."""

    _durable_mkdir(path.parent)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as file:
            for line in lines:
                encoded = line.encode("utf-8")
                file.write(encoded)
                digest.update(encoded)
                size_bytes += len(encoded)
            file.flush()
            os.fsync(file.fileno())
        return _commit_temp_file(temp_path, path, digest.hexdigest(), size_bytes)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
            _fsync_directory(path.parent)
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> AtomicWriteResult:
    return atomic_write_lines(path, [json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), "\n"])


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> AtomicWriteResult:
    return atomic_write_lines(path, (f"{canonical_json(record)}\n" for record in records))


def atomic_write_npy(path: Path, array: np.ndarray) -> AtomicWriteResult:
    """Publish an NPY file atomically and never replace different existing content."""

    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy.ndarray, got {type(array).__name__}")
    _durable_mkdir(path.parent)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            # Passing the open file is intentional: np.save(Path) would append '.npy'
            # to the temporary filename and break the atomic publication protocol.
            np.save(file, array, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        size_bytes = temp_path.stat().st_size
        digest = sha256_file(temp_path)
        return _commit_temp_file(temp_path, path, digest, size_bytes)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
            _fsync_directory(path.parent)
        raise


def quarantine_file(source: Path, quarantine_root: Path, relative_path: str) -> Path:
    """Move a regular file to a recoverable same-filesystem quarantine without overwriting."""

    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Expected a regular cache file to quarantine, got {source}")
    destination = safe_join(quarantine_root, relative_path)
    _durable_mkdir(destination.parent)
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_file() and not destination.is_symlink() and os.path.samefile(source, destination):
            # Recovery from a crash after link(source, destination) but before source.unlink().
            _fsync_directory(destination.parent)
            source.unlink()
            _fsync_directory(source.parent)
            return destination
        raise FileExistsError(f"Refusing to replace an existing quarantined file: {destination}") from None
    _fsync_directory(destination.parent)
    source.unlink()
    _fsync_directory(source.parent)
    return destination


def durable_unlink(path: Path) -> None:
    """Remove one regular file and durably persist the directory entry change."""

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular file to unlink, got {path}")
    path.unlink()
    _fsync_directory(path.parent)


def validate_npy(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any] | type[np.floating[Any]],
) -> NpyValidation:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular NPY file, got {path}")
    expected_numpy_dtype = np.dtype(expected_dtype)
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Cannot read NPY file {path}: {error}") from error
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected ndarray in {path}, got {type(array).__name__}")
    if array.shape != expected_shape:
        raise ValueError(f"Wrong shape for {path}: expected {expected_shape}, got {array.shape}")
    if array.dtype != expected_numpy_dtype:
        raise ValueError(f"Wrong dtype for {path}: expected {expected_numpy_dtype}, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite value in {path}")
    size_bytes = path.stat().st_size
    return NpyValidation(path, array.shape, array.dtype.str, sha256_file(path), size_bytes)


def read_append_only_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a single-writer progress log, ignoring only a truncated final line."""

    with path.open("rb") as file:
        lines = file.readlines()
    for line_number, encoded in enumerate(lines, start=1):
        if not encoded.endswith(b"\n"):
            if line_number != len(lines):
                raise ValueError(f"Truncated non-final line at {path}:{line_number}")
            break
        if not encoded.strip():
            continue
        try:
            value = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Expected an object at {path}:{line_number}, got {type(value).__name__}")
        yield value


class JsonlProgressWriter:
    """A single-attempt, single-rank append-only JSONL writer."""

    def __init__(self, path: Path, fsync_interval: int = 100) -> None:
        if fsync_interval <= 0:
            raise ValueError(f"fsync_interval must be positive, got {fsync_interval}")
        _durable_mkdir(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self._descriptor = os.open(path, flags, 0o644)
        self._fsync_interval = fsync_interval
        self._pending = 0
        try:
            _fsync_directory(path.parent)
        except BaseException:
            os.close(self._descriptor)
            self._descriptor = -1
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
            raise

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = f"{canonical_json(record)}\n".encode()
        written = os.write(self._descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"Short progress write to {self.path}: {written}/{len(encoded)} bytes")
        self._pending += 1
        if self._pending >= self._fsync_interval:
            self.flush()

    def flush(self) -> None:
        if self._descriptor < 0:
            return
        os.fsync(self._descriptor)
        self._pending = 0

    def close(self) -> None:
        if self._descriptor < 0:
            return
        try:
            self.flush()
        finally:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}, got {type(value).__name__}")
            yield value
