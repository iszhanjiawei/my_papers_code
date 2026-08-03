"""Shared invariants and safe I/O helpers for the Semantic-VAE cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_temp_file(temp_path: Path, destination: Path, digest: str, size_bytes: int) -> AtomicWriteResult:
    try:
        # A same-filesystem hard link is an atomic create-if-absent operation.  Unlike
        # exists()+replace(), it cannot overwrite a file published concurrently by another rank.
        os.link(temp_path, destination)
    except FileExistsError:
        if destination.is_symlink():
            temp_path.unlink()
            raise RuntimeError(f"Refusing to compare cache output through a symlink: {destination}") from None
        existing_size = destination.stat().st_size
        existing_digest = sha256_file(destination)
        temp_path.unlink()
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

    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> AtomicWriteResult:
    return atomic_write_lines(path, [json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), "\n"])


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> AtomicWriteResult:
    return atomic_write_lines(path, (f"{canonical_json(record)}\n" for record in records))


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
