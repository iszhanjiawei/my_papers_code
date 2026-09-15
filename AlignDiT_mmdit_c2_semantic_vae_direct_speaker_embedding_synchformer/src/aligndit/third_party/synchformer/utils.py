"""Small dependency replacements; no automatic network downloads."""
from collections.abc import Iterable
from pathlib import Path


def to_2tuple(value):
    # Equivalent to timm.layers.helpers._ntuple(2).
    return tuple(value) if isinstance(value, Iterable) and not isinstance(value, str) else (value, value)


def check_if_file_exists_else_download(path, *args, **kwargs):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required vendored Synchformer file is absent: {path}")
