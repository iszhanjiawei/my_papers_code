#!/usr/bin/env python3
"""Install the official, pinned SyncNet assets outside experiment snapshots.

Run this with the existing aligndit Python environment. Existing scientific
packages are never upgraded. Downloads are verified before an atomic rename;
an existing file with an unexpected checksum is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "https://github.com/joonson/syncnet_python.git"
REVISION = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
DOWNLOAD_BASE = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/"
# SHA-256 values measured from the official HTTPS files, pinned with this tool.
ASSETS = {
    "syncnet_v2.model": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
    "sfd_face.pth": "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491",
    "example.avi": "1724121b3a50141cfc9e77c034f7de914024d015b92762ae12d3e1a7151b8621",
}


def default_asset_dir() -> Path:
    return Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/syncnet")


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, asset_dir: Path) -> dict:
    destination = asset_dir / name
    expected = ASSETS[name]
    url = DOWNLOAD_BASE + name
    if destination.exists():
        actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for existing {destination}: {actual}; expected {expected}. "
                "Inspect/remove the invalid file before retrying; it was not overwritten."
            )
        print(f"Verified existing {name}", flush=True)
    else:
        for attempt in range(3):
            fd, temporary_name = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=asset_dir)
            temporary = Path(temporary_name)
            try:
                print(f"Downloading {url} (attempt {attempt + 1}/3)", flush=True)
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=90) as response:
                    downloaded = 0
                    next_report = 16 * 1024 * 1024
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(f"  {name}: {downloaded / 1024**2:.0f} MiB downloaded", flush=True)
                            next_report += 16 * 1024 * 1024
                actual = sha256(temporary)
                if actual != expected:
                    raise RuntimeError(f"Downloaded {name} checksum mismatch: {actual}; expected {expected}")
                os.replace(temporary, destination)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return {"path": str(destination), "url": url, "sha256": expected, "bytes": destination.stat().st_size}


def install_source(asset_dir: Path) -> Path:
    source = asset_dir / "syncnet_python"
    if not source.exists():
        # A failed clone must not leave a directory that looks installed.
        with tempfile.TemporaryDirectory(prefix="syncnet-clone-", dir=asset_dir) as temporary:
            clone = Path(temporary) / "source"
            for attempt in range(3):
                try:
                    run(["git", "-c", "http.version=HTTP/1.1", "clone", REPOSITORY, str(clone)])
                    break
                except subprocess.CalledProcessError:
                    if clone.exists():
                        shutil.rmtree(clone)
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            run(["git", "checkout", "--detach", REVISION], cwd=clone)
            os.replace(clone, source)
    if not (source / ".git").is_dir():
        raise RuntimeError(f"Existing source directory is not a Git checkout: {source}")
    actual = run(["git", "rev-parse", "HEAD"], cwd=source)
    if actual != REVISION:
        raise RuntimeError(
            f"Source revision is {actual}, expected {REVISION}: {source}. "
            "Use a new --asset-dir or inspect the checkout; existing files were not reset."
        )
    if run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source):
        raise RuntimeError(f"Tracked upstream source files were modified: {source}")
    print(f"Verified SyncNet source {REVISION}", flush=True)
    return source


def link_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.is_file() and sha256(destination) == sha256(source):
            return
        raise RuntimeError(f"Refusing to overwrite an existing asset path: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def ensure_dependencies(skip_install: bool) -> dict[str, str]:
    core = {"torch": "torch", "numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python"}
    missing_core = [module for module in core if importlib.util.find_spec(module) is None]
    if missing_core:
        raise RuntimeError("Use the existing aligndit environment; missing core modules: " + ", ".join(missing_core))
    # --no-deps prevents pip from replacing torch/numpy/OpenCV in this shared environment.
    optional = {
        "python_speech_features": ("python-speech-features", "0.6"),
        "tqdm": ("tqdm", "4.67.3"),
        "click": ("click", "8.4.1"),
        "platformdirs": ("platformdirs", "4.10.0"),
        "scenedetect": ("scenedetect", "0.6.7.1"),
    }
    missing = [
        f"{package}=={version}"
        for module, (package, version) in optional.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        if skip_install:
            raise RuntimeError("Missing dependencies: " + " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *missing], check=True)
        importlib.invalidate_caches()
    modules = {**core, **{module: package for module, (package, _) in optional.items()}}
    versions = {}
    for module, distribution in modules.items():
        imported = importlib.import_module(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(imported, "__version__", "unknown")
        versions[module] = version
    if versions["scenedetect"] != "0.6.7.1":
        raise RuntimeError(
            "Pinned pipeline requires scenedetect==0.6.7.1; found "
            + versions["scenedetect"]
            + ". No existing package was changed. Select the matching environment."
        )
    return versions


def resolve_binary(name: str, explicit: str | None) -> dict[str, str]:
    candidates = [explicit] if explicit else [str(Path(sys.executable).parent / name), shutil.which(name)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            executable = str(Path(candidate).resolve())
            return {"path": executable, "version": run([executable, "-version"]).splitlines()[0]}
    raise RuntimeError(f"Cannot find {name}; install ffmpeg into this environment or provide --{name}.")


def verify_models(source: Path, asset_dir: Path) -> dict:
    import torch

    sys.path.insert(0, str(source))
    from detectors.s3fd.nets import S3FDNet
    from SyncNetModel import S

    syncnet = S()
    syncnet.load_state_dict(
        torch.load(asset_dir / "syncnet_v2.model", map_location="cpu", weights_only=True), strict=True
    )
    syncnet.eval()
    detector = S3FDNet(device="cpu")
    detector.load_state_dict(torch.load(asset_dir / "sfd_face.pth", map_location="cpu", weights_only=True), strict=True)
    detector.eval()
    print("Both checkpoints loaded successfully on CPU with strict state-dict validation.", flush=True)
    return {"device": "cpu", "weights_only": True, "strict_state_dict": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=default_asset_dir())
    parser.add_argument("--with-example", action="store_true", help="Also download the official demo for validation.")
    parser.add_argument("--skip-dependency-install", action="store_true", help="Check dependencies without pip writes.")
    parser.add_argument(
        "--ffmpeg", help="Explicit ffmpeg executable; otherwise use this Python's bin directory or PATH."
    )
    parser.add_argument(
        "--ffprobe", help="Explicit ffprobe executable; otherwise use this Python's bin directory or PATH."
    )
    args = parser.parse_args()
    asset_dir = args.asset_dir.expanduser().resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    dependencies = ensure_dependencies(args.skip_dependency_install)
    ffmpeg = resolve_binary("ffmpeg", args.ffmpeg)
    ffprobe = resolve_binary("ffprobe", args.ffprobe)
    source = install_source(asset_dir)
    names = ["syncnet_v2.model", "sfd_face.pth"] + (["example.avi"] if args.with_example else [])
    assets = {name: download(name, asset_dir) for name in names}
    link_asset(asset_dir / "syncnet_v2.model", source / "data" / "syncnet_v2.model")
    link_asset(asset_dir / "sfd_face.pth", source / "detectors" / "s3fd" / "weights" / "sfd_face.pth")
    if args.with_example:
        link_asset(asset_dir / "example.avi", source / "data" / "example.avi")
    verification = verify_models(source, asset_dir)
    metadata = {
        "schema_version": 1,
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"repository": REPOSITORY, "revision": REVISION, "path": str(source)},
        "models": assets,
        "python": {"executable": sys.executable, "version": sys.version},
        "dependencies": dependencies,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "verification": verification,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="installation.", suffix=".tmp", dir=asset_dir, delete=False
    ) as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, asset_dir / "installation.json")
    print(f"Installation verified: {asset_dir / 'installation.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
