"""Frozen HunyuanVideo-Foley Synchformer conditioning and validated caches.

RGB clips are resampled to 25 Hz, spatially transformed exactly as Foley, then
encoded in 16-frame windows with stride 8. Eight 768-D vectors per window are
flattened in the same order as Foley. Clips shorter than 16 frames repeat their
last frame. Longer clips retain Foley's complete-window rule (up to seven final
frames are outside a window); there is no 15-second truncation.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor, nn

MODEL_ID = "tencent/HunyuanVideo-Foley/synchformer"
CHECKPOINT_SHA256 = "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c"
SCHEMA_VERSION = 1
PREPROCESSING = {
    "fps": 25, "sampling": "first_frame_at_or_after_grid_time_relative_to_first_pts",
    "color": "RGB", "resize_short_side": 224, "interpolation": "bicubic",
    "antialias": True, "center_crop": 224, "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5], "segment_frames": 16, "stride_frames": 8,
    "tokens_per_segment": 8, "feature_dim": 768, "short_clip": "repeat_last_to_16",
    "tail": "complete_windows_only", "duration_limit_seconds": None,
}



@lru_cache(maxsize=1)
def _malloc_trim_function():
    """glibc trim is optional; other platforms still collect native cycles."""
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def release_extraction_host_memory() -> None:
    """Collect released PyAV cycles, then return freed glibc arenas to the OS.

    Successful extraction has released its frame/CPU tensor locals by this point.
    During an error, traceback-owned locals survive until the caller handles it;
    this still reclaims older unreachable allocations. Active tensors and CUDA
    weights are never released by trimming.
    """
    gc.collect()
    trim = _malloc_trim_function()
    if trim is not None:
        trim(0)


@contextmanager
def decoded_video_frames(video_path):
    """Close the decoder explicitly: PyAV 11 Container.close leaves it open.

    See https://github.com/PyAV-Org/PyAV/issues/1117. Closing the container alone
    defers codec buffers to cyclic garbage collection across thousands of clips.
    """
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        stream.thread_type = "AUTO"
        codec.thread_count = 2
        decoder = container.decode(stream)
        try:
            yield decoder
        finally:
            try:
                decoder.close()
            finally:
                if codec.is_open:
                    codec.close()


def default_checkpoint_path() -> Path:
    return Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects/data/pretrained_models/HunyuanVideo-Foley/synchformer_state_dict.pth")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_clip_key(value: str) -> str:
    value = str(value).replace("\\", "/")
    if value.startswith("celebvdub/"):
        value = value[len("celebvdub/"):]
    path = PurePosixPath(value)
    if path.suffix in {".mp4", ".wav", ".npy", ".pt"}:
        path = path.with_suffix("")
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError(f"Expected full relative split/video/clip key, got {value!r}")
    return str(path)


def cache_path(cache_dir: str | Path, clip_key: str) -> Path:
    return Path(cache_dir) / (canonical_clip_key(clip_key) + ".pt")


def source_identity(video_path: str | Path, clip_key: str) -> dict:
    stat = Path(video_path).stat()  # Follow dataset symlinks intentionally.
    return {"relative_path": canonical_clip_key(clip_key) + ".mp4",
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_synchformer_feature(cache_dir: str | Path, clip_key: str, payload: dict) -> Path:
    path = cache_path(cache_dir, clip_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def load_synchformer_payload(cache_dir: str | Path, clip_key: str,
                              expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
                              expected_dim: int = 768, video_path: str | Path | None = None) -> dict:
    key = canonical_clip_key(clip_key)
    path = cache_path(cache_dir, key)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Invalid Synchformer cache payload: {path}")
    features, meta = payload.get("features"), payload["metadata"]
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] != expected_dim or features.shape[0] < 8 or features.shape[0] % 8 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"Invalid Synchformer features: {path}")
    expected = {"schema_version": SCHEMA_VERSION, "clip_key": key, "model_id": MODEL_ID,
                "checkpoint_sha256": expected_checkpoint_sha256, "preprocessing": PREPROCESSING}
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"Synchformer {field} mismatch in {path}")
    frames = meta.get("num_sampled_frames", 0)
    segments = max(1, (frames - 16) // 8 + 1)
    if frames < 1 or features.shape[0] != segments * 8 or meta.get("segment_start_frames") != list(range(0, segments * 8, 8)):
        raise ValueError(f"Invalid Synchformer frame/segment metadata in {path}")
    if len(meta.get("sample_timestamps_seconds", [])) != frames or len(meta.get("source_timestamps_seconds", [])) != frames:
        raise ValueError(f"Invalid Synchformer timestamp metadata in {path}")
    sample_times, source_times = meta["sample_timestamps_seconds"], meta["source_timestamps_seconds"]
    origin = meta.get("source_first_pts_seconds", float("nan"))
    if not math.isfinite(origin) or any(not math.isfinite(t) or abs(t - index / 25) > 1e-7 for index, t in enumerate(sample_times)) or any(not math.isfinite(t) or t + 1e-7 < origin + sample_times[index] for index, t in enumerate(source_times)) or any(b < a for a, b in zip(source_times, source_times[1:])):
        raise ValueError(f"Invalid Synchformer timestamp grid in {path}")
    if abs(meta.get("duration_seconds", 0) - frames / 25) > 1e-7:
        raise ValueError(f"Synchformer frame/duration mismatch in {path}")
    if meta.get("source", {}).get("relative_path") != key + ".mp4":
        raise ValueError(f"Source identity mismatch in {path}")
    if not math.isfinite(meta.get("duration_seconds", float("nan"))) or meta["duration_seconds"] <= 0:
        raise ValueError(f"Invalid duration in {path}")
    if video_path is not None and meta["source"] != source_identity(video_path, key):
        raise ValueError(f"Source video changed since Synchformer extraction: {video_path}")
    return payload


def load_synchformer_feature(cache_dir: str | Path, clip_key: str,
                              expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
                              expected_dim: int = 768) -> Tensor:
    return load_synchformer_payload(cache_dir, clip_key, expected_checkpoint_sha256, expected_dim)["features"].float()


def read_inventory(path: str | Path) -> list[dict]:
    records = []
    keys = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = canonical_clip_key((record.get("utterance_key") or record["audio_relative_path"]))
            if key in keys:
                raise ValueError(f"Duplicate clip key in inventory: {key}")
            keys.add(key)
            records.append(record)
    return records


def validate_synchformer_cache(cache_dir: str | Path, expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
                                records: list[dict] | None = None, expected_keys: list[str] | None = None,
                                expected_manifest_sha256: str | None = None) -> dict:
    """Fast startup gate based on a complete full-file audit, then key existence.

    Extraction's `--audit-only` produces coverage_report.json after checking all
    tensors and source fingerprints. Every loaded sample is independently checked
    again by load_synchformer_feature; an audit is tied to its inventory SHA256.
    """
    report_path = Path(cache_dir) / "coverage_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run Synchformer full coverage audit first: {report_path}")
    report = json.loads(report_path.read_text())
    if not report.get("complete") or report.get("invalid", 0) or report.get("missing", 0):
        raise ValueError(f"Synchformer coverage audit is incomplete: {report_path}")
    for field, value in {"schema_version": SCHEMA_VERSION, "model_id": MODEL_ID, "checkpoint_sha256": expected_checkpoint_sha256, "preprocessing": PREPROCESSING}.items():
        if report.get(field) != value:
            raise ValueError(f"Synchformer coverage {field} mismatch: {report_path}")
    audited = report.get("valid_keys", [])
    if not isinstance(audited, list) or len(set(audited)) != len(audited) or report.get("valid") != len(audited) or report.get("expected_count") != len(audited) or not audited:
        raise ValueError(f"Synchformer coverage count/keyset mismatch: {report_path}")
    if expected_manifest_sha256 and report.get("inventory_sha256") != expected_manifest_sha256:
        raise ValueError(f"Synchformer inventory SHA256 mismatch: {report_path}")
    keys = expected_keys
    if records is not None:
        keys = [canonical_clip_key((record.get("utterance_key") or record["audio_relative_path"])) for record in records]
    if keys is not None:
        keys = [canonical_clip_key(key) for key in keys]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate expected Synchformer keys")
        audited_keys = set(report.get("valid_keys", []))
        missing = [key for key in keys if key not in audited_keys or not cache_path(cache_dir, key).is_file()]
        if missing:
            raise ValueError(f"Missing {len(missing)} audited Synchformer caches; first: {missing[:3]}")
    return report


class FrozenSynchformerExtractor(nn.Module):
    """Official visual branch with strict checkpoint loading, permanently frozen."""
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | torch.device = "cuda",
                 batch_size: int = 8, expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
                 verify_checkpoint: bool = True, cleanup_interval: int = 16):
        super().__init__()
        from aligndit.third_party.synchformer.motionformer import MotionFormer
        from torchvision.transforms import v2
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
        self.checkpoint_sha256 = expected_checkpoint_sha256
        if verify_checkpoint:
            actual = sha256_file(self.checkpoint_path)
            if actual != expected_checkpoint_sha256:
                raise ValueError(f"Synchformer checkpoint SHA256 mismatch: {actual}")
        self.vfeat_extractor = MotionFormer(extract_features=True, factorize_space_time=True,
            agg_space_module="TransformerEncoderLayer", agg_time_module="torch.nn.Identity", add_global_repr=False)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        prefix = "vfeat_extractor."
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not state:
            raise ValueError("Checkpoint has no vfeat_extractor weights")
        self.vfeat_extractor.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()
        self.batch_size = int(batch_size)
        self.cleanup_interval = int(cleanup_interval)
        self._clips_since_cleanup = 0
        if self.cleanup_interval < 0:
            raise ValueError("cleanup_interval must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, segments: Tensor) -> Tensor:
        """[B,16,3,224,224] -> [B,8,768], exact Foley visual forward."""
        if segments.ndim != 5 or tuple(segments.shape[1:]) != (16, 3, 224, 224):
            raise ValueError(f"Expected [B,16,3,224,224], got {tuple(segments.shape)}")
        vis = segments.to(self.device).unsqueeze(1).permute(0, 1, 3, 2, 4, 5)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            output = self.vfeat_extractor(vis)
        return output[:, 0]

    @torch.inference_mode()
    def extract(self, video_path: str | Path, clip_key: str | None = None) -> dict[str, Any]:
        """Extract features while bounding decoder and host allocator lifetime."""
        failed = True
        try:
            payload = self._extract_video(video_path, clip_key)
            failed = False
            return payload
        finally:
            self._clips_since_cleanup = getattr(self, "_clips_since_cleanup", 0) + 1
            interval = getattr(self, "cleanup_interval", 16)
            if failed or (interval > 0 and self._clips_since_cleanup >= interval):
                release_extraction_host_memory()
                self._clips_since_cleanup = 0

    def _extract_video(self, video_path: str | Path, clip_key: str | None = None) -> dict[str, Any]:
        """Inner scope releases decoded frames before periodic host cleanup."""
        key = canonical_clip_key(clip_key or "/".join(Path(video_path).parts[-3:]))
        identity = source_identity(video_path, key)
        frames, sample_times, source_times = [], [], []
        origin = None
        next_sample = 0
        last_pts = None
        with decoded_video_frames(video_path) as decoder:
            for frame in decoder:
                if frame.time is None:
                    raise ValueError(f"Video frame has no presentation timestamp: {video_path}")
                pts = float(frame.time)
                if last_pts is not None and pts < last_pts:
                    raise ValueError(f"Video timestamps are not monotonic: {video_path}")
                last_pts = pts
                if origin is None:
                    origin = pts
                relative_time = pts - origin
                processed = None
                while relative_time + 1e-7 >= next_sample / 25:
                    if processed is None:
                        rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                        # Spatial operations on uint8 match the official v2 pipeline.
                        processed = self.preprocess(rgb)
                    frames.append(processed)
                    sample_times.append(next_sample / 25)
                    source_times.append(pts)
                    next_sample += 1
        if not frames:
            raise ValueError(f"Video contains no decoded RGB frames: {video_path}")
        n_frames = len(frames)
        if n_frames < 16:
            frames += [frames[-1]] * (16 - n_frames)
        starts = list(range(0, len(frames) - 15, 8))
        outputs = []
        for offset in range(0, len(starts), self.batch_size):
            batch_starts = starts[offset:offset + self.batch_size]
            batch = torch.stack([torch.stack(frames[start:start + 16]) for start in batch_starts])
            outputs.append(self(batch).flatten(0, 1).to(device="cpu", dtype=torch.float16))
        features = torch.cat(outputs)
        if not torch.isfinite(features).all():
            raise ValueError(f"Synchformer produced nonfinite values: {video_path}")
        if source_identity(video_path, key) != identity:
            raise RuntimeError(f"Source changed during extraction: {video_path}")
        return {"features": features, "metadata": {
            "schema_version": SCHEMA_VERSION, "clip_key": key, "model_id": MODEL_ID,
            "checkpoint_sha256": self.checkpoint_sha256, "preprocessing": PREPROCESSING.copy(),
            "source": identity, "num_sampled_frames": n_frames, "duration_seconds": n_frames / 25,
            "segment_start_frames": starts, "sample_timestamps_seconds": sample_times,
            "source_timestamps_seconds": source_times, "source_first_pts_seconds": origin,
            "padded_frames": max(0, 16 - n_frames), "unwindowed_tail_frames": max(0, n_frames - (starts[-1] + 16)),
            "compute_dtype": "float16" if self.device.type == "cuda" else "float32", "storage_dtype": "float16",
        }}
