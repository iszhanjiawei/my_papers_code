"""Frozen AV-HuBERT audio-only targets for dual-role representation supervision.

This helper intentionally is not an ``nn.Module``: construct it in the trainer,
outside the generator, DDP, EMA, and optimizer. Ground-truth audio only enters
this frozen target extractor; inference does not need it or its checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)
_PREPROCESSING_VERSION = "avhubert_audio_only_pcm_logfbank26_stack4_v1"


def _import_avhubert(avhubert_root: str):
    """Locate the existing fairseq checkout and register AV-HuBERT's user dir."""
    root = Path(avhubert_root).expanduser().resolve()
    candidates = (root, root / "avhubert", root / "avhubert" / "avhubert")
    user_dir = next(
        (p for p in candidates if (p / "hubert.py").is_file() and (p / "__init__.py").is_file()),
        None,
    )
    if user_dir is None:
        raise FileNotFoundError(f"Cannot locate AV-HuBERT user module beneath {root}")

    fairseq_roots = []
    for parent in (user_dir, *list(user_dir.parents)[:4]):
        fairseq_roots.extend((parent / "fairseq", parent / "fairseq" / "fairseq"))
    fairseq_root = next((p for p in fairseq_roots if (p / "fairseq" / "__init__.py").is_file()), None)
    if fairseq_root is not None and str(fairseq_root) not in sys.path:
        sys.path.insert(0, str(fairseq_root))
    try:
        fairseq = importlib.import_module("fairseq")
    except ImportError as exc:
        raise ImportError(f"Cannot import fairseq for AV-HuBERT at {root}") from exc
    # Upstream chooses incorrect top-level sibling imports in its interactive
    # debug mode (len(sys.argv) == 1), registering some models twice. Force its
    # normal package-import branch for Python API/notebook callers, then restore
    # the caller's exact arguments. Training CLI arguments remain untouched.
    original_argv = sys.argv
    try:
        if len(original_argv) == 1:
            sys.argv = [*original_argv, "--avhubert-package-import"]
        fairseq.utils.import_user_module(SimpleNamespace(user_dir=str(user_dir)))
    finally:
        sys.argv = original_argv
    return fairseq, user_dir


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _identity_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AVHubertAudioTeacher:
    """Extract final-layer contextual features at approximately 25 Hz.

    ``avhubert_root`` may name the outer av_hubert checkout, the inner
    avhubert repository, or its Python user-module directory. Cache namespaces
    record the resolved checkpoint identity (path, byte size, modification
    time), extraction semantics and encoder source digest. Each cache entry
    additionally validates its waveform identity. Files are published by atomic
    replacement so ranks may safely compute the same missing item concurrently.
    """

    def __init__(
        self,
        checkpoint_path: str,
        avhubert_root: str,
        device,
        cache_dir: str | None = None,
        microbatch_size: int = 4,
    ):
        if microbatch_size < 1:
            raise ValueError("AV-HuBERT teacher microbatch_size must be positive")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing AV-HuBERT checkpoint: {checkpoint}")
        self.device = torch.device(device)
        self.microbatch_size = int(microbatch_size)
        # Checkpoint loading first initializes a model on CPU. Preserve the
        # generator's RNG stream despite these discarded random initial values.
        with torch.random.fork_rng(devices=[]):
            fairseq, user_dir = _import_avhubert(avhubert_root)
            models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        if len(models) != 1:
            raise ValueError("Expected exactly one AV-HuBERT teacher")
        self.model = models[0].float().eval().requires_grad_(False).to(self.device)
        self.stack_order_audio = int(task.cfg.stack_order_audio)
        self.normalize = bool(task.cfg.normalize)
        self.feature_dim = int(self.model.encoder_embed_dim)
        if self.stack_order_audio != 4 or self.feature_dim != 1024:
            raise ValueError(
                "This experiment requires the large AV-HuBERT checkpoint with "
                f"stack_order_audio=4 and feature_dim=1024; got {self.stack_order_audio}, {self.feature_dim}"
            )

        self.identity = {
            "format_version": 1,
            "checkpoint": _file_identity(checkpoint),
            "preprocessing": _PREPROCESSING_VERSION,
            "sample_rate_hz": 16000,
            "filterbank_bins": 26,
            "stack_order_audio": self.stack_order_audio,
            "normalize_per_frame": self.normalize,
            "source_modality": "audio_only",
            "output_layer": "final_contextual",
            "feature_dim": self.feature_dim,
            "cache_dtype": "float16",
            "hubert_source_sha256": hashlib.sha256((user_dir / "hubert.py").read_bytes()).hexdigest(),
        }
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve() / _identity_key(self.identity)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self.cache_dir / "teacher_metadata.json"
            if metadata_path.exists():
                with metadata_path.open(encoding="utf-8") as handle:
                    if json.load(handle) != self.identity:
                        raise ValueError(f"AV-HuBERT cache provenance mismatch: {metadata_path}")
            else:
                _atomic_json(metadata_path, self.identity)
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        logger.info(
            "Frozen audio-only AV-HuBERT teacher: dim=%d, device=%s, microbatch=%d, cache=%s",
            self.feature_dim, self.device, self.microbatch_size, self.cache_dir,
        )

    def _load_filterbank(self, audio_path: str) -> torch.Tensor:
        # Match AV-HuBERT hubert_dataset.py: retain scipy's PCM amplitude scale,
        # stack four 10 ms logfbank frames, and normalize each 104-D frame.
        from python_speech_features import logfbank
        from scipy.io import wavfile

        sample_rate, waveform = wavfile.read(audio_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Expected a nonempty mono/stereo waveform: {audio_path}")
        if sample_rate != 16000:
            import librosa

            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        features = logfbank(waveform, samplerate=16000).astype(np.float32)
        remainder = len(features) % self.stack_order_audio
        if remainder:
            features = np.pad(features, ((0, self.stack_order_audio - remainder), (0, 0)))
        features = torch.from_numpy(features.reshape(-1, self.stack_order_audio * 26))
        if self.normalize:
            features = F.layer_norm(features, (features.shape[-1],))
        if not torch.isfinite(features).all():
            raise ValueError(f"Nonfinite AV-HuBERT input features: {audio_path}")
        return features

    def _cache_path(self, audio_identity: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = _identity_key(audio_identity)
        return self.cache_dir / key[:2] / f"{key}.npz"

    def _read_cache(self, path: Path | None, audio_identity: dict) -> torch.Tensor | None:
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data["audio_identity"].item())) != audio_identity:
                    raise ValueError("waveform identity mismatch")
                if str(data["teacher_identity"].item()) != _identity_key(self.identity):
                    raise ValueError("teacher identity mismatch")
                features = data["features"]
                if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.feature_dim:
                    raise ValueError(f"unexpected feature shape {features.shape}")
                if features.dtype != np.float16 or not np.isfinite(features).all():
                    raise ValueError("invalid feature dtype/values")
                return torch.from_numpy(features.astype(np.float32))
        except (ValueError, KeyError, OSError, EOFError) as exc:
            logger.warning("Recomputing invalid AV-HuBERT cache %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path, audio_identity: dict, features: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(
                    handle,
                    features=features.numpy().astype(np.float16),
                    audio_identity=json.dumps(audio_identity, sort_keys=True),
                    teacher_identity=_identity_key(self.identity),
                )
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @torch.no_grad()
    def encode(self, audio_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded float32 ``[B,T,1024]`` targets and true ``[B]`` lengths.

        Both outputs are on the teacher device. Padding positions are zeros and
        MUST be excluded by the representation loss. Cache-enabled misses are
        rounded through float16 before return, matching subsequent cache hits.
        """
        self.model.eval()
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        if not audio_paths:
            return (
                torch.empty((0, 0, self.feature_dim), device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        identities = [_file_identity(Path(path)) for path in audio_paths]
        cache_paths = [self._cache_path(identity) for identity in identities]
        targets = [self._read_cache(path, identity) for path, identity in zip(cache_paths, identities)]
        missing = [index for index, target in enumerate(targets) if target is None]
        self.last_cache_misses = len(missing)
        self.last_cache_hits = len(targets) - len(missing)

        # A caller may use bf16 autocast for its generator; never inherit that
        # context for the frozen target encoder or its preprocessing.
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(missing), self.microbatch_size):
                indices = missing[start : start + self.microbatch_size]
                inputs = [self._load_filterbank(audio_paths[index]) for index in indices]
                lengths = torch.tensor([len(item) for item in inputs], device=self.device, dtype=torch.long)
                features = pad_sequence(inputs, batch_first=True).to(self.device)
                padding_mask = torch.arange(features.shape[1], device=self.device)[None, :] >= lengths[:, None]
                encoded, output_padding = self.model.extract_finetune(
                    source={"audio": features.transpose(1, 2).contiguous(), "video": None},
                    padding_mask=padding_mask,
                    mask=False,
                    output_layer=None,
                )
                if encoded.shape != (len(indices), features.shape[1], self.feature_dim):
                    raise RuntimeError(f"Unexpected AV-HuBERT teacher output shape: {encoded.shape}")
                if output_padding is None or not torch.equal(output_padding, padding_mask):
                    raise RuntimeError("AV-HuBERT teacher changed the expected padding mask")
                for offset, index in enumerate(indices):
                    target = encoded[offset, : lengths[offset].item()].float().cpu()
                    if not torch.isfinite(target).all():
                        raise ValueError(f"Nonfinite AV-HuBERT teacher output: {audio_paths[index]}")
                    if cache_paths[index] is not None:
                        target = target.half().float()
                        if not torch.isfinite(target).all():
                            raise ValueError(f"AV-HuBERT teacher output overflows fp16: {audio_paths[index]}")
                        self._write_cache(cache_paths[index], identities[index], target)
                    targets[index] = target

        lengths = torch.tensor([len(target) for target in targets], device=self.device, dtype=torch.long)
        return pad_sequence(targets, batch_first=True).to(self.device), lengths
