"""Frozen acoustic-only wrapper for the released MingTok-Audio VAE.

This module intentionally imports only ``Encoder`` and ``Decoder`` from the
official ``audio_tokenizer/vae_modules.py``.  In particular, it does not import
``modeling_audio_vae.py`` or ``audio_encoder.py``: those modules instantiate the
unused 1280-D semantic path and make FlashAttention a hard import dependency.

The public representation contract is deliberately small:

* mono 16 kHz waveform input;
* raw (un-normalized) 64-D acoustic latent at 50 Hz;
* encoder layout ``[batch, frames, 64]``;
* decoder layout ``[batch, 1, frames * 320]``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn


MINGTOK_SAMPLE_RATE = 16_000
MINGTOK_HOP_SIZE = 320
MINGTOK_LATENT_DIM = 64
MINGTOK_LATENT_FPS = 50

EXPECTED_CONFIG_SHA256 = "e65fa0aec76f058308f75a4f7f892d8bdb3a3a7d79116b2b163f36b00d118c4a"
EXPECTED_MODEL_SHA256 = "c36d876de086d13eb1cdcfb9d08e22c3d806cd7893d64fdaf7ea6d30b7d521cd"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _checkpoint_contract_cached(checkpoint_dir_string: str) -> dict[str, str]:
    checkpoint_dir = Path(checkpoint_dir_string)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"MingTok config not found: {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"MingTok weights not found: {model_path}")

    config_sha256 = _sha256_file(config_path)
    model_sha256 = _sha256_file(model_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "MingTok config SHA256 mismatch: "
            f"expected {EXPECTED_CONFIG_SHA256}, got {config_sha256} ({config_path})"
        )
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "MingTok model SHA256 mismatch: "
            f"expected {EXPECTED_MODEL_SHA256}, got {model_sha256} ({model_path})"
        )
    return {
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
    }


def checkpoint_contract(checkpoint_dir: str | Path) -> dict[str, str]:
    """Verify and return the immutable local MingTok checkpoint contract."""

    resolved = str(Path(checkpoint_dir).expanduser().resolve())
    return dict(_checkpoint_contract_cached(resolved))


def stable_sample_seed(sample_key: str, base_seed: int = 666) -> int:
    """Derive a process-independent posterior seed from a stable sample key."""

    payload = f"mingtok-audio-v1\0{int(base_seed)}\0{sample_key}".encode()
    # torch.Generator.manual_seed accepts signed 64-bit values on every backend.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        result = dtype
    else:
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            result = aliases[str(dtype).lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported MingTok dtype: {dtype!r}") from error
    if result not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported MingTok dtype: {result}")
    return result


def _load_official_vae_modules(repo_path: Path):
    """Load the official vae_modules under a private package name.

    A private package avoids accidentally resolving another checkout already
    imported as ``audio_tokenizer`` while retaining the official relative
    import of ``.istft``.
    """

    tokenizer_dir = repo_path / "audio_tokenizer"
    module_path = tokenizer_dir / "vae_modules.py"
    istft_path = tokenizer_dir / "istft.py"
    if not module_path.is_file() or not istft_path.is_file():
        raise FileNotFoundError(
            "repo_path must point to MingTok-Audio and contain "
            f"audio_tokenizer/vae_modules.py: {repo_path}"
        )

    suffix = hashlib.sha256(str(repo_path.resolve()).encode("utf-8")).hexdigest()[:12]
    package_name = f"_aligndit_mingtok_{suffix}"
    module_name = f"{package_name}.vae_modules"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package = types.ModuleType(package_name)
    package.__file__ = str(tokenizer_dir)
    package.__path__ = [str(tokenizer_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load official MingTok module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise
    return module


def _with_attention_backend(backbone: dict, backend: str) -> dict:
    if backend not in {"eager", "sdpa"}:
        raise ValueError("backend must be 'eager' or 'sdpa'")
    updated = dict(backbone)
    # Transformers has used both spellings across supported releases.  Setting
    # both makes the choice explicit and prevents the checkpoint's FA2 request
    # from leaking through.
    updated["_attn_implementation"] = backend
    updated["attn_implementation"] = backend
    return updated


def _load_prefixed_safetensors(
    module: nn.Module,
    model_path: Path,
    prefix: str,
    allowed_checkpoint_only_prefixes: Iterable[str] = (),
) -> None:
    """Strictly copy one checkpoint prefix into a module.

    Every module state key must exist with exactly the expected shape.  Extra
    keys are rejected unless they belong to an explicitly allowed, unused
    checkpoint-only path (the semantic decoder branch).
    """

    expected = set(module.state_dict().keys())
    allowed = tuple(allowed_checkpoint_only_prefixes)
    with safe_open(str(model_path), framework="pt", device="cpu") as archive:
        archive_keys = set(archive.keys())
        prefixed = {key for key in archive_keys if key.startswith(prefix)}
        selected = {
            key[len(prefix) :]
            for key in prefixed
            if not any(key.startswith(extra_prefix) for extra_prefix in allowed)
        }
        missing = sorted(expected - selected)
        unexpected = sorted(selected - expected)
        rejected_extra = sorted(
            key
            for key in prefixed
            if key[len(prefix) :] not in expected and not any(key.startswith(extra) for extra in allowed)
        )
        if missing or unexpected or rejected_extra:
            raise RuntimeError(
                f"Strict MingTok load failed for prefix {prefix!r}: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"rejected_extra={rejected_extra[:8]}"
            )

        target_state = module.state_dict()
        with torch.no_grad():
            for local_key in sorted(expected):
                source_key = prefix + local_key
                source = archive.get_tensor(source_key)
                target = target_state[local_key]
                if tuple(source.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {source_key}: checkpoint={tuple(source.shape)}, "
                        f"module={tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))


class MingTokAcousticCodec(nn.Module):
    """Frozen 64-D/50-Hz MingTok acoustic encoder and/or decoder."""

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        backend: str = "eager",
        load_encoder: bool = True,
        load_decoder: bool = True,
    ) -> None:
        super().__init__()
        if not load_encoder and not load_decoder:
            raise ValueError("At least one of load_encoder/load_decoder must be True")

        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)
        self.backend = backend
        if self.device.type == "cpu" and self.dtype != torch.float32:
            raise ValueError("CPU MingTok execution requires dtype=float32")

        self.checkpoint = checkpoint_contract(self.checkpoint_dir)
        config_path = self.checkpoint_dir / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.config = config
        self._validate_config(config)

        official = _load_official_vae_modules(self.repo_path)
        Encoder = official.Encoder
        Decoder = official.Decoder
        model_path = self.checkpoint_dir / "model.safetensors"

        self.encoder: nn.Module | None = None
        self.decoder: nn.Module | None = None
        if load_encoder:
            enc_kwargs = config["enc_kwargs"]
            self.encoder = Encoder(
                encoder_args=_with_attention_backend(enc_kwargs["backbone"], backend),
                input_dim=enc_kwargs["input_dim"],
                hop_size=enc_kwargs["hop_size"],
                latent_dim=enc_kwargs["latent_dim"],
                patch_size=config["patch_size"],
            ).to(device=self.device, dtype=self.dtype)
            _load_prefixed_safetensors(self.encoder, model_path, "encoder.")

        if load_decoder:
            dec_kwargs = config["dec_kwargs"]
            self.decoder = Decoder(
                decoder_args=_with_attention_backend(dec_kwargs["backbone"], backend),
                output_dim=dec_kwargs["output_dim"],
                latent_dim=dec_kwargs["latent_dim"],
                semantic_model=None,
                patch_size=config["patch_size"],
            ).to(device=self.device, dtype=self.dtype)
            _load_prefixed_safetensors(
                self.decoder,
                model_path,
                "decoder.",
                allowed_checkpoint_only_prefixes=(
                    "decoder.semantic_model.",
                    "decoder.fc2.",
                    "decoder.fc3.",
                ),
            )

        self.requires_grad_(False)
        super().train(False)

    @staticmethod
    def _validate_config(config: dict) -> None:
        enc = config.get("enc_kwargs") or {}
        dec = config.get("dec_kwargs") or {}
        observed = {
            # The released JSON omits this default; AudioVAEconfig defines it
            # as 16 kHz when absent.
            "sample_rate": config.get("sample_rate", MINGTOK_SAMPLE_RATE),
            "input_dim": enc.get("input_dim"),
            "hop_size": enc.get("hop_size"),
            "encoder_latent_dim": enc.get("latent_dim"),
            "decoder_latent_dim": dec.get("latent_dim"),
            "output_dim": dec.get("output_dim"),
            "patch_size": config.get("patch_size"),
        }
        expected = {
            "sample_rate": MINGTOK_SAMPLE_RATE,
            "input_dim": MINGTOK_HOP_SIZE,
            "hop_size": MINGTOK_HOP_SIZE,
            "encoder_latent_dim": MINGTOK_LATENT_DIM,
            "decoder_latent_dim": MINGTOK_LATENT_DIM,
            "output_dim": MINGTOK_HOP_SIZE,
            "patch_size": -1,
        }
        if observed != expected:
            raise RuntimeError(f"Unsupported MingTok acoustic config: expected {expected}, got {observed}")
        if MINGTOK_SAMPLE_RATE // MINGTOK_HOP_SIZE != MINGTOK_LATENT_FPS:
            raise AssertionError("Internal MingTok 50-Hz contract is inconsistent")

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def train(self, mode: bool = True):
        """Keep the frozen codec in eval mode even when a parent module trains."""

        return super().train(False)

    @torch.inference_mode()
    def encode(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor | Sequence[int],
        mode: str = "sample",
        seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mono waveforms to raw ``[B,T,64]`` MingTok latents.

        ``mode='sample'`` reproduces the official Oobleck posterior:
        ``mean + (softplus(scale) + 1e-4) * epsilon``.  Passing one seed per
        sample makes posterior sampling independent of rank and batch order.
        ``mode='mean'`` is deterministic and intended only for ablations.
        """

        if self.encoder is None:
            raise RuntimeError("This MingTokAcousticCodec was created without an encoder")
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError(f"waveform must have shape [B,N] or [B,1,N], got {tuple(waveform.shape)}")
        batch_size, padded_samples = waveform.shape
        lengths_tensor = torch.as_tensor(lengths, device=self.device, dtype=torch.long).reshape(-1)
        if lengths_tensor.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} waveform lengths, got {lengths_tensor.numel()}")
        if torch.any(lengths_tensor <= 0) or torch.any(lengths_tensor > padded_samples):
            raise ValueError(
                f"Waveform lengths must be in [1,{padded_samples}], got {lengths_tensor.detach().cpu().tolist()}"
            )
        if mode not in {"sample", "mean"}:
            raise ValueError("mode must be 'sample' or 'mean'")

        waveform = waveform.to(device=self.device, dtype=self.dtype)
        with self._autocast():
            moments, _ = self.encoder(waveform)
        moments = moments.transpose(1, 2)  # [B,128,T], matching official code
        mean, scale = moments.chunk(2, dim=1)

        if mode == "mean":
            latent = mean
        else:
            std = F.softplus(scale) + 1e-4
            if seeds is None:
                noise = torch.randn_like(mean)
            else:
                seed_values = torch.as_tensor(seeds, dtype=torch.long).reshape(-1).tolist()
                if len(seed_values) != batch_size:
                    raise ValueError(f"Expected {batch_size} posterior seeds, got {len(seed_values)}")
                noise_parts = []
                for index, seed in enumerate(seed_values):
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(int(seed))
                    noise_parts.append(
                        torch.randn(
                            mean[index].shape,
                            generator=generator,
                            device=self.device,
                            dtype=mean.dtype,
                        )
                    )
                noise = torch.stack(noise_parts, dim=0)
            latent = mean + std * noise

        latent = latent.transpose(1, 2).contiguous()  # [B,T,64]
        frame_lengths = torch.div(
            lengths_tensor + MINGTOK_HOP_SIZE - 1,
            MINGTOK_HOP_SIZE,
            rounding_mode="floor",
        )
        if latent.shape[-1] != MINGTOK_LATENT_DIM:
            raise RuntimeError(f"MingTok returned latent dim {latent.shape[-1]}, expected 64")
        if int(frame_lengths.max()) > latent.shape[1]:
            raise RuntimeError(
                f"MingTok frame length {int(frame_lengths.max())} exceeds tensor length {latent.shape[1]}"
            )
        return latent, frame_lengths

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode raw ``[B,T,64]`` latents to ``[B,1,T*320]`` waveform."""

        if self.decoder is None:
            raise RuntimeError("This MingTokAcousticCodec was created without a decoder")
        if latent.ndim != 3 or latent.shape[-1] != MINGTOK_LATENT_DIM:
            raise ValueError(f"latent must have shape [B,T,64], got {tuple(latent.shape)}")
        frames = latent.shape[1]
        if frames <= 0:
            raise ValueError("latent must contain at least one frame")
        latent = latent.to(device=self.device, dtype=self.dtype)
        with self._autocast():
            waveform = self.decoder.low_level_reconstruct(latent)
        expected_shape = (latent.shape[0], 1, frames * MINGTOK_HOP_SIZE)
        if tuple(waveform.shape) != expected_shape:
            raise RuntimeError(f"MingTok decoder returned {tuple(waveform.shape)}, expected {expected_shape}")
        return waveform


__all__ = [
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_MODEL_SHA256",
    "MINGTOK_HOP_SIZE",
    "MINGTOK_LATENT_DIM",
    "MINGTOK_LATENT_FPS",
    "MINGTOK_SAMPLE_RATE",
    "MingTokAcousticCodec",
    "checkpoint_contract",
    "stable_sample_seed",
]
