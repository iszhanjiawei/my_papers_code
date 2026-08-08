"""Extract CelebV-Dub Semantic-VAE latents with the audited base cache protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from aligndit.script.misc import extract_librispeech_svae_latents as base


GOLDEN_UTTERANCE_KEY = "celebvdub/train/saP2eLOlPAc/0_0_0"
GOLDEN_POSTERIOR_SEED = 3_721_164_253_472_998_262
GOLDEN_RAW_LATENT_SHA256 = "7e3bd4e044b7f0a4f1d0295ece831e639a1e7abd78f13fd3ce85e3e1b9feccce"
GOLDEN_LATENT_FRAMES = 947
BASE_ENGINE_PATH = Path(base.__file__).resolve(strict=True)
BASE_BUILD_AND_VERIFY_SPEC = base.build_and_verify_spec
BASE_VALIDATE_STORED_SPEC_RESOURCES = base.validate_stored_spec_resources
GOLDEN_RECORD: dict[str, Any] = {
    "audio_relative_path": "train/saP2eLOlPAc/0_0_0.wav",
    "latent_frames": GOLDEN_LATENT_FRAMES,
    "num_channels": 1,
    "original_num_samples": 378_579,
    "padded_num_samples": 378_800,
    "sample_rate": base.SAMPLE_RATE,
    "source_num_channels": 1,
    "source_num_samples": 378_579,
    "source_sample_rate": base.SAMPLE_RATE,
    "utterance_key": GOLDEN_UTTERANCE_KEY,
}


def build_and_verify_spec(
    args: Any,
    context: base.DistributedContext,
) -> dict[str, Any]:
    """Bind both this dataset adapter and the delegated extraction engine."""

    spec = BASE_BUILD_AND_VERIFY_SPEC(args, context)
    spec["extraction"]["code"].update(
        {
            "base_engine_path": str(BASE_ENGINE_PATH),
            "base_engine_sha256": base.sha256_file(BASE_ENGINE_PATH),
        }
    )
    return spec


def validate_stored_spec_resources(args: Any, spec: Mapping[str, Any]) -> None:
    BASE_VALIDATE_STORED_SPEC_RESOURCES(args, spec)
    extraction_code = spec.get("extraction", {}).get("code", {})
    stored_path = Path(extraction_code.get("base_engine_path", ""))
    if stored_path != BASE_ENGINE_PATH:
        raise RuntimeError(f"Stored base extractor path changed: {stored_path} != {BASE_ENGINE_PATH}")
    expected_hash = extraction_code.get("base_engine_sha256")
    actual_hash = base.sha256_file(BASE_ENGINE_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Stored base extractor changed: {BASE_ENGINE_PATH}: expected {expected_hash}, got {actual_hash}"
        )


def load_and_validate_waveform(
    record: Mapping[str, Any],
    dataset_root: Path,
    device: torch.device,
) -> torch.Tensor:
    """Decode the recorded source, then apply the exact CelebV-Dub 16-kHz contract."""

    audio_path = base.safe_join(dataset_root, record["audio_relative_path"])
    waveform, decoded_sample_rate = torchaudio.load(audio_path)
    expected_source_shape = (int(record["source_num_channels"]), int(record["source_num_samples"]))
    if decoded_sample_rate != int(record["source_sample_rate"]):
        raise ValueError(
            f"Decoded source sample rate mismatch for {audio_path}: "
            f"expected {record['source_sample_rate']}, got {decoded_sample_rate}"
        )
    if waveform.ndim != 2 or tuple(waveform.shape) != expected_source_shape:
        raise ValueError(
            f"Decoded source shape mismatch for {audio_path}: expected {expected_source_shape}, "
            f"got {tuple(waveform.shape)}"
        )
    if not torch.isfinite(waveform).all():
        raise FloatingPointError(f"Decoded non-finite source waveform: {audio_path}")

    waveform = waveform.to(dtype=torch.float32).mean(dim=0, keepdim=True)
    if decoded_sample_rate != base.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, decoded_sample_rate, base.SAMPLE_RATE)
    expected_target_shape = (int(record["num_channels"]), int(record["original_num_samples"]))
    if int(record["sample_rate"]) != base.SAMPLE_RATE or tuple(waveform.shape) != expected_target_shape:
        raise ValueError(
            f"16-kHz target contract mismatch for {audio_path}: expected {expected_target_shape} at "
            f"{base.SAMPLE_RATE} Hz, got {tuple(waveform.shape)} at {record['sample_rate']} Hz"
        )
    if not torch.isfinite(waveform).all():
        raise FloatingPointError(f"Non-finite waveform after CelebV-Dub preprocessing: {audio_path}")

    waveform = waveform.unsqueeze(0)
    right_pad = int(record["padded_num_samples"]) - waveform.shape[-1]
    if not 0 <= right_pad < base.SEMANTIC_VAE_HOP_LENGTH:
        raise ValueError(f"Invalid right padding for {record['utterance_key']}: {right_pad}")
    return F.pad(waveform, (0, right_pad)).to(device=device, dtype=torch.float32, non_blocking=True)


def run_golden_self_test(model: base.SemanticVaePosterior, dataset_root: Path, device: torch.device) -> None:
    waveform = load_and_validate_waveform(GOLDEN_RECORD, dataset_root, device)
    latent = base.extract_latent(
        model,
        waveform,
        GOLDEN_POSTERIOR_SEED,
        GOLDEN_LATENT_FRAMES,
        device,
    )
    if latent.shape != (GOLDEN_LATENT_FRAMES, base.SEMANTIC_VAE_LATENT_DIM) or latent.dtype != np.float32:
        raise RuntimeError(f"Unexpected CelebV-Dub golden latent contract: {latent.shape}/{latent.dtype}")
    actual_hash = hashlib.sha256(latent.tobytes(order="C")).hexdigest()
    if actual_hash != GOLDEN_RAW_LATENT_SHA256:
        raise RuntimeError(
            f"Semantic-VAE CelebV-Dub golden self-test failed: expected {GOLDEN_RAW_LATENT_SHA256}, got {actual_hash}"
        )


def main() -> None:
    # Reuse the battle-tested atomic/resume/DDP implementation while binding the
    # immutable spec to this CelebV-Dub wrapper and its waveform/golden contract.
    base.__file__ = __file__
    base.GOLDEN_UTTERANCE_KEY = GOLDEN_UTTERANCE_KEY
    base.GOLDEN_POSTERIOR_SEED = GOLDEN_POSTERIOR_SEED
    base.GOLDEN_RAW_LATENT_SHA256 = GOLDEN_RAW_LATENT_SHA256
    base.build_and_verify_spec = build_and_verify_spec
    base.load_and_validate_waveform = load_and_validate_waveform
    base.run_golden_self_test = run_golden_self_test
    base.validate_stored_spec_resources = validate_stored_spec_resources
    base.main()


if __name__ == "__main__":
    main()
