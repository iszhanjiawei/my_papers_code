"""Serialized identity of WavLM TRA; validation never opens the teacher cache.

TRA changes the training objective without changing the inference graph. Thus
matching tensor keys alone cannot distinguish constant REPA from TRA, or two
different TRA schedules. Every checkpoint embeds this contract and inference
also requires an agreeing, resolved training sidecar.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WAVLM_MODEL_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WAVLM_CHECKPOINT_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
CHECKPOINT_CONTRACT_KEY = "wavlm_tra_contract"
TRA_SCHEDULE = "taro_clean_fraction"


def _object(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _positive_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _require_values(actual: Mapping, expected: Mapping, label: str) -> None:
    for key, expected_value in expected.items():
        value = actual.get(key)
        if type(value) is not type(expected_value) or value != expected_value:
            raise RuntimeError(f"{label} mismatch: {key}={value!r}, expected {expected_value!r}")


def _require_exact(actual, expected: dict, label: str) -> None:
    _object(actual, label)
    # JSON equality preserves bool/int identity, unlike Python dictionary ==.
    if json.dumps(dict(actual), sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise RuntimeError(f"{label} mismatch: checkpoint, config and training sidecar must describe the same WavLM TRA")


def build_wavlm_tra_contract(model_cfg) -> dict:
    """Build objective identity from the full resolved experiment config.

    All teacher information comes from pinned config values. Inference needs
    neither cached training features nor the frozen WavLM model installation.
    """
    config = _object(model_cfg, "experiment config")
    model = _object(config.get("model"), "model config")
    arch = _object(model.get("arch"), "architecture config")
    datasets = _object(config.get("datasets"), "datasets config")
    coefficient = _positive_float(model.get("repa_lambda"), "model.repa_lambda")
    epsilon = _positive_float(model.get("repa_schedule_eps"), "model.repa_schedule_eps")
    if model.get("repa_schedule") != TRA_SCHEDULE:
        raise RuntimeError(f"WavLM TRA requires model.repa_schedule={TRA_SCHEDULE!r}")
    _require_values(arch, {
        "dim": 768, "depth": 18, "n_mm_layers": 12, "n_text_layers": 12,
        "text_joint_attention": True, "temporal_band_enabled": True,
        "temporal_band_mode": "fixed", "temporal_band_bidirectional": True,
        "temporal_band_fixed_window_radius_frames": 4,
        "repa_layer": 9, "repa_target_dim": 768, "repa_projector_dim": 2048,
    }, "WavLM TRA architecture")
    _require_values(datasets, {
        "repa_feature_dim": 768, "repa_teacher_layer": 12,
        "repa_model_id": WAVLM_MODEL_ID, "repa_model_revision": WAVLM_MODEL_REVISION,
        "repa_checkpoint_sha256": WAVLM_CHECKPOINT_SHA256,
    }, "WavLM TRA teacher")
    return {
        "schema_version": 1,
        "method": "wavlm_timestep_adaptive_representation_alignment",
        "lambda": coefficient,
        "schedule": TRA_SCHEDULE,
        "schedule_epsilon": epsilon,
        "schedule_formula": "sigmoid(log(alpha/(sigma+eps))) = alpha/(alpha+sigma+eps)",
        "interpolant": "x_t=(1-t)*noise+t*clean; alpha=t; sigma=1-t",
        "implemented_weight": "t/(1+eps)",
        "time_direction": "t=0 noise, t=1 clean; stronger alignment towards clean audio",
        "time_source": "the same per-sample diffusion timestep used by flow matching",
        "optimization_step_warmup": False,
        "weight_normalization": False,
        "reduction": "sum_b w(t_b)*sum_valid_generated_frames(1-cosine)/total_valid_generated_frames",
        "loss": "masked cosine distance in float32; teacher detached",
        "student_layer_zero_based": 9,
        "student_stream": "audio output after the tenth AVT MM-DiT block",
        "projector": {"dimensions": [768, 2048, 2048, 768], "activation": "SiLU", "bias": True},
        "teacher": {
            "model_id": WAVLM_MODEL_ID, "model_revision": WAVLM_MODEL_REVISION,
            "checkpoint_sha256": WAVLM_CHECKPOINT_SHA256,
            "teacher_layer": 12, "feature_dim": 768,
            "source_audio": "complete_unmasked_waveform", "frozen": True,
            "sample_rate": 16000, "frame_rate_hz": 50.0, "cache_dtype": "float16",
        },
        "alignment": "linear interpolation of each unpadded 50-Hz teacher to valid 40-Hz latent length; align_corners=False",
        "attention": {"mode": "avt_joint", "av_va_window_radius_frames": 4, "mm_layers": 12, "audio_only_layers": 6},
        "inference": "projector weights load strictly but the auxiliary head is not called; no WavLM teacher or cache required",
    }


def validate_wavlm_tra_checkpoint_metadata(checkpoint: Mapping, expected_contract: dict) -> None:
    """Reject unlabelled or different-objective weights on resume/inference."""
    checkpoint = _object(checkpoint, "checkpoint")
    if CHECKPOINT_CONTRACT_KEY not in checkpoint:
        raise RuntimeError("Checkpoint is missing embedded wavlm_tra_contract; plain hard9 and fixed-REPA weights are not TRA")
    _require_exact(checkpoint[CHECKPOINT_CONTRACT_KEY], expected_contract, "Embedded WavLM TRA contract")


def validate_wavlm_tra_inference_contract(config, checkpoint_path: Path, checkpoint: Mapping) -> dict:
    """Require consistent config, embedded identity and original sidecar."""
    expected = build_wavlm_tra_contract(config)
    validate_wavlm_tra_checkpoint_metadata(checkpoint, expected)
    sidecar_path = Path(checkpoint_path).parent / "speaker_training_contract.json"
    if not sidecar_path.is_file():
        raise RuntimeError(f"WavLM TRA inference requires its training sidecar: {sidecar_path}")
    sidecar = _object(json.loads(sidecar_path.read_text(encoding="utf-8")), "training sidecar")
    _require_exact(sidecar.get("repa"), expected, "Training sidecar WavLM TRA contract")
    recorded = build_wavlm_tra_contract(sidecar.get("config"))
    _require_exact(recorded, expected, "Resolved training WavLM TRA config")
    # Validate the teacher extraction receipt without touching its original path.
    cache_metadata = _object(sidecar.get("repa_cache_metadata"), "training WavLM cache metadata")
    _require_values(cache_metadata, {
        "feature": "wavlm_hidden_state", "model_id": WAVLM_MODEL_ID,
        "model_revision": WAVLM_MODEL_REVISION, "checkpoint_sha256": WAVLM_CHECKPOINT_SHA256,
        "teacher_layer": 12, "output_dim": 768, "output_dtype": "float16",
        "source_audio": "complete_unmasked_waveform", "sample_rate": 16000,
        "frame_rate_hz": 50.0, "status": "complete",
    }, "Training WavLM cache metadata")
    return expected
