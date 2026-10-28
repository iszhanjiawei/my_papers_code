"""Identity checks for the parameter-free fixed AV/VA prior in AVT attention.

Weight tensors cannot distinguish this experiment from its no-band AVT parent.
Keep the actual geometry and topology in the training sidecar and require that
sidecar when loading fixed-band weights for inference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


FIXED_BAND_KEYS = (
    "temporal_band_enabled", "temporal_band_mode", "temporal_band_audio_fps",
    "temporal_band_video_fps", "temporal_band_fixed_offset_seconds",
    "temporal_band_fixed_sigma_seconds", "temporal_band_bidirectional",
)
EXPERIMENT_MARKER = "avt_joint_fixed_band_av_va"
HARD_WINDOW_MARKER = "avt_joint_hard_window9_av_va"
RADIUS_KEY = "temporal_band_fixed_window_radius_frames"


def _window_radius(arch):
    radius = arch.get(RADIUS_KEY)
    if radius is not None and (type(radius) is not int or radius < 0):
        raise ValueError(f"{RADIUS_KEY} must be a nonnegative integer or None")
    return radius


def _validate_arch(arch) -> bool:
    enabled = arch.get("temporal_band_enabled", False)
    bidirectional = arch.get("temporal_band_bidirectional", False)
    radius = _window_radius(arch)
    if type(enabled) is not bool or type(bidirectional) is not bool:
        raise TypeError("temporal_band_enabled and temporal_band_bidirectional must be bools")
    if not enabled:
        if bidirectional or radius is not None:
            raise ValueError("temporal_band_bidirectional requires an enabled fixed temporal band")
        return False
    if arch.get("temporal_band_mode") != "fixed" or not bidirectional:
        raise ValueError("This experiment requires enabled fixed AV+VA temporal bias")
    if arch.get("text_joint_attention") is not True:
        raise ValueError("Fixed AV+VA temporal bias in this snapshot requires text_joint_attention=True")
    for key in ("temporal_band_audio_fps", "temporal_band_video_fps",
                "temporal_band_fixed_offset_seconds", "temporal_band_fixed_sigma_seconds"):
        value = arch.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key != "temporal_band_fixed_offset_seconds" and value <= 0:
            raise ValueError(f"{key} must be positive")
    if radius is not None and (
        arch["temporal_band_audio_fps"] != arch["temporal_band_video_fps"]
        or arch["temporal_band_fixed_offset_seconds"] != 0.0
    ):
        raise ValueError("Hard temporal windows require equal audio/video frame rates and zero offset")
    return True


def validate_fixed_band_training_config(config) -> bool:
    """Reject incompatible or mislabeled runs before model/data allocation."""
    arch = config.model.arch
    enabled = _validate_arch(arch)
    run_name, save_dir = str(config.model.name), Path(config.ckpts.save_dir)
    radius = _window_radius(arch)
    if any("hard_window" in value for value in (run_name, str(save_dir))) and radius is None:
        raise ValueError("Hard-window run names and checkpoint paths require a hard temporal window")
    if not enabled:
        if any("fixed_band" in value for value in (run_name, str(save_dir))):
            raise ValueError("Fixed-band run names and checkpoint paths require an enabled fixed AV+VA bias")
        return False
    for key, expected in (("depth", 18), ("n_mm_layers", 12), ("n_text_layers", 12),
                          ("audio_video_ratio", 1)):
        if arch.get(key) != expected:
            raise ValueError(f"The fixed AVT experiment retains {key}={expected}")
    frame_rate = float(config.model.audio_representation.frame_rate)
    if frame_rate != 40.0 or any(float(arch[key]) != frame_rate for key in (
        "temporal_band_audio_fps", "temporal_band_video_fps"
    )):
        raise ValueError("Temporal-band rates must match the already-interpolated 40-Hz cache")
    marker = EXPERIMENT_MARKER
    if radius is not None:
        if radius != 4:
            raise ValueError("This hard_window9_av_va experiment requires window radius 4 (nine frames)")
        marker = HARD_WINDOW_MARKER
    if any(marker not in value for value in (run_name, str(save_dir))):
        raise ValueError(f"{marker} runs require dedicated model.name and ckpts.save_dir")
    if not (save_dir / "speaker_training_contract.json").is_file() and any(save_dir.glob("*.pt")):
        raise RuntimeError("Refusing to label existing weights as fixed-band without their original training contract")
    return True


def make_fixed_band_contract(arch, band) -> dict:
    """Describe geometry without modifying the inherited initialization policy."""
    if not _validate_arch(arch):
        raise ValueError("Cannot make a fixed-band contract for disabled temporal bias")
    from aligndit.model.fixed_temporal_band import FixedTemporalBand

    if not isinstance(band, FixedTemporalBand) or list(band.parameters()) or band.state_dict():
        raise RuntimeError("Fixed temporal band must have zero parameters and serialized state")
    radius = _window_radius(arch)
    if band.window_radius_frames != radius or type(band.window_radius_frames) is not type(radius):
        raise RuntimeError(f"Actual fixed temporal band and configuration disagree: {RADIUS_KEY}")
    for attribute, key in (
        ("audio_fps", "temporal_band_audio_fps"), ("video_fps", "temporal_band_video_fps"),
        ("offset_seconds", "temporal_band_fixed_offset_seconds"),
        ("sigma_seconds", "temporal_band_fixed_sigma_seconds"),
    ):
        if getattr(band, attribute) != arch[key]:
            raise RuntimeError(f"Actual fixed temporal band and configuration disagree: {key}")
    contract = {
        "mode": "fixed",
        "formula": "B_av[i,j] = -(t_video[j] - t_audio[i] - offset)**2 / (2*sigma**2); B_va = B_av.T",
        "predictor_input": "none; fixed constants independent of content, CFG branch and diffusion time",
        "sharing": "shared over all samples, heads and first 12 AVT MM blocks",
        "scope": "AV and VA logits only (transposed prior); AA/VV/AT/VT/TA/TV/TT biases remain zero; "
                 "shared joint softmax over all audio/video/text keys retained",
        "text_joint_attention": True,
        "initialization": f"fixed offset={arch.temporal_band_fixed_offset_seconds} seconds and "
                          f"sigma={arch.temporal_band_fixed_sigma_seconds} seconds throughout training",
        "function_preserving_when_enabled": False,
        "extra_loss": False,
        "mass_preservation": False,
        "parameters": {key: arch[key] for key in FIXED_BAND_KEYS},
        "parameter_count": 0,
    }
    if radius is not None:
        contract["parameters"][RADIUS_KEY] = radius
        contract.update({
            "attention_mode": "avt_joint",
            "kernel": "hard_window",
            "formula": "B_av[i,j] = 0 if abs(i-j) <= radius_frames else -inf; B_va = B_av.T",
            "window_radius_frames": radius,
            "window_size_frames": 2 * radius + 1,
            "initialization": f"fixed zero offset; radius={radius} aligned 40-Hz frames throughout training",
            "boundary_policy": "clip to existing keys; do not shift or pad the window",
            "gaussian_sigma_active": False,
        })
    return contract


def validate_fixed_band_contract(config, checkpoint_path: Path) -> None:
    """Validate both directions so fixed-band weights cannot silently lose bias."""
    arch = config.model.arch
    requested_fixed = _validate_arch(arch)
    contract_path = Path(checkpoint_path).parent / "speaker_training_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    if not isinstance(contract, Mapping):
        raise TypeError("Training contract must be a JSON object")
    recorded = contract.get("temporal_band", {})
    if not isinstance(recorded, Mapping):
        raise TypeError("Training temporal_band contract must be a JSON object")
    recorded_fixed = recorded.get("mode") == "fixed"
    if not (requested_fixed or recorded_fixed):
        return
    if not requested_fixed or not recorded_fixed:
        raise RuntimeError("Fixed-band inference requires the matching fixed-band training contract beside the checkpoint")
    recorded_parameters = recorded.get("parameters", {})
    if not isinstance(recorded_parameters, Mapping):
        raise TypeError("Training temporal_band parameters must be a JSON object")
    for key in FIXED_BAND_KEYS:
        requested, stored = arch.get(key), recorded_parameters.get(key)
        if key not in recorded_parameters or type(stored) is not type(requested) or stored != requested:
            raise RuntimeError(f"Fixed-band inference/training configuration mismatch: {key}")
    requested_radius = _window_radius(arch)
    try:
        stored_radius = _window_radius(recorded_parameters)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if requested_radius is not None and RADIUS_KEY not in recorded_parameters:
        raise RuntimeError(f"Hard-window training contract is missing {RADIUS_KEY}")
    if type(stored_radius) is not type(requested_radius) or stored_radius != requested_radius:
        raise RuntimeError(f"Fixed-band inference/training configuration mismatch: {RADIUS_KEY}")
    if requested_radius is not None:
        expected_geometry = {
            "attention_mode": "avt_joint", "kernel": "hard_window",
            "window_radius_frames": requested_radius, "window_size_frames": 2 * requested_radius + 1,
            "boundary_policy": "clip to existing keys; do not shift or pad the window",
            "gaussian_sigma_active": False,
        }
        for key, expected in expected_geometry.items():
            if type(recorded.get(key)) is not type(expected) or recorded.get(key) != expected:
                raise RuntimeError(f"Hard-window inference/training configuration mismatch: {key}")
    elif recorded.get("kernel", "gaussian") != "gaussian" or recorded.get("gaussian_sigma_active", True) is not True:
        raise RuntimeError("Gaussian inference cannot load a hard-window training contract")
    if type(recorded.get("parameter_count")) is not int or recorded["parameter_count"] != 0:
        raise RuntimeError("Fixed-band checkpoint contract must record zero temporal-band parameters")
    recorded_arch = contract.get("config", {}).get("model", {}).get("arch", {})
    if recorded_arch.get("text_joint_attention") is not True or recorded.get("text_joint_attention") is not True:
        raise RuntimeError("Fixed-band inference/training configuration mismatch: text_joint_attention")
    # Compare the nested resolved config as well, preventing contradictory
    # sidecars from identifying the same tensors as two attention experiments.
    for key in FIXED_BAND_KEYS:
        if recorded_arch.get(key) != arch.get(key):
            raise RuntimeError(f"Fixed-band training config and sidecar disagree: {key}")
    if (type(recorded_arch.get(RADIUS_KEY)) is not type(requested_radius)
            or recorded_arch.get(RADIUS_KEY) != requested_radius):
        raise RuntimeError(f"Fixed-band training config and sidecar disagree: {RADIUS_KEY}")
