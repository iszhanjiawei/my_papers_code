"""Checkpoint identity for parameter-free changes to paired attention."""


def checkpoint_architecture(arch) -> dict | None:
    """Retain the historical checkpoint identity when the Gaussian prior is off."""
    order = arch.get("paired_joint_order")
    if not arch.get("temporal_band_enabled", False):
        return {"paired_joint_order": order} if order is not None else None
    return {
        "paired_joint_order": order,
        "temporal_band_enabled": True,
        "temporal_band_mode": arch.get("temporal_band_mode", "fixed"),
        "temporal_band_audio_fps": arch.get("temporal_band_audio_fps", 40.0),
        "temporal_band_video_fps": arch.get("temporal_band_video_fps", 40.0),
        "temporal_band_fixed_offset_seconds": arch.get("temporal_band_fixed_offset_seconds", 0.0),
        "temporal_band_fixed_sigma_seconds": arch.get("temporal_band_fixed_sigma_seconds", 0.150),
        "temporal_band_bidirectional": arch.get("temporal_band_bidirectional", False),
    }
