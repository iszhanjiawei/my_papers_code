"""Checkpoint identity for parameter-free changes to paired attention."""


from aligndit.model.text_audio_checkpoint import PRETRAIN_STAGE


def checkpoint_architecture(arch, initialization_mode=None) -> dict | None:
    """Retain the historical checkpoint identity when the Gaussian prior is off."""
    order = arch.get("paired_joint_order")
    identity = {"paired_joint_order": order} if order is not None or arch.get("temporal_band_enabled") else {}
    if arch.get("repa_layer") is not None:
        identity.update(
            repa_layer=arch["repa_layer"],
            repa_projector_dim=arch.get("repa_projector_dim") or (2 * arch["dim"]),
            repa_target_dim=arch["repa_target_dim"],
        )
    if not arch.get("temporal_band_enabled", False):
        return identity or None
    return {
        **identity,
        **({"parent_stage": PRETRAIN_STAGE} if initialization_mode == PRETRAIN_STAGE else {}),
        "temporal_band_enabled": True,
        "temporal_band_mode": arch.get("temporal_band_mode", "fixed"),
        "temporal_band_audio_fps": arch.get("temporal_band_audio_fps", 40.0),
        "temporal_band_video_fps": arch.get("temporal_band_video_fps", 40.0),
        "temporal_band_fixed_offset_seconds": arch.get("temporal_band_fixed_offset_seconds", 0.0),
        "temporal_band_fixed_sigma_seconds": arch.get("temporal_band_fixed_sigma_seconds", 0.150),
        "temporal_band_bidirectional": arch.get("temporal_band_bidirectional", False),
    }
