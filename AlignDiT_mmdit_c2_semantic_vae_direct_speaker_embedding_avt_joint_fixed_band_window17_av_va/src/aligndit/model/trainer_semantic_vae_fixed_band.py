"""Unchanged AVT speaker training with parameter-free temporal-band diagnostics."""

from __future__ import annotations

import math

import torch

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer


class SemanticVaeFixedBandTrainer(SemanticVaeDirectC2SpeakerTrainer):
    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        backbone = self.accelerator.unwrap_model(self.model).transformer
        valid = backbone.last_temporal_band_valid_mask
        for label, tensor in (
            ("offset_ms", backbone.last_temporal_band_offset_seconds),
            ("sigma_ms", backbone.last_temporal_band_sigma_seconds),
        ):
            if tensor is None:
                raise RuntimeError("Fixed-band training did not produce temporal parameters")
            values = tensor.detach().float()
            values = values[valid] if valid is not None else values.reshape(-1)
            if values.numel() == 0:
                raise RuntimeError("Fixed-band diagnostics have no valid audio queries")
            values = values * 1000.0
            stats = torch.stack((values.mean(), values.std(unbiased=False), values.min(), values.max()))
            for statistic, value in zip(("mean", "std", "min", "max"), stats.tolist()):
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite fixed-band {label}/{statistic}")
                diagnostics[f"temporal_band/{label}_{statistic}"] = value
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if self.accelerator.sync_gradients and self.max_grad_norm > 0:
            band = self.accelerator.unwrap_model(self.model).transformer.temporal_band
            if not isinstance(band, FixedTemporalBand) or list(band.parameters()) or band.state_dict():
                raise RuntimeError("Fixed temporal band must remain parameter-free and have no state tensors")
        return super()._clip_gradients()
