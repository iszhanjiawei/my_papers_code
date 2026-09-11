"""Unchanged speaker training policy, with temporal-band learning diagnostics."""

from __future__ import annotations

import math

import torch

from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer


class SemanticVaeAdaptiveBandTrainer(SemanticVaeDirectC2SpeakerTrainer):
    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        backbone = self.accelerator.unwrap_model(self.model).transformer
        valid = backbone.last_temporal_band_valid_mask
        for label, tensor in (
            ("offset_ms", backbone.last_temporal_band_offset_seconds),
            ("sigma_ms", backbone.last_temporal_band_sigma_seconds),
        ):
            if tensor is None:
                raise RuntimeError("Adaptive-band training did not produce temporal parameters")
            values = tensor.detach().float()
            if valid is not None:
                values = values[valid]
            else:
                values = values.reshape(-1)
            if values.numel() == 0:
                raise RuntimeError("Temporal-band diagnostics have no valid audio queries")
            values = values * 1000.0
            stats = torch.stack((values.mean(), values.std(unbiased=False), values.min(), values.max()))
            for statistic, value in zip(("mean", "std", "min", "max"), stats.tolist()):
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite temporal-band {label}/{statistic}")
                diagnostics[f"temporal_band/{label}_{statistic}"] = value
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if self.accelerator.sync_gradients and self.max_grad_norm > 0:
            band = self.accelerator.unwrap_model(self.model).transformer.temporal_band
            gradients = [p.grad.detach().float().norm() for p in band.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("Temporal-band predictor is disconnected from the training loss")
            norm = torch.stack(gradients).norm()
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite temporal-band gradient norm: {norm}")
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar("temporal_band/grad_norm", norm.item(), self.completed_updates + 1)
        return super()._clip_gradients()
