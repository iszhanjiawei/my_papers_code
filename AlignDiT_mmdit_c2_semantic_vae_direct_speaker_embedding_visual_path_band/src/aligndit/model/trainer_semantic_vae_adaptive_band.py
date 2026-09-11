"""Unchanged speaker training policy, with fixed/adaptive band diagnostics."""

from __future__ import annotations

import math

import torch

from aligndit.model.fixed_temporal_band import FixedTemporalBand
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
                raise RuntimeError("Temporal-band training did not produce temporal parameters")
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
        if backbone.temporal_band_mode == "visual_path":
            increments = backbone.last_visual_path_increments
            edge_mask = backbone.last_visual_path_valid_mask
            if increments is None or edge_mask is None or increments.shape != edge_mask.shape:
                raise RuntimeError("Visual-path alignment did not produce edge diagnostics")
            if not torch.isfinite(increments).all():
                raise FloatingPointError("Non-finite visual-path increments")
            active = increments[edge_mask].float()
            diagnostics["visual_path/visible_edge_fraction"] = (
                edge_mask.float().mean().item() if edge_mask.numel() else 0.0
            )
            # Zero visible edges is valid under whole-video CFG dropout or
            # one-frame clips; record an explicit fraction and finite zeros.
            stats = (
                torch.stack((active.mean(), active.std(unbiased=False), active.min(), active.max()))
                if active.numel() else increments.new_zeros(4)
            )
            for statistic, value in zip(("mean", "std", "min", "max"), stats.tolist()):
                diagnostics[f"visual_path/increment_{statistic}"] = value
            diagnostics["visual_path/path_sigma"] = backbone.temporal_band.path_sigma
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if self.accelerator.sync_gradients and self.max_grad_norm > 0:
            band = self.accelerator.unwrap_model(self.model).transformer.temporal_band
            if isinstance(band, FixedTemporalBand):
                if list(band.parameters()):
                    raise RuntimeError("Fixed temporal band unexpectedly has trainable parameters")
                # Existing speaker/global gradient checks and clipping still run.
                # There is deliberately no fabricated predictor gradient scalar.
                return super()._clip_gradients()
            gradients = [p.grad.detach().float().norm() for p in band.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("Temporal-band predictor is disconnected from the training loss")
            norm = torch.stack(gradients).norm()
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite temporal-band gradient norm: {norm}")
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar("temporal_band/grad_norm", norm.item(), self.completed_updates + 1)
        return super()._clip_gradients()
