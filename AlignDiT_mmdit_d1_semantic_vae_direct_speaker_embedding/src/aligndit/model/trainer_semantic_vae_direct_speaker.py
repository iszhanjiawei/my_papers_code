"""Direct-D1 CTC warmup with diagnostics for the added CAM++ projection."""

from __future__ import annotations

import math

import torch

from aligndit.model.trainer_semantic_vae_direct_ctc_warmup import SemanticVaeDirectD1CtcWarmupTrainer


class SemanticVaeDirectD1SpeakerTrainer(SemanticVaeDirectD1CtcWarmupTrainer):
    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        total = float(loss.detach())
        if not math.isfinite(total) or any(not math.isfinite(float(v)) for v in loss_components.values()):
            raise FloatingPointError(f"Non-finite training loss: total={total}, components={loss_components}")
        projection = self.accelerator.unwrap_model(self.model).transformer.speaker_proj
        diagnostics["speaker_proj_weight_norm"] = projection.weight.detach().float().norm().item()
        weighted_ctc = float(loss_components.get("ctc_loss", 0.0)) * self.current_ctc_lambda
        diagnostics["ctc_weighted_loss"] = weighted_ctc
        diagnostics["ctc_fraction_of_total"] = weighted_ctc / total if total > 0 else 0.0
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if not self.accelerator.sync_gradients or self.max_grad_norm <= 0:
            return None
        projection = self.accelerator.unwrap_model(self.model).transformer.speaker_proj
        # bf16 does not use gradient scaling. Record before clipping, matching
        # the global norm returned by clip_grad_norm_.
        if projection.weight.grad is not None:
            speaker_grad = projection.weight.grad.detach().float().norm().item()
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar("speaker_proj_grad_norm", speaker_grad, self.completed_updates + 1)
        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite pre-clipping gradient norm: {norm}")
        return float(norm)

    def _before_update(self, global_update: int) -> None:
        self.completed_updates = global_update
        super()._before_update(global_update)
