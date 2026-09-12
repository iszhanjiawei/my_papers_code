"""Direct-C2 CTC warmup with diagnostics for the added CAM++ projection."""

from __future__ import annotations

import math

import torch

from aligndit.model.locat_contract import validate_locat_checkpoint_contract
from aligndit.model.trainer_semantic_vae_direct_ctc_warmup import SemanticVaeDirectC2CtcWarmupTrainer


class SemanticVaeDirectC2SpeakerTrainer(SemanticVaeDirectC2CtcWarmupTrainer):
    def load_checkpoint(self):
        # Tensor shapes alone do not encode Gaussian versus uniform bias, input
        # frame rates, width bounds, or the effective direction/layer scope.
        backbone = self.accelerator.unwrap_model(self.model).transformer
        validate_locat_checkpoint_contract(backbone, self.checkpoint_path, require_existing=False)
        return super().load_checkpoint()

    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        total = float(loss.detach())
        if not math.isfinite(total) or any(not math.isfinite(float(v)) for v in loss_components.values()):
            raise FloatingPointError(f"Non-finite training loss: total={total}, components={loss_components}")
        backbone = self.accelerator.unwrap_model(self.model).transformer
        projection = backbone.speaker_proj
        diagnostics["speaker_proj_weight_norm"] = projection.weight.detach().float().norm().item()
        weighted_ctc = float(loss_components.get("ctc_loss", 0.0)) * self.current_ctc_lambda
        diagnostics["ctc_weighted_loss"] = weighted_ctc
        diagnostics["ctc_fraction_of_total"] = weighted_ctc / total if total > 0 else 0.0
        if getattr(backbone, "locat_config", {}).get("locat_enabled", False):
            for name, tensor in backbone.locat_diagnostics().items():
                value = float(tensor.detach()) if isinstance(tensor, torch.Tensor) else float(tensor)
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite LocAt diagnostic {name}: {value}")
                diagnostics[f"locat/{name}"] = value
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if not self.accelerator.sync_gradients or self.max_grad_norm <= 0:
            return None
        backbone = self.accelerator.unwrap_model(self.model).transformer
        projection = backbone.speaker_proj
        # bf16 does not use gradient scaling. Record before clipping, matching
        # the global norm returned by clip_grad_norm_.
        if projection.weight.grad is not None:
            speaker_grad = projection.weight.grad.detach().float().norm().item()
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar("speaker_proj_grad_norm", speaker_grad, self.completed_updates + 1)
        for direction in ("av", "va"):
            parameters = [
                parameter for name, parameter in backbone.named_parameters()
                if f".locat_{direction}." in name
            ]
            if not parameters:
                continue
            gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
            gradient_norm = float(torch.stack(gradients).norm()) if gradients else 0.0
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite LocAt {direction} gradient norm: {gradient_norm}")
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar(f"locat/{direction}/grad_norm", gradient_norm, self.completed_updates + 1)
                self.writer.add_scalar(
                    f"locat/{direction}/missing_grad_count",
                    sum(p.grad is None for p in parameters), self.completed_updates + 1,
                )
        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite pre-clipping gradient norm: {norm}")
        return float(norm)

    def _before_update(self, global_update: int) -> None:
        self.completed_updates = global_update
        super()._before_update(global_update)
