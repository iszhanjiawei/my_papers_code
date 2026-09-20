"""Direct-C2 CTC warmup with CAM++ and optional visual-gate diagnostics."""

from __future__ import annotations

import math

import torch

from aligndit.model.trainer_semantic_vae_direct_ctc_warmup import SemanticVaeDirectC2CtcWarmupTrainer


class SemanticVaeDirectC2SpeakerTrainer(SemanticVaeDirectC2CtcWarmupTrainer):
    def _visual_gate_parameters(self):
        transformer = self.accelerator.unwrap_model(self.model).transformer
        return [
            (index, block.av_visual_delta_gate)
            for index, block in enumerate(transformer.transformer_blocks)
            if getattr(block, "av_visual_delta_gate", None) is not None
        ]

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
        gates = self._visual_gate_parameters()
        if gates:
            values = torch.stack([gate.detach().float() for _, gate in gates]).cpu().tolist()
            if any(not math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite visual attention gate: {values}")
            diagnostics.update({
                f"av_visual_gate/layer_{index:02d}": value
                for (index, _), value in zip(gates, values)
            })
            diagnostics["av_visual_gate/mean"] = sum(values) / len(values)
            diagnostics["av_visual_gate/min"] = min(values)
            diagnostics["av_visual_gate/max"] = max(values)
            diagnostics["av_visual_gate/abs_max"] = max(abs(value) for value in values)
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
        gates = self._visual_gate_parameters()
        if gates:
            missing = [index for index, gate in gates if gate.grad is None]
            if missing:
                raise RuntimeError(f"Visual attention gates have no backward path in layers: {missing}")
            gradients = torch.stack([gate.grad.detach().float() for _, gate in gates]).cpu().tolist()
            if any(not math.isfinite(value) for value in gradients):
                raise FloatingPointError(f"Non-finite visual attention gate gradients: {gradients}")
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar(
                    "av_visual_gate/grad_norm", math.hypot(*gradients), self.completed_updates + 1
                )
        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite pre-clipping gradient norm: {norm}")
        return float(norm)

    def _before_update(self, global_update: int) -> None:
        self.completed_updates = global_update
        super()._before_update(global_update)
