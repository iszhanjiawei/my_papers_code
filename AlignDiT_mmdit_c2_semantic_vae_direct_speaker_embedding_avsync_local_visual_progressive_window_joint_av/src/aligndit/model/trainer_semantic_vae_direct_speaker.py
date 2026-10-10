"""Direct-C2 CTC warmup with CAM++ and optional visual-local gate diagnostics."""

from __future__ import annotations

import math

import torch
from torch.utils.tensorboard import SummaryWriter

from aligndit.model.trainer_semantic_vae_direct_ctc_warmup import SemanticVaeDirectC2CtcWarmupTrainer


class SemanticVaeDirectC2SpeakerTrainer(SemanticVaeDirectC2CtcWarmupTrainer):
    def load_checkpoint(self):
        update = super().load_checkpoint()
        if update > 0 and self.is_main:
            if self.logger == "tensorboard":
                # Hide metrics from updates that were not saved before the
                # interrupted run. Keep the checkpoint update and its history.
                logdir = self.writer.log_dir
                self.writer.close()
                self.writer = SummaryWriter(log_dir=logdir, purge_step=update + 1)
            print(
                f"Resumed at update {update}: model, EMA, optimizer and scheduler restored; "
                "TensorBoard discards unsaved updates after this checkpoint",
                flush=True,
            )
        return update

    def _local_visual_gates(self):
        transformer = self.accelerator.unwrap_model(self.model).transformer
        return [
            (layer, block.local_visual_attn.gate)
            for layer, block in enumerate(transformer.transformer_blocks)
            if getattr(block, "local_visual_attn", None) is not None
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
        if self.is_main:
            gates = self._local_visual_gates()
            if gates:
                # Transfer all tiny gate reductions together, avoiding one CUDA
                # synchronization per layer and retaining only detached values.
                values = torch.stack([
                    torch.stack((gate.detach().float().mean(), gate.detach().float().abs().amax()))
                    for _, gate in gates
                ]).cpu().tolist()
                for (layer, _), (mean, absmax) in zip(gates, values):
                    diagnostics[f"local_visual/layer_{layer}/gate_mean"] = mean
                    diagnostics[f"local_visual/layer_{layer}/gate_absmax"] = absmax
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
        if self.is_main and self.logger == "tensorboard":
            gates_with_grad = [(layer, gate) for layer, gate in self._local_visual_gates() if gate.grad is not None]
            if gates_with_grad:
                grad_norms = torch.stack([
                    gate.grad.detach().float().norm() for _, gate in gates_with_grad
                ]).cpu().tolist()
                for (layer, _), grad_norm in zip(gates_with_grad, grad_norms):
                    self.writer.add_scalar(
                        f"local_visual/layer_{layer}/gate_grad_norm", grad_norm, self.completed_updates + 1
                    )
        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite pre-clipping gradient norm: {norm}")
        return float(norm)

    def _before_update(self, global_update: int) -> None:
        self.completed_updates = global_update
        super()._before_update(global_update)
