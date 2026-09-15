"""The speaker experiment with diagnostics for the added Synchformer path."""

from __future__ import annotations

import torch

from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer


class SemanticVaeDirectC2SpeakerSynchformerTrainer(SemanticVaeDirectC2SpeakerTrainer):
    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        transformer = self.accelerator.unwrap_model(self.model).transformer
        projection = transformer.sync_in[2].w2
        diagnostics["synchformer/output_projection_weight_norm"] = projection.weight.detach().float().norm().item()
        diagnostics["synchformer/position_embedding_norm"] = transformer.sync_pos_emb.detach().float().norm().item()
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if self.accelerator.sync_gradients:
            transformer = self.accelerator.unwrap_model(self.model).transformer
            modules = {"input": transformer.sync_in[0], "output": transformer.sync_in[2].w2}
            for name, module in modules.items():
                gradient = module.weight.grad
                if gradient is None:
                    continue
                norm = gradient.detach().float().norm()
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Non-finite Synchformer {name} projection gradient")
                if self.is_main and self.logger == "tensorboard":
                    self.writer.add_scalar(
                        f"synchformer/{name}_projection_grad_norm", norm.item(), self.completed_updates + 1
                    )
        return super()._clip_gradients()
