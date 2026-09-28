"""Isolated CAM++ training with a fixed AV-HuBERT audio InfoNCE teacher."""

from __future__ import annotations

import copy
import math

import torch

from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer


def infonce_lambda_for_update(completed_updates: int, *, target: float, warmup_updates: int) -> float:
    """Weight evaluated before the next update: zero at step 0, target at 10k.

    The schedule uses completed optimizer updates, so gradient accumulation and
    resuming do not change its progress. The first forward has zero weight.
    """
    if completed_updates < 0 or warmup_updates <= 0:
        raise ValueError("InfoNCE requires nonnegative completed updates and positive warmup updates")
    if not math.isfinite(target) or target < 0:
        raise ValueError("InfoNCE target weight must be finite and nonnegative")
    return float(target) * min(completed_updates / warmup_updates, 1.0)


class SemanticVaeDirectC2SpeakerAVHuBERTInfoNCETrainer(SemanticVaeDirectC2SpeakerTrainer):
    def __init__(
        self,
        *args,
        infonce_target_lambda: float,
        infonce_warmup_updates: int,
        infonce_training_contract: dict,
        **kwargs,
    ) -> None:
        infonce_lambda_for_update(0, target=infonce_target_lambda, warmup_updates=infonce_warmup_updates)
        self.infonce_target_lambda = float(infonce_target_lambda)
        self.infonce_warmup_updates = int(infonce_warmup_updates)
        self.infonce_training_contract = copy.deepcopy(infonce_training_contract)
        self.current_infonce_lambda = 0.0
        super().__init__(*args, **kwargs)

    def _before_update(self, global_update: int) -> None:
        super()._before_update(global_update)
        value = infonce_lambda_for_update(
            global_update, target=self.infonce_target_lambda, warmup_updates=self.infonce_warmup_updates
        )
        self.accelerator.unwrap_model(self.model).infonce_lambda = value
        self.current_infonce_lambda = value

    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        # CTC is deliberately off during the first 10k updates. Keep its raw
        # scalar visible from the first TensorBoard event, without evaluating it.
        loss_components.setdefault("ctc_loss", 0.0)
        diagnostics = super()._forward_diagnostics(loss, loss_components)
        required = (
            "diff_loss", "infonce_loss", "infonce_weighted_loss", "infonce_valid_anchors",
            "infonce_positive_similarity", "infonce_negative_similarity",
        )
        missing = [key for key in required if key not in loss_components]
        if missing:
            raise RuntimeError(f"InfoNCE forward did not report required diagnostics: {missing}")
        total = float(loss.detach())
        weighted = float(loss_components["infonce_weighted_loss"])
        projector = self.accelerator.unwrap_model(self.model).transformer.context_alignment_projector
        squared = sum(parameter.detach().float().square().sum() for parameter in projector.parameters())
        diagnostics.update(
            infonce_lambda=self.current_infonce_lambda,
            infonce_fraction_of_total=weighted / total if total > 0 else 0.0,
            infonce_projector_weight_norm=squared.sqrt().item(),
        )
        return diagnostics

    def _clip_gradients(self) -> float | None:
        if self.accelerator.sync_gradients:
            projector = self.accelerator.unwrap_model(self.model).transformer.context_alignment_projector
            gradients = [parameter.grad.detach().float() for parameter in projector.parameters() if parameter.grad is not None]
            if not gradients:
                raise RuntimeError("InfoNCE projector is disconnected from the backward graph")
            gradient_norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite InfoNCE projector gradient norm: {gradient_norm}")
            if self.is_main and self.logger == "tensorboard":
                self.writer.add_scalar("infonce_projector_grad_norm", gradient_norm.item(), self.completed_updates + 1)
        return super()._clip_gradients()

    def _checkpoint_metadata(self):
        return {
            "avhubert_infonce_schema_version": 1,
            "avhubert_infonce_training_contract": self.infonce_training_contract,
        }

    def _validate_checkpoint_metadata(self, checkpoint):
        if checkpoint.get("avhubert_infonce_schema_version") != 1:
            raise RuntimeError("Refusing to resume a checkpoint without the AV-HuBERT InfoNCE schema")
        if checkpoint.get("avhubert_infonce_training_contract") != self.infonce_training_contract:
            raise RuntimeError("InfoNCE checkpoint does not match this model, teacher, loss schedule and optimizer contract")
        required = {"model_state_dict", "optimizer_state_dict", "ema_model_state_dict", "scheduler_state_dict", "update"}
        if required - checkpoint.keys():
            raise RuntimeError(f"Incomplete InfoNCE resume checkpoint: missing {sorted(required - checkpoint.keys())}")
        if type(checkpoint["update"]) is not int or checkpoint["update"] < 0:
            raise RuntimeError("InfoNCE checkpoint update must be a nonnegative integer")
