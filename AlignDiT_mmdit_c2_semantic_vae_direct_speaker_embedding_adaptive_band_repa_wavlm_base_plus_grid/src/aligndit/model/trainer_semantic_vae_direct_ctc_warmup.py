"""Direct-C2 trainer with a delayed linear CTC-weight warmup."""

from __future__ import annotations

import math

from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectC2Trainer


def ctc_lambda_for_update(
    completed_updates: int,
    *,
    target: float,
    warmup_start: int,
    warmup_end: int,
) -> float:
    """Return the CTC weight for the next optimizer update.

    The first ``warmup_start`` optimizer updates use zero CTC weight.  The
    weight then increases linearly and reaches ``target`` on update
    ``warmup_end``.
    """

    if completed_updates < 0:
        raise ValueError(f"completed_updates must be non-negative, got {completed_updates}")
    if not math.isfinite(target) or target < 0:
        raise ValueError(f"target CTC lambda must be finite and non-negative, got {target}")
    if warmup_start < 0 or warmup_end <= warmup_start:
        raise ValueError(
            "CTC warmup boundaries must satisfy 0 <= warmup_start < warmup_end, "
            f"got start={warmup_start}, end={warmup_end}"
        )

    next_update = completed_updates + 1
    if next_update <= warmup_start:
        return 0.0
    if next_update >= warmup_end:
        return float(target)
    progress = (next_update - warmup_start) / (warmup_end - warmup_start)
    return float(target) * progress


class SemanticVaeDirectC2CtcWarmupTrainer(SemanticVaeDirectC2Trainer):
    """Keep Direct-C2 training unchanged except for the requested CTC schedule."""

    def __init__(
        self,
        *args,
        ctc_target_lambda: float,
        ctc_warmup_start: int,
        ctc_warmup_end: int,
        **kwargs,
    ) -> None:
        # Validate once before allocating the model/trainer state.
        ctc_lambda_for_update(
            0,
            target=ctc_target_lambda,
            warmup_start=ctc_warmup_start,
            warmup_end=ctc_warmup_end,
        )
        self.ctc_target_lambda = float(ctc_target_lambda)
        self.ctc_warmup_start = int(ctc_warmup_start)
        self.ctc_warmup_end = int(ctc_warmup_end)
        self.current_ctc_lambda = 0.0
        super().__init__(*args, **kwargs)

    def _before_update(self, global_update: int) -> None:
        value = ctc_lambda_for_update(
            global_update,
            target=self.ctc_target_lambda,
            warmup_start=self.ctc_warmup_start,
            warmup_end=self.ctc_warmup_end,
        )
        model = self.accelerator.unwrap_model(self.model)
        model.ctc_lambda = value
        self.current_ctc_lambda = value

    def _forward_diagnostics(self, loss, loss_components) -> dict[str, float]:
        return {"ctc_lambda": self.current_ctc_lambda}
