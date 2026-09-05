"""Direct-D1 trainer with a delayed linear CTC-weight warmup."""

from __future__ import annotations

import json
import math
from pathlib import Path

from aligndit.model.trainer_semantic_vae_direct import SemanticVaeDirectD1Trainer


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


class SemanticVaeDirectD1CtcWarmupTrainer(SemanticVaeDirectD1Trainer):
    """Apply the CTC schedule to child optimizer updates, including resumed runs."""

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
        return {
            "ctc_lambda": self.current_ctc_lambda,
            "ctc_weighted_loss": self.current_ctc_lambda * loss_components.get("ctc_loss", 0.0),
        }

    def load_checkpoint(self):
        """Reject a resume that would silently change the CTC schedule."""

        checkpoint_dir = Path(self.checkpoint_path)
        contract_path = checkpoint_dir / "ctc_schedule.json"
        expected = {
            "experiment": "semantic_vae_direct_d1",
            "ctc_lambda": self.ctc_target_lambda,
            "ctc_warmup_start": self.ctc_warmup_start,
            "ctc_warmup_end": self.ctc_warmup_end,
            "update_convention": "completed_child_optimizer_updates_plus_one",
        }
        # One writer and a barrier prevent non-main ranks from observing a
        # newly created but only partially written JSON file.
        if self.is_main:
            if contract_path.exists():
                actual = json.loads(contract_path.read_text(encoding="utf-8"))
                if actual != expected:
                    raise RuntimeError(f"CTC schedule differs from the existing run: {contract_path}")
            else:
                if any(checkpoint_dir.glob("*.pt")) or any(checkpoint_dir.glob("*.safetensors")):
                    raise RuntimeError(f"Cannot resume checkpoints without their CTC schedule contract: {contract_path}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with contract_path.open("x", encoding="utf-8") as file:
                    json.dump(expected, file, indent=2, sort_keys=True)
                    file.write("\n")
        self.accelerator.wait_for_everyone()
        return super().load_checkpoint()
