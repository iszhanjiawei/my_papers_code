"""CPU smoke test for the Semantic-VAE C2 trainer's resumable ramps and safety guards."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

from aligndit.model.trainer_semantic_vae_c2 import SemanticVaeC2Trainer


class _FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    def __init__(self, *, clip_result: float = 1.0) -> None:
        self.clip_result = clip_result

    def unwrap_model(self, model):
        return model

    def clip_grad_norm_(self, parameters, max_norm):
        del parameters, max_norm
        return torch.tensor(self.clip_result)

    def reduce(self, value, reduction="sum"):
        if reduction != "sum":
            raise RuntimeError(f"Unexpected reduction {reduction!r}")
        return value


def _bare_trainer(*, clip_result: float = 1.0, abort_threshold: float = 100.0) -> SemanticVaeC2Trainer:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    trainer = object.__new__(SemanticVaeC2Trainer)
    trainer.stage = "s3"
    trainer.max_stage_updates = 200_000
    trainer.ctc_target_lambda = 0.1
    trainer.ctc_ramp_updates = 20_000
    trainer.max_grad_norm = 1.0
    trainer.global_grad_norm_abort_threshold = abort_threshold
    trainer.accelerator = _FakeAccelerator(clip_result=clip_result)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.AdamW([parameter])
    return trainer


def main() -> None:
    signature = inspect.signature(SemanticVaeC2Trainer.__init__)
    legacy_defaults = {
        name: signature.parameters[name].default
        for name in (
            "ctc_ramp_updates",
            "global_grad_norm_abort_threshold",
            "group_grad_norm_log_interval",
            "raw_text_rms_abort_threshold",
        )
    }
    if legacy_defaults != {
        "ctc_ramp_updates": 0,
        "global_grad_norm_abort_threshold": 0.0,
        "group_grad_norm_log_interval": 0,
        "raw_text_rms_abort_threshold": 0.0,
    }:
        raise RuntimeError(f"Legacy-safe trainer defaults changed: {legacy_defaults}")

    trainer = _bare_trainer()
    expected_ramp = {0: 0.1 / 20_000, 19_999: 0.1, 20_000: 0.1, 80_000: 0.1}
    for update, expected in expected_ramp.items():
        actual = trainer.ctc_lambda_at(update)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"CTC ramp mismatch at update {update}: expected={expected}, got={actual}")
    trainer.ctc_ramp_updates = 0
    if trainer.ctc_lambda_at(0) != 0.1 or trainer.ctc_lambda_at(80_000) != 0.1:
        raise RuntimeError("Legacy stages no longer retain a constant CTC lambda of 0.1")

    transformer = SimpleNamespace(
        last_text_context_raw_rms=torch.tensor(1.25),
        last_text_context_post_rms=torch.tensor(0.999),
    )
    trainer.model = SimpleNamespace(transformer=transformer)
    if trainer._text_context_rms() != (1.25, float(torch.tensor(0.999).item())):
        raise RuntimeError("Text-context RMS telemetry changed")

    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])
    trainer.optimizer = torch.optim.AdamW(
        [
            {"params": [first], "group_name": "first"},
            {"params": [second], "group_name": "second"},
        ]
    )
    group_norms = trainer._optimizer_group_grad_norms()
    if group_norms != {"first": 5.0, "second": 12.0}:
        raise RuntimeError(f"Pre-clip optimizer-group norms changed: {group_norms}")

    trainer = _bare_trainer(clip_result=12.5, abort_threshold=100.0)
    if trainer._clip_gradients_or_fail() != 12.5:
        raise RuntimeError("Finite global gradient norm was not returned")

    trainer = _bare_trainer(clip_result=float("inf"), abort_threshold=100.0)
    try:
        trainer._clip_gradients_or_fail()
    except FloatingPointError as error:
        if "failed_ranks=1/1" not in str(error):
            raise RuntimeError(f"Unexpected gradient fail-fast message: {error}") from error
    else:
        raise RuntimeError("Non-finite pre-clip global gradient norm did not fail fast")

    trainer = _bare_trainer(clip_result=101.0, abort_threshold=100.0)
    try:
        trainer._clip_gradients_or_fail()
    except FloatingPointError:
        pass
    else:
        raise RuntimeError("Above-threshold pre-clip global gradient norm did not fail fast")

    trainer = _bare_trainer(clip_result=float("inf"), abort_threshold=0.0)
    if not math.isinf(trainer._clip_gradients_or_fail()):
        raise RuntimeError("The disabled legacy gradient guard changed the clip return value")

    print("Semantic-VAE C2 trainer guard smoke test passed", flush=True)


if __name__ == "__main__":
    main()
