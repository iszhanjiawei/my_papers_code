"""Deterministic Gaussian-width curriculum indexed by optimizer updates.

This schedule has no tensor state and consumes no randomness. The checkpoint's
completed update count is sufficient to resume it; diffusion time is unrelated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CosineBandSchedule:
    start_sigma_seconds: float
    end_sigma_seconds: float
    start_update: int
    end_update: int

    def __post_init__(self) -> None:
        for name in ("start_sigma_seconds", "end_sigma_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.start_sigma_seconds <= self.end_sigma_seconds:
            raise ValueError("Cosine band curriculum requires start sigma > end sigma")
        for name in ("start_update", "end_update"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer optimizer update")
        if not 0 <= self.start_update < self.end_update:
            raise ValueError("Curriculum requires 0 <= start_update < end_update")

    @classmethod
    def from_config(cls, config) -> CosineBandSchedule:
        # Reject unknown keys rather than silently ignoring a mistyped boundary.
        return cls(**dict(config))

    def sigma_at(self, update: int) -> float:
        """Width used for one-based update ``update`` (zero means initialization).

        For the 10k..50k schedule, updates 1..10000 use the start width,
        update 30000 is the midpoint, and update 50000 reaches the end width.
        """
        if type(update) is not int or update < 0:
            raise ValueError("update must be a non-negative integer")
        if update <= self.start_update:
            return float(self.start_sigma_seconds)
        if update >= self.end_update:
            return float(self.end_sigma_seconds)
        progress = (update - self.start_update) / (self.end_update - self.start_update)
        alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(self.start_sigma_seconds + alpha * (self.end_sigma_seconds - self.start_sigma_seconds))

    def to_contract(self) -> dict:
        return {
            "type": "cosine",
            **asdict(self),
            "update_unit": "optimizer_update_1_based",
            "resume": "sigma_at(checkpoint.update + 1); never use parent EMA.step",
        }


def schedule_from_config(config) -> CosineBandSchedule | None:
    raw = config.get("temporal_band_curriculum")
    return None if raw is None else CosineBandSchedule.from_config(raw)
