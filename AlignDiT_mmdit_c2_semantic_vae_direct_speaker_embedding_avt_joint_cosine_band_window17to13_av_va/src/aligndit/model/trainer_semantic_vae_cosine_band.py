"""Keep AVT training unchanged except for its scheduled Gaussian AV/VA width."""

from __future__ import annotations

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.temporal_band_curriculum import CosineBandSchedule
from aligndit.model.trainer_semantic_vae_fixed_band import SemanticVaeFixedBandTrainer


class SemanticVaeCosineBandTrainer(SemanticVaeFixedBandTrainer):
    def __init__(self, *args, temporal_band_schedule: CosineBandSchedule, **kwargs) -> None:
        if not isinstance(temporal_band_schedule, CosineBandSchedule):
            raise TypeError("temporal_band_schedule must be a validated CosineBandSchedule")
        self.temporal_band_schedule = temporal_band_schedule
        super().__init__(*args, **kwargs)
        self._set_band_sigma(self.temporal_band_schedule.sigma_at(0))

    def _set_band_sigma(self, sigma: float) -> None:
        models = [self.accelerator.unwrap_model(self.model)]
        if self.is_main:
            # EMA copies tensors, but sigma is deliberately a Python attribute.
            models.append(self.ema_model.ema_model)
        for model in models:
            band = model.transformer.temporal_band
            if not isinstance(band, FixedTemporalBand):
                raise TypeError("Cosine curriculum requires a parameter-free FixedTemporalBand")
            band.sigma_seconds = sigma
        self.current_band_sigma_seconds = sigma

    def _before_update(self, global_update: int) -> None:
        # Preserve speaker bookkeeping and the inherited CTC warmup. Repeated
        # microbatches in one accumulation group use the same optimizer update.
        super()._before_update(global_update)
        self._set_band_sigma(self.temporal_band_schedule.sigma_at(global_update + 1))
