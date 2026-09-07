"""Speaker C2 training with independent TPCA and audio-CTC schedules."""
from aligndit.model.trainer_semantic_vae_direct_speaker import SemanticVaeDirectC2SpeakerTrainer


class SemanticVaeDirectC2SpeakerTPCATrainer(SemanticVaeDirectC2SpeakerTrainer):
    def _before_update(self, global_update):
        super()._before_update(global_update)
        self.accelerator.unwrap_model(self.model).transformer.set_tpca_step(global_update)

    def _after_update(self, completed_updates):
        # Integer schedules are completed optimizer updates, never EMA averages.
        self.accelerator.unwrap_model(self.model).transformer.set_tpca_step(completed_updates)
        if self.is_main:
            self.ema_model.ema_model.transformer.set_tpca_step(completed_updates)

    def load_checkpoint(self):
        update = super().load_checkpoint()
        model = self.accelerator.unwrap_model(self.model)
        if int(model.transformer.tpca_step.item()) != update:
            raise RuntimeError("Online TPCA step differs from resumed optimizer update")
        if self.is_main and int(self.ema_model.ema_model.transformer.tpca_step.item()) != update:
            raise RuntimeError("EMA TPCA step differs from resumed optimizer update")
        return update

    def save_checkpoint(self, update, last=False):
        self._after_update(update)
        super().save_checkpoint(update, last=last)
        if self.is_main and self.logger == "tensorboard":
            self.writer.flush()
