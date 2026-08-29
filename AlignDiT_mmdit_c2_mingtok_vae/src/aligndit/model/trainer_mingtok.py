from __future__ import annotations

import math
import os

import torch
import torchaudio
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.trainer import Trainer
from f5_tts.model.utils import exists


class Trainer_MingTok(Trainer):
    """Original C2 trainer loop with MingTok latent batches and sample decoding."""

    def __init__(
        self,
        *args,
        mingtok_repo_path: str,
        mingtok_checkpoint_dir: str,
        ctc_warmup_start: int = 0,
        ctc_warmup_end: int = 0,
        **kwargs,
    ):
        if ctc_warmup_start < 0:
            raise ValueError(f"ctc_warmup_start must be non-negative, got {ctc_warmup_start}")
        if ctc_warmup_end < ctc_warmup_start:
            raise ValueError(
                f"ctc_warmup_end must be >= ctc_warmup_start, got {ctc_warmup_end} < {ctc_warmup_start}"
            )
        self.mingtok_repo_path = mingtok_repo_path
        self.mingtok_checkpoint_dir = mingtok_checkpoint_dir
        self.ctc_warmup_start = int(ctc_warmup_start)
        self.ctc_warmup_end = int(ctc_warmup_end)
        super().__init__(*args, **kwargs)
        self.ctc_lambda_target = float(self.accelerator.unwrap_model(self.model).ctc_lambda)

    def _ctc_lambda_for_update(self, update: int) -> float:
        if update <= self.ctc_warmup_start:
            return 0.0
        if update >= self.ctc_warmup_end or self.ctc_warmup_end == self.ctc_warmup_start:
            return self.ctc_lambda_target
        progress = (update - self.ctc_warmup_start) / (self.ctc_warmup_end - self.ctc_warmup_start)
        return self.ctc_lambda_target * progress

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int | None = None):
        codec = None
        if self.log_samples:
            from aligndit.model.mingtok_codec import MingTokAcousticCodec
            from f5_tts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef

            # Only the process that writes samples needs the frozen decoder.  It
            # remains external to the AlignDiT model, optimizer, EMA, and saved
            # training state.
            if self.accelerator.is_local_main_process:
                codec = MingTokAcousticCodec(
                    repo_path=self.mingtok_repo_path,
                    checkpoint_dir=self.mingtok_checkpoint_dir,
                    device=self.accelerator.device,
                    dtype=torch.bfloat16,
                    backend="eager",
                    load_encoder=False,
                    load_decoder=True,
                )
            target_sample_rate = 16000
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)
            elif (
                hasattr(train_dataloader, "batch_sampler")
                and hasattr(train_dataloader.batch_sampler, "batch_sampler")
                and hasattr(train_dataloader.batch_sampler.batch_sampler, "set_epoch")
            ):
                train_dataloader.batch_sampler.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    ctc_lambda = self._ctc_lambda_for_update(global_update + 1)
                    self.accelerator.unwrap_model(self.model).ctc_lambda = ctc_lambda

                    text_inputs = batch["text"]
                    audio_latent = batch["audio_latent"].permute(0, 2, 1)
                    audio_latent_lengths = batch["audio_latent_lengths"]
                    text_lengths = batch["text_lengths"]
                    video = batch["video"]
                    video_lengths = batch["video_lengths"]

                    loss, loss_components, _cond, _pred = self.model(
                        audio_latent,
                        text=text_inputs,
                        lens=audio_latent_lengths,
                        text_lens=text_lengths,
                        video=video,
                        video_lens=video_lengths,
                        noise_scheduler=self.noise_scheduler,
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update), loss=loss.item(), ctc_lambda=ctc_lambda, **loss_components
                    )

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], "ctc_lambda": ctc_lambda},
                        step=global_update,
                    )
                    self.accelerator.log(loss_components, step=global_update)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
                        self.writer.add_scalar("ctc_lambda", ctc_lambda, global_update)
                        for key, value in loss_components.items():
                            self.writer.add_scalar(key, value, global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = audio_latent_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        audio_video_ratio = self.accelerator.unwrap_model(self.model).audio_video_ratio
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=audio_latent[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                video=torch.cat(
                                    [
                                        video[0][: ref_audio_len // audio_video_ratio],
                                        video[0][: ref_audio_len // audio_video_ratio],
                                    ]
                                ).unsqueeze(0),
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                max_duration=5000,
                            )
                            generated = generated.to(torch.float32)
                            gen_latent = generated[:, ref_audio_len:, :]
                            ref_latent = audio_latent[0][:ref_audio_len].unsqueeze(0).to(torch.float32)

                            gen_audio = codec.decode(gen_latent).squeeze(1).to(torch.float32).cpu()
                            ref_audio = codec.decode(ref_latent).squeeze(1).to(torch.float32).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()


__all__ = ["Trainer_MingTok"]
