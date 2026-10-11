from __future__ import annotations

import copy
import gc
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


# trainer
class Trainer_VT(Trainer):
    @staticmethod
    def _adapt_mmaudio_audio_stream(src_sd, tgt_sd, state_prefix, n_mm_layers):
        """Map a native audio DiT into the audio branch of AVT joint blocks.

        The audio-only checkpoint stores separate Q/K/V linear projections and
        a linear attention output.  MMAudio joint blocks use fused QKV and a
        three-tap Conv1d output projection.  The conversion below is exact for
        Q/K/V and embeds the linear output in the centre convolution tap.

        The source GELU FFN is intentionally not forced into the structurally
        different SwiGLU ConvMLP.  Its three AdaLN FFN chunks are consequently
        kept at zero, so the new FFN is introduced through a learnable gate
        instead of perturbing the pretrained audio path immediately.
        """

        adapted = dict(src_sd)
        mapped_keys = []

        def _put(dst_key, value):
            if dst_key not in tgt_sd:
                raise KeyError(f"audio warm-start target is missing {dst_key}")
            if value.shape != tgt_sd[dst_key].shape:
                raise ValueError(
                    f"audio warm-start shape mismatch for {dst_key}: "
                    f"{tuple(value.shape)} vs {tuple(tgt_sd[dst_key].shape)}"
                )
            adapted[dst_key] = value
            mapped_keys.append(dst_key)

        for layer_index in range(n_mm_layers):
            source = f"{state_prefix}transformer.transformer_blocks.{layer_index}."
            target = f"{source}audio_block."

            source_adaln_weight = src_sd[source + "attn_norm.linear.weight"]
            source_adaln_bias = src_sd[source + "attn_norm.linear.bias"]
            target_adaln_weight = tgt_sd[target + "attn_norm.linear.weight"]
            target_adaln_bias = tgt_sd[target + "attn_norm.linear.bias"]
            if source_adaln_weight.shape != target_adaln_weight.shape:
                raise ValueError(f"incompatible AdaLN weight at joint layer {layer_index}")
            if source_adaln_bias.shape != target_adaln_bias.shape:
                raise ValueError(f"incompatible AdaLN bias at joint layer {layer_index}")

            modulation_dim = source_adaln_weight.shape[0] // 6
            adaln_weight = torch.zeros(
                target_adaln_weight.shape,
                dtype=source_adaln_weight.dtype,
                device=source_adaln_weight.device,
            )
            adaln_bias = torch.zeros(
                target_adaln_bias.shape,
                dtype=source_adaln_bias.dtype,
                device=source_adaln_bias.device,
            )
            # shift_msa, scale_msa, gate_msa are compatible.  The remaining
            # shift/scale/gate chunks belong to the incompatible source FFN.
            adaln_weight[: 3 * modulation_dim].copy_(source_adaln_weight[: 3 * modulation_dim])
            adaln_bias[: 3 * modulation_dim].copy_(source_adaln_bias[: 3 * modulation_dim])
            _put(target + "attn_norm.linear.weight", adaln_weight)
            _put(target + "attn_norm.linear.bias", adaln_bias)

            qkv_weight = torch.cat(
                [
                    src_sd[source + "attn.to_q.weight"],
                    src_sd[source + "attn.to_k.weight"],
                    src_sd[source + "attn.to_v.weight"],
                ],
                dim=0,
            )
            qkv_bias = torch.cat(
                [
                    src_sd[source + "attn.to_q.bias"],
                    src_sd[source + "attn.to_k.bias"],
                    src_sd[source + "attn.to_v.bias"],
                ],
                dim=0,
            )
            _put(target + "qkv.weight", qkv_weight)
            _put(target + "qkv.bias", qkv_bias)

            source_out_weight = src_sd[source + "attn.to_out.0.weight"]
            target_out_weight = tgt_sd[target + "out_proj.weight"]
            if target_out_weight.ndim != 3 or target_out_weight.shape[-1] % 2 != 1:
                raise ValueError(
                    f"expected an odd-width Conv1d output projection at joint layer {layer_index}, "
                    f"got {tuple(target_out_weight.shape)}"
                )
            out_weight = torch.zeros(
                target_out_weight.shape,
                dtype=source_out_weight.dtype,
                device=source_out_weight.device,
            )
            centre = target_out_weight.shape[-1] // 2
            out_weight[:, :, centre].copy_(source_out_weight)
            _put(target + "out_proj.weight", out_weight)
            _put(target + "out_proj.bias", src_sd[source + "attn.to_out.0.bias"])

        return adapted, mapped_keys

    def load_pretrained(self, pretrained_path):
        self.accelerator.wait_for_everyone()
        checkpoint = torch.load(pretrained_path, weights_only=True, map_location="cpu")

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        def _is_main_print(*args, **kwargs):
            if getattr(self, "is_main", True):
                print(*args, **kwargs)

        def _safe_merge(src_sd, tgt_sd, prefix=""):
            out = {}
            loaded_tensors = 0
            loaded_parameters = 0
            shape_mismatches = []
            extra_keys = 0
            for k, v in src_sd.items():
                if k not in tgt_sd:
                    extra_keys += 1
                    continue
                tv = tgt_sd[k]
                if v.shape != tv.shape:
                    shape_mismatches.append((k, tuple(v.shape), tuple(tv.shape)))
                    continue
                out[k] = v
                loaded_tensors += 1
                loaded_parameters += tv.numel()
            for k in tgt_sd:
                if k not in out:
                    out[k] = tgt_sd[k]
            total_parameters = sum(tensor.numel() for tensor in tgt_sd.values())
            coverage = 100.0 * loaded_parameters / total_parameters
            _is_main_print(
                f"[INFO] {prefix}audio-pretrained coverage: {loaded_tensors}/{len(tgt_sd)} tensors, "
                f"{loaded_parameters}/{total_parameters} parameters ({coverage:.2f}%); "
                f"{extra_keys} source-only keys ignored; "
                f"{len(shape_mismatches)} shape mismatches kept at target initialization"
            )
            for key, source_shape, target_shape in shape_mismatches[:12]:
                _is_main_print(
                    f"[WARN] {prefix}{key}: shape mismatch {source_shape} vs {target_shape}, skip"
                )
            return out

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        transformer = getattr(unwrapped_model, "transformer", None)
        n_mm_layers = int(getattr(transformer, "n_mm_layers", 0))

        if self.is_main:
            ema_state_dict = self.ema_model.state_dict()
            ckpt_ema_state_dict = copy.deepcopy(checkpoint["ema_model_state_dict"])
            if n_mm_layers:
                ckpt_ema_state_dict, mapped_keys = self._adapt_mmaudio_audio_stream(
                    ckpt_ema_state_dict,
                    ema_state_dict,
                    state_prefix="ema_model.",
                    n_mm_layers=n_mm_layers,
                )
                _is_main_print(
                    f"[INFO] [EMA] mapped native audio attention into {n_mm_layers} "
                    f"MMAudio joint blocks ({len(mapped_keys)} target tensors)"
                )
            change_key = "ema_model.transformer.input_embed.proj.weight"
            src = ckpt_ema_state_dict[change_key]
            tgt = ema_state_dict[change_key]
            if src.size(0) == tgt.size(0) and src.size(1) < tgt.size(1):
                _is_main_print(f"[INFO] [EMA] expand {change_key}: {src.shape} -> {tgt.shape}")
                ckpt_ema_state_dict[change_key] = torch.cat(
                    [src, tgt[:, src.size(1) :].to(device=src.device, dtype=src.dtype)], dim=1
                )

            ema_filtered = _safe_merge(ckpt_ema_state_dict, ema_state_dict, prefix="[EMA] ")
            self.ema_model.load_state_dict(ema_filtered)

        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "update", "step"]
        }
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        if n_mm_layers:
            checkpoint["model_state_dict"], mapped_keys = self._adapt_mmaudio_audio_stream(
                checkpoint["model_state_dict"],
                state_dict,
                state_prefix="",
                n_mm_layers=n_mm_layers,
            )
            _is_main_print(
                f"[INFO] mapped native audio attention into {n_mm_layers} "
                f"MMAudio joint blocks ({len(mapped_keys)} target tensors)"
            )
        change_key = "transformer.input_embed.proj.weight"
        src = checkpoint["model_state_dict"][change_key]
        tgt = state_dict[change_key]
        if src.size(0) == tgt.size(0) and src.size(1) < tgt.size(1):
            _is_main_print(f"[INFO] expand {change_key}: {src.shape} -> {tgt.shape}")
            checkpoint["model_state_dict"][change_key] = torch.cat(
                [src, tgt[:, src.size(1) :].to(device=src.device, dtype=src.dtype)], dim=1
            )

        filtered = _safe_merge(checkpoint["model_state_dict"], state_dict)
        self.accelerator.unwrap_model(self.model).load_state_dict(filtered)

        del checkpoint
        gc.collect()

    def finetune(self, pretrained_path, train_dataset: Dataset, num_workers=16, resumable_with_seed: int | None = None):
        self.load_pretrained(pretrained_path)
        self.train(train_dataset, num_workers, resumable_with_seed)

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int | None = None):
        if self.log_samples:
            from aligndit.script.eval.utils import load_vocoder
            from f5_tts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
                device=self.accelerator.device,
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
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
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
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

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
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

            # Set epoch for the batch sampler if it exists
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
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    text_lengths = batch["text_lengths"]
                    video = batch["video"]
                    video_lengths = batch["video_lengths"]

                    loss, loss_components, _cond, _pred = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
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
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item(), **loss_components)

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                    self.accelerator.log(loss_components, step=global_update)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
                        for k, v in loss_components.items():
                            self.writer.add_scalar(k, v, global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                video=torch.cat(
                                    [video[0][: ref_audio_len // 4], video[0][: ref_audio_len // 4]]
                                ).unsqueeze(0),
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                max_duration=5000,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)

                            gen_audio = vocoder(gen_mel_spec).squeeze(1).cpu()
                            ref_audio = vocoder(ref_mel_spec).squeeze(1).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
