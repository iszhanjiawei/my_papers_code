"""Text-conditioned flow matching over fixed Semantic-VAE audio latents.

No video tensors, video-length proxies, or visual CFG branches are used.
The model state remains compatible with CFM_VT for downstream AVT tuning.
"""

from __future__ import annotations

import math
from random import random

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from f5_tts.model.cfm import CFM
from f5_tts.model.utils import get_epss_timesteps, lens_to_mask, list_str_to_idx, list_str_to_tensor


class CFMTextAudio(CFM):
    def __init__(self, *args, text_drop_prob=0.0, ctc_lambda=0.03, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.transformer, "text_audio_only", False):
            raise ValueError("CFMTextAudio requires a text_audio_only backbone")
        self.text_drop_prob = float(text_drop_prob)
        self.ctc_lambda = float(ctc_lambda)
        for name in ("audio_drop_prob", "cond_drop_prob", "text_drop_prob"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cond_drop_prob + self.text_drop_prob > 1:
            raise ValueError("All-condition and text-only dropout probabilities must sum to at most 1")
        if not math.isfinite(self.ctc_lambda) or self.ctc_lambda < 0:
            raise ValueError("ctc_lambda must be finite and nonnegative")
        low, high = self.frac_lengths_mask
        if not 0 < low <= high <= 1:
            raise ValueError("frac_lengths_mask must satisfy 0 < low <= high <= 1")

    def _check_audio(self, audio, *, allow_empty=False):
        if audio.ndim != 3 or audio.shape[-1] != self.num_channels or not audio.is_floating_point():
            raise ValueError(f"Expected floating-point latents [batch, frames, {self.num_channels}]")
        if audio.shape[0] == 0 or (audio.shape[1] == 0 and not allow_empty):
            raise ValueError("Audio batch and training sequences must be nonempty")

    @staticmethod
    def _lengths(lengths, batch, maximum, device, *, allow_zero=False):
        if lengths is None:
            lengths = torch.full((batch,), maximum, device=device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=device)
            if lengths.is_floating_point() or lengths.dtype == torch.bool:
                raise ValueError("Lengths must be integer counts")
            lengths = lengths.long()
        if lengths.shape != (batch,) or torch.any(lengths < (0 if allow_zero else 1)) or torch.any(lengths > maximum):
            raise ValueError(f"Lengths must have shape [{batch}] and lie within the padded sequence")
        return lengths

    def _text_inputs(self, text, batch, device, text_lens=None):
        if isinstance(text, list):
            if len(text) != batch or not all(isinstance(item, str) and item for item in text):
                raise ValueError("Text must contain one nonempty string per sample")
            if self.vocab_char_map is not None:
                unknown = sorted({character for item in text for character in item if character not in self.vocab_char_map})
                if unknown:
                    raise ValueError(f"Text contains characters absent from the fixed vocabulary: {unknown!r}")
                text = list_str_to_idx(text, self.vocab_char_map)
            else:
                text = list_str_to_tensor(text)
        if not isinstance(text, torch.Tensor) or text.ndim != 2 or text.shape[0] != batch:
            raise ValueError("Text must be a [batch, tokens] tensor or a list of strings")
        if text.is_floating_point() or text.dtype == torch.bool:
            raise ValueError("Text token IDs must be integers")
        text = text.to(device=device, dtype=torch.long)
        if text_lens is None:
            text_lens = (text != -1).sum(dim=1)
        text_lens = self._lengths(text_lens, batch, text.shape[1], device)
        text_mask = lens_to_mask(text_lens, length=text.shape[1])
        if torch.any(text[text_mask] < 0) or torch.any(
            text[text_mask] >= self.transformer.text_embed.text_embed.num_embeddings - 1
        ):
            raise ValueError("Nonpadding text tokens are outside the fixed vocabulary")
        text = text.masked_fill(~text_mask, -1)
        return text, text_lens, text_mask

    def forward(
        self, inp, text, speaker_embedding=None, *, lens=None, text_lens=None, noise_scheduler=None,
    ):
        """Return (loss, component_losses, cond, pred), matching Trainer_VT.

        ``inp`` is normalized [B,T,C] latent audio; text describes the complete
        utterance, including the randomly retained audio prompt. Speaker and
        audio-prompt dropout are coupled. CTC weight may be changed per update.
        """
        if noise_scheduler is not None:
            raise ValueError("CFMTextAudio uses uniform flow time; noise_scheduler must be None")
        self._check_audio(inp)
        batch, length = inp.shape[:2]
        lens = self._lengths(lens, batch, length, inp.device)
        mask = lens_to_mask(lens, length=length)
        text, text_lens, text_mask = self._text_inputs(text, batch, inp.device, text_lens)
        clean = inp.masked_fill(~mask[..., None], 0.0)
        fractions = torch.empty((batch,), device=inp.device).uniform_(*self.frac_lengths_mask)
        span_lengths = (fractions * lens).long().clamp(min=1)
        starts = (torch.rand((batch,), device=inp.device) * (lens - span_lengths + 1)).long()
        positions = torch.arange(length, device=inp.device)[None]
        generation_mask = (positions >= starts[:, None]) & (positions < (starts + span_lengths)[:, None]) & mask
        noise = torch.randn_like(clean).masked_fill(~mask[..., None], 0.0)
        time = torch.rand((batch,), device=inp.device, dtype=inp.dtype)
        t = time[:, None, None]
        noisy = (1 - t) * noise + t * clean
        target = clean - noise
        cond = clean.masked_fill(generation_mask[..., None], 0.0)
        drop_audio = random() < self.audio_drop_prob
        draw = random()
        drop_all = draw < self.cond_drop_prob
        drop_text = draw < self.cond_drop_prob + self.text_drop_prob
        drop_audio = drop_audio or drop_all
        pred, intermediates = self.transformer(
            x=noisy, cond=cond, text=text, video=None, time=time,
            mask=mask, text_mask=text_mask, generation_mask=generation_mask,
            speaker_embedding=speaker_embedding, drop_audio_cond=drop_audio,
            drop_speaker=drop_audio, drop_text=drop_text,
            compute_ctc=self.ctc_lambda > 0,
        )
        flow_loss = F.mse_loss(pred[generation_mask].float(), target[generation_mask].float())
        loss = flow_loss
        components = {"diff_loss": flow_loss.detach().item(), "ctc_loss": 0.0}
        if self.ctc_lambda > 0:
            if not intermediates:
                raise RuntimeError("Positive CTC weight requires configured CTC taps")
            ctc = sum(
                F.ctc_loss(
                    item["z_tilde"].float().transpose(0, 1).log_softmax(-1),
                    text, item["z_lens"], text_lens,
                    blank=self.transformer.text_embed.text_embed.num_embeddings,
                    reduction="mean", zero_infinity=True,
                ) for item in intermediates.values()
            ) / len(intermediates)
            loss = loss + self.ctc_lambda * ctc
            components["ctc_loss"] = ctc.detach().item()
        return loss, components, cond, pred

    @torch.no_grad()
    def sample(
        self, cond, text, duration, speaker_embedding=None, *, lens=None,
        text_lens=None, steps=32, cfg_strength=1.0, sway_sampling_coef=None,
        seed=None, max_duration=4096, use_epss=False, no_ref_audio=False,
    ):
        """Generate normalized latent audio; return (full_audio, trajectory).

        ``duration`` counts total frames INCLUDING the prefix prompt. ``text``
        describes prompt plus target. ``lens`` specifies valid prefix frames;
        [B,0,C] plus zero lens supports generation with no audio prefix.
        The caller removes the prefix and reverses latent normalization before
        decoding. CFG follows AlignDiT: cond + strength * (cond - uncond).
        """
        self._check_audio(cond, allow_empty=True)
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(cfg_strength) or cfg_strength < 0:
            raise ValueError("cfg_strength must be finite and nonnegative")
        if sway_sampling_coef is not None and not -1 <= sway_sampling_coef <= 1:
            raise ValueError("sway_sampling_coef must lie in [-1, 1]")
        device = self.device
        cond = cond.to(device=device, dtype=next(self.parameters()).dtype)
        batch, prompt_length = cond.shape[:2]
        lens = self._lengths(lens, batch, prompt_length, device, allow_zero=True)
        durations = torch.as_tensor(duration, device=device)
        if durations.is_floating_point() or durations.dtype == torch.bool:
            raise ValueError("duration must contain integer frame counts")
        if durations.ndim == 0:
            durations = durations.expand(batch)
        durations = self._lengths(durations, batch, max_duration, device)
        if torch.any(durations <= lens):
            raise ValueError("Each total duration must exceed its audio prompt length")
        total_length = int(durations.max())
        text, _, text_mask = self._text_inputs(text, batch, device, text_lens)
        valid = lens_to_mask(durations, length=total_length)
        prompt_mask = lens_to_mask(lens, length=total_length)
        if no_ref_audio:
            prompt_mask = torch.zeros_like(prompt_mask)
        generation_mask = valid & ~prompt_mask
        cond = F.pad(cond[:, :total_length], (0, 0, 0, max(0, total_length - prompt_length)))
        step_cond = cond.masked_fill(~prompt_mask[..., None], 0.0)
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.to(device=device, dtype=torch.float32)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        initial = torch.randn((batch, total_length, self.num_channels), device=device,
                              dtype=cond.dtype, generator=generator).masked_fill(~valid[..., None], 0.0)

        def velocity(time, state):
            common = dict(
                x=state, cond=step_cond, text=text, video=None, time=time,
                mask=valid, text_mask=text_mask, generation_mask=generation_mask,
                speaker_embedding=speaker_embedding, compute_ctc=False, cache=True,
            )
            prediction, _ = self.transformer(
                **common, drop_audio_cond=no_ref_audio, drop_speaker=no_ref_audio,
            )
            if cfg_strength > 0:
                null_prediction, _ = self.transformer(
                    **common, drop_audio_cond=True, drop_speaker=True, drop_text=True,
                )
                prediction = prediction + cfg_strength * (prediction - null_prediction)
            return prediction.masked_fill(~valid[..., None], 0.0)

        times = get_epss_timesteps(steps, device=device, dtype=cond.dtype) if use_epss else torch.linspace(
            0, 1, steps + 1, device=device, dtype=cond.dtype,
        )
        if sway_sampling_coef is not None:
            times = times + sway_sampling_coef * (torch.cos(torch.pi / 2 * times) - 1 + times)
        was_training = self.training
        self.eval()
        try:
            trajectory = odeint(velocity, initial, times, **self.odeint_kwargs)
            output = torch.where(prompt_mask[..., None], step_cond, trajectory[-1])
            output = output.masked_fill(~valid[..., None], 0.0)
        finally:
            self.transformer.clear_cache()
            self.train(was_training)
        return output, trajectory
