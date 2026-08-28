"""C2 conditional flow matching over raw MingTok latents.

This class deliberately preserves the original C2 flow-matching, masking,
classifier-free guidance, CTC, and ODE behavior.  It only replaces the mel
front end with an explicit ``[B, T, 64]`` latent contract and makes the 2:1
audio/video frame-rate ratio an asserted model property.
"""

from __future__ import annotations

import torch
from torch import nn

from aligndit.model.cfm_vt import CFM_VT


class _MingTokLatentSpec(nn.Module):
    """Metadata-only stand-in for the mel module expected by the base CFM."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.n_mel_channels = num_channels
        self.target_sample_rate = 16_000
        self.hop_length = 320

    def forward(self, _waveform):
        raise ValueError(
            "CFM_MingTok does not encode waveforms. Pass cached MingTok latents "
            "with shape [B, T, 64]."
        )


class CFM_MingTok(CFM_VT):
    """The original C2 CFM operating directly on 64D, 50 Hz MingTok latents."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        num_channels: int = 64,
        audio_video_ratio: int = 2,
        ctc_lambda: float = 0.1,
        vocab_char_map: dict[str, int] | None = None,
        **kwargs,
    ):
        if num_channels != 64:
            raise ValueError(f"MingTok C2 requires num_channels=64, got {num_channels}")
        if audio_video_ratio != 2:
            raise ValueError(f"MingTok C2 requires audio_video_ratio=2, got {audio_video_ratio}")

        transformer_ratio = getattr(transformer, "audio_video_ratio", None)
        if transformer_ratio != audio_video_ratio:
            raise ValueError(
                "CFM/backbone audio_video_ratio mismatch: "
                f"CFM={audio_video_ratio}, backbone={transformer_ratio}"
            )

        transformer_channels = getattr(getattr(transformer, "proj_out", None), "out_features", None)
        if transformer_channels is not None and transformer_channels != num_channels:
            raise ValueError(
                f"CFM/backbone channel mismatch: CFM={num_channels}, backbone output={transformer_channels}"
            )

        forbidden = {"mel_spec_module", "mel_spec_kwargs", "num_channels"}.intersection(kwargs)
        if forbidden:
            raise TypeError(f"CFM_MingTok manages {sorted(forbidden)} internally")

        super().__init__(
            transformer=transformer,
            num_channels=num_channels,
            mel_spec_module=_MingTokLatentSpec(num_channels),
            mel_spec_kwargs={},
            vocab_char_map=vocab_char_map,
            audio_video_ratio=audio_video_ratio,
            ctc_lambda=ctc_lambda,
            **kwargs,
        )

    def _validate_latent(self, latent: torch.Tensor, name: str) -> None:
        if latent.ndim != 3:
            raise ValueError(f"{name} must have shape [B, T, 64], got {tuple(latent.shape)}")
        if latent.shape[-1] != self.num_channels:
            raise ValueError(
                f"{name} channel mismatch: expected {self.num_channels}, got {latent.shape[-1]}"
            )

    def _validate_video(self, video: torch.Tensor) -> None:
        if video is None or video.ndim != 3:
            shape = None if video is None else tuple(video.shape)
            raise ValueError(f"video must have shape [B, T_video, D], got {shape}")

    def sample(self, cond, text, duration, video, **kwargs):
        self._validate_latent(cond, "cond")
        self._validate_video(video)
        if cond.shape[0] != video.shape[0]:
            raise ValueError(f"cond/video batch mismatch: {cond.shape[0]} vs {video.shape[0]}")

        lens = kwargs.get("lens")
        if lens is not None and torch.any(torch.as_tensor(lens) % self.audio_video_ratio != 0):
            raise ValueError(f"prompt latent lengths must be divisible by {self.audio_video_ratio}")

        return super().sample(cond=cond, text=text, duration=duration, video=video, **kwargs)

    def forward(
        self,
        inp,
        text,
        video,
        *,
        lens=None,
        text_lens=None,
        video_lens=None,
        noise_scheduler=None,
    ):
        self._validate_latent(inp, "inp")
        self._validate_video(video)
        if inp.shape[0] != video.shape[0]:
            raise ValueError(f"inp/video batch mismatch: {inp.shape[0]} vs {video.shape[0]}")
        if inp.shape[1] != video.shape[1] * self.audio_video_ratio:
            raise ValueError(
                "padded MingTok/video lengths must preserve the 2:1 contract, got "
                f"audio={inp.shape[1]}, video={video.shape[1]}"
            )
        if lens is not None and video_lens is not None:
            audio_lens = torch.as_tensor(lens)
            expected_audio_lens = torch.as_tensor(video_lens, device=audio_lens.device) * self.audio_video_ratio
            if not torch.equal(audio_lens, expected_audio_lens):
                raise ValueError(
                    "MingTok/video lengths must preserve the 2:1 contract, got "
                    f"audio={audio_lens.tolist()}, video={torch.as_tensor(video_lens).tolist()}"
                )

        return super().forward(
            inp,
            text=text,
            video=video,
            lens=lens,
            text_lens=text_lens,
            video_lens=video_lens,
            noise_scheduler=noise_scheduler,
        )


__all__ = ["CFM_MingTok"]
