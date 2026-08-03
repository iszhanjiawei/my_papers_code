"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aligndit.model.modules import DownsampleLayer
from f5_tts.model.backbones.dit import ConvPositionEmbedding, DiT


class InputEmbedding_noText(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        drop_audio_cond=False,
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond), dim=-1))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.conv_pos_embed(x, mask=mask) + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class DiT_noText(DiT):
    def __init__(
        self,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        layer_indices=[12],
        projector_dim=None,
        z_dim=None,
        projector_strides=(2, 1),
        mask_during_training=False,
        padding_safe_projector=False,
    ):
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
            long_skip_connection=long_skip_connection,
            checkpoint_activations=checkpoint_activations,
        )
        self.text_embed = None
        self.input_embed = InputEmbedding_noText(mel_dim, dim)
        self.mask_during_training = mask_during_training
        self.padding_safe_projector = padding_safe_projector

        projector_dim = self.dim if projector_dim is None else projector_dim
        z_dim = self.dim if z_dim is None else z_dim
        if isinstance(projector_strides, (str, bytes)):
            raise ValueError("projector_strides must be an iterable of positive integers")
        try:
            projector_strides = tuple(projector_strides)
        except TypeError as error:
            raise ValueError("projector_strides must be an iterable of positive integers") from error
        if not projector_strides or any(not isinstance(stride, int) or stride <= 0 for stride in projector_strides):
            raise ValueError(f"projector_strides must contain positive integers: {projector_strides}")
        self.layer_map = {v: i for i, v in enumerate(layer_indices)}
        self.projectors = nn.ModuleList(
            [DownsampleLayer(tuple(projector_strides), self.dim, projector_dim, z_dim) for _ in self.layer_map]
        )

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        drop_audio_cond: bool = False,
        mask=None,
    ):
        x = self.input_embed(x, cond, drop_audio_cond=drop_audio_cond, mask=mask)

        return x

    def run_projector(self, projector, x, lens):
        if not self.padding_safe_projector:
            return projector(x, lens)

        lengths = lens.detach().cpu().tolist()
        if any(length <= 0 or length > x.shape[1] for length in lengths):
            raise ValueError(f"Projector lengths must be within [1, {x.shape[1]}], got {lengths}")
        projected = []
        projected_lens = []
        for index, length in enumerate(lengths):
            sample, sample_lens = projector(x[index : index + 1, : int(length)], lens[index : index + 1])
            projected.append(sample)
            projected_lens.append(sample_lens)
        max_length = max(sample.shape[1] for sample in projected)
        projected = [F.pad(sample, (0, 0, 0, max_length - sample.shape[1])) for sample in projected]
        return torch.cat(projected, dim=0), torch.cat(projected_lens, dim=0)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
        drop_audio_cond: bool = False,  # cfg for cond audio
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        if self.padding_safe_projector and not cache and self.layer_map and mask is None:
            raise ValueError("padding_safe_projector requires an explicit valid-frame mask")

        block_mask = mask if self.mask_during_training or not self.training else None
        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(x, cond, mask=block_mask)
            x_uncond = self.get_input_embed(x, cond, drop_audio_cond=True, mask=block_mask)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            block_mask = torch.cat((block_mask, block_mask), dim=0) if block_mask is not None else None
        else:
            x = self.get_input_embed(x, cond, drop_audio_cond=drop_audio_cond, mask=block_mask)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        intermediates = {}
        for layer_i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    t,
                    block_mask,
                    rope,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    t,
                    mask=block_mask,
                    rope=rope,
                )

            # DiTBlock masks attention outputs, but its time-conditioned FFN can
            # repopulate padded tokens. Keep padding zero so the convolutional
            # alignment projector cannot leak those values into valid boundary frames.
            if block_mask is not None:
                x = x.masked_fill(~block_mask.unsqueeze(-1), 0.0)

            if not cache and layer_i in self.layer_map:  # hack
                projector = self.projectors[self.layer_map[layer_i]]
                lens = (
                    mask.sum(dim=1)
                    if mask is not None
                    else torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
                )
                z_tilde, z_lens = self.run_projector(projector, x, lens)
                intermediates[layer_i] = {"z_tilde": z_tilde, "z_lens": z_lens}

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output, intermediates
