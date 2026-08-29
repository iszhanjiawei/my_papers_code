"""MMAudio-style three-stream MM-DiT for implicit-alignment dubbing.

The first ``n_mm_layers`` blocks keep independent audio, video, and ordered
text states.  Each stream produces its own Q/K/V, then all three are
concatenated along the token axis for one joint attention operation.  Audio
and video use aligned physical-time RoPE; text keeps the character encoder's
ordinal position features and intentionally receives no joint-level RoPE.

After the joint stage, only the audio stream is refined.  As in MMAudio, the
last joint block treats video and text as ``pre_only`` conditions: their Q/K/V
still condition audio, but their residual states and FFNs are not updated.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb

from aligndit.model.modules import DownsampleLayer
from cosyvoice.transformer.encoder import ConformerEncoder
from f5_tts.model.backbones.dit import ConvPositionEmbedding, DiT
from f5_tts.model.modules import (
    AdaLayerNorm,
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    DiTBlock,
    RMSNorm,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


def _require_mask(mask: torch.Tensor | None, batch: int, length: int, device, name: str) -> torch.Tensor:
    """Return a validated boolean prefix/key mask, defaulting to all-valid."""
    if mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be bool, got {mask.dtype}")
    if mask.shape != (batch, length):
        raise ValueError(f"{name} must have shape {(batch, length)}, got {tuple(mask.shape)}")
    if mask.device != device:
        raise ValueError(f"{name} must be on {device}, got {mask.device}")
    return mask


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Return the mutation counter; inference tensors intentionally have none."""
    try:
        return tensor._version
    except RuntimeError:
        return None


class ChannelLastConv1d(nn.Conv1d):
    """Conv1d accepting and returning ``[batch, time, channels]`` tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class SwiGLUFeedForward(nn.Module):
    """MMAudio-style MLP/ConvMLP with padding-safe temporal convolutions."""

    def __init__(
        self,
        dim: int,
        ff_mult: float,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        requested_hidden = int(2 * (dim * ff_mult) / 3)
        multiple_of = min(256, dim)
        hidden_dim = multiple_of * math.ceil(max(1, requested_hidden) / multiple_of)
        padding = kernel_size // 2
        projection = nn.Linear if kernel_size == 1 else ChannelLastConv1d
        projection_kwargs = {} if kernel_size == 1 else {"kernel_size": kernel_size, "padding": padding}

        self.w1 = projection(dim, hidden_dim, bias=False, **projection_kwargs)
        self.w2 = projection(hidden_dim, dim, bias=False, **projection_kwargs)
        self.w3 = projection(dim, hidden_dim, bias=False, **projection_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden = F.silu(self.w1(x)) * self.w3(x)
        if mask is not None:
            # This intermediate mask is required for ConvMLP: otherwise a value
            # produced just outside the valid prefix can leak back through w2.
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        out = self.dropout(self.w2(hidden))
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


class AudioInputEmbeddingMM(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        drop_audio_cond: bool = False,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond), dim=-1))
        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        return x


class VideoInputEmbeddingMM(nn.Module):
    """Embed native 25 Hz AV-HuBERT features without temporal upsampling."""

    def __init__(self, video_dim: int, out_dim: int, use_conformer: bool = False):
        super().__init__()
        self.vid_null_emb = nn.Parameter(torch.randn(1, video_dim) / video_dim**0.5)
        self.proj = nn.Linear(video_dim, out_dim)
        self.use_conformer = use_conformer
        if use_conformer:
            self.vid_conformer = ConformerEncoder(
                input_size=out_dim,
                output_size=out_dim,
                attention_heads=4,
                linear_units=1024,
                num_blocks=2,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer="linear",
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                use_cnn_module=False,
                macaron_style=False,
            )
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        video: torch.Tensor,
        *,
        drop_video: bool = False,
        video_mask: torch.Tensor | None = None,
        complementary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if complementary_mask is not None:
            if complementary_mask.dtype != torch.bool or complementary_mask.shape != video.shape[:2]:
                raise ValueError(
                    "complementary_mask must be bool with shape "
                    f"{tuple(video.shape[:2])}, got {tuple(complementary_mask.shape)}"
                )
            video = torch.where(
                complementary_mask.unsqueeze(-1),
                self.vid_null_emb.expand(video.size(0), video.size(1), -1),
                video,
            )
        if drop_video:
            video = self.vid_null_emb.expand(video.size(0), video.size(1), -1)

        if video_mask is None:
            video_lens = torch.full((video.size(0),), video.size(1), device=video.device, dtype=torch.long)
        else:
            video_lens = video_mask.sum(dim=1)

        video = self.proj(video)
        if video_mask is not None:
            video = video.masked_fill(~video_mask.unsqueeze(-1), 0.0)
        if self.use_conformer:
            video, _ = self.vid_conformer(video, video_lens)
        video = self.conv_pos_embed(video, mask=video_mask) + video
        if video_mask is not None:
            video = video.masked_fill(~video_mask.unsqueeze(-1), 0.0)
        return video


class TextInputEmbeddingMM(nn.Module):
    """Project ordered character features to the shared MM-DiT width."""

    def __init__(self, text_dim: int, out_dim: int, ff_mult: float, dropout: float):
        super().__init__()
        self.proj = nn.Linear(text_dim, out_dim)
        self.mlp = SwiGLUFeedForward(out_dim, ff_mult=ff_mult, kernel_size=1, dropout=dropout)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor | None) -> torch.Tensor:
        text = self.proj(text)
        if text_mask is not None:
            text = text.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        text = self.mlp(text, mask=text_mask)
        if text_mask is not None:
            text = text.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        return text


class OrderedTextEmbeddingMM(nn.Module):
    """Character encoder whose temporal aggregation excludes padded tokens."""

    def __init__(self, text_num_embeds: int, text_dim: int, conv_layers: int):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.precompute_max_pos = 4096
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(text_dim, self.precompute_max_pos),
            persistent=False,
        )
        self.text_blocks = nn.ModuleList([ConvNeXtV2Block(text_dim, text_dim * 2) for _ in range(conv_layers)])

    @staticmethod
    def _masked_convnext(
        block: ConvNeXtV2Block,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one ConvNeXt block without letting GRN observe padding."""
        expanded_mask = valid_mask.unsqueeze(-1)
        residual = x.masked_fill(~expanded_mask, 0.0)
        hidden = block.dwconv(residual.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.masked_fill(~expanded_mask, 0.0)
        hidden = block.norm(hidden)
        hidden = block.act(block.pwconv1(hidden))
        hidden = hidden.masked_fill(~expanded_mask, 0.0)
        hidden = block.grn(hidden)
        hidden = hidden.masked_fill(~expanded_mask, 0.0)
        hidden = block.pwconv2(hidden)
        hidden = hidden.masked_fill(~expanded_mask, 0.0)
        return (residual + hidden).masked_fill(~expanded_mask, 0.0)

    def forward(
        self,
        text: torch.Tensor,
        seq_len: int,
        drop_text: bool = False,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_mask  # Native-rate ordered text is never expanded to audio length.
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=-1)
        valid_mask = text != -1
        token_ids = text + 1
        if drop_text:
            token_ids = torch.zeros_like(token_ids)
        features = self.text_embed(token_ids)

        if len(self.text_blocks) > 0:
            batch_start = torch.zeros(text.shape[0], device=text.device, dtype=torch.long)
            position_indices = get_pos_embed_indices(
                batch_start,
                seq_len,
                max_pos=self.precompute_max_pos,
            )
            features = features + self.freqs_cis[position_indices]
            features = features.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            for block in self.text_blocks:
                features = self._masked_convnext(block, features, valid_mask)
        return features.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class MMDiTStream(nn.Module):
    """One modality-specific branch around a shared joint attention operation."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        dropout: float,
        qk_norm: str | None,
        *,
        kernel_size: int,
        pre_only: bool,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.pre_only = pre_only

        self.attn_norm = AdaLayerNorm_Final(dim) if pre_only else AdaLayerNorm(dim)
        self.qkv = nn.Linear(dim, self.inner_dim * 3)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if pre_only:
            self.out_proj = None
            self.ff_norm = None
            self.ff = None
        else:
            padding = kernel_size // 2
            if kernel_size == 1:
                self.out_proj = nn.Linear(self.inner_dim, dim)
            else:
                self.out_proj = ChannelLastConv1d(self.inner_dim, dim, kernel_size=kernel_size, padding=padding)
            self.out_dropout = nn.Dropout(dropout)
            self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff = SwiGLUFeedForward(dim, ff_mult=ff_mult, kernel_size=kernel_size, dropout=dropout)

    def pre_attention(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple | None]:
        if self.pre_only:
            norm = self.attn_norm(x, time_embed)
            modulation = None
        else:
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=time_embed)
            modulation = (gate_msa, shift_mlp, scale_mlp, gate_mlp)

        batch, length = norm.shape[:2]
        qkv = self.qkv(norm).view(batch, length, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        return (query, key, value), modulation

    def post_attention(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        modulation: tuple | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pre_only:
            return x
        gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        attn_out = attn_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        delta = self.out_dropout(self.out_proj(attn_out))
        delta = delta.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + gate_msa.unsqueeze(1) * delta
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm, mask=mask)
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class MMDiTBlockAVT(nn.Module):
    """MMAudio-style Audio/Video/Text joint-attention block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = True,
        av_ff_kernel_size: int = 3,
        text_ff_kernel_size: int = 1,
        pre_only: bool = False,
    ):
        super().__init__()
        if attn_backend != "torch":
            raise NotImplementedError("MMDiTBlockAVT currently supports torch SDPA only")
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        self.pre_only = pre_only

        self.audio_block = MMDiTStream(
            dim,
            heads,
            dim_head,
            ff_mult,
            dropout,
            qk_norm,
            kernel_size=av_ff_kernel_size,
            pre_only=False,
        )
        self.video_block = MMDiTStream(
            dim,
            heads,
            dim_head,
            ff_mult,
            dropout,
            qk_norm,
            kernel_size=av_ff_kernel_size,
            pre_only=pre_only,
        )
        self.text_block = MMDiTStream(
            dim,
            heads,
            dim_head,
            ff_mult,
            dropout,
            qk_norm,
            kernel_size=text_ff_kernel_size,
            pre_only=pre_only,
        )

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, rope):
        if rope is None:
            return query, key
        freqs, xpos_scale = rope
        q_scale, k_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        if self.pe_attn_head is None:
            return (
                apply_rotary_pos_emb(query, freqs, q_scale),
                apply_rotary_pos_emb(key, freqs, k_scale),
            )
        rope_heads = self.pe_attn_head
        if not 0 <= rope_heads <= self.heads:
            raise ValueError(f"pe_attn_head must be in [0, {self.heads}], got {rope_heads}")
        if rope_heads == 0:
            return query, key
        query = torch.cat([apply_rotary_pos_emb(query[:, :rope_heads], freqs, q_scale), query[:, rope_heads:]], dim=1)
        key = torch.cat([apply_rotary_pos_emb(key[:, :rope_heads], freqs, k_scale), key[:, rope_heads:]], dim=1)
        return query, key

    def build_joint_key_mask(
        self,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([audio_mask, video_mask, text_mask], dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        text: torch.Tensor,
        time_embed: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        audio_rope=None,
        video_rope=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = audio.shape[0]
        if video.shape[0] != batch or text.shape[0] != batch:
            raise ValueError("audio, video, and text batch sizes must match")
        audio_mask = _require_mask(audio_mask, batch, audio.shape[1], audio.device, "audio_mask")
        video_mask = _require_mask(video_mask, batch, video.shape[1], video.device, "video_mask")
        text_mask = _require_mask(text_mask, batch, text.shape[1], text.device, "text_mask")

        audio_qkv, audio_mod = self.audio_block.pre_attention(audio, time_embed)
        video_qkv, video_mod = self.video_block.pre_attention(video, time_embed)
        text_qkv, text_mod = self.text_block.pre_attention(text, time_embed)

        qa, ka = self._apply_rope(audio_qkv[0], audio_qkv[1], audio_rope)
        qv, kv = self._apply_rope(video_qkv[0], video_qkv[1], video_rope)
        # Text positions are ordinal character indices, not 100 Hz physical
        # time.  Its character encoder already supplied position features.
        qt, kt = text_qkv[0], text_qkv[1]

        query = torch.cat([qa, qv, qt], dim=2)
        key = torch.cat([ka, kv, kt], dim=2)
        value = torch.cat([audio_qkv[2], video_qkv[2], text_qkv[2]], dim=2)

        # MMAudio explicitly makes these tensors contiguous before SDPA.  The
        # concatenation currently produces contiguous storage, but retaining
        # the explicit contract avoids backend/version-dependent failures.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        if self.attn_mask_enabled:
            key_mask = self.build_joint_key_mask(audio_mask, video_mask, text_mask)
            attn_mask = key_mask[:, None, None, :]
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch, -1, self.heads * self.dim_head)
        out = out.to(query.dtype)

        audio_len = audio.shape[1]
        video_len = video.shape[1]
        audio_out = out[:, :audio_len]
        video_out = out[:, audio_len : audio_len + video_len]
        text_out = out[:, audio_len + video_len :]

        audio = self.audio_block.post_attention(audio, audio_out, audio_mod, audio_mask)
        if not self.pre_only:
            video = self.video_block.post_attention(video, video_out, video_mod, video_mask)
            text = self.text_block.post_attention(text, text_out, text_mod, text_mask)
        return audio, video, text


class DiT_VT_MMAudio(DiT):
    """D1 staging with six AVT joint blocks and an audio-only refinement tail."""

    def __init__(
        self,
        dim: int,
        depth: int = 18,
        heads: int = 12,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: float = 2,
        mel_dim: int = 80,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        text_embedding_average_upsampling: bool = False,
        qk_norm: str | None = "rms_norm",
        conv_layers: int = 4,
        pe_attn_head: int | None = 1,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = True,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        use_conformer: bool = True,
        layer_indices_ctc=(5, 11),
        projector_dim: int | None = None,
        n_mm_layers: int = 6,
        audio_video_ratio: int = 4,
        video_dim: int = 1024,
        video_rope_scaled: bool = True,
        av_ff_kernel_size: int = 3,
        text_ff_kernel_size: int = 1,
        text_input_ff_mult: float = 4,
        last_joint_pre_only: bool = True,
    ):
        if text_embedding_average_upsampling:
            raise ValueError(
                "DiT_VT_MMAudio keeps ordered text at its native token rate; "
                "text_embedding_average_upsampling must be False"
            )
        if text_dim is None:
            text_dim = mel_dim
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            text_mask_padding=text_mask_padding,
            text_embedding_average_upsampling=text_embedding_average_upsampling,
            qk_norm=qk_norm,
            conv_layers=conv_layers,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
            long_skip_connection=long_skip_connection,
            checkpoint_activations=checkpoint_activations,
        )
        if not 1 <= n_mm_layers <= depth:
            raise ValueError(f"n_mm_layers must be in [1, {depth}], got {n_mm_layers}")
        self.audio_video_ratio = int(audio_video_ratio)
        if self.audio_video_ratio < 1:
            raise ValueError(f"audio_video_ratio must be positive, got {audio_video_ratio}")
        self.video_rope_scaled = bool(video_rope_scaled)
        self.n_mm_layers = int(n_mm_layers)
        self.last_joint_pre_only = bool(last_joint_pre_only)

        try:
            ctc_layer_indices = tuple(layer_indices_ctc)
        except TypeError as error:
            raise TypeError("layer_indices_ctc must be an iterable of zero-based integers") from error
        if any(type(index) is not int for index in ctc_layer_indices):
            raise TypeError(f"layer_indices_ctc must contain integers, got {ctc_layer_indices}")
        if any(a >= b for a, b in pairwise(ctc_layer_indices)):
            raise ValueError(f"layer_indices_ctc must be unique and increasing, got {ctc_layer_indices}")
        if any(not 0 <= index < depth for index in ctc_layer_indices):
            raise ValueError(f"layer_indices_ctc must lie in [0, {depth}), got {ctc_layer_indices}")
        self.layer_indices_ctc = ctc_layer_indices

        self.input_embed = AudioInputEmbeddingMM(mel_dim, dim)
        self.video_embed = VideoInputEmbeddingMM(video_dim, dim, use_conformer=use_conformer)
        self.text_embed = OrderedTextEmbeddingMM(text_num_embeds, text_dim, conv_layers)
        self.text_joint_embed = TextInputEmbeddingMM(text_dim, dim, ff_mult=text_input_ff_mult, dropout=dropout)

        audio_block_kwargs = {
            "dim": dim,
            "heads": heads,
            "dim_head": dim_head,
            "ff_mult": ff_mult,
            "dropout": dropout,
            "qk_norm": qk_norm,
            "pe_attn_head": pe_attn_head,
            "attn_backend": attn_backend,
            "attn_mask_enabled": attn_mask_enabled,
        }
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlockAVT(
                    **audio_block_kwargs,
                    av_ff_kernel_size=av_ff_kernel_size,
                    text_ff_kernel_size=text_ff_kernel_size,
                    pre_only=last_joint_pre_only and layer_index == n_mm_layers - 1,
                )
                if layer_index < n_mm_layers
                else DiTBlock(**audio_block_kwargs)
                for layer_index in range(depth)
            ]
        )

        projector_dim = dim if projector_dim is None else projector_dim
        ctc_vocab_size = self.text_embed.text_embed.num_embeddings + 1
        self.layer_map_ctc = {
            layer_index: projector_index for projector_index, layer_index in enumerate(ctc_layer_indices)
        }
        self.projectors_ctc = nn.ModuleList(
            [DownsampleLayer([2, 1], dim, projector_dim, ctc_vocab_size) for _ in ctc_layer_indices]
        )
        self._text_cache_signatures = {"text_cond": None, "text_uncond": None}
        self._initialize_mmaudio_weights()

    def _initialize_mmaudio_weights(self) -> None:
        # Match MMAudio's scratch initialization: Xavier for all Linear
        # projections, a small-normal timestep MLP, then AdaLN-Zero and a zero
        # final projection. Conv1d layers retain their native initialization,
        # as in MMAudio's ConvMLP implementation.
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.time_mlp[2].weight, std=0.02)

        for block in self.transformer_blocks:
            if isinstance(block, MMDiTBlockAVT):
                for stream in (block.audio_block, block.video_block, block.text_block):
                    nn.init.constant_(stream.attn_norm.linear.weight, 0)
                    nn.init.constant_(stream.attn_norm.linear.bias, 0)
            else:
                nn.init.constant_(block.attn_norm.linear.weight, 0)
                nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def get_input_embed(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        video: torch.Tensor,
        *,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_video: bool = False,
        cache: bool = True,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        complementary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_mask is None:
            text_mask = text != -1
        else:
            text_mask = _require_mask(text_mask, text.shape[0], text.shape[1], text.device, "text_mask")
        source_text = text
        source_text_mask = text_mask
        source_signature = (
            source_text,
            _tensor_version(source_text),
            source_text_mask,
            _tensor_version(source_text_mask),
        )
        if text_mask is not None:
            # TextEmbedding derives its ConvNeXt padding mask from token value
            # ``-1``. Canonicalize explicit padding before that encoder so a
            # caller-provided mask cannot disagree with the token contents.
            text = text.masked_fill(~text_mask, -1)
        max_text_len = int(text_mask.sum(dim=1).max().item())
        if max_text_len < 1:
            raise ValueError("every batch must contain at least one valid text token")
        text_mask = text_mask[:, :max_text_len]

        if cache:
            cache_name = "text_uncond" if drop_text else "text_cond"
            text_features = getattr(self, cache_name)
            cached_signature = self._text_cache_signatures[cache_name]
            signature_matches = (
                cached_signature is not None
                and cached_signature[0] is source_signature[0]
                and cached_signature[1] == source_signature[1]
                and cached_signature[2] is source_signature[2]
                and cached_signature[3] == source_signature[3]
                and cached_signature[4] == max_text_len
            )
            if text_features is None or not signature_matches:
                text_features = self.text_embed(text, max_text_len, drop_text=drop_text, audio_mask=audio_mask)
                text_features = self.text_joint_embed(text_features, text_mask)
                setattr(self, cache_name, text_features)
                self._text_cache_signatures[cache_name] = (*source_signature, max_text_len)
        else:
            text_features = self.text_embed(text, max_text_len, drop_text=drop_text, audio_mask=audio_mask)
            text_features = self.text_joint_embed(text_features, text_mask)

        audio_features = self.input_embed(x, cond, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
        video_features = self.video_embed(
            video,
            drop_video=drop_video,
            video_mask=video_mask,
            complementary_mask=complementary_mask,
        )
        return audio_features, text_features, video_features

    def clear_cache(self) -> None:
        super().clear_cache()
        self._text_cache_signatures = {"text_cond": None, "text_uncond": None}

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        video: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        complementary_mask: torch.Tensor | None = None,
        generation_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_video: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        batch, audio_len = x.shape[:2]
        video_len = video.shape[1]
        if generation_mask is None:
            raise ValueError("generation_mask is required to identify synthesized audio frames")
        _require_mask(generation_mask, batch, audio_len, x.device, "generation_mask")
        if text_mask is None:
            text_mask = text != -1
        if time.ndim == 0 or (time.ndim == 1 and time.numel() == 1):
            time = time.expand(batch)
        elif time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must be scalar, [1], or [{batch}], got {tuple(time.shape)}")
        time_embed = self.time_embed(time)

        embed_kwargs = {
            "x": x,
            "cond": cond,
            "text": text,
            "video": video,
            "cache": cache,
            "audio_mask": mask,
            "text_mask": text_mask,
            "video_mask": video_mask,
            "complementary_mask": complementary_mask,
        }
        if cfg_infer:
            branches = []
            if not (drop_text or drop_video):
                branches.append(self.get_input_embed(**embed_kwargs))
                branches.append(self.get_input_embed(**embed_kwargs, drop_video=True))
            else:
                branches.append(self.get_input_embed(**embed_kwargs, drop_text=drop_text, drop_video=drop_video))
            branches.append(
                self.get_input_embed(
                    **embed_kwargs,
                    drop_audio_cond=True,
                    drop_text=True,
                    drop_video=True,
                )
            )
            repeat_count = len(branches)
            audio_features = torch.cat([branch[0] for branch in branches], dim=0)
            text_features = torch.cat([branch[1] for branch in branches], dim=0)
            video_features = torch.cat([branch[2] for branch in branches], dim=0)
            time_embed = time_embed.repeat(repeat_count, 1)
            mask = mask.repeat(repeat_count, 1) if mask is not None else None
            text_mask = text_mask.repeat(repeat_count, 1)
            video_mask = video_mask.repeat(repeat_count, 1) if video_mask is not None else None
            generation_mask = generation_mask.repeat(repeat_count, 1)
        else:
            audio_features, text_features, video_features = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_video=drop_video,
            )

        text_mask = text_mask[:, : text_features.shape[1]]
        audio_joint_mask = _require_mask(mask, audio_features.shape[0], audio_len, audio_features.device, "audio_mask")
        video_joint_mask = _require_mask(
            video_mask, video_features.shape[0], video_len, video_features.device, "video_mask"
        )
        text_joint_mask = _require_mask(
            text_mask,
            text_features.shape[0],
            text_features.shape[1],
            text_features.device,
            "text_mask",
        )

        audio_lens = (
            mask.sum(dim=1)
            if mask is not None
            else torch.full((audio_features.size(0),), audio_len, device=audio_features.device, dtype=torch.long)
        )
        audio_rope = self.rotary_embed.forward_from_seq_len(audio_len)
        video_positions = torch.arange(video_len, device=audio_features.device)
        if self.video_rope_scaled:
            video_positions = video_positions * self.audio_video_ratio
        video_rope = self.rotary_embed.forward(video_positions)

        if self.long_skip_connection is not None:
            residual = audio_features

        intermediates_ctc = {}
        for layer_index, block in enumerate(self.transformer_blocks):
            if isinstance(block, MMDiTBlockAVT):
                block_args = (
                    audio_features,
                    video_features,
                    text_features,
                    time_embed,
                    audio_joint_mask,
                    video_joint_mask,
                    text_joint_mask,
                    audio_rope,
                    video_rope,
                )
                if self.checkpoint_activations:
                    audio_features, video_features, text_features = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block), *block_args, use_reentrant=False
                    )
                else:
                    audio_features, video_features, text_features = block(*block_args)
            else:
                block_args = (audio_features, time_embed, audio_joint_mask, audio_rope)
                if self.checkpoint_activations:
                    audio_features = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block), *block_args, use_reentrant=False
                    )
                else:
                    audio_features = block(*block_args)

                # The stock audio-only DiT block masks attention keys/outputs,
                # but its point-wise FFN may recreate non-zero padded states.
                # Clear them before the next block and before a CTC projector.
                audio_features = audio_features.masked_fill(~audio_joint_mask.unsqueeze(-1), 0.0)

            if not cache and layer_index in self.layer_map_ctc:
                projector = self.projectors_ctc[self.layer_map_ctc[layer_index]]
                logits, logit_lens = projector(audio_features, audio_lens)
                intermediates_ctc[layer_index] = {"z_tilde": logits, "z_lens": logit_lens}

        if self.long_skip_connection is not None:
            audio_features = self.long_skip_connection(torch.cat((audio_features, residual), dim=-1))
        audio_features = self.norm_out(audio_features, time_embed)
        output = self.proj_out(audio_features)
        output = output.masked_fill(~audio_joint_mask.unsqueeze(-1), 0.0)
        return output, intermediates_ctc
