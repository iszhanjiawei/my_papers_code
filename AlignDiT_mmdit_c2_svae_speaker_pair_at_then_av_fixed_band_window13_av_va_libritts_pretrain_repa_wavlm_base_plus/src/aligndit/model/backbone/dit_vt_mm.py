"""
ein notation:
b - batch
n - sequence (audio, 100 Hz)
nt - text sequence
nv - video sequence (native 25 Hz, NOT upsampled)
nw - raw wave length
d - dimension

MM-DiT backbone for multimodal dubbing:
- dual-stream (audio latent stream + video stream) joint attention in the first
  `n_mm_layers` blocks, with aligned RoPE across the two frame rates
- text is injected via cross-attention in the first `n_text_layers` blocks;
  `prompt_isolated_ca` controls whether only synthesized frames receive it
- blocks after `n_mm_layers` stop video interaction; blocks after
  `n_text_layers` are text-free audio-only DiT blocks
- audio-stream parameter names are kept identical to DiTBlock so
  that the audio-only pretrained checkpoint loads directly by key matching
- `text_joint_attention=True` replaces the MM-stage text cross-attention with
  three evolving audio/video/text streams sharing the same attention softmax;
  the embeddings, audio/video FFNs, CTC taps and audio-only tail are unchanged
- paired_joint_order divides the audio heads equally between bidirectional AV
  and AT attention, with the same order/fusion rule in every multimodal block
- the optional fixed time prior biases only the AV and VA logits of the
  second stage of each AT-then-AV block
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb

from aligndit.model.fixed_temporal_band import FixedTemporalBand
from aligndit.model.modules import DiTCrossBlock, DownsampleLayer
from cosyvoice.transformer.encoder import ConformerEncoder
from f5_tts.model.backbones.dit import ConvPositionEmbedding, DiT
from f5_tts.model.modules import AdaLayerNorm, AttnProcessor, Attention, DiTBlock, FeedForward


def _validate_generation_mask(x, generation_mask):
    if generation_mask is None:
        raise ValueError("generation_mask is required for prompt-isolated text cross-attention")
    if generation_mask.dtype != torch.bool:
        raise TypeError(f"generation_mask must be bool, got {generation_mask.dtype}")
    if generation_mask.shape != x.shape[:2]:
        raise ValueError(
            f"generation_mask must have shape {tuple(x.shape[:2])}, got {tuple(generation_mask.shape)}"
        )


# 音频流输入层
class AudioInputEmbedding_MM(nn.Module):
    # same shapes/names as pretraining InputEmbedding_noText -> checkpoint compatible
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

# 视频流输入层
class VideoInputEmbedding_MM(nn.Module):
    # video kept at native 25 Hz (no transposed-conv upsampling)
    def __init__(self, video_dim, out_dim, use_conformer=False):
        super().__init__()
        self.register_buffer("vid_null_emb", nn.Parameter(torch.randn(1, video_dim) / video_dim**0.5))
        self.proj = nn.Linear(video_dim, out_dim)
        self.use_conformer = use_conformer
        if self.use_conformer:
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
        video: float["b nv d"],  # noqa: F722
        drop_video=False,
        video_mask: bool["b nv"] | None = None,  # noqa: F722
        complementary_mask: bool["b nv"] | None = None,  # noqa: F722
    ):
        if complementary_mask is not None and complementary_mask.shape[1] == video.shape[1]:
            video = torch.where(
                complementary_mask[..., None], self.vid_null_emb.expand(video.size(0), video.size(1), -1), video
            )

        if drop_video:
            video = self.vid_null_emb.expand(video.size(0), video.size(1), -1)

        if video_mask is not None and video_mask.shape[1] == video.shape[1]:
            video_lens = video_mask.sum(dim=1)
        else:
            video_lens = torch.full((video.size(0),), video.size(1), device=video.device, dtype=torch.long)

        v = self.proj(video)
        if self.use_conformer: # baseline的DiT是先采样到100Hz(长度*4)再在100Hz上跑Conformer 
            v, _ = self.vid_conformer(v, video_lens) # 这里不进行上采样，直接在原生25Hz上跑Conformer同样有上下文的建模能力,但序列短4倍
        v = self.conv_pos_embed(v) + v
        return v


class AudioTextDiTBlock(DiTCrossBlock):
    """Audio-only refinement block that optionally isolates text from prompt frames."""

    def __init__(self, *args, prompt_isolated_ca=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_isolated_ca = prompt_isolated_ca

    def forward(
        self,
        x,
        t,
        mask=None,
        rope=None,
        text=None,
        text_mask=None,
        generation_mask=None,
    ):
        if self.prompt_isolated_ca:
            _validate_generation_mask(x, generation_mask)

        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        ca_output, _ = self.cross_attn(
            x,
            text,
            text,
            key_padding_mask=~text_mask if text_mask is not None else None,
            need_weights=False,
        )
        if self.prompt_isolated_ca:
            ca_output = ca_output.masked_fill(~generation_mask.unsqueeze(-1), 0.0)
        x = x + ca_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        return x


class MMDiTBlock_VT(DiTCrossBlock):
    """Dual-stream MM-DiT block.

    audio stream: attn_norm / attn / cross_attn / ff_norm / ff (names kept for checkpoint compat)
    video stream: v_attn_norm / v_attn / v_ff_norm / v_ff (new parameters)
    the two streams communicate via joint attention (concat q/k/v along sequence).
    """

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
        text_dim=512,
        prompt_isolated_ca=True,
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
            text_dim=text_dim,
        )
        if attn_backend != "torch":
            raise NotImplementedError("only torch attn backend is supported for MMDiTBlock_VT")

        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        self.prompt_isolated_ca = prompt_isolated_ca

        # video stream 视频流独立参数
        self.v_attn_norm = AdaLayerNorm(dim)
        self.v_attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.v_ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.v_ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

        # text cross-attention AdaLN modulation + gate (audio stream only)
        # 文本 cross-attention 的 AdaLN 调制(shift/scale) + 门控(gate)，条件于时间步 t
        # 与 HunyuanVideo-Foley 的 cross-attn 一致：调制输入、门控输出，训练更稳定
        self.cross_attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn_ada = nn.Linear(dim, dim * 3)  # -> shift_ca, scale_ca, gate_ca

    def _qkv(self, attn, x):
        batch_size = x.shape[0]
        head_dim = attn.inner_dim // attn.heads
        query = attn.to_q(x).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = attn.to_k(x).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = attn.to_v(x).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        return query, key, value

    def _apply_rope(self, query, key, rope):
        if rope is None:
            return query, key
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        if self.pe_attn_head is not None:
            pn = self.pe_attn_head
            query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
            key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
        else:
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        return query, key

    def joint_attn(self, norm_x, norm_v, mask=None, v_mask=None, rope=None, v_rope=None):
        batch_size = norm_x.shape[0]
        n_a, n_v = norm_x.shape[1], norm_v.shape[1]
        heads = self.attn.heads
        head_dim = self.attn.inner_dim // heads

        q_a, k_a, v_a = self._qkv(self.attn, norm_x)
        q_v, k_v, v_v = self._qkv(self.v_attn, norm_v)

        q_a, k_a = self._apply_rope(q_a, k_a, rope)   # 将 Q 和 K 进行位置编码  
        q_v, k_v = self._apply_rope(q_v, k_v, v_rope) # 将 Q 和 K 进行位置编码

        query = torch.cat([q_a, q_v], dim=2)
        key = torch.cat([k_a, k_v], dim=2)
        value = torch.cat([v_a, v_v], dim=2)

        if self.attn_mask_enabled and mask is not None:
            if v_mask is None:
                v_mask = torch.ones((batch_size, n_v), dtype=torch.bool, device=mask.device)
            key_mask = torch.cat([mask, v_mask], dim=1)
            attn_mask = key_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, heads, n_a + n_v, n_a + n_v)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        out = out.to(query.dtype)

        out_a, out_v = out[:, :n_a], out[:, n_a:]

        out_a = self.attn.to_out[0](out_a)
        out_a = self.attn.to_out[1](out_a)
        out_v = self.v_attn.to_out[0](out_v)
        out_v = self.v_attn.to_out[1](out_v)

        if mask is not None:
            out_a = out_a.masked_fill(~mask.unsqueeze(-1), 0.0)
        if v_mask is not None:
            out_v = out_v.masked_fill(~v_mask.unsqueeze(-1), 0.0)

        return out_a, out_v

    def forward(
        self,
        x,
        v,
        t,
        mask=None,
        v_mask=None,
        rope=None,
        v_rope=None,
        text=None,
        text_mask=None,
        generation_mask=None,
    ):
        if self.prompt_isolated_ca:
            _validate_generation_mask(x, generation_mask)
        # pre-norm & modulation for attention input (per stream) 1、各流独立的adaLN调制(用时间步t生成scale/shift/gate)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm(x, emb=t)
        norm_v, v_gate_msa, v_shift_mlp, v_scale_mlp, v_gate_mlp = self.v_attn_norm(v, emb=t)

        # joint attention across audio and video streams 2、两流信息双向交换
        attn_x, attn_v = self.joint_attn(norm_x, norm_v, mask=mask, v_mask=v_mask, rope=rope, v_rope=v_rope)
        # x_gate_msa 门控，是控制注意力输出以多大比例加回残差流的。门控都是加载处理网络之后的，用来控制网络输出的多少，而不是加在输入，是作用于输出，而不是输入
        x = x + x_gate_msa.unsqueeze(1) * attn_x # gate 控制音频流接受的信息量
        v = v + v_gate_msa.unsqueeze(1) * attn_v # gate 控制视频流接受的信息量

        # text cross attention (audio stream as query) 3、文本 cross-attention 只有音频流作为query，文本不进入视频流
        # 3.1 用时间步 t 生成 cross-attn 专属的 shift/scale/gate（AdaLN-zero，零初始化）
        ca_shift, ca_scale, ca_gate = self.cross_attn_ada(F.silu(t)).chunk(3, dim=-1)
        # 3.2 调制 cross-attn 的输入
        norm_ca = self.cross_attn_norm(x) * (1 + ca_scale[:, None]) + ca_shift[:, None]
        ca_output, _ = self.cross_attn(
            norm_ca, text, text, key_padding_mask=~text_mask if text_mask is not None else None, need_weights=False
        )
        # 3.3 In the isolated variants, only synthesized audio queries receive
        # the text residual; C2 deliberately leaves the residual global.
        if self.prompt_isolated_ca:
            ca_output = ca_output.masked_fill(~generation_mask.unsqueeze(-1), 0.0)
        x = x + ca_gate.unsqueeze(1) * ca_output

        # ff_norm是归一化的作用  x_scale_mlp / x_shift_mlp：在 FFN 之前，对归一化后的特征做仿射变换
        norm = self.ff_norm(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp.unsqueeze(1) * self.ff(norm) # 4、各流独立 FFN

        norm = self.v_ff_norm(v) * (1 + v_scale_mlp[:, None]) + v_shift_mlp[:, None]
        v = v + v_gate_mlp.unsqueeze(1) * self.v_ff(norm) # 5、各流独立 FFN

        return x, v


class MMDiTBlock_AVT(DiTBlock):
    """Three-stream joint attention with the original audio/video sublayers.

    Q/K/V are concatenated in audio, video, text order. Text retains the
    character encoder's ordinal positions and receives no physical-time RoPE.
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None,
        pe_attn_head=None, attn_backend="torch", attn_mask_enabled=True, condition_heads=None,
    ):
        super().__init__(
            dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
            dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head,
            attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled,
        )
        if attn_backend != "torch":
            raise NotImplementedError("only torch attn backend is supported for MMDiTBlock_AVT")
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        for prefix in ("v", "t"):
            setattr(self, f"{prefix}_attn_norm", AdaLayerNorm(dim))
            setattr(self, f"{prefix}_attn", Attention(
                processor=AttnProcessor(
                    pe_attn_head=pe_attn_head, attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                ),
                dim=dim, heads=heads if condition_heads is None else condition_heads, dim_head=dim_head,
                dropout=dropout, qk_norm=qk_norm,
            ))
            setattr(self, f"{prefix}_ff_norm", nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{prefix}_ff", FeedForward(
                dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh",
            ))

    _qkv = MMDiTBlock_VT._qkv
    _apply_rope = MMDiTBlock_VT._apply_rope

    def joint_attn(
        self, norm_x, norm_v, norm_text, mask=None, v_mask=None,
        rope=None, v_rope=None, text_mask=None,
    ):
        batch = norm_x.shape[0]
        n_a, n_v, n_t = norm_x.shape[1], norm_v.shape[1], norm_text.shape[1]
        q_a, k_a, v_a = self._qkv(self.attn, norm_x)
        q_v, k_v, v_v = self._qkv(self.v_attn, norm_v)
        q_t, k_t, v_t = self._qkv(self.t_attn, norm_text)
        q_a, k_a = self._apply_rope(q_a, k_a, rope)
        q_v, k_v = self._apply_rope(q_v, k_v, v_rope)
        query = torch.cat((q_a, q_v, q_t), dim=2)
        key = torch.cat((k_a, k_v, k_t), dim=2)
        value = torch.cat((v_a, v_v, v_t), dim=2)

        # Preserve the source's A/V padding policy, including its training
        # mask suppression. Text padding was always masked in cross-attention
        # and must remain masked now that text keys share the joint softmax.
        key_mask = None
        if text_mask is not None or (self.attn_mask_enabled and (mask is not None or v_mask is not None)):
            valid_a = mask if self.attn_mask_enabled and mask is not None else torch.ones(
                (batch, n_a), dtype=torch.bool, device=norm_x.device,
            )
            valid_v = v_mask if self.attn_mask_enabled and v_mask is not None else torch.ones(
                (batch, n_v), dtype=torch.bool, device=norm_v.device,
            )
            valid_t = text_mask if text_mask is not None else torch.ones(
                (batch, n_t), dtype=torch.bool, device=norm_text.device,
            )
            key_mask = torch.cat((valid_a, valid_v, valid_t), dim=1)[:, None, None, :]
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=key_mask, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch, n_a + n_v + n_t, self.attn.inner_dim)
        outputs = []
        for stream_out, attention, valid in zip(
            out.split((n_a, n_v, n_t), dim=1),
            (self.attn, self.v_attn, self.t_attn), (mask, v_mask, text_mask),
        ):
            stream_out = attention.to_out[1](attention.to_out[0](stream_out))
            if valid is not None:
                stream_out = stream_out.masked_fill(~valid.unsqueeze(-1), 0.0)
            outputs.append(stream_out)
        return tuple(outputs)

    def forward(
        self, x, v, t, mask=None, v_mask=None, rope=None, v_rope=None,
        text=None, text_mask=None, generation_mask=None,
    ):
        if text is None:
            raise ValueError("AVT joint attention requires the text stream")
        norm_x, x_gate, x_shift, x_scale, x_ff_gate = self.attn_norm(x, emb=t)
        norm_v, v_gate, v_shift, v_scale, v_ff_gate = self.v_attn_norm(v, emb=t)
        norm_t, t_gate, t_shift, t_scale, t_ff_gate = self.t_attn_norm(text, emb=t)
        attn_x, attn_v, attn_t = self.joint_attn(
            norm_x, norm_v, norm_t, mask=mask, v_mask=v_mask,
            rope=rope, v_rope=v_rope, text_mask=text_mask,
        )
        x = x + x_gate[:, None] * attn_x
        v = v + v_gate[:, None] * attn_v
        text = text + t_gate[:, None] * attn_t
        x = x + x_ff_gate[:, None] * self.ff(self.ff_norm(x) * (1 + x_scale[:, None]) + x_shift[:, None])
        v = v + v_ff_gate[:, None] * self.v_ff(self.v_ff_norm(v) * (1 + v_scale[:, None]) + v_shift[:, None])
        text = text + t_ff_gate[:, None] * self.t_ff(
            self.t_ff_norm(text) * (1 + t_scale[:, None]) + t_shift[:, None],
        )
        if text_mask is not None:
            text = text.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        return x, v, text


class MMDiTBlock_PairedJoint(MMDiTBlock_AVT):
    """Two bidirectional pair attentions with a fixed, disjoint audio head bank for each."""

    ORDERS = ("av_then_at", "at_then_av", "av_at_parallel")

    def __init__(self, *, heads, paired_joint_order, **kwargs):
        if paired_joint_order not in self.ORDERS:
            raise ValueError(f"paired_joint_order must be one of {self.ORDERS}, got {paired_joint_order!r}")
        if heads < 2 or heads % 2:
            raise ValueError(f"paired joint attention requires an even head count >= 2, got {heads}")
        super().__init__(heads=heads, condition_heads=heads // 2, **kwargs)
        self.paired_joint_order = paired_joint_order
        self.pair_heads = heads // 2
        self.head_dim = self.attn.inner_dim // heads

    def _serial_audio_qkv(self, norm_x, start):
        # The second pair sees an updated audio state. Project only the rows used
        # by this pair, keeping the parent's full-width Q/K/V parameters intact.
        rows = slice(start * self.head_dim, (start + self.pair_heads) * self.head_dim)
        q, k, v = (
            F.linear(norm_x, proj.weight[rows], proj.bias[rows])
            .unflatten(-1, (self.pair_heads, self.head_dim)).transpose(1, 2)
            for proj in (self.attn.to_q, self.attn.to_k, self.attn.to_v)
        )
        if self.attn.q_norm is not None:
            q = self.attn.q_norm(q)
            k = self.attn.k_norm(k)
        return q, k, v

    def _group_rope(self, q, k, rope, start):
        # pe_attn_head counts global audio heads. With pe_attn_head=1 only AV
        # head 0 receives physical-time RoPE. Text gets its encoder positions.
        count = self.pair_heads if self.pe_attn_head is None else max(0, min(self.pair_heads, self.pe_attn_head - start))
        if rope is None or count == 0:
            return q, k
        freqs, scale = rope
        q_rot = apply_rotary_pos_emb(q[:, :count], freqs, scale if scale is not None else 1.0)
        k_rot = apply_rotary_pos_emb(k[:, :count], freqs, scale**-1.0 if scale is not None else 1.0)
        if count == self.pair_heads:
            return q_rot, k_rot
        return torch.cat((q_rot, q[:, count:]), dim=1), torch.cat((k_rot, k[:, count:]), dim=1)

    @staticmethod
    def _mask_output(x, valid):
        return x if valid is None else x.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _pair_attention(
        self, audio_qkv, condition, is_video, audio_mask, condition_mask, rope, v_rope, temporal_band_bias=None,
    ):
        start = 0 if is_video else self.pair_heads
        attn = self.v_attn if is_video else self.t_attn
        q_a, k_a, v_a = audio_qkv
        q_a, k_a = self._group_rope(q_a, k_a, rope, start)
        q_c, k_c, v_c = self._qkv(attn, condition)
        if is_video:
            q_c, k_c = self._group_rope(q_c, k_c, v_rope, start)

        key_mask = None
        valid_a = audio_mask if self.attn_mask_enabled else None
        valid_c = condition_mask if self.attn_mask_enabled or not is_video else None
        batch, _, n_audio, _ = q_a.shape
        n_condition = condition.size(1)
        if valid_a is not None or valid_c is not None:
            if valid_a is None:
                valid_a = torch.ones((batch, n_audio), dtype=torch.bool, device=q_a.device)
            if valid_c is None:
                valid_c = torch.ones((batch, n_condition), dtype=torch.bool, device=q_a.device)
            key_mask = torch.cat((valid_a, valid_c), dim=1)[:, None, None, :]

        key = torch.cat((k_a, k_c), dim=2)
        value = torch.cat((v_a, v_c), dim=2)
        if temporal_band_bias is None:
            out = F.scaled_dot_product_attention(
                torch.cat((q_a, q_c), dim=2), key, value, attn_mask=key_mask, dropout_p=0.0,
            )
        else:
            if not is_video or temporal_band_bias.shape != (batch, n_audio, n_condition):
                raise ValueError("temporal_band_bias requires the AV pair and shape [batch, audio, video]")
            # Both query groups normalize over the complete [A,V] sequence.
            # The zero blocks leave AA and VV logits unchanged.
            av_bias = temporal_band_bias.to(q_a.dtype)
            audio_bias = F.pad(av_bias, (n_audio, 0))[:, None]
            video_bias = F.pad(av_bias.transpose(-1, -2), (0, n_condition))[:, None]
            if key_mask is not None:
                audio_bias = audio_bias.masked_fill(~key_mask, float("-inf"))
                video_bias = video_bias.masked_fill(~key_mask, float("-inf"))
            out_a = F.scaled_dot_product_attention(q_a, key, value, attn_mask=audio_bias, dropout_p=0.0)
            out_v = F.scaled_dot_product_attention(q_c, key, value, attn_mask=video_bias, dropout_p=0.0)
            out = torch.cat((out_a, out_v), dim=2)
        out_a, out_c = out.transpose(1, 2).flatten(2).split((n_audio, n_condition), dim=1)
        out_c = attn.to_out[1](attn.to_out[0](out_c))
        return out_a, self._mask_output(out_c, condition_mask)

    def forward_text_audio(self, x, text, t, mask, text_mask, rope):
        """Run the first stage of AT→AV and the audio/text FFNs on LibriTTS."""
        norm_x, gate, shift, scale, ff_gate = self.attn_norm(x, emb=t)
        norm_t, t_gate, t_shift, t_scale, t_ff_gate = self.t_attn_norm(text, emb=t)
        qkv = self._serial_audio_qkv(norm_x, self.pair_heads)
        heads, text_delta = self._pair_attention(qkv, norm_t, False, mask, text_mask, rope, None)
        proj, dropout = self.attn.to_out
        # AT contributes the upper half of W_out. The shared bias belongs to
        # the downstream AV stage and stays at its S2c value during pretraining.
        delta = F.linear(heads, proj.weight[:, self.pair_heads * self.head_dim:])
        x = x + gate[:, None] * dropout(delta)
        text = text + t_gate[:, None] * text_delta
        x = x + ff_gate[:, None] * self.ff(self.ff_norm(x) * (1 + scale[:, None]) + shift[:, None])
        text = text + t_ff_gate[:, None] * self.t_ff(
            self.t_ff_norm(text) * (1 + t_scale[:, None]) + t_shift[:, None],
        )
        return self._mask_output(x, mask), self._mask_output(text, text_mask)

    def forward(
        self, x, v, t, mask=None, v_mask=None, rope=None, v_rope=None,
        text=None, text_mask=None, generation_mask=None, temporal_band_bias=None,
    ):
        if text is None:
            raise ValueError("paired joint attention requires the text stream")
        if temporal_band_bias is not None and self.paired_joint_order != "at_then_av":
            raise ValueError("temporal_band_bias requires paired_joint_order='at_then_av'")
        # Compute the audio AdaLN modulation once. Both serial stages normalize
        # their own current state with the same timestep-conditioned scale/shift.
        shift, scale, gate, ff_shift, ff_scale, ff_gate = self.attn_norm.linear(
            self.attn_norm.silu(t)
        ).chunk(6, dim=-1)
        norm_x = self.attn_norm.norm(x) * (1 + scale[:, None]) + shift[:, None]
        norm_v, v_gate, v_shift, v_scale, v_ff_gate = self.v_attn_norm(v, emb=t)
        norm_t, t_gate, t_shift, t_scale, t_ff_gate = self.t_attn_norm(text, emb=t)
        proj, dropout = self.attn.to_out

        if self.paired_joint_order == "av_at_parallel":
            qkv = self._qkv(self.attn, norm_x)
            av_qkv = tuple(tensor[:, :self.pair_heads] for tensor in qkv)
            at_qkv = tuple(tensor[:, self.pair_heads:] for tensor in qkv)
            av_audio, attn_v = self._pair_attention(av_qkv, norm_v, True, mask, v_mask, rope, v_rope)
            at_audio, attn_t = self._pair_attention(at_qkv, norm_t, False, mask, text_mask, rope, None)
            update = dropout(proj(torch.cat((av_audio, at_audio), dim=-1)))
            x = x + gate[:, None] * self._mask_output(update, mask)
        else:
            # Use the matching columns of the same W_out. Add its single bias
            # only on the final projection; the intermediate carries no bias.
            first_is_video = self.paired_joint_order == "av_then_at"
            conditions = ((norm_v, v_mask), (norm_t, text_mask))
            order = (0, 1) if first_is_video else (1, 0)
            condition_updates = [None, None]
            # Reuse one output-dropout mask for the two column contributions,
            # matching dropout on the complete audio output projection.
            dropout_mask = None
            for stage, index in enumerate(order):
                condition, condition_mask = conditions[index]
                start = index * self.pair_heads
                audio_qkv = self._serial_audio_qkv(norm_x, start)
                audio_heads, condition_updates[index] = self._pair_attention(
                    audio_qkv, condition, index == 0, mask, condition_mask, rope, v_rope,
                    temporal_band_bias=temporal_band_bias if index == 0 else None,
                )
                width = self.pair_heads * self.head_dim
                columns = slice(index * width, (index + 1) * width)
                delta = F.linear(audio_heads, proj.weight[:, columns], proj.bias if stage == 1 else None)
                if stage == 0 and dropout.training and dropout.p:
                    dropout_mask = dropout(torch.ones_like(delta))
                if dropout_mask is not None:
                    delta = delta * dropout_mask
                x = x + gate[:, None] * self._mask_output(delta, mask)
                if stage == 0:
                    norm_x = self.attn_norm.norm(x) * (1 + scale[:, None]) + shift[:, None]
            attn_v, attn_t = condition_updates

        v = v + v_gate[:, None] * attn_v
        text = text + t_gate[:, None] * attn_t
        x = x + ff_gate[:, None] * self.ff(self.ff_norm(x) * (1 + ff_scale[:, None]) + ff_shift[:, None])
        v = v + v_ff_gate[:, None] * self.v_ff(self.v_ff_norm(v) * (1 + v_scale[:, None]) + v_shift[:, None])
        text = text + t_ff_gate[:, None] * self.t_ff(
            self.t_ff_norm(text) * (1 + t_scale[:, None]) + t_shift[:, None]
        )
        return x, v, self._mask_output(text, text_mask)


class DiT_VT_MMDiT(DiT):
    def __init__(
        self,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_conformer=True,
        layer_indices_ctc=[6, 12],
        ctc_sampling_ratios=(2, 1),
        projector_dim=None,
        n_mm_layers=12,
        n_text_layers=None,
        prompt_isolated_ca=True,
        audio_video_ratio=4,
        video_dim=1024,
        video_rope_scaled=True,
        normalize_text_context=False,
        speaker_dim=None,
        speaker_condition_start_layer=None,
        text_joint_attention=False,
        paired_joint_order=None,
        temporal_band_enabled=False,
        temporal_band_mode="fixed",
        temporal_band_audio_fps=40.0,
        temporal_band_video_fps=40.0,
        temporal_band_fixed_offset_seconds=0.0,
        temporal_band_fixed_sigma_seconds=0.150,
        temporal_band_bidirectional=False,
        text_audio_only=False,
        repa_layer=None,
        repa_target_dim=None,
        repa_projector_dim=None,
    ):
        if type(text_audio_only) is not bool:
            raise TypeError("text_audio_only must be a boolean")
        if text_audio_only and (paired_joint_order != "at_then_av" or temporal_band_enabled):
            raise ValueError("text_audio_only requires AT-then-AV with the AV band disabled")
        if type(temporal_band_enabled) is not bool or type(temporal_band_bidirectional) is not bool:
            raise TypeError("temporal_band_enabled and temporal_band_bidirectional must be bools")
        if temporal_band_mode != "fixed":
            raise ValueError("Only temporal_band_mode='fixed' is supported")
        if temporal_band_enabled != temporal_band_bidirectional:
            raise ValueError("temporal_band_enabled and temporal_band_bidirectional must match")
        if temporal_band_enabled and paired_joint_order != "at_then_av":
            raise ValueError("Fixed AV/VA bias requires paired_joint_order='at_then_av'")
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
        self.audio_video_ratio = audio_video_ratio
        self.video_rope_scaled = video_rope_scaled
        self.prompt_isolated_ca = prompt_isolated_ca
        self.normalize_text_context = bool(normalize_text_context)
        # Runtime-only diagnostics.  These are deliberately not buffers: the
        # repair must remain state-dict compatible with the S2c parent and the
        # completed Direct-C2 control.
        self.last_text_context_raw_rms: torch.Tensor | None = None
        self.last_text_context_post_rms: torch.Tensor | None = None
        self.n_mm_layers = n_mm_layers
        self.n_text_layers = n_mm_layers if n_text_layers is None else n_text_layers
        self.text_audio_only = text_audio_only
        self.paired_joint_order = paired_joint_order
        if paired_joint_order is not None and not text_joint_attention:
            raise ValueError("paired_joint_order requires text_joint_attention=True")
        self.text_joint_attention = bool(text_joint_attention)
        if self.text_joint_attention and (
            self.n_mm_layers == 0 or self.n_text_layers != self.n_mm_layers or prompt_isolated_ca
        ):
            raise ValueError("AVT joint attention requires n_text_layers == n_mm_layers > 0 and prompt_isolated_ca=False")
        if not 0 <= self.n_mm_layers <= self.n_text_layers <= depth:
            raise ValueError(
                "expected 0 <= n_mm_layers <= n_text_layers <= depth, got "
                f"n_mm_layers={self.n_mm_layers}, n_text_layers={self.n_text_layers}, depth={depth}"
            )
        self.speaker_dim = speaker_dim
        if self.speaker_dim is None:
            if speaker_condition_start_layer is not None:
                raise ValueError("speaker_condition_start_layer requires speaker_dim")
            self.speaker_condition_start_layer = None
        else:
            if self.speaker_dim <= 0:
                raise ValueError(f"speaker_dim must be positive, got {self.speaker_dim}")
            self.speaker_condition_start_layer = (
                self.n_text_layers
                if speaker_condition_start_layer is None
                else speaker_condition_start_layer
            )
            if not self.n_text_layers <= self.speaker_condition_start_layer < depth:
                raise ValueError(
                    "speaker conditioning is restricted to the text-free audio tail; expected "
                    f"n_text_layers <= speaker_condition_start_layer < depth, got "
                    f"{self.n_text_layers} <= {self.speaker_condition_start_layer} < {depth}"
                )
        if text_audio_only and repa_layer is not None:
            raise ValueError("REPA supervision requires the multimodal dubbing backbone")
        self.repa_layer = repa_layer
        self.repa_target_dim = repa_target_dim
        if self.repa_layer is None:
            if self.repa_target_dim is not None or repa_projector_dim is not None:
                raise ValueError("repa_target_dim/repa_projector_dim require repa_layer")
        else:
            if type(self.repa_layer) is not int or not 0 <= self.repa_layer < self.n_mm_layers:
                raise ValueError(
                    "repa_layer must be a zero-based MM-DiT layer index in "
                    f"[0, {self.n_mm_layers}), got {self.repa_layer!r}"
                )
            if type(self.repa_target_dim) is not int or self.repa_target_dim <= 0:
                raise ValueError(f"repa_target_dim must be a positive integer, got {self.repa_target_dim!r}")
            if repa_projector_dim is None:
                repa_projector_dim = self.dim * 2
            if type(repa_projector_dim) is not int or repa_projector_dim <= 0:
                raise ValueError(f"repa_projector_dim must be a positive integer, got {repa_projector_dim!r}")
        try:
            ctc_layer_indices = tuple(layer_indices_ctc)
        except TypeError as error:
            raise TypeError("layer_indices_ctc must be an iterable of zero-based integer layer indices") from error
        if any(type(layer_i) is not int for layer_i in ctc_layer_indices):
            raise TypeError(f"layer_indices_ctc must contain only integers, got {ctc_layer_indices}")
        if any(current >= following for current, following in zip(ctc_layer_indices, ctc_layer_indices[1:])):
            raise ValueError(
                f"layer_indices_ctc must be unique and strictly increasing, got {ctc_layer_indices}"
            )
        if any(not 0 <= layer_i < depth for layer_i in ctc_layer_indices):
            raise ValueError(
                f"layer_indices_ctc must use zero-based indices in [0, {depth}), got {ctc_layer_indices}"
            )
        self.layer_indices_ctc = ctc_layer_indices
        try:
            ctc_sampling_ratios = tuple(ctc_sampling_ratios)
        except TypeError as error:
            raise TypeError("ctc_sampling_ratios must be an iterable of positive integers") from error
        if not ctc_sampling_ratios or any(type(ratio) is not int or ratio <= 0 for ratio in ctc_sampling_ratios):
            raise ValueError(f"ctc_sampling_ratios must contain positive integers, got {ctc_sampling_ratios}")
        self.ctc_sampling_ratios = ctc_sampling_ratios

        self.input_embed = AudioInputEmbedding_MM(mel_dim, dim)
        self.video_embed = VideoInputEmbedding_MM(video_dim, dim, use_conformer=use_conformer)

        audio_block_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
        )
        text_block_kwargs = {
            **audio_block_kwargs,
            "text_dim": text_dim,
            "prompt_isolated_ca": prompt_isolated_ca,
        }
        if self.paired_joint_order is not None:
            mm_block = MMDiTBlock_PairedJoint
            mm_kwargs = {**audio_block_kwargs, "paired_joint_order": self.paired_joint_order}
        elif self.text_joint_attention:
            mm_block, mm_kwargs = MMDiTBlock_AVT, audio_block_kwargs
        else:
            mm_block, mm_kwargs = MMDiTBlock_VT, text_block_kwargs

        self.transformer_blocks = nn.ModuleList(
            [
                mm_block(**mm_kwargs) if i < self.n_mm_layers
                else AudioTextDiTBlock(**text_block_kwargs)
                if i < self.n_text_layers
                else DiTBlock(**audio_block_kwargs)
                for i in range(depth)
            ]
        )

        projector_dim = self.dim if projector_dim is None else projector_dim
        z_dim = self.text_embed.text_embed.num_embeddings + 1
        self.layer_map_ctc = {v: i for i, v in enumerate(self.layer_indices_ctc)}
        self.projectors_ctc = nn.ModuleList(
            [
                DownsampleLayer(self.ctc_sampling_ratios, self.dim, projector_dim, z_dim)
                for _ in self.layer_map_ctc
            ]
        )
        self.speaker_proj = (
            nn.Linear(self.speaker_dim, self.dim, bias=False) if self.speaker_dim is not None else None
        )
        self.text_joint_proj = (
            nn.Linear(self.text_embed.text_embed.embedding_dim, dim) if self.text_joint_attention else None
        )

        # Initialize the re-created blocks without double-zeroing a gated branch.
        #
        # For an AdaLN-Zero residual y = x + gate * branch(x), only the gate is
        # initialized to zero. The branch output projection must stay normally
        # initialized so that the gate receives a non-zero gradient on the first
        # update; otherwise both factors remain zero forever.
        self.initialize_weights()
        for block in self.transformer_blocks[: self.n_mm_layers]:
            # Gated MM branches: zero only the modulation/gates. Keep
            # cross_attn.out_proj and v_attn.to_out normally initialized.
            if self.text_joint_attention:
                nn.init.constant_(block.t_attn_norm.linear.weight, 0)
                nn.init.constant_(block.t_attn_norm.linear.bias, 0)
            else:
                nn.init.constant_(block.cross_attn_ada.weight, 0)
                nn.init.constant_(block.cross_attn_ada.bias, 0)
            nn.init.constant_(block.v_attn_norm.linear.weight, 0)
            nn.init.constant_(block.v_attn_norm.linear.bias, 0)
        for block in self.transformer_blocks[self.n_mm_layers : self.n_text_layers]:
            # These tail text blocks have no extra output gate. Zeroing only
            # cross_attn.out_proj preserves the pretrained audio path while
            # still giving that projection a gradient on the first update.
            nn.init.constant_(block.cross_attn.out_proj.weight, 0)
            nn.init.constant_(block.cross_attn.out_proj.bias, 0)
        if self.speaker_proj is not None:
            # Preserve the parent Direct-C2 model at initialization. A second
            # zero gate here would block gradients into the projection.
            nn.init.constant_(self.speaker_proj.weight, 0)

        self.temporal_band = None
        if temporal_band_enabled:
            self.temporal_band = FixedTemporalBand(
                dim=dim, audio_fps=temporal_band_audio_fps, video_fps=temporal_band_video_fps,
                offset_seconds=temporal_band_fixed_offset_seconds, sigma_seconds=temporal_band_fixed_sigma_seconds,
            )
            if abs(self.temporal_band.audio_fps / self.temporal_band.video_fps - self.audio_video_ratio) > 1e-6:
                raise ValueError("temporal-band audio/video frame-rate ratio must match audio_video_ratio")

        if self.text_audio_only:
            self.text_embed.mask_padding = True
            self.video_embed.requires_grad_(False)
            for block in self.transformer_blocks:
                block.attn.processor.attn_mask_enabled = True
            for block in self.transformer_blocks[:self.n_mm_layers]:
                block.attn_mask_enabled = True
                block.attn.to_out[0].bias.requires_grad_(False)
                for name, parameter in block.named_parameters():
                    if name.startswith("v_"):
                        parameter.requires_grad_(False)

        # Construct the auxiliary head last to preserve the initialization of shared weights.
        self.repa_projector = (
            nn.Sequential(
                nn.Linear(self.dim, repa_projector_dim),
                nn.SiLU(),
                nn.Linear(repa_projector_dim, repa_projector_dim),
                nn.SiLU(),
                nn.Linear(repa_projector_dim, self.repa_target_dim),
            )
            if self.repa_layer is not None
            else None
        )

    def _forward_text_audio(
        self, x, cond, text, time, *, mask, text_mask, generation_mask,
        drop_audio_cond, drop_text, speaker_embedding, drop_speaker, cache, compute_ctc,
    ):
        _validate_generation_mask(x, generation_mask)
        if generation_mask.device != x.device:
            raise ValueError("generation_mask must be on the latent device")
        batch, length = x.shape[:2]
        if cond.shape != x.shape:
            raise ValueError("Text/audio conditioning must match the noisy latent shape")
        if mask is None:
            mask = torch.ones((batch, length), device=x.device, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != (batch, length) or mask.device != x.device:
            raise ValueError("Audio mask must be bool [batch, audio frames] on the latent device")
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must contain valid audio frames")
        if text_mask is None:
            text_mask = text != -1
        if text.ndim != 2 or text.shape[0] != batch or text_mask.shape != text.shape:
            raise ValueError("Text and text_mask must have matching [batch, tokens] shapes")
        if text_mask.dtype != torch.bool or text_mask.device != x.device or not text_mask.any(dim=1).all():
            raise ValueError("Every sample requires a nonempty boolean text mask on the latent device")
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        speaker_delta = self.get_speaker_delta(
            speaker_embedding, t,
            drop_speaker=drop_audio_cond if drop_speaker is None else drop_speaker,
        )
        text = text.masked_fill(~text_mask, -1)
        text_embed = self.text_embed(text, text.shape[1], drop_text=drop_text, audio_mask=mask)
        if self.normalize_text_context:
            text_embed, text_mask = self._stabilize_text_context(text_embed, text_mask)
        text_embed = self.text_joint_proj(text_embed).masked_fill(~text_mask[..., None], 0.0)

        # Mask before and between input convolutions: padded latent values must
        # never leak into a real frame through convolution or attention.
        x = x.masked_fill(~mask[..., None], 0.0)
        cond = torch.zeros_like(cond) if drop_audio_cond else cond.masked_fill(~mask[..., None], 0.0)
        x = self.input_embed.proj(torch.cat((x, cond), dim=-1)).masked_fill(~mask[..., None], 0.0)
        positional = x.transpose(1, 2)
        for layer in self.input_embed.conv_pos_embed.conv1d:
            positional = layer(positional).masked_fill(~mask[:, None], 0.0)
        x = (x + positional.transpose(1, 2)).masked_fill(~mask[..., None], 0.0)
        residual = x if self.long_skip_connection is not None else None
        rope = self.rotary_embed.forward_from_seq_len(length)
        lens = mask.sum(dim=1)
        intermediates_ctc = {}
        for index, block in enumerate(self.transformer_blocks):
            block_t = t + speaker_delta if (
                speaker_delta is not None and index >= self.speaker_condition_start_layer
            ) else t
            if index < self.n_mm_layers:
                args = (x, text_embed, block_t, mask, text_mask, rope)
                if self.checkpoint_activations and self.training:
                    x, text_embed = torch.utils.checkpoint.checkpoint(
                        block.forward_text_audio, *args, use_reentrant=False,
                    )
                else:
                    x, text_embed = block.forward_text_audio(*args)
            else:
                if self.checkpoint_activations and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, block_t, mask, rope, use_reentrant=False)
                else:
                    x = block(x, block_t, mask=mask, rope=rope)
                x = x.masked_fill(~mask[..., None], 0.0)
            if compute_ctc and not cache and index in self.layer_map_ctc:
                z_tilde, z_lens = self.projectors_ctc[self.layer_map_ctc[index]](x, lens)
                intermediates_ctc[index] = {"z_tilde": z_tilde, "z_lens": z_lens}
        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        output = self.proj_out(self.norm_out(x, t)).masked_fill(~mask[..., None], 0.0)
        return output, intermediates_ctc

    def get_speaker_delta(self, speaker_embedding, t, *, drop_speaker=False):
        if self.speaker_proj is None:
            if speaker_embedding is not None:
                raise ValueError("speaker_embedding was provided, but this backbone has no speaker_dim")
            return None
        if speaker_embedding is None:
            raise ValueError("speaker_embedding is required when speaker_dim is configured")
        if speaker_embedding.ndim != 2 or speaker_embedding.shape != (t.shape[0], self.speaker_dim):
            raise ValueError(
                f"speaker_embedding must have shape {(t.shape[0], self.speaker_dim)}, "
                f"got {tuple(speaker_embedding.shape)}"
            )
        speaker_embedding_fp32 = speaker_embedding.to(device=t.device, dtype=torch.float32)
        # Cached vectors are validated on CPU; avoid synchronizing CUDA in
        # every flow step. Normalize in float32 before the learned projection.
        speaker_embedding_normalized = F.normalize(speaker_embedding_fp32, dim=-1)
        if drop_speaker:
            speaker_embedding_normalized = torch.zeros_like(speaker_embedding_normalized)
        return self.speaker_proj(speaker_embedding_normalized.to(dtype=t.dtype))

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        video,  # b nv d
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_video: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,  # noqa: F722
        text_mask: bool["b nt"] | None = None,  # noqa: F722
        video_mask: bool["b nv"] | None = None,  # noqa: F722
        complementary_mask: bool["b nv"] | None = None,  # noqa: F722
    ):
        seq_len = text_mask.sum(dim=1).max().item()
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)

        x = self.input_embed(x, cond, drop_audio_cond=drop_audio_cond)
        v = self.video_embed(
            video,
            drop_video=drop_video,
            video_mask=video_mask,
            complementary_mask=complementary_mask,
        )

        return x, text_embed, v

    @staticmethod
    def _masked_text_context_rms(text_context: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        context_float = text_context.float()
        valid = text_mask.unsqueeze(-1)
        squared_sum = context_float.square().masked_fill(~valid, 0.0).sum()
        valid_elements = text_mask.sum(dtype=torch.float32) * text_context.shape[-1]
        return torch.sqrt(squared_sum / valid_elements.clamp_min(1.0))

    def _stabilize_text_context(
        self,
        text_context: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bound the shared text K/V scale before all twelve cross-attentions.

        The original text conditioner is a four-block residual ConvNeXt tower
        without an output normalization.  A single parameter-free per-token
        LayerNorm here fixes that unbounded interface without adding checkpoint
        keys or changing which blocks receive text.
        """

        if text_context.ndim != 3:
            raise ValueError(f"text context must have shape [batch, tokens, dim], got {tuple(text_context.shape)}")
        batch, context_len, context_dim = text_context.shape
        if text_mask is None:
            context_mask = torch.ones((batch, context_len), dtype=torch.bool, device=text_context.device)
        else:
            if text_mask.dtype != torch.bool:
                raise TypeError(f"text_mask must be bool, got {text_mask.dtype}")
            if text_mask.ndim != 2 or text_mask.shape[0] != batch or text_mask.shape[1] < context_len:
                raise ValueError(
                    "text_mask must have shape [batch, tokens] and cover the encoded context, "
                    f"got context={tuple(text_context.shape)}, mask={tuple(text_mask.shape)}"
                )
            if text_mask.device != text_context.device:
                raise ValueError(f"text_mask must be on {text_context.device}, got {text_mask.device}")
            if text_mask.shape[1] > context_len and text_mask[:, context_len:].any().item():
                raise ValueError("text_mask contains valid tokens beyond the encoded text context length")
            context_mask = text_mask[:, :context_len]

        raw_rms = self._masked_text_context_rms(text_context, context_mask)
        normalized = F.layer_norm(text_context, (context_dim,), weight=None, bias=None, eps=1e-6)
        normalized = normalized.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        post_rms = self._masked_text_context_rms(normalized, context_mask)
        self.last_text_context_raw_rms = raw_rms.detach()
        self.last_text_context_post_rms = post_rms.detach()
        return normalized, context_mask

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        video: float["b nv d"],  # video  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
        text_mask: bool["b nt"] | None = None,  # noqa: F722
        video_mask: bool["b nv"] | None = None,  # noqa: F722
        complementary_mask: bool["b nv"] | None = None,  # noqa: F722
        generation_mask: bool["b n"] | None = None,  # explicit synthesized audio region  # noqa: F722
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        drop_video: bool = False,  # cfg for video
        speaker_embedding: float["b ds"] | None = None,  # full, unmasked reference audio
        drop_speaker: bool | None = None,
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        compute_ctc: bool = True,
        return_repa: bool = False,
    ):
        if return_repa and self.repa_projector is None:
            raise RuntimeError("return_repa=True requires a configured REPA projector")
        if return_repa and (cfg_infer or cache):
            raise ValueError("REPA activations are training-only and cannot be requested with cfg_infer/cache")
        if self.text_audio_only:
            if video is not None:
                raise ValueError("text_audio_only expects video=None; visual features are not used")
            if cfg_infer:
                raise ValueError("Use CFMTextAudio.sample for CFG without AVT packed branches")
            return self._forward_text_audio(
                x, cond, text, time, mask=mask, text_mask=text_mask,
                generation_mask=generation_mask, drop_audio_cond=drop_audio_cond,
                drop_text=drop_text, speaker_embedding=speaker_embedding,
                drop_speaker=drop_speaker, cache=cache, compute_ctc=compute_ctc,
            )
        batch, seq_len = x.shape[0], x.shape[1]
        video_len = video.shape[1]
        if generation_mask is None:
            raise ValueError("generation_mask is required to separate prompt and synthesized audio regions")
        if generation_mask.dtype != torch.bool:
            raise TypeError(f"generation_mask must be bool, got {generation_mask.dtype}")
        if generation_mask.shape != (batch, seq_len):
            raise ValueError(
                f"generation_mask must have shape {(batch, seq_len)}, got {tuple(generation_mask.shape)}"
            )
        if generation_mask.device != x.device:
            raise ValueError(f"generation_mask must be on {x.device}, got {generation_mask.device}")
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, x: noised input audio (stream 1), v: video (stream 2)
        t = self.time_embed(time)
        if drop_speaker is None:
            # Keep identity conditioning coupled to prompt-audio dropout.
            drop_speaker = drop_audio_cond

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

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b/3b n d
            x_list, text_embed_list, v_list = [], [], []
            speaker_delta_list = []
            speaker_delta_cond = self.get_speaker_delta(
                speaker_embedding,
                t,
                drop_speaker=drop_speaker,
            )
            if not (drop_text or drop_video):
                x_cond, text_embed_cond, v_cond = self.get_input_embed(**embed_kwargs)
                x_list.append(x_cond)
                text_embed_list.append(text_embed_cond)
                v_list.append(v_cond)
                if speaker_delta_cond is not None:
                    speaker_delta_list.append(speaker_delta_cond)
                x_tts, text_embed_tts, v_tts = self.get_input_embed(
                    **embed_kwargs,
                    drop_video=True,
                )
                x_list.append(x_tts)
                text_embed_list.append(text_embed_tts)
                v_list.append(v_tts)
                if speaker_delta_cond is not None:
                    speaker_delta_list.append(speaker_delta_cond)
            else:
                x_cond, text_embed_cond, v_cond = self.get_input_embed(
                    **embed_kwargs,
                    drop_text=drop_text,
                    drop_video=drop_video,
                )
                x_list.append(x_cond)
                text_embed_list.append(text_embed_cond)
                v_list.append(v_cond)
                if speaker_delta_cond is not None:
                    speaker_delta_list.append(speaker_delta_cond)

            x_uncond, text_embed_uncond, v_uncond = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=True,
                drop_text=True,
                drop_video=True,
            )
            x_list.append(x_uncond)
            text_embed_list.append(text_embed_uncond)
            v_list.append(v_uncond)
            speaker_delta_null = self.get_speaker_delta(speaker_embedding, t, drop_speaker=True)
            if speaker_delta_null is not None:
                speaker_delta_list.append(speaker_delta_null)

            rep_n = len(x_list)
            x = torch.cat(x_list, dim=0)
            v = torch.cat(v_list, dim=0)
            # Match the branch-major ordering of torch.cat for batch > 1.
            t = t.repeat((rep_n, 1))
            text_embed = torch.cat(text_embed_list, dim=0)
            masks_to_repeat = [mask, text_mask, video_mask, complementary_mask]
            (
                mask,
                text_mask,
                video_mask,
                complementary_mask,
            ) = [
                m.repeat((rep_n,) + (1,) * (m.ndim - 1)) if m is not None else None
                for m in masks_to_repeat
            ]
            generation_mask = generation_mask.repeat(rep_n, 1) if generation_mask is not None else None
            speaker_delta = torch.cat(speaker_delta_list, dim=0) if speaker_delta_list else None

        else:
            x, text_embed, v = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_video=drop_video,
            )

            speaker_delta = self.get_speaker_delta(
                speaker_embedding,
                t,
                drop_speaker=drop_speaker,
            )

        if self.normalize_text_context:
            text_embed, text_mask = self._stabilize_text_context(text_embed, text_mask)
        if self.text_joint_proj is not None:
            text_embed = self.text_joint_proj(text_embed)
            if text_mask is not None:
                text_mask = text_mask[:, :text_embed.shape[1]]
                text_embed = text_embed.masked_fill(~text_mask.unsqueeze(-1), 0.0)

        band_args = ()
        band_kwargs = {}
        if self.temporal_band is not None:
            # Construct once after CFG packing and reuse across all paired layers.
            offset, sigma = self.temporal_band(v, seq_len)
            temporal_band_bias = self.temporal_band.bias(offset, sigma, video_len)
            band_args = (temporal_band_bias,)  # ckpt_wrapper forwards positional arguments.
            band_kwargs = {"temporal_band_bias": temporal_band_bias}

        lens = (
            mask.sum(dim=1)
            if mask is not None
            else torch.full((x.size(0),), seq_len, device=x.device, dtype=torch.long)
        )
        # Aligned RoPE的生成  
        rope = self.rotary_embed.forward_from_seq_len(seq_len) # 音频位置0,1,2...T-1
        # aligned RoPE for the video stream: scale positions by the audio/video frame-rate ratio
        video_pos = torch.arange(video_len, device=x.device) # 视频位置0, 4, 8...T-4
        if self.video_rope_scaled:
            video_pos = video_pos * self.audio_video_ratio
        v_rope = self.rotary_embed.forward(video_pos) # 视频流的attention mask

        # video-stream attention mask at native video rate
        if mask is not None:
            if video_mask is not None and video_mask.shape[1] == video_len:
                v_mask = video_mask
            else:
                v_mask = mask[:, :: self.audio_video_ratio][:, :video_len]
                if v_mask.shape[1] < video_len:
                    v_mask = F.pad(v_mask, (0, video_len - v_mask.shape[1]), value=False)
        else:
            v_mask = None

        if self.long_skip_connection is not None:
            residual = x

        intermediates_ctc = {}
        repa_projection = None
        for layer_i, block in enumerate(self.transformer_blocks):
            is_mm = isinstance(block, (MMDiTBlock_VT, MMDiTBlock_AVT))
            has_tail_text = isinstance(block, AudioTextDiTBlock)
            block_mask = None if self.training else mask  # memory issue
            block_v_mask = None if self.training else v_mask
            block_t = (
                t + speaker_delta
                if speaker_delta is not None and layer_i >= self.speaker_condition_start_layer
                else t
            )
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                if is_mm:
                    mm_output = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x,
                        v,
                        block_t,
                        block_mask,
                        block_v_mask,
                        rope,
                        v_rope,
                        text_embed,
                        text_mask,
                        generation_mask,
                        *band_args,
                        use_reentrant=False,
                    )
                elif has_tail_text:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x,
                        block_t,
                        block_mask,
                        rope,
                        text_embed,
                        text_mask,
                        generation_mask,
                        use_reentrant=False,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x,
                        block_t,
                        block_mask,
                        rope,
                        use_reentrant=False,
                    )
            else:
                if is_mm:
                    mm_output = block(
                        x,
                        v,
                        block_t,
                        mask=block_mask,
                        v_mask=block_v_mask,
                        rope=rope,
                        v_rope=v_rope,
                        text=text_embed,
                        text_mask=text_mask,
                        generation_mask=generation_mask,
                        **band_kwargs,
                    )
                elif has_tail_text:
                    x = block(
                        x,
                        block_t,
                        mask=block_mask,
                        rope=rope,
                        text=text_embed,
                        text_mask=text_mask,
                        generation_mask=generation_mask,
                    )
                else:
                    x = block(
                        x,
                        block_t,
                        mask=block_mask,
                        rope=rope,
                    )

            if is_mm:
                if self.text_joint_attention:
                    x, v, text_embed = mm_output
                else:
                    x, v = mm_output

            if compute_ctc and not cache and layer_i in self.layer_map_ctc:  # hack
                projector = self.projectors_ctc[self.layer_map_ctc[layer_i]]
                z_tilde, z_lens = projector(x, lens)
                intermediates_ctc[layer_i] = {"z_tilde": z_tilde, "z_lens": z_lens}
            if return_repa and layer_i == self.repa_layer:
                repa_projection = self.repa_projector(x)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if return_repa:
            if repa_projection is None:
                raise RuntimeError(f"failed to collect REPA activation from layer {self.repa_layer}")
            return output, intermediates_ctc, repa_projection
        return output, intermediates_ctc
