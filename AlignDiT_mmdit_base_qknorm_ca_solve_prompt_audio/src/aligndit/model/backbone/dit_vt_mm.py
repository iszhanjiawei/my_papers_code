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
- text is injected via cross-attention (query: audio stream), identical to the
  baseline DiT_VT_CrossAttn conditioning
- remaining `depth - n_mm_layers` blocks are audio-only single-stream blocks
- audio-stream parameter names are kept identical to DiTBlock/DiTCrossBlock so
  that the audio-only pretrained checkpoint loads directly by key matching
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb

from aligndit.model.modules import DiTCrossBlock, DownsampleLayer
from cosyvoice.transformer.encoder import ConformerEncoder
from f5_tts.model.backbones.dit import ConvPositionEmbedding, DiT
from f5_tts.model.modules import AdaLayerNorm, AttnProcessor, Attention, FeedForward

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
        # 3.3 门控控制 cross-attn 输出加回残差流的比例
        x = x + ca_gate.unsqueeze(1) * ca_output

        # ff_norm是归一化的作用  x_scale_mlp / x_shift_mlp：在 FFN 之前，对归一化后的特征做仿射变换
        norm = self.ff_norm(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp.unsqueeze(1) * self.ff(norm) # 4、各流独立 FFN

        norm = self.v_ff_norm(v) * (1 + v_scale_mlp[:, None]) + v_shift_mlp[:, None]
        v = v + v_gate_mlp.unsqueeze(1) * self.v_ff(norm) # 5、各流独立 FFN

        return x, v


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
        projector_dim=None,
        n_mm_layers=12,
        audio_video_ratio=4,
        video_dim=1024,
        video_rope_scaled=True,
    ):
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
        self.n_mm_layers = min(n_mm_layers, depth)

        self.input_embed = AudioInputEmbedding_MM(mel_dim, dim)
        self.video_embed = VideoInputEmbedding_MM(video_dim, dim, use_conformer=use_conformer)

        block_kwargs = dict(
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
        )  # 前12层：双流    后6层：纯音频
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock_VT(**block_kwargs) if i < self.n_mm_layers else DiTCrossBlock(**block_kwargs)
                for i in range(depth)
            ]
        )

        projector_dim = self.dim if projector_dim is None else projector_dim
        z_dim = self.text_embed.text_embed.num_embeddings + 1
        self.layer_map_ctc = {v: i for i, v in enumerate(layer_indices_ctc)}
        self.projectors_ctc = nn.ModuleList(
            [DownsampleLayer([2, 1], self.dim, projector_dim, z_dim) for _ in self.layer_map_ctc]
        )

        # Initialize the re-created blocks without double-zeroing a gated branch.
        #
        # For an AdaLN-Zero residual y = x + gate * branch(x), only the gate is
        # initialized to zero. The branch output projection must stay normally
        # initialized so that the gate receives a non-zero gradient on the first
        # update; otherwise both factors remain zero forever.
        self.initialize_weights()
        for block in self.transformer_blocks:
            if isinstance(block, MMDiTBlock_VT):
                # Gated MM branches: zero only the modulation/gates. Keep
                # cross_attn.out_proj and v_attn.to_out normally initialized.
                nn.init.constant_(block.cross_attn_ada.weight, 0)
                nn.init.constant_(block.cross_attn_ada.bias, 0)
                nn.init.constant_(block.v_attn_norm.linear.weight, 0)
                nn.init.constant_(block.v_attn_norm.linear.bias, 0)
            else:
                # Audio-only blocks add cross-attention directly without an
                # additional gate, so a zero output projection is safe and
                # preserves the pretrained audio path at initialization.
                nn.init.constant_(block.cross_attn.out_proj.weight, 0)
                nn.init.constant_(block.cross_attn.out_proj.bias, 0)

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
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
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
            if not (drop_text or drop_video):
                x_cond, text_embed_cond, v_cond = self.get_input_embed(**embed_kwargs)
                x_list.append(x_cond)
                text_embed_list.append(text_embed_cond)
                v_list.append(v_cond)
                x_tts, text_embed_tts, v_tts = self.get_input_embed(
                    **embed_kwargs,
                    drop_video=True,
                )
                x_list.append(x_tts)
                text_embed_list.append(text_embed_tts)
                v_list.append(v_tts)
            else:
                x_cond, text_embed_cond, v_cond = self.get_input_embed(
                    **embed_kwargs,
                    drop_text=drop_text,
                    drop_video=drop_video,
                )
                x_list.append(x_cond)
                text_embed_list.append(text_embed_cond)
                v_list.append(v_cond)

            x_uncond, text_embed_uncond, v_uncond = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=True,
                drop_text=True,
                drop_video=True,
            )
            x_list.append(x_uncond)
            text_embed_list.append(text_embed_uncond)
            v_list.append(v_uncond)

            rep_n = len(x_list)
            x = torch.cat(x_list, dim=0)
            v = torch.cat(v_list, dim=0)
            t = t.repeat_interleave(rep_n, dim=0)
            text_embed = torch.cat(text_embed_list, dim=0)
            masks_to_repeat = [mask, text_mask, video_mask, complementary_mask]
            (
                mask,
                text_mask,
                video_mask,
                complementary_mask,
            ) = [m.repeat_interleave(rep_n, dim=0) if m is not None else None for m in masks_to_repeat]
            generation_mask = generation_mask.repeat(rep_n, 1) if generation_mask is not None else None

        else:
            x, text_embed, v = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_video=drop_video,
            )

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
        for layer_i, block in enumerate(self.transformer_blocks):
            is_mm = isinstance(block, MMDiTBlock_VT)
            block_mask = None if self.training else mask  # memory issue
            block_v_mask = None if self.training else v_mask
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                if is_mm:
                    x, v = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x,
                        v,
                        t,
                        block_mask,
                        block_v_mask,
                        rope,
                        v_rope,
                        text_embed,
                        text_mask,
                        generation_mask,
                        use_reentrant=False,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x,
                        t,
                        block_mask,
                        rope,
                        text_embed,
                        text_mask,
                        use_reentrant=False,
                    )
            else:
                if is_mm:
                    x, v = block(
                        x,
                        v,
                        t,
                        mask=block_mask,
                        v_mask=block_v_mask,
                        rope=rope,
                        v_rope=v_rope,
                        text=text_embed,
                        text_mask=text_mask,
                        generation_mask=generation_mask,
                    )
                else:
                    x = block(
                        x,
                        t,
                        mask=block_mask,
                        rope=rope,
                        text=text_embed,
                        text_mask=text_mask,
                    )

            if not cache and layer_i in self.layer_map_ctc:  # hack
                projector = self.projectors_ctc[self.layer_map_ctc[layer_i]]
                z_tilde, z_lens = projector(x, lens)
                intermediates_ctc[layer_i] = {"z_tilde": z_tilde, "z_lens": z_lens}

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output, intermediates_ctc
