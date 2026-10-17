# Fixed AV+VA + Synchformer

本实验从 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_fixed_band_av_va` 独立复制。没有复制旧 checkpoint、训练日志和 TensorBoard 运行产物，没有修改源实验。源码在本目录内独立保存，外部只复用固定数据缓存和预训练权重。

## 实现方式

```text
原始 RGB 视频
  → 冻结 Synchformer visual encoder
  → 原生 [窗口数 × 8, 768] 特征缓存
  → 窗口内 8-token 位置编码 + Linear/SiLU/门控 MLP
  → 按每个样本有效长度 nearest-exact 对齐到 40 Hz 音频 latent
  → c_audio[t] = timestep_embedding + sync_embedding[t]
  → 音频流各层 AdaLN / attention、MLP gate / 最终输出 AdaLN
```

实现对应 HunyuanVideo-Foley 论文公式 (2) 和参考代码的 `sync_modulation=True` 路径。参考仓库发布的 XL/XXL 配置实际上使用 `sync_modulation=False, add_sync_feat_to_audio=True`；本实验实现的是用户指定的 **Synchformer 与 timestep 相加进行调制**，并不声称原样复现其发布模型。

视频流使用全局 timestep，Synchformer 不直接调制视频流。前 12 个多模态块、后 6 个音频块及输出层的音频条件支持逐 token 调制。原先的 AV-HuBERT 输入、固定 100 ms AV/VA 双向先验、64D/40 Hz Semantic-VAE、CAM++ speaker 的末 6 层条件，以及学习率和 CTC warmup 均保留。

新增 6 个参数张量：`sync_pos_emb`、`sync_in.0.{weight,bias}` 和 `sync_in.2.{w1,w2,w3}.weight`。最后的输出投影 `w2` 初始化为零，加载父模型时先保留原有输出，再学习同步条件。编码器离线冻结，训练时不加载它。

时间轴和条件丢弃约定：

- 每个样本只对有效同步 token 插值，其他样本的 padding 不改变其时间比例。
- prompt 和 padding 区域在同步投影后置零，避免位置编码、Linear bias 泄漏。
- CFG 完整条件分支保留同步特征，TTS/无条件分支与视频一起丢弃；支持 batch 大于 1。
- `CFM.sample(sync_feat=..., sync_lens=..., sync_target_lens=...)` 接收目标视频原生特征，自动将其放到参考音频之后。`sync_target_lens` 是目标片段的 40 Hz latent 长度。截断输出时保留原时间尺度。
- 训练 `CFM.forward` 使用全片段特征，由生成区间 mask 决定调制范围，不传 `sync_target_lens`。

## 权重与缓存

默认预训练权重：

```text
${ROOT_PREFIX}/zjw524/projects/data/pretrained_models/HunyuanVideo-Foley/synchformer_state_dict.pth
```

固定 SHA256：`8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c`。

默认同步缓存：

```text
${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/synchformer_25fps_16f_stride8
```

输入为原始 RGB 视频，按时间戳采样 25 fps，短边缩放/中心裁剪到 224×224，归一化到 [-1,1]；16 帧窗口、8 帧步长，每窗输出 8×768 特征。不足 16 帧重复末帧，之后只使用完整窗口。不会用灰度嘴部视频或 AV-HuBERT 特征代替 Synchformer 输入。

缓存记录编码器 SHA256、预处理、原视频指纹、完整相对路径和覆盖报告。Dataset 会检查完整 inventory 身份及覆盖，每次读取仍检查特征形状、有限值和元数据。现有 79,826 条缓存可复用；无需重新生成。

需要在另一台机器重新准备时：

```bash
source env.sh
export PYTHONPATH=src
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
bash scripts/download_synchformer.sh
bash scripts/extract_synchformer_multigpu.sh
"$PYTHON_BIN" -u scripts/extract_synchformer.py --audit-only
```

`--limit` 仅用于独立 debug 缓存目录，不能覆盖已完成的全量缓存。编码器源码与来源、许可证位于 `src/aligndit/third_party/synchformer/`。额外提取依赖列在 `pyproject.toml` 的 `synchformer` extra；当前环境已有所需依赖。

## 训练入口

在本副本目录运行：

```bash
source env.sh
export PYTHONPATH=src
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$PYTHON_BIN" -u scripts/preflight_synchformer.py --report logs/synchformer_preflight.json
```

配置为 `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va_synchformer.yaml`，继承原 AV+VA 配置。默认仍由经过哈希校验的 **S2c 70k EMA 音频父权重** 初始化；并非自动续训已经训练过的 AV+VA checkpoint。

正式启动四卡训练：

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_fixed_band_av_va_synchformer_4x4090.sh \
  > logs/train_fixed_band_av_va_synchformer.log 2>&1 < /dev/null &
```

启动脚本先完成 preflight，再启动对应 run 的 TensorBoard 并核验 HTTP 响应，最后启动 Accelerate。`TENSORBOARD_PORT` 默认 6020，冲突时显式选择其他端口。实际 PID、端口和 logdir 记录在 `logs/tensorboard_synchformer.json`。日志包含原有总损失、flow/CTC、speaker、固定带宽，以及新增同步投影梯度/权重范数。

新 checkpoint 默认目录：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_fixed_band_av_va_synchformer_ctc003_warmup10k30k_40hz_CelebVDub_char
```

训练入口要求启用本实验同步条件并使用专属 run/checkpoint 名。`speaker_training_contract.json` 同时记录 AV/VA 方向、固定时间带宽、同步调制作用流、编码器身份和缓存预处理。推理要求这些字段匹配，避免把张量形状相同的旧实验权重误当成本实验。

## 推理与评测

训练得到新 checkpoint 后运行：

```bash
CKPT_STEP=200000 EVAL_GPU=0 \
  bash src/aligndit/run/eval/infer_celebvdub_s1_svae_speaker_fixed_band_av_va_synchformer.sh
```

生成并执行原有四指标评测：

```bash
CKPT_STEP=200000 EVAL_GPU=0 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_fixed_band_av_va_synchformer.sh
```

保持原 CelebVDub Setting 1 的同片段 GT 音频参考协议，默认 EMA、32 NFE、CFG text/video=5/2。测试读取目标视频同步缓存，不在线加载编码器。推理和指标输出默认进入新实验 checkpoint 目录。

## 验证

```bash
PYTHONPATH=src "$PYTHON_BIN" -m unittest discover -s tests -v
PYTHONPATH=src "$PYTHON_BIN" scripts/test_synchformer_cache.py -v
PYTHONPATH=src "$PYTHON_BIN" scripts/test_synchformer_dataset.py -v
PYTHONPATH=src "$PYTHON_BIN" src/aligndit/script/misc/smoke_test_fixed_temporal_band_bidirectional.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_synchformer_real_parent.py
```

最后一项只做真实父权重和真实数据的前向/反向检查，不执行 optimizer update，不生成训练 checkpoint。运行记录见本目录 `logs/`；本次验证结果见 `VALIDATION_SYNCHFORMER.md`。功能正确性测试不能替代训练后的配音质量和同步性评测。
