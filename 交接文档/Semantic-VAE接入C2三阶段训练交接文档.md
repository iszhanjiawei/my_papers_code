# Semantic-VAE 接入 C2 三阶段训练交接文档

> 更新日期：2026-08-02
> 工作区：`/zjw524/projects/alignDiT_idea6`
> Git 仓库：`/zjw524/projects/alignDiT_idea6/my_papers_code`
> 远程：`https://github.com/iszhanjiawei/my_papers_code.git`
> 当前主分支：`main`

## 1. 给新会话的一句话摘要

当前工作是把 AlignDiT 的 **C2 结构（12 层 MM-DiT + 文本 Cross-Attention，后 6 层纯音频 DiT）** 从 `80 维、100 Hz mel + HiFi-GAN` 改造为 `64 维、40 Hz Semantic-VAE latent + Semantic-VAE decoder`。

已完成 Semantic-VAE codec 上限评测和总体技术方案，证明 Semantic-VAE 1000k 的重建上限显著高于现有 mel+HiFi-GAN。**尚未完成 Semantic-VAE 的训练数据缓存、模型代码接入和三阶段训练。** 当前最大的外部阻塞是本机没有已解压的 LibriSpeech 训练集。

## 2. 为什么选 C2 作为主干

现有 C0-C3、D0-D2 实验表明：

- C2：前 12 层 MM-DiT + 文本 CA，后 6 层纯音频 DiT；
- C2 没有在后 6 层反复注入文本，综合指标最好；
- D1 证明双层 CTC 能明显修复 D0 单层 CTC 导致的高 WER；
- D2 证明在中间 6 层继续注入文本 CA 只能带来早期语义/情感收敛优势，会伤害 SPKSIM 和 AVSync，且 200k 时 WER 优势消失。

C2-200k 的关键结果：

| SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---:|---:|---:|---:|
| 0.63084 | 0.04794 | 0.78610 | 0.51161 |

因此 Semantic-VAE 新主线固定为：

```text
前 12 层：MM-DiT + 文本 Cross-Attention
后  6 层：无文本、无视频交互的纯音频 DiT
```

对应现有工程：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/
  AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio
```

注意：

```text
AlignDiT_mmdit_wav_vae_base_qknorm_ca
```

虽然目录名中有 `wav_vae`，但当前受 Git 跟踪的源码并没有完成 Semantic-VAE 接入，而且不是最新 C2 主干。后续建议从 C2 快照新建独立实验目录，不要直接覆盖 C2 mel 实验。

## 3. Semantic-VAE 资源与论文

### 3.1 论文

```text
/zjw524/projects/alignDiT_idea6/papers/mel-vae-papers/
  Semantic-VAE- Semantic-Alignment Latent Representation for Better Speech Synthesis/
  Semantic-VAE- Semantic-Alignment Latent Representation for Better Speech Synthesis.md
```

### 3.2 复现代码

```text
/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE
```

### 3.3 权重

```text
/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae
/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k
```

当前主方案使用：

```text
/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth
```

权重元数据已核对：

- sample rate：16 kHz；
- encoder rates：`[4, 4, 5, 5]`；
- 总下采样倍率：400；
- latent 帧率：`16000 / 400 = 40 Hz`；
- latent 维度：64。

## 4. 已完成的工作

### 4.1 阶段0 codec 上限评测已完成

已在 CelebVDub 正式 213 条测试集上对比：

- mel + HiFi-GAN；
- acoustic VAE 64D；
- Semantic-VAE 600k；
- Semantic-VAE 1000k。

评测脚本已提交：

- `5a3c92c feat(eval): add CelebVDub codec ceiling benchmark`
- `5e7287b fix(eval): register AV-HuBERT task in codec benchmark`

评测结果：

| Codec | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ | SNR dB ↑ |
|---|---:|---:|---:|---:|---:|
| mel + HiFi-GAN | 0.79952 | 0.04710 | 0.85774 | 0.97969 | -6.4051 |
| Acoustic-VAE 64D sample | 0.92812 | 0.04289 | 0.94263 | 0.99282 | 9.8322 |
| Semantic-VAE 600k sample | 0.92792 | **0.03869** | 0.94534 | 0.99294 | 9.8238 |
| Semantic-VAE 1000k sample | **0.93038** | 0.04037 | **0.94910** | **0.99311** | **10.0196** |

结论：Semantic-VAE 1000k 的重建上限显著高于现有 mel+HiFi-GAN，具备继续接入 C2 的价值。

结果文件：

```text
/zjw524/projects/data/codec_ceiling_celebvdub/_codec_ceiling_summary.json
/zjw524/projects/data/codec_ceiling_celebvdub/_reconstruction_summaries.json
```

注意：

```text
/zjw524/projects/data/codec_ceiling_celebvdub/semantic_vae_1000k_sample
```

里保存的是 codec 重建 WAV 及评测产物，**不是训练所需的 40 Hz latent 缓存**。

### 4.2 已核对 Semantic-VAE 的 latent 实现

Semantic-VAE encoder 输出 `mu` 和 `log_var`，官方提取脚本使用：

```text
z = mu + exp(0.5 * log_var) * eps
```

即 posterior sample。VAE 的 KL 约束只是将总体 posterior 向标准高斯推近，并不代表每个实际 latent 都已经严格零均值、单位方差。

已确定的主方案：

- 沿用 codec ceiling 已测试的 `sample` 路线；
- 每条 utterance 的 sample 必须固定；
- 使用稳定 utterance key + base seed 生成与 GPU rank/顺序无关的随机种子；
- latent 离线提取一次，后续训练不在每个 epoch 重新采样；
- 只用 LibriSpeech train 统计全局逐通道 mean/std；
- 阶段一、二、三必须使用同一份 mean/std；
- decoder 前必须反归一化。

`mu` 路线可作为后续消融，但不能与 sample latent 混放在同一缓存目录。

### 4.3 已分析 mel 预训练权重的可迁移范围

已直接检查：

```text
/zjw524/datasets/AlignDiT_pretrain_LibriSpeech_500000.pt
```

形状变化：

| 参数 | mel checkpoint | Semantic-VAE 模型 | 处理 |
|---|---:|---:|---|
| `input_embed.proj.weight` | `[768,160]` | `[768,128]` | 重新初始化 |
| `input_embed.proj.bias` | `[768]` | `[768]` | 随整个输入层重置 |
| `proj_out.weight` | `[80,768]` | `[64,768]` | 重新零初始化 |
| `proj_out.bias` | `[80]` | `[64]` | 重新零初始化 |

可迁移的纯音频主干约 153.1M 参数，包括：

- time embedding；
- 18 层 attention / FFN / AdaLN；
- convolutional position embedding；
- `norm_out`。

但形状相同不等于语义完全相同：

- mel 是 80D / 100 Hz；
- Semantic-VAE 是 64D / 40 Hz；
- 每帧时间从 10 ms 变为 25 ms；
- 位置卷积、RoPE 和 attention 学到的物理时间意义都改变了。

因此 mel checkpoint 只能做 warm start，不能当成可直接 resume 的完整 checkpoint。

### 4.4 已确定时间对齐策略

当前方案是：

- Semantic-VAE audio latent：40 Hz；
- AV-HuBERT 原生视频特征：25 Hz；
- 阶段三将每条视频特征精确插值到该条音频的 `T_latent`；
- 音频与视频在模型内都是 40 Hz；
- `audio_video_ratio=1`。

不能只写固定 `scale_factor=1.6` 然后假设长度必然相同；必须对每条数据以 `T_latent` 作为目标长度。

### 4.5 已确定解码长度策略

Semantic-VAE 会将输入 WAV 右补齐到 400 samples 的整数倍：

```text
T_latent = ceil(original_num_samples / 400)
decoder_native_length = T_latent * 400
```

codec ceiling 实测 Semantic-VAE 平均比原 WAV 多约 198 samples。每条 manifest 必须保存 `original_num_samples`，最终 WAV 必须裁剪到原始样本数，不能用浮点 duration 反推。

## 5. 当前代码的真实状态

### 5.1 现有 LibriSpeech 预训练入口

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/
  AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/
  src/aligndit/script/train/pretrain.py
```

它当前构造：

```text
CFM_notext(DiT_noText)
```

数据集固定为：

```text
CustomDataset_mel_rep
```

输入是：

```text
mel_tacotron:       [T100, 80]
hubert_large_ll60k: [T50, 1024]
```

损失是：

```text
mel flow matching MSE
+
HuBERT cosine projection loss
```

当前配置：

```text
src/aligndit/config/pretrain.yaml
```

启动脚本：

```text
src/aligndit/run/train/pretrain.sh
```

### 5.2 重要纠正：LibriSpeech 预训练没有 CTC

`pretrain.py` 中的 `projectors` 是 HuBERT 表征对齐头，不是 CTC。

原 mel 预训练的对齐是：

```text
100 Hz block hidden
  -> DownsampleLayer([2,1])
  -> 50 Hz / 1024D
  -> 与 HuBERT 50 Hz 特征计算 cosine loss
```

如果不修改就用在 40 Hz latent：

```text
40 Hz -> [2,1] -> 20 Hz
```

会与 HuBERT 50 Hz 严重错位。正确方案：

```text
HuBERT 50 Hz -> 按每条有效长度插值到 40 Hz
40 Hz hidden -> stride [1,1] projector -> 40 Hz / 1024D
```

阶段一先 `proj_lambda=0`，只做 latent flow loss；阶段二修复 40/50 Hz 对齐后再逐步引入 HuBERT projection loss。

### 5.3 现有代码还不能直接训 latent

现有代码仍然存在以下 mel 假设：

- `pretrain.py` 构造 `MelSpec_tacotron`；
- dataset 硬编码读取 `/mel_tacotron/`；
- CFM 从 mel module 推断 `num_channels=80`；
- sample logger 把模型输出当 mel 交给 HiFi-GAN；
- Trainer 只有单一学习率；
- Trainer 没有 `max_updates`，只按 epoch 停止；
- 缺少严格的冻结/逐层解冻能力；
- 缺少“仅权重初始化”与“同阶段严格 resume”的区分；
- EMA 和 optimizer 在当前 Trainer 中创建得过早；
- 没有 Semantic-VAE decoder 的 sample logging。

因此现在不能只把 `n_mel_channels: 80` 改成 64 就启动训练。

## 6. 当前卡点

### 6.1 本机缺少 LibriSpeech 数据

本机存在：

```text
/zjw524/datasets/AlignDiT_pretrain_LibriSpeech_500000.pt
```

但不存在：

```text
/zjw524/projects/data/LibriSpeech_notext
```

也没有发现已解压的：

```text
train-clean-100
train-clean-360
train-other-500
dev-clean
dev-other
```

开始阶段一/二前，必须先从旧服务器挂载或复制这些数据。

### 6.2 Semantic-VAE latent 训练缓存尚未生成

尚需为：

- LibriSpeech train/dev；
- CelebVDub train/dev/test；

提取固定 `[T40,64]` latent，并生成 manifest 和 mean/std。

预计 float32 缓存大小：

- LibriSpeech 960h：约 35 GB；
- CelebVDub 91h：约 3.4 GB。

### 6.3 三阶段训练基础设施尚未实现

尚缺：

- latent dataset/collate；
- 64D / 40 Hz CFM 接口；
- 预训练权重白名单迁移；
- 参数冻结和多 LR parameter groups；
- 精确 `max_updates`；
- 阶段间 weights-only init；
- 40 Hz HuBERT projector；
- 40 Hz CTC；
- 视频 25 -> 40 Hz 精确插值；
- Semantic-VAE decoder 日志采样。

## 7. 已确定的三阶段训练计划

论文上分三阶段，实际执行建议分为 6 个独立任务：

| 论文阶段 | 实际任务 | 数据 | 建议 updates |
|---|---|---|---:|
| 阶段一 | S1：新接口校准 | LibriSpeech latent | 10k |
| 阶段二 | S2a：解冻后 6 层 | LibriSpeech latent | 10k |
| 阶段二 | S2b：解冻后 12 层 | LibriSpeech latent | 10k |
| 阶段二 | S2c：全音频主干适配 | LibriSpeech latent | 70k |
| 阶段三 | S3a：训练新多模态模块 | CelebVDub latent/text/video | 5k |
| 阶段三 | S3b：完整 C2 微调 | CelebVDub latent/text/video | 195k |

建议初始预算：

```text
LibriSpeech latent 适配：100k
CelebVDub C2 微调：200k
```

100k 只是首轮正式配置，应根据 LibriSpeech dev 结果决定是否延长到 200k。

### 7.1 阶段一 S1：新接口校准

数据：

```text
LibriSpeech train-clean-100/360/other-500 固定归一化 latent
```

模型：

```text
DiT_noText
depth=18, dim=768, heads=12, ff_mult=2
qk_norm=rms_norm
audio_dim=64
```

从 mel 500k EMA 加载兼容主干，重置：

```text
transformer.input_embed.proj.*
transformer.proj_out.*
transformer.projectors.*
```

只训练：

```text
transformer.input_embed.proj.{weight,bias}
transformer.proj_out.{weight,bias}
```

冻结：

```text
time_embed
input_embed.conv_pos_embed
transformer_blocks.0-17
norm_out
Semantic-VAE
```

损失：

```text
latent flow matching only
proj_lambda=0
```

建议配置：

```text
max_updates=10000
LR=1e-4
warmup=500
AdamW weight_decay=0.01（bias 不衰减）
BF16
grad_clip=1.0
EMA beta=0.999
seed=666
```

`proj_out` 零初始化时，第一个 update 主要只有输出层获得梯度；从第二个 update 开始，梯度才逐渐传到新 input projection，这是正常现象。

### 7.2 阶段二 S2：逐步解冻音频主干

S2a（10k）：

- 训练 input/output interface；
- 解冻 `conv_pos_embed`、`norm_out`、blocks 12-17；
- interface LR `5e-5`；
- 已加载主干 LR `1e-5`；
- warmup 500。

S2b（10k）：

- 解冻范围扩展到 blocks 6-17；
- interface LR `3e-5`；
- 主干 LR `1e-5`；
- blocks 0-5 和 time embedding 仍冻结；
- warmup 500。

S2c（70k）：

- 全部 18 层音频主干解冻；
- interface LR `2e-5`；
- blocks 6-17 / time / conv-pos / norm-out LR `1e-5`；
- 刚解冻的 blocks 0-5 LR `5e-6`；
- 新 40 Hz HuBERT projector LR `2e-5`；
- warmup 1000。

S2c 中的 HuBERT projection loss：

```text
HuBERT 50 Hz -> 按有效长度插值到 T40
block-13 hidden 40 Hz -> stride [1,1] projector -> 40 Hz/1024D
cosine loss
```

`proj_lambda` 建议前 5k 从 0 线性 ramp 到 0.1，先检查 projection 梯度与 flow 梯度量级，再决定是否提高到原预训练使用的 1.0。

### 7.3 阶段三 S3：C2 CelebVDub 多模态微调

数据：

```text
CelebVDub 40 Hz normalized latent
+ char text
+ AV-HuBERT video feature interpolated exactly to T_latent
```

模型固定：

```text
depth=18
n_mm_layers=12
n_text_layers=12
prompt_isolated_ca=False
qk_norm=rms_norm
audio_dim=64
audio_video_ratio=1
layer_indices_ctc=[6,12]
ctc_lambda=0.1
```

CTC 保留 C2 的两个监督位置，但 projector 必须保持 40 Hz，不能继续下采样到 20 Hz，并且 CTC head 完整重新初始化。

从 S2c EMA 加载：

- 64D input projection；
- position embedding；
- time embedding；
- 18 层的音频 attention / FFN / AdaLN / QK-Norm；
- `norm_out`；
- 64D output projection。

新初始化：

- text embedding；
- 前 12 层文本 cross-attention；
- video embedding / Conformer；
- video attention / FFN / gate；
- CTC heads。

S3a（5k）：

- 冻结 S2c 加载的音频路径；
- 只训练新文本/视频/MM-DiT/CTC 参数；
- LR `5e-5`；
- warmup 500。

S3b（195k）：

- 全量解冻；
- 新文本/视频/MM-DiT/CTC LR `5e-5`；
- latent input/output interface LR `2e-5`；
- 已适配音频主干 LR `1e-5`；
- 必要时前部敏感音频层 LR `5e-6`；
- warmup 5k；
- CelebVDub 总预算保持 `5k + 195k = 200k`。

## 8. 数据组织和质量检查

### 8.1 latent 缓存 manifest 必须记录

每条至少记录：

```text
relative_utterance_key
audio_path
latent_path
original_num_samples_16k
padded_num_samples
latent_frames
duration
vae_checkpoint_id/hash
latent_mode=sample
per_utterance_seed
normalization_stats_id/hash
```

缓存质量检查：

- shape 必须为 `[T,64]`；
- 所有值 finite；
- `T == ceil(original_num_samples / 400)`；
- 没有漏文件；
- 没有重复 key；
- 随机解码 100 条，裁剪后长度与原 WAV 一致；
- 使用临时文件 + atomic rename 防止中断留下半文件。

### 8.2 归一化

- 只用 LibriSpeech train 的有效帧统计；
- 使用 float64 Welford 累计；
- padding 在归一化之后填 0，使 padding 对应训练均值；
- CelebVDub 不重新定义坐标系；
- checkpoint 要保存 stats hash。

### 8.3 CelebVDub CTC 可行性

现有 CelebVDub train Arrow：

- 79,613 条；
- 约 91.06 小时。

40 Hz 预检发现：

- 72 条 `T_latent < len(text)`；
- 考虑相邻重复字符需要 blank 后，约 105 条不满足 CTC 最短路径；
- 其中存在明显错标数据。

正式训练前必须用真实 tokenizer 再检查，修复或过滤这些样本。不能仅依赖 `zero_infinity=True` 将它们静默变成零损失。

## 9. Batch 与 GPU 配置

为保持每个 update 看到的音频秒数一致，不能直接沿用 mel 的 frame threshold。

### 9.1 LibriSpeech 阶段

原配置：

```text
13500 mel frames/GPU @100 Hz = 135 sec/GPU
```

40 Hz 配置：

```text
5400 latent frames/GPU @40 Hz = 135 sec/GPU
max_samples=32
```

如需匹配原 8 GPU 全局 batch：

| GPU 数 | frames/GPU | grad accumulation |
|---:|---:|---:|
| 8 | 5400 | 1 |
| 4 | 5400 | 2 |
| 1 | 5400 | 8 |

### 9.2 CelebVDub C2 阶段

原 C2：

```text
9000 mel frames/GPU @100 Hz = 90 sec/GPU
```

40 Hz 配置：

```text
3600 latent frames/GPU @40 Hz = 90 sec/GPU
max_samples=32
```

| GPU 数 | frames/GPU | grad accumulation |
|---:|---:|---:|
| 4 | 3600 | 1 |
| 1 | 3600 | 4 |

首轮不要因为 40 Hz 序列更短就直接放大每 update 的音频秒数，否则无法将性能变化干净归因于 Semantic-VAE。

## 10. Checkpoint、EMA 与 resume 规则

### 10.1 同一小阶段中断

例如 S2c 训到 35k 中断，可严格恢复：

- online model；
- EMA；
- optimizer；
- scheduler；
- update=35k。

### 10.2 跨阶段

S1 -> S2a、S2a -> S2b、S2c -> S3a 等跨阶段操作必须：

1. 读取上一阶段 EMA 权重；
2. 仅作为新模型 weights-only initialization；
3. 重建 optimizer；
4. 重建 scheduler；
5. 重建 EMA 并重置 EMA step；
6. 新阶段 update 从 0 开始。

每个阶段使用独立目录：

```text
ckpts/svae_s1_interface/
ckpts/svae_s2a_tail6/
ckpts/svae_s2b_tail12/
ckpts/svae_s2c_full_audio/
ckpts/svae_s3a_c2_new_modalities/
ckpts/svae_s3b_c2_full/
```

当前 Trainer 会扫描 `model_last.pt` 并自动恢复 optimizer，所以跨阶段不得共用 checkpoint 目录。

正确初始化顺序：

```text
构造模型
-> 加载上一阶段 EMA / mel EMA 白名单权重
-> 显式重置指定参数
-> 设置 requires_grad
-> 构造多 LR optimizer groups
-> 创建 EMA
-> Accelerator.prepare
-> 开始训练
```

## 11. 验收和评估计划

### 11.1 阶段一 -> 阶段二

使用固定 LibriSpeech dev manifest：

- 固定 mask / noise / time 的 deterministic flow loss；
- input/output 梯度 finite；
- 第二个 update 开始 input projection 有非零梯度；
- 无 NaN/Inf；
- 固定 audio inpainting 解码长度正确；
- 最后 2k updates 的 dev loss 如果仍下降超过约 2%，则 S1 延长到 20k。

### 11.2 阶段二 -> 阶段三

在 dev-clean + dev-other 上监控：

- flow loss；
- HuBERT projection loss；
- 固定 200 条 audio inpainting ASR WER；
- SPKSIM；
- 长度正确性；
- 早/中/晚层 gradient norm 和 hidden norm。

选定唯一 S2c EMA 作为 C2 latent 音频预训练权重。

### 11.3 阶段三

- 从 CelebVDub train 按顶层 ID/说话人分组划出固定 dev；
- 内部 dev 每 5k 监控 diff/CTC loss；
- 保存并评估累计 50k / 100k / 150k / 200k；
- 正式 213 条 test 不参与早停和 checkpoint 选择；
- 最终指标：SPKSIM、WER、EMOSIM、AVSync；
- 小于约 0.002 的差异需要 paired bootstrap，不直接宣称稳定提升。

## 12. 论文公平性与必做消融

主方案在 mel 500k checkpoint 之后又增加约 100k LibriSpeech latent 适配。不能将最终提升全部归因于 Semantic-VAE，而忽略额外训练预算。

至少需要：

1. 现有 mel C2 对照；
2. latent C2 完全随机初始化；
3. latent C2 部分加载但不分阶段；
4. latent C2 部分加载 + 三阶段训练（主方案）；
5. mel checkpoint 继续训练相同 LibriSpeech updates 的计算量控制组。

如果训练预算允许，最严谨的补充是：

```text
mel 原生 LibriSpeech 预训练 -> C2 CelebVDub
latent 原生 LibriSpeech 预训练 -> C2 CelebVDub
```

两者使用相同预训练数据和 update 预算。

## 13. 下一步执行顺序

新会话应按以下顺序继续，不要直接启动训练。

### 步骤 1：恢复数据

- 确认 LibriSpeech 在新服务器的真实路径；
- 恢复 train-clean-100/360/other-500；
- 恢复 dev-clean/dev-other；
- 重建/repair `raw.arrow` 和 `duration.json`；
- 检查 Arrow 是否仍包含旧服务器 `/home/...` 绝对路径；
- 优先改成相对路径 + 显式 data root。

### 步骤 2：创建独立 C2 Semantic-VAE 快照

建议从：

```text
AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio
```

复制为新目录，例如：

```text
AlignDiT_mmdit_c2_semantic_vae
```

不要覆盖已完成的 C2 mel 快照。

### 步骤 3：实现可复现 latent 提取

- 使用 Semantic-VAE 1000k EMA；
- per-utterance deterministic posterior sample；
- 输出 `[T,64]` float32；
- 生成 manifest；
- 全量 QC；
- 统计 Libri train mean/std；
- 生成 Libri dev 和 CelebVDub latent。

### 步骤 4：实现 latent 纯音频预训练

建议新建：

```text
src/aligndit/script/train/pretrain_semantic_vae.py
```

而不是改变旧 `pretrain.py` 的 mel 语义。

需要增加：

- latent dataset；
- 64D CFM；
- init-only loader；
- strict resume loader；
- freeze groups；
- parameter-specific LR；
- max updates；
- Semantic-VAE sample decode；
- 40 Hz HuBERT alignment。

### 步骤 5：单卡 smoke test

当前机器如仍只有 1 张 4090，先测试：

- 构造模型；
- 白名单权重加载；
- input/output 确实重置；
- 冻结名单正确；
- 2 个 optimizer updates；
- 第 2 步 input projection 梯度非零；
- VAE 始终无梯度；
- latent 两次读取一致；
- decode 长度裁剪正确；
- 同阶段 resume 与跨阶段 weights-only init 均正确。

### 步骤 6：按 S1 -> S2a -> S2b -> S2c 训练

- 每段使用独立 config；
- 每段使用独立 checkpoint 目录；
- 阶段间只加载上一阶段 EMA 权重；
- 先完成 Libri dev 验收，再进入 CelebVDub。

### 步骤 7：实现并训练 S3 C2

- 25 Hz video -> exact 40 Hz；
- C2 结构保持 12+6；
- CTC 保留 `[6,12]`；
- CTC 头保持 40 Hz；
- S3a 5k 先训新多模态参数；
- S3b 195k 全量微调；
- 评估 50k/100k/150k/200k。

## 14. Git 与实验跟踪要求

用户已明确要求：每一个代码/配置/启动脚本步骤都要 Git commit 并 push，便于回滚和论文实验跟踪。

建议提交拆分：

1. `feat(data): add deterministic Semantic-VAE latent extraction`
2. `feat(data): add fixed latent normalization and manifests`
3. `feat(pretrain): add 64D Semantic-VAE flow pretraining`
4. `feat(train): add staged freezing and optimizer groups`
5. `feat(pretrain): align 50Hz HuBERT targets to 40Hz latent`
6. `feat(model): add C2 Semantic-VAE backbone integration`
7. `feat(train): add 40Hz video alignment and CTC heads`
8. `test(svae): cover migration, gradients, masks and decode length`
9. `feat(run): add S1-S3 launch configs and scripts`

每次 push 后要核对本地 HEAD 与远端分支一致。

不得提交：

- LibriSpeech/CelebVDub 数据；
- latent 缓存；
- checkpoint；
- 训练日志；
- 生成 WAV；
- Hydra outputs；
- TensorBoard/W&B 产物。

当前工作树中还有两个未跟踪 `data` 软链接：

```text
AlignDiT_mmdit_base/data
AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/data
```

这些是本机数据路径，不要误加入 commit。

## 15. 新会话开始时的建议检查清单

1. 先完整阅读本文档和 `my_papers_code/AGENTS.md`。
2. 执行 `git status --short --branch`，不要覆盖用户改动。
3. 确认 Semantic-VAE 1000k EMA 权重仍存在。
4. 确认 mel 500k LibriSpeech checkpoint 仍存在。
5. 确认 LibriSpeech 是否已恢复；如未恢复，先处理数据，不启动训练。
6. 不要把 codec ceiling WAV 目录当成 latent cache。
7. 不要直接修改旧 `pretrain.py` 破坏 mel 复现性。
8. 新建 C2 Semantic-VAE 独立快照后再开发。
9. 代码完成后先单卡 smoke test，再切换到 4/8 卡正式训练。
10. 每个实现步骤单独 commit/push。
