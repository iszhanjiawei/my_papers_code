# Semantic-VAE 接入 C2 与 500k 从头音频预训练交接文档

> 更新日期：2026-08-04
> 工作区：`/zjw524/projects/alignDiT_idea6`
> Git 仓库：`/zjw524/projects/alignDiT_idea6/my_papers_code`
> 远程：`https://github.com/iszhanjiawei/my_papers_code.git`
> 当前主分支：`main`

## 0. 2026-08-04 当前权威路线

当前主线不再采用 mel 500k checkpoint 的 S1/S2 分阶段 warm-start，而是在独立快照：

```text
AlignDiT_mmdit_c2_semantic_vae/
```

中使用 LibriSpeech 960h 从头训练一个 `64 维、40 Hz` Semantic-VAE latent 音频模型，预算固定为
`500000 optimizer updates`。训练目标为：

```text
normalized Semantic-VAE latent flow matching
+
同长度 40 Hz HuBERT representation alignment
```

该音频预训练权重完成后，再用于初始化 C2 多模态模型的 audio path。下文原有的 mel warm-start
三阶段方案仅保留为设计演进记录和后续消融，不是当前正式执行路线。

### 0.1 当前状态

| 项目 | 状态 | 完成判据 |
|---|---|---|
| LibriSpeech immutable inventory | 已完成 | train 281,241，dev 5,551，总计 286,792 |
| Semantic-VAE raw latent cache | 已完成 | 286,792 个 FP32 `[T,64]` fixed posterior sample；full completion/index 已发布并通过只读全量验证 |
| HuBERT exact-40-Hz cache | 已完成并独立审计 | 286,792 个 FP32 `[T,1024]` 文件；139,999,682 个目标帧；complete/index/资源 SHA、全 inventory 顺序和 68 个确定性抽样全部通过 |
| train-only normalization | 已完成 | 281,241 条 train、138,504,846 帧、64 通道 float64 Welford population mean/std 已原子发布并校验 |
| 500k 音频预训练代码 | 已实现并推送 | 基础实现 `ea7b304`；padding 修复 `282dad8`；A40 launcher 参数 `aac4e59` |
| 正式 6×A40 训练 | 已启动并通过启动验收 | 6 rank contract 正确；30,000 frames/GPU、max 64；连续超过 20 个真实 update，loss 有限，无 OOM、NCCL、长度错配或数据错误 |
| CelebVDub/C2 latent 微调 | 后续工作 | 不属于本轮 LibriSpeech 纯音频预训练任务 |

缓存根目录：

```text
/home/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32
```

### 0.2 当前已实现代码

- `src/aligndit/config/pretrain_semantic_vae.yaml`
  - 从头初始化 18 层 `DiT_noText`；
  - 64D、40 Hz、16 kHz、hop 400；
  - HuBERT projector stride `[1,1]`，不再把 40 Hz 下采样到 20 Hz；
  - `max_updates: 500000`、LR `7.5e-5`、warmup 20k、BF16、seed 666、`proj_lambda: 1.0`；
  - 正式实测配置为 30,000 frames/GPU、`max_samples=64`；不得在同一 contract 下改动。
- `SemanticVaePretrainDataset`
  - 强制 latent/HuBERT 都是 full completion；
  - 校验 inventory、consolidated index、SHA、shape、dtype、finite 和逐条等长；
  - 只接受 train-only mean/std，并在归一化之后把 padding 填为 0。
- `PrecomputedAudioRepresentation`
  - 跳过 mel frontend，同时向上游 CFM 提供 64 通道、16 kHz、hop 400 元数据。
- `DiT_noText`
  - 支持 Hydra `ListConfig` 的可配置 projector stride；
  - 训练时严格使用有效帧 attention mask；
  - 每层重新清零 padding；
  - projector 按单条真实长度计算，避免 GroupNorm 受同 batch padding 比例影响。
- `ConvPositionEmbedding`
  - mask 模式在每个卷积栈子层后重新清零 padding；
  - 修复第一层卷积在 padding 区生成值、第二层再把它卷回有效边界帧的问题；
  - standalone-vs-padded CPU 回归差异从旧行为约 `0.131` 降到 `2.24e-08`。
- `Trainer_notext`
  - 精确停在 500k optimizer updates；
  - scheduler/warmup 按 Accelerate world size 正确换算；
  - 以 global update + rank 派生随机流，使同一 contract 的 resume 不重复 noise/time/span/dropout；
  - 当前确定性 resume 模式明确要求 `grad_accumulation_steps=1`。
- `pretrain_semantic_vae.py`
  - update 0 前发布 immutable `training_contract.json`；
  - contract 绑定 resolved config、world size、BF16/runtime、缓存 completion SHA、normalization SHA 和关键源码 SHA；
  - checkpoint 存在但 contract 缺失时 fail closed，并拒绝外来 pretrained checkpoint 混入。

旧入口 `src/aligndit/script/train/pretrain.py` 保持原有 mel 语义，没有被改写。

### 0.3 正式训练前强制门禁

1. latent 与 HuBERT 都必须有 full completion marker，并通过 index/hash/count/frame 校验。
2. 运行 `compute_librispeech_svae_train_stats.sh`，验证 64 个 mean/std 全部 finite、`std > 0`，记录文件 SHA。
3. 构造完整 `SemanticVaePretrainDataset`，抽查首尾和固定随机样本，确认 latent/HuBERT 严格同长度。
4. 用真实模型和真实最坏动态 batch，在一张 A40 上完成至少两次 forward/backward/AdamW/EMA step。
5. 单卡只用于筛选 frame budget；必须再用 6 个 DDP rank 跑同一最坏 batch canary，计入 DDP/NCCL 额外显存。
6. benchmark 必须确认总 loss、各 loss 分量和 gradient norm 都 finite，并且不写正式 checkpoint。
7. 正式任务启动后，必须看到 6 个 worker 连续完成多个 update，且 contract、LR、loss、显存、GPU 利用率和日志全部正常。

动态 frame batching稳定的是每步有效帧数，不代表增大 frame budget 后必须线性放大学习率。首轮继续使用
`7.5e-5`，避免同时改变 batch 和优化语义。

### 0.4 已完成门禁、正式运行与恢复依据

权威文件与 SHA256：

| 文件 | SHA256 |
|---|---|
| `state/latents/complete.json` | `e255f8ddea5181436283510538ad1bd6bf6808bbe61d3081f3f38977c91be69b` |
| `state/hubert_40hz/complete.json` | `2e66525965d3d48495036c7c60772520d1a233832425fbc669614127df1b0f45` |
| `state/latents/train_normalization.json` | `65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6` |
| 正式 `training_contract.json` | `0f19420a6fd6244c45c5c4020e8779392fbd32e716faadebde86ea03c069e580` |

HuBERT completion 记录 286,792 条、139,999,682 个 40 Hz target frames、174,674,621 个 native
frames 和 573,475,406,848 bytes。最终发布前发现的 3 个旧原子临时文件与对应正式文件逐字节一致，
已可恢复地隔离到：

```text
/home/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/
  quarantine/hubert_40hz/orphans/finalize-orphan-cleanup-20260804T0418CST/
```

它们没有被删除，也不在 active feature tree 内。

显存门禁结果：

| 配置 | 单卡 peak reserved | 6-rank peak reserved | 结论 |
|---|---:|---:|---|
| 13,500 frames、max 32 | 17.799 GiB | 未作为最终配置测试 | 安全但显存利用不足 |
| 22,000 frames、max 32 | 27.902 GiB | 未作为最终配置测试 | max 32 已接近 batch 数饱和 |
| 30,000 frames、max 64 | 41.182 GiB | rank0 41.822 GiB；其余约 41.168 GiB | 正式采用；留有 DDP/NCCL 余量 |

正式训练固定使用：

```text
Git HEAD:       aac4e590a947142148f072f7543c4fc8ec3c56be
GPU:            physical 2,3,4,5,6,7；一张卡一个 rank
world size:     6
precision:      BF16
frame budget:   30000 / GPU
max samples:    64 / GPU
max updates:    500000
warmup:         20000 updates
learning rate:  7.5e-5
```

正式日志：

```text
AlignDiT_mmdit_c2_semantic_vae/logs/
  pretrain_semantic_vae_6xa40_fb30000_ms64_20260804T051453.log
```

正式 checkpoint：

```text
/home/zjw524/projects/data/ckpts/
  AlignDiT_SemanticVAE_pretrain_semantic_vae_40hz_LibriSpeech_svae40/
```

启动后已确认物理 GPU 2–7 分别严格映射 rank 0–5，29620 为实际 rendezvous 端口；连续超过 20 个
真实 update 的 loss/diff_loss/proj_loss 均有限，日志没有 feature/latent 长度 warning、NaN/Inf、OOM、
NCCL、Traceback 或 ChildFailed。GPU 0/1 属于其他任务，整个缓存门禁、benchmark、canary 和正式启动
过程都没有使用它们。

若后续需要恢复，必须继续使用同一源码、同一 6 卡 world、BF16、30,000/64 和同一缓存/normalization；
脚本会验证现有 contract，并从 `model_last.pt` 恢复。`model_last.pt` 每 5,000 updates 才写入，
`model_<update>.pt` 每 50,000 updates 写入，因此刚启动时只有 contract、没有 checkpoint 是正常行为。

## 1. 给新会话的一句话摘要

当前工作是把 AlignDiT 的 **C2 结构（12 层 MM-DiT + 文本 Cross-Attention，后 6 层纯音频 DiT）**
从 `80 维、100 Hz mel + HiFi-GAN` 改造为 `64 维、40 Hz Semantic-VAE latent + Semantic-VAE decoder`。
当前正在执行的第一步，是先在 LibriSpeech 上从头完成 500k 的 40 Hz 纯音频预训练；详细状态以第 0 节为准。

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

## 5. 历史代码状态（已由第 0 节的新实现取代）

本节记录 2026-08-02 方案制定时旧 mel 快照的状态，不得据此判断当前代码仍缺少 latent 接口。

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

## 6. 历史卡点（现已解决或进入新流程）

本节是旧服务器/旧方案的历史记录。当前服务器已有 LibriSpeech、独立 Semantic-VAE 快照和完整 latent cache；
HuBERT、normalization 与正式启动状态以第 0 节为准。

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

## 7. 历史方案：mel 500k warm-start 三阶段适配（当前不执行）

以下 S1/S2/S3 方案是此前针对“部分复用 mel 预训练权重”的设计。当前实验已经改为在相同
LibriSpeech 数据上从头进行 500k Semantic-VAE 音频预训练，因此不得按本节启动正式任务。
本节仅用于记录设计演进，以及后续 `scratch vs mel warm-start` 消融。

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

## 9. 历史 warm-start 方案的 Batch 与 GPU 配置（当前不执行）

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

## 10. 历史分阶段方案的 Checkpoint、EMA 与 resume 规则

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

## 13. 历史方案的执行顺序（当前不执行）

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

工作树长期可能包含其他实验的 `data` 软链接、日志、TensorBoard events 和未完成快照。每次只显式
`git add` 本步骤的目标文件，不要使用 `git add .`，也不要把这些本机产物误加入 commit。

## 15. 新会话开始时的建议检查清单

1. 先完整阅读本文档和 `my_papers_code/AGENTS.md`。
2. 执行 `git status --short --branch`，不要覆盖用户改动。
3. 检查 latent/HuBERT 两个 full completion marker、index 和 SHA，不以文件数量或进度日志代替完成标记。
4. 检查 train-only normalization 存在且与 train manifest、latent completion SHA 绑定。
5. 检查正式 checkpoint 目录：空目录可从头开始；有 checkpoint 时必须存在并严格匹配 training contract。
6. 先运行单卡最坏真实 batch benchmark，再运行 6-rank DDP canary；不要从普通首个 batch 推断显存上限。
7. 正式训练固定使用 `grad_accumulation_steps=1`、BF16 和 500k exact updates，不能临时改变 world size 后续训。
8. 不要把 codec ceiling WAV 目录当成 latent cache，也不要修改旧 `pretrain.py` 的 mel 语义。
9. 启动后同时核对 worker、连续 update、loss/LR、GPU 显存/利用率和日志错误，不能只看 launcher PID。
10. 每个实现或文档步骤单独 commit/push，并核对本地 HEAD 与 `origin/main` 一致。
