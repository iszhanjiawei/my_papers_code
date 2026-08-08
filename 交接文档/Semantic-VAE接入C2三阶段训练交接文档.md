# Semantic-VAE 接入 C2 与分阶段 mel warm-start 交接文档

> 更新日期：2026-08-08
> 工作区：`/zjw524/projects/alignDiT_idea6`
> Git 仓库：`/zjw524/projects/alignDiT_idea6/my_papers_code`
> 远程：`https://github.com/iszhanjiawei/my_papers_code.git`
> 当前主分支：`main`

## 0. 2026-08-08 当前权威路线

原 6×A40、500k scratch 路线因训练时间过长，已按用户要求安全停止。随后在独立快照：

```text
AlignDiT_mmdit_c2_semantic_vae/
```

从已有 mel 500k checkpoint 的 EMA 严格迁移兼容音频主干，并已完成 S1、S2a、S2b、S2c
四个独立任务，完成 `80D/100 Hz mel -> 64D/40 Hz Semantic-VAE latent` 的纯音频适配。
后续唯一选定的纯音频初始化权重是 S2c `model_70000.pt`，不使用已停止的 scratch 3500
或 S1/S2a/S2b 中间权重启动 C2 多模态训练。

### 0.1 当前状态

| 项目 | 状态 | 说明 |
|---|---|---|
| LibriSpeech immutable inventory | 已完成 | train 281,241，dev 5,551，总计 286,792 |
| Semantic-VAE latent cache | 已完成 | FP32 fixed posterior sample `[T,64]`，40 Hz |
| HuBERT exact-40-Hz cache | 已完成 | FP32 `[T,1024]`，与 latent 逐条等长 |
| train-only normalization | 已完成 | 281,241 条 train、138,504,846 帧、64 通道 mean/std |
| 500k scratch | **已停止** | 进程在 update 3676 停止；最近完整 `model_last.pt` 为 update 3500；后续不使用 |
| S1 | **已完成** | 10k updates；新 64D input/output interface 校准 |
| S2a | **已完成** | 10k updates；解冻 interface、conv-pos、norm-out、blocks 12–17 |
| S2b | **已完成** | 10k updates；解冻范围扩展到 blocks 6–17 |
| S2c | **已完成** | 70k updates；全音频主干与 40 Hz HuBERT projector 已适配 |
| S2c 最终权重 | **已选定、已上传 only-VAE** | `model_70000.pt`，update/EMA step 均为 70000，见第 0.4 节 |
| CelebVDub 64D/40 Hz latent cache | **未完成** | only-VAE 当前只有旧 80D/100 Hz mel 缓存 |
| 25 Hz video -> exact 40 Hz | **未完成** | 现有 AV-HuBERT 为 25 Hz，尚未在 latent dataset 中逐条对齐到 `T_latent` |
| S3a/S3b C2 latent 训练闭环 | **未完成、未启动** | 尚需 latent dataset、40 Hz CTC、S2c EMA 严格迁移、分阶段 trainer/config/launcher 和冒烟测试 |

scratch 保留目录：

```text
/home/zjw524/projects/data/ckpts/
  AlignDiT_SemanticVAE_pretrain_semantic_vae_40hz_LibriSpeech_svae40/
```

停止时 update 3500 之后有 176 个尚未原子落盘的 update，已放弃。不要删除该目录的
`training_contract.json`、`model_last.pt` 或日志；除非用户明确重新启用 scratch，也不要恢复它。

### 0.2 四阶段执行语义

| 阶段 | 本阶段 updates | 可训练参数 | projection loss |
|---|---:|---|---:|
| S1 | 10k | 新 64D input/output interface | 0 |
| S2a | 10k | interface、conv-pos、norm-out、blocks 12–17 | 0 |
| S2b | 10k | interface、conv-pos、norm-out、blocks 6–17 | 0 |
| S2c | 70k | 全音频主干、新 40 Hz HuBERT projector | 前 5k 从 0 ramp 到 0.1 |

S2c 的 scheduler 和 immutable contract 从一开始就按 70k 规划，但 launcher 首次默认只运行 S2c 20k，
此时总适配预算为 `10k + 10k + 10k + 20k = 50k`。通过 dev gate 后，在同一 S2c 目录设置
`RUN_UNTIL_UPDATE=70000` 严格恢复，不能新建一个 50k scheduler 或重置 S2c LR。

四阶段都使用 LibriSpeech train 的同一份固定归一化 latent 与 exact-40-Hz HuBERT：

```text
/home/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32
```

默认 6×A40 配置为物理 GPU 2–7、BF16、seed 666、`7200 frames/GPU`、`max_samples=32`。这等价于
每个 update 暴露 `6 × 7200 / 40 = 1080` 音频秒，与旧 mel 预训练的
`8 × 13500 / 100 = 1080` 秒一致。当前 A40 服务器的 DDP canary 证明必须使用：

```text
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
```

开启 P2P/IB 会在 DDP 参数同步时挂住；launcher 已默认设为 1。换到其他拓扑时只有在独立 NCCL canary
通过后才可显式覆盖为 0。

### 0.3 严格迁移与恢复规则

mel 500k 父权重：

```text
/home/zjw524/datasets/AlignDiT_pretrain_LibriSpeech_500000.pt
size:   2763050034 bytes
sha256: 4a9fc0e526ce47745aee839348406ca99597d32f5ed028bda42a3de3ec900fcd
update: 500000
```

S1 目标是启用 RMS QK-Norm 的 64D 模型：source 277 keys、target 313 keys；加载 263 keys；显式重置
50 keys；只允许 input/output 的 3 个预期 shape mismatch；按参数量复用 88.6682%。重置集合包含：

- 64D input projection 和 output projection；
- 40 Hz stride-1 HuBERT projector；
- mel checkpoint 不存在的 36 个 Q/K RMSNorm 参数。

只能读取父 checkpoint 的 EMA；必须验证父 SHA/size/update、`EMA initted=True`、EMA step、父 stage 与父
contract。跨阶段只迁移相邻阶段 EMA weights，不能继承 online model、optimizer、scheduler、EMA 计数或
local update。同阶段 resume 才恢复全部状态。任何未知 key/mismatch、外来 `pretrained_*.pt`、safetensors、
checkpoint 无 contract 或 contract 内容变化都必须 fail closed，禁止用 `strict=False` 绕过。

### 0.4 已完成的纯音频权重与 only-VAE 现场核验

| 用途 | 文件 |
|---|---|
| 迁移和冻结策略 | `src/aligndit/model/semantic_vae_warmstart.py` |
| 阶段训练器 | `src/aligndit/model/trainer_semantic_vae_warmstart.py` |
| Hydra 入口 | `src/aligndit/script/train/pretrain_semantic_vae_warmstart.py` |
| 公共配置 | `src/aligndit/config/pretrain_semantic_vae_warmstart_base.yaml` |
| 阶段配置 | `src/aligndit/config/pretrain_semantic_vae_warmstart_{s1,s2a,s2b,s2c}.yaml` |
| 6×A40 launcher | `src/aligndit/run/train/pretrain_semantic_vae_warmstart_6xa40.sh` |
| 专项测试 | `src/aligndit/script/misc/test_semantic_vae_warmstart.py` |

已通过：

- Ruff、format、py_compile、launcher `bash -n`；
- 7 项专项 unittest；
- 四份 Hydra stage 配置解析；
- 真实 mel500k 完整迁移；
- 真实单卡 S1 backward：接口梯度/更新非零，冻结主干梯度和变化为 0；
- 真实单卡 S2c backward：diff/proj loss 有限，projector 梯度非零；
- 2-rank BF16 DDP 真实数据 1-update canary：loss 2.03，EMA step/update/contract/scheduler/checkpoint 一致。

canary 使用独立临时目录，核对后已删除，没有混入正式 stage checkpoint。S1、S2a、S2b、S2c
此后已全部完成，不再重新启动这四个阶段。

only-VAE 服务器当前已核对：

```text
GPU: 4 × RTX 4090 24 GB
checkpoint:
  /zjw524/projects/data/ckpts/
    AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt
contract:
  /zjw524/projects/data/ckpts/
    AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/training_contract.json
```

| 校验项 | 值 |
|---|---|
| checkpoint size | 约 2.6 GB |
| checkpoint SHA256 | `02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4` |
| contract SHA256 | `3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d` |
| checkpoint schema | 1 |
| warm-start stage | `s2c` |
| update | 70000 |
| EMA initialized | `True` |
| EMA step | 70000 |
| model state keys | 313 |

checkpoint 内记录的 contract SHA256 与上传的 `training_contract.json` 一致。后续 S3 必须只读取
`ema_model_state_dict` 中的纯音频路径作 weights-only initialization，不继承 S2c optimizer、scheduler 或 update。

### 0.5 only-VAE 启动 S3 前的实际阻断项

2026-08-08 现场检查结果：

- Semantic-VAE 1000k encoder/decoder 权重存在：
  `/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth`；
- CelebVDub 原始音频、25 Hz AV-HuBERT 视频特征、Arrow 和字符词表存在；
- CelebVDub 尚未生成可训练的 fixed posterior sample `[T,64]` 40 Hz latent cache/manifest/complete marker；
- only-VAE 当前缺 LibriSpeech train-only `train_normalization.json`。只有它的 SHA256
  `65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6` 记录在 S2c contract 中，
  mean/std 数值本身不在 checkpoint 内；
- 当前 `finetune.py` / `CustomDataset_mel_video` / `Trainer_VT` 仍是 80D/100 Hz mel + HiFi-GAN 链路；
- S3a/S3b 的严格 EMA 迁移、冻结策略、多 LR optimizer groups、exact update 契约和 4×4090 launcher
  尚未实现。

因此正式训练不得立即用旧 C2 launcher 启动。必须先按“迁移 normalization -> 生成 CelebVDub
latent -> 实现 40 Hz dataset/model/trainer -> CTC 可行性预检 -> 单卡与 4-rank canary -> S3a -> S3b”
顺序执行。

## 1. 给新会话的一句话摘要

当前工作是把 AlignDiT 的 **C2 结构（12 层 MM-DiT + 文本 Cross-Attention，后 6 层纯音频 DiT）**
从 `80 维、100 Hz mel + HiFi-GAN` 改造为 `64 维、40 Hz Semantic-VAE latent + Semantic-VAE decoder`。
500k scratch 已停止；mel500k EMA 到 40 Hz latent 的 S1–S2c 分阶段适配已全部训练完成，
并固定选用 S2c `model_70000.pt`。当前任务从 CelebVDub 数据与 S3 C2 训练闭环开始，
详细状态以第 0 节为准。

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

## 7. 当前方案：mel 500k warm-start 与后续 C2 接入

S1–S2c 已完成实现、canary 和全部正式训练；最终使用 S2c 70k EMA。S3a/S3b 仍是后续设计，
不能用当前 80D/100 Hz CelebVDub 入口直接运行。

论文上分三阶段，实际执行建议分为 6 个独立任务：

| 论文阶段 | 实际任务 | 数据 | updates | 实现状态 |
|---|---|---|---:|---|
| 阶段一 | S1：新接口校准 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2a：解冻后 6 层 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2b：解冻后 12 层 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2c：全音频主干适配 | LibriSpeech latent | 70k | **已完成，选用 `model_70000.pt`** |
| 阶段三 | S3a：训练新多模态模块 | CelebVDub latent/text/video | 5k | 尚未实现 |
| 阶段三 | S3b：完整 C2 微调 | CelebVDub latent/text/video | 195k | 尚未实现 |

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
离线缓存的 exact-40-Hz HuBERT（由 native 50 Hz 按每条 T40 生成）
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

## 9. warm-start 的 Batch 与 GPU 配置

为保持每个 update 看到的音频秒数一致，不能直接沿用 mel 的 frame threshold。

### 9.1 LibriSpeech 阶段

原配置：

```text
13500 mel frames/GPU @100 Hz = 135 sec/GPU
```

单卡等音频秒数是 `5400 frames/GPU`；当前只有 6 张 A40，为保持旧 8 卡实验的全局 1080 秒，正式配置为：

```text
7200 latent frames/GPU @40 Hz = 180 sec/GPU
6 GPU * 180 sec/GPU = 1080 sec/update
max_samples=32
grad_accumulation_steps=1
```

其他卡数只有在新建独立实验 contract 时才按下表换算；不能在已有阶段目录中改变 world size：

| GPU 数 | frames/GPU | grad accumulation |
|---:|---:|---:|
| 8 | 5400 | 1 |
| 6 | 7200 | 1 |
| 4 | 10800 | 1 |
| 1 | 43200 | 1（需先做显存 canary） |

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

## 10. 分阶段 Checkpoint、EMA 与 resume 规则

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

S1–S2c 当前实现使用独立目录：

```text
AlignDiT_SemanticVAE_mel_warmstart_s1_40hz_LibriSpeech/
AlignDiT_SemanticVAE_mel_warmstart_s2a_40hz_LibriSpeech/
AlignDiT_SemanticVAE_mel_warmstart_s2b_40hz_LibriSpeech/
AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/
```

S3a/S3b 的目录只能在对应代码和数据闭环完成后创建。

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

## 13. 后续执行顺序

### 步骤 1：正式启动并验收 S1（已完成）

- 已验证 6 rank、immutable contract、连续有限 loss 和 checkpoint；
- 已精确完成 S1 10k，并以其 EMA 作为 S2a weights-only 初始化。

### 步骤 2：依次执行 S2a 与 S2b（已完成）

- 每段使用独立 config 和 checkpoint 目录；
- 父 checkpoint 必须是相邻阶段 update 10000 的 EMA；
- 检查每阶段 migration report、冻结参数集合和 LR groups；
- 每段精确完成 10k，不把 optimizer/scheduler/EMA step 带入下一阶段。

### 步骤 3：执行 S2c 与累计 50k 门禁（已完成）

- S2c contract/scheduler 固定 70k；
- 首次只运行本阶段 20k，检查 flow loss、HuBERT projection loss、WER/SPKSIM 和梯度量级；
- 通过 gate 后设置 `RUN_UNTIL_UPDATE=70000` 恢复同一 S2c；
- 已选定唯一 S2c EMA：`model_70000.pt`，供后续 C2 使用。

### 步骤 4：补齐 CelebVDub Semantic-VAE 数据闭环（当前进行中）

- 为 train/dev/test 生成同模式 fixed posterior `[T,64]` latent；
- 继续使用 LibriSpeech train mean/std，不重新统计 CelebVDub 坐标系；
- manifest 保存原始采样点数，decoder 后精确裁剪；
- 以每条 `T_latent` 为目标把 25 Hz 视频特征精确插值到 40 Hz；
- 用真实 tokenizer 处理约 105 条不满足 CTC 最短路径的样本。

### 步骤 5：实现并训练 S3 C2

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
3. only-VAE 必须先获取 SHA256 为 `65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6`
   的 LibriSpeech train-only normalization；不得在 CelebVDub 上重算新坐标系。
4. 核对 S2c `model_70000.pt` 的 SHA、stage、update、EMA step 和 contract，并只做 EMA weights-only S3 初始化。
5. 检查 CelebVDub latent full completion marker、index 和 manifest SHA，不以文件数量或进度日志代替完成标记。
6. 用真实 tokenizer 做 CTC 最短路径预检，明确过滤名单，不依赖 `zero_infinity=True` 静默吞掉错标样本。
7. 不要把 codec ceiling WAV 目录当成 latent cache，也不要修改旧 `pretrain.py` 的 mel 语义。
8. S3 先做单卡 forward/backward 和 4-rank 1-update canary，再启动正式 S3a；同时核对 worker、连续 update、loss/LR、GPU 显存/利用率和 checkpoint。
9. 每个后续实现或文档步骤单独 commit/push，并核对本地 HEAD 与 `origin/main` 一致。
