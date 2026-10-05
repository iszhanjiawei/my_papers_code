# Direct-C2 + speaker + framewise visual modulation

本实验独立复制自 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding`，只增加逐帧视觉条件调制，原项目不修改。`BASELINE_SNAPSHOT.json` 记录复制时的源码提交和文件哈希；训练产物、数据和缓存不复制。

## 实验语义

在前 12 个 MM-DiT block 的音频 AdaLN 中增加视觉相关的 scale、shift 和 residual gate 增量：

```text
aligned 40 Hz AV-HuBERT cache (1024 D)
  -> non-affine LayerNorm -> shared Linear(1024, 128)
  -> add per-block projected flow-time embedding
  -> SiLU -> per-block zero-initialized Linear(128, 6 * 768)
  -> add to the six inherited audio AdaLN parameters
```

新增 8,445,568 个参数，当前完整配置共 330,294,786 个参数。条件取自原始对齐视频缓存，不经过正在更新的 MM 视频分支。AV-HuBERT 缓存自身仍包含上下文，因此不声称该条件只包含当前单帧信息。

- AA、AV、VA、VV 均沿用基线的全局注意力及共享 softmax；没有 adaptive band、局部窗口或关闭 VA。
- 保留原文本 cross-attention、后 6 层 CAM++ speaker 调制、Semantic-VAE、flow matching 和 CTC。
- **不加入 WavLM REPA、InfoNCE、时间差分损失或其他辅助损失。**
- 新调制只施加于生成区域内有效的音视频位置；参考前缀、padding、互补掩码位置及 CFG 的视频丢弃分支得到零增量。所有新模块在丢弃视频时仍进入 autograd 图，避免 DDP 未使用参数问题。
- 仅新增输出投影零初始化，没有额外零初始化乘法门控。原权重与随机初始化次序保留，零增量时前向行为与基线一致。
- 音视频使用已有 40 Hz 公共索引，不做新的时间插值。推理时视频短于音频，只对新增调制条件右侧补零并屏蔽；不改变原视频 attention 序列。
- 模型支持 `visual_modulation_mode=pooled` 作为未来同参数量对照；本次训练入口固定为 `framewise`，没有自动启动其他实验。

此实现借鉴 FLOAT 的逐帧条件调制思路，在原有 joint-attention 上增加分支，并非复现其整套 motion 生成网络。

## 训练

配置：`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_framewise_visual_mod.yaml`。

| 项目 | 设置 |
|---|---|
| 初始化 | 与基线相同的 S2c 70k EMA；新优化器和 update 计数 |
| 数据 | CelebV-Dub 79,613 条；原 64D/40 Hz latent、视频和 CAM++ 缓存 |
| GPU | 4 张 RTX 4090，bf16 |
| 每卡 batch | 3,600 帧，最多 32 条，梯度累积 1 |
| 学习率 | 5e-5，原 20k warmup 和 200 epoch 学习率日程 |
| 停止位置 | 200,000 optimizer updates |
| CTC | 10k 前为 0，10k–30k 线性升至 0.03 |
| Seed / EMA | 666 / beta=0.999 |
| 保存 | 每 50k 独立权重，每 5k 更新 last；保留全部独立权重 |

在本实验目录执行：

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_framewise_visual_mod_4x4090.sh \
  > logs/train_200k.log 2>&1 &
```

脚本使用环境内的 Python、`PYTHONPATH=src` 和独立 DDP 端口 `29661`（可用 `TRAIN_PORT` 设置）。启动前检查四卡空闲；同目录已有匹配 checkpoint 时使用原续训逻辑。不要同时启动两份同目录训练。

checkpoint 根目录（前面可加 `ROOT_PREFIX`）：

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_framewise_visual_mod_ctc003_warmup10k30k_40hz_CelebVDub_char
```

`speaker_training_contract.json` 记录实际配置、初始化、注意力拓扑、新分支语义和 TensorBoard 路径。迁移严格校验父权重哈希及 303 个继承张量，明确允许 50 个新增视觉参数；不会静默忽略不匹配张量。

## TensorBoard

训练开始时必须同时启动并核验损失曲线：

```bash
setsid bash scripts/run_tensorboard.sh > logs/tensorboard.log 2>&1 &
```

默认端口 `6006`，可通过 `TENSORBOARD_PORT` 设置。默认读取本实验 `runs/`，实际单个 run 为：

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_framewise_visual_mod_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

保留总损失、flow、CTC 及权重、学习率、梯度范数；CTC 未启用时原始 `ctc_loss` 尚不存在，`ctc_weighted_loss` 为零。首步和每 100 步记录 block 0 视觉输出投影的权重和梯度范数。查看 IDE 底部“端口”页签中的当次转发地址；服务器的 `127.0.0.1` 地址不等于客户端转发地址。

## 验证

```bash
source env.sh
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
PYTHONPATH=src "$python_bin" -m unittest discover -s scripts -p test_framewise_visual_modulation.py -v
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$python_bin" -u \
  src/aligndit/script/misc/smoke_test_framewise_visual_mod_real_parent.py
```

CPU 测试覆盖真实基线源码回归、零初始化等价性、视觉/时间依赖、framewise/pooled 区别、CFG、prompt/padding、短视频尾部、两步梯度、激活 checkpoint 和 flow+CTC 损失组合。真实父权重测试使用实际数据执行 bf16 前向/反向，检查迁移张量及新分支有效梯度，不更新模型或写训练 checkpoint。

原 speaker 推理/评测脚本在本副本中的默认配置和 checkpoint 已指向新实验。评估沿用既有协议，增加分支或通过运行检查不代表 AVSync 已提升。
