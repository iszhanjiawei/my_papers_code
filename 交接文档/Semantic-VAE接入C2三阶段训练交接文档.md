# Semantic-VAE 接入 C2 与 mel warm-start 交接文档

> 更新日期：2026-08-09
> 工作区：`/zjw524/projects/alignDiT_idea6`
> Git 仓库：`/zjw524/projects/alignDiT_idea6/my_papers_code`
> 远程：`https://github.com/iszhanjiawei/my_papers_code.git`
> 当前主分支：`main`

## 0. 2026-08-09 当前权威路线

原 6×A40、500k scratch 路线因训练时间过长，已按用户要求安全停止。随后在独立快照：

```text
AlignDiT_mmdit_c2_semantic_vae/
```

从已有 mel 500k checkpoint 的 EMA 严格迁移兼容音频主干，并已完成 S1、S2a、S2b、S2c
四个独立任务，完成 `80D/100 Hz mel -> 64D/40 Hz Semantic-VAE latent` 的纯音频适配。
后续唯一选定的纯音频初始化权重是 S2c `model_70000.pt`，不使用已停止的 scratch 3500
或 S1/S2a/S2b 中间权重启动 C2 多模态训练。

only-VAE 服务器上的 CelebVDub manifest、79,826 个 64D/40 Hz latent、79,826 个
25 Hz -> 40 Hz 视频 cache 均已完成。S2c contract 绑定的 LibriSpeech train-only
`train_normalization.json` 已按 SHA、schema、计数和 64 通道统计严格核验。旧 S3a→S3b 在恢复后虽然
继续推进到累计约 116k，但审计证明它在约 42.8k 后已经因梯度裁剪溢出而停止有效学习，因此已由本会话
主动停止并废弃，最新完整累计 115k checkpoint 只保留作故障审计。当前权威路线是从干净 S2c 70k EMA
直接启动一个连续 200k 的单阶段 `s3`；2026-08-09 23:20 已从严格验证的 update 1 checkpoint 恢复到
20k 门禁。下文第 0.9 节关于“继续旧 S3b”的内容仅是历史记录，凡与本节冲突均以本节为准。

### 0.1 当前状态

| 项目 | 状态 | 说明 |
|---|---|---|
| LibriSpeech immutable inventory | 已完成 | train 281,241，dev 5,551，总计 286,792 |
| Semantic-VAE latent cache | 已完成 | FP32 fixed posterior sample `[T,64]`，40 Hz |
| HuBERT exact-40-Hz cache | 已完成 | FP32 `[T,1024]`，与 latent 逐条等长 |
| train-only normalization 统计 | **已传输并严格核验** | 281,241 条 train、138,504,846 帧、64 通道 mean/std；SHA256 `65b8ab...d02c3f6` |
| 500k scratch | **已停止** | 进程在 update 3676 停止；最近完整 `model_last.pt` 为 update 3500；后续不使用 |
| S1 | **已完成** | 10k updates；新 64D input/output interface 校准 |
| S2a | **已完成** | 10k updates；解冻 interface、conv-pos、norm-out、blocks 12–17 |
| S2b | **已完成** | 10k updates；解冻范围扩展到 blocks 6–17 |
| S2c | **已完成** | 70k updates；全音频主干与 40 Hz HuBERT projector 已适配 |
| S2c 最终权重 | **已选定、已上传 only-VAE** | `model_70000.pt`，update/EMA step 均为 70000，见第 0.4 节 |
| CelebVDub manifest | **已完成** | inventory 79,826；train 79,613；test 213；CTC-valid train 79,508 |
| CelebVDub 64D/40 Hz latent cache | **已完成** | 79,826 个 FP32 `[T,64]` 文件；全量音频/latent 只读校验已通过 |
| 25 Hz video -> exact 40 Hz cache | **已完成** | 79,826 个 FP32 `[T40,1024]` 文件；逐条目标长度等于 latent 长度 |
| 视频 cache 全量只读校验 | **已通过** | 4-rank validate-only 验证 79,826/79,826，退出码 0，无 error/traceback |
| 模型/数据严格对齐 | **已完成** | commit `73f9256`；40 Hz mask、padding、CTC 和 artifact 约束均已收紧 |
| 真实 CelebVDub dataset gate | **已通过** | 79,508 条 CTC-valid train 全量检查；4,376 个动态 batch 无遗漏/重复/超限 |
| 单卡最坏 batch gate | **已通过** | BF16 forward/backward、冻结权重不漂移、关键新路径梯度及一次临时 optimizer/EMA step 均通过 |
| 旧 4 卡 1-update canary | **已通过** | 证明原训练数据/DDP/checkpoint 闭环可运行，但不能证明旧长期策略稳定 |
| 旧 S3a | **仅保留审计，不再作父权重** | diff loss 当时正常，但文本最终 RMS 已从约 1.31 漂到 4.43 |
| 旧 S3b | **已停止并废弃** | 最新完整累计 115k；50k 后 optimizer momentum 几乎全为最小次正规数，禁止续训/评测 |
| 新单阶段 S3 update 1 gate | **已通过并严格验证** | diff 1.343、grad norm 0.388、raw/post text RMS 1.301/1.000；EMA/update 均为 1 |
| 新单阶段 S3 20k gate | **正式训练中** | 从同一 update 1 的 model/optimizer/scheduler/EMA 严格恢复；20k 通过前不直接跑 200k |
| LibriSpeech train normalization | **阻塞已解除** | only-VAE 原文件 SHA256、schema、count、frame_count 和 64D mean/std 全部匹配 |

### 0.1.1 旧 S3b 的确定根因与新单阶段修复

这次高 loss 不是 TensorBoard 显示错误，也不是 resume、scheduler 卡数缩放、latent normalization、视频插值
或 CTC 本身造成。固定同一真实样本得到：S2c 初始化时四层文本 ConvNeXt RMS 约
`1.27→1.28→1.30→1.31`，旧 S3a 5k 已变为 `1.37→1.72→2.69→4.43`，旧 S3b 50k 进一步变为
`2.00→10.51→76.21→219.59`。第一层文本 CA 输出 RMS 达到约 122，block 0 音频状态达到约 356。

旧 50k checkpoint 在正式 batch 上所有梯度元素仍 finite，但稳定 FP64 global norm 约 `3e23`；PyTorch
FP32 clipping 返回 `inf` 并把全部梯度乘成 0。50k/100k 的 696 个 Adam state 中，693 个 `exp_avg`
只剩 float32 最小次正规数 `5.605e-45`，其余 3 个为 0。模型表面仍增加 update，实际只剩 AdamW
weight decay。diff-only 同样复现，因此 CTC 不是主触发源。

三项修复已经分别 commit/push：

| commit | 修复 |
|---|---|
| `e345c32` | 所有文本 CA 前统一无参数、padding-safe LayerNorm；不增加 checkpoint state key |
| `66791d8` | 捕获 pre-clip grad norm，DDP 同步 fail-fast，并写入 CTC/grad/text TensorBoard 指标 |
| `5ad4095` | 从 S2c 70k EMA 启动连续 200k 单阶段策略，取消 S3a/S3b optimizer/EMA 重置 |

新策略固定：20k warmup；CTC 在第 1 个 update 使用 `0.1/20000`、第 20k 个 update 达到 0.1；
`text_conditioner=5e-6`、`multimodal_core=1e-5`、`multimodal_gates=1e-6`、`ctc_heads=1e-5`、
`interface=5e-6`、`audio_blocks_0_5=2e-6`、`audio_backbone_rest=5e-6`、
`shared_conditioning=1e-6`。raw text RMS > 3、global pre-clip norm > 100 或非有限值会在 optimizer step 前停止。

当前新实验现场（进程号会过期，必须实时复查）：

```text
checkpoint:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3_single_stage_v2_40hz_CelebVDub_char/
update-1 contract SHA256:
  f82ea4dbd61e27b32518061906181f87beec0d3a00a8b1ed24d9d21327b84c4c
20k launcher PID/SID at launch:
  443478 / 443478
20k log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    train_semantic_vae_c2_s3_single_stage_resume1_to20k_4x4090_20260809.log
TensorBoard run:
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3_single_stage_v2_s3_CelebVDub_svae40_ctc_valid
TensorBoard service at launch:
  PID 419778, 127.0.0.1:34951 (Devin 端口面板点击同一转发地址)
```

20k 门禁必须同时满足：diff loss 未发生旧实验那样的阶跃；raw text RMS 始终 < 3；post RMS 约 1；
global grad norm 全程 finite 且 < 100；Adam momentum 正常、no-decay 参数确实更新；完整 20k checkpoint
通过 validator。只有全部通过，才设置 `S3_RUN_UNTIL_STAGE_UPDATE=200000` 从同一目录继续。

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
| checkpoint size | `2,762,690,094` bytes |
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

### 0.5 only-VAE 数据闭环与可审计 SHA/count

CelebVDub 当前权威 cache 根目录：

```text
/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32
```

固定 posterior sample 使用 base seed 666。inventory 共 79,826 条，train/test 分别为 79,613/213；
总原始音频采样点数 5,256,590,059，总 latent 帧数 13,179,773。CTC 40 Hz 预检后，train 中
79,508 条可训练、105 条进入明确排除清单；test 中另有 1 条不可行，但 test 不参与训练。

| manifest / source artifact | 条数或大小 | SHA256 |
|---|---:|---|
| `inventory.jsonl` | 79,826 条；68,395,999 bytes | `a6478cce785748cbcefd87af54eafa9f654d735afa1c41b8f846e041cbc1286d` |
| `train.jsonl` | 79,613 条；68,218,316 bytes | `0d16d5c8f00eb25ee51c7de604299a37cace1bc0e65b7127a45420c433b4d395` |
| `test.jsonl` | 213 条；177,683 bytes | `f724fa12947365e7bb08a5665c23aaf649a73855647e0667135352fa473e36b8` |
| `train_ctc40_valid.jsonl` | 79,508 条；68,131,856 bytes | `cbeb9f02d805d403ce6b24a6a56f170ad278d3187cd65e289ef1e80cf665d58c` |
| `ctc40_excluded.jsonl` | 105 条；86,460 bytes | `d4071db62d412ebeba144fe5e8a665530dfbc57ac52e19b4ad42212841ddcb0a` |
| `ctc40_report.json` | 797 bytes | `bc4887aca72cf8c996c9fd89538aa9f83c66476606087201d8de1d3cc9f3042c` |
| `inventory_meta.json` | 2,816 bytes | `c1aadbcef9dc83cd8f6a2c0d094311cc527b971f8921abbcfee1afd2c4e67bc2` |
| `manifest_spec.json` | 970 bytes | `3343c5ff5b80999b05b049db55b19c6335133307c6c38b4e69611718a9ff7986` |
| 原始 `CelebVDub_char/raw.arrow` | source | `99da14538f85eca3a039282d1cb5126f2a5598dd3c513422fe58b454af9437ef` |
| `CelebVDub_char/vocab.txt` | vocab size 159 | `225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d` |

字符目标实际范围为 0–158；CTC class 159 保留未用，blank id 为 160，因此输出类别数为 161。

latent cache 为 79,826 个 FP32 `[T,64]` 文件，feature id 是
`semantic_vae_posterior_sample_v1`，总 `.npy` 大小 3,384,239,616 bytes：

| latent artifact | 条数或大小 | SHA256 |
|---|---:|---|
| `state/latents/spec.json` | 3,488 bytes | `ce413cba5f75cd3fb63a2bf74a8117277a04992af4716b1d2a319c971ff379d4` |
| `state/latents/index.jsonl` | 79,826 条；22,641,889 bytes | `1a8dbdb5d5c8f08482f2091120ee4a2a0c54a822a99dda9fab6b23f490594475` |
| `state/latents/complete.json` | 682 bytes | `dc06eb31449c0e12558a69874b0d91ef83a3ca543418533fe60cfa2b0d6e8767` |

使用的 Semantic-VAE 1000k 权重为：

```text
/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth
sha256: 7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516
```

视频 cache 为 79,826 个 FP32 `[T40,1024]` 文件，feature id 是
`avhubert_video_25hz_to_40hz_linear_align_corners_false_v1`；原生 25 Hz 总帧数 8,238,378，
插值后总帧数 13,179,773，与 latent 总帧数完全相等；总 `.npy` 大小 53,994,567,936 bytes：

| video artifact | 条数或大小 | SHA256 |
|---|---:|---|
| `state/video_40hz/spec.json` | 1,829 bytes | `0622c768216dfd01e18720c590dae6bfa128402b5e9835eb558a90293c21cb5c` |
| `state/video_40hz/index.jsonl` | 79,826 条；35,375,799 bytes | `6a58072e13848378e83c661f8843655a3368aab24cffd246545356c454c07325` |
| `state/video_40hz/complete.json` | 745 bytes | `47d868473c0a133b2bcf9414d2511df285d40fd3636fe116f092fe5eaf3d60fc` |

音频/latent 的 4-rank 全量 read-only validation 已成功读完 79,826 条，各 rank 分别处理
19,957/19,957/19,956/19,956 条，日志为：

```text
AlignDiT_mmdit_c2_semantic_vae/logs/validate_celebvdub_svae_latents_20260808.log
```

视频的 4-rank 全量 validate-only 已成功读完 79,826 条，各 rank 分别处理
19,957/19,957/19,956/19,956 条，进程退出码为 0，无 error/traceback，日志为：

```text
AlignDiT_mmdit_c2_semantic_vae/logs/validate_celebvdub_video_40hz_20260808.log
```

完成标记、spec、完整 index、selected manifest SHA 和文件 stat/size 必须同时匹配；不能仅凭 `.npy`
数量或生成日志认定 cache 合格。

### 0.6 已完成的模型、数据和 S3 trainer 实现

正式启动前已核对仓库 `HEAD == origin/main == c6bef85`；后续文档提交会自然推进 HEAD，因此应以
`git rev-parse HEAD` 和 `git ls-remote origin refs/heads/main` 做实时核对。关键提交为：

| commit | 内容 | 当前结论 |
|---|---|---|
| `3c8ffc8` | CelebVDub 40 Hz cache/manifest 与审计链路 | 数据生成已完成，音频和视频全量只读校验均通过 |
| `73f9256` | 模型/数据严格 Semantic-VAE C2 对齐 | 已完成 exact-40-Hz mask、padding-safe conv、CTC 与 artifact fail-closed 修复 |
| `8ae61ee` | 旧 S3a/S3b staged trainer、config 和 chain launcher | 历史实现；长期训练已确定失稳，禁止继续使用 |
| `ed80a0b` | 修复 checkpoint semantic digest 对 0 维 optimizer/EMA tensor 的字节哈希 | 同一 4 卡 canary checkpoint 已重新严格验证通过 |
| `c6bef85` | 用原子目录锁替换 DPC 文件系统上无法可靠释放的 `flock` | 旧 chain 当时使用；现仅保留历史实现 |
| `0df98e2` | 记录 normalization、全部门禁和旧 S3 首次启动现场 | 已被本文第 0.9 节的事故历史补充 |
| `e345c32` | 在所有文本 CA 前增加无参数、padding-safe LayerNorm | 当前单阶段 S3 必需的结构修复 |
| `66791d8` | 梯度范数、文本 RMS、CTC ramp 监控和 DDP fail-fast | 当前单阶段 S3 必需的训练安全修复 |
| `5ad4095` | 从 S2c 70k EMA 直接连续训练 200k 的单阶段 S3 | 当前唯一正式 S3 路线 |

`73f9256` 固定了音频和视频相同的 padded length/mask；输入卷积和 CTC projector 均保证 padding-safe；
两个 CTC head 位于 `[6,12]`，保持 40 Hz、ratio `[1,1]`，使用 FP32 CTC、精确可行性检查和
`zero_infinity=False`。S2c 313-key source 到 C2 703-key target 的迁移为：加载 303 个兼容 key，
仅忽略 10 个已知旧 projector key，新建 400 个多模态/CTC key；未知 key、shape 或 artifact 绑定变化
均 fail closed。严格对齐修复后的 10 项 regression tests 已通过；包含 Conformer 和随机 gate 的整模型
padding invariant 误差约为 output `7.5e-8`、CTC `3e-7`，padding 输出为 0。

`8ae61ee` 的以下阶段定义仅用于解释旧事故，**不是当前启动配置**：

- S3a：5k updates，只训练新多模态路径，LR `5e-5`，warmup 500；
- S3b：195k updates，全量 702 个参数解冻；新参数 `5e-5`、interface `2e-5`、audio blocks 0–5
  `5e-6`、其余已加载音频路径 `1e-5`，warmup 5000；
- 阶段间重建 optimizer/scheduler/EMA；chain 使用锁、空闲 GPU 拒绝、S3a 完成验证和严格父
  checkpoint SHA/contract 后才允许启动 S3b；
- 4×4090 默认 BF16、每卡 3,600 个 40 Hz frames、`max_samples=32`，即全局每 update 360 秒音频。

这套旧策略的问题不是“分阶段”三个字本身，而是 S3a 已把无输出归一化的文本塔推向尺度漂移，S3b 又在
5k 边界重置 Adam/EMA、以最高 `5e-5` 学习率继续训练文本/CA，并一次性解冻音频主干。当前 `5ad4095`
已经用一个连续 optimizer/scheduler/EMA 时间线替代它；准确参数见第 0.1.1 节。

### 0.7 warm-start 初始漂移风险

迁移在 schema、key 和 shape 层面是严格的，但不是 function-preserving。使用真实 S2c
`model_70000.pt` EMA、303 个共享 key、seed 666、完整 768×18/12MM + Conformer，并在 CPU 上用
`T=24` 做审计时：

- 纯音频 source 与接入随机视频后的 S3 输出 relative RMS drift 为 15.73% 和 18.03%，cosine 为
  0.9886 和 0.9855；
- 交换两组随机视频时，S3 输出 relative RMS 改变 5.64%；
- 真实视频与 null/drop video 的输出 relative RMS 改变 10.04%。

原因是已加载、非零的 audio attention gate 立即作用于 joint softmax；随机初始化的视频 embedding/K/V
会从第一个 update 开始影响音频。`v_attn_norm` 的 zero gate 只阻断 attention residual 回写视频，不能阻断
video K/V -> audio。因此在旧策略中，S3a 只能称为“新多模态路径适配”，不能称为“零漂移接口校准”。
这一初始漂移审计仍适用于从 S2c 初始化的新单阶段 S3，必须由当前 20k 门禁持续监控。

### 0.8 数据阻塞与门禁结果，以及旧 staged 训练现场

#### 0.8.1 当前仍然有效的数据与计算图门禁

原来唯一缺失的文件已经传入并核验：

```text
/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/
  state/latents/train_normalization.json
size:   3634 bytes
sha256: 65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6
```

它与源服务器原文件逐字节一致；`channel_count=64`、`count=281241`、`frame_count=138504846`，mean/std
各 64 项且全部 finite，所有 std 均大于 0。因此使用的是 S2c 所在的 LibriSpeech train-only 坐标系，
没有在 CelebVDub 上重算或从 checkpoint 反推。

真实数据 gate 已全量通过：dataset 长度 79,508，帧长 18–1,199，总有效帧 13,146,829；CTC target
范围 1–662、最小需求 1–676，所有样本均可行。正式 `3600 frames/GPU, max_samples=32` 分批器生成
4,376 个 batch，79,508 条恰好覆盖一次，batch 总帧数 661–3,600，无重复、遗漏或超限；混合长度
collate 的音频/视频 mask 完全一致，padding 全零。

单卡 gate 使用真实最坏 attention batch：`B=3`、长度 1,188–1,190、总 3,566 帧。严格从 S2c EMA
加载 303/703 个 key，新建 400 个 key；BF16 得到 `diff=1.4271`、双层 CTC 平均 `8.9692`、
`total=2.3240`，loss 和梯度均 finite。关键视频 embedding、video K/V、文本 CA gate 和两个 CTC head
都有非零梯度；一次不落盘 AdamW step 后，302 个冻结且加载的 tensor SHA 完全不变，新参数发生预期更新。
峰值显存约 allocated 9.269 GiB、reserved 9.566 GiB。末个 MM block 有 6 个仅产生无人消费视频输出的
residual 参数无梯度，这是精确白名单；同层 video K/V -> audio 条件路径梯度非零，不能误报成视频条件失效。

4 卡 BF16 canary 在独立目录完成 1 个真实 update：

```text
checkpoint dir:
  /zjw524/projects/data/ckpts/_canary/
    AlignDiT_MMDiT_c2_svae_s3a_u1_20260808_gate1
log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    canary_s3a_u1_4x4090_20260808.log
model_last.pt:
  size:   4346037616 bytes
  sha256: 2adb4e8c17b243d779cabd07a57896ebeb64cc94024b9767dce95d9a1d0f8d4a
contract sha256:
  849cb3584529a85bcbb688231f2865e2952b4005f26d2e93696865bbcd42751a
```

canary 的 `stage_update=1`、`cumulative_update=1`、EMA step 1、703 个模型 key、optimizer 和 scheduler
resume state 均由严格 validator 验证。首次 validator 只因 AdamW step 是 0 维 tensor 而触发 PyTorch
字节 view 限制；`ed80a0b` 仅修复 semantic digest 的标量兼容性，随后对同一 checkpoint 重验通过，
没有重跑或替换训练结果。canary 目录与正式目录完全隔离并保留作审计证据。

#### 0.8.2 历史上的 S3a/S3b 正式启动现场（禁止恢复）

以下只记录旧 staged 实验的基础设施和启动现场。其 checkpoint、PID、日志和 chain 均不是当前路线。

`/zjw524` 是 DPC 分布式文件系统，canary 退出后原 `flock` 出现无法可靠释放的残留锁；`c6bef85`
改用带 `owner.txt` 的原子目录锁，仍然跨主机 fail-closed，且正常/信号退出时精确清理。当时旧正式串联
任务为：

```text
启动时间: 2026-08-08T20:48:57+08:00
launcher PID/SID: 64078 / 64078（启动现场；重连后须实时核对）
outer log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    train_semantic_vae_c2_chain_4x4090_20260808.log
S3a log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    train_semantic_vae_c2_s3a_4x4090_20260808_204857.log
S3a checkpoint dir:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3a_40hz_CelebVDub_char
S3b checkpoint dir:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3b_40hz_CelebVDub_char
```

任务使用 `setsid` 脱离终端，启动后 PPID 已变为 1、SID 独立、TTY 为 `?`。旧 S3a contract SHA256
为 `ce26935d98ab4bafe8ea583cde151754af72e34e4acb66a5096397744d4a1b50`，父 S2c SHA/contract 和
303/703 迁移结果与 canary 一致。启动后的独立 122 秒监控窗口从 update 142 推进到 369，稳定速度约
`1.86 update/s`；最近 100 步 total/diff/CTC 均值分别为 1.6806/1.4040/2.7676，全部 finite。显存采样
峰值为 GPU0–3：13,062/11,436/11,618/11,922 MiB（每卡 24,564 MiB），未见 traceback、NaN、OOM、
NCCL 或 ChildFailed。按该启动窗口估算 S3a 约在 2026-08-08 21:34–21:35 到达 5,000 updates，另需
checkpoint 保存和验证时间；实时 update、速度和 ETA 必须重新读取日志，不能把本段启动快照当成当前进度。
旧 chain 当时会在 S3a 精确 5,000 updates 后保存并严格验证，只有验证通过才自动启动 S3b 195,000
updates；`model_last.pt` 每 5,000 updates 更新。该入口现已废弃，禁止重新启动。

### 0.9 旧 S3a/S3b 事故历史（仅供审计，禁止照此恢复）

本节记录为什么旧训练曾中断、如何在当时从 85k 恢复，以及后来如何确认其优化已经失效。这里出现的
PID、恢复命令、checkpoint 和“继续 S3b”决定都已过期。当前唯一正式路线是第 0.1.1 节的新单阶段 S3。

#### 0.9.1 历史 S3a 曾完成，但权重现已废弃

首个正式 chain 于 2026-08-08 20:48:57 启动。S3a 在 21:34 左右完成 5,000 updates，chain 于
21:36:50 完成严格 validator 后自动进入 S3b。S3a 两份最终文件均为 4,346,037,616 bytes，且当时
byte SHA256 和四类 resumable-state digest 完全一致：

```text
directory:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3a_40hz_CelebVDub_char
model_5000.pt SHA256:
  23a5d6696315722f3189b9bf9f590a6b436cacbcae919f1753a5469136e5a7c7
model_last.pt SHA256:
  23a5d6696315722f3189b9bf9f590a6b436cacbcae919f1753a5469136e5a7c7
contract SHA256:
  ce26935d98ab4bafe8ea583cde151754af72e34e4acb66a5096397744d4a1b50
stage/cumulative/EMA:
  5000 / 5000 / 5000
```

S3b 当时完整加载了 S3a EMA 的 703/703 个 key，ignored/new/shape mismatch 均为空，loaded fraction 为
1.0；这证明迁移机制正确，却不能证明权重优化健康。后续审计发现 S3a 的文本最终 RMS 已漂移至 4.43，
因此当前路线必须绕过 S3a，从干净的 S2c 70k EMA 重新启动。

#### 0.9.2 S3b 首次运行停止的直接原因

S3b 首次运行日志：

```text
AlignDiT_mmdit_c2_semantic_vae/logs/
  train_semantic_vae_c2_s3b_4x4090_20260808_204857.log
```

进程运行到累计 update 89,572 时，于 `2026-08-09 15:38:37 +08:00` 停止。直接证据是 rank 0
PID 81105 收到 `SIGKILL`，exit code 为 `-9`；Torch Elastic 检测到 rank 0 消失后才向其余 rank
发送 SIGTERM。已经排除：

- Python/模型代码 traceback；
- CUDA OOM；
- CPU/cgroup OOM：检查时 cgroup `oom=0`、`oom_kill=0`，系统仍有约 442 GiB available；
- NaN/Inf、FloatingPointError；
- NCCL error/timeout；
- 磁盘不足；
- 服务器重启：当时 uptime 已超过 43 天。

因此这不是训练代码自行停止，而是外部强杀。用户在相近时间切换网络，最合理推断是 Bitahub/远程会话
平台的清理机制被连接变化触发，但没有预先开启 audit/eBPF 时，Linux 事后不能给出 `SIGKILL` 发送者，
所以不能把该推断写成 100% 已证明。`setsid` 只能防普通 SSH 断开产生的 SIGHUP，无法抵抗平台主动
发送的 SIGKILL；以后不能仅凭“使用了 setsid”就断言平台级强杀绝不会发生。

#### 0.9.3 恢复点为什么是 85k，而不是 89,572

S3b `last_per_updates=5000`，所以中断前最近一次原子落盘是：

```text
/zjw524/projects/data/ckpts/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3b_40hz_CelebVDub_char/model_last.pt

resume-time metadata:
  semantic_vae_c2_stage: s3b
  stage_update:          80000
  cumulative_update:     85000
  update:                85000
  EMA step:              80000
  EMA initted:           true
  model keys:            703
  EMA keys:              705
  contract SHA256:       cca7cf315b956cee8a125337be2f3759efc946219413e8b70670c74c0ed8c1f3
```

89,572 只是进程内存中的最新步数，85,001–89,572 没有完整 checkpoint，不能恢复，也不能伪造 update。
恢复后必须重新计算这 4,572 步。

#### 0.9.4 历史上的 85k 恢复任务

用户当时在了解初步 loss 风险后要求保持原计划：不新建 S3b-v2、不修改 LR/解冻策略，直接从最近完整
85k checkpoint 继续。因此 2026-08-09 的那次恢复没有修改任何代码或配置，使用原 S3b 目录做
same-stage strict resume。后续拿到梯度和 optimizer 的确定证据后，该决定已经被停止旧任务、重开新单阶段
S3 的决定取代；下列信息只能用于事故复盘：

```text
恢复启动时间:
  2026-08-09T15:49:00+08:00
恢复 launcher PID/SID（启动现场，必须实时复查）:
  347787 / 347787
outer log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    train_semantic_vae_c2_chain_resume_s3b_from_85000_4x4090_20260809.log
stage log:
  AlignDiT_mmdit_c2_semantic_vae/logs/
    train_semantic_vae_c2_s3b_4x4090_20260809_154901.log
atomic lock:
  /zjw524/projects/data/ckpts/
    .aligndit_semantic_vae_c2_s3_chain_4x4090.lock.d
lock owner:
  host=bitahub-a20751278753968128132306
  pid=347787
  started_at=2026-08-09T15:49:00+08:00
  start_stage=s3b
```

恢复 chain 先重新验证 S3a 的两份 5k 文件和 contract，再由 S3b trainer 验证原 S3b contract，随后完整加载：

- online model；
- AdamW optimizer state/moments；
- scheduler state；
- EMA model、initted 和 step；
- stage/cumulative update。

数据 epoch/batch 位置由 stage update 重建；`deterministic_update_seed=true` 按 stage update 和 rank 重建
flow noise/time/dropout 随机流。恢复后日志已连续出现累计 85,001 之后的更新。语义上这是完整断点续训，
不是只载入模型权重重新微调；但 CUDA/DDP 运算不承诺逐 bit 与未中断轨迹完全相同。

`2026-08-09 17:15:38 +08:00` 状态快照：累计 update 91,303，最新完整 `model_last.pt` 为累计 90k，
四卡显存约 16.1–18.5 GiB，速度约 1.25 update/s，无新的 OOM/NCCL/traceback。该 update、PID、速度
都只是历史快照，新会话必须从实时进程、日志和 checkpoint 元数据重新读取。

#### 0.9.5 loss 风险：机械恢复正确不等于优化健康

S3b 的 parent、offset、contract、scheduler 和恢复机制都已核对正确，但首次运行的 loss 有持续恶化：

| 区间 | diffusion loss 现象 |
|---|---:|
| S3b stage 1–10k | 均值约 1.39，正常 |
| stage 约 14k | 开始快速上升 |
| stage 15–37k | 长期约 15.9 |
| stage 约 38k 以后 | 长期约 21–22 |

512 条真实 latent 抽样中，“模型恒输出 0”的理论 flow MSE 约为 2.10，因此 21–22 不能解释为 latent
天然尺度；CTC 也从早期约 0.8–1.7 恶化到约 4。最可能风险是 S3b 一次性解冻全音频主干后，
multimodal `5e-5`、interface `2e-5`、audio backbone 最高 `1e-5`，并只 warmup 5k，导致长期定向漂移；
`max_grad_norm=1.0` 只能限制单步梯度，不能阻止累计漂移。

这段初步判断后来被更强证据取代：根因不是一般性的“累计漂移”，而是无界文本上下文导致 global norm
达到约 `3e23`，FP32 clipping 溢出后把每步梯度静默清零。旧 S3b 已经停止，任何新会话都不得恢复、评测
或以其权重初始化；只能监控和验收第 0.1.1 节的新单阶段 S3。

### 0.10 已踩过的坑：新会话不要重复

1. **不要只改通道数就把 mel 入口当 Semantic-VAE 入口。** 80D/100 Hz mel 与 64D/40 Hz latent 的
   dataset、input/output、CTC、mask、video 对齐、decoder 和 checkpoint 契约都不同。
2. **不要在 CelebVDub 上重算 normalization。** 唯一坐标系是 S2c 使用的 LibriSpeech train-only
   mean/std，文件 SHA256 必须为 `65b8ab...d02c3f6`。
3. **不要把 25 Hz 视频假装成 40 Hz。** 当前做法是按每条 latent 目标长度，以 linear、
   `align_corners=false` 精确插值，缓存必须由 completion/spec/index/manifest SHA 联合绑定。
4. **CTC 不能继续下采样到 20 Hz。** 两个 CTC head 固定在 `[6,12]`，sampling ratio `[1,1]`，
   保持 40 Hz，并在训练前按重复字符的精确最短路径过滤 105 条不可行 train 样本。
5. **跨阶段与同阶段恢复不能混用。** 当前 S2c -> 单阶段 S3 只取 S2c EMA weights-only，并新建一次
   optimizer/scheduler/EMA；进入 S3 后的任何中断恢复都必须恢复 online/optimizer/scheduler/EMA/update
   全部状态，且 0–200k 期间不得再人为切阶段或重置这些状态。
6. **不要给新实验复用含旧 `model_last.pt` 的目录。** trainer 会自动 same-stage resume；若实验语义、
   结构或 LR 改变，必须使用全新目录和 contract。新单阶段 S3 已使用独立 `_single_stage_v2_` 目录；只有
   该目录内部的中断才允许 same-stage strict resume。
7. **DPC 文件系统上的 `flock` 不可靠。** 已由 `c6bef85` 改成带 owner 的原子目录锁；不要改回
   `flock`，也不要直接删除活跃锁目录来绕过单实例保护。
8. **checkpoint validator 必须支持 0 维 optimizer tensor。** `ed80a0b` 已在字节 reinterpret 前 flatten；
   不要恢复旧的 `tensor.view(torch.uint8)`，否则 AdamW scalar step 会让有效 checkpoint 校验失败。
9. **`setsid` 不是 SIGKILL 防护。** 它能抵抗 SSH SIGHUP，但平台级强杀仍会停止任务。网络切换前后应检查
   远程平台生命周期；中断后必须从日志确认 root cause，不能把所有停止都归咎于代码。
10. **日志 finite 不代表训练健康。** 旧 S3b 在没有 NaN/OOM 时已经静默失效。新 trainer 虽已增加
    raw/post text RMS、pre-clip global/per-group grad norm 和 fail-fast，仍必须同时看 loss 趋势、
    optimizer moments 与真实评估，不能只看进程、GPU 或 `update/s`。
11. **最后一个 MM block 的 6 个纯视频输出参数无梯度是当前计算图的已知事实。** 末层之后视频输出无人
    消费，但同层 video K/V -> audio 梯度非零；不要误判成整个视频条件路径失效。
12. **不要混用 canary、正式目录、旧 S3a/S3b 或其他服务器结果。** 新单阶段实验有独立目录和
    TensorBoard run；评测必须记录 checkpoint SHA、推理目录和样本数。
13. **checkpoint 间隔内的进度不可恢复。** `model_last.pt` 每累计 5k 更新；进程日志中的更高 update 若未
    原子落盘，只能重算，不能手工改 metadata 冒充恢复。
14. **不要提交训练产物。** checkpoint、日志、TensorBoard、Hydra outputs、cache 和两个既有未跟踪
    `data` 软链接都不得混入 Git；只显式暂存本次源码或文档。
15. **旧 S3b 的累计/阶段 offset 只属于历史格式。** 新单阶段 S3 从 0 开始，所以
    `stage_update == cumulative_update == update`；最终 `model_200000.pt` 三者都应为 200k。

## 1. 给新会话的一句话摘要

当前工作是把 AlignDiT 的 **C2 结构（12 层 MM-DiT + 文本 Cross-Attention，后 6 层纯音频 DiT）**
从 `80 维、100 Hz mel + HiFi-GAN` 改造为 `64 维、40 Hz Semantic-VAE latent + Semantic-VAE decoder`。
500k scratch 已停止；mel500k EMA 到 40 Hz latent 的 S1–S2c 分阶段适配已全部训练完成，并固定选用 S2c
`model_70000.pt`。CelebVDub 40 Hz 音频/视频数据、两类全量只读校验与 C2 trainer 均已完成。旧 S3a/S3b
已经因文本塔尺度爆炸和静默零梯度而废弃。当前从 S2c 70k EMA 运行连续 200k 的新单阶段 S3，并先在
20k 自动停止做长期稳定性门禁；详细现场、阈值与代码提交以第 0 节为准。

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

当前 Semantic-VAE C2 工程：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/
  AlignDiT_mmdit_c2_semantic_vae
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

### 5.3 旧 mel 入口不能直接训 latent（历史状态）

2026-08-02 检查的旧入口存在以下 mel 假设：

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

因此当时不能只把 `n_mel_channels: 80` 改成 64 就启动训练。当前独立快照
`AlignDiT_mmdit_c2_semantic_vae` 已通过 `73f9256` 和 `8ae61ee` 完成 64D/40 Hz 的底层
dataset/model/trainer 基础，再由 `e345c32`、`66791d8`、`5ad4095` 完成当前稳定单阶段入口；旧 staged
launcher 已废弃。本小节只用于解释为什么不能回退使用旧 `finetune.py`、`CustomDataset_mel_video`
或旧 mel launcher。

## 6. 历史卡点（现已解决或进入新流程）

本节是旧服务器/旧方案的历史记录。以下三个旧卡点中的数据、normalization 和训练基础设施问题均已解决；
第 0.8 节记录了文件核验、早期门禁和旧 S3 启动历史；当前单阶段现场以第 0.1.1 节和实时日志为准。

### 6.1 本机缺少 LibriSpeech 数据（旧服务器问题，已解决）

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

该问题随后在执行 S1–S2c 的服务器上解决，四阶段纯音频适配已完成。only-VAE 不需要重新训练
LibriSpeech 阶段；与 S2c contract 完全一致的 `train_normalization.json` 也已复制并核验完成。

### 6.2 Semantic-VAE latent 训练缓存尚未生成（历史状态，现已完成）

原计划需要为：

- LibriSpeech train/dev；
- CelebVDub train/dev/test；

提取固定 `[T40,64]` latent，并生成 manifest 和 mean/std。

预计 float32 缓存大小：

- LibriSpeech 960h：约 35 GB；
- CelebVDub 91h：约 3.4 GB。

当前 LibriSpeech latent/HuBERT 缓存和 S2c 已完成；CelebVDub 79,826 个 latent 与 79,826 个 40 Hz
视频 cache 也已完成。CelebVDub 不生成自己的训练 mean/std，必须继续使用 LibriSpeech train stats。

### 6.3 三阶段训练基础设施尚未实现（历史状态，现已实现）

旧方案当时尚缺：

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

正式 S3 所需的 latent dataset/collate、64D/40 Hz CFM、严格迁移、冻结/多 LR、精确 updates、
weights-only init、40 Hz CTC 与 25 -> 40 Hz 视频 cache 由 `73f9256`/`8ae61ee` 奠定；旧 staged chain
已经废弃。文本归一化、训练门禁与当前单阶段入口分别由 `e345c32`、`66791d8`、`5ad4095` 取代。
解码与最终评测仍按第 11 节单独验收，不能把“trainer 已实现”误写成“最终指标已验证”。

## 7. 当前方案：mel 500k warm-start 与后续 C2 接入

S1–S2c 已完成实现、canary 和全部正式训练；最终使用 S2c 70k EMA。不能回退使用 80D/100 Hz
CelebVDub 入口，也不能使用旧 S3a/S3b 权重。纯音频适配仍保留 S1–S2c 的渐进阶段，但 CelebVDub C2
阶段已经改成一个连续任务，因此实际执行是 5 个任务：

| 论文阶段 | 实际任务 | 数据 | updates | 实现状态 |
|---|---|---|---:|---|
| 阶段一 | S1：新接口校准 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2a：解冻后 6 层 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2b：解冻后 12 层 | LibriSpeech latent | 10k | **已完成** |
| 阶段二 | S2c：全音频主干适配 | LibriSpeech latent | 70k | **已完成，选用 `model_70000.pt`** |
| 阶段三 | S3：连续完整 C2 微调 | CelebVDub latent/text/video | 200k | **20k 稳定性门禁训练中** |

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

单阶段 S3（200k）：

- 从 S2c 70k EMA 严格白名单迁移一次，随后全量解冻；
- 整个 0–200k 使用同一个 optimizer、scheduler 和 EMA，warmup 20k；
- 文本 conditioner `5e-6`，视频/MM core `1e-5`，新 gate `1e-6`，CTC heads `1e-5`；
- latent interface `5e-6`，音频 blocks 0–5 `2e-6`，其余音频主干 `5e-6`，共享 time/norm `1e-6`；
- CTC 权重由首步 `5e-6` 线性升至第 20k 步 `0.1`；
- 所有文本 CA 共享经过无参数 LayerNorm 的 context；
- raw text RMS、pre-clip global/per-group grad norm 进入 TensorBoard，超过门限在更新前同步终止；
- 首次运行精确停在 20k，验收通过后从同一完整状态继续到 200k。

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

40 Hz 预检最终结果：

- train 79,613 条中，79,508 条可行、105 条不满足 CTC 最短路径并已写入排除 manifest；
- test 213 条中另有 1 条不可行，但 test 不参与训练；
- CTC 使用 `zero_infinity=False`，不能静默吞掉不可行或错标样本。

正式 dataset 必须绑定 `train_ctc40_valid.jsonl` 及其 SHA，且启动 canary 时再次用真实 tokenizer/blank
规则核对。

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

S1 -> S2a、S2a -> S2b、S2b -> S2c、S2c -> S3 等跨阶段操作必须：

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

新单阶段 S3 的代码、数据闭环、normalization、两类全量校验、真实 4-rank update-1 gate 已全部通过。
正式 `_single_stage_v2_` 目录只属于新策略；旧 S3a/S3b 目录仅供只读事故审计，禁止恢复。

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

### 步骤 4：补齐 CelebVDub Semantic-VAE 数据闭环（主体已完成）

- 79,826 条 fixed posterior `[T,64]` latent 已生成，音频/latent 全量只读校验已通过；
- 79,826 条 25 Hz -> exact 40 Hz 视频 cache 已生成，全量视频 validate-only 已通过；
- manifest 已保存原始采样点数和 artifact 绑定；decoder 后仍须按原始采样点数精确裁剪；
- 105 条 train CTC 不可行样本已写入明确排除 manifest；
- CelebVDub 不重算坐标系；LibriSpeech train `train_normalization.json` 原文件已取得并严格核验。

### 步骤 5：验收并正式训练单阶段 S3 C2（20k 门禁进行中）

- 旧 S3a/S3b 已经确定失稳并停止；禁止从其任何 checkpoint 续训或初始化；
- 新 `s3` 只从固定 S2c 70k EMA 初始化一次，200k 内保持同一 optimizer/scheduler/EMA；
- update 1 的真实 4-rank BF16 gate、TensorBoard tags 和 5.57 GB 完整 checkpoint validator 已通过；
- 当前从 update 1 严格恢复至 20k，实时现场见第 0.1.1 节；
- 到 20k 后先审计 diff/CTC、raw/post text RMS、global/per-group grad norm、optimizer momentum 和
  no-decay 参数变化，再验证 checkpoint；任何一项失败都不能继续到 200k；
- 门禁通过后设置 `S3_RUN_UNTIL_STAGE_UPDATE=200000`，仍使用同一 checkpoint 目录做 same-stage resume；
- 最终再保存并评估 50k/100k/150k/200k，正式 213 条 test 不参与 checkpoint 选择。

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
3. 核对 only-VAE 上 LibriSpeech train-only normalization 仍为 SHA256
   `65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6`；该文件已传输完成，不得在
   CelebVDub 上重算或替换坐标系。
4. 核对 S2c `model_70000.pt` 的 SHA、stage、update、EMA step 和 contract，并只做 EMA weights-only S3 初始化。
5. 重新检查 CelebVDub latent/video full completion marker、spec、完整 index 和 manifest SHA；两类全量
   validate-only 已于 2026-08-08 成功结束，不以文件数量或进度日志代替完成标记。
6. 用真实 tokenizer 做 CTC 最短路径预检，明确过滤名单，不依赖 `zero_infinity=True` 静默吞掉错标样本。
7. 不要把 codec ceiling WAV 目录当成 latent cache，也不要修改旧 `pretrain.py` 的 mel 语义。
8. 旧 S3b 已停止且禁止恢复。先核对新单阶段 launcher PID/SID、20k 日志、GPU、TensorBoard 新 run、
   最新 update 和 checkpoint；不能把旧 S3a/S3b event 曲线混作新实验结果。
9. 若进程再次消失，先查 stage 日志最后的 root cause、服务器 uptime、cgroup `memory.events` 和 GPU；不要
   未诊断就重启，也不要把外部 SIGKILL 误报成代码错误。
10. 同阶段只从 `model_last.pt` 的完整累计 update 恢复；日志中高于该值但未落盘的步数必须重算。
11. 新单阶段实验在 20k 门禁通过前不得直接扩展到 200k；通过后只能从同一完整 `model_last.pt` 恢复，
    不得重建 optimizer、scheduler 或 EMA。
12. 每个后续实现或文档步骤单独 commit/push，并核对本地 HEAD 与 `origin/main` 一致。
