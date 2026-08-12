# Semantic-VAE C2 旧 S3 高损失与静默零梯度错误总结

> 记录日期：2026-08-09  
> 影响项目：`AlignDiT_mmdit_c2_semantic_vae`  
> 事故范围：已废弃的 `S3a -> S3b` CelebVDub 多模态训练  
> 当前正式路线：S2c 70k EMA -> 连续单阶段 S3 200k

## 1. 先给结论

旧 mel C2 可以加载纯音频预训练权重后直接训练；Semantic-VAE 在完成 S2c 纯音频适配后，同样可以、
也应该直接进入一个连续的 C2 多模态训练阶段。

真正需要分阶段解决的是前面的表示迁移：

```text
80D / 100 Hz mel 纯音频预训练
    -> S1 / S2a / S2b / S2c
    -> 64D / 40 Hz Semantic-VAE latent 纯音频模型
```

S2c 已经完成输入输出维度、时间频率、latent 分布和音频主干的适配。此后再把 CelebVDub C2 拆成
`S3a 5k + S3b 195k`，不是 VAE 的必需条件，而是一次过度保守且最终被证明有害的训练策略设计。

旧 S3 的失败由三层问题叠加造成：

1. **结构直接根因**：四层 TextEmbedding/ConvNeXt 没有最终输出归一化，放大的文本 context 被连续送入
   12 层 Cross-Attention；
2. **训练触发条件**：旧 S3a/S3b 对文本、视频、CTC、Cross-Attention 等约 1.96 亿新参数统一使用
   `5e-5`，warmup 很短，且在 5k 边界重建 optimizer/scheduler、清空 Adam moments；EMA 也被重置，
   但 EMA 不参与反向传播，不是爆炸触发器；
3. **错误隐藏机制**：旧 trainer 丢弃 `clip_grad_norm_` 返回的 pre-clip norm。FP32 norm 溢出为
   `inf` 后，clip coefficient 变成 0，梯度被静默清零，但训练进程仍继续增加 update。

因此，旧实验不是简单的“loss 偏高”或“收敛慢”，而是从约累计 42.8k 起已经基本停止有效学习。

## 2. 为什么没有 VAE 时可以直接加载纯音频预训练模型

### 2.1 旧 mel 预训练和下游微调处在同一表示空间

旧 C2 的 LibriSpeech 预训练输入和 CelebVDub 微调目标都是：

```text
80 维 mel
100 Hz（hop_length=160, sample_rate=16 kHz）
相同的 mel 计算方式和数值语义
```

因此纯音频 checkpoint 中的 input projection、audio blocks、time conditioning、输出层及其上下游尺度
基本处在下游任务可以直接使用的坐标系。新增的文本、视频和 MM-DiT 参数仍需学习，但不需要先改变音频
表示本身。

旧 C2 配置也确实是一个连续训练任务：

- `finetune_celebvdub_mm_c2.yaml`：统一 LR `5e-5`；
- warmup 为 20k；
- 直接加载 `AlignDiT_pretrain_LibriSpeech_500000.pt`；
- `AdamW(model.parameters())` 从一开始覆盖全部参数，没有隐藏的冻结阶段；
- 部分新增 residual 分支采用 zero gate 起步，降低但不能消除随机分支的初始扰动；随机 video K/V 仍可
  经已加载的 audio attention gate 从首步影响音频；
- 只构造一次 optimizer、scheduler 和 EMA，中途不在 5k 处重置。

对应代码：

```text
AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/
  src/aligndit/config/finetune_celebvdub_mm_c2.yaml:13-18,65-72
  src/aligndit/script/train/finetune.py:50-110
  src/aligndit/model/trainer_vt.py:21-91,146-162
  src/f5_tts/model/trainer.py:101-141
  src/aligndit/model/backbone/dit_vt_mm.py:451-470
```

### 2.2 Semantic-VAE 多了一次“表示迁移”，但这已由 S1-S2c 完成

mel checkpoint 不能直接当作 Semantic-VAE checkpoint，原因包括：

- 通道从 80D 变成 64D；
- 帧率从 100 Hz 变成 40 Hz；
- 训练目标从 mel 变成 Semantic-VAE posterior latent；
- input/output projection 形状发生变化；
- 位置、时间和 CTC/HuBERT 路径面对的序列统计发生变化。

所以先做 S1-S2c 是合理且必要的。它们解决的是“mel 音频模型如何变成 latent 音频模型”，而不是
“多模态模型是否必须分两段训练”。

完成 S2c 70k 后，已经得到经过 LibriSpeech 适配的 64D/40 Hz 纯音频模型。此时接入 CelebVDub 的任务
与旧 mel C2 的逻辑重新一致：以一个可靠的纯音频父权重为起点，加入随机初始化的多模态分支并连续训练。

严格迁移审计也支持这一点：S2c source 有 313 个 state key，C2 target 有 703 个；加载 303 个兼容 key，
只忽略 10 个旧 HuBERT projector key，新建 400 个文本/视频/MM-DiT/CTC key，且共同 key 没有 shape
变化。也就是说，纯音频主干已经是可复用的成熟父模型，剩余问题是正常的多模态联合优化，不是再次做
mel-to-latent 接口适配。

### 2.3 为什么当时仍拆成 S3a/S3b

当时的考虑是：

- S3a 先冻结已适配音频主干，只训练新文本/视频/MM-DiT/CTC 路径；
- S3b 再解冻全部网络，避免随机多模态模块立即破坏音频能力。

这个想法在概念上像“接口校准”，但对当前 MM-DiT 计算图并不成立。即使冻结音频参数，随机视频 K/V
也会从第一个 update 起进入音频 joint attention；S3a 并不是 function-preserving 的零扰动阶段。

更重要的是，S3a/S3b 实现同时引入了以下不利条件：

| 项目 | 旧 S3a | 旧 S3b |
|---|---:|---:|
| 训练长度 | 5k | 195k |
| 新多模态统一 LR | `5e-5` | `5e-5` |
| warmup | 500 | 5k |
| 音频主干 | 冻结 | 一次性全部解冻 |
| optimizer/scheduler/EMA | 新建 | 在 5k 边界再次新建 |

阶段切换丢掉了 S3a 已形成的 Adam moments，又给已经开始漂移的文本塔一次短 warmup、高 LR 重启。
EMA step 的重置还破坏了评估权重的时间线连续性，但不参与梯度爆炸的因果链。所以这里的两阶段不是
“因为用了 VAE 所以必须这样做”，而是一次后来被实验证伪的保护策略。

## 3. 训练错误到底是什么

### 3.1 表面现象

旧 S3b 的 TensorBoard 大致经历了两次跃迁：

| 累计 update | diffusion loss 现象 |
|---:|---|
| 5k-15k | 约 1.39，表面正常 |
| 约 19.1k | 第一次快速恶化，随后进入约 15-16 |
| 约 42.65k-42.8k | 第二次灾变，短时超过 40，之后长期约 21-22 |

进程没有出现 CUDA OOM、NCCL error、NaN loss 或 Python traceback，GPU 也一直在工作，update/s 正常。
这使它最初看起来像是“模型还在训练，只是 loss 比较高”。这个判断是错误的。

### 3.2 文本塔尺度爆炸

固定同一个真实 CelebVDub 样本、同一组 `x0/t/mask`，四层文本 ConvNeXt 输出 RMS 为：

| checkpoint | block 0 | block 1 | block 2 | block 3/final |
|---|---:|---:|---:|---:|
| S2c -> S3 初始化 | 1.27 | 1.28 | 1.30 | 1.31 |
| 旧 S3a 5k | 1.37 | 1.72 | 2.69 | 4.43 |
| 旧 S3b 50k | 2.00 | 10.51 | 76.21 | 219.59 |

TextEmbedding 的四个 ConvNeXtV2 block 都是残差结构，旧实现没有最终输出 norm。最终文本 context 直接
作为 12 层文本 Cross-Attention 的 K/V，因此尺度增长被反复注入音频流。

旧 S3b 50k 的固定 forward 进一步得到：

- text context RMS：约 219.6；
- block 0 text Cross-Attention 输出 RMS：约 122；
- block 0 音频 hidden RMS：约 356；
- 末端 hidden RMS：超过 1000；
- 最终输出 RMS：约 4.6；
- 固定样本 diffusion MSE：约 24，而健康 checkpoint 约 1.8。

最终 `norm_out` 把输出压回有限范围，所以 loss 没有变成 NaN；但内部残差和反向 Jacobian 已经爆炸。

### 3.3 梯度范数溢出并被静默清零

在旧 S3b 50k 上，真实 batch 的梯度元素本身仍全部 finite，但用稳定 FP64 计算得到 global L2 norm 约：

```text
3 × 10^23
```

PyTorch/Accelerate 原来的 FP32 global norm 在平方求和时先溢出成 `inf`。随后梯度裁剪近似执行：

```text
clip_coef = max_grad_norm / (inf + eps) = 0
gradient = gradient * 0
```

旧 trainer 只调用：

```python
self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```

但没有读取返回值，也没有在 `optimizer.step()` 前检查 pre-clip norm。本事故的直接故障源是修复提交
`66791d8` 之前的 Semantic-VAE trainer：

```text
git show 66791d8^:AlignDiT_mmdit_c2_semantic_vae/src/aligndit/model/trainer_semantic_vae_c2.py
# 历史版本约 429-435 行
```

旧 mel `trainer_vt.py:210-226` 也存在“丢弃 clip 返回值”的同类潜在缺陷，但它不是这次 Semantic-VAE
事故日志所对应的直接代码版本。

结果是：loss finite、参数 grad 元素 finite、进程无异常，但裁剪后的有效梯度几乎全为 0。

### 3.4 optimizer 审计证明模型已经停止有效学习

旧 50k/100k checkpoint 的 optimizer state 有确定性证据：

- 共 696 份 Adam state；
- 693 个 `exp_avg` 只剩 float32 最小次正规数 `5.605e-45`；
- 其余 3 个为精确 0；
- 从 50k 到 100k，703 个在线模型 tensor 中 379 个逐 bit 不变；
- 其余大多只是 AdamW weight decay 引起的衰减，不是正常梯度学习。

所以 update 仍从 50k 增加到 100k，不代表模型完成了 50k 次有效优化。

## 4. 为什么不是数据、CTC 或恢复造成的

已经排除的方向：

### 4.1 不是 latent normalization

用固定 LibriSpeech train mean/std 抽样 500 条 CelebVDub、80,730 帧：

- normalized latent 全局 mean 约 0.0023；
- 通道方差均值约 1.079；
- `E[x1^2]` 约 1.091；
- 零预测器的理论 flow MSE 约 2.091。

坏模型的 diffusion loss 约 21-24，远超目标数据的自然尺度，不能由 latent 分布解释。

### 4.2 不是 CTC 主触发

- 去掉 CTC、只反传 diffusion loss，仍复现约 `3e23` 的 global grad norm；
- 最大梯度来自 block 0 text Cross-Attention 的 Q/K projection 和 CA AdaLN；
- CTC 在失稳前后也恶化，说明它受到共享 hidden state 影响，但不是本次爆炸的直接根因。

CTC 随机头立即以 `lambda=0.1` 回传仍然是不良放大因素，因此新策略保留 20k ramp，但不能把事故简单
归结成“CTC 层数或位置错误”。

### 4.3 不是 resume、scheduler、EMA 或 GPU 型号

- scheduler state、world-size 步进和实际 LR 与配置一致；
- resume 前后重叠的 4,572 个 update，TensorBoard loss 逐点、逐 bit 一致；
- EMA 与 online model 都忠实跟随了坏模型，EMA 不是触发器；
- 固定 checkpoint 离线 forward 可以稳定复现高 loss 和巨大梯度，与使用 4090/A40 无关；
- TensorBoard 只是正确记录了坏模型的 loss，不是显示错误。

### 4.4 不是视频 25 Hz -> 40 Hz 插值

同一坏 checkpoint 中 audio input、video input 和 time input 的初始 RMS 都在正常量级，第一个明显异常
张量是 TextEmbedding 最终输出及其后的 block 0 text Cross-Attention。视频插值和长度对齐已通过全量校验。

## 5. 为什么旧 mel C2 没有明显触发同一事故

旧 mel 模型并非在数学上绝对不会遇到这个风险；旧 trainer 同样没有检查 `clip_grad_norm_` 返回值，
TextEmbedding 也缺少最终 norm。它只是没有在既有 C0-C3/D0-D2 训练中跨过失稳阈值。

与失败的 Semantic-VAE staged 配置相比，旧 mel C2 有几个更稳定的条件：

- 音频预训练和微调使用同一 80D/100 Hz 表示；
- 一个连续 optimizer/scheduler/EMA 时间线；
- 20k warmup，而不是 500 + 5k 两次短 warmup；
- 没有在文本塔已经漂移后清空 Adam moments 并重新升 LR；
- 实际稳定 mel D2 checkpoint 的四层文本 RMS 约 1.0-1.4，没有发生级联放大。

因此不能得出“Semantic-VAE 天生不稳定”或“必须先冻结再解冻”的结论。更准确的结论是：旧 staged
训练策略触发了原架构中潜伏的文本 context 无界风险，而旧 trainer 又把灾难性梯度裁剪隐藏了。

## 6. 已经如何修复

### 6.1 废弃旧权重，从干净 S2c 70k 重新开始

以下全部只保留故障审计，禁止续训、正式评测或作为父权重：

- 旧 S3a `model_5000.pt`；
- 旧 S3b 50k/100k/115k；
- 旧 S3b `model_last.pt`。

新单阶段 S3 只允许从严格验证的 S2c 70k EMA 初始化，并使用独立 checkpoint 目录和 contract。

### 6.2 在所有文本 CA 前统一做无参数 LayerNorm

修复位置：

```text
AlignDiT_mmdit_c2_semantic_vae/
  src/aligndit/model/backbone/dit_vt_mm.py:569-613,749
```

处理规则：

- 对每个有效文本 token 的 feature 维执行 `F.layer_norm`；
- `weight=None, bias=None`，不新增 checkpoint state key；
- norm 后重新把 padding 置 0；
- 12 层 Cross-Attention 共用同一个受控 context；
- 同时记录 raw/post text RMS。

在已经崩坏的旧 50k checkpoint 上，用另一组固定合成审计输入做只读反事实实验，仅增加这一层 context
norm 就把 raw grad norm 从约 `2.7e18` 降到约 `83.9`。这组数值和第 3.3 节正式真实 batch 的约 `3e23`
来自不同输入与审计设置，不能横向当成同一 batch；两者共同证明 context norm 会把异常梯度降低多个数量级。

提交：`e345c320ee4f9c28503b5f562ea8237fea7c5d3b`。

### 6.3 增加梯度与文本尺度 fail-fast

修复位置：

```text
AlignDiT_mmdit_c2_semantic_vae/
  src/aligndit/model/trainer_semantic_vae_c2.py:159-239,526-611
```

现在会：

- 捕获 `clip_grad_norm_` 返回的 pre-clip global norm；
- 检查非有限值和阈值；
- 用 DDP reduce 同步所有 rank 的失败状态；
- 在任一 `optimizer.step()` 之前抛出 `FloatingPointError`；
- 每 100 update 用 FP64 记录各 optimizer group 的 pre-clip norm；
- 写入 TensorBoard：`grad_norm/global`、`grad_norm/group/*`、
  `text_context/raw_rms`、`text_context/post_rms`、`ctc_lambda`。

当前硬门限：

```text
raw text context RMS <= 3
global pre-clip grad norm <= 100
所有 loss / RMS / grad norm 必须 finite
```

提交：`66791d840243eb0e069dbb2bff9ebc03ccf5d273`。

### 6.4 用连续单阶段策略替代 S3a/S3b

当前配置：

```text
AlignDiT_mmdit_c2_semantic_vae/
  src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_s3.yaml
```

固定策略：

| 参数组 | 最大 LR |
|---|---:|
| text conditioner + text Cross-Attention | `5e-6` |
| video/MM core | `1e-5` |
| 新 Cross-Attention/video gates | `1e-6` |
| CTC heads | `1e-5` |
| latent input/output interface | `5e-6` |
| audio blocks 0-5 | `2e-6` |
| 其余 audio backbone | `5e-6` |
| shared time embedding + norm-out | `1e-6` |

此外：

- 总预算 200k；
- warmup 20k；
- CTC lambda 首步为 `0.1/20000=5e-6`，第 20k 步达到 0.1；
- 从 0 到 200k 只构造一次 optimizer、scheduler 和 EMA；
- 中断后必须恢复 online model、optimizer、scheduler、EMA 和 update 全部状态；
- 不能在 5k 或 20k 处重建训练状态。

提交：`5ad4095146c1128c46bf4cc2fc679b7a87495ae5`。

## 7. 20k 门禁不是新的“两阶段”

当前 launcher 首次设置 `run_until_stage_update=20000`，只是让同一个 S3 任务在 20k 自动保存并退出，
便于审计。它不改变 200k scheduler horizon，也不会在继续训练时重建 optimizer/scheduler/EMA。

20k 验收通过后，从同一 `model_last.pt` 做 exact same-stage resume 到 200k，语义等同于同一训练被安全
暂停后继续。因此：

```text
错误说法：新方案又分成了 0-20k 和 20k-200k 两个训练阶段
正确说法：新方案只有一个 S3；20k 是同一状态轨迹上的人工验收断点
```

设置 20k 门禁是因为旧实验第一次明显失稳出现在累计约 19.1k。即使 20k 通过，也不能宣称整个 200k
必然安全；旧实验第二次灾变在约 42.8k，因此继续训练时仍由 fail-fast 和 TensorBoard 全程监控，并在
50k/100k/150k/200k 做 checkpoint 与 optimizer 审计。

## 8. 当前验证状态

真实 4×4090 update-1 gate 已通过：

```text
diff loss                 1.343
raw CTC loss             10.263
CTC lambda                0.000005
global pre-clip norm      0.388
raw/post text RMS         1.301 / 1.000
model/update/EMA step     1 / 1 / 1
```

完整 checkpoint 的 model、optimizer、scheduler、EMA、update 和 contract 已通过严格 validator。

正式 20k 门禁训练正在运行。只有满足以下全部条件，才能从同一状态继续到 200k：

1. diffusion loss 没有旧实验式阶跃；
2. raw text RMS 始终小于 3，post RMS 约为 1；
3. global/per-group grad norm finite，global 小于 100；
4. Adam moments 处于正常数值范围；
5. no-decay 参数确实发生梯度驱动的更新；
6. `model_last.pt` 通过严格 resume validator。

在 20k 完成以前，只能说“修复已实现、早期门禁正常”，不能写成“长期问题已经完全解决”。

## 9. 必须记住的坑与禁令

1. **不要把 VAE 当作必须拆 S3a/S3b 的理由。** VAE 表示迁移已经由 S1-S2c 完成。
2. **不要恢复任何旧 S3a/S3b checkpoint。** 即使文件完整、loss finite，也可能没有有效梯度。
3. **不要只监控 loss、GPU 利用率和 update/s。** 必须看 raw/post text RMS、pre-clip grad norm 和
   optimizer moments。
4. **不要忽略 `clip_grad_norm_` 返回值。** “设置了 max_grad_norm=1”不等于训练安全。
5. **不要在阶段边界无理由重置 Adam/EMA。** 如果只想暂停验收，应 exact resume，而不是 weights-only
   重新微调。
6. **不要把 CTC 相关恶化误判成唯一根因。** diff-only 已复现爆炸；CTC ramp 是稳定措施，不是直接修复。
7. **不要把 4090/A40 差异、视频插值或 TensorBoard 当作本事故解释。** checkpoint 离线复现已经排除。
8. **不要在 CelebVDub 重算 latent mean/std。** 坐标系固定使用 S2c 的 LibriSpeech train-only stats。
9. **不要把 20k 门禁说成第二套 staged training。** 它是同一 optimizer 状态的安全暂停点。
10. **不要在 20k 通过后关闭监控。** 仍须覆盖旧第二灾变区间约 42.8k，并审计 50k checkpoint。

## 10. 代码与提交索引

| commit | 内容 |
|---|---|
| `e345c32` | 给共享文本 context 增加无参数、padding-safe LayerNorm |
| `66791d8` | 增加 CTC ramp、文本 RMS、pre-clip grad norm、DDP fail-fast 和 TensorBoard 指标 |
| `5ad4095` | 用从 S2c 70k 开始的连续单阶段 S3 替代旧 S3a/S3b |
| `48d8092` | 更新 AGENTS 与 Semantic-VAE 主交接文档，禁止恢复旧权重 |

权威入口：

```text
配置:
  AlignDiT_mmdit_c2_semantic_vae/src/aligndit/config/
    finetune_celebvdub_mm_c2_semantic_vae_s3.yaml
launcher:
  AlignDiT_mmdit_c2_semantic_vae/src/aligndit/run/train/
    finetune_celebvdub_mm_c2_semantic_vae_single_stage_4x4090.sh
checkpoint:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3_single_stage_v2_40hz_CelebVDub_char/
```

## 11. 2026-08-10 最终纠正：稳定化 S3 不能冒充“只替换 VAE”的 C2 对照

前述 single-stage v2 解决的是旧 S3 已经观测到的数值灾变，因此加入了文本 LayerNorm、八组学习率、
CTC ramp、RMS/梯度硬门禁等稳定化设计。这些设计本身可以继续作为独立的稳定性实验，但它们同时改变了
训练策略和网络前向，不能回答“原 C2 只把 mel 换成 Semantic-VAE 后效果如何”。此前把它称为当前唯一
训练路线，会混淆两个不同的科研问题。

最终按用户要求新建严格对照快照：

```text
AlignDiT_mmdit_c2_semantic_vae_direct/
```

它直接从原 mel C2 `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` 复制，只允许以下必要变化：

1. `80D/100 Hz mel -> 64D/40 Hz Semantic-VAE latent`，使用 S2c 绑定的 LibriSpeech-train mean/std；
2. 视频使用已经预提取的精确 40 Hz cache，音视频长度比改为 1:1；
3. CTC tap 仍为 `[6,12]`，只将时间 stride 从 `[2,1]` 改为 `[1,1]`，避免 40 Hz 再降到 20 Hz；
4. frame batch 从 `9000@100 Hz` 等时长换算为 `3600@40 Hz`；
5. 从已完成表示适配的纯音频 S2c 70k EMA 做严格权重迁移。

下列内容全部恢复原 C2，不再加入稳定化变量：

```text
单阶段、全部参数可训练、单一 AdamW 参数组、LR=5e-5、warmup=20k、
CTC lambda 从第1步固定为0.1、200 epochs、原始 EMA 默认节奏、
无额外 text LayerNorm、无冻结、无分组 LR、无 CTC ramp、无 RMS 硬停止。
```

训练数据使用完整 79,613 条，与原 mel C2 的 `raw.arrow` 逐条一致。40 Hz 下新增的 105 条 CTC
不可行样本仍参与 diffusion loss，并沿用原 C2 的 `zero_infinity=True` 令其 CTC 项为 0。不能使用
79,508 条过滤集，否则会额外改变训练数据集合。

代码提交为 `44d9df2`。正式 4×4090 Direct-C2 任务已经从干净的 S2c 70k 启动，独立输出到：

```text
/zjw524/projects/data/ckpts/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_c2_40hz_CelebVDub_char/
```

启动验收已连续超过 200 updates，约 `1.53-1.55 update/s`；早期 `diff_loss` 约 1.3-1.5、原始
`ctc_loss` 约 7-10、总 `loss` 约 2.0-2.5，四卡进程、梯度更新和 TensorBoard 均正常。总 loss
高于使用 CTC ramp 的实验是预期现象，因为原 C2 从首步就计算 `0.1 * ctc_loss`，不能据此再次擅自
改回 ramp。

后续纪律：

- Direct-C2 是“只替换 Semantic-VAE”的主对照；single-stage v2 是额外稳定化实验，二者不能混写；
- Direct-C2 若后续失稳，应先保留现场并把它作为实验事实分析，不能在同一实验中途改变训练公式；
- 任意 LayerNorm、分组学习率、CTC ramp、冻结或硬门禁，都必须另建实验目录和名称；
- 禁止将旧 S3a/S3b、旧 single-stage v2 50k 权重续入 Direct-C2 输出目录。

## 12. 2026-08-12 Direct-C2 长程结果：严格对照同样失稳

Direct-C2 最终连续运行到约 update 208.9k。它没有 S3a/S3b 阶段边界，因而给出了一个重要纠正：

> 两阶段重置会放大旧 S3 的问题，但不是必要条件，也不是唯一根因。即使完全复刻原 C2 的一次性
> optimizer/scheduler/EMA，Semantic-VAE Direct-C2 仍会触发同一种文本入口尺度失稳和静默零梯度。

### 12.1 精确时间线

TensorBoard 共记录 208,903 个 update：

| 区间 | 现象 |
|---:|---|
| 0-29.7k | `diff_loss` 约 1.37-1.40，表面健康 |
| 29,998-30,401 | 首次灾变，总 loss 峰值 174.397 |
| 30.5k-85.2k | 暂时回落到约 3.7-5，但已明显劣化 |
| 85,215 起 | loss 持续大于 10，随后长期约 21-23 |

第一次灾变由 diffusion 分支主导，CTC 同时恶化但不是单独触发源。Direct 配置已经证明“照旧 C2、
只改 VAE 必需接口”不是尚未尝试的方案，而是已经被长程实验否定的方案。

### 12.2 checkpoint 证明 50k 前已停止有效学习

50k/100k/150k/200k 四个编号 checkpoint 均无 NaN/Inf，但 optimizer 有决定性证据：

- 318,749,570 个 Adam `exp_avg` 元素中，约 0.93% 为 0，其余全部衰减到不超过
  `5.605193857e-45` 的 FP32 次正规数；
- 150k->200k 的代表性 `exp_avg_sq` 比值为 `1.8823e-22`，与
  `0.999^50000 = 1.8811e-22` 基本相同，证明这 50k 步没有新的有效平方梯度贡献；
- 同期代表性权重仅缩小约 0.994，符合 AdamW weight decay，而不是正常梯度学习；
- S2c 70k 父权重的 `exp_avg` RMS 为 `2.07e-6`，故障不是从纯音频父模型继承的。

因此 Direct 的 50k/100k/150k/200k/last 只保留用于事故审计，禁止续训、评测或作为父权重。

### 12.3 激活与 attention 的直接证据

固定输入审计结果：

| checkpoint | Text final RMS | time embedding RMS | block-0 gated CA residual RMS |
|---|---:|---:|---:|
| mel D2 50k | 1.406 | 0.483 | 0.054 |
| stable-v2 50k | 2.641，送 CA 前归一为 1 | 0.711 | 0.054 |
| Direct-C2 50k | 268.1 | 13.83 | 2492 |

Direct 50k 的 block-0 CA attention-logit RMS 达到约 `1.63e7`，注意力熵接近 0。这里配置中的
RMS QK-Norm 没有保护文本 CA：文本 CA 使用 `nn.MultiheadAttention`，其文本 K/V 未经过该 QK-Norm。
由于 30k 附近没有保存权重和分组梯度日志，不能声称 TextEmbedding 是时间上第一个异常参数；严谨结论是
最早可审计的异常簇集中在 TextEmbedding/GRN、block-0 audio attention、block-0 text CA 和 time/AdaLN。

### 12.4 为什么 mel C2 没触发、Semantic-VAE 却触发

这不是“VAE 让 TextEmbedding 天生失效”，而是换表示后旧超参的稳定裕度不足：

- S2c 到 C2 target 只迁移 303/703 个 key，约 47.63% 参数量；400 个文本、视频、MM、CTC key 新建；
- Direct 丢弃 S2c optimizer 状态，把所有 701 个参数统一放入 `AdamW(lr=5e-5, weight_decay=0.01)`；
- 标准化 latent 的健康初期 diffusion loss 约 1.4，高于 mel C2 约 0.7-0.8；
- 同等 90 秒/GPU 下，loss 平均的音频标量数由 `9000*80` 降为 `3600*64`，少约 3.125 倍，
  梯度噪声更大；
- 视频 token 从相对音频的 1/4 变成 1:1，随机新视频路径的相对参与度上升；
- CTC 时间网格由 50 Hz 变为 40 Hz，且随机 CTC 头从首步固定权重 0.1。

这些因素改变了梯度和 Jacobian 的尺度，使原本无输出归一化的文本入口跨过失稳阈值。latent mean/std、
25->40 Hz 视频插值、GPU 型号、resume 计数和 EMA 已分别通过数据或 checkpoint 审计排除为直接根因。

## 13. 彻底修复：独立 minimal-fix v1

修复没有覆盖 Direct 历史语义，而是在 `AlignDiT_mmdit_c2_semantic_vae_direct` 中新增独立配置、入口、
实验名称和 checkpoint 目录。它仍是一个连续单阶段任务：不冻结、不拆 S3a/S3b、不重置 optimizer，
使用完整 79,613 条训练集，12 MM + 12 text、固定 CTC 0.1、单一 AdamW 参数组。

只增加两项训练语义变化：

1. 同一个文本 context 进入 12 层 CA 前执行一次无参数、padding-safe、per-token LayerNorm；
2. 已被 Direct 长程证伪的全局 LR `5e-5` 降到 `1e-5`，20k warmup 和其余 C2 公式不变。

工程保护不会改变健康轨迹：

- native clip 前用 FP64 per-tensor norm + host `math.hypot` 得到不会静默溢出的 pre-clip global norm；
- norm 非有限、`<=1e-12` 或 `>100` 时在全部 DDP rank 同步、在 optimizer step 前终止；
- TensorBoard 每步记录 loss、diff/CTC、raw/post text RMS 和 pre-clip norm；
- checkpoint 采用同盘临时文件、fsync、原子 replace；
- checkpoint 固定 schema/policy/contract hash，并绑定 seed=666、world-size=4、batch 策略、完整 manifest、
  normalization、vocab、S2c parent 及 resolved config；加载时选择 update 最大且 contract 匹配的完整文件；
- 每个 update 根据 `seed + update + rank` 重建 Python/Torch 随机流，固定四卡、grad accumulation=1。

固定入口：

```text
配置:
  AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/config/
    finetune_celebvdub_mm_c2_semantic_vae_minimal_fix.yaml
launcher:
  AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/run/train/
    finetune_celebvdub_mm_c2_semantic_vae_minimal_fix_4x4090.sh
checkpoint:
  /zjw524/projects/data/ckpts/
    AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v1_40hz_CelebVDub_char/
```

真实 4x4090 已完成 0->1、1->2 的真实数据/真实父权重 resume smoke：S2c 迁移仍为
`303 loaded / 10 ignored / 400 new`；update 1 的 `diff=1.343`、raw CTC=12.700、
pre-clip norm=2.102、raw/post text RMS=`1.301/1.000`；update 2 续训也保持 finite，并且 checkpoint
schema、contract、optimizer、scheduler、EMA step 均通过读取验证。

这说明代码层的故障链和静默假训练已被切断。长期数值稳定仍必须用同一个正式轨迹跨过旧危险区
30k、50k、85-90k 和 100k 才能最终确认，不能仅凭两步 smoke 宣称 200k 指标已经得到保证。
