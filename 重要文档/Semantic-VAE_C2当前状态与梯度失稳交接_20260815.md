# Semantic-VAE C2 minimal-fix v2 本次训练交接

> 记录日期：2026-08-15
> 本文只记录本次 Semantic-VAE C2 minimal-fix v2 的训练、梯度和评测情况；
> 不重复 C0–C3、D0–D2、数据预处理和旧 S3 历史。

## 1. 本次训练是什么

实验项目：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/
  AlignDiT_mmdit_c2_semantic_vae_direct/
```

实验名：

```text
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_40hz_CelebVDub_char
```

配置：

```text
src/aligndit/config/
  finetune_celebvdub_mm_c2_semantic_vae_minimal_fix.yaml
```

训练主要设置：

- 64D/40 Hz Semantic-VAE latent；
- 视频特征 25 Hz 离线插值到 40 Hz；
- 18 层，前 12 层 MM-DiT，后 6 层 audio-only DiT；
- 前 12 层使用文本 Cross-Attention；
- 连续单阶段、全参数训练，不冻结、不在中途重建 optimizer；
- 单一 AdamW learning rate `1e-5`；
- CTC 权重固定为 `0.1`；
- `max_grad_norm: 1.0`；
- 在 12 层文本 CA 共享 context 前使用无参数 padding-safe LayerNorm；
- 4 张 RTX 4090，BF16，seed 666。

父权重是已适配 64D/40 Hz latent 的 LibriSpeech 纯音频 S2c 70k EMA，不是从旧的坏
Semantic-VAE C2 checkpoint 续训。

## 2. 训练完成情况

- 已完整训练到 `200000` updates；
- 根据 TensorBoard wall time，1–200k 总训练时间约 `40.03` 小时；
- 全程平均速度约 `1.388 update/s`；
- checkpoint 正常保存到 50k、100k、150k 和 200k；
- 本次训练已结束，目前没有 minimal-fix v2 训练进程需要恢复。

checkpoint 目录：

```text
/zjw524/projects/data/ckpts/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_40hz_CelebVDub_char/
```

其中已有：

```text
model_50000.pt
model_100000.pt
model_150000.pt
model_200000.pt
model_last.pt
training_contract.json
parent_migration.json
gradient_spikes.jsonl
```

每个正式 checkpoint 约 5.12 GB，不得提交到 Git。

## 3. TensorBoard 与训练速度

TensorBoard event 目录：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/
  AlignDiT_mmdit_c2_semantic_vae_direct/runs/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_semantic_vae_40hz_CelebVDub_char/
```

主 event 文件约 69 MB，包含 1–200k 的 loss、LR 和 `grad_norm/global` 等指标。

分段速度：

| update 区间 | 速度 |
|---|---:|
| 1–10k | 1.383 update/s |
| 10k–50k | 1.420 update/s |
| 50k–100k | 1.404 update/s |
| 100k–140k | 1.357 update/s |
| 140k–150k | 1.424 update/s |
| 150k–160k | 1.436 update/s |
| 160k–170k | 1.413 update/s |
| 170k–180k | 1.370 update/s |
| 180k–190k | 1.360 update/s |
| 190k–200k | 1.267 update/s |

最后 10k 变慢与当时几乎每步触发详细梯度快照高度相关，但没做同机 A/B，不能宣称速度下降
全部由监控造成。

## 4. 梯度异常

minimal-fix v2 记录的是 DDP 同步、AMP unscale 之后、全局裁剪之前的 scale-safe FP64
global L2 norm。

| update 区间 | 裁剪前平均梯度范数 |
|---|---:|
| 140k–150k | 0.487 |
| 150k–160k | 10.79 |
| 160k–170k | 97.82 |
| 170k–180k | 289.51 |
| 190k–200k | 887.97 |

关键时间点：

- 第一次超过 10：update `149184`；
- 第一次超过 100：update `157239`；
- 第一次超过 1000：update `165726`；
- 全程共有 32,039 次范数超过 100。

异常快照中，约 95.2% 的“最大单参数张量梯度”来自：

```text
transformer_blocks.0.attn_norm.linear.weight
```

即长程失稳主要集中在第一层 MM-DiT attention normalization 路径。`95.2%` 不表示该张量
占全局范数的 95.2%，只表示它在 95.2% 的异常记录中排名第一。

`max_grad_norm: 1.0` 确实执行了裁剪，因此 `887.97` 不会原样进入 optimizer。当范数为 `G`时，
所有梯度大致一起乘上 `min(1, 1/G)`。但这会让异常的 block 0 决定全局缩放系数，
同时压小其他本来正常的梯度，因此仍属于持续性训练失稳，不是普通的偶发尖峰。

梯度详细记录：

```text
/zjw524/projects/data/ckpts/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_40hz_CelebVDub_char/
  gradient_spikes.jsonl
```

## 5. 评测结果

150k 和 200k 均完成 CelebV-Dub Setting 1 的 213/213 条完整推理和四项评测。

| 权重 | SPKSIM↑ | WER↓ | EMOSIM↑ | AVSync↑ |
|---|---:|---:|---:|---:|
| model_150000.pt | **0.37617** | **0.39529** | **0.62847** | **0.47884** |
| model_200000.pt | 0.14321 | 0.68419 | 0.45488 | 0.32316 |

200k 相比 150k 四项全面退化，与 150k 之后梯度持续增大的时间点一致。

评测汇总和日志：

```text
/zjw524/projects/data/eval_results/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_setting1/
```

生成音频：

```text
model_150000:
/zjw524/projects/data/eval_results/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_setting1/
  model_150000/celebvdub_test_s1/
  seed0_euler_nfe32_svae40_ss-1_cfgt5.0_cfgv2.0_gt-dur/

model_200000:
/zjw524/projects/data/eval_results/
  AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_setting1/
  model_200000/celebvdub_test_s1/
  seed0_euler_nfe32_svae40_ss-1_cfgt5.0_cfgv2.0_gt-dur/
```

## 6. 监控为什么会拖慢后期

单纯将一个梯度 scalar 写入 TensorBoard 开销很小。当前后期额外开销主要来自：

1. 每步额外用 FP64 遍历全模型计算 safe global norm；
2. `G > 100` 后再遍历全模型，计算每个参数张量的 norm 并排序 top-12；
3. 大量 GPU→CPU `.item()` 同步；
4. 每条 spike 都写 JSON、`flush()` 并 `os.fsync()`；
5. 全程 32,039 条 spike，`gradient_spikes.jsonl` 约 52 MB。

如果以后继续使用该监控，top-parameter 详细记录应改为首次越阈、前几次和每 100/500 步
抽样，JSON 应缓冲后定期落盘，不要每步 `fsync`。梯度裁剪和非有限值 fail-fast 仍应保留。

## 7. 本次训练的最终结论

1. 本次 minimal-fix v2 已完整跑完，不是进程中断或 checkpoint 未保存。
2. 150k 已显著低于原 mel C2，200k 因长程梯度失稳继续恶化。
3. `model_200000.pt` 不是正常收敛的最终权重，不应继续追加训练步数。
4. `model_150000.pt` 只能作为本次失败路线中相对较好的审计点，不能作为论文主结果。
5. 如果开启新 Semantic-VAE 结构实验，必须从干净的 S2c 70k EMA 开始，不得从本次
   150k/200k 续训。
6. 新实验应首先针对第一层 MM-DiT 和 40 Hz 视频进入 joint softmax 的干扰做受控修改，
   先跑 1/100 update 和 5k/20k 门禁，不要再直接启动 200k。

## 8. 与本次训练直接相关的代码

```text
config:
AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/config/
  finetune_celebvdub_mm_c2_semantic_vae_minimal_fix.yaml

trainer:
AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/model/
  trainer_semantic_vae_minimal_fix.py

common training loop:
AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/model/
  trainer_vt.py

launcher:
AlignDiT_mmdit_c2_semantic_vae_direct/src/aligndit/run/train/
  finetune_celebvdub_mm_c2_semantic_vae_minimal_fix_4x4090.sh
```

新会话如要继续排查，先读本文，再查 TensorBoard event 和 `gradient_spikes.jsonl`；不需要重新整理前面
已完成的所有实验。
