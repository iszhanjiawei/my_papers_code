# D1 CTC lambda=0.2 评测结果

记录日期：2026-09-06；评测完成时间：2026-09-04

## 实验与权重

- 实验快照：`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio`
- 结构：D1，前 6 层 MM-DiT、后 12 层 Audio DiT，并在第 6、12 层施加双层 CTC 监督
- 训练设置：从零开始训练，`ctc_lambda=0.2`，4×RTX 4090
- `model_150000.pt`：checkpoint 内部记录为 update 150000
- `model_last.pt`：checkpoint 内部记录为 update 185000

Checkpoint 目录：

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_6mm_12audio_dual_ctc6_12_ctc02_finetune_hifigan_16k_CelebVDub_char/
```

## 评测口径

- 数据集：CelebV-Dub Setting 1
- 样本数：213
- 使用 EMA 权重
- seed：0
- ODE solver：Euler
- NFE：32
- sway sampling：-1
- 文本 CFG：5
- 视频 CFG：2
- 时长：ground-truth duration
- vocoder：HiFi-GAN 16 kHz
- 指标：SPKSIM↑、WER↓、EMOSIM↑、AVSync↑
- 指标统一以 `[0, 1]` 小数记录，例如 WER `0.06098` 等于 `6.098%`

## 正式结果

| Checkpoint | Update | SPKSIM↑ | WER↓ | EMOSIM↑ | AVSync↑ |
|---|---:|---:|---:|---:|---:|
| `model_150000.pt` | 150000 | **0.61936** | **0.06098** | **0.75623** | **0.51571** |
| `model_last.pt` | 185000 | 0.61889 | 0.06182 | 0.75070 | 0.51337 |

## 完整性与对比

两个 checkpoint 均完成：

- 213/213 条生成 WAV；
- 213/213 个生成音频对应的 AV-HuBERT 特征；
- SPKSIM、WER、EMOSIM、AVSync 四份完整逐样本结果及汇总值；
- 最终评测日志未出现 Traceback、CUDA OOM、子进程失败或数量校验失败。

从 150k 继续到 185k 后：

- SPKSIM 从 `0.61936` 降至 `0.61889`，变化 `-0.00047`，相对下降约 `0.08%`；
- WER 从 `0.06098` 升至 `0.06182`，绝对退化 `0.00084`，相对退化约 `1.38%`；
- EMOSIM 从 `0.75623` 降至 `0.75070`，变化 `-0.00553`，相对下降约 `0.73%`；
- AVSync 从 `0.51571` 降至 `0.51337`，变化 `-0.00234`，相对下降约 `0.45%`。

因此，`model_150000.pt` 在四项指标上均略优于 `model_last.pt`，是这两个权重中更可靠的选择。
其中 SPKSIM 基本持平，最明显的变化是 185k 的 EMOSIM 小幅下降；总体差距仍较小。

## 原始证据

150k 结果目录：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/results/finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_ctc02_150000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/
```

185k 结果目录：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/results/finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_ctc02_185000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/
```

评测日志：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/logs/eval_d1_ctc02_150k_last_1x4090_20260904_retry.log
```

可复现评测脚本：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/src/aligndit/run/eval/run_celebvdub_s1_d1_ctc02_150k_last_1x4090.sh
```
