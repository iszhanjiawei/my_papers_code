# D1 CTC lambda=0.3 评测结果

记录日期：2026-09-06；评测完成时间：2026-09-04

## 实验与权重

- 实验快照：`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio`
- 结构：D1，前 6 层 MM-DiT、后 12 层 Audio DiT，并在第 6、12 层施加双层 CTC 监督
- 训练设置：`ctc_lambda=0.3`，4×RTX 4090
- `model_150000.pt`：update 150000
- `model_last.pt`：checkpoint 内部记录为 update 185000

Checkpoint 目录：

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_6mm_12audio_dual_ctc6_12_ctc03_fresh_finetune_hifigan_16k_CelebVDub_char/
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
- 指标统一以 `[0, 1]` 小数记录，例如 WER `0.05929` 等于 `5.929%`

## 正式结果

| Checkpoint | Update | SPKSIM↑ | WER↓ | EMOSIM↑ | AVSync↑ |
|---|---:|---:|---:|---:|---:|
| `model_150000.pt` | 150000 | **0.60345** | 0.05929 | 0.72959 | **0.51381** |
| `model_last.pt` | 185000 | 0.60312 | **0.05635** | **0.75147** | 0.49958 |

## 完整性与对比

两个 checkpoint 均完成：

- 213/213 条生成 WAV；
- 213/213 个生成音频对应的 AV-HuBERT 特征；
- SPKSIM、WER、EMOSIM、AVSync 四份完整逐样本结果及汇总值；
- 最终评测日志未出现 Traceback、CUDA OOM、子进程失败或数量校验失败。

从 150k 继续到 185k 后：

- SPKSIM 从 `0.60345` 降至 `0.60312`，变化 `-0.00033`，基本持平；
- WER 从 `0.05929` 降至 `0.05635`，绝对改善 `0.00294`，相对改善约 `4.96%`；
- EMOSIM 从 `0.72959` 升至 `0.75147`，提高 `0.02188`，相对提高约 `3.00%`；
- AVSync 从 `0.51381` 降至 `0.49958`，下降 `0.01423`，相对下降约 `2.77%`。

因此，这两个权重形成明确的指标权衡：185k 的内容正确性和情感一致性更好；150k 的说话人相似度
略高且音视频同步更好。不能声称其中一个权重在四项指标上全面优于另一个。

## 原始证据

150k 结果目录：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/results/finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_ctc03_fresh_150000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/
```

185k 结果目录：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/results/finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_ctc03_fresh_185000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/
```

评测日志：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/logs/eval_d1_ctc03_fresh_150k_1x4090_20260904_retry1.log
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/logs/eval_d1_ctc03_fresh_last185k_1x4090_20260904_retry1.log
```

可复现评测脚本：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/src/aligndit/run/eval/run_celebvdub_s1_d1_ctc03_fresh_150k_last_1x4090.sh
```
