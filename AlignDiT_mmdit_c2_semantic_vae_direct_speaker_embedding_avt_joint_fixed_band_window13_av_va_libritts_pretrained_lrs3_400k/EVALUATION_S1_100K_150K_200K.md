# AVT Joint：100k / 150k / 200k CelebV-Dub Setting 1 评测

评测日期：2026-10-19。目标项目为 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint`，即 speaker-only 源项目的 12 层 AVT joint + 6 层 audio-only 副本。

三份 EMA 权重分别固定在 GPU 0 / 1 / 2，独立完成推理及四项指标计算。每组覆盖 **213/213** 条，未筛选成功子集。

| EMA 权重 | GPU | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| 100k | 0 | 0.62015 | 0.05761 | 0.76118 | 0.55162 |
| 150k | 1 | 0.64554 | 0.05046 | 0.74958 | 0.54952 |
| 200k | 2 | 0.65360 | 0.05214 | 0.76118 | 0.54150 |

200k 的 SPKSIM 最高，150k 的 WER 最低，100k 的 AVSync 最高。100k 与 200k 的 EMOSIM 在五位小数下均为 0.76118。三份权重各有优势，没有一份在四项指标上全面最优。

## 固定协议

- CelebV-Dub Setting 1，213 条；完整同一 GT clip 作为 prompt，CAM++ 来自同一参考音频。此为同句参考协议。
- 输入真实目标文本，保留历史 prompt + target 文本拼接；不使用 VSR/ASR 预测文本作为模型条件。
- EMA，seed 0，Euler + EPSS，32 NFE，sway=-1，CFG text/video=5/2，真实时长。
- 原 Semantic-VAE-1000k decoder、64-D/40-Hz latent、原 LibriSpeech-train 归一化与 speaker cache。
- SPKSIM：既有 WavLM-Large + ECAPA 权重；EMOSIM：既有 emotion2vec-plus-large 分类 score 向量余弦，沿用历史口径。
- WER：既有 faster-whisper-large-v3，英语 beam size=5，沿用既有标点/大小写处理。WER 为小数比例，使用 corpus edit counts，不平均逐句 WER。
- AVSync：生成音频与对应真实嘴部视频共同提取 AV-HuBERT 特征，与既有 GT 音视频特征逐帧余弦后逐样本平均。

## 完整性与复算

- 每组均检查 213 个 WAV、213 个 AV-HuBERT 特征以及四份各 213 条的指标记录；日志末尾汇总行不计为样本。
- WAV 为 16 kHz 单声道，样本数与推理 receipt 一致，逐个 SHA256 一致，采样值有限。
- 特征形状与 GT 对齐且数值有限；逐样本 AVSync 从原始特征重新计算。
- 原单 worker 结果按测试清单顺序核对 basename，审计输出使用完整 `test/视频ID/片段ID`，保留跨视频同名 clip。
- SPKSIM/EMOSIM/AVSync 均从逐样本值复算均值，WER 从总词错误数/参考词数复算；所有结果与原日志五位小数一致。

| EMA 权重 | 词级错误总数 | 参考词总数 | WER |
|---|---:|---:|---:|
| 100k | 137 | 2378 | 0.05761 |
| 150k | 120 | 2378 | 0.05046 |
| 200k | 124 | 2378 | 0.05214 |

## 权重与产物

Checkpoint 根目录：`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char_avt_joint`。

| EMA 权重 | checkpoint SHA256 | 输出子目录 |
|---|---|---|
| 100k | `6dcb9d7804f4e458fce68a06a1172b4562cc6573ea923c40767d76882af63f2d` | `eval_s1_100000_speaker_avt_joint_cfgv2.0` |
| 150k | `5bd50c314faba4e45ae164675fbcdc7b3ef03ce18c870584cadb28160196d934` | `eval_s1_150000_speaker_avt_joint_cfgv2.0` |
| 200k | `de00fa08732eeed609d19411694f3cd2d136ac64432ab6c26b259d5d50b29a41` | `eval_s1_200000_speaker_avt_joint_cfgv2.0` |

每个输出目录保留 `inference_summary.json`、`test/` WAV、`avhubert_feat/test/`、四份原始 `_*_results.jsonl`、完整 ID 的 `evaluation_per_clip.jsonl` 及 `evaluation_audit.json`。

本项目运行记录：`logs/evaluation_s1_20261019_runs.json`；各任务日志为 `logs/eval_s1_<update>_speaker_avt_joint_20261019.log`；复算日志为 `logs/audit_s1_100k_150k_200k_20261019.log`。原推理/评测入口与模型代码均未改动。

## 复现命令

各组在相应 GPU 上独立运行；首次运行需使用空输出目录：

```bash
for spec in 100000:0 150000:1 200000:2; do
    eval_step="${spec%:*}"
    eval_gpu="${spec#*:}"
    setsid env PYTHONUNBUFFERED=1 CKPT_STEP="$eval_step" EVAL_GPU="$eval_gpu" \
        MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
        > "logs/eval_s1_${eval_step}_speaker_avt_joint.log" 2>&1 < /dev/null &
done
```

全部阶段完成后，使用 `scripts/audit_avt_joint_s1.py --output-dir <输出目录>` 核验；可重复传入 `--output-dir` 同时复核三个目录。运行产物不纳入 Git。
