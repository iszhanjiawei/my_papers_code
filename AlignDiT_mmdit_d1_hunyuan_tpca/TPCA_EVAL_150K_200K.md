# TPCA 150k / 200k CelebV-Dub Setting 1 评测

评测完成：2026-09-07 19:12:58（Asia/Shanghai）。运行代码提交：`426fe4e9b0946e27e5fb1f489691c607e39dd815`。

| EMA checkpoint | WER ↓ | SPKSIM ↑ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 150k | 0.05887 | 0.59178 | 0.76320 | 0.50861 |
| 200k | 0.05088 | 0.59201 | 0.74554 | 0.50882 |
| 200k − 150k | -0.00799 | +0.00023 | -0.01766 | +0.00021 |

200k 的 WER 从 140/2378 降至 121/2378，相对降低 13.57143%，绝对降低 0.79899 个百分点。
150k 的词级替换/删除/插入为 57/65/18，200k 为 53/55/13。
200k 的 SPKSIM 和 AVSync 数值略高，但变化很小；EMOSIM 则更低。
优先台词准确率时可选择 200k；优先当前 EMOSIM 时 150k 更合适。
这是同一次 TPCA 训练的 checkpoint 比较，不能单凭此结果归因于 TPCA 或证明其优于原 D1。
未进行多 seed、统计显著性检验或人工听评。

## 测试协议

- 配置：`src/aligndit/config/finetune_celebvdub_mm_d1_hunyuan_tpca.yaml`，Hydra 展开继承配置。
- `celebvdub_test_s1.lst` 全部 213 条；seed 0；EMA；FP32；Euler/EPSS；32 NFE；sway=-1；CFG text/video=5/2。
- 真实目标时长；参考音频使用同一测试片段 GT 音频；推理提供 prompt/target 文本边界。
  因而结果属于既有 Setting 1，不应当作跨句参考协议结果。
- HiFi-GAN 16 kHz：`../hifigan_16k_LRS3/g_01000000`。
- SPKSIM：既有 WavLM/ECAPA；WER：本地 faster-whisper-large-v3、英语、beam=5，沿用既有文本规范化。
- EMOSIM：既有 emotion2vec_plus_large scores 余弦相似度。
- AVSync：生成音频+对应嘴部视频提取 AV-HuBERT 特征，与 GT AV 特征逐帧余弦相似度后取均值。
  使用 `data/CelebVDub/video_mouth/test/test` 的对应嘴部视频。
- SPKSIM、EMOSIM、AVSync 为样本均值；WER 为语料级总词错误数/总参考词数，不平均逐句 WER。
- WAV、AV 特征及四项指标均核对完整 `test/视频ID/clipID`：每组均 213 条，无遗漏、重复或额外样本。
  WAV 为有效单声道 16 kHz；AV 特征形状与 GT 一致；所有音频、特征和指标为有限值。
  四项指标独立复算与输出汇总相符。

## Checkpoint 与产物

Checkpoint 目录：
`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_TPCA_6MM12A_CTC6_12_finetune_hifigan_16k_CelebVDub_char`

分别加载 `model_150000.pt`、`model_200000.pt`。检查文件内 `update` 与 EMA `transformer.tpca_step`，
均与 150000/200000 相符；推理严格加载 EMA state dict，并再次断言加载后的 TPCA step。

- [150k 产物](results/finetune_celebvdub_mm_d1_hunyuan_tpca_150000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/)：生成 WAV 在 `test/`，AV 特征在 `avhubert_feat/test/`，四份 `_*_results.jsonl` 含逐条指标与完整样本 ID；`verified_summary.json` 保存复算结果。
- [200k 产物](results/finetune_celebvdub_mm_d1_hunyuan_tpca_200000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur/)：生成 WAV 在 `test/`，AV 特征在 `avhubert_feat/test/`，四份 `_*_results.jsonl` 含逐条指标与完整样本 ID；`verified_summary.json` 保存复算结果。

日志：`logs/eval_tpca/150000.log`、`logs/eval_tpca/200000.log`；均有 `COMPLETE` 标记。
运行产物、日志、数据软链接和 checkpoint 不加入 Git。

## 复现与续跑

在本项目目录执行；`data` 指向既有 `/zjw524/projects/data` 数据目录（可按 `ROOT_PREFIX` 调整）。

```bash
setsid env INFER_GPUS=0,1 EVAL_PORT=29581 PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/run_celebvdub_s1_tpca.sh 150000 > logs/eval_tpca/150000_rerun.log 2>&1 < /dev/null &
setsid env INFER_GPUS=2,3 EVAL_PORT=29582 PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/run_celebvdub_s1_tpca.sh 200000 > logs/eval_tpca/200000_rerun.log 2>&1 < /dev/null &
```

每组两卡推理、第一张卡单进程逐项评测；脚本只跳过通过逐样本验证的已有阶段。
两组本次完整推理分别约 4.4 分钟，随后执行四指标及特征提取。
