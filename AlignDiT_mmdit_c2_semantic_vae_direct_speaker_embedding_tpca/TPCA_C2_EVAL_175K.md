# C2 Semantic-VAE + Speaker + TPCA：175k 评测

评测完成时间：2026-09-09（Asia/Shanghai）。

| EMA checkpoint | WER ↓ | SPKSIM ↑ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 150k | 0.06056 | 0.64852 | 0.76103 | 0.55667 |
| **175k** | **0.05467** | **0.65651** | **0.76680** | **0.56213** |
| 175k − 150k | **−0.00589** | **+0.00799** | **+0.00577** | **+0.00546** |

175k 在四项指标上均优于同一训练运行的 150k checkpoint；WER 相对下降约 9.7%。
这些差值描述同一模型在两个训练时点的结果，不能单独证明 TPCA 的因果收益。

175k WER 独立按语料级总编辑距离复算：参考词数 2,378，替换 47、删除 65、
插入 18，即 `(47 + 65 + 18) / 2378 = 0.05466779`。SPKSIM、EMOSIM 和
AVSync 均为 213 条样本分数的算术平均，未使用句级 WER 平均。

## 协议

- CelebV-Dub Setting 1 全部 213 条，使用相同 GT 片段作为说话人参考音频。
- 配置：`finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_tpca_ctc003_warmup.yaml`。
- 捕获 `model_last.pt` 在 update 175,000 的状态并加载 EMA 权重；checkpoint `update`、
  在线和 EMA `transformer.tpca_step` 均核对为 175,000。该状态的 SHA256 为
  `9e36bc6aadc84c90f345cf0f2da03be24ada239abfa2b2dfad7c4d405dfb2457`。
- seed 0、Euler/EPSS、32 NFE、sway -1、text/video CFG 5/2、GT 目标时长。
- 推理文本沿用历史 Setting 1 的 `prompt + 两个空格 + target`，TPCA 只对齐 target
  occurrence，排除 prompt 文本和分隔符。
- 64 维、40 Hz Semantic-VAE latent 使用固定 LibriSpeech train 归一化；输出由固定
  Semantic-VAE EMA decoder 解码。说话人条件读取完整、未遮挡的 GT 波形 CAM++ 缓存。
- AVSync 使用生成音频与对应嘴部视频重新提取 AV-HuBERT 特征，再与相应 GT
  AV-HuBERT 特征逐帧计算余弦相似度。

## 完整性检查

- 生成 WAV 213/213：完整相对 ID 与测试列表一致，无缺失、重复或额外样本；均为
  单声道 16 kHz、非空且数值有限。
- 生成 AV-HuBERT 特征 213/213：完整相对 ID 与测试列表一致，形状逐条匹配 GT，
  且所有数值有限。
- 四份指标文件各有 213 条 JSON 样本记录及一条汇总行；每条记录包含完整
  `test/video_id/clip_id`，四项指标均为有限值。
- 三项相似度均从逐样本记录独立复算；WER 从逐样本转录按语料级编辑数独立复算；
  四项结果均与文件末尾五位小数一致。
- 推理生成耗时 558.02 秒；评测与同期训练共享 GPU 1，未发生 OOM 或流水线错误。
  评测完成后训练仍正常运行，已继续至至少 update 176,668。

## 产物

评测所用 rolling checkpoint 状态：

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_40hz_CelebVDub_char/model_last.pt`

评测目录：

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_40hz_CelebVDub_char/eval_s1_175000_speaker_tpca_cfgv2.0`

目录中的 `inference_summary.json` 固化了 update 175,000 checkpoint SHA256、固定 decoder、
归一化、生成参数和每条 WAV SHA256，因此 rolling checkpoint 后续被训练覆盖时仍可核查
本次评测输入；`_*_results.jsonl` 保存四项逐样本结果。运行日志为
`logs/eval_175k/pipeline.log`。这些运行产物、模型和日志不提交 Git。

本结果仅比较同一 TPCA 训练运行的 150k 与 175k 状态。没有相同架构、相同初始化和
相同评测协议的无 TPCA 配对结果时，不能把分数差异直接归因于 TPCA。
