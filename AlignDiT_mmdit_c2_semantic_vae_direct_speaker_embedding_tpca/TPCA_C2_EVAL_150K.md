# C2 Semantic-VAE + Speaker + TPCA：150k 评测

评测完成时间：2026-09-09（Asia/Shanghai）。

| EMA checkpoint | WER ↓ | SPKSIM ↑ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 150k | 0.06056 | 0.64852 | 0.76103 | 0.55667 |

WER 独立按语料级总编辑距离复算：参考词数 2,378，替换 54、删除 75、插入 15，
即 `(54 + 75 + 15) / 2378 = 0.06055509`。SPKSIM、EMOSIM 和 AVSync 均为
213 条样本分数的算术平均，未使用句级 WER 平均。

## 协议

- CelebV-Dub Setting 1 全部 213 条，使用相同 GT 片段作为说话人参考音频。
- 配置：`finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_tpca_ctc003_warmup.yaml`。
- 加载 `model_150000.pt` 的 EMA 权重；checkpoint `update`、在线和 EMA
  `transformer.tpca_step` 均核对为 150,000。
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
- 四项汇总均从逐样本记录独立复算，并与文件末尾五位小数一致。
- 推理生成耗时 561.60 秒；评测与同期训练共享 GPU 1，未发生 OOM 或流水线错误。

## 产物

Checkpoint：

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_40hz_CelebVDub_char/model_150000.pt`

评测目录：

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_tpca_cfgv2.0`

目录中的 `inference_summary.json` 记录 checkpoint SHA256、固定 decoder、归一化、
生成参数和每条 WAV SHA256；`_*_results.jsonl` 保存四项逐样本结果。
运行日志为 `logs/eval_150k/pipeline.log`。这些运行产物、模型和日志不提交 Git。

本结果仅描述当前 TPCA 150k checkpoint。没有相同架构、相同初始化和相同评测协议的
无 TPCA 配对结果时，不能把该分数差异直接归因于 TPCA。
