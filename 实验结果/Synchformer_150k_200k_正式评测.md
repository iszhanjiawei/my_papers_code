# Semantic-VAE Direct-C2 + CAM++ + Synchformer 正式评测

评测日期：2026-09-20。

## 结果

| Checkpoint | SPKSIM↑ | WER↓ | EMOSIM↑ | AVSync↑ |
|---:|---:|---:|---:|---:|
| 150k | 0.65647 | **0.05635** | 0.75623 | 0.56146 |
| 200k | **0.65950** | 0.06013 | **0.76175** | **0.56737** |

200k 相比 150k 的 SPKSIM、EMOSIM 和 AVSync 分别提高 `0.00303`、`0.00552` 和
`0.00592`，但 WER 增加 `0.00378`。因此 200k 是身份、情感和同步指标优先时的综合选择；
若优先考虑内容正确性，则 150k 更合适。两者形成清晰的 Pareto 权衡，不能描述为 200k
四项全面优于 150k。

## 评测协议

- CelebV-Dub Setting 1，同一份 `213/213` 测试列表；测试列表 SHA256 为
  `1ad609deefaf1293e00b168a3418611ef779c4e9b49f3b51539721a8360a1e49`。
- 使用 checkpoint 的 EMA 权重；150k SHA256 为
  `6fc35052b423774624d2ec9817878ba6a654ad0c21adb0d99ced22f8ec83fa37`，200k SHA256 为
  `78d6cb0d5dafb15f90e6b59c1cbf95a2aa0fdfee2b302686b0cbb317245ed7fe`。
- seed 0、Euler/EPSS、32 NFE、sway `-1`、文本 CFG 5、视频 CFG 2、真实目标时长。
- Setting 1 的 prompt 音频、CAM++ speaker embedding 与目标来自同一 GT 片段；
  Synchformer 条件来自目标 RGB 视频。该协议不是独立参考片段的 voice-cloning 测试。
- Semantic-VAE 1000k EMA decoder，输出为 16 kHz PCM WAV。
- 两组均完成 213 条 WAV 与 213 条 AV-HuBERT 特征；四项结果 JSONL 各含 213 条样本。

## 独立复算

SPKSIM、EMOSIM 和 AVSync 使用 213 条逐样本值的算术平均。WER 使用所有句子的词级编辑距离
总和除以参考词总数。独立复算得到：

| Checkpoint | SPKSIM | WER | EMOSIM | AVSync |
|---:|---:|---:|---:|---:|
| 150k | 0.6564716047 | 0.0563498738 | 0.7562288853 | 0.5614563264 |
| 200k | 0.6595000828 | 0.0601345669 | 0.7617452559 | 0.5673745641 |

五位小数舍入后与评测脚本写入的汇总行完全一致，全部逐样本值均为有限值。

## 证据目录

- 150k：`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_synchformer_cfgv2.0/`
- 200k：`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_200000_speaker_synchformer_cfgv2.0/`
- 运行日志：`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_synchformer/logs/eval_synchformer_150k_200k/`

每个评测目录均保留 `inference_summary.json`、生成 WAV、AV-HuBERT 特征及
`_sim_results.jsonl`、`_wer_results.jsonl`、`_emosim_results.jsonl`、`_avsync_results.jsonl`。
这些运行产物不加入 Git。
