# AlignDiT 实验结果归档

本目录汇总当前项目已经完成、失败或正在进行的实验。文档中的下游生成指标统一使用
`[0, 1]` 小数表示，避免与百分数混用。

## 文件索引

- `实验结果总汇.md`：实验定义、完整结果表、跨实验分析、权重选择和未完成事项。
- `CelebVDub_Setting1_正式指标.csv`：CelebV-Dub Setting 1、213 条样本的机器可读结果。
- `D1_CTC_lambda_0.2_评测结果.md`：D1、CTC 权重 0.2 的 150k 与 185k 完整评测记录。
- `D1_CTC_lambda_0.3_评测结果.md`：D1、CTC 权重 0.3 的 150k 与 185k 完整评测记录。
- `Codec重建上限指标.csv`：同一测试集上的 codec 重建上限，不属于生成模型结果。
- `训练阶段与状态.csv`：Semantic-VAE scratch、warm-start、旧 S3 事故和新 S3 的状态。

## 证据等级

- `artifact_verified`：本服务器仍有原始评测日志或 JSON，已重新核对。
- `user_recorded`：来自用户保存的评测汇总截图；数值已按截图原样转录。
- `status_only`：只有训练日志/checkpoint 状态，尚无正式四指标评测。

不同证据等级不会改变数值本身，但论文出表前应优先保存每条样本的原始预测、逐样本指标、
评测脚本版本和 checkpoint SHA256。硬件标签只标识训练来源，不能单独证明性能差异由 GPU 型号造成。
