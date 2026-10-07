# CelebVDub Setting 2：同一说话人的另一句参考（自动筛选 v1）

已从本机现有 Setting 1 的 213 个测试片段构造 **115 对**目标—参考配对，来自 21 个源视频、30 个自动说话人分组，使用 83 个不同参考片段。其余 **98 个目标**保留排除记录；没有用目标自身音频补位。

这是本项目构造的 **CelebVDub_Setting2_auto_v1**，不是官方发布的 CelebVDub Setting 2 清单，也不是人工核实身份的标注集。说话人身份通过独立的 CAMPplus 模型自动筛选；工程阈值没有经过人工身份标注校准。用于最终论文前，应试听 `review.html` 的配对并记录人工复核结果。不同源视频之间未合并身份，因此“30 个分组”不表示恰好 30 个不同自然人。

## 文件入口

| 文件 | 内容 / 用途 |
|---|---|
| `pairs.tsv` | 115 对目标、参考、自动说话人组、时长及筛选分数；方便直接查看 |
| `celebvdub_test_s2.tsv` | 两列 `target_id`、`reference_id`；保留 `test/` 前缀 |
| `manifest.jsonl` | 完整路径、参考文本、数据来源和审计字段 |
| `inference.jsonl` | **生成模型使用**：静音目标视频、嘴部视频、视频特征、另一句参考音频及其文本；不包含目标真值音频/文本字段 |
| `evaluation.jsonl` | **评测使用**：目标真值文本、目标真值音频、配对参考、GT AV-HuBERT 特征 |
| `targets.lst` | 与这 115 对对应的目标列表；不代表原始全部 213 条 |
| `excluded.jsonl` | 98 条排除记录 |
| `candidate_audit.jsonl` | 源视频内所有不同片段候选及逐项排除原因 |
| `summary.json` | 构造参数、模型来源、数量、文件哈希 |
| `review.html` | 本地试听目标/参考和查看静音目标视频的审阅页面 |
| `media/silent_video/` | 115 个完整目标静音视频，路径保留 `test/<folder>/<clip>.mp4` |
| `media/mouth_video/` | 115 个嘴部静音视频；25 fps |
| `media/target_audio/`、`media/reference_audio/` | 审阅用音频链接；原始文件未修改 |
| `media_audit.json` | 230 个视频均仅有视频流，尺寸、帧率和帧数保持一致的检查记录 |
| `wavlm_pair_audit.tsv` | 独立 WavLM 检查值，未用于选参考或调整保留集合 |
| `identity_sources.md` | 官方资料中身份分组方法和文件前缀的证据与限制 |

示例：

```text
target_id:     test/-2KGPYEFnsU/3_1
reference_id:  test/-2KGPYEFnsU/3_0
```

不要按下划线解析 YouTube/source-video ID。读取 TSV 两列，或直接读取 JSONL 的显式字段。

## 固定的构造规则

1. 目标池和参考池均来自 `/zjw524/projects/data/celebvdub_test_s1.lst` 的 213 个测试片段；没有从训练集借用参考。
2. 源视频文件夹只用于限制候选范围，不作为说话人身份标签。
3. 使用已有的 CAMPplus 192 维说话人特征，按源视频进行 complete-linkage 层次聚类，余弦阈值为 **0.65**。同一组中任意两句均需满足阈值；不会通过相似度链式传播把不同人合并。
4. 参考句必须是另一个片段，时长 **2–10 秒**，至少 3 个词；拒绝相同音频哈希、相同归一化转录，以及包含对方完整转录的至少 3 词片段。
5. 对潜在配对做波形重复检查：下采样到 4 kHz，扫描 0.5 秒窗口、步长 0.25 秒，与另一段音频各位置做归一化互相关；值达到 **0.90** 时排除。实际排除了 `-HWkUQJjVrI/11_2` 与 `12_1` 这一对的两个方向（约 0.935）。这不是原始时间戳无交叠的证明，也不保证发现任意变速/失真后的重复。
6. 在合格候选中采用固定种子 **20261003** 的 SHA256 排序选择参考，**不挑最高相似度**参考。参考允许被多个目标复用。
7. 保存官方训练源码使用的文件名前缀作为审计字段：49 对同前缀、66 对跨前缀。前缀不是已确认的人工身份标签，因此未单独以此决定保留。

98 条排除中，87 条在保守聚类下没有另一句同组参考，11 条的同组候选未通过时长/文本/重复等规则。详细候选依据保存在 `candidate_audit.jsonl`，没有静默漏掉样本。

独立 WavLM 检查的配对余弦均值约 **0.71719**，最低约 **0.54393**；这些是**真实目标与真实参考之间的数据审计值**，不是 AlignDiT 或其他模型生成结果的 spkSIM。WavLM 没有参与聚类、筛选或参考排名，避免用最终评测模型挑选测试对。

## 在 VTS 实验中如何使用

- AlignDiT 输入：`video_feature`、`reference_audio`、`reference_text`，以及从目标静音视频识别出的唇读伪文本。`pseudo_text_path` 是预留路径，目前并不表示伪文本已生成。
- LipVoicer 输入：`video`；其 VSR 预测应导出到同一份 `pseudo_text/`，供 AlignDiT 使用，以统一文本识别来源。它本身使用人脸信息估计音色，不需把参考音频强塞进其架构。
- Intelligible 输入：`mouth_video`，以及**从 reference_audio 提取**的 RTVC speaker embedding；不得用目标真值音频提 speaker embedding。
- 目标真值字幕只能用于 WER。不能把目标真实字幕写进 `pseudo_text_path`。
- WER = Whisper-large-v3 对生成音频识别后，与目标真值字幕比较的 corpus micro WER ×100。
- spkSIM = 生成音频与本清单中 **另一句参考音频**的 WavLM-large speaker-verification embedding 余弦相似度，再按样本平均。
- AVSync = 同一目标视频分别配真实音频与生成音频，提取联合 AV-HuBERT 特征，先按帧求余弦平均，再按样本平均。

所有模型必须使用相同 115 个目标和同一配对清单。若有模型失败，不应单独删掉失败样本后比较均值。若比较 Setting 1 与 Setting 2 的影响，也应重新在这 **115 个共同目标**上跑 Setting 1。

本集合没有提供训练集，也没有启动模型训练或完整 VTS 推理。CelebVDub 的数据转录本身可能含伪标注误差，WER 应理解为相对现有数据转录的误差。

## 复建与评分命令

```bash
cd /zjw524/projects/alignDiT_idea6/Video-to-Speech-benchmark
PY=/zjw524/ENTER/envs/aligndit/bin/python

# 原始数据只读；固定参数重新生成相同配对与路径。
$PY scripts/build_celebvdub_setting2.py
$PY scripts/materialize_setting2.py

# 查看数据，不加载模型。
head -n 5 setting2/pairs.tsv

# 生成音频必须保存为 $GEN/test/<folder>/<clip>.wav。
# 将 GEN 设置为已生成音频所在目录后再执行下面的评分。
GEN=/absolute/path/to/generated_wavs
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_setting2.py --generated "$GEN" --metric wer
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_setting2.py --generated "$GEN" --metric spksim
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_setting2_avsync.sh "$GEN"
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_setting2.py --generated "$GEN" --metric avsync
```

评分入口使用已有 `../AlignDiT_baseline` 的 WER/spkSIM 实现，并显式传入本机权重。AVSync 使用该 baseline 的 AV-HuBERT 提取器。缺少任一目标的生成音频/特征会报错，不会改动评测集合。新评分入口已完成语法和缺失文件预检查；尚未用模型生成结果完成端到端评分。

最终 `manifest.jsonl` SHA256：

```text
5d31f42d9766b4c500904dcca8e9a50230f5678744c57b49dc808e45accd7cf8
```

已复建验证 manifest 字节完全一致；记录在 `validation.json`。Intelligible 的 115 条模型输入已经另外准备到 `../prepared/IntelligibleL2S/setting2`，参考说话人 embedding 全部由本清单的另一句参考语音提取。此步骤没有调用第一阶段语音生成模型。
