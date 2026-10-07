# Setting 2 — 真实台词条件的配音测试集

该目录可随本仓库 `git pull` 完整获取，用于其他服务器 / 其他模型的配音推理与评分。
包含 **115 条目标、83 条独立跨句参考、21 个源视频、30 个自动推断说话人**。
目标视频为 25 fps，共 **13,283 帧 / 531.32 秒（约 8.86 分钟）**。
使用本地固定 CelebVDub Setting 2 配对，参考身份为自动筛选结果，尚未人工核验。

本任务的输入条件是 **目标静音视频 + 真实目标台词 + 明确配对的另一句话参考音频**。
参考转录也提供；模型若不使用参考转录，应在结果中说明其实际条件。
所有文件均为真实副本，清单路径相对此目录；不依赖原服务器的绝对路径、软链接或 VSR 缓存。

## 获取与检查

在另一台服务器的 `my_papers_code` 仓库根目录执行：

```bash
git pull origin main
python Setting2-Dubbing-test/verify_dataset.py
python Setting2-Dubbing-test/prepare_inputs.py
```

两个脚本只需要 Python 标准库，无需 GPU 或本机原环境。
校验成功应显示 115 条目标、83 条参考与 `complete=true`。
第三条命令在 `Setting2-Dubbing-test/resolved/` 生成当前服务器的绝对路径清单，方便已有模型入口使用。
也可通过 `--output-dir <目录>` 改变解析后的输出位置；生成的 `resolved/` 不提交 Git。

## 清单与文件

| 文件 | 用途 |
|---|---|
| `inference.jsonl` | 通用配音输入；明确包含 `target_text`，没有目标 GT 音频或联合 GT AV 特征 |
| `evaluation.jsonl` | 评分输入；目标真实音频、真实转录、跨句参考、GT AV 特征与原始视频 |
| `aligndit_inference.jsonl` | 当前 AlignDiT GT 台词入口的兼容输入；目标台词由 `--target-text-dir` 读取 |
| `pairs.jsonl`、`targets.lst` | 冻结的目标 / 参考映射、来源哈希和完整目标 ID |
| `metadata.json`、`checksums.json` | 测试集统计与逐文件 SHA256 |
| `feature_spec.json` | 可选 AV-HuBERT 特征的用途与格式 |

媒体 / 文本位置：

- `media/silent_video/test/<视频ID>/<片段ID>.mp4`：完整目标静音视频。
- `media/mouth_video/test/<视频ID>/<片段ID>.mp4`：96×96 静音嘴部视频。
- `media/original_audio/test/<视频ID>/<片段ID>.wav`：115 条目标真值与 83 条参考的原始音频并集。
  **生成时仅使用清单的 `reference_audio`，目标音频只用于评分。**
- `text/test/<视频ID>/<片段ID>.txt`：真实目标与参考台词。
- `features/video/`：video-only AV-HuBERT 25 Hz / 1024D；供使用该前端的模型选择。
  其他模型可以从目标静音视频自行预处理。
- `features/gt_av/`：目标 GT 联合音视频特征，只供原 AVSync 指标评分。
- `media/original_video/`：原始脸部视频，只在评分清单提供，可用于 SyncNet / LSE。
  它保留原音轨；评分应明确指定模型生成音频，不把该视频作为配音生成输入。

## 接入其他模型

每条 `inference.jsonl` 有：

- `id`、`reference_id`：唯一完整路径 ID 与不同语句参考 ID；
- `video`、`mouth_video`：静音视觉输入，按模型需求选用；
- `target_text`、`target_text_path`：**真实目标台词**；
- `reference_audio`、`reference_text`：跨句参考音频与参考转录；
- `fps=25`、`num_frames`、`duration_seconds`、`target_samples_16khz`：目标视频长度信息。

将这些字段映射到各模型自己的接口即可。相对文件路径以本目录为根解析，或使用 `resolved/inference.jsonl`。
输出统一保存为 `<generated-root>/<id>.wav`，例如 `test/-2KGPYEFnsU/3_1.wav`。
不得只用片段 basename 命名；应去掉参考 prompt 段，保留目标配音，采用相同的视频时长约束。
正式对比覆盖完整 115 条，记录模型、权重、种子、采样参数和实际使用的输入条件。

模型权重、前端、vocoder 和评分器环境由各模型单独提供，这些不属于测试集数据。
本目录不包含某模型的生成样本，也不读取伪文本。

## 接入当前 AlignDiT 的真实台词入口

当前入口会从台词目录读取目标文本，使用 `aligndit_inference.jsonl`，不要传入通用清单。
在模型项目中调用 `aligndit.script.eval.infer_celebvdub_setting2_dubbing` 时设置：

```text
--manifest <本目录绝对路径>/aligndit_inference.jsonl
--target-text-dir <本目录绝对路径>/text
--reference-cache <新参考缓存目录>
--output-dir <新生成目录>
```

`--manifest` 直接指向根目录相对路径版，便于入口通过同目录的 `checksums.json` 校验视觉特征。
当前模型的依赖、权重与 decoder 另按模型运行说明配置，不受数据集迁移影响。

## 评分与可比性

`evaluation.jsonl` 中的目标音频 / 转录只用于评分。沿用此前四指标时：

- WER：统一 ASR / 文本规范化，全部词级编辑错误 / 1,721 个规范化参考词 ×100%；不平均逐句 WER。
- SPKSIM：生成语音与明确跨句参考的 WavLM-large + ECAPA cosine；按样本平均。
- EMOSIM：生成与目标 GT 的 emotion2vec 分类分数向量 cosine；按样本平均。
- AVSync：生成联合 AV-HuBERT 特征与目标 GT 特征的逐帧 cosine，先句内平均再样本平均；非 SyncNet 分数。

要为已生成完整 115 条音频导出 SyncNet / LSE 配对，可执行：

```bash
python prepare_inputs.py --generated-root /path/to/generated
```

它会检查全部生成 WAV 是否存在，并写出 `resolved/lse_manifest.jsonl`，每行明确指定原始脸部视频
和生成音频路径。LSE 检测器 / 权重 / 环境仍按对应独立评分器安装。

协议名称为 `CelebVDub_Setting2_GTText_Dubbing_v1`。
这是输入真实台词的 CelebVDub 配音适配测试集；与使用 VSR 预测文本的 Video-to-Speech 结果分别报告。
