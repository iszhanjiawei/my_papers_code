# CelebVDub Setting 2 测试集副本

该目录是完成本地 Video-to-Speech / Visual Forced Alignment 评测的固定 Setting 2 副本。
包含 **115 条目标、83 条独立跨句参考、21 个源视频、30 个自动推断说话人**。
目标视频为 25 fps、共 **13,283 帧 / 531.32 秒（8.86 分钟）**；参考必须是另一条语句。
身份经 CAMP++ 自动筛选，尚未人工核验；这是本地 CelebVDub 适配，非论文 LRS3 官方测试集。

## 包含的文件

- `manifest.jsonl`：完整目标、参考、GT 与配对信息；`pairs.tsv` / `targets.lst` 为配对与目标清单。
- `inference.jsonl`：只含目标视觉输入、实际 VSR 伪文本路径和显式参考输入，不含目标 GT 文本 / 音频字段。
- `evaluation.jsonl`：GT 音频、转录与评分输入；只用于评分。
- `media/silent_video/`、`media/mouth_video/`：实际使用的 115 条目标静音视频和嘴部视频。
- `media/target_audio/`、`media/reference_audio/`：原目录中物化的目标 / 参考音频副本。
- `media/original_audio/`：原始 GT / 参考音频，根清单引用这些与原清单逐字节一致的音频。
  部分参考原始采样率 / 声道不同于物化的 16 kHz 版本，两者分别保留。
- `media/original_video/`、`media/original_mouth_video/`：原始视频副本，保留原始数据。
- `text/`：目标和参考转录；`pseudo_text/`：此前测评实际使用的 115 条 LipVoicer 视觉识别预测。
- `features/video/`：目标 / 参考的 video-only AV-HuBERT 特征；`features/gt_av/`：目标 GT 联合特征。
- `audio_mono/`、筛选审计、WavLM 诊断及其余原目录元数据按原样保留。
- `provenance/source_*.jsonl` 与 `source_README.md`：原清单 / 说明的逐字副本。
  原筛选审计中的绝对路径是来源记录，不作为该副本的运行依赖。
- `bundle_summary.json`、`checksums.json`：统计、来源哈希和逐文件 SHA256。

原目录中的软链接均已展开为实际文件；原始 benchmark 目录未修改。
根目录三个 JSONL 的文件路径统一相对此目录，源资产逐字节核验；
VSR 伪文本从实际缓存补齐，原清单中未落地的伪文本占位路径已替换。
这些预测与 GT 转录分开保存，不允许以 GT 填充生成文本。

## 完整性检查与使用

在该目录执行（Python 标准库即可）：

```bash
python verify_bundle.py
python resolve_manifests.py
```

第二条命令在 `resolved/` 生成当前机器绝对路径版本，可传入此前的生成 / 评估适配器。
例如将 `resolved/inference.jsonl` 传给生成入口，`resolved/evaluation.jsonl` 传给评分入口；
VSR 目录设置为本目录 `pseudo_text/`。`resolved/` 属于本机临时输入，不提交 Git。
也可通过 `--output-dir <目录>` 指定解析后的输出位置。

原清单固定来源哈希与协议见 `bundle_summary.json`；模型权重、生成音频和评测结果不属于本测试集副本。
