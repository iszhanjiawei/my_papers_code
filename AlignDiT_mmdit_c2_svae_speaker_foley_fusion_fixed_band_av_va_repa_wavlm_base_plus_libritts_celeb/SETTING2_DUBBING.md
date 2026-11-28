# Setting 2：输入真实台词的配音评测

本入口使用目标静音视频、**目标真实台词**、同说话人另一句话的参考音频及参考转录。
沿用冻结的 115 条目标 / 83 条跨句参考；目标 GT 音频、目标声学 latent、目标 speaker
embedding 与 REPA teacher 均不进入生成，GT 音频只用于评分。

这与 `infer_celebvdub_setting2.py` 的 VTS 入口不同：后者以 VSR 预测作为目标文本。
两个入口保留独立的输出和参考缓存，真实台词条件的结果标记为 `Setting2-GTText-Dubbing`。

## 在当前机器运行

在本项目根目录执行（默认 EMA 200k，完成生成及 WER / SPKSIM / EMOSIM / AVSync）：

```bash
setsid env DUBBING_STEP=200000 PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/eval/infer_celebvdub_setting2_dubbing.sh \
  > setting2_gttext_200k.log 2>&1 < /dev/null &
```

`DUBBING_STEP=100000` / `150000` 选择对应权重。`RUN_METRICS=0` 仅生成，
`EVAL_GPU=0` 指定生成 GPU。默认固定 EMA、seed 0、float32、32 NFE、Euler/EPSS、
sway −1、文本 / 视频 CFG=5/2。时长由 25 fps 目标视频帧数决定。

默认输入来自兄弟目录 `../celebvdub_setting2/`：

- `inference.jsonl`：目标视频特征与明确的跨句参考信息；
- `text/test/<video>/<clip>.txt`：真实目标台词。由 `--target-text-dir` 指定，完全不读取 VSR。
- `checksums.json`：校验复制后的 video-only 特征，保留原始视觉条件的来源契约。

Python 入口为 `aligndit.script.eval.infer_celebvdub_setting2_dubbing`，支持 `--manifest`、
`--target-text-dir`、`--output-dir`、`--reference-cache` 与 `--dry-run`。
无需将真实文本写入伪文本缓存，也无需修改原 VTS 入口。

适配器复用固定 SHA256 的原生成实现，仅替换目标文本来源、便携视觉特征校验、默认路径与
真实台词条件的元数据；采样、参考提取、归一化、模型严格 EMA 加载与 decoder 不变。
共享实现变化时会明确报错，需重新审计，避免静默改变协议。

## 输出与评分

输出分别为仓库外 benchmark 的
`results/setting2_gttext_dubbing/{Ours_100k,Ours_150k,Ours_200k}/`，
保存 16 kHz 单声道 WAV、逐条 GT 台词与参考哈希、`run_config.json`、生成覆盖记录和四项指标。
参考 posterior latent / CAM++ 缓存单独保存，不使用历史同句参考或目标声学数组。

WER 为 corpus 编辑总数 / 1,721 个参考词 ×100%；其他三项按完整 115 条取样本均值。
SPKSIM 对跨句参考音频，EMOSIM 对目标 GT 情感分类分数，AVSync 为 AV-HuBERT 帧余弦，
不是 SyncNet / LSE-D。参考说话人身份为自动筛选结果，尚未人工核验。

三权重批量入口：`Video-to-Speech-benchmark/scripts/run_setting2_gttext_dubbing.py`。
最终报告为 `reports/setting2_gttext_dubbing.md`，机器汇总为
`results/setting2_gttext_dubbing/summary.json`。日志、生成音频、缓存与评分不提交 Git。
