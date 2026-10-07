# AlignDiT baseline：Setting 2 真实台词配音评测

输入为 `Setting2-Dubbing-test/` 冻结的 115 条目标视频特征、真实目标台词和明确配对的
跨句参考音频 / 转录，共 83 条独立参考。推理不读取目标波形或目标联合 AV 特征，
参考身份仍是自动筛选结果，尚未人工核验。

使用仓库外 `aligndit-baseline_1003/alignDiT_baseline/AlignDiT` 的原模型及训练配置，
严格加载 `aligndit-baseline_1003/ckpts/model_{100000,150000,200000}.pt` 的 EMA。
字符词表、16 kHz HiFi-GAN、原始参考 Tacotron mel、reference visual zeroing、
视频时长、seed 0、float32、32 NFE Euler/EPSS、sway=-1、CFG text/video=5/2 保持既有配置。

此入口由 benchmark 的已审计 baseline 推理脚本派生，保留 EMA 加载、采样、声码器、
参考音量处理和 prompt 裁剪；目标文本改为显式真实台词，并验证数据包 SHA256。
与历史 VTS 的识别文字输入分别保存。只提交评测代码与实验文档，不提交运行产物。

在 `my_papers_code` 根目录运行三个权重的完整流水线：

```bash
ROOT_PREFIX="${ROOT_PREFIX:-}"
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
LOG="../Video-to-Speech-benchmark/logs/setting2_gttext_dubbing_baseline/pipeline.log"
mkdir -p "$(dirname "$LOG")"
setsid env PYTHONUNBUFFERED=1 "$PYTHON" -u evaluation/setting2_dubbing/run_baseline.py \
  > "$LOG" 2>&1 < /dev/null &
BASELINE_DUBBING_PID=$!
ps -o pid,ppid,sid,tty,stat,comm -p "$BASELINE_DUBBING_PID"
```

单独生成指定权重或预检（将 `--dry-run` 去掉即生成）：

```bash
"$PYTHON" evaluation/setting2_dubbing/infer_aligndit_baseline.py \
  --checkpoint ../aligndit-baseline_1003/ckpts/model_150000.pt \
  --expected-update 150000 \
  --output-dir ../Video-to-Speech-benchmark/results/setting2_gttext_dubbing/AlignDiT_150k \
  --dry-run
```

`--manifest` 默认 `Setting2-Dubbing-test/aligndit_inference.jsonl`，`--target-text-dir`
默认该包的 `text/`。包内路径、文字和视频特征必须通过其 `checksums.json`。
采样器内部持有 GPU 0 锁，不要再套外层 flock。完整流水线依次完成生成、四指标和独立复算。

评分复用 `../Video-to-Speech-benchmark/scripts/run_setting2_evaluation.py --with-emosim`：
faster-whisper-large-v3 corpus WER（百分数，1721 参考词）、WavLM 跨句参考 SPKSIM、
emotion2vec 对目标 GT 的分类分数 EMOSIM、AV-HuBERT 生成 / GT 联合特征 AVSync。
后三项取样本均值；AVSync 不等同于 SyncNet / LSE 时间同步指标。

产物为 benchmark 下 `results/setting2_gttext_dubbing/AlignDiT_{100,150,200}k/`，
每组含 `test/` WAV / 输入哈希、`run_config.json`、`avhubert_feat/`、四份
`setting2_*.json` 与 `evaluation_verification.json`。流水线状态 / 汇总分别为
`baseline_pipeline_progress.json` / `baseline_summary.json`；日志为
`logs/setting2_gttext_dubbing_baseline/`。

全部结束后可再次核验：

```bash
"$PYTHON" evaluation/setting2_dubbing/verify_baseline.py
```

移机时模型源码、checkpoint、词表、声码器、统一评分代码和评分模型须另行准备；
Git 中的测试数据包本身可独立校验与解析，使用方式见 `Setting2-Dubbing-test/README.md`。
