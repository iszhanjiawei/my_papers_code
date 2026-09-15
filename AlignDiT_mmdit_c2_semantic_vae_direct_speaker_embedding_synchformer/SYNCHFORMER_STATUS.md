# Synchformer 实施与验证状态

更新日期：2026-09-15（本机 Asia/Shanghai）。

## 已完成

- 独立复制 234 个原项目文件；功能修改均位于 `_speaker_embedding_synchformer` 新目录。
- 官方 Synchformer 权重下载完成：950,058,171 字节，SHA256 与官方一致。
- 全部 79,826 个源视频存在，源检查结果为 79,613 train / 213 test，零缺失、零无效文件。
- 随机及固定抽查 520 个视频的流时长，与音频清单的最大差异为 0.0386875 秒。
- 变长条件、有效长度插值、padding、三路 CFG、activation checkpointing、梯度和 ODE 推理测试通过。
- 新旧模型公共参数在相同初始化种子下逐项完全一致。
- 6 项 dataset / S1 推理接线测试通过，8 项视频提取 / 缓存校验测试通过（包括并行启动元数据竞争回归）。
- 首批 8 个真实视频使用官方权重提取成功，8 个缓存均通过严格校验。23.66 秒视频得到 `[584, 768]`，未截断为 15 秒。
- 使用真实 S2c 70k 父 checkpoint 和两条真实样本进行 bf16 前向、反向验证，未执行 optimizer update。
- 独立只读代码复查未发现需要修复的额外问题。

## 真实父权重集成测试

父权重 SHA 验证通过。313 个源键中精确迁移 303 个，按原规则忽略 10 个；新模型共 710 个键，407 个保持初始化，其中 6 个为新增同步分支参数。

两条样本的 latent 长度为 `[126, 98]`，同步特征长度为 `[64, 48]`。

| CTC 权重 | 总 loss | flow loss | CTC loss | 同步输出投影梯度范数 | GPU 峰值 GiB |
|---|---:|---:|---:|---:|---:|
| 0 | 1.50622 | 1.50622 | 未启用 | 50.45190 | 2.56694 |
| 0.03 | 1.78564 | 1.50622 | 9.31390 | 50.64784 | 2.59075 |

以上是一次无参数更新的工程验证，不是训练或生成质量评测结果。

另一次接近正式 batch 大小的验证选中真实长度 `[1199, 1190, 1188]`，有效 3577 帧、padding 后 3597 帧。额外驻留 4.88295 GiB 的字节缓冲，估计 AdamW 两份 FP32 状态、EMA 副本与 DDP 梯度 bucket。保持 `checkpoint_activations=false` 时，CTC 0 / 0.03 的前后向均通过，GPU allocated 峰值分别为 15.57293 / 15.57117 GiB，reserved 峰值为 16.30469 / 16.39648 GiB。无参数更新。该容量估计不包含 NCCL、optimizer step 临时张量或 DDP bucket 重建峰值，正式启动仍需核验完整更新。日志：`logs/test_synchformer_real_parent_3600_reserved.log`。


## 全量数据与训练

全量目标为 79,826 条。首次长时间提取出现 PyAV 11 解码资源循环引用及 glibc 内存保留，导致 worker 被 OOM kill；已停止该轮进程并保留 45,379 个已落盘缓存文件，续跑时逐项校验复用。修复包括显式关闭 decoder/codec、每 16 个新提取视频 GC/trim，以及 worker 异常、单进程 RSS 6144 MiB 和 cgroup 90% 内存保护。正常退出、异常退出、RSS 超限和 supervisor 收到 SIGTERM 的实际子进程验证均通过。

修复后连续 300 条真实视频提取通过：最终 RSS 1820.35 MiB，10 Hz 观测峰值 2591.11 MiB，8 条已有真实缓存与修复后结果逐项完全一致。已于 12:06 左右以 4 张 GPU、24 个受监督 worker 续跑（独立 session PID 143750）。完成后的 `coverage_report.json` 必须包含 79,826 条有效记录且无失败，随后运行 `scripts/preflight_synchformer.py` 才能确认训练准备完成。

正式训练尚未启动。本文件会在全量检查及启动完成后更新实际状态。

## 复现检查

在本目录执行：

```bash
export PYTHONPATH=src
PYTHON_BIN=/zjw524/ENTER/envs/aligndit/bin/python
"$PYTHON_BIN" -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_synchformer.py
"$PYTHON_BIN" -u scripts/test_synchformer_pipeline.py
"$PYTHON_BIN" -u scripts/test_synchformer_cache.py
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_synchformer_real_parent.py
```

最后一条默认要求完整缓存。初期两样本验证使用 `--partial-cache`，该选项仅属于无参数更新的 smoke test，正式训练的完整覆盖检查没有放宽。

实际日志位于 `logs/test_synchformer_model.log`、`logs/test_synchformer_real_parent.log`、`logs/extract_synchformer_first8.log`、`logs/synchformer_extraction_guarded/`、`logs/synchformer_rss_stress.json`；运行产物不加入 Git。
