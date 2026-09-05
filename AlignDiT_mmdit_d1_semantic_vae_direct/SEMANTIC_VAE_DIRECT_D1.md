# D1 + Semantic-VAE 独立实验

从 `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` 的工作区源码复制，复制基线 commit 为 `6dc3291`。本目录独立保存源码、配置和入口，不导入其他实验快照的 Python 源码。原 D1 和参考目录 `AlignDiT_mmdit_c2_semantic_vae_direct` 均不在本次修改范围内。保留的旧 mel/LRS3 配置和脚本仅用于追溯，请使用下述新入口。

## 模型与数据

保留 D1 的 18 层骨干：前 6 层 MM、后 12 层 audio；6 层 text stream，global text cross-attention，QK RMSNorm，CTC 位于第 6、12 层（零基索引 `[5, 11]`）。不移植 C2 的 12 层 MM 布局、额外 text-context normalization 或 speaker embedding。

参照 C2 的 Semantic-VAE 接法：

- 输入/输出由 80D mel 改成固定的 64D、40 Hz Semantic-VAE posterior-sample latent；16 kHz 音频每帧对应 400 个采样点。训练时读取缓存，不在线训练 VAE。
- 使用已经对齐到 40 Hz 的 1024D 视频缓存；`audio_video_ratio=1`、`video_rope_scaled=False`。
- 两个 CTC projector 均使用 `ctc_sampling_ratios=[1, 1]`，不再将 latent 降采样。原 mel D1 的默认值 `[2, 1]` 在旧入口保持不变。
- CelebVDub 全部 79,613 条训练记录保留；其中 79,508 条满足 40 Hz CTC 对齐条件，另外 105 条沿用 `zero_infinity=True`，只将不可行样本的 CTC 项归零，仍训练 diffusion 项。
- 采用与 C2 相同的固定 LibriSpeech-train 通道均值/标准差。配置固定 manifest、词表、归一化和预训练权重的 SHA256；不重算或覆盖缓存。
- 帧预算为每 GPU 3,600 帧，即 90 秒，等价于原 D1 的 9,000 mel 帧。其余主要优化设置保留：全局学习率 `5e-5`、LR warmup 20,000 updates、200 epochs、seed 666、EMA beta 0.999。

训练 latent 缓存位于 `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32`。推理解码器为该缓存绑定的冻结 Semantic-VAE 1000k decoder；加载时验证元数据、代码版本与权重身份。推理先反归一化，再解码，并按每条记录的 `original_num_samples` 裁回原始长度。

## 配置与 CTC 调度

唯一推荐配置：[finetune_celebvdub_mm_d1_semantic_vae_direct.yaml](src/aligndit/config/finetune_celebvdub_mm_d1_semantic_vae_direct.yaml)。

```yaml
ctc_lambda: 0.03
ctc_warmup_start: 10000
ctc_warmup_end: 30000
```

权重按本实验的 optimizer update 计算，不包括父模型预训练步数，且不是 epoch 或 microbatch 计数：

| 本实验 update | 有效 CTC 权重 |
| --- | ---: |
| 1–10,000 | 0 |
| 10,001 | 0.0000015 |
| 20,000 | 0.015 |
| 30,000 及以后 | 0.03 |

`total_loss = diff_loss + 有效权重 × ctc_loss`；`ctc_loss` 是两个 CTC 头的平均值。前 10,000 步不计算 CTC loss，仅训练 diffusion；此时不把未计算的 raw CTC 伪装成 0，而记录实际为 0 的 `ctc_weighted_loss` 和 `ctc_lambda`。

续训读取已完成的子实验 update，并校验 checkpoint 目录内的 `ctc_schedule.json`，禁止静默更换调度。新实验目录没有 checkpoint 时，从 update 0 开始；存在 checkpoint 则保留原训练器的恢复行为。若要另外重跑，指定全新的 `ckpts.save_dir`，不要删除或覆盖旧目录。

## 初始化与保存位置

使用与 C2 相同的 **S2c LibriSpeech 70k EMA** 作为音频预训练父模型：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt
```

不是从原 D1 的 mel checkpoint 续训（80D 与 64D 不兼容）。新 CelebVDub optimizer/scheduler/update 从 0 开始，EMA 内部计数按 C2 延续父模型的 70k bookkeeping。严格迁移加载 303 个兼容张量，忽略 10 个父模型 HuBERT projector 张量，256 个 D1 新增张量保持初始化；目标 state dict 共 559 个张量键。非预期键/形状会报错，迁移报告保存到 `parent_migration.json`。

默认 checkpoint 目录：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_ctc003_warmup10k30k_40hz_CelebVDub_char
```

每 50,000 updates 保存一个编号 checkpoint；每 5,000 updates 以及训练结束时保存 `model_last.pt`。所有 checkpoint、日志、推理输出均不提交 Git。

## 训练入口（本次尚未启动正式训练）

在本目录执行 `source env.sh` 后，四卡入口为：

```bash
bash src/aligndit/run/train/finetune_celebvdub_mm_d1_semantic_vae_direct_4x4090.sh
```

该脚本设置本快照的 `PYTHONPATH=src`，使用环境 Python `-u`、`PYTHONUNBUFFERED=1`、4 GPU bf16，并检查 GPU 占用。正式后台训练必须遵循上级 `AGENTS.md`：外层使用 `setsid`，日志选择新文件，启动后核对 SID、TTY、worker、GPU 和日志进展；不能用 `nohup`。

仅 global main 写 TensorBoard。相对于本目录的 logdir 为：

```text
runs/AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Scalars 包含 `loss`、`diff_loss`、`ctc_lambda`、`ctc_weighted_loss`、`lr`，CTC 启用后还有 `ctc_loss`。真正启动训练时，必须同时用 `setsid` 启动 TensorBoard，检查 event、端口和 HTTP，并按 `AGENTS.md` 报告当次真实转发地址；本说明不代表已有训练或 TensorBoard 服务运行。

## 推理与评测入口

```bash
# 在确实生成对应的新 D1 VAE checkpoint 后执行；不接受旧 mel 权重。
bash src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae.sh 150000
bash src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae.sh last
```

支持第二个参数指定新的输出目录，以及 `EVAL_GPU`、`CHECKPOINT_DIR`、`CONFIG` 等环境变量。拒绝非空输出目录以避免覆盖。`last` 的实际 update 从 checkpoint 读取，EMA 权重严格加载。保持 CelebVDub Setting 1 的 213 条样本、seed 0、NFE 32、text CFG 5、video CFG 2 和历史 prompt-text 拼接协议；依次计算 SIM、WER、EMOSIM、AVSync，并检查产物完整性。长时间评测也使用 `setsid`。

## 开发验证

可重复的小模型检查（不训练正式实验、不保存 checkpoint）：

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_direct.py
```

该检查覆盖 D1 层布局、40 Hz CTC、warmup 边界、loss 组合及反向梯度。小模型仅用于代码检查，不用于性能结论。正式训练后仍需使用上述完整评测入口取得 WER/SIM 等指标，不能将 smoke loss 当成收敛结果。

有上述缓存和父模型的机器还可运行完整尺寸的 bf16 检查（不执行 optimizer update、不写 checkpoint）：

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_direct.py --device cuda --real-data
```

2026-09-06 验证通过：CPU/CUDA 小模型检查；79,613 条训练 manifest 与固定归一化身份；303 个迁移张量逐值一致；包含短 CTC 可行/不可行样本和常规长度样本的完整尺寸 bf16 forward/backward；权重 0、0.015、0.03 的加权公式与两个 CTC 头梯度；2-step latent sampling 有限值及 prompt 原样保留。另已检查全部 213 条测试 latent/video 和参考资源；真实 26 帧 latent 经绑定 decoder 得到 10,400 个采样点，裁回原始 10,240 点，数值均有限。这些是接口与数值验证，不代表训练收敛或性能提升。
