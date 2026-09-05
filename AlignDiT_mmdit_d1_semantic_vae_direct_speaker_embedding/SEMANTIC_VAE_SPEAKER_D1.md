# D1 + Semantic-VAE + CAM++ speaker embedding

本目录独立复制自正在训练的 `AlignDiT_mmdit_d1_semantic_vae_direct`；复制基线 commit 为 `8ec0df3`。参考 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` 的 speaker 实现，所有修改只在本目录中进行。原 D1 VAE、C2 speaker 的源码和运行目录保持不变；不复制训练日志、数据、TensorBoard event 或 checkpoint，不建立跨实验源码软链接。

## 注入范围：跟随 D1 的 audio-only 分段

C2 的 audio-only 分段只有最后 6 层，D1 则是后 12 层。这里移植的是“向 audio-only 分段注入 speaker 条件”的方法，**不能照搬 C2 的固定起始层 12**。

| D1 层号（从 1 开始） | 类型 | Speaker 条件 | CTC 位置 |
| --- | --- | --- | --- |
| 1–6 | audio/video MM + text cross-attention | 不注入 | 第 6 层之后 |
| 7–12 | audio-only | 注入 | 第 12 层之后 |
| 13–18 | audio-only | 注入 | 无新增 CTC 头 |

因此配置为 `speaker_condition_start_layer: 6`（零基索引），覆盖 blocks `6..17`；`n_mm_layers=6`、`n_text_layers=6`、CTC taps `[5,11]` 保持 D1 原设置。第二个 CTC 头处于 speaker-conditioned 分段，不能将其描述为完全不受 speaker 条件影响。模型最终输出归一化的 timestep conditioning 仍使用原始 `t`，不添加 speaker delta。

## Speaker 条件与保持不变的设置

- 读取固定的双语 CAM++ 缓存：每条原始完整、未掩码 waveform 对应一个 L2-normalized `float32[192]` 向量，不在线训练 speaker encoder，不从 VAE 重建音频提取向量。
- 通过零初始化、无 bias 的 `Linear(192,768)` 得到 speaker delta，加入上述 12 个 audio-only block 的 timestep conditioning。新增 147,456 个可训练参数；不添加额外 speaker loss，也不加第二个零初始化 gate。
- Speaker dropout 与最终的 prompt-audio dropout 绑定；包括全条件 dropout。推理 full/TTS CFG 分支保留 speaker，null 分支去掉 speaker 和 prompt。多样本 CFG 的 timestep/mask/speaker 均采用一致的 branch-major 拼接顺序。
- 不移植 C2 的 12-MM 骨干、额外 text-context normalization，保留 D1 默认 audio-only text cross-attention 模式。
- 保持 64D/40Hz Semantic-VAE latent、1024D/40Hz 视频、`audio_video_ratio=1`、CTC strides `[1,1]`、固定 LibriSpeech-train 通道归一化及全部 79,613 条训练记录。105 条 40Hz CTC 不可行记录仍通过 `zero_infinity=True` 保留 diffusion 学习。
- LR `5e-5`、LR warmup 20k、seed 666、rank RNG `666+rank`、bf16、每 GPU 3,600 latent frames、EMA beta 0.999。参照 speaker 实验在 update **200,000** 停止，但不缩短原 200-epoch LR scheduler 的时间跨度。

新配置：[finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup.yaml](src/aligndit/config/finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup.yaml)，继承本快照内的 D1 VAE 配置。

```yaml
speaker_dim: 192
speaker_condition_start_layer: 6
ctc_lambda: 0.03
ctc_warmup_start: 10000
ctc_warmup_end: 30000
```

CTC 权重在 updates 1–10000 为 0；随后线性增加，update 20000 为 0.015，update 30000 达到 0.03。`total_loss = diff_loss + effective_ctc_lambda × mean(two_CTC_losses)`。早期不计算 raw CTC，日志中的 `ctc_weighted_loss=0` 不意味着已计算的 raw CTC 恰好为 0。

## 缓存、初始化和恢复约束

Speaker cache：

```text
${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/campplus_spk_emb_zh_en_16k
```

固定 encoder ID 为 `iic/speech_campplus_sv_zh_en_16k-common_advanced`，checkpoint SHA256 为 `92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199`。缓存 metadata/coverage 必须标记完整 79,826 条（79,613 train + 213 test）；实际读取时逐条校验 shape、dtype、有限值和单位范数。正式运行将 metadata/coverage 的哈希以及解析后的配置记录到 `speaker_training_contract.json`。

新训练从与父 D1 VAE 相同的固定 **S2c LibriSpeech 70k EMA** 初始化，不从正在跑的 D1 child checkpoint 续训。严格迁移加载 303 个兼容张量、忽略相同的 10 个 HuBERT projector 张量；目标 state dict 为 560 个键，257 个新键保持初始化，其中包含全零 speaker projection。新 optimizer/scheduler/child update 从 0 开始；EMA 内部 bookkeeping 延续父模型计数。

默认 checkpoint 目录：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char
```

编号 checkpoint 每 50k 保存一次，`model_last.pt` 每 5k 和训练结束保存。Speaker contract 与 `ctc_schedule.json` 会拒绝不兼容配置或缺少配置身份记录的恢复。正常恢复使用已完成的 child update；在已达到 200k 的 checkpoint 上不会再执行训练 update。重新比较必须选择新的输出目录，不删除旧实验。

## 启动与 TensorBoard

在本目录中，先检查空闲 GPU、端口及独立输出目录。后台启动必须使用 `setsid`，日志文件选择未使用的名称：

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_speaker.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

训练入口固定本快照 `PYTHONPATH=src`，使用环境 Python `-u` 和 GPU 0–3。默认 rendezvous port 29588，可用 `MAIN_PROCESS_PORT` 覆盖。TensorBoard 使用当前环境支持的 `python -m tensorboard.main`，独立 session，默认端口 6007，可用 `TENSORBOARD_PORT` 覆盖；它的默认 logdir 是精确的单个 run：

```text
runs/AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_speaker_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

仅 global main 写 TensorBoard。Scalars 包括 `loss`、`diff_loss`、启用后的 `ctc_loss`、`ctc_lambda`、`ctc_weighted_loss`、`ctc_fraction_of_total`、`grad_norm/global`、`speaker_proj_grad_norm`、`speaker_proj_weight_norm` 和 `lr`。Loss 延续 rank-0 batch 的统计口径，不假装是所有 rank 的全局均值。

按上级 `AGENTS.md`，实际启动后还必须检查 SID/TTY、四个 worker、GPU、日志增长、event/scalar 更新、HTTP 和实际客户端转发链接。`127.0.0.1` 是服务器本机地址，不是已验证的远程转发地址；启动脚本打印 PID 本身也不代表这些检查已经完成。具体 PID、日志和端口应记录在 ignored 的 `logs/` 运行记录中，不写死成未来运行的状态。

## 验证与推理

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_speaker.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/audit_semantic_vae_speaker_cache.py --full-audit
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_speaker_real_parent.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/audit_semantic_vae_d1_speaker_s1.py

# 在对应 speaker checkpoint 生成之后运行完整 213 条评测。
bash src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae_speaker.sh 200000
bash src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae_speaker.sh last
```

验证需覆盖 audio-only 12 层的真实注入范围、零初始化时的基线一致性、prompt/speaker 联合 dropout、批量 CFG、两处 CTC 和 speaker projection 的反向梯度、父权重严格迁移及真实数据数值稳定性。正式训练的指标不能由 smoke test loss 推断。

推理沿用固定 Semantic-VAE decoder、反归一化与精确采样长度裁切，严格加载 speaker 版 EMA，保持 Setting 1 的 NFE 32 / seed 0 / text CFG 5 / video CFG 2，计算 WER、SIM、EMOSIM、AVSync。Speaker vector 来自 prompt 对应的原始 waveform cache。既有 CelebVDub Setting 1 使用同一 GT clip 作为 prompt 和 target，本次保留并在 inference summary 中说明；不能称为独立参考音频的验证。
