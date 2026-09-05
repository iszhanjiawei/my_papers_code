# 原生 mel500k 与 Semantic-VAE S2c70k 纯音频对照评测

## 任务与边界

用户要求实际比较两条完整原生推理链路，而不是把两个不同空间的 MSE 直接比较：

| 对照 | 权重与原生链路 |
|---|---|
| 原始模型 | LibriSpeech mel500k EMA → 80D/100 Hz mel → HiFi-GAN 16 kHz |
| 适配模型 | Semantic-VAE warm-start S2c70k EMA → 64D/40 Hz 标准化 latent → LibriSpeech train mean/std 反归一化 → Semantic-VAE 1000k decoder |

本任务只评测，不启动、恢复或修改任何训练。两个纯音频模型均无文本/视频条件，因此这是音频缺失区域补全测试，不能将其当作文本合成 WER 或 CelebVDub AVSync 测试。

当前服务器实际为 `xju-aslp8`，目录前缀 `/s7home`；`/home/zjw524` 指向同一用户目录。不要根据旧会话中的 only-VAE / 4090 描述判断当前机器。

## 保持原训练源码

本地原训练/评测基线为 `daaf5ad`。核查时远端 main 已增加其他实验改动，包括被 S2c immutable training contract 绑定哈希的模型源码。为避免改变这次原生评测的运行语义，新增评测使用独立分支 `eval/native-audio-pretrain-20260906`，不更新或改写原模型源码。

只修改 `AlignDiT_mmdit_c2_semantic_vae` 的新增评测入口、测试、launcher 和评测依赖。本报告放在仓库交接文档目录。其他实验快照不改。

## 固定协议

- LibriSpeech dev-clean / dev-other 各 25 条，4–10 秒，50 个不同说话人；沿用旧评测的固定样本选择，但此次重新冻结物理遮挡区间与输出目录。
- 每条音频一个约 70% 的连续缺失区间，边界量化到 800 samples / 50 ms：精确等于 5 个 mel 帧或 2 个 VAE 帧。两组不独立取整。
- 每条样本使用 666、667、668 三个采样 seed，两组各生成 150 条；具体 ODE seed 由样本 key 确定。同 seed 不意味着不同维数的噪声张量逐元素相同。
- EMA、FP32、关闭 TF32、Euler、EPSS、32 步、batch=1；代码中的 CFG strength=1，即 `2 * conditional - unconditional`。
- 原始波形缺失区间先置零，再分别计算 mel 或 VAE 条件。模型仅看剩余约 30% 的原始音频信息，不输入文本、视频、HuBERT。
- Semantic-VAE 条件使用严格原版 posterior 编码器，固定每样本 posterior 噪声并使用训练集统计归一化；三次 ODE 采样共用这一份条件。
- 解码只按原始 sample 数裁掉右侧 padding，不做拉伸、自动时移对齐或响度归一化。音频保存为 float32 WAV，写后核对精确 round trip。
- 相似度只计算缺失区间，不能用模型复制的已知音频抬高指标。说话人另外与输入的已知上下文比较。
- 先在每个样本内平均三次采样，再按 utterance 配对、按 dev 子集分层 bootstrap；不把 150 个结果当成 150 个独立样本。

### 为什么要先遮挡波形

Semantic-VAE 编码器包含卷积及覆盖历史帧的 causal attention。完整原音频先编码再遮挡中间 latent 时，后缀 latent 可能携带缺失区间内容。因此当前严格协议先遮挡波形，完整 GT 编码只用于独立的 codec 重建参考，不得进入 ODE 条件或最终已知帧复制。

这消除了该信息泄漏路径，但也带来明确解释边界：训练时采用缓存的完整 latent 再遮挡，而本评测改为波形先遮挡。特别是 VAE 的条件分布可能发生变化；因此本测试不能单独判定是否充分收敛。两模型的 QK-Norm 配置也不同，所以这是完整适配系统的比较，不是仅微调步数这一个变量的消融。

## 必须绑定的资源

以下路径以 `${ROOT_PREFIX}/zjw524` 为前缀；当前 `ROOT_PREFIX=/s7home`。

| 资源 | 路径后缀 | SHA256 |
|---|---|---|
| mel500k | `datasets/AlignDiT_pretrain_LibriSpeech_500000.pt` | `4a9fc0e526ce47745aee839348406ca99597d32f5ed028bda42a3de3ec900fcd` |
| S2c70k | `projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt` | `02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4` |
| S2c contract | 与 S2c70k 同目录的 `training_contract.json` | `3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d` |
| train normalization | `projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json` | `65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6` |
| Semantic-VAE EMA | `projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth` | `7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516` |
| HiFi-GAN | `projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000` | `af3f49e21c70b5fe4a120fee27ddca56d076a52bc8ebbc9bc02db7903b61bd07` |

说话人指标使用已有 WavLM-Large / ECAPA 权重；情绪指标使用已有 emotion2vec_plus_large，分别报告真正的 `feats` embedding cosine 与旧项目的 `scores` cosine，不能混称一个指标。补充 STOI、SI-SDR 仅衡量与原缺失音频的一致性；由于未提供缺失文字，不能将波形一致性当成唯一质量标准。另保留两种 codec 的完整原音频重建作为控制。

## 入口与复现

项目目录：`${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae`。

- `src/aligndit/script/eval/compare_native_audio_pretrains.py`：冻结样本、编码遮挡后条件、两种模型采样、VAE 反归一化与解码。
- `src/aligndit/script/eval/evaluate_native_audio_pretrains.py`：严格验证生成清单、两种 embedding 与波形指标、配对 bootstrap、试听 HTML。
- `src/aligndit/run/eval/compare_native_audio_pretrains.sh`：按 `prepare` → `canary` → `formal` 串行运行；也可使用 `all` 自动串联。任一步失败即停止，拒绝覆盖已生成的正式目录/日志。
- AlignDiT Python：`${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python`；Semantic-VAE 独立 Python：`${ROOT_PREFIX}/zjw524/ENTER/venvs/semantic-vae/bin/python`。
- 本次补充评测依赖 `pystoi==0.4.1`，不重装、升级训练环境的 torch/numpy 等核心库。

示例（必须先确认选定 GPU 可用，并使用一个全新的输出目录）：

```bash
setsid env ROOT_PREFIX=/s7home GPU_ID=0 \
  RUN_ROOT=/s7home/zjw524/projects/data/evals/native_mel500k_vs_svae70k_waveform_masked_50x3_20260906 \
  bash src/aligndit/run/eval/compare_native_audio_pretrains.sh all \
  > /s7home/zjw524/projects/data/evals/native_pair_20260906_launcher.log 2>&1 &
```

模型权重、数据、日志、latent、WAV、HTML 等运行产物均放在仓库外，不提交 Git。正式结果必须检查 `metrics/complete.json` 与两组 `generation_complete.json`，不能把 canary 或不完整运行当成结果。

## 运行结果

待当前实际评测完成后更新；本版本只记录已核验的协议与资源，不预先填写或推测评测数值。
