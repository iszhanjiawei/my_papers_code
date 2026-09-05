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

2026-09-06（北京时间）已完成正式评测，两组各 50 条 × 3 seeds，共 300 条生成音频和 100 条 codec 重建。三种子先在句内平均，配对统计的样本数为 50，不是 150。平均遮挡比例为 0.695639，样本实际时长 4.005–9.765 秒。

### 生成的缺失区域：原始模型在音色保持上明显领先

Δ = Semantic-VAE S2c70k − 原始 mel500k，括号为按句配对、按 dev 子集分层的 10,000 次 bootstrap 95% 区间。

| 指标 ↑ | mel500k | S2c70k | Δ / 95% CI |
|---|---:|---:|---|
| SPKSIM，与输入已知上下文比较 | 0.408721 | 0.214098 | −0.194623 / [−0.219974, −0.170373] |
| SPKSIM，与原始缺失区间比较 | 0.374484 | 0.172783 | −0.201701 / [−0.227106, −0.175080] |
| 情绪 embedding cosine | 0.956095 | 0.953166 | −0.002929 / [−0.008080, +0.002120] |
| 旧 EMOSIM：情绪分类 scores cosine | 0.550246 | 0.549741 | −0.000505 / [−0.074156, +0.071957] |
| STOI，与原始缺失区间比较 | 0.184303 | 0.174854 | −0.009450 / [−0.022923, +0.003770] |

两种 SPKSIM 的差异在 dev-clean 和 dev-other 中方向一致、区间都低于零；不仅是与隐藏 GT 说话人比较时低，与模型实际看到的参考音频比较也明显低。情绪指标和 STOI 的差值区间包含零，不能宣称这几项已证明退化或等效。

按句平均三种子后，S2c 的原始区间 SPKSIM 在 49/50 条低于 mel，上下文 SPKSIM 在 50/50 条低于 mel。因此，当前观察到的音色差距不是由少数异常样本拉低平均值造成的。

生成音频平均 RMS：mel 0.085536，S2c 0.057730。完整报告保留幅值越界比例、SI-SDR 和逐条结果。RMS 较低是观察到的现象，不等于已经证明是音色差距的原因。

### 只经过 codec 的重建：Semantic-VAE 明显更好

这里不使用扩散模型，直接把原始音频经各自原生表示和 decoder 重建，再在同一缺失区间评分。

| 指标 ↑ | mel + HiFi-GAN | Semantic-VAE |
|---|---:|---:|
| SPKSIM，与原始缺失区间比较 | 0.837763 | 0.966359 |
| 情绪 embedding cosine | 0.985472 | 0.998028 |
| 旧 EMOSIM scores cosine | 0.849601 | 0.947696 |
| STOI | 0.925181 | 0.979429 |

因此，不能把生成阶段的说话人相似度差距简单解释成“64D/40 Hz codec 没能力保留音色”。当前证据更直接地指向生成模型、条件利用或适配方案仍有差距，但尚未识别唯一原因。

SI-SDR 不作为两条 codec 链路的主要音质排名依据：波形级相位一致性对该指标影响很大，HiFi-GAN 原生重建并不要求与 GT 逐采样点同相。没有文字条件的生成 STOI/SI-SDR 也不能当作 WER 或主观发音清晰度。

### “微调够不够”应分开回答

1. 作为表示适配，70k 是已训练阶段中有完整验证支持的最佳候选，不是未加载或不能运行的权重。之前 5,551 条完整 dev 的同分布评测：60k→70k Flow MSE 为 1.3364042413→1.3357761681，相对改善约 0.047%；HuBERT cosine 为 0.6524509715→0.6536650244。尾段仍有改善，但幅度很小。
2. 作为纯音频补全生成模型，不能据此宣布“已完全达到原始 mel 模型水平”。此次音色保持存在明确差距，dev loss 趋稳不等于最终生成质量已经达标。
3. 这还不能证明“只要再微调更多步就能解决”。本测试的条件构造与训练分布不同，两模型还存在表示、QK-Norm、codec 等差异。应先复核与实际参考音频使用方式一致的条件协议和生成效果，再决定是否增加训练，而不是直接重启相同 scheduler 继续跑。
4. 本次没有主观盲听评分、WER 或 AVSync，也未启动 C2 多模态训练。50 条 pilot 不替代完整 dev 或下游配音评测。

旧同分布评测证据目录：

```text
/s7home/zjw524/projects/data/evals/svae_warmstart_dev_full_fp32_seed666_r1_20260807/
  s2c_60k.summary.json
  s2c_70k.summary.json
  comparison_vs_s2c60.md
```

### 附加条件诊断的限制

只读比较了本次 zero-wave 条件 latent 与原始缓存 latent：远离卷积遮挡边界的已知前缀/后缀平均 normalized MSE 约为 0.844976 / 0.843536。两种安全区域都存在的 33 条样本，后缀减前缀的 MSE 均值仅 +0.000257。

两份 latent 的 posterior seed 在全部 50 条上都不同，连不接触遮挡区的前缀也存在明显随机差异，因此不能借这组诊断证明后缀有很大的额外分布偏移。代码上的条件依赖风险是确定的；它是否为本次性能差距的主要原因仍未证实。

## 验证与运行记录

- 新增 18 项 CPU 单元测试通过：物理 mask、输入泄漏拒绝、已知区域篡改、WAV round trip、禁止覆盖、完整记录、重复种子、配对统计、静音退化和 canary 隔离。
- 现有 MM-DiT CPU smoke test 通过，包括模型构造、前反向、条件丢弃、checkpoint 兼容和采样；本次未改模型代码。
- 新增 Python 文件 Ruff、format、py_compile 通过；launcher `bash -n` 通过。
- 两条 canary 的完整推理、解码、指标及文件完整性校验通过后才执行正式对照。
- 第一次 canary 在生成前遇到 CUDA allocator 统计尚未初始化的问题；已在读取统计前初始化所选设备并通过独立检查。旧失败日志保留，没有混入正式输出。
- emotion2vec 日志的 10 个缺失参数仅属于不用的预训练 decoder。CPU schema 核查确认实际推理使用的其余 185 个参数全部加载、形状一致，包含情绪分类头。
- 正式 launcher PID/SID 曾为 1934806，TTY 为 `?`；正式任务现已退出，没有启动或停止其他训练任务。
- 独立复核正式完成标记绑定的 5 个报告文件 SHA；从原始 JSONL 重算 48 个统计项，均值、差值与区间和报告一致（误差小于 1e-12）。无静音生成；3 个 S2c 文件共 6 个采样点恰好达到绝对值 1，没有超过 1 的采样点，float32 WAV 未施加写入削波。

同一张 A40（物理 GPU 2），FP32、batch=1 的记录如下。耗时包含加载、校验和落盘，显存为 PyTorch allocator 峰值，不是整卡占用，也不是训练吞吐 benchmark。

| 阶段 | 耗时 | allocator 峰值 |
|---|---:|---:|
| 50 条 VAE 遮挡后条件编码 | 11.043 s | 536.093 MiB |
| mel：150 次采样及 HiFi-GAN 解码 | 262.906 s | 982.039 MiB |
| S2c：150 次 latent 采样 | 185.744 s | 749.933 MiB |
| VAE 反归一化、150 条生成与 50 条重建解码 | 38.533 s | 393.189 MiB |
| 两组统一指标及报告 | 149.810 s | 未单独记录 |

## 结果位置与 Git

正式目录：

```text
/s7home/zjw524/projects/data/evals/native_mel500k_vs_svae70k_waveform_masked_50x3_20260906/
  common/manifest.jsonl
  mel/generation_complete.json
  svae_context/context_complete.json
  svae_latents/latent_generation_complete.json
  svae/generation_complete.json
  metrics/complete.json
  metrics/summary.json
  metrics/summary.md
  metrics/generated_metrics.jsonl
  metrics/codec_metrics.jsonl
  metrics/listening.html
```

试听页保留原音频、输入上下文、两条 codec 重建、三种子生成完整音频及缺失区间。它是供人工检查的试听页面，不代表已经完成盲听/MOS。

所有代码提交均已推送到 `eval/native-audio-pretrain-20260906`，不推送或改写原始训练权重：

- `6462c80`：原生模型比较、原始波形遮挡防泄漏及生成协议测试。
- `5f4ba36`：严格配对指标、codec 对照、置信区间、试听页及指标测试。
- `d10a010`：分阶段 launcher 与协议交接文档。
- 本节正式结果另行提交；最终提交号以该文档的 `git log` 为准。
