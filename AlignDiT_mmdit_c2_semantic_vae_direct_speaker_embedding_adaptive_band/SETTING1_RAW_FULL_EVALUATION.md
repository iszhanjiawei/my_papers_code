# Setting 1：仅目标文本，直接解码完整原始采样状态

2026-10-14 完成。本实验对应不含 REPA 的本项目，EMA 150k / 200k 各 213 条。
**直接解码完整原始采样状态没有解决 target-only 的问题：整段 WER 从约 32% 升至约 64% / 67%，主要增加的是插入错误。首句缺失的 `you think` 也未被整段 ASR 恢复。**

## 实验实际改变了什么

沿用 [上一轮 target-only 对照](SETTING1_TARGET_ONLY_EVALUATION.md) 的模型、数据、参考音频、CAM++、视频和采样参数，只改变最终解码区域。Setting 1 的参考和目标来自同一 GT clip，P=T=N；参考台词与目标台词相同，但本次文本只输入一次目标原文。

```python
generated, trajectory = model.sample(..., text=[target_text], duration=original_duration)
z_normalized = trajectory[-1]       # 完整 ODE 末态，未替换参考前缀
z = z_normalized * latent_std + latent_mean
wav = decoder(z.transpose(1, 2))    # 所有采样帧一次解码
save(wav)                          # 不裁 latent，不截 WAV，不拼参考音频
```

`model.sample()` 原本同时返回已回填参考前缀的 `generated` 和原始 `trajectory`。本模式直接选择后者的末态，前者不用于解码；不能把已回填的完整返回值拿来证明缺失词去了前缀。源码位置是 `src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py` 的 `select_output_latent` 和 `run`，开关为 `--output-region raw-full`。默认 `target` 模式保留既有裁剪行为。

- 213 条、2378 个参考词；同一 CelebV-Dub Setting 1 清单，EMA 150000 / 200000，分别物理 GPU 0 / 1。
- FP32/high TF32，seed 0，Euler/EPSS，32 NFE，sway=-1，CFG text/video=5/2；同一 64D/40 Hz latent 标准化、1000k EMA Semantic-VAE decoder、完整参考音频的 192D CAM++、目标嘴部条件。
- 原固定音频条件和视频参考前缀保留；本次没有改变噪声长度或训练模型。
- 212 条采样长度为 P+T=2N；`0qkVyahL10U/1_2` 保留历史采样器长度保护，P=T=29、布局 58 帧、实际采样 67 帧。本次解码所有 67 帧，额外 9 帧也保留，不偷偷裁成 58 帧。
- 每 latent 帧解码为 400 个 16 kHz 样本；保存单声道 PCM16 WAV，包含全部解码输出，末尾也不按 GT 时长截断。

## 指标范围和结果

SPKSIM、WER、EMOSIM 均读取整段保存的 WAV。WER 仍与一次原始目标台词比较，分母为 2378；不重复 GT、不平均逐句 WER。传统/上一轮结果的 WAV 是目标段，本次 WAV 是完整采样段，因此这是不同输出范围的诊断对照。

原嘴部视频只有目标 T 帧，不能将完整 P+T 音频与它冒充标准整段 AVSync。本次 AVSync 是明确标注的**目标时间窗诊断**：在内存中取完整解码 WAV 的 `[P*400:P*400+original_num_samples]`，与原嘴部视频提取联合特征；磁盘 WAV 不裁剪。未重复视频、拉伸音频或搜索最佳偏移。

| EMA | 文本与解码方式 | SPKSIM ↑ | WER (%) ↓ | EMOSIM ↑ | AVSync（目标窗）↑ |
|---|---|---:|---:|---:|---:|
| 150k | 传统拼接文本，目标段 | 0.65820 | 5.92935 | 0.76552 | 0.56968 |
| 150k | 仅目标文本，裁剪目标段 | 0.60127 | 32.38015 | 0.71370 | 0.47681 |
| 150k | 仅目标文本，完整原始采样 | 0.59970 | 64.17157 | 0.42046 | 0.47674 |
| 200k | 传统拼接文本，目标段 | 0.66823 | 5.38267 | 0.76102 | 0.57056 |
| 200k | 仅目标文本，裁剪目标段 | 0.60539 | 32.25399 | 0.71919 | 0.48204 |
| 200k | 仅目标文本，完整原始采样 | 0.60521 | 66.61060 | 0.41834 | 0.48164 |

完整模式相对上一轮裁剪模式：150k WER 增加 31.79142 个百分点，200k 增加 34.35660 个百分点；EMOSIM 分别下降 0.29324 / 0.30085，SPKSIM 接近上一轮。目标窗 AVSync 也基本相同，没有恢复传统方式的同步分数。

完整解码后再取 waveform 目标窗，与先裁 latent 再解码存在 decoder 边界上下文差异；不能要求两者 WAV 或 AVSync 数值严格相等。

## 词错误组成

S/D/I 是同一词级编辑距离对齐下的替换、删除、插入，不能把 S/D 的下降直接等同于确定的词位置恢复。

| EMA | 输出方式 | S | D | I | 编辑数 / 2378 | 完整句零错误 |
|---|---|---:|---:|---:|---:|---:|
| 150k | 传统目标段 | 49 | 71 | 21 | 141 / 2378 | 137 |
| 150k | target-only 裁剪 | 346 | 386 | 38 | 770 / 2378 | 27 |
| 150k | target-only 完整原始采样 | 217 | 124 | 1185 | 1526 / 2378 | 35 |
| 200k | 传统目标段 | 44 | 64 | 20 | 128 / 2378 | 147 |
| 200k | target-only 裁剪 | 365 | 367 | 35 | 767 / 2378 | 28 |
| 200k | target-only 完整原始采样 | 219 | 123 | 1242 | 1584 / 2378 | 34 |

插入错误分别增加 1147 / 1207 个，超过 S+D 的减少。整段包含额外识别内容，有重复台词的实际案例，不能当作只合成一次目标台词的有效修复。也并非每条都变差：对比上一轮裁剪，150k 有 54 条改善、47 条持平、112 条变差；200k 有 45 条改善、53 条持平、115 条变差。

## `you think that's all I do` 的直接检查

clip 为 `test/0_ArO8UCfyk/0_0`，P=T=44，完整状态 88 帧，WAV 为 35200 样本（2.2 秒），P/T 边界为 1.1 秒。传统两组正确；上一轮 target-only 裁剪两组均为 `thats all i do`。

本次整段主评测 ASR 和带词时间戳诊断的转写一致：

| EMA | 整段 ASR | 词时间戳所示范围 |
|---|---|---|
| 150k | `That's all I do. That's all I do.` | 第一遍 0.00–0.90 秒，第二遍 1.30–1.98 秒 |
| 200k | `That's all I do.` | ASR 返回的一遍为 0.00–0.90 秒 |

150k 的前 P 区域确实识别出目标台词的一部分，后段也识别出同一部分；两组整段均未识别出 `you think`。因此这个例子不支持“缺失的两个词只是被裁掉了”。200k 整段 ASR 只返回一遍，不能据此断言后段物理上没有语音，ASR 可能合并或漏掉重复内容。

时间戳来自 faster-whisper large-v3 的估计，不是声学真值；没有识别出某词也不证明该词绝对未生成。Setting 1 的参考本来就是同一句台词，即使原始采样前缀含有相同词，也不能仅凭它区分“条件参考的重建”和“目标文本迁移”。

词时间戳诊断固定选择测试列表前 8 条及上述长度异常样本，每权重 9 条，共 18 条，无失败；选择在观察本轮结果前固定。不能把这 18 条的定位结果推广成全部 426 条音频的逐词位置结论。

## 核验与结论

- 两组生成与四指标流水线均返回 0；426 个完整 WAV、426 个原始 latent、426 个目标窗联合特征、1704 条新指标记录逐条覆盖完整测试清单。
- 每条文本等于目标原文，完整 latent 有限且 shape 与实际采样长度相等；WAV 长度严格为采样帧数 × 400，无 latent/波形截断。
- 426 条原始前缀均不与条件参考 latent 逐元素相等，解码源确为 `trajectory[-1]`；每条目标 latent 与采样器已回填返回值的对应目标区相等。
- 新旧 checkpoint、decoder、数据集、speaker、标准化和采样参数一致；3530 项源码、输入、模型资产和历史产物哈希与启动前一致。
- 独立 DP 和 jiwer 重算 corpus WER、S/D/I；其他指标从完整样本均值重算并核对日志；NumPy 重算 AVSync 最大差为 `1.1118876053206606e-7`。
- 旧 JSONL 只有 basename，按单 worker 的固定测试顺序、basename、真值绑定完整 clip 路径，未跨视频用 basename 合并。
- 全部后台进程已结束，GPU 0 / 1 已释放。本轮原始输出及旧结果均保留。

当前两个权重下，**保留完整原始采样段不能作为 target-only 的替代修复方案，Setting 1 仍应使用传统文本拼接与目标段输出。** 本轮证明前缀可出现可识别语音、整段易出现额外内容，但没有证明所有漏词都跑进前 P 帧，也没有隔离文本对齐、参考重建和 decoder 上下文各自的贡献。

## 源码、复现与产物

源码提交 `bdb8527cdf29ae36769b0907c9394ebe81ac25e2`，已推送 main。`select_output_latent` 的实际函数体用不同前缀的 sentinel tensor 检查：raw-full 保留全部原始帧、target 保留原目标区、非法模式拒绝；音频窗 offset/长度与越界拒绝检查通过。两个 Shell 入口语法检查通过。

复现命令见 [ADAPTIVE_TEMPORAL_BAND.md](ADAPTIVE_TEMPORAL_BAND.md) 的 Raw full-sequence diagnostic 部分：设置 `TEXT_CONDITIONING=target-only OUTPUT_REGION=raw-full`，为两权重各指定新的输出目录。原始 latent 位于各权重目录的 `raw_latent/test/`，完整 WAV 位于 `test/`。

产物根目录：`/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111`。

- [完整比较和独立核验](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/comparison.json)
- [启动前输入和历史产物哈希](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/preflight.json)
- [全流程完成记录](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/pipeline_complete.json)
- [150k 词时间戳](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/150000/word_timing_diagnostic.json)、[200k 词时间戳](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/200000/word_timing_diagnostic.json)
- [150k 首句完整音频](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/150000/test/0_ArO8UCfyk/0_0.wav)、[200k 首句完整音频](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_raw_full_150k_200k_20261014_015111/200000/test/0_ArO8UCfyk/0_0.wav)

逐条推理记录、四份指标 JSONL、音频窗绑定 `audio_region_protocol.json`、配对记录、运行脚本及日志均在产物目录，不提交 Git。
