# Setting 1：仅目标文本对照实验

2026-10-14 完成。本记录仅对应本项目 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band`，不包含 REPA。**在 EMA 150k 和 200k 上，去掉参考台词后四项指标均退化，当前 Setting 1 应保留传统拼接文本。**

## 对照范围与协议

Setting 1 的参考音频和目标来自同一条 GT clip，因此参考台词等于目标台词。传统模型文本为 `text + "  " + text`；新模式只输入一次 `text`。参考 latent 和完整音频提取的 192D CAM++ 均保留。这不是独立参考片段的 voice-cloning 评测。

- CelebV-Dub Setting 1，213 条完整测试样本、2378 个参考词；测试列表 SHA256：`1ad609deefaf1293e00b168a3418611ef779c4e9b49f3b51539721a8360a1e49`。
- 分别严格加载 EMA 150000 / 200000；配置为 `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band.yaml`。
- FP32、`torch.set_float32_matmul_precision("high")`、seed 0、Euler/EPSS、32 NFE、sway=-1、CFG text/video=5/2；150k 使用物理 GPU 0，200k 使用物理 GPU 1，单卡指标计算 `-n 1`。
- 同一 64D/40 Hz 参考 latent、LibriSpeech 训练集标准化、1000k EMA Semantic-VAE decoder、目标视频、视频参考前缀、参考 CAM++、目标裁剪和真实输出时长。
- 保留原采样长度。通常 P=T=N、长度 2N。clip `0qkVyahL10U/1_2` 的 N=29，原重复文本有 66 个字符，会触发采样器的文本长度下限，实际长度为 67，而音频/视频布局为 58。新模式显式保留 67 帧噪声和原 58 帧视频输入，仍只裁剪 [29:58] 的目标 latent；额外 9 帧不解码。因此没有把文本与噪声长度变化混为一个干预。
- 沿用原四指标实现和模型资源。SPKSIM、EMOSIM、AVSync 为样本等权均值；WER 为词级编辑距离总和 / 2378，不平均逐句 WER。EMOSIM 为既有 emotion2vec classifier score 向量余弦；AVSync 为生成音频/GT 音频与相同嘴部视频的 AV-HuBERT 联合特征逐帧余弦均值。

| 权重 | Checkpoint SHA256 |
|---|---|
| EMA 150k | `3a6f4c49dfb884040f685db9605378260c5e064a7898640929b3438b26fcd8e8` |
| EMA 200k | `c6c49fcaf78ec9bb0aa4e092375ea0ddf381045013cec5f2a012c9d70cc309f2` |

## 四指标结果

| EMA 权重 | 模型文本 | 覆盖 | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---|---:|---:|---:|---:|---:|
| 150k | 参考台词 + 目标台词 | 213/213 | 0.65820 | 0.05929 | 0.76552 | 0.56968 |
| 150k | 仅目标台词 | 213/213 | 0.60127 | 0.32380 | 0.71370 | 0.47681 |
| 200k | 参考台词 + 目标台词 | 213/213 | 0.66823 | 0.05383 | 0.76102 | 0.57056 |
| 200k | 仅目标台词 | 213/213 | 0.60539 | 0.32254 | 0.71919 | 0.48204 |

WER 表中为小数。例如 150k 新模式为 **770/2378 = 32.38015%**，200k 为 **767/2378 = 32.25399%**。

下表差值为“仅目标台词 − 传统拼接”，使用未舍入原始均值计算。相似度负值和 WER 正值均表示退化。

| EMA 权重 | ΔSPKSIM | ΔWER | ΔEMOSIM | ΔAVSync | WER 增加百分点 |
|---|---:|---:|---:|---:|---:|
| 150k | -0.05693 | +0.26451 | -0.05182 | -0.09286 | +26.45080 |
| 200k | -0.06285 | +0.26871 | -0.04183 | -0.08852 | +26.87132 |

## 错误组成与逐句对比

| EMA 权重 | 模型文本 | S | D | I | 编辑数 / 参考词 | 完整句零错误 |
|---|---|---:|---:|---:|---:|---:|
| 150k | 参考台词 + 目标台词 | 49 | 71 | 21 | 141 / 2378 | 137 |
| 150k | 仅目标台词 | 346 | 386 | 38 | 770 / 2378 | 27 |
| 200k | 参考台词 + 目标台词 | 44 | 64 | 20 | 128 / 2378 | 147 |
| 200k | 仅目标台词 | 365 | 367 | 35 | 767 / 2378 | 28 |

150k：4 条改善、44 条持平、165 条变差；200k：4 条改善、43 条持平、166 条变差。主要增加的是替换和删除，不是少量插入造成的 WER 波动。

例如首条目标台词归一化后为 `you think thats all i do`，两组新模式均识别为 `thats all i do`，丢掉句首的 `you think`。传统两组在这一句均无词级错误。

## 可得出的结论与机制边界

当前项目的 Setting 1 **不适合直接去掉参考台词**。参考音频条件和目标视频并不能硬性保证仅输入一次的目标台词完整落在后 T 帧。当前 C2 的 `prompt_isolated_ca=False`，前后音频位置都能读取文本，采样状态的参考前缀也没有逐步硬回填。

一个与结果相容的解释是：同句参考与目标的两段布局配合两份文本时，对齐更合适；去掉一份文本会改变长度、位置和文本/音频条件的关系，导致目标区域漏词和替换增多。当前结果没有解码原始采样前缀、检查注意力或进行词时间对齐，不能据此证明缺失台词都被生成到了前 P 帧，也不能声称模型存在显式的“参考音频与文本是否匹配”判断。

此前 GRID Setting 2 的 target-only 改善不能直接外推到本项目：那是不同数据集、带 REPA 的不同 checkpoint，并且使用独立参考台词。本次对照严格支持的是本项目两个权重下的 Setting 1 文本选择，不能把跨实验差异只归因于 Setting 编号。

## 实现、验证与复现

源码提交：`ca0e94d9ba2917c2e08b63d331ea136aa1f91ef5`，已推送 main。

- `infer_celebvdub_semantic_vae_s1.py` 新增 `--text-conditioning target-only`、`setting1_model_text` 和保留历史有效采样长度的 `setting1_sampling_duration`。参考 latent、视频、speaker 和裁剪路径沿用原代码。
- 两个 adaptive-band 推理/四指标 Shell 入口支持 `TEXT_CONDITIONING`；默认仍为 `reference-target`，target-only 默认输出目录带独立后缀。
- 213 条 CPU 文本与有效长度检查通过：默认字符串保持完全一致、target-only 等于目标原文、有效噪声长度一致、非法模式拒绝。Shell 语法与 Git diff 检查通过。
- 两组各 213 WAV、213 生成联合特征和四份各 213 条指标全部覆盖；426 WAV 的 SHA256、16 kHz/mono/PCM16 和逐条真实时长通过核验。输入文本、参考 speaker、checkpoint、数据集、标准化和 decoder 与历史对照一致。
- 独立动态规划及 jiwer 重算词级编辑距离和 S/D/I；四指标重新汇总，日志舍入值一致。NumPy 重算 AVSync 最大差为 1.1578510838994305e-7。
- 2666 项源码、输入、资产及历史结果文件与启动前哈希一致；历史结果未覆盖。旧指标文件只保存 basename，因此依据既有单 worker 测试列表顺序、basename 和 ASR 真值检查绑定完整相对路径，绑定记录另存，不能用 basename 跨视频关联。
- 所有生成、评测和核验进程已结束，两张 GPU 均已释放。

复现命令见 `ADAPTIVE_TEMPORAL_BAND.md`。本次具体命令和独立 session PID 保存于 `pipeline_state.json`，使用 `setsid` 启动，未使用 nohup。

产物根目录：`/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419`。

- [完整比较与独立核验](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419/comparison.json)
- [启动前文件哈希与历史基线](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419/preflight.json)
- [150k 推理记录](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419/150000/inference_summary.json)
- [200k 推理记录](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419/200000/inference_summary.json)
- [完整相对路径的指标配对](/zjw524/projects/data/evaluations/CelebVDub_s1_adaptive_band_target_only_150k_200k_20261014_003419/paired_metric_bindings.jsonl)

两组 WAV、AV-HuBERT 特征、四份指标文件分别位于 `150000/` 和 `200000/`；执行日志为 `150000.log`、`200000.log`，核验脚本与日志保存在同一根目录。运行产物不提交 Git。传统结果仍为原 checkpoint 目录下的 `eval_s1_{150000,200000}_adaptive_band_cfgv2.0/`。
