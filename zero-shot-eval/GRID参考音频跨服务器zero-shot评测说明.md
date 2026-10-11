# GRID 参考音频与 CelebV-Dub 跨服务器 zero-shot 评测说明

更新日期：2026 年 10 月 11 日。用途：在具备原生 CelebV-Dub 视觉特征的另一台服务器上，测试 CelebV-Dub 训练的 StyleDubber 和 ProDubber，并与此前 AlignDiT 的 GRID 参考实验保持相同目标、参考配对及评分口径。

**核心协议：用 GRID 的声音，为 CelebV-Dub 的视频和台词配音。** CelebV-Dub 提供目标视频及真实目标台词；GRID 只提供外部参考声音。最终输出应说 CelebV-Dub 台词，并尽可能具有所配 GRID 说话人的音色。两份指定权重不在 GRID 上继续训练或微调。

当前服务器只完成了权重及参考输入核验；StyleDubber 和 ProDubber 的本次 GRID 推理尚未运行，没有这两组的新指标。阻塞原因是缺少训练时对应的 CelebV-Dub 唇部及脸部特征。以下原生模型步骤是下一台服务器需要完成的适配与运行要求，不是已经完成的实验结果。

## 一 数据在推理与评分中的分工

| 数据 | 模型推理中的用途 | 评分中的用途 |
|---|---|---|
| CelebV-Dub 目标视频 | 提供嘴部动作及脸部视觉条件 | 与生成音频一起提取 AV-HuBERT 联合特征 |
| CelebV-Dub 目标真实台词 | 指定生成内容 | WER 的正确文本 |
| CelebV-Dub 目标真实音频 | 不作为声学条件；只沿用已有时长元数据 | 两种情感相似度的参照，以及 GT AV-HuBERT 联合特征的音轨 |
| GRID 配对参考音频 | 提供外部音色与参考风格 | SPKSIM 的参考音频 |
| GRID 参考台词 | AlignDiT 的音频 prompt 配套文本；两种原生 Dubber 不需要它作为文本输入 | 不是目标 WER 的正确文本 |
| GRID 视频或唇部特征 | 不使用 | 不使用 |

因此，缺少的是 **CelebV-Dub 目标视频的特征**，不是 GRID 视频特征。原项目的目录名 `extrated_embedding_Grid_152_gray` 和 `Grid_VA_feature` 沿用了 GRID 命名，但这里应读取 CelebV-Dub 视频 ID 对应的文件。

例如第一条配对：

```text
目标 ID：       test/0_ArO8UCfyk/0_0
目标台词：      you think that's all i do
GRID 参考 ID：  grid/s29/bbio9s
GRID 参考台词： bin blue in o nine soon
```

这一条应生成 “you think that's all i do”，参考音色来自 GRID s29 的 bbio9s；不能把 GRID 台词当成待生成台词，也不能把两句话一起保存在最终评分 WAV 中。

## 二 冻结的目标与参考配对

使用已有清单，不重新抽样：

```text
/zjw524/projects/data/Grid_reference_celebvdub_s1_seed0_svae1000k_campplus_v1/pairs.jsonl
```

原始清单 SHA256：

```text
cd1ab0adf681946c389a5e674eb44d0c923a686a0f9bcf5418794d0a3e72c729
```

- 目标为历史 CelebV-Dub Setting 1 的同一组 213 条片段，保留清单顺序。
- 每条目标只分配一条 GRID 参考，共 213 条不同参考；不是 213 个目标与所有说话人的笛卡尔积。
- 覆盖 GRID 的 33 位说话人：s1 至 s34，排除 s21。排除原因是当前 GRID 副本缺少 s21 对应的参考转录。
- 配对 seed 为 0；先平衡说话人，再随机选取不重复的音频。每位说话人出现 6 或 7 次。
- 候选来自本地 GRID `audio_25k` 中具有对应 `.lab` 的文件，**没有按 GRID 官方训练或测试划分筛选**。不要在新服务器重新限定为训练集或测试集，否则会变成另一套实验。
- `.lab` 中去掉 `sp`、`sil`、`<sil>` 等静音标记，保留六个实际单词。清单里的 `ref_text` 已保存处理结果。

原始来源路径用于追溯：

```text
音频：/zjw524/datasets/Grid_Dataset/Grid_dataset_Raw/audio_25k/{speaker}/{clip}.wav
转录：/zjw524/datasets/Grid_Dataset/Grid_resample_ABS/Grid_Wav_22050_Abs/{speaker}/{speaker}-{clip}.lab
```

推理及 SPKSIM 使用的是已冻结的 16 kHz 单声道 PCM16 参考 WAV，即清单 `ref_audio` 指向的文件，不是上述 25 kHz 原文件，也不是另一个重采样版本。

使用完整参考句，不额外裁静音、拼接、拉伸或按目标台词筛选。当前参考时长约为 1.22006 至 2.63006 秒，不能一律当成 3 秒。213 条参考音频已逐文件核验采样率、长度和 SHA256。

“zero-shot”在这里指 CelebV-Dub 训练权重不经 GRID 微调，使用外部 GRID 参考进行跨数据集音色迁移；它不意味着无台词输入、无已知时长，或已经证明所有外部预训练组件都没见过 GRID。

## 三 可以直接迁移的参考包

本机已生成：

```text
/zjw524/projects/alignDiT_idea6/my_papers_code/zero-shot-eval/GRID_reference_213_seed0_迁移包.tar.gz
```

约 15 MiB，SHA256 为：

```text
9e6d7913ddf28b03e7109d5af884971f8eb4e2628112b6f1e514c00cdfdc2b47
```

包内结构：

```text
Grid_reference_celebvdub_s1_seed0_svae1000k_campplus_v1/
  pairs.jsonl          原始 213 条配对，包含目标和参考台词
  metadata.json        抽样方法与缓存来源
  plan.jsonl           原始准备计划
  audio/s29/bbio9s.wav 等，共 213 个固定参考 WAV
  latents/            AlignDiT 使用的 Semantic-VAE 参考 latent
  speakers/           AlignDiT 使用的 192 维 CAM++ 参考 embedding
  records/            缓存提取记录
```

**StyleDubber 和 ProDubber 只需要本包的配对信息与 `audio/`；不能把其中的 latent 或 CAM++ embedding 送给它们。** 保留这些额外缓存是为了也能追溯此前 AlignDiT 输入。

在新服务器先核对压缩包哈希，再解压到一个新的目录；下面的 `/path/to/transfer` 替换成实际路径：

```bash
sha256sum /path/to/transfer/GRID_reference_213_seed0_迁移包.tar.gz
mkdir -p /path/to/transfer/grid_reference_bundle
tar --keep-old-files -xzf /path/to/transfer/GRID_reference_213_seed0_迁移包.tar.gz \
  -C /path/to/transfer/grid_reference_bundle
```

这个包不包含模型工程、权重、CelebV-Dub 视频或 GT 音频、原生视觉特征、评分模型和原始 `.lab` 文件。转录文本已在清单中，原始路径与哈希供追溯；如要重新运行原始来源审计，再另行迁移原始 `.lab` 和 25 kHz WAV。

### 迁移后的路径处理

清单内包含当前服务器的绝对路径。**复制文件并不会自动更新 JSON 内的路径。** 保留原始 `pairs.jsonl` 不变，用它证明配对没有变化。

适配器可在内存中解析新路径；如现有评分脚本要求真实路径，则另写 `pairs.runtime.jsonl`，只替换路径字段，保留全部 ID、台词、长度和内容哈希。记录原始与运行清单两个 SHA256。运行清单的哈希会变化，不能声称仍是上面的原始哈希。

| 原字段 | 新服务器解析方法 |
|---|---|
| `ref_audio` | 从 `ref_id=grid/s29/bbio9s` 得到 `{参考包根}/audio/s29/bbio9s.wav` |
| `ref_latent_path` | 如跑 AlignDiT，得到 `{参考包根}/latents/s29/bbio9s.npy` |
| `ref_speaker_path` | 如跑 AlignDiT，得到 `{参考包根}/speakers/s29/bbio9s.npy` |
| `target_gt_audio` | 用完整 `target_id` 定位新服务器的 CelebV-Dub GT；文件内容应与已有哈希一致 |
| `target_video_path` | 这是 AlignDiT 的 40 Hz AV-HuBERT video-only 缓存；原生 Dubber 不读取它，另按目标 ID 读取 25 Hz 唇部及脸部缓存 |
| `ref_source_audio`、`ref_transcript_path` | 原始来源追溯路径；原生推理不需要读取，不能以“旧路径不存在”为由重新抽参考 |

使用新路径时仍需核对对应文件的原有内容哈希。若另一台服务器的 CelebV-Dub 是不同裁剪或重编码版本，不要仅因 ID 相同就当成相同输入。

### 解压后的最小核验

设置 `GRID_BUNDLE` 为解压后的缓存根目录，使用装有 numpy、soundfile 的 Python 运行：

```bash
export GRID_BUNDLE=/path/to/transfer/grid_reference_bundle/Grid_reference_celebvdub_s1_seed0_svae1000k_campplus_v1
python - <<'PY'
import os, json, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
root = Path(os.environ['GRID_BUNDLE'])
manifest = root / 'pairs.jsonl'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == 'cd1ab0adf681946c389a5e674eb44d0c923a686a0f9bcf5418794d0a3e72c729'
rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
assert len(rows) == len({r['target_id'] for r in rows}) == len({r['ref_id'] for r in rows}) == 213
assert len({r['ref_speaker_id'] for r in rows}) == 33
for row in rows:
    prefix, speaker, clip = row['ref_id'].split('/')
    assert prefix == 'grid' and speaker == row['ref_speaker_id']
    path = root / 'audio' / speaker / (clip + '.wav')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['ref_audio_sha256']
    wave, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == 16000 and wave.shape == (row['ref_num_samples'], 1)
    assert np.isfinite(wave).all() and np.any(wave)
assert sum(r['target_num_samples'] for r in rows) == 11536944
print('PASS: 213 targets, 213 fixed reference WAVs, 33 GRID speakers')
PY
```

## 四 三类模型如何使用同一个 GRID 参考

### AlignDiT 的已有实验

1. 同一 GRID 参考 WAV 编码成 Semantic-VAE latent prompt，并提取 192 维 CAM++ embedding。
2. 文本输入为 GRID 参考台词加既有分隔符再加 CelebV-Dub 目标台词。
3. 时间轴为参考段再接目标段；参考段视觉条件置零，目标段使用 CelebV-Dub 的 40 Hz video-only AV-HuBERT 特征。
4. 采样后去掉参考 prompt，只解码并保存目标段，按清单 `target_num_samples` 保留目标长度。
5. 已有采样配置为 EMA、seed 0、32 NFE、Euler/EPSS、sway=-1、CFG text/video=5/2。这些是 AlignDiT 参数，不应强行用于另两种模型。

已有完整协议见：

```text
my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus/GRID_REFERENCE_EVALUATION.md
```

### StyleDubber 的原生参考条件

工程和指定权重：

```text
对比实验/StyleDubber_Project_Compare/StyleDubber/
对比实验/celebV-Dub-checkpoints/styledubber/100000.pth.tar
step = 100000
checkpoint SHA256 = 8c291cca8d73ed6d2d5dc92942e4638774caf749fae1bee0a992e8165f992a10
```

参考实现文件为 `infer_setting2_dubbing.py`、`setting2_dubbing_inputs.py` 和 `celebvdub_runtime.py`。新 GRID 入口应复用它们的原生模型及参考前端，但更换清单读取逻辑：

- 将配对的完整 GRID 16 kHz WAV 按既有前端重采样到 22.05 kHz、峰值归一化至 0.95、转 PCM16。
- 以原训练的 TacotronSTFT 提取参考 mel：n_fft/win=1024、hop=256、80 个 mel 通道、频带 0 至 8000 Hz。
- 以原 `stage6_spk_embedding.wav_to_mel` 和 `dvector.pt` 流程提取 256 维 GE2E speaker embedding；保持训练前端一致。
- **参考 mel 和 speaker embedding 必须都来自配对 GRID 音频。** 不可保留 CelebV-Dub 目标原声的 embedding，也不可用包内 192 维 CAM++ 代替 GE2E。
- 文本只用 `target_text`，按既有 g2p_en 转 ARPAbet。GRID `ref_text` 不拼接进目标文本。
- 视觉输入为 CelebV-Dub 目标片段的唇部和脸部特征，不是 GRID 的。
- 用 `build_models` 构建模型，对 checkpoint 的 `model`、`fusion_model` 都严格加载。使用 `predict` 的无目标声学监督分支。
- 解析器需要的 `mel_target`、`D`、`f0`、`energy` 可以按现有入口使用占位零值；不能读取目标 GT mel、MFA 时长、pitch、energy 来填充。
- 使用对应 22.05 kHz HiFi-GAN，最后转为 16 kHz 单声道 PCM16，保存目标语音，不加参考音频前缀。

还需检查配置里的 `preprocessed_path/stats.json` 以及 vocoder 的 `config_path`、`checkpoint_file`。旧 `/home/...`、`/s7home/...` 路径应显式解析为新服务器路径，不更换训练配置的模型维度或权重。

### ProDubber 的原生参考条件

工程和指定权重：

```text
对比实验/ProDubber_Project_Compare/ProDubber/
对比实验/celebV-Dub-checkpoints/produbber/epoch_2nd_0012.pth
epoch = 12；checkpoint 内 iters = 216215
checkpoint SHA256 = 43ba90ac1d508c7c754b48062b462214e5dce00e12f0fc9e33dcac29de804df2
```

参考实现为 `inference_grid.py` 和 `setting2_dubbing_runner.py`。注意文件叫 `inference_grid.py` 并不代表默认设置已实现本次跨数据集协议。

- 在 `compute_style` 中传入配对的 **GRID WAV**，不能传 CelebV-Dub 目标 WAV。
- 沿用 native librosa 24 kHz 加 mel 前端，n_fft=2048、win=1200、hop=300、80 个 mel 通道。
- 保留既有训练上下文：编码 style 时两端各加 5000 个零采样；这是模型内部前端，不是在最终结果里添加 GRID prompt。
- 用 `model.style_encoder` 和 `model.predictor_encoder` 提取并拼接参考向量，不使用 CAM++ 或 StyleDubber 的 GE2E 向量。
- 文本只用 CelebV-Dub 的 `target_text`，沿用 native espeak phonemizer。
- `emotion_feature` 和 `lip_feature` 分别来自 CelebV-Dub 目标脸部和唇部缓存。
- 保留原生采样：seed 0、alpha=0.0、beta=0.3、diffusion_steps=5、embedding_scale=1。
- `mel_len` 不能由目标 GT WAV 或 GRID 参考时长计算；沿用视频约束 `round(N25 * 24000 / (25 * 300))`。
- checkpoint `net` 的各组件应严格加载；若带 `module.` 前缀只做前缀适配，不用 `strict=False` 掩盖缺失组件。
- 原 `inference` 会删去输出末尾 50 个 24 kHz 采样点，沿用并记录，不因本次换参考而调整模型内部逻辑。
- native 24 kHz 输出重采样到 16 kHz 单声道 PCM16，只保存目标语音。

新服务器需有可用 ProDubber 环境、espeak，以及配置对应的 ASR、JDC、PLBERT 等资源。检查 `ASR_path`、`ASR_config`、`F0_path`、`PLBERT_dir` 和模型配置里的本地路径。

## 五 CelebV-Dub 视觉特征与目标时长

两种原生 Dubber 使用训练时一致的 25 Hz 特征。以目标 `test/0_ArO8UCfyk/0_0` 为例：

```text
{CELEB_NATIVE_FEATURE_ROOT}/extrated_embedding_Grid_152_gray/0_ArO8UCfyk-face-0_0.npy
{CELEB_NATIVE_FEATURE_ROOT}/Grid_VA_feature/0_ArO8UCfyk-feature-0_0.npy
```

唇部 shape 应为 `[N25, 512]`，脸部 shape 应为 `[N25, 256]`；长度、时间起点与同一目标视频一致，全部为有限值。若 ProDubber 的旧配置使用 `VA_feature` 别名，应核对它确为同一套 CelebV-Dub 脸部特征，不是另一个提取器的缓存。

清单中的 `target_frames` 是 **AlignDiT 的 40 Hz 帧数**，不能当成 `N25`。要从对应原生视觉缓存读取 N25，并核对 25 fps 视频时间轴。不能把 AV-HuBERT 的 1024 维缓存截断或投影成 512/256 维来代替原生输入。

对于时长，应明确采用与既有对照一致的已知目标时长协议：

1. 原生生成阶段按目标视频/25 Hz 缓存长度约束总时长。StyleDubber 内部为 `round(N25 * 22050 / (25 * 256))` 个 mel 帧；ProDubber 为前述 80 Hz mel 长度。
2. 为与此前 AlignDiT 输出完全一致，最终 16 kHz WAV 应具有清单 `target_num_samples` 个采样点。这是原测试缓存已有的 **GT 时长元数据**，不是仅视频帧数乘 640，也不应宣称全流程完全未使用 GT 元数据。
3. native hop 舍入、ProDubber 尾端删点、视频帧量化造成的小差值，只在尾部裁去或补零；逐条记录调整量。沿用已有前端的 100 ms 容差，超过 1600 个 16 kHz 采样点则报错核查，不自动修补。
4. 不做整段时间拉伸、不按最优同步偏移移动音频、不添加参考 prompt；目标音频内容和目标声学缓存不能参与生成。

如果改成严格的纯视频时长协议，则应重新评测所有对照，不能把不同输出长度政策的结果放在同一表中不加说明。

## 六 新服务器需要适配的入口

**现有 Setting 2 入口不是本次 GRID 入口，不能直接把本包传给 `--bundle` 或 `--setting2_bundle`。** 它们硬编码了 115 条、`test/...` 参考 ID、同说话人跨句参考和 Setting 2 协议。本次是 213 条、`grid/...` 参考 ID、跨数据集跨身份参考。

在各自工程新增独立 GRID 入口，保留旧入口。至少需要完成：

1. 读取固定 213 条 `pairs.jsonl`，检查目标和参考完整 ID 唯一，按原顺序执行。
2. 单独解析 CelebV-Dub 目标视觉路径与 GRID 参考音频路径，不要求目标和参考 speaker 相同。
3. 从配对 GRID 波形重提各模型原生参考条件，禁止读取目标声学输入。
4. 按上述原生参数严格加载模型并生成；StyleDubber 用 GPU 0，ProDubber 用 GPU 1。
5. 先跑同样的前 3 条小样本，再在另一个独立目录跑全量 213 条；检查正式输出前 3 条与小样本的配置一致。
6. 输出形如 `{OUT}/test/0_ArO8UCfyk/0_0.wav`，共 213 条目标 WAV。参考转换缓存放到 OUT 外的独立位置，避免与目标覆盖统计混淆。
7. 保存输入、权重、代码和输出哈希，输出采样率/长度、N25、尾部调整、seed、模型特有参数及逐条 reference ID。

当前尚没有已实现并验证的 `infer_grid_reference.py` 或对应 ProDubber GRID runner；不要把未来的适配要求当成已有可执行命令。另一台服务器应按本节实现后实际跑通再报告完成。

长任务使用 `setsid`、Python `-u` 和独立日志。设置 `CUDA_VISIBLE_DEVICES=1` 后，ProDubber 进程内部的 `cuda:0` 对应物理 GPU 1，不要再指定内部 `cuda:1`。

## 七 与此前一致的五项评分

| 指标任务 | 对比对象 | 汇总规则与方向 |
|---|---|---|
| `sim` 即 SPKSIM | 生成音频与该条配对 GRID 参考 WAV；WavLM-large 加 ECAPA | 213 条 cosine 的均值，越高越好 |
| `wer` | 生成音频的识别结果与 CelebV-Dub `target_text` | 全语料词级编辑错误除以参考词总数，越低越好 |
| `emosim` | 生成音频与 CelebV-Dub 目标 GT 的 emotion2vec 分类分数向量 | cosine 的样本均值，越高越好 |
| `emoembed` | 生成音频与 CelebV-Dub 目标 GT 的 emotion2vec utterance embedding | cosine 的样本均值，越高越好 |
| `avsync` | 同一目标嘴部视频加生成音频的 AV-HuBERT 联合特征，与同视频加 GT 音频的联合特征 | 先逐帧 cosine 均值再样本均值，越高越好 |

WER 沿用 faster-whisper-large-v3、英语、beam=5 和既有大小写/标点规范化。固定 213 条目标的参考词总数为 **2378**，全量目标采样点为 **11,536,944**，总时长 **721.059 秒**。不能平均逐句 WER；报告 `错误数 / 2378` 和百分数。

SPKSIM 必须对 GRID 参考评分，而不是对 CelebV-Dub 目标原声评分。反之，两种情感指标应对 CelebV-Dub 目标 GT 评分，而不是对 GRID 参考评分。AVSync 是历史 AV-HuBERT 特征相似度，不是 SyncNet/LSE 音画同步指标；跨说话人下情感及 AV 特征分数都只能作辅助诊断。

### 评分代码和资源

复用已有五指标入口：

```text
my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus/
  src/aligndit/script/eval/eval_celebvdub_grid_reference.py
  src/aligndit/script/eval/utils.py
  src/f5_tts/eval/utils_eval.py
  src/aligndit/script/misc/extract_avhubert.py
```

新服务器的评分资源需要与既有对照一致：

```text
wavlm_large_finetune.pth
wavlm_large_s3prl.pt
faster-whisper-large-v3/
emotion2vec_plus_large/
large_vox_iter5.pt
对应版本的 AV-HuBERT 与 fairseq
CelebV-Dub 原始嘴部视频与 GT 联合 AV-HuBERT 特征
```

本次原生模型也可使用这个评分入口，但需要先保存它要求的 `inference_summary.json`：

```text
protocol = celebvdub_grid_reference_one_per_target_v1
pair_manifest_sha256 = 实际交给评分的清单 SHA256
original_pair_manifest_sha256 = 上述冻结原始清单 SHA256
count = 213
partial_smoke_test = false
method = StyleDubber 或 ProDubber
checkpoint = 如实记录 path、sha256、step 或 epoch、实际 weights 类型
generation.target_acoustic_inputs = false
outputs = 原清单顺序的 213 条输出
```

每条 output 至少包含 `pair_id`、`target_id`、`ref_id`、`ref_speaker_id`、`relative_path`（如 `test/0_ArO8UCfyk/0_0.wav`）、`sha256`、`samples`。另存原生输入的来源和哈希。不能把非 EMA 权重写成 EMA，也不能把 epoch 12 伪写成 update 150000 来通过检查。

在完成推理及上述 summary 后，可使用下面的评分命令。所有变量先替换为新服务器实际路径；它不是推理命令：

```bash
export ALIGN=/path/to/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus
export ALIGN_PY=/path/to/metric_environment/bin/python
export PAIRS=/path/to/pairs.runtime.jsonl
export OUT=/path/to/model_grid_reference_output
export WAVLM_FT=/path/to/wavlm_large_finetune.pth
export WAVLM_BASE=/path/to/wavlm_large_s3prl.pt
export ASR=/path/to/faster-whisper-large-v3
export EMO=/path/to/emotion2vec_plus_large
export GPU=0  # StyleDubber；ProDubber 设为 1
cd "$ALIGN"
for task in sim wer emosim emoembed; do
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 PYTHONPATH="$ALIGN/src" \
  "$ALIGN_PY" -u -m aligndit.script.eval.eval_celebvdub_grid_reference \
    --manifest "$PAIRS" --gen-wav-dir "$OUT" -e "$task" \
    --wavlm-ckpt "$WAVLM_FT" --wavlm-base-ckpt "$WAVLM_BASE" \
    --asr-ckpt "$ASR" --emo-ckpt "$EMO"
done
```

AVSync 前，按固定目标清单，用生成音频加目标嘴部视频提取联合特征，保存到 `{OUT}/avhubert_feat/test/{video_id}/{clip}.npy`。下面 `MOUTH_TEST` 应是其下直接包含视频 ID 子目录的目录；检查它与本次 213 个 ID 一一对应：

```bash
export MOUTH_TEST=/path/to/CelebVDub/video_mouth/test/test
export FAIRSEQ_ROOT=/path/to/directory_containing_fairseq_package
export AVHUBERT_USER=/path/to/avhubert_user_directory
export AVHUBERT_CKPT=/path/to/large_vox_iter5.pt
export GT_AV=/path/to/CelebVDub/avhubert_feat
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 \
PYTHONPATH="$ALIGN/src:$FAIRSEQ_ROOT" \
"$ALIGN_PY" -u src/aligndit/script/misc/extract_avhubert.py \
  --nshard 1 --rank 0 --v-input-dir "$MOUTH_TEST" \
  --a-input-dir "$OUT/test" --output-dir "$OUT/avhubert_feat/test" \
  --ckpt-path "$AVHUBERT_CKPT" --user_dir "$AVHUBERT_USER"
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 PYTHONPATH="$ALIGN/src" \
"$ALIGN_PY" -u -m aligndit.script.eval.eval_celebvdub_grid_reference \
  --manifest "$PAIRS" --gen-wav-dir "$OUT" -e avsync --gt-av-feat "$GT_AV"
```

已有 `verify_grid_reference_results.py` 含 AlignDiT 专用 EMA 和 update 检查，不能原样用于这两份原生权重。应新增或适配原生模型的独立核验，而不是篡改 checkpoint 元数据。已有 Setting 2 的评分脚本也硬编码 115 条，不用于本实验。

## 八 完成后的验收与交付

- 两种模型各有 213 个唯一目标 WAV，16 kHz、单声道、PCM16、有限且非静音，逐条长度符合冻结清单。
- 五项指标各覆盖全部 213 个完整 target/pair ID，无漏条、重复、失败后静默跳过或仅用 `0_0` basename 配对。
- 每个模型完整核验 GRID 参考 ID 和音频内容哈希，确认 speaker/style 条件没有来自目标原声。
- 213 份生成 AV 特征与相应 GT shape 相同，有限且完整；不要平均成功子集冒充全量分数。
- 从逐句记录独立复算四个 cosine 均值及 corpus WER；WER 分母核对为 2378；报告五位小数并注明 WER 是比例还是百分数。
- 保存原始与运行清单、输入/权重/代码哈希、参考原生缓存、运行配置、进度、日志、逐条分数、独立核验文件及结果表。
- 结果注明“CelebV-Dub 目标加 GRID 外部参考，213 条，已知目标时长”。不能与历史同片 GT 音频参考或 115 条 Setting 2 结果混作相同协议。

可直接给下一台服务器的执行请求：

> 请按本说明和迁移包的冻结配对，评测 CelebV-Dub 训练的 StyleDubber step 100000 与 ProDubber epoch 12。StyleDubber 使用 GPU 0，ProDubber 使用 GPU 1；不微调、不重新抽 GRID 参考。使用 CelebV-Dub 原训练一致的 25 Hz 唇部/脸部特征及真实目标台词，配对 GRID WAV 编码成各模型原生参考条件；不读取目标声学条件。新增独立 GRID 推理入口，先小样本再完整生成 213 条，按本说明的已知目标时长政策保存目标段。复用相同的 SPKSIM、corpus WER、EMOSIM scores、EMO embedding cosine、AV-HuBERT similarity 五指标，完成全量独立核验后报告结果与产物路径。现有 Setting 2 入口不可直接使用。
