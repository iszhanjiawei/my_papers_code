# Chem 训练

本目录是原 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus` 的独立代码副本。原目录不修改。Chem 配置继承用户指定的 CelebV-Dub 配置，替换数据、缓存、输出位置与训练总步数。

## 启动与恢复

在本目录执行；`env.sh` 默认 `ROOT_PREFIX=/home`，也接受显式空值以使用云服务器 `/zjw524` 路径。使用现有 `aligndit` Python，通过 `PYTHONPATH=src` 加载本副本，不修改环境的 editable-install 指向。

```bash
source env.sh
CUDA_VISIBLE_DEVICES=5 bash src/aligndit/run/train/finetune_chem_svae_speaker_adaptive_band_repa_wavlm_base_plus.sh
```

后台执行示例（先检查 GPU 可用资源；项目锁会拒绝重复启动）：

```bash
mkdir -p output
setsid env ROOT_PREFIX=/home CUDA_VISIBLE_DEVICES=5 \
  bash src/aligndit/run/train/finetune_chem_svae_speaker_adaptive_band_repa_wavlm_base_plus.sh \
  > output/chem_train.log 2>&1 < /dev/null &
```

配置：`src/aligndit/config/finetune_chem_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml`。

- 总计 **120,000 optimizer updates**；单 GPU、BF16、3600 个 40 Hz 音频帧/批、梯度累积 1。
- 每 **20,000** 步保留 `model_20000.pt` … `model_120000.pt`；每 **5,000** 步更新 `model_last.pt`，结束时也保存。
- 权重目录：`ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_Chem_120k/`。
- 相同命令自动从步数最大的完整 checkpoint 恢复模型、AdamW、scheduler、EMA、随机状态和批位置。临时写入后原子替换，不覆盖仍完整的旧文件；配置/数据合同不一致会拒绝恢复。
- 学习率 `5e-5`、20k warmup、原 LinearLR 和 EMA 参数保留。显式设置 120k scheduler horizon，并按实际批次数计算足够的 epoch，避免小数据集沿用 200 epochs 而提前结束。

TensorBoard：

```bash
TENSORBOARD_PORT=6064 bash scripts/start_adaptive_band_repa_tensorboard.sh
```

服务读取本目录 `runs/`；本机为 `http://127.0.0.1:6064`，服务器内网为 `http://10.109.119.146:6064`。远程 IDE 可转发 6064 端口。正式曲线和独立 smoke 曲线使用不同 run。实际启动 PID、路径、命令记录在 `output/chem_runtime.json`；服务日志在 `logs/tensorboard_adaptive_band_repa_6064.log`。

## 数据与模型

数据源是 `/home/zjw524/datasets/Chem_dataset`。复用已验收的基线 Chem 划分和纯视觉 AV-HuBERT 特征：
`/home/zjw524/projects/aligndit_project_gird/aligndit_project_chem/alignDiT_baseline/AlignDiT/data_chem_v2`。

| 划分 | 条数 | 用途 |
| --- | ---: | --- |
| train | 5821 | 梯度更新、WavLM REPA |
| val | 311 | 留出的验证集 |
| test | 196 | 原测试集 |

保留原训练入口的训练行为，本次未新增自动验证/测试指标或音频生成评测。验证视频与训练、测试视频均不重叠；原 train/test 划分本身共享一个视频 ID `myN3PqD38Ds`，没有调整原测试集，也不把它描述成说话人/视频完全互斥的测试集。

| 输入 | 格式 | 本目录缓存 |
| --- | --- | --- |
| Semantic-VAE | 64 维 FP32，40 Hz，固定 posterior sample | `data_chem/svae1000k_sample_seed666_fp32/latents` |
| 口型 AV-HuBERT | 1024 维 FP32，25 Hz 线性插值到 latent 等长 | `data_chem/svae1000k_sample_seed666_fp32/video_40hz` |
| CAM++ | 192 维 FP32，L2 归一化 | `data_chem/campplus_spk_emb_zh_en_16k` |
| WavLM Base+ | 最后第 12 层，768 维 FP16，约 50 Hz | `data_chem/wavlm_base_plus_repa_final_fp16` |

所有音频保留原 16 kHz 单声道 WAV。SVAE 仅在右侧补零至 400 samples 的整数倍，所有 latent 共 1,272,032 帧；原种子协议为 `SHA256("666:" + utterance_key)` 派生逐样本 seed。训练时继续使用 **LibriSpeech 固定归一化统计**，不在 Chem 上重估。CAM++ 和 WavLM 从完整原始音频离线提取，训练不更新这两个 teacher。WavLM 逐样本插值到有效 latent 长度，只在生成区间计算 REPA。

复用基线已有的全局音画 offset=0 估计。视频时长与音频的量化差不超过约 20 ms；这是全局同步估计和帧率转换，并非对所有片段精确同步的证明。

保持 12 个 MM block + 6 个 audio block、共享 adaptive temporal band、CAM++ 线性投影注入 block 12–17、REPA 第 10 个 block/projector 和 `lambda=0.1`。CTC 前 10k 步权重为 0，10k–30k 线性增加到 0.03。起始阶段日志没有非零 CTC loss 是原配置的预期行为。

初始化仍为 S2c **70k EMA parent**，Chem optimizer 和 update 从零开始；严格迁移验证 313 个源 key / 714 个目标 key，其中加载 303、明确忽略 10、新建 411。不是从 CelebV-Dub 完整训练 checkpoint 接续。

预训练资产：

- SVAE：`/home/zjw524/projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k`。
- S2c：`/home/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`。
- 固定归一化：`/home/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json`。
- WavLM：`/home/zjw524/projects/data/wavlm-base-plus`，原配置指定 revision/SHA 校验通过。
- CAM++：本副本 `data_chem/pretrained_models/campplus/campplus_cn_en_common.pt`，来自 ModelScope 官方 `iic/speech_campplus_sv_zh_en_16k-common_advanced` v1.0.0，SHA 与原配置一致。资产来源和哈希见 `output/chem_audio_teacher_assets.json`。

## 重建和检查

缓存已经生成。以下命令可以复用完成文件并重新校验；读取基线与上述固定权重路径，因此重建时仍需保留这些资产。

```bash
source env.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$PYTHON_BIN" src/aligndit/script/misc/prepare_chem_semantic_vae.py --stages manifest,video,latent,audit
bash src/aligndit/run/misc/extract_chem_audio_teachers.sh
"$PYTHON_BIN" src/aligndit/script/misc/audit_chem_multimodal.py
```

官方 SVAE 导入所需新增的 11 个小依赖固定在 `requirements-chem-preprocess.txt`，以 `pip --no-deps` 安装；未升级现有 Torch、Torchaudio、NumPy、Transformers、Protobuf 或 TensorBoard。

验收记录位于本副本 `output/` 和缓存 `state/`，数据、日志、events、权重不入 Git：

- `output/chem_multimodal_audit.json`：全 6328 条真实音频、latent、video、speaker，以及 5821 条 REPA 与真实短/长样本 loader 校验。
- `data_chem/svae1000k_sample_seed666_fp32/state/audit.json`：全量数组/hash/固定归一化检查。
- 同目录 `state/latents/golden_test.json`：原 LibriSpeech golden latent SHA 完全一致。
- `output/chem_audio_teacher_audit.json`：全部 teacher 输出、归一化与逐条音频/REPA 帧数检查。
- `output/chem_real_parent_smoke.log`：真实 S2c+Chem，CTC 0 / 0.03 两种前后向；speaker、band、REPA 梯度有限且非零。
- `output/chem_gpu_train_first.log`、`output/chem_gpu_train_resume.log`：实际 3600-frame AdamW 保存与恢复检查，单独的权重/曲线目录。
- `output/source_snapshot.json`：复制时全部 264 个源文件的 SHA256；`output/source_unchanged_audit.json`：最终源目录复核结果。

修改集中在 Chem 数据准备、路径/合同、独立 trainer 的步数和恢复逻辑；未改 backbone、CFM、VAE 或 REPA 损失公式。
