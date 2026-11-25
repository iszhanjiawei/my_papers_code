# AlignDiT · GRID 配音训练（AT → AV，Window13，WavLM-Base+ REPA）

本项目复制自 `AlignDiT_mmdit_c2_svae_speaker_pair_at_then_av_fixed_band_window13_av_va_libritts_pretrain_grid`。GRID 训练从配对模型的 LibriTTS 50,000 步 EMA 权重 `AlignDiT_LibriTTS_svae_speaker_pair_at_then_av_6AT6AV_50k/model_00050000.pt` 初始化，从第 0 步开始更新。前 12 层先用 6 个头融合音频与文本，再用 6 个头融合音频与视频。双向音视频注意力使用标准差 0.150 秒的固定高斯偏置。后 6 层通过 CAM++ 说话人向量调节音频。[GRID 配置](src/aligndit/config/finetune_grid.yaml)

GRID 入口读取 `${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32` 中的 40 Hz Semantic-VAE、AV-HuBERT 和 CAM++ 特征。沿用参考实验按语句划分的 29,531 条训练样本和 3,274 条留出样本，训练只读取训练分片的特征。字符词表和潜变量归一化沿用原配音模型。加载父权重时按字符和 CTC blank 映射词表相关参数。

训练从第 1 次更新起，将第 10 个 AT → AV 融合块输出的音频状态送入 768 → 2048 → 2048 → 768 的 SiLU 投影头。教师为冻结的 WavLM-Base+ 第 12 层，特征来自每条完整录音，在 GRID 缓存的 `repa/train/` 下按语句读取。每条 50 Hz 教师特征裁去填充并线性插值到对应的 40 Hz 有效长度，只在流匹配损失的生成帧上计算余弦距离。总损失为 `flow + 0.1 × REPA + λ_CTC(step) × CTC`。推理沿用原有流程，无需教师特征。

在本目录执行以下命令。启动器校验缓存、父权重与四卡训练步数，启动 TensorBoard 并进入训练。

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/finetune_grid.sh > logs/train_grid.log 2>&1 < /dev/null &
```

默认使用 4 张 RTX 4090、BF16、每卡 3,600 帧和种子 666。学习率调度沿用 GRID 参考配置的 1,000 个 epoch 预算。训练在第 100,000 步保存 `model_100000.pt` 后停止。学习率为 `5e-5`，预热 20,000 步。CTC 在前 10,000 步的权重为 0，到 30,000 步线性升至 0.03。最新断点每 5,000 步保存一次，带步数的权重每 50,000 步保存一次。

权重目录为 `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_pair_at_then_av_fixed_band_window13_av_va_libritts50k_GRID_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_char`。TensorBoard 读取当前副本的 `runs/`，默认从端口 6067 开始选择可用端口。实际端口和进程号记录在 `logs/tensorboard_grid.json`。TensorBoard 中的 `repa_loss` 为原始余弦损失，`repa_weighted_loss` 为其对总损失的贡献。

在正式训练前可使用以下命令，以 30 条真实训练语句检查 CTC 权重为 0 和 0.03 时的总损失、教师填充、投影头梯度及显存占用。

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/check_grid_repa_gpu.py
```

## 来源项目的 LibriTTS 与 CelebV-Dub 配方

本快照复制自 `AlignDiT_mmdit_c2_svae_speaker_pair_at_then_av_fixed_band_window13_av_va`。默认前 12 层重复同一个 **AT → AV 融合块**，后 6 层只更新音频。先让音频与台词做双流联合注意力，把六个 AT 头的输出投影回 768 维并加到音频。AV 的六个头从更新后的音频重新计算 Q/K/V，再与嘴型做双流联合注意力。文本因此可以在同一层经音频影响嘴型。

每个双流注意力都同时更新音频和对应条件。视频和文本各自保留 768 维状态，传入下一融合层。每个头 64 维，音频头 0–5 用于 AV，6–11 用于 AT。每层共用原来的音频输入和输出投影参数，每种模态各执行一次 FFN。位置编码沿用基线配置，音频与视频的全局第一个头使用时间 RoPE。训练继续使用 CelebV-Dub 的 64 维、40 Hz Semantic-VAE 表征、音频尾部 CAM++ 说话人条件和原 CTC 配置。

窗口沿用 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va_libritts_pretrained` 的 `fixed_band_window13_av_va` 设置。每个融合层的第二阶段在音频查询与视频键的注意力分数上加入固定高斯偏置，视频查询与音频键使用其转置。音视频各为 40 Hz，中心偏移为 0，标准差为 0.150 秒（6 帧）。中心及左右各 6 帧构成 13 帧核心，核心外的偏置随时间差平方连续衰减。每个查询仍在全部有效音频键和视频键上统一归一化，同模态的分数和第一阶段 AT 注意力沿用基线。窗口不引入可训练参数。

## LibriTTS 预训练（8 × RTX 4090）

在完整 LibriTTS train-clean-100 和 train-clean-360 上训练 50,000 步。从 LibriSpeech S2c 70k EMA 初始化音频参数。前 12 层使用配对模型中的 AT 六个头，视觉分支与 Window13 在本阶段关闭。后 6 层使用音频和 CAM++ 说话人条件。默认种子 666，bf16，每卡 2,400 帧，8 卡每次更新的名义预算为 19,200 帧。CTC 权重为 0。训练每 1,000 步保存断点，最终输出 `model_00050000.pt`。[预训练配置](src/aligndit/config/pretrain_libritts_text_audio.yaml)

在项目根目录执行以下命令。特征提取默认使用 GPU 0–7，每张卡运行两个进程，共 16 个进程。每个进程处理独立的数据分片，逐条生成 Semantic-VAE latent 和 CAM++ 向量，复用通过校验的已有记录。可用 `WORKERS_PER_GPU` 调整提取并发数。正式预训练使用每张卡一个 DDP 进程。

```bash
bash src/aligndit/run/misc/prepare_libritts_text_audio.sh all
source env.sh
pretrain_event_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_LibriTTS_svae_speaker_pair_at_then_av_6AT6AV_50k/tensorboard"
TENSORBOARD_LOGDIR="$pretrain_event_dir" TENSORBOARD_PORT=6064 bash scripts/start_speaker_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/pretrain_libritts_text_audio_8x4090.sh > logs/pretrain_libritts.log 2>&1 < /dev/null &
```

预训练权重目录为 `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_LibriTTS_svae_speaker_pair_at_then_av_6AT6AV_50k`。TensorBoard 事件位于其中的 `tensorboard/`。可将 `ckpts.resume=/绝对路径/model_NNNNNNNN.pt` 传给启动脚本恢复同一实验。

## CelebV-Dub 下游训练

在本项目根目录运行。启动器读取本目录的 `env.sh`，通过 `PYTHONPATH` 导入本份源码。默认使用本机四张 GPU、bf16、每卡 3,600 个 latent 帧，从本项目训练完成的 LibriTTS 50k EMA 初始化，训练至第 200,000 次更新。有 `/home/zjw524` 路径前缀的服务器请先执行 `export ROOT_PREFIX=/home`，并确认训练数据缓存、父权重及环境已就位。

```bash
bash scripts/start_speaker_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh > logs/train.log 2>&1 < /dev/null &
```

正式配置同时设置 `paired_joint_order: at_then_av` 和 `temporal_band_enabled: true`。训练、推理、评测和 TensorBoard 使用本实验独立的路径。默认 DDP 端口为 29654，下游 TensorBoard 端口为 6065。[训练配置](src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml)

## 结构验证与层数消融

在本目录运行结构测试，可以核对双向窗口与显式 softmax 的数值结果、同一层中条件传递的方向、CFG 和反向传播，以及新旧权重的识别。

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -m aligndit.script.misc.smoke_test_paired_joint
```

在原 AT → AV 模型基础上，本 GRID 实验的 REPA 投影头新增 7,344,896 个可训练参数。

固定 S2c 父模型的总深度为 18。比如在训练配置中把 `model.arch.n_mm_layers` 改为 8，就得到 8 层融合加 10 层音频。文本流和说话人条件的边界会一同调整。层数消融请使用单独的 `model.name` 和 `ckpts.save_dir`，并在推理时选用同一配置。

## CelebV-Dub Setting 1 推理和评测

训练保存第 200,000 步权重后，运行以下命令即可生成样本并计算四项指标。若要选择其他已保存步数，设置 `CKPT_STEP`。

```bash
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

只生成音频时，可单独运行 [推理脚本](src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh)，用 `OUTPUT_DIR` 指定独立输出目录。

原三流对照的历史文档随源码保留，其训练记录及指标属于 AVT 基线。[基线说明](AVT_JOINT_ATTENTION.md) · [基线评测](EVALUATION_S1_100K_150K_200K.md)

# AlignDiT

Official PyTorch implementation for the following paper:
> **AlignDiT: Multimodal Aligned Diffusion Transformer for Synchronized Speech Generation**<br>
> [Jeongsoo Choi](https://choijeongsoo.github.io), [Ji-Hoon Kim](https://sites.google.com/view/jhoonkim), [Kim Sung-Bin](https://sites.google.com/view/kimsungbin), [Tae-Hyun Oh](https://ami.kaist.ac.kr/members/tae-hyun-oh), [Joon Son Chung](https://mm.kaist.ac.kr/joon)<br>
> ACM MM 2025<br>
> \[[Paper](https://arxiv.org/abs/2504.20629)\] \[[Project](https://mm.kaist.ac.kr/projects/AlignDiT)\]

<div align="center"><img width="80%" src="https://mm.kaist.ac.kr/projects/AlignDiT/imgs/fig1.png?raw=true"/></div>

## Datasets
To help verify your setup and ensure reproducibility, we provide the following debug data.

  Path | Dataset | Debug data
  :---:|:---:|:---:
  data/LibriSpeech_debug | [LibriSpeech](https://www.openslr.org/12) | [download](https://drive.google.com/file/d/1tO8qcddtfhL7yh_svDr9Jaif-mNHdflL/view)
  data/LRS3_debug | [LRS3](https://mmai.io/datasets/lip_reading) | [download](https://drive.google.com/file/d/1u-AbgT_O4Xl14XKnui-IVCMoA0mZEJRU/view)

## Model Checkpoints
  Path | Train Dataset | Model
  :---:|:---:|:---:
  ckpts/AlignDiT_pretrain_hifigan_16k_LibriSpeech_notext/model_500000.pt | [LibriSpeech](https://www.openslr.org/12) | [download](https://drive.google.com/file/d/1-LddbcvUQmPOy_iLJ7I85eaMaYx1kgav/view)
  ckpts/AlignDiT_finetune_hifigan_16k_LRS3_char/model_400000.pt | [LRS3](https://mmai.io/datasets/lip_reading) | [download](https://drive.google.com/file/d/1gDRpB-5UKYy3NDz6uysNNMZQ25leXLrV/view)

## Test Samples
We provide audio samples generated by AlignDiT. For VTS task, we use a lip reading model ([Auto-AVSR](https://github.com/mpc001/auto_avsr)) to transcribe text from the silent video before inference.

  Task | Test Dataset | WER ↓ | AVSync ↑ | spkSIM ↑ | Samples
  :---:|:---:|:---:|:---:|:---:|:---:|
  ADR (automated dialogue replacement) | [LRS3-cross](https://github.com/kaistmm/AlignDiT/blob/main/data/lrs3_test_cross_sentence.lst) | 1.401 | 0.751 | 0.515 | [download](https://drive.google.com/file/d/1amU29Ax_E2USyJz5IjXwC9GFaBGlnCDk/view)
  VTS (video-to-speech synthesis) | [LRS3-cross](https://github.com/kaistmm/AlignDiT/blob/main/data/lrs3_test_cross_sentence.lst) | 19.513 | 0.688 | 0.508 | [download](https://drive.google.com/file/d/1L3MeKwACURdJ-ihxFml0kbK6B1ZyLk57/view)

<br>

## 1. Installation
```bash
conda create -y -n aligndit python=3.10 && conda activate aligndit

git clone https://github.com/kaistmm/AlignDiT.git && cd AlignDiT

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e .[eval]  # For evaluation
```

## 2. Data Preparation
We crop the mouth region from each video following [Auto-AVSR](https://github.com/mpc001/auto_avsr) and place the resulting videos in `data/LRS3_debug/autoavsr/video`. For both training and inference, we use these cropped videos.

#### Metadata
```bash
bash src/aligndit/run/misc/prepare_librispeech_notext.sh
bash src/aligndit/run/misc/prepare_lrs3.sh
```

#### Mel spectrogram
```bash
bash src/aligndit/run/misc/extract_mel.sh
```

#### HuBERT feature
```bash
bash src/aligndit/run/misc/extract_hubert.sh
```

#### AV-HuBERT video feature
This requires [Fairseq](https://github.com/facebookresearch/fairseq) and [AV-HuBERT](https://github.com/facebookresearch/av_hubert).
```bash
bash src/aligndit/run/misc/extract_avhubert_from_only_video.sh
```

## 3. Training
```bash
# 1. Pre-train on LibriSpeech for 500k updates
bash src/aligndit/run/train/pretrain.sh

# 2. Fine-tune on LRS3 for 400k updates
bash src/aligndit/run/train/finetune.sh
```

## 4. Inference
```bash
# ADR (automated dialogue replacement)
bash src/aligndit/run/eval/infer.sh

# VTS (video-to-speech synthesis)
bash src/aligndit/run/eval/infer_w_lipreader.sh
```

## 5. Evaluation
We follow [F5-TTS](https://github.com/SWivid/F5-TTS) for evaluation. Further details are avilable [here](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval).
```bash
# For AVSync metric, run this script beforehand
bash src/aligndit/run/misc/extract_avhubert.sh

bash src/aligndit/run/eval/eval_lrs3_test.sh
```

<br>

## Acknowledgement
This repository is built using [F5-TTS](https://github.com/SWivid/F5-TTS), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [Fairseq](https://github.com/pytorch/fairseq), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [HiFi-GAN](https://github.com/jik876/hifi-gan), [V2SFlow](https://github.com/kaistmm/V2SFlow). We appreciate the open source of the projects.

## Citation
If our work is useful for you, please cite the following paper:
```bibtex
@inproceedings{choi2025aligndit,
  title={AlignDiT: Multimodal Aligned Diffusion Transformer for Synchronized Speech Generation},
  author={Choi, Jeongsoo and Kim, Ji-Hoon and Sung-Bin, Kim and Oh, Tae-Hyun and Chung, Joon Son},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025}
}
```

## License
This project is released under the MIT License. Please note that the use of [AV-HuBERT](https://github.com/facebookresearch/av_hubert) models is subject to their original license terms.
