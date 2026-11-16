# Current experiment: isolated LRS3 training for 400k updates

This directory is a physical copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va_libritts_pretrained`
at Git `78f293f9f5b468a7f645d7e776e2de2c45eb5051`. Model/trainer source is local
to this snapshot. Runtime logs, events and checkpoints were not copied.

The active recipe is
`src/aligndit/config/finetune_lrs3_svae_speaker_window13_libritts.yaml`.
It reads the published strict LRS3 cache at
`${ROOT_PREFIX}/zjw524/datasets/LRS3/AlignDiT_SVAE_LRS3_v2_sharded`:
289,727 training clips / 422.998268 hours, 1,200 validation clips and 1,321 test
clips. This is the accepted training publication, not a lossless restoration
of every official LRS3 clip. The optional P1 publication has a separate config
and output identity; it is not selected by default.

Training starts a fresh downstream optimizer/update counter from the LibriTTS
50k EMA parent, not from a CelebV-Dub checkpoint. It retains Semantic-VAE 64-D
40-Hz latents, 1024-D aligned AV-HuBERT, 192-D CAM++, 12 AVT joint blocks plus
6 audio blocks, the 150-ms Gaussian AV/VA prior, the 159-character vocabulary,
and the same latent normalization. Four GPUs use BF16, 3,600 frames and at most
32 samples per GPU, LR 5e-5, 20k LR warmup and CTC 0 through 10k then a linear
increase to 0.03 at 30k. DataLoader uses 8 workers per rank for LRS3.

`optim.run_until_update: 400000` stops at exactly 400k downstream optimizer
updates. The inherited 200-epoch linear learning-rate schedule is retained;
it does not decay to zero at 400k. The accepted dataset produces 4,623 updates
per epoch on four ranks, so the 200-epoch budget is sufficient. Numbered
checkpoints are saved every 50k and `model_last.pt` every 5k and at normal stop.

From this directory:

```bash
# Validate data/source contracts, parent checkpoint, and freeze the launch plan.
bash train_lrs3_400k.sh
# Validate and launch the supervisor, four ranks and TensorBoard in new sessions.
bash train_lrs3_400k.sh --start
```

The launcher uses setsid via `start_new_session=True`, unbuffered Python and
this snapshot's explicit `PYTHONPATH`; no editable install is needed. After a
stopped run, the same command resumes a matching downstream checkpoint. It
refuses to reuse an active launch or a checkpoint with a conflicting contract.
Use `--help` for optional GPU, port and output overrides.

Default runtime paths:

- Launch plan, frozen YAML, status and logs: `logs/lrs3_400k_strict/`.
- Checkpoints: `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_LRS3_400k_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts`.
- Event directory: `runs/AlignDiT_LRS3_400k_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_semantic_vae_40hz_LRS3_char`.
- TensorBoard logdir: `logs/lrs3_400k_strict/tensorboard/`, port 6032; DDP port 29592.

TensorBoard is started by the launcher and records total/flow/CTC losses and
diagnostics. Its server-local URL is `http://127.0.0.1:6032`; an external URL
requires the IDE's actual port-forwarding service. The PID/status files report
the process from the current launch rather than an assumed PID.

Data caches, source provenance, frozen text recovery, vocabulary, normalization
and pretrained model files remain shared read dependencies. Do not remove the
v1 provenance/raw media or external dependencies listed in the dataset's
`DELIVERY.txt`. No source directory, source experiment checkpoints or source
events are reused for writes by the LRS3 entry above.

The copied CelebV-Dub/Slurm/evaluation scripts and reports below are historical
references; they are not the entry points or results of this LRS3 experiment.
Use `train_lrs3_400k.sh` for this run.

# Historical source experiment: 13-frame Gaussian soft core

This independent copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_av_va`
changes only the effective Gaussian sigma from 0.100 to 0.150 seconds. At the
unchanged 40-Hz feature rate, +/-1 sigma now spans +/-6 frames (13 samples)
instead of +/-4 frames (9 samples). Gaussian tails remain; this is not a hard window.

The 12 AVT joint blocks, 6 audio-only blocks, model weights/layout, offset=0,
AV/VA directions, data, S2c 70k EMA initialization and entire training recipe
are unchanged. Experiment/checkpoint/log identities are isolated.
See [AVT_FIXED_BAND_WINDOW13_AV_VA.md](AVT_FIXED_BAND_WINDOW13_AV_VA.md) for the
current config, commands and checkpoint paths. Other inherited reports describe
prior experiments and are not validation or evaluation results for this run.

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
