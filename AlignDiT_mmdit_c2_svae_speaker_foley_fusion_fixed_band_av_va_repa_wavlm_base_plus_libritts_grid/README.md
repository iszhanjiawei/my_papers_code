# LibriTTS 50k 初始化的 GRID Foley MM-DiT

GRID 下游训练从已完成的 LibriTTS 50,000 步 EMA 初始化音频和文本模块，重新建立优化器并从第 0 步计数。前 12 个 MM-DiT 块对音视频做 Joint Attention，使用 σ = 4 帧（100 ms）的双向固定高斯时间偏置。更新后的音视频 Query 沿序列维拼接，共享文本 Key/Value。后 6 个块处理音频并接收 CAM++ 说话人条件，第 10 层使用 WavLM-Base+ REPA，权重为 0.1。[模型与训练配置](src/aligndit/config/finetune_grid.yaml)

## 数据与配置

两种配置沿用各自已有的 GRID 训练和验证清单，共用 40 Hz、64 维的 Semantic-VAE 音频 latent、40 Hz 对齐的 1024 维 AV-HuBERT 视频特征、192 维 CAM++ 和 768 维 WavLM-Base+。训练只读取 train 清单，val 留作评测。两种配置各自写入独立的 GRID 权重目录。

| 配置 | 数据目录（相对于 `$ROOT_PREFIX/zjw524/projects/data`） | 数据来源 |
| --- | --- | --- |
| `finetune_grid` | `GRID_svae1000k_sample_seed666_fp32` | 与参考项目的同步 MPG 流程一致。参考缓存含 29,531 条训练样本、3,274 条验证样本 |
| `finetune_grid_local` | `GRID_mmdit_svae` | 当前服务器已有的完整 GRID 缓存，含 29,557 条训练样本、3,281 条验证样本 |

父权重路径为 `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_LibriTTS_foley_text_audio_12CA6A_50k/model_00050000.pt`。训练入口根据字符身份迁移词表权重，加载父模型的 EMA 参数，并为下游 WavLM REPA 投影头初始化参数。[训练入口](src/aligndit/script/train/finetune_grid.py)

## 在当前服务器训练

在本项目根目录设置路径前缀。在 tmux 中可提交八卡 Slurm 作业，使用已有的本机 GRID 缓存。

```bash
export ROOT_PREFIX=/home
NUM_PROCESSES=8 bash sbatch_train_c2.sh --config-name finetune_grid_local
```

直接运行四卡训练时，可执行下列命令。启动脚本检查训练清单和预计可执行的更新步数，启动 TensorBoard，然后开始训练。默认目标为 200,000 步，学习率 5e-5，预热 20,000 步。CTC 权重在第 10,000 至 30,000 步之间线性增加到 0.03。

```bash
ROOT_PREFIX=/home CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash src/aligndit/run/train/finetune_grid.sh --config-name finetune_grid_local
```

## 复现同步 MPG 参考协议

在具有 `${ROOT_PREFIX}/zjw524/datasets/Grid_Dataset` 的机器上运行准备脚本，依据已有 train/val 列表从原始 MPG 提取完整音轨，与嘴部视频对齐后生成特征。已有此格式完整缓存时，可直接运行验证和训练。

```bash
bash scripts/prepare_grid.sh --plan-only
bash scripts/prepare_grid_parallel.sh --gpus 0,1,2,3
bash scripts/prepare_grid.sh --verify-only
bash src/aligndit/run/train/finetune_grid.sh
```

## 检查真实样本和父权重

下列检查使用训练入口中的模型和数据加载函数，以两条真实 GRID 样本执行 BF16 前向和反向，并检查 Foley 文本注意力、说话人和 REPA 投影参数的梯度。

```bash
export ROOT_PREFIX=/home
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" \
  src/aligndit/script/misc/smoke_test_grid_real_parent.py --config-name finetune_grid_local
```

# AlignDiT

Official PyTorch implementation for the following paper:
> **AlignDiT: Multimodal Aligned Diffusion Transformer for Synchronized Speech Generation**<br>
> [Jeongsoo Choi](https://choijeongsoo.github.io), [Ji-Hoon Kim](https://sites.google.com/view/jhoonkim), [Kim Sung-Bin](https://sites.google.com/view/kimsungbin), [Tae-Hyun Oh](https://ami.kaist.ac.kr/members/tae-hyun-oh), [Joon Son Chung](https://mm.kaist.ac.kr/joon)<br>
> ACM MM 2025<br>
> \[[Paper](https://arxiv.org/abs/2504.20629)\] \[[Project](https://mm.kaist.ac.kr/projects/AlignDiT)\]

<div align="center"><img width="80%" src="https://mm.kaist.ac.kr/projects/AlignDiT/imgs/fig1.png?raw=true"/></div>

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
