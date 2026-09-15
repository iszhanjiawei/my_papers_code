# Semantic-VAE Direct-C2 + CAM++ + Synchformer

本实验从 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` 独立复制，新增 HunyuanVideo-Foley 风格的视频同步条件。源码、配置和运行目录属于本实验；数据、预训练权重和特征缓存保存在项目外。原实验的运行日志、checkpoint 和 TensorBoard 文件没有复制。

## 1. 新增了什么

```text
原始 RGB 视频 → 冻结 Synchformer 视觉编码器 → [窗口数 × 8, 768] 离线缓存
                                                    ↓
                               窗口内位置编码 + Linear / SiLU / 门控 MLP
                                                    ↓
                               按每个样本实际长度插值至 40 Hz 时间轴
                                                    ↓
                 同步条件 + flow 时间步 → 各层 AdaLN / gate → 生成音频 latent
```

- 保留原来的 AV-HuBERT 视频条件、文字条件、64 维 / 40 Hz Semantic-VAE latent 和 192 维 CAM++ speaker 条件。
- Synchformer 使用预训练视觉编码器，提取时冻结；训练只读取缓存，GPU 中不加载此编码器。
- 沿用 Foley 的 8 个窗口内位置编码与门控 MLP，把同步条件加入 12 个多模态块的音频和视频流、6 个音频块，以及最终输出 AdaLN。speaker 仍只进入零起始编号 12–17 的音频块。
- 对每个样本的有效同步 token 单独做 `nearest-exact` 插值，再补零。不同长度样本组成 batch 时，padding 不改变有效片段的时间比例。
- 同步条件和 AV-HuBERT 视频一起被 CFG 丢弃；完整条件分支保留两者，TTS 和空条件分支移除两者。音频 prompt 所覆盖的区域也执行原模型的互补视频遮罩。
- 新增 MLP 的最后一层初始化为零，以保留迁移后的原音频路径；加载原音频模型后，该层可以在第一步获得梯度。

针对现有模型的两处适配是：丢弃视频时把投影后的同步条件置零，不另设 Foley 的可训练 empty-sync token；对每个样本单独对齐并保留原来的 prompt 互补遮罩。新增参数的零初始化用于兼容已有音频父模型。

这是针对现有配音架构接入同步条件的改动。未复现 Foley 的整套 SigLIP2、CLAP、DAC-VAE、REPA 或训练数据方案；效果需要通过后续配音评测确认。

## 2. 需要下载的权重

只需新增官方 [`synchformer_state_dict.pth`](https://huggingface.co/tencent/HunyuanVideo-Foley/blob/3abd4e833b95b8db0fc9c687afc52483a48e9a97/synchformer_state_dict.pth)，950,058,171 字节。现有 Semantic-VAE、CAM++、AV-HuBERT 缓存与 S2c 70k 训练父权重继续复用。

默认下载位置：

```text
/zjw524/projects/data/pretrained_models/HunyuanVideo-Foley/synchformer_state_dict.pth
```

官方文件 SHA256：

```text
8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c
```

```bash
bash scripts/download_synchformer.sh
```

脚本固定 Hugging Face revision，支持中断后续传，并核对最终 SHA256。默认使用并行 HTTP 分段下载；`DOWNLOAD_WORKERS=1` 使用标准 `hf download`。`SYNCHFORMER_MODEL_DIR` 可覆盖目录。整个 checkpoint 包含音频编码器和同步分类头，但本任务仅严格加载全部 `vfeat_extractor.*` 权重。

视觉编码器源码放在 `src/aligndit/third_party/synchformer/`，来源为本机 Foley 复现代码 commit `df7b005b5023df2a9b73e1d66dd51d452799884e`。附带来源、LICENSE 和 NOTICE；无跨实验源码软链接。

## 3. 数据处理约定

输入为 `/zjw524/projects/data/CelebVDub/video/{train,test}/<video_id>/<clip>.mp4` 的原 RGB 视频，允许其指向原始数据的软链接。现有 AV-HuBERT `.npy` 不能用来计算 Synchformer 特征；也不使用灰度嘴部裁剪替代原视频。

1. 按视频时间戳采样为 **25 fps**。
2. RGB，短边缩放为 **224**，bicubic + antialias，中心裁剪 **224 × 224**，以均值和标准差 `[0.5, 0.5, 0.5]` 归一化。
3. **16 帧窗口、8 帧步长**；每窗输出 **8 × 768**，按窗口顺序拼接，保存 float16 特征。
4. 不足 16 帧时重复最后一帧；其余片段沿用 Foley 的完整窗口规则，末尾不足一个步长的帧不另起窗口。记录补帧数和未入窗的尾帧数。
5. 完整处理每个视频，不采用官方推理示例中的 15 秒上限。原始采样时间、源文件大小和修改时间、模型 SHA256、预处理参数均写入缓存元数据。
6. 以 `split/video_id/clip` 完整相对路径作为键；缓存原子写入，续跑逐项校验。全部完成后逐项检查张量、源文件对应关系和 inventory 覆盖，再生成完整性报告。

固定 inventory 含 **79,826** 条：训练 **79,613** 条，测试 **213** 条；约 **91.26 小时**，最长 **29.96 秒**。其中 **2,452** 条短于 16 帧。特征张量预计约 **10.5 GiB**，另有少量元数据与文件格式开销。

缓存默认目录：

```text
/zjw524/projects/data/CelebVDub/synchformer_25fps_16f_stride8
```

## 4. 环境

直接使用现有环境，无需重新安装 editable 项目：

```bash
source env.sh
export PYTHONPATH=src
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
```

当前环境提供 PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121 和 PyAV 11.0.0。仅提取特征需要额外的 `av`、`torchvision`，其余使用现有 `einops` 和 `omegaconf`；视觉编码器不依赖完整 Foley 软件栈。新机器可参考 `pyproject.toml` 的 `synchformer` extra，保持 torch/torchvision 版本匹配。

## 5. 训练与推理

一条命令完成下载、四卡提取、全量缓存 audit 和训练前检查，可中断后重跑：

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash scripts/prepare_synchformer.sh \
  > logs/prepare_synchformer.log 2>&1 < /dev/null &
```

`SYNC_GPUS=0,1,2,3`、`SYNC_BATCH_SIZE=8` 可覆盖提取使用的 GPU 和每次前向的窗口数。默认 `SYNC_WORKERS_PER_GPU=4`，即 4 张卡共 16 个进程，利用 CPU 并行视频解码；CPU/内存资源较少时可设为 1 或 2。提取子入口为 `scripts/extract_synchformer_multigpu.sh`；单独 audit 使用 `PYTHONPATH=src python scripts/extract_synchformer.py --audit-only`。`--limit` 仅用于调试，不能生成完整训练所需的通过证明。全量 audit 默认使用 8 个 CPU 进程（`SYNC_AUDIT_WORKERS` 可覆盖），逐项读取并校验全部缓存；不加载 Synchformer 编码器。逐卡日志在 `logs/synchformer_extraction/`。 每个 worker 默认 RSS 上限为 6144 MiB（`SYNC_MAX_WORKER_RSS_MIB` 可覆盖）；进程异常退出、超过该内存上限或 cgroup 内存使用达到 90% 时，监督进程会停止其余 worker，避免带故障继续运行。已写入且校验通过的缓存可直接续用。PyAV 11 的解码上下文会显式关闭，每 16 个新提取视频执行垃圾回收并归还可释放的 glibc 内存，避免大量视频累积占用。

新训练配置继承原 speaker 实验：S2c 70k EMA 初始化、学习率 `5e-5`、20k LR warmup、CTC 在 10k 前为 0，在 30k 增至 `0.03`、200 epoch 的 LR 调度范围、最多 200k updates、每卡 3,600 latent frames、4 张 GPU。新增独立 checkpoint 和 TensorBoard run 名。

训练前检查（需要先完成全量同步特征提取与 audit）：

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u scripts/preflight_synchformer.py
```

正式训练：

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_synchformer_4x4090.sh \
  > logs/train_synchformer.log 2>&1 < /dev/null &
```

入口先完成数据和父权重校验，并实际启动 TensorBoard。默认端口 **6007**，可通过 `TENSORBOARD_PORT` 覆盖；`logs/tensorboard_synchformer.json` 保存本次 PID、端口及确切 logdir。TensorBoard 除原有 flow/CTC/speaker 曲线外，还记录同步输入和输出投影梯度、输出权重范数和窗口内位置编码范数。

推理入口：

```bash
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_synchformer.sh
```

沿用原 CelebVDub Setting 1 协议：同一 GT 音频片段作为 prompt，生成对应目标片段；speaker 取同一 prompt，目标视频对应的 Synchformer 特征拼接同长度的空 prompt 前缀。推理读取新 checkpoint 的同步分支参数及测试缓存，不需要在线加载 Synchformer。现有 Setting 1 不是独立参考音频协议，结果描述应保持这一限制。

## 6. 验证记录

实际完成的检查、下载、缓存覆盖和训练状态见本目录的 `SYNCHFORMER_STATUS.md`。只有全量 audit 与 preflight 通过后，才能确认现有数据可直接用于正式训练。
