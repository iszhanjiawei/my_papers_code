# GRID 训练

本目录是 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus`
的独立源码副本，先逐文件复制并验证一致，再接入 GRID。原目录保持不变。模型仍使用
Semantic-VAE Direct-C2、CAM++ speaker embedding、adaptive temporal band 和 WavLM-Base+ REPA。

本次已完成全量源数据审计、正式清单冻结及真实小样本特征/模型验证。
全量特征提取与训练状态见末尾本机运行记录；下方命令用于复现和断点续跑。

## 数据协议

- 数据根目录：`${ROOT_PREFIX}/zjw524/datasets/Grid_Dataset`。
- 按用户确认，沿用 `0_Grid_Wav_22050_Abs_Feature/Grid_Wav_22050_Abs_Feature/train.txt`
  和 `val.txt`。原始列表分别有 29,694 / 3,300 条；缺损样本逐项记录，绝不重新分配 split。
  这是现有的说话人重叠划分，没有独立 test split；val 不进入训练。
- 清除文本中的 `sp`、`sil`、`<sil>` 静音标记，校验六个单词与 GRID 文件名编码一致。
- 现有 `audio_25k` / 22,050 Hz WAV 含静音裁剪，不能直接与原视频配对训练。
  从原始 MPG 完整音轨解码，按音频和视频首时间戳差对齐为单声道 16 kHz PCM16，
  仅在 120 ms 容差内补齐/裁剪至实际视频帧时长；超过容差或解码失败即报告失败。
- 视频须为 25 fps、实际解码 74 或 75 帧，并与连续编号的 96×96 灰度嘴部 JPEG 一致。
  校验每张 JPEG、MPG 解码、音视频时间戳，保存输入哈希及排除原因。
  沿用数据集提供的嘴部裁剪，其原始人脸对齐算法未随数据提供。
- 在这批嘴部图像上运行 AV-HuBERT Large **video-only**（`audio=None`），按 checkpoint
  的中心裁剪与归一化提取 1,024 维特征，再线性插值到 40 Hz latent 长度。
  现有的 512 维 lip / 256 维 face 特征不会作为本模型的输入。
- 从同一份完整、对齐后的 WAV 提取固定 Semantic-VAE posterior sample（64 维 / 40 Hz）、
  CAM++（192 维 L2 单位向量）与 WavLM-Base+ 最后一层（768 维 FP16）目标。
  后验采样按 utterance ID 和 seed 666 确定；REPA 对齐、掩码与损失沿用原模型。
- 为兼容 S2c 70k EMA 初始化，复用原模型的字符词表和固定 LibriSpeech latent 标准化参数；
  不使用验证集估计统计量。所有 checkpoint、词表和标准化资源均校验 SHA256。

## 准备

在本目录运行。脚本设置 `PYTHONPATH=本目录/src`，优先使用现有 `aligndit` 环境；无需
`pip install -e .`，不会覆盖其他实验的 editable 安装。另一种机器目录布局可设置 `ROOT_PREFIX=/home`。

```bash
source env.sh
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
GRID_CACHE="${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32"

# 全量源数据审计及清单；不加载 GPU 模型。
bash scripts/prepare_grid.sh --plan-only

# 全部特征：复用冻结清单，12 个 CPU worker 提取音频，再使用四张 GPU。
# 可中断后用相同命令续跑；日志与逐 worker 状态保存到缓存 parallel_runs/。
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash scripts/prepare_grid_parallel.sh \
  --gpus 0,1,2,3 --gpu-workers-per-device 2 --audio-workers 12 \
  > logs/grid_prepare_parallel.log 2>&1 < /dev/null &
PREPARE_PID=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$PREPARE_PID"

# 全量校验完成标记和缓存，不重新提取模型特征。
bash scripts/prepare_grid.sh --verify-only
```

可用 `--output-dir` 指定另一份独立缓存，`--workers` 控制源数据审计并行度。
若机器缺少 WavLM-Base+ 权重，可在首次准备时加 `--download-missing-models`，只下载并校验
固定 revision 的三个文件，默认存入 `${ROOT_PREFIX}/zjw524/projects/data/pretrained_models/wavlm-base-plus`。
也可用 `--wavlm-model-dir` 指向相同版本的本地权重。
`--stage audio|latent|video|speaker|repa` 可独立执行对应阶段；只有全部阶段完成且校验通过，
才发布训练所需的 `complete.json`。输入、模型或协议不一致时拒绝混用旧缓存。
断点续跑会核对已有文件内容与提取记录；不要手工伪造完成标记。
并行入口仅将原始清单分成互不重叠的行，各 worker 复用原提取函数、逐条随机种子和
数值配置。父进程持有唯一写锁，所有 worker 成功并通过全量校验后才发布 `complete.json`；
任一 worker 失败会停止本轮其余 worker，保留已校验产物。单 GPU 串行入口仍为
`CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_grid.sh`。
`--gpu-workers-per-device` 默认 1；本机短片提取经验证可设为 2，以增加逐帧读取与推理的
并发度。该参数只改变任务调度，每个样本仍由唯一 worker 按原数值协议提取。

小样本检查使用独立目录，每个 split 各取 N 条，输出明确标记为 smoke，正式训练器拒绝使用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_grid.sh \
  --output-dir "${GRID_CACHE}_smoke" --limit 2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" -u \
  src/aligndit/script/misc/smoke_test_grid_real_parent.py \
  --cache-root "${GRID_CACHE}_smoke"
```

此检查加载真实 S2c 权重与 GRID 特征，验证 CTC 权重 0 / 0.03 时的 BF16 前向、总损失、
speaker / adaptive-band / REPA 梯度；不执行 optimizer update，不保存训练 checkpoint。

缓存结构：

```text
GRID_svae1000k_sample_seed666_fp32/
  manifests/{inventory,train,val}.jsonl
  manifests/inventory_meta.json
  audio/{train,val}/s*/id.wav
  latents/{train,val}/s*/id.npy
  video_40hz/{train,val}/s*/id.npy
  speakers/{train,val}/s*/id.npy
  repa/{train,val}/s*/id.npy
  complete.json
```

## 训练

数据准备和全量校验完成后：

```bash
mkdir -p logs
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_grid.sh \
  > logs/train_grid.log 2>&1 < /dev/null &
TRAIN_PID=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$TRAIN_PID"
tail -f logs/train_grid.log
```

单卡可设 `CUDA_VISIBLE_DEVICES=0`，脚本按可见 GPU 数启动对应 worker；也可显式指定
`NUM_PROCESSES` 和 `TRAIN_PORT`。Hydra 参数直接追加，例如
`datasets.num_workers=4 datasets.batch_size_per_gpu=2400`。
启动前检查实际 batch 数、总步数和完整缓存，防止小 batch / 多 GPU 配置提前耗尽 epoch。

主配置：[finetune_grid.yaml](src/aligndit/config/finetune_grid.yaml)。默认参数：

| 参数 | 值 |
| --- | --- |
| 初始化 | 原模型相同、SHA 固定的 S2c 70k EMA，GRID optimizer/update 从零开始 |
| 每卡 batch | 3,600 latent 帧，最多 32 条 |
| 学习率 / LR warmup | 5e-5 / 20,000 updates |
| CTC | 前 10k 为 0，10k–30k 线性增加到 0.03 |
| REPA | 固定 0.1，从第一个 update 启用 |
| 精度 / seed | BF16 / 666 |
| 训练停止 | 200,000 optimizer updates |
| epoch / LR 衰减期限 | 1,000 epochs，适配更少的 GRID batch；与原 CelebV-Dub 200 epochs 不同 |
| checkpoint | 每 50k 保存里程碑，每 5k 更新 `model_last.pt` |

帧 batch 经固定种子按 epoch 打乱；DDP 丢弃末尾不足一个 GPU 组的 batch（每轮最多
`GPU数−1` 个 batch），各 rank 更新次数一致。实际数量和调度信息写入训练 contract。
重复启动会从本实验 checkpoint 续训；改变数据或训练配置须用新的 `ckpts.save_dir`，
避免与旧 contract 冲突。

GRID checkpoint 默认存于：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char
```

训练入口会同时启动 TensorBoard（默认寻找 6020 起的空闲端口，可设置 `TENSORBOARD_PORT`），
记录 PID、端口和 logdir 到 `logs/tensorboard_grid.json`。打开底部“端口”页签，找到实际端口，
点击该行“转发地址”链接。只有主 rank 写 event，包含总损失、diff/CTC/REPA 分项及各模块梯度。
logdir 为本副本 `runs/` 下的 GRID run，checkpoint 和日志不会写入原实验目录。

当前实现提供 GRID 训练及前向/反向检查。val 留作后续独立生成评测；本目录继承的 CelebV-Dub
推理脚本有其原始测试协议，不能直接作为 GRID 验证集评测入口。

## 本机验证记录（2026-10-12）

本次实际保留训练 **29,531** 条、验证 **3,274** 条，两个 split 无 utterance 交集，均含 33 名说话人。
训练集含 67 条 74 帧样本，验证集含 6 条，其余均为 75 帧，总时长约 27.33669 小时。
共排除 189 条：156 条嘴部帧不足、30 条 MPG 解码损坏、3 条缺少音频或视频流。
原有列表之外的 6 条原始视频不加入任何 split。

全量审计和逐条排除记录位于：

```text
${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32/source_audit/
  source_records.jsonl
  source_exclusions.jsonl
  source_summary.json
  correction_report.json
```

修正记录保留了初轮审计对 MPEG 容器时长估计过严造成的 15 条误排；最终以实际解码帧数
和完整嘴部帧为准。当前数量由本地数据实际审计得到，不使用其他实验 README 的历史数量。
正式 `manifests/` 已生成，首次全量特征提取会复用该清单，不再重复解码审计整个数据集。

- 35 项测试通过：源数据/时间轴 13 项、数据集/多卡调度 12 项、缓存完整性/断点续跑 10 项。
- 配置解析、Python 编译、Shell 语法及 Git diff 检查通过。
- 2 条 train + 2 条 val 的全部 20 个音频/特征文件完成真实提取与哈希校验。
- 真实 S2c EMA 严格迁移：313 个来源张量、714 个目标张量，导入 303 个、忽略 10 个、初始化 411 个。
- 两条真实 GRID 训练样本完成 BF16 前向/反向：CTC 权重为 0 和 0.03 时总损失及
  speaker/adaptive-band/REPA 梯度均有限且非零；未执行 optimizer update。
  小 batch 峰值显存约 2.63 GiB，不代表正式训练 batch 的显存需求。
- 13 项并行调度测试通过，验证精确分片、写锁、失败清理、同卡并发和完整性发布约束。
- 四卡小样本并行与同卡双进程分别复现原串行流程的全部 20 个输出 SHA256；
  另有 64 条视频特征在同卡并发下逐条 SHA256 一致。短时窗口中，同卡三个视频 worker
  合计约 20.7 条/秒，单 worker 约 8 条/秒；正式续跑采用较保守的每卡两个 worker。
- 将两条真实样本重复到 30 条的容量检查通过：BF16 前向/反向峰值约 8.81 GiB，
  CTC 权重 0 / 0.03 时损失和梯度均正常。此检查未执行 optimizer update；额外的
  EMA、Adam 两份动量和 DDP bucket 估算约 4.91 GiB，实际训练显存以四卡实跑为准。

检查日志在本副本的 `logs/grid_final_tests.log`、`logs/grid_prepare_smoke_final.log`、
`logs/grid_model_smoke.log` 中，日志与缓存均不提交 Git。
容量检查日志为 `logs/grid_batch30_capacity.log`。
