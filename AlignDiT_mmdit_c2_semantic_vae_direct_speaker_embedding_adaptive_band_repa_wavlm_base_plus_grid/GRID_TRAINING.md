# AlignDiT-MM-DiT on GRID

本目录从同级 `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus`
复制，代码独立，不导入该目录的 Python 包。原项目、现有 GRID baseline 和数据缓存只读复用。
所有命令从本目录执行；`env.sh` 自动识别当前服务器的 `/home` 路径前缀，
也可以显式设置 `ROOT_PREFIX`。使用已有 `aligndit` 环境，不重新安装 editable package。

## 数据与模型

- 复用 `projects/aligndit_project_gird/alignDiT_baseline/AlignDiT/data_grid` 的
  29,557 条 train、3,281 条 val，以及原 baseline 的异常样本排除结果。
  两个集合按 utterance 隔离，但包含相同说话人，不是未见说话人测试协议。
- 使用从原始 MPG 完整音轨对齐得到的 16 kHz WAV 和 AV-HuBERT Large video-only
  1024D/25 Hz 特征。不能用 GRID 中经过裁剪的 WAV，也不能用 StyleDubber 的 512D lip 特征替代。
- 新缓存为 `${ROOT_PREFIX}/zjw524/projects/data/GRID_mmdit_svae`，包含固定 posterior sample
  64D/40 Hz latent、按单条有效长度插值的 1024D/40 Hz video、L2 CAM++ 192D speaker embedding、
  WavLM-Base+ 第 12 层 768D/50 Hz FP16 REPA 特征。REPA 在加载时插值到实际 latent 长度。
- 使用同一句完整 GT 音频提取 speaker/REPA，保留原训练条件语义。val 不进入训练迭代器。
  该入口不执行自动验证集生成或四指标评测。
- 复用 S2c LibriSpeech 70k EMA 初始化，重新创建 GRID optimizer 和 update 计数。
  保留原迁移器的 EMA counter 语义。固定 LibriSpeech latent mean/std 和原字符词表，
  不用 GRID val 重新估计归一化，不从 CelebV-Dub 训练中间权重恢复。
- 本机的 S2c、Semantic-VAE 1000k EMA、CAM++ 和 WavLM-Base+ 文件已找到，
  预处理核对其 SHA256，不自动下载。具体来源和哈希写入缓存 contract。

## 配置

主配置：`src/aligndit/config/finetune_grid_mmdit.yaml`。
保持 18 层（前 12 层 MM-DiT）、后 6 层 CAM++ 条件、adaptive temporal band、
第 10 个 MM block 的 REPA、REPA 权重 0.1 和原 CTC 0→0.03/10k→30k 调度。
学习率 5e-5、warmup 20k、BF16、3,600 latent frames/GPU、seed 666。

GRID 使用 **100,000 optimizer updates** 的学习率调度与停止点，
不沿用在 GRID 上可能提前结束的 CelebV-Dub 200-epoch 循环。
编号权重 `model_20000.pt`、`model_40000.pt`、…、`model_100000.pt` 全部保留；
`model_last.pt` 每 5,000 步更新用于恢复。
checkpoint 目录独立，数据及配置 contract 不符时拒绝恢复。
恢复包含 optimizer、scheduler、EMA；不承诺逐位复现中断时的随机流。

## 准备、检查、训练

一条命令执行完整预处理、全量校验，再启动正式训练：

```bash
mkdir -p logs
setsid env PREP_GPU=7 TRAIN_GPUS=3,4,5,6 TENSORBOARD_PORT=6017 \
  bash scripts/run_grid.sh > logs/grid_pipeline.log 2>&1 < /dev/null &
```

如果已经单独启动本副本的全量预处理，可给这个命令设置
`WAIT_FOR_EXISTING_PREP=1`。流水线等待预处理文件锁释放，并且只有完整
32,838 条特征的校验通过才启动训练；预处理失败会终止流水线。

也可以分别执行：

```bash
source env.sh
# GPU 索引仅为示例，启动前检查 nvidia-smi。
PREP_GPU=7 bash scripts/prepare_grid.sh

# launcher 自动启动并检查 TensorBoard；使用独立 session 跑长任务。
mkdir -p logs
setsid env TRAIN_GPUS=3,4,5,6 TENSORBOARD_PORT=6017 \
  bash scripts/train_grid.sh > logs/train_grid.log 2>&1 < /dev/null &
```

`scripts/prepare_grid.sh` 支持恢复未完成提取。完整特征审计成功后才发布
`data_contract.json` 与 `complete.json`，缺失、损坏或只准备了调试子集时拒绝正式训练。
小样本诊断必须使用独立 cache 和 checkpoint 目录，不能混入正式缓存。

训练日志在本目录 `logs/`，TensorBoard event 在 `runs/`，
包括总 loss、diffusion、CTC、REPA 和 adaptive-band/speaker 梯度诊断。
`scripts/start_grid_tensorboard.sh` 输出实际 PID、端口、logdir 和本机 HTTP 地址。
远程 IDE 需将该端口转发后访问，脚本不能生成 IDE 的转发 URL。

默认 checkpoint 目录：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_GRID_svae_speaker_adaptive_band_repa_wavlm_base_plus_100k
```

恢复时重新运行相同训练命令。请保持 GPU 数、batch、seed 和其他训练配置不变。
训练运行状态以本次日志、进程与 TensorBoard 为准；这里不记录易过期的 PID 或步数。

## 实际验证

- 原 adaptive-band 的 19 项、band + REPA 的 8 项模型回归检查通过。
- 真实 GRID 74/75 帧样本的数据提取与 checksum 审计通过；对应 latent 119/120 帧、
  WavLM 147/149 帧，CAM++ norm 为 1。
- 实际 S2c EMA 严格迁移，真实 GRID BF16 前向/反向在 CTC=0、0.03 均通过；
  speaker、band offset/width、REPA 梯度有限且非零。
- 4×A40 调试运行跨 epoch 完成 update 1–2、保存完整 checkpoint；
  重启后成功恢复至 update 2，并完成 update 3 和再次保存。
- TensorBoard event 实际包含 loss、diff_loss、repa_loss、CTC 调度与模块诊断。
  CTC warmup 前权重为 0，原模型不计算原始 ctc_loss；正式生效后才记录该分项。
- 新增源码静态检查通过；继承的 `trainer_vt.py` 存在原有 lint/格式告警，未进行无关重构。

详细运行证据在忽略 Git 的 `logs/smoke_grid_real.log`、
`logs/train_grid_smoke.log`、`logs/train_grid_smoke_resume.log`。
短检查只验证执行链路；长期训练稳定性和配音质量需看正式训练及后续评测结果。

## 正式训练与 100k 评估结果（2026-10-07）

正式训练已完成 100,000 optimizer updates，并保存 `model_100000.pt` 与
`model_last.pt`。本次评估使用 `model_100000.pt` 的 EMA 参数；checkpoint SHA256 为
`e435062c33a26115eac145a2de018ddcdb86b50a40869d3731c22afb8274afa3`。

评估覆盖共享 GRID validation manifest 的全部 3,281 条样本，采用 Setting 2：每个目标使用
同说话人、不同 validation 句子的参考音频提取 prompt latent 和 CAM++，推理时不读取目标音频。
采样参数为 Euler/EPSS、NFE 32、sway -1、text/video CFG 5/2，基础 seed 666，并按
manifest 全局序号为每条样本派生 seed。指标均用完整集合计算；WER 使用数字展开并按全语料
词级编辑距离统计。

| Checkpoint | 样本数 | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| EMA 100k | 3,281 | 0.66492 | 0.49101 | 0.80150 | 0.68977 |

完整 WAV、逐样本指标、日志及机器可读汇总保存在：

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_GRID_svae_speaker_adaptive_band_repa_wavlm_base_plus_100k/eval_grid_setting2_100000_seed666_cfg5_2
```

`metrics_summary.json` SHA256 为
`9865e56fa34ca4ba7d5493ea399e0af3fe530742884fe08742329cfc13cd4ccb`。
