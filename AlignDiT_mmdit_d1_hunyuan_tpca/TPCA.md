# D1 Hunyuan Dual CA / All-RoPE + TPCA

## 来源与隔离

本实验从 `../AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` 的 208 个 Git 跟踪文件独立复制。
源仓库提交：`83554362d7231b8a7bdf667e798d7525b807091f`。源配置为
`src/aligndit/config/finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope.yaml`。
原项目未修改；未复制原训练日志、TensorBoard events、checkpoint 或缓存。
两个项目没有共享源码软链接。数据和 LibriSpeech 音频预训练权重沿用既有外部路径。

本次只实现 TPCA，没有加入 AV-HuBERT 音频分支表征监督。新配置：
`src/aligndit/config/finetune_celebvdub_mm_d1_hunyuan_tpca.yaml`。

## 实现

1. **视觉条件对齐。** 独立 `OccurrenceCTCAligner` 读取原始 AV-HuBERT 视觉特征，使用
   LayerNorm、256 维投影、depthwise 时间卷积和字符分类头。识别头不接收台词。
   每个 25 Hz 视频帧预测两个具有独立分类参数的子位置，形成 50 Hz CTC 发射网格；
   这提高路径容量，并不增加新的视觉观测，也不是把同一份 logits 重复两次。
2. **台词出现位置后验。** 给定台词，以 float32 log-space CTC forward–backward
   计算扩展状态后验。同一字符的不同出现位置保持不同列；相邻相同字符必须经 blank。
   blank 状态聚合到最后的 null 列，两个子位置平均映射回对应 25 Hz 视频帧。
   blank 不是静音标签，该后验不是音素持续时长真值。对齐 CTC 保留梯度，后验 detach。
3. **实际路径组合。** 在第 4、5、6 个 MM block（零基 `[3,4,5]`），读取前 4 个
   AV joint-attention heads 实际使用的 Q/K（已经 QK norm 和 RoPE），提取音频→视频
   子分布，仅在有效且非 complementary-mask 的视频 keys 上重新归一化，得到 R。
   分查询块计算 `C = R @ P`，没有另建 Q/K 对齐网络，也没有用固定对角矩阵代替 R。
   这是在“选择视觉 keys”条件下的分布，不等同于完整 joint attention 分给视频的总质量。
4. **局部文本读取。** 对同层前 4 个音频→文本 heads 加入 null key/value（均为零），
   用 `softmax(S + bias_strength * log(C_smooth))` 读取文本。
   `C_smooth = 0.9*C + 0.1*Uniform(valid target positions + null)`。
   其余 8 个文本 heads、全部视频→文本 CA 和后 12 层 audio-only refinement 保留。
5. **独立 raw-attention 一致性。** 对未加入 prior 的 `softmax(S)` 计算
   `KL(stop_gradient(C_smooth) || softmax(S))`；不是比较已经被 prior 修正后的注意力。
   只统计有效生成音频 queries，按 queries×受约束 heads 归一化，再对选定层取均值。
   R/P 到 teacher prior 的路径 stop-gradient，原生成路径仍正常反传。

代码：`src/aligndit/model/tpca.py`、`tpca_attention.py` 和 `backbone/dit_vt_mm.py`。
CTC 约束视觉→台词的合法顺序；R 仍是可学习的全局注意力，**整个 TPCA 不保证硬单调**。
没有额外加入固定时间窗或高斯 AV 注意力先验。

## 损失与预热

```
loss = flow_loss + 0.1 * mean(audio_ctc_layer6, audio_ctc_layer12)
       + 0.03 * visual_ctc_loss
       + 0.01 * path_scale * raw_path_kl
```

`path_scale = clamp((completed_updates - 2000) / 8000, 0, 1)`。
前 2,000 次更新只训练新视觉 CTC，局部注意力输出与原 CA 一致；随后逐步启用 prior 和 KL，
第 10,000 次完成更新时 scale 为 1。`bias_strength = path_scale`。
online 与 EMA 的 `tpca_step` 是持久化整数 buffer，恢复训练沿用已完成更新数。

默认 CFG 保持原配置：缺失 text 或 video 的分支完全禁用 TPCA prior 和新损失；
仅丢弃 audio prompt 时，仍可使用 text+video。推理 packed CFG 只有 full 分支启用 TPCA。
训练日志中的 `tpca_active` 标记条件是否齐全；warmup 是否结束需另看 `tpca_path_scale`。
原有音频 CTC 的 raw ID/blank 约定不变。新视觉 CTC 使用 raw IDs `0..V-1`、blank `V`。

真实 79,613 条训练记录中，原生 25 Hz 字符 CTC 有 9,743 条长度不可行；
50 Hz 网格仅有 3 条不满足 `frames >= tokens + adjacent repeated characters`。
这些样本不会被移除出训练集，仍参与原损失，但排除新 CTC/path loss，并记录 feasible fraction。

## prompt / target、padding 与兼容修复

- 训练仍是完整台词加随机音频缺失片段。独立对齐头看完整原始视频；生成视频流仍沿用
  complementary masking。新 prior/KL 只作用生成区域，其 R 不使用 prompt 区域的视频 keys。
- 推理视频由零填充参考段和目标视频拼接。新增 `sample(prompt_text_lens=...)`，显式给出
  拼接台词中目标字符的起点；视频起点由对齐到 4 倍网格的 prompt 音频长度确定。
  只将目标嘴唇对齐目标台词，S1 的两份相同台词不混为同一次 occurrence。
  `utils.py`、`infer.py` 和训练器定期采样均已接入。带 prompt 的 TPCA 调用缺少边界会报错。
- 保留全局 heads 读取整句语境，局部 prior 的均匀下限仅覆盖目标文本位置与 null。
- 新副本内修复原 B>1 视频 mask 的错误上采样、packed CFG batch 顺序错配，以及空文本
  SDPA 在 Torch 2.4 上潜在的 NaN 反向。源项目保持原样。
- 推理按条件缓存 P，并在 sample 开始和退出（包括异常）时清理。
  sample 将固定 inference-mode 条件转换为有版本计数的普通无梯度 tensor，闭包内复用；
  直接调用 backbone 时，无 mutation counter 的外来条件仍禁用缓存以避免错误复用。
  R/C 每层每步更新。

## 可复核验证

```
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u tests/test_tpca_alignment.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u tests/test_tpca_attention.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u tests/test_tpca_cfm.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_hunyuan_dual_ca.py
PYTHONPATH=src OMP_NUM_THREADS=2 /zjw524/ENTER/envs/aligndit/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=4 tests/smoke_tpca_ddp.py
PYTHONPATH=src OMP_NUM_THREADS=4 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/validate_tpca_training.py --device cuda:0 --frames 9000 --batch-size 4 --pretrained
```

测试覆盖穷举 CTC 路径、重复字符 occurrence、独立子帧、padding/spans、真实 R 数值对照、
raw KL 梯度、global/prompt 保留、CFG 隔离、预热等价、激活 checkpoint 等价、cache、
四卡 BF16 不同分支和预热切换，以及真实 D1/数据/预训练权重的完整前反向压力检查。
这些验证说明实现与训练链路可运行，不说明 WER/AVSync 已经改善。

## 正式训练与曲线

正式入口 `scripts/train_tpca_4x4090.sh` 继承原实验条件：4 GPU、BF16、9,000 frames/GPU、
最多 32 样本/GPU、16 workers/GPU、200 epochs、LR=5e-5、LR warmup=20,000、seed=666、
activation checkpointing=False、log_samples=True、audio CTC=0.1。
使用同一 LibriSpeech 500k 纯音频 EMA 权重初始化，不恢复历史 D1 多模态训练状态。
新 model.name 隔离 checkpoints/Hydra outputs/events；运行始终显式优先本副本的 `src`。

```
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash scripts/train_tpca_4x4090.sh \
  > logs/train_tpca.log 2>&1 < /dev/null &
setsid env PYTHONUNBUFFERED=1 TPCA_TB_PORT=6006 bash scripts/tensorboard_tpca.sh \
  > logs/tensorboard_tpca.log 2>&1 < /dev/null &
```

脚本支持 `CUDA_VISIBLE_DEVICES`、`TPCA_MASTER_PORT`、`TPCA_CONFIG_NAME` 和 Hydra overrides；
`TPCA_DRY_RUN=1` 只打印命令。不要对已有正式 run 任意改结构后续训。

TensorBoard 由 global main 记录 `loss / diff_loss / ctc_loss / tpca_visual_ctc_loss /
tpca_path_loss`，并记录两项加权损失、实际权重、path_scale、active、feasible_fraction、
active_fraction、bias_strength、lr。事件是 rank 0 当前 batch 的值，不是跨卡平均值。
logdir 为本项目 `runs/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_TPCA_6MM12A_CTC6_12_finetune_hifigan_16k_char_CelebVDub`。
实际 PID、端口、启动命令和验证结果保存在未纳入 Git 的 `logs/launch_status.json`。

当前工具不能读取 Devin UI 的“端口”转发地址；须在底部“端口”页签找到实际监听端口，
点击该行“转发地址”的链接。本机 HTTP 检查与 UI 外部转发检查是两项不同的验证，不能以猜测 URL 代替。

## 后续实验对照

- `model.arch.tpca_enabled=false`：本副本中的原 D1 架构。
- `model.arch.tpca_bias_strength=0`：保留同容量视觉头及视觉 CTC，关闭 prior 和 path KL。
- `model.tpca_path_lambda=0`：保留 prior，去除 raw KL。

每项对照必须给独立 model.name/输出目录；完整结果再比较 WER 的替换/删除/插入项、
重复词/长句分组、AVSync、音质、SIM 与 EMOSIM，不预先认定某项必然提升。
