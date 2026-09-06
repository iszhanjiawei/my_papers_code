# D1：AV-HuBERT 双角色表征监督

本快照独立复制自 `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio`，基线配置是用户指定的
`finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope.yaml`。源码为独立文件，不通过软链接或硬链接共享；数据、预训练权重使用已有外部路径。

## 实验定义

- 视觉角色：保留原有 AV-HuBERT visual-only 嘴唇特征条件。
- 音频角色：使用同一 `large_vox_iter5.pt` 预训练体系的冻结 audio-only 路径，为真实训练语音生成最终 Transformer 层的 1024 维上下文表征。不是音视频联合输入特征，不使用现有 `avhubert_feat` 联合特征充当音频目标。
- 学生位置：第 6 个 block（零基索引 5），即最后一个 MM block 的音频隐藏状态。将 100 Hz 隐状态每 4 帧平均池化为 25 Hz，再经 `Linear(768,768) → SiLU → Linear(768,1024)` 投影。
- 表征损失：有效生成区域上的平均 `1 - cosine(projected_hidden, stop_gradient(audio_teacher))`，以 FP32 计算。按固定时间网格对应，教师真实长度和生成掩码取交集；不把 padding 当作目标，不按 padded 序列长度拉伸。教师尾部和视频/mel 存在少量帧数差时截取共同有效区域。
- 总损失：`flow_loss + 0.1 × mean(CTC_layer6, CTC_layer12) + 0.1 × avhubert_rep_loss`。
- 第 6 层和权重 0.1 是首轮实验选择，尚未调优或验证性能收益。
- 本次只实现表征监督；没有实现 TPCA。

音频教师由 trainer 单独持有，冻结并处于 eval 模式，不进入生成器参数、DDP、optimizer、EMA 或生成器 checkpoint。GT 音频只用于监督目标，不作为生成器条件。推理仍使用原有文本、嘴唇与参考音频，不需要目标音频或 AV-HuBERT 音频教师；默认 backbone 两元素返回接口保持兼容。

## 数据与缓存

实际训练集沿用 `CelebVDub_char/raw.arrow`。数据集同时保留每条 mel/video 样本的原音频路径，音频教师按相同顺序提取目标，读取失败直接报错，避免错配或隐式丢样本。

音频预处理沿用 AV-HuBERT：16 kHz PCM 幅值、26 维 log-filterbank、25 ms 窗长和 10 ms 帧移、每 4 帧堆叠，依据 checkpoint 配置逐帧归一化。音频不足完整 stack 的最后一帧沿用教师预处理补零规则，batch padding 则独立遮蔽。

每个训练 rank 使用冻结 FP32 教师，microbatch 为 4。首次遇到样本时计算特征，随后从独立缓存读取。缓存位于 `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/avhubert_audio_teacher_cache` 的身份子目录；记录 checkpoint 文件路径/大小/修改时间、提取语义和源码摘要，逐样本记录音频文件身份，并原子写入。缓存保存 FP16，首次计算结果也经同样量化后返回，避免命中前后目标精度不一致。该缓存为训练产物，不提交 Git。

## 运行

新配置：`src/aligndit/config/finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope_avhubert_dual_role.yaml`。

正式配置继承基线的 4 GPU、BF16、每 GPU 9000 mel 帧、最多 32 条样本、学习率 5e-5、20,000 次 warmup、200 epochs、CTC 权重 0.1、LibriSpeech 500k 音频预训练初始化、`checkpoint_activations=False`、`log_samples=True` 及原 checkpoint 保存策略。新模型名称确保不会恢复或覆盖历史 D1 checkpoint。

```bash
bash scripts/start_avhubert_dual_role_training.sh
```

启动器用两个独立 `setsid` session 启动训练与 TensorBoard，默认 master port 29586、TensorBoard port 6008；启动前检查端口。运行记录、PID、训练日志和服务日志保存在本快照 `logs/`。启动记录不代表运行验证完成：需要实际检查四个 worker、有限损失、GPU、event 更新与 TensorBoard HTTP。不要重复运行启动器占用相同 GPU。

TensorBoard 记录 `loss`、`diff_loss`、`ctc_loss`、`avhubert_rep_loss`、`avhubert_rep_weighted_loss`、`avhubert_rep_valid_frames`、`lr`，仅全局 rank 0 写 event。

```text
runs/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_AVHuBERTDualRole_L6_W01_finetune_hifigan_16k_char_CelebVDub
```

checkpoint 路径为 `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_AVHuBERTDualRole_L6_W01_finetune_hifigan_16k_CelebVDub_char`。

如果需要在 Devin 中访问 TensorBoard，打开底部「端口」页签，在本次端口所在行点击「转发地址」。客户端转发 URL 无法从当前工具直接读取时，运行记录保持 `forwarded_address: null`，不能用猜测地址替代。

## 验证边界

已通过 7 项独立 CPU 单元检查和原 Hunyuan 架构 CPU 回归。真实音频教师测试确认长度、冻结状态、缓存完全一致及 BF16 外层环境下教师 FP32 提取。完整 268,784,146 参数 D1 已在两条真实训练样本上通过 BF16 前向/反向检查，投影头有非零梯度，所有已有梯度与各分项损失有限；此检查不执行 optimizer update。

复查命令：

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -m unittest discover -s tests -p test_avhubert_dual_role.py -v
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u scripts/check_avhubert_dual_role_real_batch.py --output logs/real_batch_gradient_check.json
```

单元检查应覆盖：有效生成区域与教师真实长度的交集、忽略 padding/参考区域、教师无梯度、投影头有梯度、固定 4 帧池化、功能关闭时的参数兼容及推理默认接口。实际教师检查应使用真实语音核对形状、长度、冻结状态和缓存一致性。正式启动还需验证真实 D1 的反向传播和四卡运行。

损失正常下降仅代表训练正常进行，不能据此断言 WER、AVSync 或音质改善；仍需与原 D1 做相同推理设置下的独立评测。
