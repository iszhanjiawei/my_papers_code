# GRID 双台词条件实验

本副本基于原 GRID adaptive-band + CAM++ + WavLM-Base+ REPA 实验。输入同一说话人的参考音频和对应台词 A，以及目标嘴部视频和台词 B。输出长度和内容监督均由 B 决定。

## 模型和时间轴

- 两句共享字符词表与编码器权重，各自从位置零编码，在文本卷积前加入可学习的参考或目标角色嵌入。推理支持结构化参数 `reference_text=[A], text=[B]`、标记形式 `text=["[REF] A [TGT] B"]`，也支持 GRID 旧接口的 `text=[A + "  " + B]`。标记用于拆分输入，不占用字符词表。单句内部含多个连续空格时应使用显式标记形式。
- 参考 latent 作为查询，读取参考文本的键和值。注意力结果逐帧调节参考声学特征，再对有效 P 帧取平均，得到一个 128 维向量。它经过零初始化输出投影，与来自同一参考音频的 CAM++ 一起调制第 12–17 块（从 0 编号）。池化提供信息瓶颈，不保证风格向量完全不含词信息。
- 主干保留 P+T 布局，训练和推理都在输入卷积前屏蔽前 P 帧的噪声状态和逐帧参考条件。只有后 T 帧接收目标文本交叉注意力的更新。目标嘴部视频与后 T 帧对齐。解码器只接收目标区域，输出截到目标的真实采样点数。
- 全条件与 TTS 两个 CFG 分支使用参考风格，无条件分支屏蔽它。参考文本和声学条件在同次 ODE 求解中固定。
- train 清单内按说话人、台词和片段 ID 生成确定的一对一异句配对；每个片段各作一次目标和参考。配对算法与有序 ID 对的 SHA256 写入训练 contract。数据集返回参考 CAM++、两个角色的文本及逐样本 P，虚拟参考视频帧在前，真实目标视频在后。帧预算按 P+T 计。
- 流匹配仅对目标有效帧计算损失。CTC 在投影前取出每个样本的目标隐藏状态，对齐 B 的字符标签。REPA 把 B 的 WavLM 特征对齐到 B 自己的有效 T 帧。

## 训练

在本副本根目录执行。默认沿用与基线相同的 S2c 70k EMA 初始化、词表和标准化，启动新优化器和更新计数。新角色模块从本次初始化开始学习。默认训练上限为 200k 次更新。训练入口会先核验缓存和权重，再启动本副本的 TensorBoard。

```bash
mkdir -p logs
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_grid.sh \
  > logs/train_grid_roles.log 2>&1 < /dev/null &
```

配置为 [finetune_grid.yaml](src/aligndit/config/finetune_grid.yaml)。使用已审计的全量 GRID 特征缓存；TensorBoard 写入本副本的 `runs/`，权重目录名包含 `GRID_role_conditioned`。默认保留每 50k 次更新的权重。若仅需做开发验证，可运行：

```bash
source env.sh
export PYTHONPATH=src
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$PYTHON" -m unittest discover -s tests
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/aligndit/script/misc/smoke_test_grid_real_parent.py \
  --cache-root "${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32"
```

第二条检查加载真实父权重，验证不同长度样本的正反向、CTC、REPA、梯度，以及包含两段台词的 CFG 采样，不进行参数更新。

## Setting 2 推理与评估

以训练得到的 EMA 100k 权重为例，`--step` 可改成其他已经保存的更新数。使用新输出目录。此入口默认选择角色分路模式，按配置定位 checkpoint 并严格校验权重，逐条记录参考与目标文本。

```bash
source env.sh
eval_dir="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_role_conditioned_s2_100k_seed0"
mkdir -p "$eval_dir"
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_s2_100k.sh \
  --step 100000 --gpus 0,1 --workers-per-gpu 1 --output-dir "$eval_dir" \
  > "$eval_dir/run.log" 2>&1 < /dev/null &
```

除 SPKSIM、WER、EMOSIM、AVSync 外，`evaluation_summary.json` 还给出 `reference_word_matches`。分母是参考与目标台词同位置的词不同的槽位数，分子是目标转写错误且恰好等于该位置参考词的槽位数。额外插入的词没有对应槽位，仍计入 WER。原 100k 基线在相同 3,274 条上的历史对照为拼接文本 1,010/15,908、仅目标文本 4/15,908。新模型的效果需要用训练后的权重测量。

## 实现验证

2026-11-18 使用完整训练清单核对了 29,531 对样本，所有配对均为同说话人异句，参考无重复占用。有序配对 SHA256 为 `5f723308be28c1b4b549ecc3d216882de4afe5bee21cf7c7149a8a8171f3c804`。真实 S2c 70k EMA 的 303 个共享张量精确迁移，13 个新角色张量独立初始化。GPU 检查使用 P 为 120 和 119 的两个样本，CTC 权重取 0 和 0.03 均完成反向传播，角色分支得到有限非零梯度。

生产推理入口另外验证了 `val/s6/bwba2a`。使用原 GRID EMA 与新初始化角色参数组成的临时权重，输入参考句 `place red at c two soon` 与目标句 `bin white by a two again`，按 120+120 帧采样，真实解码器输出目标段 48,000 个采样点（16 kHz，3 秒）。这次验证覆盖权重加载、分路、解码、逐条记录与长度核验。正式 WER 和参考词误用率仍待新模型训练后测量。
