# 在没有说话人 ID 的 CelebV-Dub 上训练双台词模型

> 以下记录来自复制前的角色条件实验。本副本的结构、父权重、配置和训练入口见 [当前实验](README.md)。

角色由样本在任务中的用途决定。参考 A 提供音色和风格，目标 B 提供要读出的台词和对应嘴部视频。模型不需要整套数据的说话人编号。训练构造 A 和 B 时，同声异句的可信度会影响监督质量。若把不相关人声当作参考，目标 B 的音色监督会与 A 冲突。

## 训练对

1. 只读取锁定 SHA256 的 CelebV-Dub 训练清单。按源视频和去掉最后一个下划线字段的片段名前缀收集候选。例如同一视频中的 `13_0_1`、`13_0_2` 可以互为候选。前缀来自公开的数据配对代码，只表示候选范围。[VoiceCraft-Dub 配对实现](https://github.com/kaist-ami/voicecraft-dub/blob/main/data/construct_dataset.py#L82-L92)
2. 对每个目标 B，选择另一片段作为参考 A，要求 A 长度为 2–10 秒。用完整片段的 192 维 CAM++ 做成对余弦筛选，阈值为 0.65，并剔除相同的声纹向量。目标声纹只用于离线筛选。
3. 对英文转录做大小写、标点和空格等归一化。剔除空转录、完全相同、逐词连续包含，以及词序列高度重叠的候选。近重复条件见 [prepare_celeb_role_pairs.py](src/aligndit/script/misc/prepare_celeb_role_pairs.py)。
4. 多个候选通过时，以 seed 666 和两个片段的完整 ID 做哈希后确定一个参考。一个参考可以服务多个目标。将结果冻结为 `pairs.jsonl`。其余目标及排除原因写入 `exclusions.jsonl`。训练前检查清单哈希、CAM++ 元数据和完整分割覆盖，训练 contract 记录配对摘要。

| 本机冻结清单 | 数量 |
|---|---:|
| 原训练目标 | 79,613 |
| 有合格参考，进入双台词训练 | 32,183（40.42%） |
| 同前缀没有第二条片段 | 16,548 |
| 剩余候选中没有时长合适的参考 | 12,476 |
| 剩余候选中没有通过声纹门槛的参考 | 18,313 |
| 最后因转录为空或重叠排除 | 93 |

共使用 22,566 个不同的参考片段，入选目标涉及 3,946 个源视频。参考声纹余弦中位数为 0.74840，参考时长中位数为 4.23 秒。入选目标时长中位数 3.74 秒，全训练集为 2.84 秒。筛选改变了训练集组成，进行网络消融时应固定配对清单。没有人工身份标签或原视频绝对时间戳，因此不能据此给出身份正确率或证明片段在源视频中完全无重叠。

冻结清单位于 `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_role_pairs_v1/`，`pairs.jsonl` 的 SHA256 为 `5a0039628dbd64b382bcaf51da7bba791fb6ce642e94c6f67dae7a3a4139ce49`。另用 WavLM-Large + ECAPA 说话人模型独立抽查 64 对。原配对余弦中位数 0.72363，最低 0.40599。把同一批参考跨视频打乱，中位数 0.02760，最高 0.24275。抽查的选样规则、逐对记录和汇总位于同级的 `CelebVDub_role_pairs_v1_audit/`，没有根据这次打分修改清单。

## 一个实际训练样本

同一源视频 `-2UAUQnPRGk` 的 `13_0_1` 是参考 A，`13_0_2` 是目标 B。CAM++ 余弦为 0.71492。

| 输入或监督 | 该样本 |
|---|---|
| A 参考台词 | `so now we're ready to do some light chemistry review next week` |
| B 目标台词 | `and after that, we'll talk about minerals` |
| 参考声音 | A 的 113 帧 latent 和 A 的 CAM++ |
| 目标视觉 | B 的 66 帧嘴部特征，前面补 113 帧空视频 |
| 训练监督 | B 的 66 帧真实 latent、B 的台词和 B 的 WavLM-Base+ 特征 |

[数据加载器](src/aligndit/model/celeb_role_dataset.py) 把 A 和 B 的真实 latent 拼成训练时的干净序列。生成掩码只覆盖 B。参考文本与 A 的声学条件汇聚为全局风格，主干文本交叉注意力只读取 B 的字符。CFM 和 REPA 按 B 的帧计算，CTC 按 B 的字符计算。Setting 2 推理用 A 的音频和台词、B 的台词和无声视频生成，只解码后 T 帧，推理输入不需要 B 的声音或声纹。Setting 1 遵循原协议，两种角色对应同一句台词。

## 运行

在本独立副本根目录执行。当前机器已生成并验证完整配对清单。在有相同 CelebV-Dub 缓存的环境中，可用以下命令重建或核验。清单发布前会检查所有训练声纹，再原子发布结果。

```bash
source env.sh
export PYTHONPATH="$PWD/src"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$python_bin" -m aligndit.script.misc.prepare_celeb_role_pairs
"$python_bin" -m aligndit.script.misc.check_celebvdub_role_training --num-processes 4
```

[双台词配置](src/aligndit/config/finetune_celebvdub_role_conditioned.yaml) 使用与基线相同的 S2c 70k EMA 初始化共享权重，新角色参数独立初始化。训练默认至 200k 更新，CFM 帧预算为每 GPU 2,400 帧（P+T）。4 卡每个 epoch 有 1,426 个同步更新。训练脚本预检后启动本副本 TensorBoard，事件和服务信息分别写入 `runs/` 和 `logs/`，模型保存在含 `celeb_role_conditioned` 的独立权重目录。

```bash
mkdir -p logs
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_role_conditioned.sh \
  > logs/train_celeb_roles.log 2>&1 < /dev/null &
```

检查实现可运行全套单元测试和真实父权重的 GPU 正反向测试。后者核对每对样本实际读取的 A/B 张量，在 CTC 权重为 0 和 0.03 两种情况下检查梯度，全程不更新参数。

```bash
"$python_bin" -m unittest discover -s tests
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u -m aligndit.script.misc.smoke_test_adaptive_band_repa_real_parent \
  --config-name finetune_celebvdub_role_conditioned
```

训练产生权重后，以下入口分别生成 CelebV-Dub 同句参考或独立异句参考条件下的音频。默认选择 200k EMA，`CKPT_STEP` 可指定已保存的更新数。沿用原项目测试清单，Setting 2 输入真实目标台词。

```bash
CKPT_STEP=100000 bash src/aligndit/run/eval/infer_celebvdub_role.sh s1
CKPT_STEP=100000 bash src/aligndit/run/eval/infer_celebvdub_role.sh s2
```

## 已完成的验证

全量清单重复生成后，配对、排除清单和摘要逐字节相同。54 项单元测试全部通过。实际 S2c 70k 父权重精确迁移了 303 个共享张量，另有 13 个角色分支张量独立初始化。两对真实训练样本的 P/T 分别为 113/66 和 132/100；用目标 GT 和缓存逐张量核对来源，在 CTC 权重为 0 和 0.03 时，参考风格投影都得到了有限的非零梯度。

采用临时组装的 CelebV-Dub 100k EMA 共享参数及未训练的角色层，在生产推理入口各生成一条真实音频。Setting 1 按 44+44 帧采样，解码后为 17,462 个 16 kHz 采样点。Setting 2 使用两句不同台词，按 184+76 帧采样，解码后为 30,080 个采样点。这次验证覆盖输入分路、实际解码和目标长度。正式训练后的 WER 与参考台词误读情况仍待测量。
