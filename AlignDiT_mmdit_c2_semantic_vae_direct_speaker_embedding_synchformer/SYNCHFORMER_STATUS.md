# Synchformer 实施与训练状态

更新日期：2026-09-15。验证时间：`2026-09-15T04:51:09.101099+00:00`。

## 结果

功能、预训练权重和全部数据缓存已准备完成，四卡训练正在运行。交接检查时已完成 **223 次 optimizer update**；当前记录的全部标量均为有限值，Synchformer 输入和输出投影均有非零梯度，TensorBoard event 持续增长。

- 新实验目录：`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_synchformer`。
- 原项目 234 个源码文件的 SHA256 逐项复查保持不变。
- 本次训练使用的源码提交：`052ec9e607cfbe7a579818438f77c926dc0f7f3e`。此后状态文档的提交不改变正在运行的训练代码。
- 这是训练启动和工程验证记录；完整训练及生成质量评测仍待后续完成。

## 权重和数据

官方 Synchformer checkpoint 已下载并核对完整 SHA256：

- 文件：`/zjw524/projects/data/pretrained_models/HunyuanVideo-Foley/synchformer_state_dict.pth`
- 大小：950,058,171 字节。
- SHA256：`8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c`。
- 编码器在特征提取时冻结；训练只读取缓存，不加载该编码器。

同步特征根目录：`/zjw524/projects/data/CelebVDub/synchformer_25fps_16f_stride8`。

| 检查项 | 结果 |
|---|---:|
| 全量有效缓存 | 79,826 |
| train / test | 79,613 / 213 |
| 缺失 / 无效 | 0 / 0 |
| 特征 token 总数 | 7,334,616 |
| 完整内容审查耗时 | 646.86 秒 |

全量审查逐项读取特征、检查有限值、形状、权重身份、源视频指纹和音视频时长。权威报告为缓存根目录中的 `coverage_report.json`，SHA256 为 `77aad81ff6506788d083d97692412e7edbedccc39f4b4071709586194bd1441e`；简明副本为 `logs/synchformer_full_coverage_summary.json`。

已有音频 latent、40 Hz AV-HuBERT 和 speaker 缓存也完成审查：79,613 条训练样本、238,839 个 NPY 的路径、头部维度、FP32 类型和文件字节数全部通过。latent/video 权威完成索引的 SHA 一致；speaker 复用身份匹配且文件修改时间未变化的既有全数组审查。代表样本实际读取、归一化和有限值检查通过。报告：`logs/preexisting_training_data_audit.json`。

训练前预检 `logs/synchformer_preflight.json` 的 `ready=true`，未放宽完整覆盖要求。

## 正式训练

- Run：`AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`
- 启动器 PID：`233991`；worker PID：`234794, 234795, 234796, 234797`。
- 后台进程均无控制终端，使用独立 session。
- 4 × RTX 4090，bf16，每 GPU 3600 帧，seed 666。
- 使用既定 S2c 70k EMA 父权重初始化；训练 update 从 0 起计，独立 optimizer。
- LR 上限 `5e-5`、20k LR warmup；CTC 前 10k update 为 0，至 30k 线性升到 0.03。
- 保留原 `checkpoint_activations=false`；计划上限 200k update。
- Checkpoint 目录：`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_40hz_CelebVDub_char`。
- 训练日志：`logs/train_synchformer_4x4090.log`。

第 223 次更新的记录：总 loss `1.37093723`，Synchformer 输入投影梯度范数 `0.00597330`，输出投影梯度范数 `5.31657982`。此时 CTC 权重为 0，符合原 warmup 计划。损失快照用于确认运行正常，不用于评价生成质量。

## TensorBoard

- PID：`234536`；端口：`6007`。
- 完整 logdir：`/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_synchformer/runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`。
- 本机网络地址：[http://10.248.230.208:6007/](http://10.248.230.208:6007/)。localhost、机器网络地址和 Scalars 数据接口均实测 HTTP 200。
- 已存在实际 event 文件；观测到文件从 64,164 字节增长到 173,492 字节。
- 已验证 `loss`、`diff_loss`、CTC 调度及加权损失、speaker 梯度和 Synchformer 输入/输出投影梯度等 13 个标量 tag。
- 当前工具环境没有提供平台的外部转发 URL，因此尚未验证浏览器经平台转发的访问。若使用 Devin，请打开底部“端口”页签，找到 6007，点击“转发地址”列中的实际链接。

运行验证报告：`logs/synchformer_training_verification.json`。该报告区分了已经验证的机器网络地址与尚不可获取的平台转发地址。

## 工程验证

- 变长有效长度插值、padding 隔离、三路 CFG、activation checkpointing、梯度和 ODE 推理测试通过。
- 新旧模型的公共参数在相同初始化种子下逐项完全一致。
- 6 项 dataset / S1 推理接线测试，9 项视频提取 / 缓存 / 并行审查测试通过。
- 真实 S2c 70k 迁移：313 个源键中精确迁移 303 个，按原规则忽略 10 个；710 个目标键中 407 个保持初始化，其中 6 个为新增同步分支参数。
- 两条真实样本在 CTC 0 / 0.03 下前后向通过，GPU 峰值约 2.59 GiB，没有参数更新。
- 大 batch 实测长度 `[1199, 1190, 1188]`，有效 3577、padding 后 3597 帧；额外预留 4.88295 GiB 估计 AdamW、EMA 和 DDP 存储，CTC 0 / 0.03 均通过，allocated 峰值约 15.57 GiB。该测试未执行 optimizer update；之后正式四卡训练的实际更新也已通过验证。

提取中遇到的 PyAV 11 解码资源保留已修复：显式关闭 decoder/codec，每 16 个新视频 GC/trim。修复后 300 个真实视频最终 RSS 1820.35 MiB，观测峰值 2591.11 MiB，8 份已有特征与修复后结果逐位一致。恢复全量提取后新生成 34,447 条、复用 45,379 条，全部成功，无新增 OOM。worker 监督进程同时提供异常退出、单进程 RSS 6144 MiB 和 cgroup 90% 内存保护。

后续完整审查默认支持 8 个 CPU 进程；串行与并行的报告内容和错误顺序已通过真实 fixture 等价测试。本次全量审查由原串行进程完成，使用相同逐项检查。

用法、架构和重新运行命令见 [SYNCHFORMER.md](SYNCHFORMER.md)。日志、数据、TensorBoard 和 checkpoint 均不加入 Git。
