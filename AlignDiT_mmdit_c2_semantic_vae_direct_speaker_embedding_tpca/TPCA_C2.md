# C2 Semantic-VAE Direct + CAM++ + TPCA

本项目是 `../AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` 的独立代码副本。
从仓库提交 `611a61ff3cdc867563d51e7d6952b3bf41e53e94` 所在工作树复制 234 个已跟踪文件，
源目录无本地改动。原项目和 D1 TPCA 项目均不修改，不共享源码软链接。
数据缓存和固定预训练权重使用既有外部资源；未复制训练日志、checkpoint 或结果。

## 模型与移植边界

保留用户指定配置 `finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`
的全部基础设定：64 维/40 Hz 标准化 Semantic-VAE latent、25→40 Hz 已对齐视频特征、
12 层 MM + 6 层 audio-only、原 C2 音频→文本 `nn.MultiheadAttention`、原 QK norm/RoPE、
冻结 CAM++ 缓存以及第 12..17 层的零初始化说话人投影。没有更换成 Hunyuan 双 CA。

TPCA 复用 `../AlignDiT_mmdit_d1_hunyuan_tpca` 的 occurrence CTC 对齐和路径组合机制：

1. 视觉独立对齐头预测字符 CTC logits，通过前后向算法给出台词各次出现位置的后验 P。
   同一重复字符的不同出现位置是不同列；另有 blank/null 列。
2. 用同层实际 joint attention 的归一化、RoPE 后 Q_audio/K_video 重建视频条件分布 R，
   构成 detached `C = R @ P`。这是在有效视频 keys 内重新归一化的条件分布。
3. 在最后三层 MM（0-based 9、10、11）的前 4 个音频→文本 heads 中加入 `log(C)` 先验；
   其余 8 heads、参考音频 queries、其它层、说话人尾部路径保持原行为。
   C2 原有 MHA 的 Q/K/V、bias、输出投影和 attention dropout 均保留。
4. 对未加先验的 raw 文本注意力计算 `KL(stop_gradient(C_smooth) || A_raw)`；
   `C_smooth = 0.9*C + 0.1*Uniform(valid target positions + null)`。
   只在有效生成区域统计；不对已加先验的注意力自我拟合。
5. 仅 text+video 条件齐全时启用 TPCA；speaker 继续跟随参考音频一起 dropout。
   Packed CFG 只对 full 分支启用 TPCA；推理显式传入 prompt/target 字符边界并按一次 ODE 缓存 P。

新对齐头接收 C2 既有 40 Hz 视频，两个独立子位置分类器形成 80 Hz 的预测网格。
此网格不增加新的视觉观测，也不意味着原视频采样率变为 80 Hz。
与 D1 的 25 Hz 输入/50 Hz logits 是适配既有数据率后的区别。
训练列表 79,613 条在 80 Hz TPCA 网格均满足 CTC 最短路径长度；原 40 Hz 音频 CTC 有 105 条
不可行，沿用原有 zero_infinity 处理，不改变训练样本集合。
CTC 路径约束 P 的顺序，R 仍可学习；完整 TPCA 不保证硬单调。

新分支在原模型初始化完毕后构造，相同 seed 下原参数初始化保持一致。
TPCA 只新增 8 个参数张量和 1 个整数 step buffer。
S2c 严格迁移继续加载 303 个兼容 EMA 张量，忽略既有 10 个 S2c projector 张量；
新增 aligner、speaker 和既有 C2 新层保持各自初始化。

## 损失与训练

```
L = L_flow + lambda_audio_ctc(update) * L_audio_ctc
    + 0.03 * L_visual_ctc + 0.01 * tpca_scale(update) * L_raw_path_KL
```

音频 CTC 前 10k 为 0，从 10k 到 30k 线性升至 0.03，保持原配置。
TPCA 前 2k 只训练视觉 CTC，新注意力先验和路径 KL 的系数随后在 8k 更新内线性升至全量。
两种预热独立。缺失 text/video 的分支把 TPCA 损失权重置零。
在线模型和 EMA 的 `tpca_step` 都记录本轮已完成的 optimizer updates，恢复时严格检查；
它与继承的 EMA 内部 bookkeeping step（保留 S2c 70k 历史）不同。

新配置：
`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_tpca_ctc003_warmup.yaml`。

从相同固定 S2c 70k EMA 开始新的优化器和 update 计数；不从已训 C2 或 D1 TPCA 继续训练。
保留 LR=5e-5、20k LR warmup、200 epochs 的调度跨度、seed=666、bf16、每卡 3,600 latent
帧和 4×4090。达到 200k 停止；每 50k 保存编号权重，每 5k 保存 model_last.pt。

```bash
bash scripts/start_tpca_training.sh
```

入口使用 setsid 独立 session，显式 `PYTHONPATH=src`，不修改 Python 环境中的 editable 安装。
同时启动专用 TensorBoard，默认端口 6008。启动记录见日志及本文后续实测记录。

Checkpoint 根目录（可按 ROOT_PREFIX 前缀切换）：
`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_40hz_CelebVDub_char`。

TensorBoard run：
`runs/AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_semantic_vae_40hz_CelebVDub_char`。
仅 rank 0 记录总损失、flow、音频 CTC 权重/损失、视觉 CTC、路径 KL、TPCA 系数与可行比例、
说话人投影权重/梯度范数、全局梯度范数和 LR；沿用原模型的 rank-0 batch 日志口径。

## 推理

新入口：`src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_tpca_ctc003.sh`；
四指标流水线：`src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_tpca_ctc003.sh`。
默认 200k EMA、Setting 1 同片段参考协议。此时尚未评测新训练权重，不预先声称指标提升。

## 启动前验证（2026-09-08）

- 源项目 234 个已跟踪文件逐项 SHA256 不变。
- MHA 4 项单元检查、CFM/backbone 11 项集成检查通过；包括热启动非零输出的预热一致性、
  B>1 CFG、40 Hz padding、重复台词目标边界、teacher detach、dropout、speaker/aligner 梯度和 cache 清理；另验证相同 seed 的原参数一致性、在线/EMA step 及独立预热日程。
- 4 卡 bf16 DDP 连续 8 次混合 CFG 更新通过，跨 rank 说话人/TPCA 参数同步，梯度有限；
  覆盖从预热零系数进入全量 TPCA、音频 CTC 开关以及不同 rank 的不同模态 dropout。
- 完整 322,196,930 参数模型，实际 S2c 70k EMA + 4 条 900 帧训练样本：
  303 个迁移参数与源 EMA 逐元素相等；目标 713 个 state keys，410 个新 keys。
  CTC 关闭/开启、TPCA 预热/全量两次前后向均通过，speaker 与 aligner 梯度非零且有限。
  计入为 EMA、Adam 两组状态和 DDP buckets 预留的 4.801 GiB，分配峰值为 13.628/14.058 GiB，
  activation checkpointing 关闭；该测试只做前后向，没有更新或保存模型权重。
- 79,613 条训练 speaker 缓存全量审计通过，192 维 float32、有限值、单位范数，既有模型数据字段未变。

复核命令（在本项目内，统一 PYTHONPATH=src）：

```bash
/zjw524/ENTER/envs/aligndit/bin/python -u tests/test_c2_tpca_mha.py
/zjw524/ENTER/envs/aligndit/bin/python -u tests/test_c2_tpca_integration.py
CUDA_VISIBLE_DEVICES=0 /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_tpca_real_parent.py
CUDA_VISIBLE_DEVICES=0,1,2,3 /zjw524/ENTER/envs/aligndit/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=4 tests/smoke_c2_tpca_ddp.py
```

## 本次训练启动记录

- 启动时间：2026-09-08 01:37:24（Asia/Shanghai）。运行源码提交 `bcf1cb3d9c947a6839738a4daef7b84239da4fa1`。
- 运行名称：`AlignDiT_MMDiT_C2_SemanticVAE_Direct_Speaker_TPCA_CTC003_Warmup10k30k_semantic_vae_40hz_CelebVDub_char`。
- launcher PID/SID：`295862`；四个训练 worker PID：`295959,295960,295961,295962`。
  均无控制终端，worker cwd 指向本独立副本，实际运行 GPU 0..3。
- 日志：`logs/train_speaker_tpca_20260908_013724.log`；本轮 stop update=200000。
- 实际 parent_migration.json 确认 source=313、target=713、loaded=303、parent EMA update=70000；
  `speaker_tpca_training_contract.json` 确认新的 project/config/checkpoint 目录及 TPCA 开关。
- 启动检查至 update 19：20 个 TensorBoard scalar tags 均为有限值；最大总损失 2.22159；
  speaker projection 的范数由零增加，梯度非零；完整条件分支视觉 CTC 有效。
  音频 CTC 与 TPCA path 系数当前均为零，符合各自预热日程。
- 实际四卡早期显存约 14–15 GiB。此时仅确认训练正常推进，尚不能判断收敛或最终指标。
- TensorBoard PID/SID `295878`，端口 `6008`，logdir 即上文精确 run 路径。
  只有一个 rank-0 event 文件，持续写入；主页和 Scalars HTTP API 返回 200。
  服务器本机地址 `http://127.0.0.1:6008/`。客户端实际转发 URL 未暴露给当前工具，
  已请求用户提供用于核验；不将本机地址或猜测 URL 称为外部可访问的转发链接。
  用户可在 Devin 底部“端口”页签找到 6008，点击“转发地址”列中的实际链接。
- 验证证据：`logs/validation_cpu_tpca.log`、`logs/validation_ddp_tpca.log`、
  `logs/validation_real_parent_tpca.json`、`logs/tpca_speaker_cache_audit.log`、
  `logs/tensorboard_startup_audit.json`。运行产物均不提交 Git。
