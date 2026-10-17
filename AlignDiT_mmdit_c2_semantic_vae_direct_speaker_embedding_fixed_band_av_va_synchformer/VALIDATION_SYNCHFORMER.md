# 验证记录（2026-10-17）

验证使用现有 `/zjw524/ENTER/envs/aligndit/bin/python`、PyTorch 2.4.1+cu121。未安装或升级环境依赖，未启动正式训练，未更新模型权重。

## 源实验隔离

复制前后对原 `fixed_band_av_va` 实验的 253 个源码/配置/文档/附属文件计算 SHA256，全部一致。原 checkpoint、日志和运行目录未复制。新目录没有跨实验源码软链接。

## 自动检查

| 检查 | 结果 |
|---|---|
| `tests/test_synchformer_model.py` | 7 项通过：CFG、时间轴、prompt/padding、禁用兼容、零初始化梯度、activation checkpoint、CFM 训练/采样 |
| `tests/test_synchformer_contract.py` | 6 项通过：配置继承、契约匹配/拒绝、原目录误写保护 |
| `scripts/test_synchformer_cache.py` | 10 项通过：缓存身份/覆盖、短片段、解码关闭、调试提取保护、并行 audit |
| `scripts/test_synchformer_dataset.py` | 5 项通过：数据读取/组 batch/键与缺失输入检查 |
| 原 `smoke_test_fixed_temporal_band_bidirectional.py` | 16 项通过：AV/VA logits、转置 prior、mask、CFG、契约回归 |
| Python 编译、相关未定义名称检查、修改的 Shell 语法 | 通过 |
| `scripts/preflight_synchformer.py` | `ready=true`，79,613 条训练记录及父权重身份通过 |

共 44 项自动回归测试通过。新同步分支之外，`joint_attn`、speaker 条件、输入 embedding 的 AST 与源基线相同，固定带宽模块文件字节相同。

## 权重与同步缓存

实际完整读取 Synchformer 官方权重并计算 SHA256，结果匹配固定 hash。严格加载全部视觉编码器参数、参数冻结及 eval 状态检查通过；CPU 输入 `[1,16,3,224,224]`，输出 `[1,8,768]`，数值有限。

核验 79,826 个预期同步缓存文件全部存在，包括 train=79,613、test=213。读取首条、最短 0.44 秒、最长 29.96 秒及测试集代表样本，张量、原视频指纹和时长匹配。完整张量内容 audit 沿用 2026-09-15 的既有通过报告；本次未重新读取所有缓存张量。

提取模块修复 PyAV 11 提前结束时 `AUTO` 线程关闭死锁，改为 `SLICE`。真实 592 帧视频的 RGB 解码字节与原实现完全相同。`--limit` 调试提取现在拒绝覆盖已有全量缓存元数据/覆盖报告。

## 真实 GPU 前向/反向

使用实际 S2c 70k EMA 父模型和两条真实 CelebVDub 样本，音频 latent 长度 126/98，同步 token 长度 64/48。严格迁移保留 303 个原张量，父模型总 313 个 key；新目标总 710 个 key，仅允许既定新增张量。坏 shape 和未知 source key 均被拒绝。

| CTC λ | 总损失 | Flow 损失 | CTC 损失 | 同步输出投影梯度范数 | CAM++ 投影梯度范数 |
|---|---:|---:|---:|---:|---:|
| 0 | 1.49995 | 1.49995 | 未计算 | 42.30314 | 0.02187 |
| 0.03 | 1.77976 | 1.49995 | 9.32684 | 42.49297 | 0.02210 |

混合精度 bf16；所有损失和梯度有限。峰值 allocated 显存约 2.59 GiB，仅代表该两条短样本测试，不代表正式 3,600-frame/卡分布式训练的显存需求。实际 optimizer updates=0，没有保存新训练 checkpoint。

本机运行记录（不纳入 Git）：

- `logs/synchformer_preflight.json`
- `logs/synchformer_resource_validation.json`
- `logs/synchformer_real_parent_smoke.log`
- `logs/fixed_band_av_va_regression.log`
- `logs/synchformer_contract_tests.log`

正式多卡训练、完整音频生成及四指标质量评测尚未执行；需要训练后的新权重才能判断效果。
