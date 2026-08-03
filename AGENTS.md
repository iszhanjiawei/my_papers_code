# AGENTS.md

本文件适用于 `my_papers_code/` 及其全部子目录。这里保存的是 AlignDiT 论文实验的多个独立快照，目标是保持实验可复现，而不是把它们逐步合并成一个统一代码库。

## 仓库结构

- `AlignDiT_mmdit_base/`：MM-DiT 基线实验。
- `AlignDiT_mmdit_base_qknorm_ca/`：在基线上启用 RMS QK-Norm，并为文本 cross-attention 增加由时间步调制的 AdaLN/gate。
- `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/`：当前论文改进与 C0-C3 消融实验的主快照；分离视频交互层数、文本注入层数，并支持仅向待生成音频帧写入文本 cross-attention 残差。
- `AlignDiT_mmdit_c2_semantic_vae/`：从上述主快照的 C2 路线独立复制出的 Semantic-VAE 实验目录；目标是把 80 维、100 Hz mel 改为 64 维、40 Hz Semantic-VAE latent。该目录必须独立演进，不得把中间改动同步回已完成的 C0-C3/D0-D2 实验。
- `AlignDiT_mmdit_wav_vae_base_qknorm_ca/`：为 wav/Semantic-VAE 方向保留的实验快照；当前受 Git 跟踪的源码与 `AlignDiT_mmdit_base_qknorm_ca/` 基本一致，不要仅凭目录名假定 wav VAE 已完成集成。
- `hifigan_16k_LRS3/`：共享的 HiFi-GAN 配置与权重。权重属于二进制资产，不要修改、格式化或重新生成。

每个 `AlignDiT_*` 目录都是一个独立的 Python 项目，包名都为 `aligndit`。仓库根目录本身不是 Python 包。

## 修改前先确定实验目标

1. 先明确任务属于哪个 `AlignDiT_*` 子目录，只在该目录内工作。
2. 不要把一个实验目录的改动自动复制到其他实验目录。只有在任务明确要求同步，或已确认是所有快照共有的缺陷时才同步，并逐个检查差异。
3. 不要为了“去重”而创建跨实验目录的软链接、共享源码目录或大规模公共抽象；快照隔离是本仓库的一部分。
4. 如果任务描述只说“AlignDiT”而无法从上下文判断目标，优先根据涉及的配置名、模型名和路径推断；仍会影响实验语义时再询问用户。
5. 修改前后使用 `git diff -- <目标目录>` 检查范围，避免把生成文件或其他实验的变化混入。

## 主要代码位置

在每个实验目录中：

- `src/aligndit/model/backbone/`：DiT/MM-DiT 主干网络。
- `src/aligndit/model/cfm_*.py`：条件流匹配与采样逻辑。
- `src/aligndit/model/trainer_*.py`：训练循环、检查点及日志。
- `src/aligndit/model/dataset.py`：数据集与批处理。
- `src/aligndit/config/`：Hydra 训练配置。
- `src/aligndit/script/`：Python 训练、推理、评测和数据准备入口。
- `src/aligndit/run/`：封装上述入口的 Shell 脚本。
- `src/f5_tts/`、`src/cosyvoice/`、`src/gslm/`：上游/第三方代码。除非任务直接涉及它们，否则不要顺手重构或全量格式化。
- `paper/`：论文笔记和参考资料，不是运行时源码。

## 环境与路径

项目面向 Python 3.10。本机已有标准环境：

```bash
/zjw524/ENTER/envs/aligndit/bin/python
```

应优先直接使用该环境的 Python，不要为普通开发任务重复创建环境。使用 `ROOT_PREFIX=/home` 的服务器上，对应路径为 `/home/zjw524/ENTER/envs/aligndit/bin/python`；脚本中统一写成 `${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python`。

只有需要重建环境时，才进入目标实验目录执行：

```bash
conda create -y -n aligndit python=3.10
conda activate aligndit
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e '.[eval]'  # 仅评测需要
```

多个实验项目提供相同的 `aligndit` 包；同一环境中执行 `pip install -e .` 会让后安装的快照覆盖先前的 editable 指向。运行命令时应位于目标目录，并优先显式设置 `PYTHONPATH=src`，防止导入错误快照。例如：

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u path/to/script.py
```

短时交互命令可以在已激活的 Conda 环境中运行。后台或长时间任务必须直接调用环境中的 Python，不使用 `conda run`，也不增加不必要的 `bash -c`/Shell 嵌套。

路径由各项目根目录的 `env.sh` 统一切换：

```bash
source env.sh
# 当前服务器通常 ROOT_PREFIX=""
# 另一种目录布局可使用 ROOT_PREFIX=/home
```

新增本机绝对路径时沿用现有约定：

- Shell：`${ROOT_PREFIX}/zjw524/...`
- Hydra/YAML：`${oc.env:ROOT_PREFIX,''}/zjw524/...`
- Python：`os.environ.get("ROOT_PREFIX", "")`

不要提交只适用于临时机器、个人环境或某个 GPU 节点的新硬编码路径。注意部分既有脚本仍引用仓库外的 CelebVDub、Semantic-VAE、预训练权重和 Conda 环境；不要假定这些资源在所有机器上存在。

## 模型与配置约束

- 当前 `my_papers_code` 中的论文实验以 **CelebVDub** 为目标数据集。上游 `README.md` 仍以 LRS3 为主，不能据此把当前 CelebVDub 训练、推理或评测路径改回 LRS3。`finetune.yaml`/`finetune_celebvdub*.yaml` 等配置同时存在时，以任务指定的实验配置为准；不要混用数据列表、词表、视频特征或 checkpoint。
- LibriSpeech 仍可用于纯音频预训练，LRS3 相关入口仍作为上游兼容代码保留；“目标数据集是 CelebVDub”不意味着可以删除这些入口。
- 默认音频为 16 kHz，mel 为 80 维、100 Hz；视频特征通常为 1024 维、25 Hz，`audio_video_ratio=4`。修改时保持时间轴、mask、RoPE 位置和长度换算一致。
- `n_mm_layers` 表示前若干层双流 MM-DiT，后续层为 audio-only。改变深度或分层方式时，同时检查 block 构造、CTC 中间层索引及 smoke test。
- `qk_norm` 必须从 Hydra 配置完整传递到对应 attention；`base` 配置默认不启用，而 `qknorm_ca` 配置使用 `rms_norm`。
- 新增模块若要从纯音频预训练检查点微调，应保持已有 audio path 的键名和张量形状兼容。新视频流、cross-attention 或 gate 参数应有明确且稳定的初始化策略。
- 新增或重命名 Hydra 字段时，同步检查配置、模型构造、训练器和推理入口，避免只改 YAML。
- checkpoint 的兼容加载有意允许部分新参数缺失；不要随意改成宽泛的 `strict=False` 来掩盖非预期的键名或形状错误。

## 当前主实验快照与 C0-C3 消融

除非用户明确指定其他快照，当前论文改进任务默认只修改：

```text
AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/
```

当任务明确涉及 Semantic-VAE、40 Hz latent 或对应的 LibriSpeech 音频预训练时，只修改：

```text
AlignDiT_mmdit_c2_semantic_vae/
```

不要使用未完成且可能含本地运行产物的 `AlignDiT_mmdit_wav_vae_base_qknorm_ca/`，也不要覆盖原 C2 mel 快照。

四组实验均使用深度 18、前 12 层 MM-DiT、CelebVDub、字符 tokenizer、RMS QK-Norm、BF16 和相同的动态 frame batch。只允许按下表改变文本注入层数和参考音频隔离开关：

| 实验 | `n_mm_layers` | `n_text_layers` | `prompt_isolated_ca` | 语义 |
|---|---:|---:|---|---|
| C0 | 12 | 18 | `False` | 18 层全局文本 CA；参考音频帧也接收文本残差 |
| C1 | 12 | 18 | `True` | 18 层文本 CA；文本残差只写入待生成音频帧 |
| C2 | 12 | 12 | `False` | 前 12 层全局文本 CA；后 6 层为无文本 audio-only DiT |
| C3 | 12 | 12 | `True` | 前 12 层隔离文本 CA；后 6 层为无文本 audio-only DiT |

配置和单机 4×RTX 4090 启动入口固定映射如下：

| 实验 | Hydra 配置 | 启动脚本 |
|---|---|---|
| C0 | `finetune_celebvdub_mm_c0.yaml` | `finetune_celebvdub_mm_c0_4x4090.sh` |
| C1 | `finetune_celebvdub_mm_c1.yaml` | `finetune_celebvdub_mm_c1_4x4090.sh` |
| C2 | `finetune_celebvdub_mm_c2.yaml` | `finetune_celebvdub_mm_c2_4x4090.sh` |
| C3 | `finetune_celebvdub_mm.yaml` | `finetune_celebvdub_mm_c3_4x4090.sh` |

注意 C3 使用主配置 `finetune_celebvdub_mm.yaml`，不存在必须另建的 `finetune_celebvdub_mm_c3.yaml`。四个 `4x4090` 启动脚本均绑定 GPU 0-3，使用不同端口，并保持 `OMP_NUM_THREADS=1`。曾测试 C3 使用 `OMP_NUM_THREADS=4`，200 updates 仅从 2:18 缩短至 2:15，差异接近运行波动，已恢复为 1。

### D0：6+12 层比例与单层 CTC

在 C0-C3 完成后，新增 D0 用于验证参考论文常用的约 1:2 多模态/单模态层比例。D0 不覆盖或改写任何 C0-C3 配置：

| 实验 | `n_mm_layers` | `n_text_layers` | `prompt_isolated_ca` | `layer_indices_ctc` | 语义 |
|---|---:|---:|---|---|---|
| D0 | 6 | 6 | `False` | `[11]` | 前 6 层为全局文本 CA 的 MM-DiT；后 12 层为无视频、无文本 CA 的原生音频 DiT；唯一 CTC 头接在第 12 个 block 后 |

注意 `layer_indices_ctc` 使用零基索引，因此 `[11]` 表示第 12 个 block 执行完成后。D0 的入口固定为：

| Hydra 配置 | 单机 4×RTX 4090 启动脚本 |
|---|---|
| `finetune_celebvdub_mm_d0_6mm12audio_ctc12.yaml` | `finetune_celebvdub_mm_d0_6mm12audio_ctc12_4x4090.sh` |

D0 显式设置顶层 `seed: 666`。训练入口在模型构造前用该 seed 固定新增参数初始化，再在 Accelerate 初始化后使用 `seed + process_index` 生成各 DDP rank 的训练随机流；动态 batch sampler 继续使用原始实验 seed。没有顶层 `seed` 的历史 C0-C3 配置保持原有行为，避免改变已完成实验的语义。

### D1-D2：双层 CTC 与分阶段文本注入

D1-D2 用于拆分 D0 的高 WER 是来自 CTC 深监督不足，还是来自第 6 层后过早停止文本注入。两组都保持 D0 的深度、前 6 层 MM-DiT、全局文本残差、随机种子和 `ctc_lambda: 0.1`；多个 CTC loss 在应用 `ctc_lambda` 前取平均，因此不会扩大总辅助损失权重。

| 实验 | Blocks 0-5 | Blocks 6-11 | Blocks 12-17 | `layer_indices_ctc` | 主要对照 |
|---|---|---|---|---|---|
| D1 | MM-DiT + 文本 CA | 原生音频 DiT | 原生音频 DiT | `[5, 11]` | 相对 D0 增加 MM 阶段出口的 CTC 深监督 |
| D2 | MM-DiT + 文本 CA | `AudioTextDiTBlock` + 文本 CA | 原生音频 DiT | `[5, 11]` | 相对 D1 只增加中间 6 层文本 CA |

`layer_indices_ctc` 是全局零基 block 索引，CTC 头在对应 block 执行完成后读取隐藏状态。因此 `[5, 11]` 严格表示第 6、12 个 block 后，分别监督 D2 的 MM 阶段出口和文本阶段出口。已有 C2 使用 `[6, 12]`，即第 7、13 个 block 后；比较 D2 与 C2 时不能把全部差异仅归因于 MM-DiT 层数。

| 实验 | Hydra 配置 | 单机 4×RTX 4090 启动脚本 |
|---|---|---|
| D1 | `finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12.yaml` | `finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_4x4090.sh` |
| D2 | `finetune_celebvdub_mm_d2_6mm6text6audio_dual_ctc6_12.yaml` | `finetune_celebvdub_mm_d2_6mm6text6audio_dual_ctc6_12_4x4090.sh` |

D1 与 D2 必须各自从相同的 LibriSpeech 预训练权重开始训练，不能从 D0 或彼此的中途 checkpoint 续训。D1/D2 启动脚本分别使用端口 `29565`/`29566`；若在同一服务器并行启动，仍需为每个实验分配互不重叠的 GPU。

训练日志位于该快照的 `logs/`，checkpoint 位于 `/zjw524/projects/data/ckpts/` 下以各配置 `model.name` 命名的目录。日志和 checkpoint 由多台服务器通过共享文件系统写入；检查远端训练状态时，应同时确认：

1. 日志大小和 mtime 持续变化；
2. 最新 `Epoch ... update=...` 持续推进；
3. 日志末尾无 traceback、OOM、NaN、NCCL/ChildFailed 错误；
4. `model_last.pt` 和 `model_<update>.pt` 按配置周期正常生成。

不要把某一时刻的 epoch、update、PID、ETA 写进本文件；这些信息会很快过期，应在交接时从实时日志重新计算。

## 已知暂缓项

- `AlignDiT_mmdit_wav_vae_base_qknorm_ca/` 尚未真正接入 wav/Semantic-VAE；除非用户重新指定，不要在当前 C0-C3 工作中补做。
- batch 大于 1 的旧推理视频 mask 存在已知方向问题，但当前正式推理入口固定 `infer_batch_size=1`；用户已明确要求暂缓，不要顺手修改。

## 代码风格

- 遵循各项目的 `ruff.toml`：Python 3.10，行宽 120。
- 保持现有导入方式和类型/张量命名风格；注释应解释形状、时间对齐或实验动机，避免复述代码。
- 对第三方目录只格式化实际修改的文件，不运行会重写整个 `src/` 的批量格式化。
- Shell 脚本应可从对应项目根目录执行；修改后至少运行 `bash -n`。
- 不要手工编辑 `.pyc`、日志、重建音频、Hydra `outputs/`、`wandb/`、checkpoint 或其他生成物。

## 验证

根据改动范围选择最小但充分的验证。先在目标实验目录执行：

```bash
# Python 静态检查；将路径限制在改过的文件
ruff check path/to/changed.py
ruff format --check path/to/changed.py
python -m py_compile path/to/changed.py

# Shell 语法
bash -n path/to/changed.sh
```

修改 MM-DiT、CFM、mask、采样或 checkpoint 兼容逻辑时，运行该快照的 CPU smoke test：

```bash
PYTHONPATH=src python -u src/aligndit/script/misc/smoke_test_mmdit.py
```

它应覆盖模型构造、前反向传播、模态丢弃、采样和预训练键兼容；本地没有预训练权重时，对应兼容检查会跳过，应在结果中如实说明。仓库没有统一的 pytest 测试套件，不要声称“全部测试通过”而只做了语法检查。

`scripts/smoke_vae.py` 不是普通单元测试：它依赖仓库外的 Semantic-VAE、CelebVDub 和模型权重，会自动选择 CUDA，并写入 `scripts/smoke_vae_output/`。仅在任务明确涉及 VAE 且外部资源齐全时运行。

## 训练、评测与资源安全

- 不要为了验证小改动而启动训练、推理、评测、数据预处理、下载或 Slurm 作业。
- `sbatch_train_*.sh` 和 `src/aligndit/run/train/*slurm.sh` 会申请多张 GPU，并使用特定节点、网卡、端口和外部数据路径；只有用户明确要求启动作业时才执行。
- 若用户要求运行重任务，先确认目标快照、配置、checkpoint、输出目录和当前 GPU/Slurm 状态，防止覆盖或续训错误实验。
- 不要提交数据集、模型 checkpoint、日志、生成音频或新的大文件。若任务必须更新二进制资产，先向用户确认。

### 长时间后台任务

用户明确要求在当前机器启动非 Slurm 长任务时，必须用 `setsid` 创建独立 session，不使用 `nohup`。`nohup` 可能只保护外层 Shell，而 `accelerate`/`torchrun` worker 仍留在原进程组，SSH 断开后可能收到 SIGHUP。

同时避免日志缓冲：

- 单个 Python 程序直接使用环境解释器和 `-u`。
- 包含 `accelerate`/`torchrun` 的 Shell 入口设置 `PYTHONUNBUFFERED=1`。
- 不使用 `conda run`，不套多层 `bash -c`。

```bash
# 单个 Python 长任务
setsid env PYTHONPATH=src \
  /zjw524/ENTER/envs/aligndit/bin/python -u path/to/script.py \
  > path/to/task.log 2>&1 &

# 已封装 accelerate/torchrun 的训练入口
setsid env PYTHONUNBUFFERED=1 \
  bash path/to/train_script.sh \
  > path/to/train.log 2>&1 &
```

启动后记录返回的 PID，并确认任务已脱离控制终端（`SID` 独立且 `TTY` 为 `?`）：

```bash
ps -o pid,ppid,sid,tty,stat,cmd -p <PID>
```

不要仅凭外层 Shell 存活就判断训练正常；还要检查 worker 进程、日志持续更新及 GPU/Slurm 状态。复杂且需要复用的启动命令应写入目标实验目录的脚本或文档，不依赖临时 Shell 历史。

## 完成任务时

说明：

- 修改了哪个实验快照以及为什么没有（或为什么需要）同步其他快照；
- 执行了哪些检查及其结果；
- 哪些检查因 GPU、数据集、外部仓库或 checkpoint 不可用而未运行；
- 若改变了模型结构或配置，指出 checkpoint 兼容性和预期实验语义。

## Git 跟踪要求

- 用户要求当前论文实验的每个源码、配置、启动脚本或 `AGENTS.md` 修改步骤都必须使用 Git 跟踪：检查改动范围、执行对应验证、创建独立 commit，并 push 到远端当前分支。
- commit message 应明确说明原问题、实验语义或 bug，以及采用的解决方式；不要用无法区分实验步骤的笼统说明。
- push 后核对本地 `HEAD` 与远端分支一致，并报告 commit hash。
- 训练日志、Hydra outputs、TensorBoard/W&B 文件、数据集、生成样本和 checkpoint 不得加入 commit；它们属于运行产物，即使位于工作区或共享目录也只做状态检查。
- 工作树存在与当前任务无关的用户修改时，不要覆盖、回滚或混入提交；只暂存本次目标文件。
