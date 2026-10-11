# AGENTS.md

本文件适用于 `my_papers_code/` 及其全部子目录。这里保存的是 AlignDiT 论文实验的多个独立快照，目标是保持实验可复现，而不是把它们逐步合并成一个统一代码库。

## 模型推理与四指标评测

1. 确认目标实验、配置、checkpoint/update 和 EMA；在目标项目中使用 `PYTHONPATH=src`，优先复用 `src/aligndit/run/eval/` 的对应入口。VAE/speaker 分支同时核对 decoder、latent 标准化和说话人缓存。
2. 沿用该实验的测试协议；历史 CelebV-Dub Setting 1 通常为 213 条、seed 0、EMA、Euler/EPSS、32 NFE、sway=-1、CFG text/video=5/2、真实时长，用户指定参数优先。各权重/CFG 使用独立输出目录和日志。
3. 流程：生成 WAV → SPKSIM（WavLM）→ WER（既有 ASR）→ EMOSIM（emotion2vec）→ 用生成音频和对应嘴部视频提取 AV-HuBERT 特征 → AVSync。指标入口通常为 `src/aligndit/script/eval/eval_celebvdub_test.py`，任务参数 `-e sim/wer/emosim/avsync`；特征提取用 `src/aligndit/script/misc/extract_avhubert.py`。
4. 检查实际可用显存，允许按用户授权与其他进程共享 GPU；本机后台评测用 `setsid`，单卡指标计算用 `-n 1`。监控日志、子进程和 GPU 至全部阶段结束；失败时保留已验证产物，补跑失败阶段。
5. 按测试列表逐项核验 WAV、特征和四份 JSONL 的覆盖与有效值。JSONL 末尾汇总行不算样本；跨视频同名 clip 应用完整相对路径区分。SPKSIM/EMOSIM/AVSync 取样本均值，WER 用各句词级编辑距离之和除以参考词总数，不能平均逐句 WER；复算结果与日志核对，展示五位小数。
6. 交付四指标表、权重对比和产物路径；按用户要求写入 `实验结果/实验结果总汇.md`，记录实际配置、样本数、参考音频协议及 checkpoint。不同测试协议/独立训练的差值只作描述性对照；文档按本仓库 Git 规则提交，运行产物不提交。

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


### 每次训练必须提供 TensorBoard 损失曲线

每次启动任何训练（包括本机、后台和 Slurm 训练）时，必须同时完成以下事项；不得只启动训练进程而不提供可访问的损失曲线：

1. 确认训练器使用 TensorBoard event 文件持续记录损失，至少包含总损失和该实验实际使用的各分项损失（例如 flow/CFM、CTC 或 projection loss）。只有文本日志不算完成此要求。仅主进程写 event，避免 DDP 多 rank 重复记录；续训必须沿用或明确区分对应 run 目录。
2. 记录本次训练的确切 TensorBoard `logdir`，并在 Devin 可访问的机器上实际启动 TensorBoard，不能只给出一条尚未运行的命令。长时服务同样使用 `setsid`，例如：

   ```bash
   setsid /zjw524/ENTER/envs/aligndit/bin/python -m tensorboard \
     --logdir <event-logdir> --host 0.0.0.0 --port <free-port> \
     > <tensorboard-log> 2>&1 &
   ```

   若训练在其他节点而 event 文件位于共享文件系统，则在 Devin 所在机器上对该共享 `logdir` 启动 TensorBoard。
3. 启动后检查 TensorBoard 进程、监听端口和 HTTP 响应，并确认 event 文件会随训练更新、TensorBoard Scalars 页面能看到对应的 loss tag。端口被占用时改用一个经检查的空闲端口，不要盲目复用旧端口。
4. 每次启动训练后都必须在交接中同时给出：训练 run 名称、TensorBoard `logdir`、TensorBoard PID、端口和可直接点击的转发地址。并明确告诉用户：打开 Devin 底部面板的“端口”页签，找到该端口的那一行，点击“转发地址”列中的具体链接。必须报告当次运行实际显示的链接，不得把示例端口或过期链接当作当前地址。
5. TensorBoard 未成功记录 loss、未运行或无法通过转发地址访问时，不得宣称训练启动已完成；应继续排查，或如实报告阻塞原因。




## Git 跟踪要求

- 用户要求当前论文实验的每个源码、配置、启动脚本或 `AGENTS.md` 修改步骤都必须使用 Git 跟踪：检查改动范围、执行对应验证、创建独立 commit，并 push 到远端当前分支。
- commit message 应明确说明原问题、实验语义或 bug，以及采用的解决方式；不要用无法区分实验步骤的笼统说明。
- push 后核对本地 `HEAD` 与远端分支一致，并报告 commit hash。
- 训练日志、Hydra outputs、TensorBoard/W&B 文件、数据集、生成样本和 checkpoint 不得加入 commit；它们属于运行产物，即使位于工作区或共享目录也只做状态检查。
- 工作树存在与当前任务无关的用户修改时，不要覆盖、回滚或混入提交；只暂存本次目标文件。


## 修改前先确定实验目标

1. 先明确任务属于哪个 `AlignDiT_*` 子目录，只在该目录内工作。
2. 不要把一个实验目录的改动自动复制到其他实验目录。只有在任务明确要求同步，或已确认是所有快照共有的缺陷时才同步，并逐个检查差异。
3. 不要为了“去重”而创建跨实验目录的软链接、共享源码目录或大规模公共抽象；快照隔离是本仓库的一部分。
4. 如果任务描述只说“AlignDiT”而无法从上下文判断目标，优先根据涉及的配置名、模型名和路径推断；仍会影响实验语义时再询问用户。
5. 修改前后使用 `git diff -- <目标目录>` 检查范围，避免把生成文件或其他实验的变化混入。


## LSE-D / LSE-C 独立音画同步评测

用户已指定这两个新增指标使用独立工具，目录为 `evaluation/lse/`，读取各实验已经生成的 WAV 和对应原视频；不导入、修改或合并任何实验快照。原有 SPKSIM / WER / EMOSIM / AVSync 保留，增加 LSE-D / LSE-C 后可报告六项指标。

### 输入与指标协议

- `AlignDiT_mmdit_c2_semantic_vae_direct` 的模型视觉条件确实来自嘴部：`crop_mouth_celebvdub.py` 先裁成 96×96，AV-HuBERT 按 checkpoint 配置中心裁成 88×88并提取特征；训练/推理再读取相应 40 Hz 特征缓存。**这与 SyncNet 的评测输入不同**。LSE 默认读取 `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/video/test/...mp4` 中的原始脸部视频，使用官方 S3FD 检测、场景切分、轨迹追踪及 224×224 裁剪；不要把 `video_mouth`、灰度嘴部视频或 AV-HuBERT 特征送入标准 LSE。
- 本地 HPMDubbing 的 `evaluate.py` 与 StyleDubber 的 `0_evaluate_V2C_Setting1.py` 没有提供 LSE 实现。采用 [SyncNet 官方实现](https://github.com/joonson/syncnet_python)及 [Wav2Lip 的 LSE 定义](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)。代码固定在 `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`；评测前验证源码及权重 SHA256。
- LSE-D **越低越好**，LSE-C **越高越好**。先对每个候选偏移的窗口距离取均值得到 `mdist`，再计算 `LSE-D=min(mdist)`、`LSE-C=median(mdist)-min(mdist)`。固定 25 fps、16 kHz 单声道 PCM16、5 视频帧/20 MFCC 帧、`vshift=15`（±0.6 秒）；保留官方零填充和窗口端点规则，不归一化图像/MFCC/嵌入。
- 输入生成 WAV 必须只含待配音片段，已去除 prompt/reference 音频。默认时长差大于 0.10 秒记为失败；如研究协议确需只评有效重叠区，显式设置 `--duration-policy overlap`，并报告原时长及差值。不要自动拉伸、补静音或按最优 offset 移动音频再评分。
- 配音短片段默认 `--min-track 10`（要求 **超过** 10 次检测），其余沿用上游默认：检测 scale=0.25、置信度=0.9、最小脸尺寸=100、crop scale=0.4、最大漏检间隔=25。当前 213 条清单中有 19 条不超过 25 帧，最短 16 帧（0.64 秒）；上游默认 min_track=100 和较保守的 25 都会漏样本。改变门槛必须对比较的模型和 GT 一致使用并记录，短片段保留窗口数供审计。预处理适配器 `pipeline.py` 显式指定中间 AVI 音频为无损 PCM16，避免上游未指定编码器时 FFmpeg 默认 MP3 引入延迟、尾部填充及重复视频帧；其余检测/跟踪/裁剪代码来自固定的官方源码。
- 多轨迹默认 `--track-policy longest`：按视频帧数选择最长轨迹，同长取首条，不按生成音频的 LSE 挑选。可显式选择 `single`（多轨迹时报错）或 `mean`（轨迹等权均值）。所有轨迹分数、offset 和选中轨迹均保存。跨片段使用等权样本均值，不按窗口数加权。
- 最优 offset 和零偏移距离也会保存：LSE 搜索最优偏移，因此较高 LSE-C 不能独立证明音画在零偏移处同步。HPMDubbing 未公开足够的裁剪/筛选细节，本工具实现同类指标，不能据此声称精确复现其论文数值。

### 安装与检查

```bash
ROOT_PREFIX="${ROOT_PREFIX:-}"
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
LSE="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/evaluation/lse"
ASSETS="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/syncnet"

"$PYTHON" -u "$LSE/install.py" --asset-dir "$ASSETS" --with-example
"$PYTHON" -m unittest discover -s "$LSE" -p 'test_*.py' -v
```

安装器复用现有环境，只补缺少的轻量依赖；不升级 torch/numpy/OpenCV。官方仓库、`syncnet_v2.model`（音画同步网络）、`sfd_face.pth`（人脸检测器）和可选 `example.avi` 均保存在仓库外的 `$ASSETS`。安装器校验哈希、严格加载两个 checkpoint，写出 `installation.json`；禁止把权重、原视频、裁剪帧或运行结果提交 Git。FFmpeg/ffprobe 优先使用当前 Python 同目录中的可执行文件，无需手动激活环境。

官方样例已经是 224×224、25 fps 的 SyncNet 裁剪，可跳过检测进行安装自检（每次使用新的输出目录）：

```bash
"$PYTHON" -u "$LSE/evaluate.py" \
  --video "$ASSETS/example.avi" --input-kind syncnet-crop \
  --asset-dir "$ASSETS" --device cuda \
  --output-dir "${ROOT_PREFIX}/zjw524/projects/data/evaluations/lse_official_demo"
```

官方参考约为 offset=3 帧、LSE-D=5.353、LSE-C=10.021，平台/编解码版本会有小幅差异。`syncnet-crop` 只接受 224×224、25 fps 视频，仍需确认它由正确的脸部裁剪协议产生；不能把嘴部区域放大后绕过检查。CPU 可使用 `--device cpu`。

### CelebV-Dub Setting 1 评测

沿用上面的变量，并把 `GEN_WAV_DIR` 指向目标实验实际生成结果根目录（应含 `test/视频ID/片段ID.wav`）。`OUT_DIR` 必须是新的独立目录。下例只启动评测，不重新生成音频：

```bash
GEN_WAV_DIR="/path/to/experiment/eval_s1_update"
OUT_DIR="${ROOT_PREFIX}/zjw524/projects/data/evaluations/lse_experiment_update"
mkdir -p "$OUT_DIR"
setsid env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$LSE/evaluate.py" \
  --test-list "${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst" \
  --video-root "${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/video" \
  --audio-root "$GEN_WAV_DIR" --asset-dir "$ASSETS" \
  --device cuda --output-dir "$OUT_DIR" \
  > "$OUT_DIR/run.log" 2>&1 &
LSE_PID=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$LSE_PID"
```

先加 `--limit 2` 可进行小样本测试；小样本测试与正式评测使用不同输出目录。GT 对照将 `--audio-root` 换成 `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/audio`，使用另一输出目录，其余参数保持相同。单个视频可用 `--video /path/to/clip.mp4 --audio /path/to/generated.wav --output-dir /path/to/new_result`；仅在 GT 自检时省略 `--audio` 使用视频自带音轨。

其他目录结构使用 `--manifest pairs.jsonl`，每行必须明确给出 `id`、`video`、`audio`，例如 `{"id":"test/video1/clip1","video":"/data/video1.mp4","audio":"/results/clip1.wav"}`。相对路径基于清单所在目录；只有显式 `"audio":null` 才使用视频自带 GT 音轨。不通过文件 basename 猜测配对，重复完整 ID 直接报错。

输出包括 `results.jsonl`（每个请求片段一行，包含成功或失败）、`summary.json`（样本均值和覆盖率）、`protocol.json`（参数、软件版本和权重来源）、`logs/`（每片段预处理日志）。默认清理临时帧，`--keep-work` 可保留裁剪供排查。任何缺文件、无脸、片段过短、时长不符或计算异常均记录失败，并使脚本最终非零退出。正式报告必须核对清单、`requested/succeeded/failed/coverage/complete`；**不能把仅成功子集的均值冒充完整测试集结果**。同一张对比表必须使用相同样本、轨迹策略和预处理参数，分数展示五位小数。

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



