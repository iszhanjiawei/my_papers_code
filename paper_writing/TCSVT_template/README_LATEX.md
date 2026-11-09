# TCSVT 模板的持久化 LaTeX 环境

论文主文档：`manuscript.tex`，依次载入 `abstract.tex`、`introduction.tex`、`related_work.tex`、`method.tex`、`experiments.tex`、`conclusion.tex`、`acknowledgment.tex`，使用 `references.bib` 管理参考文献。标题为 **JointDub: Locality-Aware Multimodal Diffusion Transformer for Synchronized Video Dubbing**，作者信息待补充。

`method.tex` 组织为 A. Preliminaries、B. Multimodal Diffusion Transformer for Video Dubbing、C. Gaussian-Windowed Audiovisual Attention 和 D. Three-Stage Progressive Training。A 小节参考 MMAudio 的组织方式，将条件流匹配和音频压缩合并，以段内斜体标题区分两部分。流匹配正文载入 `../local_files/flow_matching_preliminaries.tex`；`../local_files/flow_matching_references.bib` 的三篇配套文献已合并到 `references.bib`，统一通过主稿参考文献库编译。Lipman 等人的 Flow Matching 论文使用 `lipman2023flow` 引用键，避免重复列项。

A 小节中的音频压缩正文载入 `../local_files/audio_compression_revised.tex`，使用波形编码与解码的精简叙述。原流匹配与音频压缩小节的引用标签统一指向合并后的 A 小节。`../local_files/audio_compression_revised_references.bib` 的文献已与主文献库去重合并：MMAudio 复用现有 CVPR 正式发表条目，Semantic-VAE 使用作者提供的 2025 年 v1 预印本及 `niu2025semantic` 引用键。

B 小节直接载入 `../local_files/mmdit_backbone_concise.tex`，保留联合注意力和训练目标两条公式，以文字说明多模态表示、流间更新及说话人条件音频细化。配套的 `../local_files/mmdit_backbone_concise_references.bib` 与主文献库统一引用键并去重，实际引用的文献复用现有条目。隐藏维度、注意力头数、FFN 扩展比和说话人嵌入维度保留在 `experiments.tex` 的实现设置中。

C 小节直接载入 `../local_files/gaussian_windowed_attention.tex`，包含高斯时间先验、双向偏置矩阵、共享 softmax 归一化及窗口配置。其使用的 `tran2026precise` 条目来自前述配套文献文件，已并入主文献库；MMAudio 复用现有条目。方法各小节通过标签交叉引用，公式保留 IEEEtran 默认 10 pt 字号。

D 小节直接载入 `../local_files/progressive_training.tex`，描述继承的 LibriSpeech 声学初始化 → LibriTTS 文本–音频适配 → 视听配音适配。正文区分第一阶段保留音频上下文的 masked infilling、第二阶段的音频/文本联合注意力及说话人条件、第三阶段的视觉交互与 CTC。第一阶段是已有权重来源，不宣称从零完成三次训练。新增 LibriSpeech、LibriTTS、HuBERT 和 CineDub 文献至主文献库，课程学习比较限定于 CineDub §3.3.1。

D 小节实现依据：`../../AlignDiT_mmdit_c2_semantic_vae/src/aligndit/model/cfm_notext.py` 及 `config/pretrain_semantic_vae_warmstart_s2c.yaml` 对应第一阶段；`../../AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va/src/aligndit/config/pretrain_libritts_text_audio.yaml` 对应中间阶段；新增 `_libritts_pretrained` 后缀的最新实验目录内 `LIBRITTS_PRETRAINED_WINDOW13.md` 对应完整 EMA 迁移。中间阶段使用 `train-clean-100` 与 `train-clean-360`，训练 50k updates，不启用视觉、Gaussian bias 或 CTC；文本和说话人条件参数参与训练。子集名称继承自 LibriSpeech，不能将名称相加后声称实际使用 460 小时 LibriTTS；论文正文使用精确子集名，实际预处理后时长需以缓存清单汇总。

训练策略证据边界（2026-11-09）：最新实验目录 `logs/libritts_window13_150k_comparison.json` 和 `logs/libritts_window13_150k_protocol_comparison.json` 记录同 CelebV-Dub Setting 1、213 条、下游 150k EMA 的独立训练对照。增加中间阶段后 SPKSIM 0.65271→0.67540、WER 5.08831%→3.82675%、EMOSIM 0.75905→0.78698，AVSync 0.58568→0.57453。这是整套中间预训练方案的描述性收益，尚未隔离额外数据、计算量和课程顺序的影响。建议后续在同 LibriTTS 数据与 50k 预算下增加关闭文本的适配对照，并补齐同协议 Setting 2 对照；若主张顺序优势，再比较同数据/预算的联合训练。现有 GRID、zero-shot、VTS/VFA 表格不据此自动认定为新初始化结果；本次仅新增方法 D，原实验表格和留空消融小节保持其原有数值来源。

`experiments.tex` 包含两数据集 Setting 1/2、跨数据集参考音频、关闭 Text 的 Video-to-Speech 和 Visual Forced Alignment；五张结果表位于 `tables/results_*.tex`。数值来自项目根目录 `实验结果.md`。VTS 的 Text 关闭和当前模型重测依据作者在本次写作会话中的明确确认；旧 benchmark 文档中的历史模型/伪文本协议不代表此次结果。消融小节按要求留空。

`conclusion.tex` 总结方法、主要实验结论和已验证的任务扩展。`acknowledgment.tex` 按作者选择采用通用致谢，感谢公开数据集、预训练模型、基线实现及评测工具的贡献者；具体资助与致谢信息待作者后续提供。致谢不编号，置于参考文献之前。

原始示例入口 `bare_jrnl_new_sample4.tex`、`IEEEtran.cls`、图片和原版示例 PDF 均保留，供后续查阅排版用法。

## 在终端编译

```bash
cd /zjw524/projects/alignDiT_idea6/my_papers_code/paper_writing/TCSVT_template
./compile.sh manuscript.tex
```

论文 PDF：`build/manuscript.pdf`。日志、辅助文件和 SyncTeX 同样在 `build/`。
`latexmk` 会自动处理重复编译和参考文献。编译其他主文件可执行 `./compile.sh your_main.tex`。
不带参数的 `./compile.sh` 仍编译原始示例入口。
也可直接调用 `/zjw524/bin/latexmk -pdf -outdir=build your_main.tex`。

## 在 Devin 编辑和预览

远程环境已安装 LaTeX Workshop，工作区 `.vscode/settings.json` 已配置固定路径的编译工具。
打开 `.tex` 后，保存自动编译；也可使用命令面板的 LaTeX Workshop: Build LaTeX project。
macOS 通常使用 Cmd+Option+B 编译、Cmd+Option+V 预览。快捷键如有冲突，以命令面板为准。
默认采用 pdfLaTeX，适合当前 IEEE/TCSVT 英文模板；需要 Unicode/系统字体时可在 recipe 中选择 XeLaTeX。

## 实例重建后的恢复

TeX Live、宏包、字体、动态库、格式文件、缓存以及编辑器扩展缓存都保存在 `/zjw524`。
新实例没有原来的 `/root` 配置时，加载：

```bash
source /zjw524/texlive/env.sh
```

模板的 `./compile.sh` 和工作区的绝对编译路径不依赖上述 source，仍可直接使用。
连接/恢复 Devin Server 后，如需重新注册编辑器扩展及 shell 环境，执行：

```bash
/zjw524/bin/latex-env-init --with-editor
source /zjw524/texlive/env.sh
```

`latex-env-init` 不重装 TeX Live；LaTeX Workshop 从持久盘的已校验 VSIX 离线安装。
扩展同时已写入 `/zjw524/devin-server-persistent/extensions`，现有 Devin 恢复脚本会从此目录恢复。

## 安装布局

- TeX Live 2026：`/zjw524/texlive/2026`；稳定入口：`/zjw524/texlive/current`。
- 编译包装命令：`/zjw524/bin/pdflatex`、`xelatex`、`lualatex`、`latexmk`、`bibtex`、`biber`。
- 用户宏包：`/zjw524/texlive/texmf-local`；格式/配置：`2026/texmf-var` 与 `2026/texmf-config`。
- 字体动态库及 fontconfig：`/zjw524/texlive/runtime`。
- 缓存与临时文件：`/zjw524/texlive/cache`、`/zjw524/texlive/tmp`。
- 安装器与编辑器 VSIX：`/zjw524/texlive/installer-cache`、`/zjw524/texlive/editor-cache`。

后续缺宏包时，可运行 `/zjw524/bin/tlmgr install 宏包名`。当前仓库为清华 CTAN 镜像。

## 已完成验证

1. 当前稿件（含三阶段训练方法、实验章节及五张结果表）完成多轮编译，生成 10 页 PDF；无未定义引用或内容越界警告。方法 D 在第 5 页，已检查渲染后的正文排版。
2. 在空 HOME、仅含 `/usr/bin:/bin` 的 PATH 下强制复编译通过，验证不依赖 `/root` 初始化文件。
3. XeLaTeX + fontspec + Latin Modern Roman 的 Unicode 文本/公式测试通过。
4. 所有 28 个 TeX 格式文件成功重建；BibTeX/Biber、latexmk 和常用 IEEE 宏包已安装。

此验证模拟了缺少用户 shell 配置的环境；未实际重启实例。
