# Setting 2：身份分组证据与局限

本清单是自动说话人筛查后的跨句参考实验集；`auto_spk` 是算法分组，不是人工核实的身份真值。CAMPplus 用于构造，独立 WavLM 只做诊断，不改变选集。

## 官方证据

1. [VoiceCraft-Dub ICCV 2025 补充材料](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Sung-Bin_VoiceCraft-Dub_Automated_Video_ICCV_2025_supplemental.pdf)，§C “Data curation pipeline for CelebV-Dub” 的 **Speaker classification** 段落。该段明确说明同一源视频可能包含不同说话人；作者先按源视频分组，再使用说话人模型的成对相似度重新聚类。旧版 [KAIST 附录](https://mm.kaist.ac.kr/pubs/pdfs/kim25e.pdf) 中对应章节是 §B，不能把两版章节编号混写。PDF 页码尚未从本机 PDF 解析器独立核对，按章节与段落定位。
2. [官方 data/README.md](https://github.com/kaist-ami/voicecraft-dub/blob/main/data/README.md) 要求输入目录已按 speaker 分类。这是上游输入契约，不能证明任何名称类似 YouTube ID 的现有目录已经完成身份分类。
3. [官方 phonemize_lrs.py](https://github.com/kaist-ami/voicecraft-dub/blob/main/data/phonemize_lrs.py#L102) 第 102–105 行将父目录与 clip stem 用下划线连接；第 157–159 行为 audio token 使用相同 segment ID。
4. [官方 construct_dataset.py](https://github.com/kaist-ami/voicecraft-dub/blob/main/data/construct_dataset.py) 将 flattened 文件名最后一个下划线字段当 utterance ID，之前的前缀当 `speaker` 分组，生成组内不同 utterance 的训练配对。对当前文件布局等价于 `folder + '/' + clip.rsplit('_', 1)[0]`。

例如 `test/0IEYKinDZyc/2_0_0` 和 `test/0IEYKinDZyc/2_0_1` 共用训练配对前缀 `test/0IEYKinDZyc/2_0`。官方公开代码没有进一步解释 clip 中每个数字字段究竟是源段、track、speaker cluster 或 sentence index。因此这里称之为 **official pairing prefix**，不把它提升为人工身份标签。不同前缀也可能仍是同一人。

## 本地证据

- 本地 `papers_codes/StyleDubber_CeleVDub/scripts/prepare_celevdub_manifest.py` 设置 `speaker_id=None`；`celevdub/dataset.py` 设置 -1，并明确说明没有 verified speaker annotations。其 `data/features/speakers.json` 只是源视频索引，不是独立身份标注。
- 本地原始 `.npy` 示例为 256 维向量；没有配套 provenance/ID 标签，不能猜测这些向量就是人工 speaker ID。
- 同 folder 的声纹分数差异大，单凭 folder 选 reference 不可靠。

## 自动筛查的可解释性

CAMPplus 的 complete-link clustering 限制同一个自动簇中所有样本对均通过预设阈值；阈值是工程参数，没有在人工身份标注集上校准，不能给出 FAR 保证。选参考由固定 seed 的 hash 顺序决定，不最大化 WavLM 或最终生成指标。

WavLM 提取使用 `scripts/extract_setting2_wavlm.py`，全 213 条均成功，输出 L2-normalized 256 维 speaker embedding。`wavlm_pair_audit.tsv` 对固定配对给出独立分数、同/异 official pairing prefix、低于 0.4 的人工复核标记；0.4 也是未经校准的诊断值。高分不等于身份认证。

剔除低声纹可分性样本会改变评测难度，所以应报告：初始 213 条、最终保留/排除条数、每个排除理由、参考时长限制、筛查模型/阈值、GT-target 与 reference 的声纹分布。对所有比较方法固定同一 manifest。最终人工复核应至少抽查每个自动簇，并优先检查跨 official pairing prefix、短语音、身份分数较低和镜头切换样本。

音频 hash、转录去重和相关性检查可发现一部分复用内容；缺少原始时间戳时，不能宣称已经证明 source-time 完全不重叠。视频输入必须去音轨，避免部分模型意外读取目标 GT audio。

## 最终固定清单的独立审计

2026-10-03 固定版本：115 对、98 条排除；manifest SHA-256=`5d31f42d9766b4c500904dcca8e9a50230f5678744c57b49dc808e45accd7cf8`。逐对检查 ID 不同、音频 hash 不同、参考时长/词数、CAMPplus 阈值、局部重复检查结果、seed hash 选择规则、全部引用文件与哈希、inference/evaluation 顺序一致、115+98 完整覆盖原 213 条。核查 115 条 media audit，230 个视频文件均只包含 video stream。

独立 WavLM：均值 0.71719、中位数 0.71012、范围 0.54393–0.91307；低于 0.4 的标记为 0。49 对使用相同 official pairing prefix，66 对跨该前缀；未根据 WavLM 分数或前缀进一步删除样本。

局部重复检测函数的独立合成检查：将 1 秒完全相同语音样信号埋入两侧不相关信号，修复后的 0.5 秒滑窗扫描输出相关系数 1.0；独立随机对照为 0.08778。该检查验证了已发现并修复的“共享片段被不相关外部片段稀释”问题，但不提供任意编码变化或任意 source-time 重叠的完备保证。
