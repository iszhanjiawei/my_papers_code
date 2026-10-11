# 非 VAE C2 + CAM++ 显式说话人条件

本目录是从 `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` 单独复制出的实验快照；原目录未修改。

## 实验变量

- 声学表示仍为 16 kHz、80 维、100 Hz mel；没有 Audio VAE。
- 文本、视频、双 CTC、HiFiGAN、训练 mask 比例和 C2 的 12+6 层结构保持不变。
- 冻结的双语 CAM++ 从每条**完整且尚未 mask 的原始音频**提取 192 维 embedding。
- embedding 经 L2 normalization 后，由零初始化且无 bias 的 `Linear(192, 768)` 投影。
- 前 12 层仍使用原 timestep embedding；仅第 12--17 层 audio-only DiT 使用 `t + speaker_delta`。
- speaker 与 prompt audio 联合 dropout；CFG 的 full/TTS 分支保留二者，null 分支同时清零。

新增参数为 147,456。speaker 投影初始化为零，因此 update 0 与原 C2 数值等价。

## CAM++ 数据契约

权重：

```text
/zjw524/projects/data/pretrained_models/3D-Speaker/
  speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
```

SHA256：

```text
92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199
```

缓存：

```text
/zjw524/projects/data/CelebVDub/campplus_spk_emb_zh_en_16k/
  train/<video_id>/<clip>.npy
  test/<video_id>/<clip>.npy
```

预处理严格使用 16 kHz、80D Kaldi FBank、逐 chunk 均值归一化、10 秒非重叠 chunk、整条语音循环补齐，最多 90 秒。多个 chunk 的 CAM++ 输出先算术平均，再做最终 L2 normalization；缓存必须为 finite `float32[192]` 且范数误差不超过 `1e-4`。

四卡提取：

```bash
bash src/aligndit/run/misc/extract_campplus_celebvdub_4x4090.sh
```

只有 `metadata.json` 的 `status` 为 `complete` 且 coverage 审计通过后，训练入口才会放行。

## 训练和推理

正式配置：

```text
src/aligndit/config/finetune_celebvdub_mm_c2_campplus_speaker.yaml
```

四卡训练：

```bash
bash src/aligndit/run/train/finetune_celebvdub_mm_c2_campplus_speaker_4x4090.sh
```

200k 推理：

```bash
bash src/aligndit/run/eval/infer_celebvdub_s1_c2_campplus_speaker_200k.sh
```

SPK/WER/EMO/AVSync 四项评测：

```bash
bash src/aligndit/run/eval/eval_celebvdub_s1_c2_campplus_speaker_200k.sh
```

正式实验从与 C2 相同的 `AlignDiT_pretrain_LibriSpeech_500000.pt` 开始训练。历史 C2 150k/200k 权重当前不在服务器上，且历史配置没有记录全局模型初始化 seed；因此与历史表格的比较不是严格同 seed A/B。后续如需严格归因，应以本配置的 `seed: 666` 同时重跑无 speaker 的 C2 control。
