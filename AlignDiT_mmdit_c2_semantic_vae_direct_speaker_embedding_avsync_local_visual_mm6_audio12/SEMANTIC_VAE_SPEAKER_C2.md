# Semantic-VAE Direct-C2 + CAM++ + local visual attention: 6 MM / 12 audio

This independent project is copied from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual`
at commit `76c5f6a`. It contains its own source files, configuration and launchers.
The source project is the 12-MM / 6-audio control and is not edited by this experiment.
Logs, checkpoints, datasets and TensorBoard events are not copied into this snapshot.

## Exact experimental change

All block indices below are zero-based. Total depth remains 18.

| Component | Source control | This experiment |
| --- | --- | --- |
| MM-DiT joint-attention blocks | 0–11 | 0–5 |
| Text cross-attention blocks | 0–11 | 0–5 |
| Audio self-attention / FFN tail | 12–17 | 6–17 |
| Gated local visual attention | 12–17 | 6–17 |
| CAM++ speaker modulation | 12–17 | 12–17 |
| CTC taps | [6, 12] | [6, 12] |

The fully resolved Hydra configs differ only in `model.arch.n_mm_layers`
(12 to 6), `model.arch.n_text_layers` (12 to 6), the experiment name and the
checkpoint output directory. Text attention follows the multimodal boundary so
all twelve tail blocks are text-free audio blocks. The existing local attention
is constructed independently in each audio block. CAM++ stays in the original
last six blocks, consistent with changing only the requested layer split.

No backbone operation is rewritten: the existing block factory already supports
this layout. The migration and inference guards, launcher description and tests
are updated to recognize the new split. The original 12/12 configuration remains
supported by these guards.

## Conditions and pretraining

Each tail block runs audio self-attention, a gated local visual residual, then
its original FFN. Visual keys/values are the cached 1024-D AV-HuBERT features at
40 Hz. The local branch does not receive globally mixed MM-DiT visual states.
Audio and visual padding, prompt regions and CFG video dropout keep the previous
masking rules. All twelve adapters have independent parameters.

The Flowley window remains omega=0, delta=4 at 8 FPS, equivalent to a **0.5 s
radius / 20 tokens at 40 Hz**. The cosine weights produce an additive
`log(weight + 1e-6)` bias, with fade scale 1 in every layer. This is a soft
window; padding alone uses negative infinity. The OmniShow-style 768-channel
gate remains initialized to **1e-5**, directly multiplying the residual output.
Q/K/V/output projections and Q/K RMSNorm use their existing initialization.

Start from the same pinned **S2c 70k EMA audio parent**. The strict migration
loads exactly 303 audio-parent tensors and ignores the same 10 source projector
tensors. The new model contains **692** state entries: 303 loaded and **389**
new, including **132** local-adapter entries. Six fewer multimodal blocks remove
144 entries, and six extra local adapters add 66 entries, relative to the
770-entry control. Fresh adapter gates and every adapter shape are checked.
The 12+6 model's training checkpoints are not initialization checkpoints for
this different architecture; this experiment uses its own optimizer/update run.

The following settings are inherited without change: 64-D / 40-Hz Semantic-VAE,
all 79,613 training records, CAM++ embeddings, seed 666, bf16, four GPUs,
3,600 audio frames per GPU, LR=5e-5, 20k LR warmup, 200-epoch LR schedule,
CTC disabled through 10k then linear to 0.03 at 30k, and stop at 200k updates.
CTC taps remain [6, 12], and CTC downsampling strides remain [1, 1].

## Run from this project directory

The active configuration is
`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_ctc003_warmup.yaml`.
The filename is inherited, while the resolved architecture and experiment name
identify this 6-MM / 12-audio snapshot.

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_ctc003_warmup

mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_4x4090.sh \
  > logs/train_mm6_audio12.log 2>&1 < /dev/null &
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_local_visual_mm6_audio12_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

The independent checkpoint directory is
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_local_visual_mm6_audio12_ctc003_warmup10k30k_40hz_CelebVDub_char`.
Numbered checkpoints are saved every 50k and `model_last.pt` every 5k.
Only rank 0 writes TensorBoard; loss, CTC weight, gradient norms and all twelve
local gates are monitored. Resuming this experiment restores its model, EMA,
optimizer and scheduler states and purges unsaved TensorBoard steps.

For inference use `src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py`
with this project's active YAML as `--config` and its matching checkpoint as
`--checkpoint`. The inherited baseline speaker shell launchers still select
the baseline configuration. Keep the original S1 evaluation protocol when
comparing the 6+12 and 12+6 experiments; the layout change alone does not establish
an AVSync improvement.
