# MM12/audio6 visual progressive-window experiment

This directory is an independent copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual` at repository commit `7396f56`. Runtime logs, checkpoints and TensorBoard runs were not copied. The original experiment is unchanged.

Only the progressive visual-window behavior is ported from the MM6/audio12 text-gate progressive experiment. The model remains 12 multimodal blocks followed by 6 audio-only blocks, with visual residual attention in zero-based blocks 12–17. No tail text attention is added. Speaker conditioning still starts at block 12, CTC taps remain [6, 12], and all existing parameter names/shapes are preserved.

## Window semantics

`local_visual_window_schedule` accepts `fixed` (the backward-compatible default) or `flowley_progressive`. For an audio tail of N blocks, its local block index l uses `beta = 1 - l/(N-1)`; N=1 uses beta=1. The fade amplitude is `local_visual_window_fade_scale * beta`.

| Zero-based block | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---:|---:|---:|---:|---:|---:|
| beta | 1 | 0.8 | 0.6 | 0.4 | 0.2 | 0 |

The outer radius remains 0.5 seconds (4 reference frames at 8 Hz; 20 tokens at 40 Hz), with core radius 0. This is a layer-depth schedule, not a training-time schedule or a shrinking geometric radius. The center weight stays 1, while the cosine fade is scaled by beta. Attention uses `log(weight + 1e-6)`: at beta=0 noncentral keys retain the epsilon floor. Padding is still excluded. The per-channel residual gate is initialized to `1e-5` as in the original experiment.

The schedule is configuration, not a checkpoint tensor. **Train, resume and infer with the new progressive YAML/entry points**, even though fixed/progressive state dictionaries have matching shapes. The training contract records the schedule and actual per-layer window values.

## Training and monitoring

Run from this directory in the existing `aligndit` environment. New training loads the same S2c 70k pure-audio EMA parent, with fresh optimizer and update counter. The inherited protocol is seed 666, four RTX 4090 GPUs, bf16, 3600 frames/GPU, LR 5e-5, 20k LR warmup, CTC 0 through 10k then linear to 0.03 at 30k, and stop at 200k. Checkpoints are saved every 50k with a rolling checkpoint every 5k.

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash \
  src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_progressive_window_4x4090.sh \
  > logs/train_progressive_window.log 2>&1 < /dev/null &
```

The model/run prefix is `AlignDiT_MMDiT_c2_svae_speaker_local_visual_flowley_progressive_ctc003_warmup10k30k`. The checkpoint directory is `${ROOT_PREFIX}/zjw524/projects/data/ckpts/` plus that prefix and `_40hz_CelebVDub_char`. TensorBoard events are under this project's `runs/`, with that prefix and `_semantic_vae_40hz_CelebVDub_char`.

```bash
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_local_visual_flowley_progressive_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

Main-rank events include `loss`, `diff_loss`, `ctc_lambda`, `ctc_weighted_loss`, global gradient norm, speaker diagnostics and local visual gate/gradient diagnostics. Raw `ctc_loss` begins after the 10k CTC warmup boundary because its forward is skipped while disabled.

## Validation and later inference

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/test_progressive_visual_window.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_progressive_window_ctc003_warmup
```

The GPU smoke validates real parent hashes and exact migrated tensors, real cached samples, the six window scales, finite forward/backward at CTC weights 0 and 0.03, and speaker/local-attention gradients. It performs no optimizer updates. At beta=0, Q/K gradients may round to zero in bf16; gates, V and output projections must still receive nonzero gradients.

For a completed checkpoint, `CKPT_STEP=150000 EVAL_GPU=0 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_local_visual_progressive_window.sh` uses the new config/checkpoint directory and runs the inherited CelebV-Dub Setting 1 inference plus four metrics. Use `infer_celebvdub_s1_svae_speaker_local_visual_progressive_window.sh` for inference alone.
