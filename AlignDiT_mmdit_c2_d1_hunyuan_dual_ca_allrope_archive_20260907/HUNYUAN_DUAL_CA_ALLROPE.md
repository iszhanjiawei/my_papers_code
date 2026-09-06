# C2/D1: Hunyuan dual text cross-attention and all-head RoPE

This variant implements the three changes requested on 2026-09-05. Historical
configs and checkpoints retain their original architecture; select the new
configs below to use the revised network. It is not a complete Hunyuan model.

## Architecture contract

- Multimodal blocks first perform Audio/Video joint attention, then both
  Audio and Video query the same ordered Text sequence via cross-attention.
- Audio/Video CA queries have separate projections; Text K/V are shared.
- CA explicitly uses per-head Q/K RMSNorm (epsilon 1e-6), separate A/V
  normalization, timestep modulation and residual gates, and output projections.
- CA RoPE follows Hunyuan: adjacent channel-pair rotation, theta 10000,
  independent ordinal Audio/Video/Text positions, on every head.
- `pe_attn_head: null` enables all-head RoPE in AV joint attention AND in the
  native audio-only blocks. This changes attention computation, not parameter
  shapes. The native audio-only attention/FFN module structure is retained.
- AV RoPE still uses Audio positions `i`, Video positions `4j`. Interleaved
  RoPE, Synchformer conditioning and ATST/REPA are NOT added in this experiment.
- The character encoder, AV-HuBERT mouth inputs, mel representation, reference
  audio mechanism, existing padding policy, and CTC objectives are retained.
- New CA gates are zero-initialized but their output projections are not, so
  the gates can learn immediately. Old text CA weights are not a valid direct
  substitute for the new module.

## Configurations and initialization

- D1: `finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope`
  - 6 multimodal + 12 native audio-only blocks.
  - CTC after blocks 6 and 12, averaged, weight 0.1.
  - New run name:
    `AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_6MM12A_CTC6_12_finetune`.
- C2: `finetune_celebvdub_mm_c2_hunyuan_dual_ca_allrope`
  - 12 multimodal + 6 native audio-only blocks.
  - Retains C2's zero-based CTC indices `[6, 12]` (blocks 7 and 13).
  - New run name: `AlignDiT_MMDiT_C2_HunyuanDualCA_AllRoPE_12MM6A_finetune`.
- Both inherit the original LibriSpeech 500k pure-audio warm start and base
  optimization settings. They do not load old C2/D1 multimodal checkpoints.
- All non-architecture training settings are inherited unchanged from the
  corresponding baseline, including `checkpoint_activations=False` and
  `log_samples=True`. Do not silently adjust them to fit memory.
- The initial 2026-09-05 launch incorrectly overrode those two settings. That
  run was stopped before its first checkpoint; its logs/events/Hydra output
  are archived separately. The corrected formal run restarts at update zero
  from the same LibriSpeech pretrained checkpoint, not from the aborted run.

## D1 launch and telemetry

From this project root, launch through `setsid` with unbuffered output:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope_4x4090.sh \
  > logs/train_d1_hunyuan_dual_ca_allrope.log 2>&1 < /dev/null &
```

The launcher accepts Hydra overrides for explicitly named smoke runs. Never
use the production checkpoint path for a canary with changed training settings.

The C2 four-GPU launcher is:

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_hunyuan_dual_ca_allrope_4x4090.sh \
  > logs/train_c2_hunyuan_dual_ca_allrope.log 2>&1 < /dev/null &
```

The formal run's TensorBoard directory is:

`runs/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_6MM12A_CTC6_12_finetune_hifigan_16k_char_CelebVDub`

It records `loss`, `diff_loss`, `ctc_loss`, and `lr`. Start TensorBoard in its own
`setsid` session, confirm scalar events, HTTP access and the actual forwarded
address, and record live PIDs/ports in the handoff (not hardcoded here).

Use `python -m tensorboard.main`, not `python -m tensorboard`, with the installed
TensorBoard distribution. Only the global main training rank creates the writer.

## Regression checks

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_hunyuan_dual_ca.py
```

The tests cover independent CA/RoPE numerical reference calculations, text
padding exclusion, both stream queries, gate gradient liveness, all-head RoPE,
legacy state-dict compatibility, and 18-layer C2/D1 forward/backward with and
without activation checkpointing. The final MM video output has no consumer in
the audio-only tail; retain the trainer's `find_unused_parameters=True`.
