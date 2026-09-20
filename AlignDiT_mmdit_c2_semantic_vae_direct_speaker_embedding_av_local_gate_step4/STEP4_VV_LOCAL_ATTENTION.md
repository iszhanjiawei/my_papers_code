# Step 4: local video self-attention

This experiment is an independent copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_av_local_gate_step3`.
The 240 source files were copied from repository commit
`0279b716db4dd3aba16d9e75ef5facacc0e1dfa5` into separate regular files.
Source hashes and inode isolation are recorded in the ignored
`logs/source_step3_snapshot.json`. No training logs, events, checkpoints or
compiled caches are inherited. Frozen datasets, speaker embeddings and the
pinned audio parent are shared read-only inputs.

## Experimental change

The active configuration adds `model.arch.vv_local_window_radius: 2`.
Each video query can read video keys at indices `i-2` through `i+2`, truncated
at sequence boundaries. Video tokens are on the existing 40-Hz grid, so this
is a +/-50 ms window. It is independent of `av_local_window_radius`, even
though both active radii are 2 in this run.

| Query -> Key/Value | Step 3 | Step 4 |
| --- | --- | --- |
| A -> A | Global | Global |
| A -> V | Radius 2 + visual delta gate | Same |
| V -> A | Blocked | Blocked |
| V -> V | Global | Radius 2 |

Only the VV quadrant of the attention mask changes in each of the first
12 MM-DiT blocks. Local rules remain active during training, evaluation and
activation recomputation independently of the inherited padding-mask switch.
The 12 scalar visual gates still compute `O_AA + gate * (O_AJ - O_AA)` before
the shared audio output projection. At the same layer inputs and weights,
step 4 leaves the audio output unchanged; later layers can use the changed
video output. Text/speaker paths, the final six audio blocks, flow/CTC losses,
reference-prefix coordinates and null-video/CFG policies retain step-3 behavior.

`vv_local_window_radius` defaults to `None` (global VV), introduces no parameter
or checkpoint keys and rejects negative, noninteger and boolean radii.
Inference must use the step-4 architecture configuration as well as its weights.
The mask limits direct VV attention within each layer. AV-HuBERT, the visual
encoder and propagation across layers still provide broader context; this is
not a strictly local end-to-end visual receptive field. No video parameters
are frozen or detached.

## Initialization and training budget

Each ablation starts independently from the same pinned S2c update-70,000 EMA
audio checkpoint, with a fresh optimizer and update counter. Step 4 does not
resume a trained step-3 checkpoint.

The active config is
`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`.
Its resolved differences from step 3 are exactly the VV radius, model run name
and checkpoint output directory.

Preserved settings: seed 666; 79,613 CelebVDub records; normalized 64-D/40-Hz
audio; CAM++ 192-D speaker conditions in blocks 12..17; four GPUs with bf16;
3,600 frames per GPU; gradient accumulation 1; LR `5e-5`; 20k LR warmup;
200-epoch scheduler; CTC zero through update 10k and reaching 0.03 at 30k;
stop at update 200k; numbered checkpoints every 50k and last checkpoint every 5k.

Run name:

```text
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step4_r2
```

Checkpoint directory (prepend `ROOT_PREFIX` on other installations):

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step4_r2_40hz_CelebVDub_char
```

TensorBoard events are isolated under this project's:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step4_r2_semantic_vae_40hz_CelebVDub_char
```

## Validation and launch

Run `src/aligndit/script/misc/smoke_test_vv_local_step4.py` with `PYTHONPATH=src`.
The inherited step-1/2/3 suites exercise the backward-compatible global-VV
default. The real-parent integration script
`smoke_test_semantic_vae_c2_speaker_real_parent.py` uses the active step-4 config
and checks actual parent migration and real-batch forward/backward without
updating weights.

Validation completed on 2026-09-21: the new CPU suite passed its independent
explicit-softmax oracle, local VV/AV and blocked VA dependency checks, global
AA and gate endpoint checks, None/wide-window fallback, padding/empty-video
handling, CFG/prefix sampling, bf16 backward and activation recomputation.
The inherited step-1/2/3 suites all passed with the VV option disabled.
The CUDA integration test passed with the actual S2c parent and two real
training records at CTC weights 0 and 0.03, including finite nonzero gradients
for all 12 visual gates. It performed zero optimizer updates.

After validation, from this experiment directory:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_av_local_gate_step4_r2.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/train_av_local_gate_step4_r2.pid
bash scripts/start_speaker_tensorboard.sh
```

The launcher selects this source copy with `PYTHONPATH=src`, checks GPU
availability and uses distributed port 29624. TensorBoard defaults to port
6009. Override these only through `TRAIN_PORT` and `TENSORBOARD_PORT` when
needed. Both long-running services use independent `setsid` sessions.

TensorBoard records total/flow loss, CTC loss and coefficient, speaker
diagnostics and all visual gates. Verify worker updates, finite losses, growing
events and HTTP scalar responses. Runtime PIDs and checks belong in ignored
`logs/launch_status.json`. Use the actual port-forwarding address displayed by
the current interface; a local HTTP URL is not a verified forwarded URL.

The active speaker inference and evaluation launchers select this experiment's
weights and architecture config. Compare against step 3 with matched training
budgets, CFG and reference-audio protocols. The inherited Setting-1 protocol
uses the target clip's full ground-truth audio as reference and is not an
independent-reference evaluation.
