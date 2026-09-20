# Step 3: block video queries from reading audio

This experiment is an independent copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_av_local_gate_step2`.
All 238 tracked files were copied from repository commit
`6ad3cf4366816dc1e56feed66a5768c06d193972` with distinct file inodes.
Source files are regular copies; logs, TensorBoard events, checkpoints and caches
are not inherited. Frozen datasets, speaker embeddings and the pinned audio
parent remain shared read-only inputs. The step-2 source directory is unchanged.

## Experimental change

The active speaker configuration sets `block_video_audio_attention: true`.
In the first 12 MM-DiT layers, video queries cannot read audio keys/values.
Here each arrow means Query -> Key/Value, so V -> A is audio feedback into video.

| Query -> Key/Value | Step 3 |
| --- | --- |
| A -> A | Global, unchanged |
| A -> V | Radius 2 on the 40-Hz grid (+/-50 ms), with the step-2 delta gate |
| V -> A | Blocked |
| V -> V | Global, unchanged |

The gate remains one unconstrained learnable scalar per MM block, initialized
to `1e-5`, and combines `O_AA + gate * (O_AJ - O_AA)` before the existing audio
output projection. The reference-prefix convention, padding behavior, null-video
and CFG handling, text and speaker conditions, final six audio blocks and all
losses otherwise retain step-2 semantics. There is no step-4 local V -> V window.

`block_video_audio_attention` defaults to `false` for historical configurations;
the active step-3 YAML explicitly enables it. It adds no parameters or checkpoint
keys, so inference must load this architecture config as well as the weights.

Blocking V -> A changes forward information flow. It does not freeze the video
branch or detach its gradients: audio generation can still train visual
representations through A -> V. Video hidden states still transform across
layers and retain the existing flow-timestep modulation. The claim is independence
from the current audio hidden states through this attention connection, not
constant visual features or guaranteed better synchronization.

## Initialization and budget

This is a fresh run from the same pinned S2c update-70,000 EMA audio parent as
steps 1 and 2. It does not resume a trained step-2 checkpoint. The optimizer and
update counter start fresh. The active configuration remains
`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`.

Preserved settings: seed 666; 79,613 CelebVDub records; normalized 64-D/40-Hz audio;
CAM++ 192-D speaker conditions in blocks 12..17; four GPUs with bf16; 3,600 frames
per GPU; gradient accumulation 1; LR `5e-5`; 20k warmup; 200-epoch scheduler;
CTC zero through 10k and linearly reaching 0.03 at 30k; stop at update 200k;
numbered checkpoints every 50k and last checkpoint every 5k.

Model run name:

```text
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step3_r2
```

Checkpoint directory (prepend `ROOT_PREFIX` on other installations):

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step3_r2_40hz_CelebVDub_char
```

TensorBoard events are isolated under this project's:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step3_r2_semantic_vae_40hz_CelebVDub_char
```

## Launch and monitoring

From this experiment directory, after validation:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_av_local_gate_step3_r2.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

The launcher selects this source copy with `PYTHONPATH=src`. The default
distributed port is 29623 and TensorBoard port is 6008, overridable through
`TRAIN_PORT` and `TENSORBOARD_PORT`. The launcher rejects busy GPUs; inspect
resource availability before starting. Long-running training and TensorBoard
services use independent `setsid` sessions.

Only the main process writes TensorBoard events, including total/flow loss,
CTC weighted loss and coefficient, speaker projection diagnostics, and each
visual gate plus its gradient norm. Verify worker progress, finite losses,
event growth and HTTP scalar responses after launch. Record actual PIDs,
logdir, observed progress and forwarding status in ignored
`logs/launch_status.json`; a local HTTP URL is not a verified public forwarding URL.

## Comparison and evaluation

Compare step 3 with step 2 at matching training budgets, inference parameters
and reference-audio protocols. The active speaker inference and four-metric
evaluation scripts select the isolated step-3 checkpoint directory and this
experiment's architecture config. Their inherited Setting-1 protocol uses the
target clip's full ground-truth audio as reference; it is not an independent
reference evaluation. Forward-path checks establish implementation behavior,
while AVSync, WER, SPKSIM and listening tests establish trained-model effects.

## Pre-launch validation

On 2026-09-21, `smoke_test_va_block_step3.py` passed the video-only explicit
softmax oracle, zero VA dependency/gradient, global AA/VV and local live AV
gradient checks, unchanged same-input audio output versus step 2, multilayer
video independence, packed CFG, reference-prefix sampling, all-video padding,
bf16 backward, activation recomputation and model/optimizer/EMA round trips.
The inherited step-1 and step-2 contract suites also passed with the new switch
disabled. No new parameter keys are introduced.

`smoke_test_semantic_vae_c2_speaker_real_parent.py` passed on CUDA with the
actual S2c 70k EMA parent and two real training records, at CTC weights 0 and
0.03. All 12 visual gates had finite nonzero gradients. Strict migration loaded
the same 303 of 716 target keys, including matching online and frozen EMA copies;
the test performed zero optimizer updates.

Resolved configuration comparison against step 2 found exactly three changes:
the run name, checkpoint directory and `model.arch.block_video_audio_attention`.
The 238 source hashes in the original step-2 directory remain unchanged. The
new run reuses only read-only data, decoder and parent-checkpoint inputs.
