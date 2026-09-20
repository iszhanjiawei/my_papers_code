# Step 2: gated visual correction of local joint attention

This is an independent copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_av_local_step1`
at repository commit `58753e7bffaae70e571b5bb5f73cf8371bce4674`.
All 236 source files were copied as independent regular files. Runtime logs,
events, checkpoints and caches were not copied. Datasets, frozen speaker inputs,
the decoder and the pinned audio parent are shared read-only inputs.

## Exact experimental change

The first 12 MM-DiT blocks retain the step-1 dense joint SDPA with A-query/V-key
radius 2 on the shared 40-Hz grid (+/-50 ms). Each block adds one unconstrained,
learnable scalar `av_visual_delta_gate`, initialized to `1e-5`:

```text
O_AA = attention(Q_A, K_A, V_A)
O_AJ = step1_local_joint_attention(...).audio_rows
O_A  = O_AA + gate * (O_AJ - O_AA)
```

The implementation uses float32 `torch.lerp` and then restores the attention
output dtype. Both endpoints retain gradients; no value-dependent branch is used.
The existing audio output projection and dropout run once after the combination.
The existing AdaLN residual gate remains in place. QKV projections and RoPE are
shared by the two audio calculations; this adds exactly 12 scalar parameters,
but also an additional audio self-attention calculation in each MM block.

At the same layer input and weights, gate 0 recovers audio-only attention and
gate 1 recovers step 1. This does not restore a historical pretrained checkpoint
or remove visual information from preceding layers. The scalar is not a
probability, confidence predictor or flow-timestep schedule; it may learn values
outside [0, 1]. No new loss is introduced.

AA, VA and VV remain global. The final six audio blocks, text and speaker paths,
CTC schedule, padding behavior, complementary masking, reference-prefix offsets,
null-video handling and CFG branches retain step-1 semantics. There is no new
generation-region-only gate. This intentionally implements step 2 only, without
the step-3 feedback block or step-4 local video self-attention.

`av_visual_delta_gate_init: null` disables the extra calculation/parameters,
allowing the inherited step-1 checks and historical configurations to run.
The active speaker config explicitly sets `1.0e-5`. Training and inference both
construct the backbone from the complete architecture config. Checkpoint loads
remain strict; a step-2 checkpoint must be paired with the step-2 config.

## Initialization and training

This is a fresh run from the same pinned S2c update-70,000 EMA audio parent used
by step 1, with new optimizer state and update counter. It does not resume a
trained step-1 checkpoint. Strict migration loads the same 303 source keys,
validates the 12 newly initialized scalar gate keys, and otherwise preserves the
parent contract: 716 target keys, 413 new target keys for the active speaker model.

The active configuration remains:

`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`

Shared settings: seed 666; 79,613 CelebVDub records; normalized 64-D/40-Hz audio;
frozen CAM++ 192-D conditions in blocks 12..17; four GPUs with bf16; 3,600 frames
per GPU; accumulation 1; LR 5e-5; 20k warmup; inherited 200-epoch scheduler;
CTC zero through 10k, linearly reaching 0.03 at 30k; stop at update 200k;
numbered checkpoints every 50k and last checkpoint every 5k.

Model run name:

```text
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step2_r2
```

Checkpoint directory (prefixed by `ROOT_PREFIX` on other installations):

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step2_r2_40hz_CelebVDub_char
```

TensorBoard events are isolated under this project's:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_gate_step2_r2_semantic_vae_40hz_CelebVDub_char
```

From this directory, the launchers select their own source via `PYTHONPATH=src`:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_av_local_gate_step2_r2.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

The default distributed port is 29622 and TensorBoard port is 6007; override via
`TRAIN_PORT` / `TENSORBOARD_PORT` when necessary. In addition to existing total,
flow and CTC loss diagnostics, TensorBoard records each gate, their summary
statistics and the gate gradient norm. Launch-time PIDs, observed losses,
memory use and actual forwarding URL belong in ignored `logs/launch_status.json`.

## Validation and inference

Run the gate contracts and inherited checks before training:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_av_visual_gate_step2.py
PYTHONPATH=src OMP_NUM_THREADS=1 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_av_local_step1.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py
```

The active inference/evaluation launchers now use the step-2 checkpoint directory
and architecture config. Their inherited Setting-1 protocol uses the target
clip's full ground-truth audio as reference; it is not an independent-reference
evaluation. Compare step 1 and step 2 with matching training budgets and CFG.
Locality and endpoint checks establish implementation behavior, not improved
AVSync or sound quality; those require trained-model evaluation.

## Pre-launch validation

On 2026-09-21, the six step-2 CPU contract groups, inherited step-1 contracts and
speaker contracts passed. They cover independent softmax endpoint checks,
shared-parameter gradients at gate 1, gate/video gradients at initialization,
bf16, checkpoint/optimizer round trips, activation recomputation and batched CFG.
The real S2c parent and two real training samples also passed CUDA bf16
forward/backward at CTC weights 0 and 0.03, with finite nonzero gradients for all
12 gates and no weight updates. Strict migration loaded 303 of 716 target keys.

Resolved config comparison against step 1 found exactly three differences:
model run name, checkpoint directory and `av_visual_delta_gate_init`. All 236
source-file hashes in the original step-1 directory remain unchanged; copied
files have separate inodes and no cross-experiment source links.
