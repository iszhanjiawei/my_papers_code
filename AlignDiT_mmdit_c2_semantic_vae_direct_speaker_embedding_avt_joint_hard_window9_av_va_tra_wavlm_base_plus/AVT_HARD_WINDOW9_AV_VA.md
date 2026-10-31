> Historical source experiment record copied for provenance. It does not report WavLM TRA training or evaluation. See [the active experiment](WAVLM_TRA_HARD_WINDOW9.md).

# AVT joint attention with a strict nine-frame A↔V window

This isolated snapshot copies
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_av_va`
and replaces its Gaussian audio/video logit prior with a strict temporal mask.
The hard-mask semantics follow
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_hard_window5_av_va`;
this experiment explicitly uses **radius 4, total width 9**, on the aligned
40-Hz audio/video grids. The source snapshots are separate experiments.

## Attention rule

For an audio query at index `i`, permitted video keys are the valid indices
`i-4, ..., i+4`, inclusive. Video queries use the transposed mask over audio keys.

```text
B_av[i,j] = 0       when abs(j - i) <= 4
B_av[i,j] = -inf    otherwise
B_va      = B_av.transpose(-1, -2)

              audio keys  video keys  text keys
 audio query      0          B_av         0
 video query      B_va        0            0
 text query       0          0            0
```

Every query still normalizes jointly across concatenated audio, video and text
keys. The `-inf` mask therefore makes out-of-window A→V and V→A attention weights
exactly zero. Valid within-window cross-modal logits receive no distance penalty.
At sequence boundaries the window is clipped to existing keys; interior
positions have nine permitted cross-modal keys. Supplied padding masks are
respected, including the inherited policy that suppresses A/V padding masks
during training. AA, VV and all text-related edges retain their original
attention behavior.

The radius is **4 frames = 100 ms** on each side of the center, with zero offset.
The full nine-position span is from -100 ms through +100 ms. In the source
Gaussian experiment, sigma 100 ms also represented four frames, but was a
standard deviation: cross-modal attention outside that range remained possible.
Thus this run imposes a new strict nine-position boundary. The inherited sigma
setting is retained for configuration compatibility and is unused in hard mode.

The mask is parameter-free, content-independent and shared across heads,
layers, samples and CFG branches. The topology remains **12 AVT MM-DiT blocks
+ 6 audio-only blocks**, with 376,563,842 trainable parameters. CAM++ speaker
conditioning, Semantic-VAE latents, text stream, CTC, padding policy and model
state keys follow the source experiment. No new loss is introduced.

## Configuration and initialization

Use
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va.yaml`.
It inherits the Gaussian AVT config and changes only:

- `model.arch.temporal_band_fixed_window_radius_frames: 4`;
- `model.name`, with the marker `avt_joint_hard_window9_av_va`;
- `ckpts.save_dir`, using the same experiment marker.

Initialize compatible weights from the same pinned **S2c 70k EMA** parent:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`.
A fresh run starts its optimizer and update counter from zero. The completed
Gaussian 150k/200k checkpoints are retained for comparisons.

Training retains seed 666, four GPUs, BF16, 3,600 frames/GPU, up to 32 samples/GPU,
LR 5e-5, 20k LR warmup, the original 200-epoch decay horizon and stopping at
200k updates. CTC is zero through 10k and ramps to 0.03 at 30k. Numbered weights
are saved every 50k and `model_last.pt` every 5k. All 79,613 source training
records remain in the split. Keep `speaker_training_contract.json` alongside
weights: the hard mask has no tensor state, so its radius and semantics are
validated through that contract when resuming or running inference.

## Real-data smoke run and formal training

Run from this snapshot's root, using the standard environment directly.
The launcher sets `PYTHONPATH=src` and uses
`${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u` for Accelerate.
Both smoke and formal launches below use independent sessions through `setsid`.

First check that physical GPUs 0–3 and DDP port 29637 are available. The launcher
rejects any of these GPUs using more than 500 MiB. The DDP port can be changed
with `TRAIN_PORT`. Start TensorBoard before training; serving `runs/` makes both
smoke and formal curves visible in one service.

```bash
source env.sh
mkdir -p logs
TENSORBOARD_LOGDIR="$PWD/runs" bash scripts/start_speaker_tensorboard.sh
smoke_name="avt_joint_hard_window9_av_va_smoke3_$(date +%Y%m%d_%H%M%S)"
smoke_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/$smoke_name"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va_4x4090.sh \
  "model.name=$smoke_name" "ckpts.save_dir=$smoke_dir" optim.run_until_update=3 \
  > "logs/$smoke_name.log" 2>&1 < /dev/null &
smoke_pid=$!
printf '%s\n' "$smoke_pid" > "logs/$smoke_name.pid"
ps -o pid,ppid,sid,tty,stat,cmd -p "$smoke_pid"
```

Confirm the smoke run reaches three finite updates, writes TensorBoard losses
and saves `model_last.pt` at update 3. Keep the original batch size, scheduler
and save intervals; the final-stop save produces this smoke checkpoint. Check
strict online/EMA checkpoint reload, then wait for every smoke worker to exit.
Use an empty formal checkpoint directory for the first formal launch; an
existing matching checkpoint directory triggers resume.

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_hard_window9_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_hard_window9_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Formal checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Formal TensorBoard event directory relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

The TensorBoard startup script defaults to port **6021** and this exact formal
event directory; override `TENSORBOARD_LOGDIR` as above to serve all runs or
`TENSORBOARD_PORT` if the port is occupied. Only rank 0 writes events. Check
finite total/flow/CTC loss, event updates, worker/GPU activity, the TensorBoard
HTTP response and actual scalar tags. Record the service PID, port, logdir and
actual forwarded URL. Open the application's bottom Ports panel and use the
forwarded-address link for the active port.

## Inference and evaluation

The copied speaker inference/evaluation entry points now default to this
hard-window9 config, checkpoint directory and dedicated output directories:

```text
src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh
src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

The protocol remains CelebV-Dub Setting 1, 213 clips, same-clip reference, EMA,
seed 0, CFG text/video 5/2 and 32 NFE. `CKPT_STEP` defaults to 200000; `EVAL_GPU`,
`CFG_VIDEO`, `CFG_TEXT`, `NFE`, `INFER_SEED` and `OUTPUT_DIR` remain configurable.
Weights with a Gaussian or other incompatible temporal contract are rejected.

The inherited `AVT_FIXED_BAND_AV_VA.md`, `AVT_JOINT_ATTENTION.md` and evaluation
reports are historical source-experiment records. Their measured results do
not describe this strict-window9 experiment. Logs, event files, generated
samples and checkpoints are runtime artifacts excluded from Git.

## Validation (2026-10-27)

- All 17 attention tests pass, including independent full joint-softmax
  equivalence, exact zero AV/VA probabilities outside ±4, inclusion of both
  endpoints, 3,073-frame integer geometry, CFG, checkpointed gradients and
  CUDA BF16 finite backward for fully masked rows. Gaussian mode with an
  omitted/None radius matches the source model initialization and outputs
  exactly. The hard mask adds no parameters, state tensors or RNG consumption.
- All 11 new hard-window contract tests and all 11 inherited Gaussian
  contract tests pass. Shell syntax and the three-field configuration
  comparison pass. Newly added/updated supporting Python files pass Ruff;
  the backbone retains the same 24 pre-existing Ruff diagnostics, with no
  new diagnostics compared with the source.
- Independently constructed production models using the real 159-symbol
  vocabulary and complete CFM wrapper both have 376,563,842 trainable
  parameters, 802 identical state keys/shapes/dtypes, 12 AVT blocks and
  6 audio-only blocks.
- The separate real-data run `avt_joint_hard_window9_av_va_smoke3_20261027`
  completes 3 updates on four GPUs at 3,600 frames/GPU. Total/flow losses are
  1.3477484, 1.4532063 and 1.3474190; global and speaker gradients are finite.
  TensorBoard records radius 4, width 9 and zero offset at all three steps.
  Its saved update-3 checkpoint reloads online and EMA weights strictly with
  all 802 model keys. All online, EMA and optimizer tensors are finite. EMA
  retains the original parent-step policy (70,000 + 3 = 70,003). The actual
  checkpoint sidecar passes the hard9 configuration and rejects Gaussian and
  radius-2 configurations. Detailed reports are retained under this project's
  ignored `logs/` directory.
