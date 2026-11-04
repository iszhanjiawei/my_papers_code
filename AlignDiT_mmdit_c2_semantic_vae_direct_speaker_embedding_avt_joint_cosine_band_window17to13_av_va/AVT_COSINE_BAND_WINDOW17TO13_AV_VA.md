# AVT joint: cosine 17-to-13-frame Gaussian AV/VA curriculum

This is an independent source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window17_av_va`.
The source experiment remains unchanged. Trained weights, logs and other runtime
artifacts are not copied. This run starts afresh from the same S2c 70k EMA parent;
it does not continue the trained 17-frame model.

## What changes

Only the width of the existing Gaussian AV/VA attention prior changes during
training. At 40 Hz, sigma=200 ms is the nominal +/-8-frame core (17 samples),
and sigma=150 ms is the nominal +/-6-frame core (13 samples). Gaussian tails
remain outside the core; there is no hard cutoff, new mask or knowledge
distillation loss.

For the optimizer update being computed, `u`:

```text
p(u) = clamp((u - 10000) / (50000 - 10000), 0, 1)
a(u) = (1 - cos(pi * p(u))) / 2
sigma(u) = 0.200 + (0.150 - 0.200) * a(u)  # seconds
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / sigma(u))**2
B_va = B_av.transpose(-1, -2)
```

Updates 1 through 10,000 use sigma=0.200 seconds, update 30,000 uses 0.175,
and updates 50,000 onward use 0.150. The schedule uses absolute optimizer
updates in this fresh run, not epochs, microbatches, diffusion/flow time,
wall-clock time or the EMA parent's inherited counter. Resume uses the saved
completed-update count, so the curriculum does not restart. Every microbatch
within one optimizer update uses the same sigma. Training hooks keep the online
and EMA transformer widths consistent with that update.

The 12 AVT joint blocks, 6 audio-only blocks, offset=0, AV/VA directions,
speaker conditioning, parameter tensors/count, AA/VV/text attention, datasets,
latent normalization and losses retain their source behavior.

## Configuration

Use
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_cosine_band_window17to13_av_va.yaml`.
It inherits the copied window17 config and adds:

```yaml
model:
  arch:
    temporal_band_fixed_sigma_seconds: 0.150
temporal_band_curriculum:
  start_sigma_seconds: 0.200
  end_sigma_seconds: 0.150
  start_update: 10000
  end_update: 50000
optim:
  run_until_update: 150000
```

The model's static sigma is deliberately the final 150-ms inference width;
training applies the curriculum at runtime. Experiment names and checkpoint
paths are distinct. The only additional training-budget change is automatic
stop at update 150k, matching the existing 150k comparisons. The inherited
200-epoch LR scheduler is retained, so shortening the stop limit does not
rescale the learning-rate schedule.

Other settings remain 4 GPUs, BF16, 3,600 frames/GPU, at most 32 samples/GPU,
LR 5e-5, 20k LR warmup, seed 666, CTC zero through 10k then linearly increasing
to 0.03 at 30k, numbered checkpoints every 50k and `model_last.pt` every 5k.

## Training and TensorBoard

From this project directory:

```bash
source env.sh
mkdir -p logs
bash scripts/start_speaker_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_cosine_band_window17to13_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_cosine_band_window17to13_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_cosine_band_window17to13_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default DDP port: 29641 (`TRAIN_PORT` override). Default TensorBoard port: 6025
(`TENSORBOARD_PORT` override). Verify that the ports are available before use.
The training entry rejects occupied GPUs under its inherited launch policy.
Use an empty experiment checkpoint directory for the initial run; matching
existing checkpoints resume their saved optimizer-update progress.

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_cosine_band_window17to13_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

TensorBoard event directory, relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_cosine_band_window17to13_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Check live total/flow/CTC loss scalars and the Gaussian sigma trace, which should
start at 200 ms, reach 175 ms at update 30k and remain 150 ms after 50k. Offset
remains 0 ms. Floating-point reductions may introduce small roundoff. Obtain
the actual forwarded TensorBoard URL from the application's Ports panel after
the service starts; the port number alone is not a forwarded URL.

## Inference

The copied speaker inference/evaluation shell entries now default to this
curriculum experiment's config, checkpoint path and output path, with
`CKPT_STEP=150000`. They retain the source Setting 1 protocol and all other
inference settings. The Python inference entry's default config also points
to this experiment. Inference uses fixed sigma=0.150 seconds and does not
replay the training curriculum.
This target-width inference policy also applies if an early checkpoint is
selected explicitly; it is not the checkpoint's intermediate training sigma.

```bash
CKPT_STEP=150000 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

Retain `speaker_training_contract.json` beside the checkpoints: sigma and its
curriculum are runtime/configuration state, not learned parameter tensors.
Configuration/contract validation prevents a mismatched fixed-width or
curriculum run from being resumed or evaluated under this identity.

All inherited `AVT_FIXED_BAND_*`, `EVALUATION_*` and earlier implementation
documents are historical source records. They do not report results for this
new curriculum experiment. Runtime verification logs remain under ignored
`logs/` and are not committed.

## Validation

The following suites passed before launch: 12 curriculum checks, 11 inherited
fixed-band contract checks, and 15 AVT attention checks including CUDA BF16
checkpointed backward. The curriculum checks cover update boundaries,
monotonicity, unchanged CTC, gradient accumulation, resumed update progress,
online/EMA sigma synchronization, actual Gaussian bias, unchanged RNG/state
keys, final-width inference identity, and resolved configuration equivalence.

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/test_temporal_band_curriculum.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u src/aligndit/script/misc/smoke_test_avt_fixed_band_contract.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u src/aligndit/script/misc/smoke_test_avt_fixed_band_av_va.py --cuda
```

The two inherited suites retain their historical fixed-window fixtures; the
new curriculum suite tests the actual new config and scheduled training hook.
All five changed/new shell entries pass `bash -n`. Backbone, flow model,
dataset, optimizer and base training-loop files are byte-identical to the
copied source.
