# Isolated Semantic-VAE C2 + speaker + fixed AV+VA temporal band

This snapshot is a real source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_fixed_band` at
`6c69daa92a6c7d204e46ff00853a699c5a204d81`. The source AV-only fixed-band,
adaptive-band and speaker baseline snapshots are not modified. Logs, events,
checkpoints, outputs and caches were not copied. Existing pinned data, frozen
codec and speaker caches remain read-only inputs.

## Intervention

In the first 12 joint blocks, retain the fixed audio-query/video-key prior and
also bias video-query/audio-key logits with its exact transpose:

```text
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / 0.100)**2
B_va      = B_av.transpose(-1, -2)
joint bias = [[0_AA, B_av], [B_va, 0_VV]]
```

- Center remains the physical query time: offset **0 seconds** throughout.
- Width remains **sigma 100 ms**, shared across examples, heads, layers and
  updates. Sigma is a Gaussian standard deviation, not a hard cutoff or a
  100-ms total window.
- `temporal_band_bidirectional: True` is valid only for an enabled fixed band.
  Its default is `False`, preserving the inherited AV-only path. Disabling the
  temporal band with the default direction recovers the legacy no-band path.
- **Zero new trainable parameters or state tensors**: no predictor, frozen MLP,
  learnable buffers or new loss. FP32 distances use the existing aligned 40-Hz
  input grids; cache interpolation, duration rules, dropout and speaker injection
  are unchanged. The prior is also active in dropped-video CFG branches.
- AA and VV logits are untouched. Existing joint softmax is retained; the
  nonpositive priors can reduce cross-modal attention mass in both directions.
  This is not two separate softmaxes or a mass-preserving reweighting. No hard
  mask, teacher, auxiliary encoder or progressive window is added.
- Both training (including no padding-attention-mask mode) and inference use
  the same rule. Mathematical symmetry of the prior does not make the learned
  attention weights symmetric.

This is a directional control against the AV-only fixed-band experiment, not
a claim of measured AVSync or speech-quality improvement.

## Matched protocol and isolated launch

Initialize from the identical SHA-pinned S2c **70k EMA** parent with a new
optimizer and update counter. Do not resume a speaker, adaptive-band or AV-only
trained checkpoint. Retain seed 666 (rank RNG 666 + rank), 4 GPUs, bf16, 3,600
latent frames/GPU, LR 5e-5, 20k LR warmup, the original 200-epoch decay horizon,
CTC zero through 10k and linear to 0.03 at 30k, and stop at 200k updates. Retain
all 79,613 training records, including the 105 CTC-infeasible records handled by
the existing zero-infinity policy. Numbered checkpoints every 50k;
`model_last.pt` every 5k. Strict parent migration expects 313 source / 704 target /
303 loaded / 10 ignored / 401 new tensors, the same tensor schema as the speaker
baseline and AV-only control. The bidirectional prior adds no tensors.

Run commands from this snapshot's root. All runtime paths below are isolated
from the source experiment. Check all four GPUs are idle first: the inherited
wrapper uses physical GPUs 0,1,2,3 and refuses startup above 500 MiB on any one.
The new wrapper uses rendezvous port **29629**; verify it is free or explicitly
set `TRAIN_PORT` to another free port.

```bash
source env.sh
bash scripts/start_fixed_band_av_va_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_fixed_band_av_va_4x4090.sh \
  > logs/train_fixed_band_av_va.log 2>&1 < /dev/null &
train_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Use only the new `fixed_band_av_va` launcher/config for this experiment. The
copied named AV-only launcher now refuses startup and points to the new wrapper;
its YAML remains unchanged for regression tests. Other copied adaptive and older
named wrappers/configurations remain historical and may still name sibling
checkpoint directories. Do not launch them without explicit isolated
`model.name` and `ckpts.save_dir` overrides. The generic speaker launcher and
copied speaker inference/evaluation wrappers default to the new AV+VA experiment.

Config: `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va.yaml`.
It inherits the AV-only fixed configuration and changes only the experiment
name, checkpoint directory and bidirectional switch. Checkpoint directory under
`ROOT_PREFIX`:

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_fixed_band_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char`

TensorBoard serves this snapshot's `runs/`, port **6010** by default. Formal
run/event directory under `runs/`:

`AlignDiT_MMDiT_c2_svae_speaker_fixed_band_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`

Only rank 0 writes scalars: total/flow/CTC losses, LR, existing global/speaker
gradient diagnostics, and fixed offset/sigma mean/std/min/max in milliseconds.
These are local-batch diagnostics, not all-GPU averages. There is no predictor
gradient scalar. The training contract records fixed mode, both biased
directions and physical-time constants. Verify TensorBoard PID, listening port,
HTTP response and actual loss tags after launching; obtain the real forwarded
URL from the application's Ports panel. Starting the script alone does not
establish that loss curves are accessible.

## Verification before the formal run

CPU regressions (use the standard environment, with the correct local source):

```bash
source env.sh
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
PYTHONPATH=src "$python_bin" -u src/aligndit/script/misc/smoke_test_fixed_temporal_band_bidirectional.py
PYTHONPATH=src "$python_bin" -u src/aligndit/script/misc/smoke_test_fixed_temporal_band.py
PYTHONPATH=src "$python_bin" -u src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
PYTHONPATH=src "$python_bin" -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

Use a unique name/directory for each real-data smoke. The following performs
three updates at the unchanged 3,600-frame/GPU batch size and retains the
200-epoch scheduler horizon; only the stop/save intervals and output names
change. Do not run formal training concurrently. Start TensorBoard first unless
the already-verified service above is serving this snapshot's `runs/`.

```bash
source env.sh
smoke_name="fixed_band_av_va_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
smoke_dir="$PWD/ckpts/$smoke_name"
test ! -e "$smoke_dir"
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_fixed_band_av_va_4x4090.sh \
  "model.name=$smoke_name" "ckpts.save_dir=$smoke_dir" \
  optim.run_until_update=3 ckpts.save_per_updates=3 ckpts.last_per_updates=3 \
  > "logs/$smoke_name.log" 2>&1 < /dev/null &
smoke_pid=$!
printf 'smoke_name=%s smoke_dir=%s PID=%s\n' "$smoke_name" "$smoke_dir" "$smoke_pid"
ps -o pid,ppid,sid,tty,stat,cmd -p "$smoke_pid"
```

After all workers have exited successfully and `model_3.pt` exists, retain
`smoke_dir` and run the read-only strict EMA/online + real-data gradient check:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/validate_fixed_band_checkpoint.py \
  --checkpoint "$smoke_dir/model_3.pt" --step 3 \
  --config src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va.yaml \
  --output-json "$smoke_dir/real_gradient_validation.json"
```

The validator performs no optimizer update and checks CTC weights 0 and 0.03.
The smoke TensorBoard event directory is
`runs/${smoke_name}_semantic_vae_40hz_CelebVDub_char`.

### Pre-launch verification (2026-10-10)

- Inherited 25-test fixed-band, 19-test adaptive-band and speaker CPU suites
  passed in the new snapshot. These preserve the AV-only/baseline regressions.
- All 16 new AV+VA CPU regressions passed: independent full joint-softmax
  reference, unequal lengths, padding/head broadcasting, nonzero-offset
  transpose, AA/VV zero bias, CFG isolation, checkpointed gradients, unchanged
  parameter/state/RNG, EMA roundtrip and strict direction contracts.
- Seven negative training-config checks rejected mislabeled names/paths,
  disabled/adaptive bidirectional mode and nonboolean flags before any model,
  trainer or data-cache allocation.
- Four-GPU BF16 smoke `fixed_band_av_va_smoke_20261010` completed three updates
  at 3,600 frames/GPU. Strict S2c migration matched 313/704/303/10/401 tensors.
  Rank-0 total losses were 1.34679, 1.41300 and 1.36200. Numbered/last checkpoints
  and TensorBoard loss/flow/weighted-CTC scalars were written successfully.
- The saved checkpoint reloaded strictly for both EMA and online weights (704
  keys, zero band tensors). Real-data backward tests gave finite global gradient
  norms 1.10955 at CTC weight 0 and 1.26594 at weight 0.03; every MM block had
  bidirectional mode enabled and fixed offset/sigma stayed 0 s / 0.100 s.
- One Setting 1 test item completed EMA -> latent -> Semantic-VAE -> 16-kHz WAV
  at two NFEs. This is a functional smoke test, not a quality evaluation. Its
  waveform contains 17,462 finite samples. Reports/audio/checkpoints are under
  `ckpts/fixed_band_av_va_smoke_20261010/`, with text logs under `logs/`.

Passing these functionality checks does not establish any quality improvement.

## Inference and evaluation

The copied `infer_celebvdub_s1_svae_direct_speaker_ctc003.sh` and corresponding
`eval_...sh` select the new AV+VA config/checkpoint/output directory. Existing
Setting 1 same-clip reference, EMA, seed 0, CFG text/video 5/2 and 32 NFE are
retained. `CKPT_STEP`, `EVAL_GPU`, `CFG_VIDEO`, `CFG_TEXT`, `NFE` and `OUTPUT_DIR`
are supported as before; the full evaluation entry runs the four metrics.

Because fixed priors have no state keys, inference requires the matching
`speaker_training_contract.json` beside the checkpoint. Keep this file when
moving weights. Configuration and sidecar must agree on AV-only versus AV+VA
direction as well as fixed constants; identical tensor schemas alone cannot
prove experiment identity. Historical sidecars without the bidirectional field
represent the legacy AV-only rule, never the AV+VA rule.

`ADAPTIVE_TEMPORAL_BAND.md` and other inherited experiment documentation are
historical references, not launch instructions for this snapshot. Runtime
reports, training logs, events and generated audio remain ignored by Git.
