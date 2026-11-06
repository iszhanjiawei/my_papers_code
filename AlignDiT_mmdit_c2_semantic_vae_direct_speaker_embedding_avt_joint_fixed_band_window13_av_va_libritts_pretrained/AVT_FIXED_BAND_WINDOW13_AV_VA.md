# AVT fixed Gaussian band: 13-frame soft core

This experiment is an independent source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_av_va`.
Its only numerical configuration change is
`model.arch.temporal_band_fixed_sigma_seconds: 0.100 -> 0.150`.
Experiment names and output paths are distinct. Source checkpoints, logs and
TensorBoard events are not copied.

## Meaning of 13 frames

The existing aligned audio/video features are 40 Hz, so one frame is 25 ms.
The source sigma of 100 ms equals 4 frames; sigma of 150 ms equals 6 frames.
The nominal +/-1-sigma core therefore grows from indices -4..4 (9 samples)
to -6..6 (13 samples). This does not truncate attention outside that core.

```text
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / 0.150)**2
B_va = B_av.transpose(-1, -2)
```

At a displacement of 6 frames the new prior is -0.5, corresponding to a
Gaussian factor exp(-0.5). These factors multiply the content-based attention
scores before joint normalization; they are not the final attention weights.
AA, VV and all text-related logits retain their existing behavior.

All model and trainer implementation files are inherited unchanged, except
the experiment-name marker used by the training contract. No hard mask,
new parameter, extra loss or adaptive window is introduced. The generic
model constructor's 100-ms default remains for historical configurations;
the new training and inference configuration explicitly supplies 150 ms.

## Configuration and training

The new config is
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va.yaml`.
It inherits the source fixed-band config and overrides only sigma, model.name
and ckpts.save_dir. All other resolved values remain identical: 12 AVT joint
blocks plus 6 audio-only blocks, 376,563,842 parameters, offset 0, bidirectional
AV/VA, the same datasets/caches, speaker conditioning, seed 666 and S2c 70k
EMA initialization. This is a fresh comparison run using that same parent.

Training retains 4 GPUs, BF16, 3,600 frames per GPU, at most 32 samples per GPU,
LR 5e-5, 20k LR warmup, 200 scheduler epochs, and a stop at update 200k.
CTC stays zero through 10k and increases to 0.03 at 30k. Numbered checkpoints
are saved every 50k; `model_last.pt` is updated every 5k.

From this project directory:

```bash
source env.sh
mkdir -p logs
bash scripts/start_speaker_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_fixed_band_window13_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_fixed_band_window13_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default DDP port is 29638 (`TRAIN_PORT` override). Use an empty checkpoint
directory for the initial run; the inherited trainer automatically resumes
matching existing checkpoints. The save directory is:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

The TensorBoard startup script serves this exact event directory:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

The default TensorBoard port is 6022 (`TENSORBOARD_PORT` override). Verify
live loss/flow/CTC tags and `temporal_band/sigma_ms_mean=150`, with
`temporal_band/offset_ms_mean=0` (allow floating-point reduction roundoff).
Obtain the current forwarded URL from the application's Ports panel.
Runtime logs and verification reports remain outside Git.

## Inference

The copied speaker inference/evaluation shell entries and the Python inference
default select the new window13 config and checkpoint/output directories.
All other inference parameters retain their source values. Keep the generated
`speaker_training_contract.json` with the checkpoints: the Gaussian width
has no tensor state, and the existing contract check rejects mixing 100-ms
weights with a 150-ms inference config, or the reverse.

Inherited evaluation reports and the original fixed-band document describe
earlier experiments; they are not results for the window13 run.

## Verification

The 2026-11-01 independent audit confirms exactly three resolved config
differences (sigma, model.name, ckpts.save_dir) and identical model/trainer
implementation. The attention suite verifies the six-frame sigma, retained
Gaussian tails, exact VA transpose, joint softmax, CFG, unchanged state and
CUDA BF16 checkpointed backward. The contract suite checks configuration and
checkpoint identity, including rejecting the original 100-ms width.

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_avt_fixed_band_av_va.py --cuda
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_avt_fixed_band_contract.py
```

Machine-readable verification and live-training records are stored in `logs/`.
