# AVT fixed Gaussian band: 17-frame soft core

This independent copy starts from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va`.
The only numerical configuration change is
`model.arch.temporal_band_fixed_sigma_seconds: 0.150 -> 0.200`.
Experiment names and output paths are distinct; runtime artifacts and trained
weights are not copied.

## Attention width

Audio and video features retain the aligned 40-Hz grids (25 ms per frame).
Sigma=200 ms equals 8 frames, so the nominal +/-1-sigma core comprises indices
-8..8 (17 samples). The source sigma=150 ms has a -6..6 core (13 samples).
Gaussian tails remain outside this range; there is no hard cutoff.

```text
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / 0.200)**2
B_va = B_av.transpose(-1, -2)
```

At displacement +/-8 frames, the new bias is -0.5. Its exp(-0.5) factor
multiplies the content-based attention score before joint normalization;
it is not the final attention weight. Only AV and VA receive the prior.
AA, VV and all text-related logits retain their source behavior.

The model and trainer implementation remain unchanged. The contract's
experiment-name marker and train/inference output routing identify the new
experiment. Historical constructor defaults remain unchanged; the new config
explicitly supplies sigma=0.200. No parameter, new loss or mask is introduced.

## Configuration and training

Use
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window17_av_va.yaml`.
It inherits the copied window13 config and overrides only sigma, model.name
and ckpts.save_dir. All other resolved settings are identical: 12 AVT joint
blocks, 6 audio-only blocks, 376,563,842 parameters, offset=0, bidirectional
AV/VA, speaker conditioning, datasets/caches, seed 666 and S2c 70k EMA
initialization. This is a fresh run from the same pretrained parent.

Training retains 4 GPUs, BF16, 3,600 frames/GPU, at most 32 samples/GPU,
LR 5e-5, 20k LR warmup, the 200-epoch scheduler and stop at update 200k.
CTC stays zero through 10k and ramps to 0.03 at 30k. Numbered checkpoints
are saved every 50k and `model_last.pt` every 5k.

From this project directory:

```bash
source env.sh
mkdir -p logs
bash scripts/start_speaker_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window17_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_fixed_band_window17_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_fixed_band_window17_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default DDP port: 29639 (`TRAIN_PORT` override). Use an empty checkpoint
directory for the initial run; the inherited trainer resumes matching existing
checkpoints. The dedicated save directory is:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window17_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Default TensorBoard port: 6023 (`TENSORBOARD_PORT` override). The startup
script serves the exact event directory:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window17_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Verify live loss/flow/CTC scalars, `temporal_band/sigma_ms_mean=200` and
`temporal_band/offset_ms_mean=0`, allowing floating-point reduction roundoff.
Obtain the current forwarded URL from the application's Ports panel.

## Inference and verification

The speaker inference/evaluation shell entries and Python inference default
use the new window17 config and checkpoint/output directories. Other inference
parameters are inherited. Keep `speaker_training_contract.json` beside the
weights: sigma is not a tensor state, so this contract prevents mixing
150-ms training weights and a 200-ms inference configuration.

The inherited attention/contract test suites are adapted to the 17-frame
case. They check +/-8-frame geometry, retained soft tails, VA transpose, joint
softmax, CFG, unchanged state, CUDA BF16 checkpointed backward, the three
resolved configuration differences, and rejection of the source width.

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_avt_fixed_band_av_va.py --cuda
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_avt_fixed_band_contract.py
```

Logs and verification reports live under `logs/` and are excluded from Git.
Other inherited documentation/evaluation reports describe prior experiments;
they are not results for this window17 run.
