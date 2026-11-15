# CelebV-Dub fixed13 LibriTTS + WavLM-Base+ REPA

This is an independent source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va_libritts_pretrained`.
Copy-only commit: `e3cd1ac8ccea07f43e6370731293541e0ea837bc` (262 files).
The source snapshot, GRID experiments and existing adaptive-band experiments are unchanged.
Inherited evaluation reports describe earlier experiments, not this new training run.

## Experiment

The configuration `src/aligndit/config/finetune_celebvdub_repa.yaml` inherits the
original CelebV-Dub recipe and adds the same REPA objective used in the fixed13
LibriTTS GRID REPA experiment and the adaptive-band CelebV-Dub REPA experiment:

- Frozen `microsoft/wavlm-base-plus`, revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, checkpoint SHA256
  `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.
- Layer-12 teacher representations from the complete unmasked training waveform,
  cached as FP16, 768 dimensions, approximately 50 Hz.
- Audio representation after the 10th AVT block (zero-based 9), with a new
  `768 -> 2048 -> 2048 -> 768` SiLU projection head.
- Interpolate each unpadded teacher sequence to its valid 40-Hz student length.
  Average `1 - cosine_similarity` over generated valid frames only; exclude
  prompt frames and padding. Teacher targets are detached.
- `total = diffusion + scheduled_CTC * CTC + 0.1 * REPA`.
  REPA starts at the first update, with no additional warmup.
- Strict checkpoint loading includes the projector, while inference does not
  execute the auxiliary head or load WavLM.

The following original settings are retained: 12 AVT joint blocks + 6 audio
blocks; fixed Gaussian AV and VA sigma 0.150 seconds (nominal 13-frame soft core);
CAM++ conditioning; vocabulary and Semantic-VAE normalization; all 79,613
CelebV-Dub train examples (including the 105 CTC-infeasible examples with the
existing `zero_infinity` behavior); four GPUs, BF16, 3,600 frames/GPU, max32,
workers16, seed666; LR5e-5, 20k LR warmup, 200-epoch LR horizon, stop at200k;
CTC0 through10k then linear to0.03 at30k; numbered checkpoints every50k and
`model_last.pt` every5k.

Initialize from the full LibriTTS 50k EMA:
`${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/libriSpeech_LibriTTS_pretrained_models/model_00050000.pt`.
SHA256: `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`.
All 802 existing tensors transfer, with five vocabulary tensors mapped by token
identity, including the correct CTC blank row. Only the six projector tensors
(7,344,896 parameters) are newly initialized. Optimizer and downstream update
start fresh at0; the existing policy carries the parent's EMA step50000.

Teacher cache:
`${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16`.
Its manifest, model identity and complete train coverage are validated. There
is no teacher forward pass or teacher optimizer in the training loop.

## Validation and launch

From this project directory, using the existing environment and explicit local
`PYTHONPATH` (do not reinstall the shared editable package):

```bash
source env.sh
export PYTHONPATH="$PWD/src"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$python_bin" -u scripts/test_celeb_repa_transfer.py --real-parent
"$python_bin" -u scripts/check_celeb_repa_training.py --audit-teacher
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u scripts/check_celeb_repa_gpu.py \
  --selection attention --output logs/celeb_repa_gpu_attention.json
CUDA_VISIBLE_DEVICES=1 "$python_bin" -u scripts/check_celeb_repa_gpu.py \
  --selection padded_frames --output logs/celeb_repa_gpu_padded_frames.json
```

The CPU suite checks unchanged inherited configuration/weights/sampling,
masked interpolation, teacher detach, gradients, rejected malformed state,
actual variable-length training data, and all808 online/EMA initialization
tensors with a fresh optimizer. GPU checks use real dynamic Celeb batches
selected for large joint-attention or padded-frame cost, at CTC0 and0.03,
including BF16 forward/backward and a reserve for EMA, Adam moments and DDP
buckets. These checks perform no optimizer updates and save no model weights.

Start once the checks pass, with an empty checkpoint directory for a fresh run:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_repa.sh \
  > logs/train_celeb_repa.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_celeb_repa.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default DDP port29648 (`TRAIN_PORT` can override). The entry validates its exact
configuration before launch and starts a detached TensorBoard service on a free
port from6030. `scripts/start_celeb_repa_tensorboard.sh` reuses only its verified
live service; `logs/tensorboard_celeb_repa.json` records PID, port and logdir.
Forward that port in the IDE to view it. The user's existing waiver of checking
the actual IDE-forwarded URL remains in effect; local HTTP and real scalar
records must still be verified.

Experiment model name:
`AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_ctc003_warmup10k30k`.

Checkpoint directory:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char`.

TensorBoard run under this project's `runs/`:
`AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`.
Only rank0 writes events: loss, diff_loss, repa_loss, weighted REPA, scheduled
CTC, global gradient norm and projector gradients. `ctc_loss` appears when CTC
becomes active; `ctc_lambda` and weighted CTC are recorded from the start.

The inherited speaker train aliases and speaker Setting1/Setting2 inference
defaults select this experiment. Historical configs remain for reproducibility.
Runtime receipts, logs, events and checkpoints are ignored by Git.

## Pre-launch verification result

All10 CPU tests passed, including real-parent loading. All79,613 teacher
arrays were read and validated (16,326,966 teacher frames, no missing or invalid
arrays). Both real BF16 GPU checks passed at CTC0 and0.03: a3-sample/3,539-frame
long-sequence batch and a32-sample/3,599-frame batch. Peak allocated memory was
16.16GiB and16.01GiB respectively, including the persistent training-state
reserve. Losses and gradients were finite, teacher gradients absent, and
poisoning padded frames did not change the REPA loss. Launch-time DDP health
and TensorBoard checks are recorded separately in
`logs/celeb_repa_training_run.json` after optimizer updates begin.
