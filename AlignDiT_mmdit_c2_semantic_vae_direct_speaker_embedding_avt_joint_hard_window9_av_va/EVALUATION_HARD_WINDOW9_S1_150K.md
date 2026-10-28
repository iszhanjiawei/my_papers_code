# Strict AVT hard-window9: 150k EMA evaluation

Target: this experiment's `model_150000.pt`, with the original strict AV/VA
radius 4 (nine frames) training/inference contract. Evaluate all 213
CelebV-Dub Setting 1 test clips with EMA, seed 0, CFG text/video 5/2,
Euler + EPSS, sway -1, 32 NFE and real target duration. The prompt and CAM++
speaker vector come from the same clip's full GT audio, following the
existing same-clip reference protocol. Generated WAVs contain only the
target segment.

Inference and the four metrics reuse the existing single-GPU entry. Physical
GPU 1 shares with the ongoing four-GPU training; inference/evaluation run
with one worker and keep the original model settings. Monitor GPU memory,
training and evaluation progress until completion.

Run from this project root:

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=1 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_hard_window9_cfgv2.0_20261028.log 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > logs/eval_s1_150000_hard_window9_cfgv2.0_20261028.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
```

The pipeline generates WAVs, computes SPKSIM (WavLM), WER
(faster-whisper-large-v3) and EMOSIM (emotion2vec-plus-large), extracts
AV-HuBERT features from generated audio plus the matching mouth video,
then computes AVSync. The decoder, latent normalization, speaker cache and
test protocol follow the existing experiment configuration.

Checkpoint/output root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Output subdirectory: `eval_s1_150000_speaker_avt_joint_hard_window9_av_va_cfgv2.0`.

After the pipeline finishes, audit using the standard interpreter and local
source snapshot:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_avt_joint_hard_window9_av_va_cfgv2.0" \
  > logs/audit_s1_150000_hard_window9_20261028.log 2>&1
```

The audit requires exactly 213 WAVs/features and records in each metric,
checks complete clip identities, finite values, WAV hashes and sample counts,
and recomputes the four reported metrics. WER uses total word edit counts
divided by total reference words. Runtime artifacts stay outside Git.

## Results

Pending pipeline completion and full-coverage audit.
