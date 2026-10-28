# Strict AVT hard-window9: 200k EMA evaluation

Evaluate this experiment's completed `model_200000.pt`, retaining strict
bidirectional AV/VA radius 4 (nine frames). Training completed at update
200,000 on 2026-10-28; the numbered checkpoint and final checkpoint were
saved around 16:41 Asia/Shanghai and the training launcher exited.

Use the same protocol as the audited 150k evaluation: all 213 CelebV-Dub
Setting 1 test clips, EMA, seed 0, CFG text/video 5/2, Euler + EPSS,
sway -1, 32 NFE and real target duration. The prompt and CAM++ speaker
vector come from the same clip's full GT audio. Generated WAVs contain
only the target segment. Decoder, latent normalization and speaker cache
follow the unchanged experiment configuration.

Run from this project root on available physical GPU 1:

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=200000 EVAL_GPU=1 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_200000_hard_window9_cfgv2.0_20261028.log 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > logs/eval_s1_200000_hard_window9_cfgv2.0_20261028.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
```

The existing single-GPU pipeline generates WAVs, computes SPKSIM (WavLM),
WER (faster-whisper-large-v3) and EMOSIM (emotion2vec-plus-large), extracts
AV-HuBERT features from generated audio and corresponding mouth video,
then computes AVSync. No model or evaluator source changes are needed.

Checkpoint/output root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Output subdirectory: `eval_s1_200000_speaker_avt_joint_hard_window9_av_va_cfgv2.0`.

After successful pipeline completion:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_200000_speaker_avt_joint_hard_window9_av_va_cfgv2.0" \
  > logs/audit_s1_200000_hard_window9_20261028.log 2>&1
```

The audit requires exactly 213 WAVs, 213 generated AV-HuBERT features and
213 records in each metric file. It checks full clip identities, finite
values, WAV checksums/sample counts, feature shapes and recomputed metric
values. WER uses total word edit counts / total reference words. Runtime
artifacts remain outside Git.

## Results

Pending full pipeline completion and independent coverage/metric audit.
