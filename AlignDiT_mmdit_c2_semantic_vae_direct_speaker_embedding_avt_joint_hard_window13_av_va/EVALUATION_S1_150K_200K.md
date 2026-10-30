> Historical source-experiment record. For the current thirteen-frame run,
> see [AVT_HARD_WINDOW13_AV_VA.md](AVT_HARD_WINDOW13_AV_VA.md).

# AVT joint + fixed AV/VA band: 150k / 200k evaluation

Evaluate the numbered 150,000 and 200,000 checkpoints of this snapshot, using
their EMA weights and the fixed-band inference contract. The original
CelebV-Dub Setting 1 protocol uses all 213 test clips, each clip's full GT
reference prompt and CAM++ speaker vector, real target duration, seed 0,
CFG text/video 5/2, Euler with EPSS, sway -1 and 32 NFE. The generated WAV
contains only the target segment. This is the same-clip reference protocol.

The existing evaluation entry generates audio, computes SPKSIM (WavLM), WER
(faster-whisper-large-v3) and EMOSIM (emotion2vec_plus_large), then extracts
AV-HuBERT features from the generated audio and corresponding mouth video
and computes AVSync. Each checkpoint has a separate output directory.

Run from this project's root on two available GPUs:

```bash
source env.sh
mkdir -p logs
for checkpoint_step in 150000 200000; do
  eval_gpu=0
  if [[ "$checkpoint_step" == 200000 ]]; then eval_gpu=1; fi
  setsid env PYTHONUNBUFFERED=1 CKPT_STEP="$checkpoint_step" EVAL_GPU="$eval_gpu" \
    CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
    > "logs/eval_s1_${checkpoint_step}_fixed_band_cfgv2.0.log" 2>&1 < /dev/null &
  eval_pid=$!
  printf '%s\n' "$eval_pid" > "logs/eval_s1_${checkpoint_step}_fixed_band_cfgv2.0.pid"
  ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
done
```

After both pipelines finish, audit each output using
`PYTHONPATH=src ${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u scripts/audit_avt_joint_s1.py --output-dir <output>`.
The audit requires exactly 213 WAVs, 213 AV-HuBERT features and 213 records per
metric; checks full clip identities, audio checksums and duration receipts;
and recomputes the reported metrics. WER is total word edits divided by total
reference words, not an average of utterance WER. Output artifacts remain
outside Git, alongside the checkpoints.

## Results

Evaluation and coverage audit completed on 2026-10-26. Both checkpoints
complete all pipeline stages on their assigned GPUs and cover **213/213**
clips. Metric directions and five-decimal results:

| EMA checkpoint | Physical GPU | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| 150k | 0 | 0.64895 | 0.04668 | 0.74785 | 0.57954 |
| 200k | 1 | 0.65431 | 0.05172 | 0.75167 | 0.57856 |

150k has lower WER and slightly higher AVSync. 200k has higher SPKSIM and
EMOSIM. No checkpoint is best on all four metrics. These are descriptive
comparisons of the same test set, without a statistical significance claim.
WER is a fraction: 150k has 111 word errors / 2,378 reference words, and 200k
has 123 / 2,378. The WER percentages are 4.66779% and 5.17241%, respectively.

## Verification and artifacts

For each checkpoint, the audit verifies all 213 WAVs, 213 generated AV-HuBERT
features and 213 records in each of the four metric files. Every waveform is
16-kHz mono, finite, has the inference receipt's sample count and matches its
SHA256. Generated feature shapes match the corresponding GT features and all
values are finite. AVSync is recomputed from the feature arrays; the other
sample means and corpus WER are recomputed from per-clip records and agree
with the reported results. Summary lines and blank lines are excluded from
sample counts. Audit records use full test/video/clip identities, retaining
clips with repeated basenames in different videos.

Checkpoint and evaluation root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

| EMA checkpoint | SHA256 | Evaluation subdirectory |
|---|---|---|
| `model_150000.pt` | `2a8040e0f8735e0e18b4afd643189e25823e0b4c80ba3e03faeb082818128db6` | `eval_s1_150000_speaker_avt_joint_fixed_band_av_va_cfgv2.0` |
| `model_200000.pt` | `a90801f8c429f2a644ab508ef70fa807dc0277350c85acf7da0503bd2267ceb8` | `eval_s1_200000_speaker_avt_joint_fixed_band_av_va_cfgv2.0` |

Each output directory contains `test/` WAVs, `inference_summary.json`,
`inference_audio_audit.json`, `avhubert_feat/test/`, the four original
`_*_results.jsonl` files, `evaluation_per_clip.jsonl` with full clip IDs, and
`evaluation_audit.json` with exact recomputed metrics and coverage.

Run records and original logs are retained in this project's ignored `logs/`:

- `evaluation_s1_20261026_runs.json`
- `eval_s1_150000_fixed_band_cfgv2.0.log`
- `eval_s1_200000_fixed_band_cfgv2.0.log`
- `audit_s1_fixed_band_150k_200k_20261026.log`

The evaluation source and model configuration are unchanged in this
evaluation task. Only this reproducibility/results document is added to Git.
