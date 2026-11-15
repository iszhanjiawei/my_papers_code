# GRID fixed-13 AVT + LibriTTS + REPA: Setting 1 at 100k

Evaluated on 2026-11-16 at the user's request. This report belongs to the
independent REPA experiment in this directory. The inherited
`EVALUATION_GRID_150K_200K.md` describes the parent **without REPA**.

## Results

All **3,274** held-out GRID targets from **33 speakers** completed generation
and all four metrics. Independent verification reports **PASS**, requested /
succeeded = 3274 / 3274, failed = 0. There are **19,644 reference words**.
WER below is shown as a percentage; the other metrics are sample means.

| Model | GRID update | SPKSIM ↑ | WER (%) ↓ | EMOSIM ↑ | AVSync ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed13 AVT + LibriTTS + REPA | 100,000 | 0.81085 | 17.10955 | 0.86635 | 0.76894 |
| Parent fixed13 AVT + LibriTTS, without REPA | 150,000 | 0.80069 | 20.01120 | 0.85615 | 0.76416 |

The REPA 100k model has **3,361 / 19,644** word errors,
WER = 0.171095499898. Relative to the previously evaluated parent at
150k, WER is lower by **2.90165 percentage points**, and all three similarity
metrics are higher. These checkpoints have different training update counts;
this is a descriptive comparison. A matched 150k evaluation is still needed to
assess REPA at the same update count.

## Checkpoint and inference protocol

- GRID checkpoint: `/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char/model_100000.pt`.
- SHA256: `624b8f370a8e48060c382534d71dcc5a35065bcb10f516a2765ba4b6fe221984`.
- Actual checkpoint update is 100,000; initialized EMA is loaded strictly.
- Initialization remains the LibriTTS text-audio 50k EMA, SHA256
  `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`:
  all 802 parent tensors transferred, with six fresh REPA projector tensors.
- Same 12 AVT joint + 6 audio-only blocks, fixed 13-frame soft Gaussian AV/VA
  prior (sigma 0.15 seconds), vocabulary and LibriSpeech latent normalization.
- Setting 1 uses the same target clip's complete canonical audio as latent
  prompt and CAM++ reference; text is the historical duplicated transcript.
- EMA, FP32/high TF32, seed 0 per target, Euler/EPSS, 32 NFE, sway -1,
  text/video CFG 5/2, batch size 1. Save only the generated target continuation,
  at the original duration, as 16-kHz mono PCM16.
- SPKSIM: WavLM-Large/ECAPA against the same-clip canonical reference.
- WER: local faster-whisper Large-v3, FP16, English, beam 5; identical symmetric
  text normalization, summed word edits divided by total reference words.
- EMOSIM: emotion2vec classifier-score cosine against canonical target audio.
- AVSync: mean frame cosine of joint 25-Hz AV-HuBERT features from identical
  mouth video paired with generated versus canonical target audio.
- Full validation manifest SHA256:
  `c86309f5c5a0b9910608e74f54f82e71f0c30340b06b7c73dad6cb2a2b4a6230`.

## Execution and verification

The run used four free RTX 4090 GPUs on `bitahub-a21875852801077248925999` with
four independent single-utterance workers per GPU (16 shards). The original
training continued on its original node, with its 150k stop unchanged.
Training was at **112,421 updates** when observed at **2026-11-16T01:41:44.187147+08:00**.

Launch source revision: `8f33a19eacf9b25e22fcbcbd73e104c66bcc30fe`. All tracked snapshot files
were hashed into `launch_source_snapshot.json` before generation. The independent
verifier rehashed the checkpoint, model assets and executable/config sources;
checked all IDs, WAVs, cache/prompt receipts and four full metric JSONLs;
recomputed corpus WER and all metric means; and validated all 3,274 generated/GT
joint-feature pairs. No samples were omitted.

Outputs (excluded from Git):

```text
/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_repa_s1_100k_seed0_nfe32_cfg5_2
```

Key files: `evaluation_summary.json`, `verification.json`, `result_comparison.json`,
`protocol.json`, `metric_assets.json`, `launch_source_snapshot.json`,
`pipeline.log`, `verification.log`, full-ID metric JSONLs, WAV receipts and
`avhubert_joint/{gen,gt}/`.

Reproduce or resume from this project with the same immutable protocol and
16-shard layout:

```bash
source env.sh
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
CKPT_DIR="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char"
OUTPUT_DIR="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_repa_s1_100k_seed0_nfe32_cfg5_2"
mkdir -p logs
setsid env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  "$PYTHON" -u -m aligndit.script.eval.run_grid_s1 \
  --setting s1 --checkpoint "$CKPT_DIR/model_100000.pt" --step 100000 \
  --output-dir "$OUTPUT_DIR" --gpus 0,1,2,3 --workers-per-gpu 4 \
  > logs/evaluate_grid_s1_100k.log 2>&1 < /dev/null &
eval_pid=$!
wait "$eval_pid"
PYTHONPATH=src "$PYTHON" -u scripts/verify_grid_evaluation.py --output-dir "$OUTPUT_DIR"
```

Wait for `pipeline_status.json` to report `complete=true` before invoking the
independent verifier. Reuse only this run's output directory with the exact
same checkpoint, configuration, inference parameters and worker count.
