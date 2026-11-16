# GRID fixed-13 AVT + LibriTTS + REPA: Setting 1 at 150k

Evaluated on 2026-11-16 at the user's request. This report belongs to the
independent REPA experiment in this directory. The inherited
`EVALUATION_GRID_150K_200K.md` describes the parent **without REPA**.

## Results

All **3,274** held-out GRID targets from **33 speakers** completed generation
and all four metrics. Independent verification reports **PASS**, requested /
succeeded = 3274 / 3274, failed = 0.
There are **19,644 reference words**. WER is shown as a percentage; the other
metrics are sample means.

| Model | GRID update | SPKSIM ↑ | WER (%) ↓ | EMOSIM ↑ | AVSync ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed13 AVT + LibriTTS + REPA, current | 150,000 | 0.80355 | 18.78436 | 0.85630 | 0.76404 |
| Parent fixed13 AVT + LibriTTS, without REPA | 150,000 | 0.80069 | 20.01120 | 0.85615 | 0.76416 |
| Fixed13 AVT + LibriTTS + REPA, earlier checkpoint | 100,000 | 0.81085 | 17.10955 | 0.86635 | 0.76894 |

The REPA 150k model has **3,690 / 19,644** word errors,
WER = 0.187843616371. Relative to the parent at the same 150k
update count, WER is lower by **1.22684 percentage points**
(**6.13076% relative reduction**). SPKSIM and EMOSIM are higher; AVSync
is slightly lower. These evaluations share the same validation protocol and
GRID update count.

The same REPA run's earlier 100k checkpoint has WER **17.10955%**,
which is **1.67481 percentage points lower** than its 150k checkpoint.
The 100k checkpoint also scores higher on the three similarity metrics.
For these evaluated checkpoints, extending GRID training from 100k to 150k
did not improve the four Setting 1 metrics.

## Checkpoint and inference protocol

- GRID checkpoint: `/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char/model_150000.pt`.
- SHA256: `cdb0d3e232f987be1a611613e6827bfde46f6be9228fe87b6ad01f2dccfe574c`.
- The actual checkpoint update is 150,000. The initialized EMA has 808 model
  tensors, including all six REPA projector tensors; loading is strict.
- EMA step is 200,000: the inherited LibriTTS EMA step 50,000 plus 150,000 GRID
  updates. GRID optimization itself starts at update zero.
- The numbered 150k checkpoint was saved at **2026-11-16 10:22:48 +08:00**.
  Training then stopped at the requested 150k limit.
- Initialization remains the LibriTTS text-audio 50k EMA, SHA256
  `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`:
  all 802 parent tensors transferred, with only six new REPA projector tensors.
- Same 12 AVT joint + 6 audio-only blocks, fixed 13-frame soft Gaussian AV/VA
  prior (sigma 0.15 seconds), vocabulary and LibriSpeech latent normalization.
- REPA training uses WavLM-Base+ layer-12 cached targets, student block 10,
  a 768→2048→2048→768 projector and weight 0.1. The auxiliary head is unused
  during inference; its weights still load strictly.
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

## Execution and independent verification

The run used four free RTX 4090 GPUs on `bitahub-a21858530364485632258746` with
four independent single-utterance workers per GPU (16 shards). Only Setting 1
was requested and run. The inference contract is identical to the original
non-REPA 150k and this REPA experiment's earlier 100k evaluation.

Launch source revision: `08adc4091471fbc3fad45306071e261a8a51ad22`. Executable/configuration
source is unchanged from the training implementation `27b6ab1`.
The tracked snapshot files are hashed in `launch_source_snapshot.json`.

The independent verifier rehashed the actual checkpoint, all 11 metric assets,
251 executable/configuration source files and the decoder. It checked every
full sample ID, generated WAV, duration and source-cache/prompt binding;
recomputed word edits and all four metric means; and checked all 3,274
generated/GT joint-feature pairs. The largest FP64 AVSync recomputation
difference was below 1.85e-7. Word errors comprise **3,052 substitutions,
419 deletions and 219 insertions**, totaling 3,690.

The pipeline finished at **2026-11-16 12:43:57 +0800**; independent verification
passed at **2026-11-16 12:47:24 +08:00**. Original experiment source was not
changed for this evaluation.

Runtime outputs are excluded from Git:

```text
/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_repa_s1_150k_seed0_nfe32_cfg5_2
```

Key files: `evaluation_summary.json`, `verification.json`, `result_comparison.json`,
`protocol.json`, `metric_assets.json`, `launch_source_snapshot.json`,
`pipeline.log`, `verification.log`, four full-ID metric JSONLs, WAV receipts
and `avhubert_joint/{gen,gt}/`.

Reproduce or resume with the same immutable protocol and 16-shard layout:

```bash
source env.sh
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
CKPT_DIR="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char"
OUTPUT_DIR="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_repa_s1_150k_seed0_nfe32_cfg5_2"
mkdir -p logs
setsid env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  "$PYTHON" -u -m aligndit.script.eval.run_grid_s1 \
  --setting s1 --checkpoint "$CKPT_DIR/model_150000.pt" --step 150000 \
  --output-dir "$OUTPUT_DIR" --gpus 0,1,2,3 --workers-per-gpu 4 \
  > logs/evaluate_grid_s1_150k.log 2>&1 < /dev/null &
eval_pid=$!
wait "$eval_pid"
PYTHONPATH=src "$PYTHON" -u scripts/verify_grid_evaluation.py --output-dir "$OUTPUT_DIR"
```

Before a new run, capture the selected project's tracked source/configuration
files and Git revision in `launch_source_snapshot.json`, preserving its layout
from the existing receipt. The verifier requires this source binding.
Wait for `pipeline_status.json` to report `complete=true` before invoking the
independent verifier. Reuse only this run's output directory with the exact
same checkpoint, configuration, inference parameters and worker count.
