# GRID fixed-13 LibriTTS: EMA 150k and 200k evaluation

Evaluate this snapshot's GRID-trained `model_150000.pt` and `model_200000.pt`,
using the complete held-out GRID validation split: 3,274 targets and 19,644
reference words. Preserve this project's AVT joint backbone, fixed sigma=0.15 s
AV/VA attention prior and its checkpoint-side training contract.

## Protocol

All six evaluations use initialized EMA, FP32 with high TF32 matmul precision,
seed 0 per target, Euler/EPSS, 32 NFE, sway=-1, text/video CFG=5/2, batch size 1,
the original tokenizer, LibriSpeech latent normalization and pinned Semantic-VAE
decoder. Save only the generated target continuation, cropped to its original
video duration, as 16-kHz mono PCM16.

1. Setting 1: same-clip canonical audio latent prompt and CAM++ vector;
   duplicated reference/target transcript, matching the reference GRID project.
2. Setting 2 reference-target: a deterministic independent same-speaker,
   different-text validation reference provides latent prompt and CAM++;
   concatenate its transcript and target transcript with the historical separator.
3. Setting 2 target-only: retain the same independent prompt, speaker vector,
   target video and prompt+target time axis; supply only the target transcript.

The user requested both Setting 2 text variants. Keep separate output directories
and freeze the same output-independent reference pairing for all four S2 runs.
Generation never consumes the target waveform, latent or CAM++ in Setting 2.
All target dialogue is supplied ground truth, without lip-reading transcription.

Four metrics follow the reference implementation:

- SPKSIM: WavLM-Large/ECAPA cosine against the prompt audio actually used
  (same-clip in S1, independent reference in S2).
- WER: frozen local faster-whisper Large-v3, FP16, English, beam 5. Normalize
  target and ASR text symmetrically with NFKC/case/punctuation/whitespace/digit
  rules, preserving GRID letter words. Sum word edits over 19,644 target words.
- EMOSIM: emotion2vec classifier-score cosine against target canonical GT.
- AVSync: mean frame cosine of joint 25-Hz AV-HuBERT features from the same
  mouth video paired with generated versus canonical target audio.

S1 and S2 SPKSIM use different references, so their differences also reflect
the metric pairing. The two S2 variants share identical reference assignments.

## Reproduce

From this project, run the batch controller in a separate session:

```bash
mkdir -p logs
setsid env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  "${ROOT_PREFIX:-}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/evaluate_grid_150k_200k.py \
  --gpus 0,1,2,3 --workers-per-gpu 4 \
  > logs/evaluate_grid_150k_200k.log 2>&1 < /dev/null &
```

The two weights run concurrently, each on two GPUs with eight independent
single-utterance workers. Both S1 pipelines and their independent verification
must finish before S2 reference-target; both of those must finish before S2
target-only. Inference, speaker similarity, ASR, emotion and joint-feature
extraction use these GPUs. ASR retains the reference evaluator's model slots
and available-memory guard. No training is started or resumed.

The checkpoint root is the GRID experiment directory recorded in
`GRID_TRAINING.md`. Actual checkpoint SHA256 values:

- 150k: `df297235d281c9185b41429a15c3d020338f9194580994e6e2af025649215424`.
- 200k: `0fd9c84ea1695ee9dbdb4c3c21bf614dfaa21f80c1faedef627c3a1cb8980fb3`.

Default outputs are under `${ROOT_PREFIX}/zjw524/projects/data/evaluations/`:

```text
GRID_avt_fixed13_libritts_s1_{150k,200k}_seed0_nfe32_cfg5_2
GRID_avt_fixed13_libritts_s2_reference_target_{150k,200k}_seed0_nfe32_cfg5_2
GRID_avt_fixed13_libritts_s2_target_only_{150k,200k}_seed0_nfe32_cfg5_2
```

Each directory contains pipeline/worker logs, `protocol.json`, WAV receipts,
four full-ID metric JSONLs, joint feature caches, `evaluation_summary.json`
and independent `verification.json`. The batch status is
`GRID_avt_fixed13_libritts_evaluation_batch.json` in the output root.
Runtime artifacts and weights are not Git content.

Resume with the same command and worker count. Completed groups are reaudited;
incomplete pipelines verify and reuse existing WAVs/shards. Never mix checkpoint,
sampling parameters or S2 text modes in an output directory.

## Results: 2026-11-14

Each row covers all 3,274 held-out validation targets from 33 speakers, with
19,644 reference words. These are EMA weights at 150,000 and 200,000 GRID
updates. Before GRID training, both were initialized from the same LibriTTS
text-audio 50k checkpoint; the parent updates are not part of the GRID counts.
The architecture remains 12 AVT joint blocks plus 6 audio-only blocks, with the
fixed 13-frame soft Gaussian AV/VA prior. No training configuration was changed
for this evaluation.

WER below is a fraction, not a percentage. Higher SPKSIM, EMOSIM and AVSync
are better; lower WER is better. Values are rounded only for display.

| Protocol | GRID checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Setting 1, same-clip prompt | 150k | 0.80069 | 0.20011 | 0.85615 | 0.76416 |
| Setting 1, same-clip prompt | 200k | 0.79334 | 0.21793 | 0.84971 | 0.75789 |
| Setting 2, reference + target text | 150k | 0.67085 | 0.34861 | 0.80329 | 0.72696 |
| Setting 2, reference + target text | 200k | 0.66627 | 0.35970 | 0.80089 | 0.72308 |
| Setting 2, target text only | 150k | 0.68309 | 0.21452 | 0.82811 | 0.75629 |
| Setting 2, target text only | 200k | 0.68033 | 0.22205 | 0.82623 | 0.75313 |

Corpus WER is recomputed from the complete per-utterance edit counts:

| Protocol | GRID checkpoint | Word edits / reference words | WER (%) |
| --- | --- | ---: | ---: |
| Setting 1 | 150k | 3,931 / 19,644 | 20.01120 |
| Setting 1 | 200k | 4,281 / 19,644 | 21.79291 |
| Setting 2, reference + target text | 150k | 6,848 / 19,644 | 34.86052 |
| Setting 2, reference + target text | 200k | 7,066 / 19,644 | 35.97027 |
| Setting 2, target text only | 150k | 4,214 / 19,644 | 21.45184 |
| Setting 2, target text only | 200k | 4,362 / 19,644 | 22.20525 |

Within each of these three protocols, 150k scores better than 200k on all four
metrics. Setting 1 WER is lower by 1.78171 percentage points at 150k. Within
Setting 2, target-only text improves all four metrics at both checkpoints,
with WER lower by 13.40867 and 13.76502 percentage points, respectively. This
comparison keeps the independent reference audio, latent prompt, CAM++, target
video, sampling settings and reference assignments fixed. It describes these
evaluations and does not by itself establish a cause for the differences.

Setting 1 uses the target clip itself as the reference prompt. Both Setting 2
variants instead use the same independent same-speaker, different-text
reference. In particular, the S1/S2 SPKSIM scores have different reference
pairings and should be interpreted with that distinction.

The full validation manifest SHA256 is
`c86309f5c5a0b9910608e74f54f82e71f0c30340b06b7c73dad6cb2a2b4a6230`.
All four Setting 2 runs use reference-pair SHA256
`eb092eeda9294fa06a21d893c1a04ce09e20bf3d2bb0fdd99558ad0e5acbd99c`.
All six `verification.json` receipts report `PASS`, `complete=true`,
`requested=succeeded=3274`, and `failed=0`. The independent audit checks full
sample IDs, generated WAVs and prompt receipts, all four metric JSONLs,
corpus WER, all 3,274 generated/GT joint-feature pairs, actual checkpoint
hash/update/EMA metadata, decoder receipts and the captured source/configuration.
The batch controller also reports `phase=complete` and `complete=true`.
Both Setting 1 runs completed and passed verification before Setting 2 began.
The four GPUs were used for generation and metrics; each checkpoint used two
GPUs with eight independent workers.

Final machine-readable results are in
`${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_comparison.json`.
It links each output directory, exact metric values, WER counts and the hashes
of its summary, protocol and independent verification receipts. Generated WAVs,
per-sample metric JSONLs, joint features and logs remain in the six output
directories listed above and are excluded from Git.
