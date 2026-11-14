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
  /zjw524/ENTER/envs/aligndit/bin/python -u scripts/evaluate_grid_150k_200k.py \
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
