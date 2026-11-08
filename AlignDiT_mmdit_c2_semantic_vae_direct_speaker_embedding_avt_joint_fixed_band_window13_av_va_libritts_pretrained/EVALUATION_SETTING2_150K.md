# Original 115-pair Setting2 GT-dialogue evaluation

This entry evaluates this LibriTTS-pretrained fixed13 AVT snapshot with the
user-designated `model_150000.pt`, strict EMA and matching fixed-band training
sidecar. The original 115 targets and 83 different-utterance references are from
`my_papers_code/Setting2-Dubbing-test`; the 66/85 outcome-selected subsets are not
used. Reference identities are automatically screened, not human verified.

Generation receives video-only target features, supplied true target dialogue,
and the explicitly paired reference waveform's Semantic-VAE prompt and CAM++
embedding. Text retains the historical reference transcript + separator + target
dialogue concatenation. No target acoustic input, VSR or REPA teacher is read.

Sampling: seed 0, FP32, strict EMA update 150000, Euler/EPSS 32 NFE, sway -1,
text/video CFG 5/2. Target length is 25-Hz video frames times 640 samples; video
features are linearly interpolated to 40 Hz (`align_corners=False`), reference
visual frames are zero placeholders, and only target audio is decoded/cropped.
Use this experiment's resolved LibriTTS configuration, including AVT joint
attention and fixed bidirectional Gaussian AV/VA sigma 150 ms. All other
attention entries retain their inherited behavior.

The historical independent-reference cache is reused read-only. Preflight checks
all input hashes against the dataset inventory and verifies exact reference
audio hashes, SVAE 1000k EMA identity, source commit, deterministic per-reference
posterior seeds, latent/speaker hashes, tensor shapes, finite values and CAM++
identity against training configuration. Training-only latent normalization and
the pinned 1000k decoder are unchanged. Model outputs are newly generated.

From this project root:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 EVAL_GPU=3 EVAL_MEMORY_FRACTION=0.22 \
  bash src/aligndit/run/eval/eval_setting2_dubbing_150k.sh \
  > logs/eval_setting2_libritts150k_20261108.log 2>&1 < /dev/null &
```

GPU and memory-fraction settings may be overridden; they do not alter model
precision, text or sampler parameters. The allocator limit caps only this
evaluation process, without changing existing training processes. The initial
one-item smoke test selected the longest combined reference/target duration and
successfully loaded the strict checkpoint and decoder and generated its WAV.

Default output, beneath this experiment's checkpoint directory:
`eval_s2_150000_gttext_original115_cfgtext5_cfgvideo2`.
Set `OUTPUT_DIR` for a separate rerun. Partial generation resumes only when the
configuration, source hashes, input receipts and existing WAV hashes match.
Completed metrics are kept for final hash/coverage audit; stale inputs fail it.

The pipeline reuses the common benchmark `evaluate_setting2.py` without edits:

- SPKSIM: WavLM-large/ECAPA cosine against the explicit cross-utterance reference.
- WER: faster-whisper-large-v3 English beam 5 and historical normalization;
  total word edits / 1721 normalized reference words, reported as percent.
- EMOSIM: emotion2vec_plus_large classifier-score cosine against target GT.
- AVSync: generated joint AV-HuBERT features versus target GT joint features,
  frame cosine averaged within each sample, then sample mean. This is not LSE.

Each stage uses the explicitly selected GPU. AV-HuBERT preprocessing follows the
existing baseline mouth-video extractor and fixed checkpoint. `audit_setting2.py`
checks all 115 WAVs, original target/reference bindings, prompt conditions,
input/source/weight hashes, full metric coverage and independent word edit
distance plus frame-level AVSync recomputation before writing
`evaluation_audit.json` with `complete=true`.

Input geometry regression checks:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 /zjw524/ENTER/envs/aligndit/bin/python \
  scripts/test_setting2_inputs.py
```

Run logs, audio, checkpoints, caches, metric outputs and smoke artifacts remain
outside Git. This document records protocol/reproduction; final scores are in
the audited output directory.
