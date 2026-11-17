# CelebV-Dub fixed13 LibriTTS + REPA: 150k EMA Setting 1

Evaluation completed on 2026-11-17 (Asia/Shanghai). The target is this isolated
`...window13_av_va_libritts_pretrained_repa_wavlm_base_plus` CelebV-Dub project.
The numbered 150k checkpoint was created at 2026-11-17 13:49:26 +08:00.
All **213/213** Setting 1 clips completed generation, all four metrics and audit.

| 150k EMA experiment | SPKSIM ↑ | WER ↓ (fraction) | WER ↓ (%) | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| Fixed13 LibriTTS, without REPA | 0.67540 | 0.03827 | 3.82675% | 0.78698 | 0.57453 |
| Fixed13 LibriTTS + WavLM-Base+ REPA | 0.67839 | 0.04205 | 4.20521% | 0.76326 | 0.54768 |

At this checkpoint, adding REPA did not reduce WER: word errors increased from
91 to 100 over the same 2,378 reference words, or +0.37847 percentage points.
SPKSIM increased by 0.00299, while EMOSIM decreased by 0.02372 and AVSync by
0.02685. These are descriptive results from one training run per configuration,
not a significance test or evidence of a general causal effect. The current
run continues toward 200k; its 200k checkpoint has not been evaluated here.

## Fixed evaluation protocol

- Full CelebV-Dub Setting 1 list (213 clips), same full GT clip as prompt and
  CAM++ speaker reference; historical prompt + two spaces + target text.
- EMA, seed0, Euler/EPSS, 32 NFE, sway=-1, CFG text/video=5/2, real duration.
- Original FP32 generation with TF32 high precision setting, no autocast;
  fixed13 Gaussian AV and VA sigma0.150s, unchanged 12 AVT + 6 audio blocks.
- Pinned Semantic-VAE1000k EMA decoder and original LibriSpeech-train latent
  normalization; output contains only the target, trimmed to original samples.
- SPKSIM: historical WavLM-Large + ECAPA weights. WER: original
  faster-whisper-large-v3 FP16, English beam5 and existing text normalization.
  EMOSIM: cosine of emotion2vec-plus-large classifier score vectors, preserving
  the original definition. AVSync: generated audio paired with real mouth video
  through AV-HuBERT, compared framewise to GT joint audio/video features.
- All11 metric-model/config assets were SHA256-checked against the previously
  pinned local evaluation assets; exact paths, hashes and versions are in
  `metric_assets.json`. No model or metric definition changed for this run.

The baseline is the original non-REPA LibriTTS fixed13 project's audited 150k
run from 2026-11-08. Its exact source/output identities and audit are recorded
in `protocol.json` and `result_comparison.json`; inherited older evaluation
reports elsewhere in this snapshot describe other experiments.

## Verification

`evaluation_audit.json` reports complete=true, count213, with213 WAVs,
213 generated AV-HuBERT features and213 records for each metric. All WAVs are
finite 16-kHz mono PCM16, match original target sample counts and their inference
SHA256 receipts. Feature arrays are finite and match GT shapes; all per-clip
AVSync values were recomputed from raw arrays. SPKSIM/EMOSIM/AVSync means and
corpusWER were recomputed and agree with the logs. Summary lines are excluded.
Full test/video/clip identities are retained, including repeated basenames.

Checkpoint identity was hashed independently before generation, matched to the
inference receipt, and its size/mtime remained unchanged. All tracked source
hashes from the launch snapshot were unchanged through evaluation completion.

## Checkpoint, outputs and execution

Checkpoint:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char/model_150000.pt
```

SHA256: `97a9338eb354526e4061b37bcc704fe52d1daecf39e6459414315ee6591ed920`.
Size:6,096,733,394 bytes. Selected weights:EMA, update150000.

Output:

```text
${ROOT_PREFIX}/zjw524/projects/data/evaluations/CelebVDub_avt_fixed13_libritts_repa_s1_150k_seed0_nfe32_cfg5_2
```

It contains `test/` WAVs, `inference_summary.json`, `inference_audio_audit.json`,
`avhubert_feat/test/`, four `_*_results.jsonl` files, `evaluation_per_clip.jsonl`,
`evaluation_audit.json`, `protocol.json`, `metric_assets.json`,
`launch_source_snapshot.json`, `result_comparison.json` and `evaluation_logs/`.

The detached launcher PID was685773. Generation shared GPU3 with training and
used a0.16 PyTorch allocator fraction; SPKSIM used GPU1, EMOSIM GPU2 and WER GPU3.
AV-HuBERT used GPU1 after SPKSIM released it, with the same allocator cap.
Generation took562.20 seconds. The full pipeline and audit completed at14:50;
four-GPU training continued normally (approximately154,260 updates at receipt).
The memory cap changes allocation limits only; generation precision and sampling
parameters remain those of the historical entry.

Launcher source commit: `2a39f4ab91613ecda6861d60732392e63afff889`.
Run receipt: `logs/eval_s1_150k_repa_20261117.json`.
Main log: `logs/eval_s1_150k_repa_20261117.log`.
Runtime artifacts are excluded from Git.

To reproduce in a fresh output directory from this project:

```bash
source env.sh
setsid env PYTHONUNBUFFERED=1 OUTPUT_DIR=/path/to/new/empty/output \
  bash src/aligndit/run/eval/eval_celebvdub_s1_repa_150k_shared.sh \
  > logs/eval_s1_150k_repa.log 2>&1 < /dev/null &
```

The script refuses existing inference output and fails on missing stages or an
incomplete audit. Reuse the verified existing artifacts to audit without rerunning
models using `scripts/audit_avt_joint_s1.py --output-dir <output>` with local
`PYTHONPATH=src` and the AlignDiT environment interpreter.
