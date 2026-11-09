# Window13 Gaussian AV/VA band: 150k EMA evaluation

Evaluation and full coverage audit completed on 2026-11-02. This report
belongs to the window13 experiment (fixed sigma 0.150 seconds, nominal
+/-6 frames at 40 Hz, Gaussian tails retained).

## Results

Both rows use 150k EMA checkpoints and the same 213-clip CelebV-Dub Setting 1
protocol. The 9-frame row is the previously completed source experiment,
read from its original `evaluation_audit.json`; it was not rerun in this task.

| Gaussian soft core | Sigma | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| Source 9 frames | 100 ms | 0.64895 | 0.04668 | 0.74785 | 0.57954 |
| **Current 13 frames** | **150 ms** | **0.65271** | **0.05088** | **0.75905** | **0.58568** |

The 13-frame run has higher SPKSIM, EMOSIM and AVSync, with worse WER.
Its corpus WER is **121 word errors / 2,378 reference words = 5.08831%**;
the source 9-frame run has 111 / 2,378 = 4.66779%. These are descriptive
comparisons of independently trained runs, without a significance claim.

## Protocol and checkpoint identity

- Full 213-clip test list; same-clip full GT reference prompt and CAM++ speaker
  conditioning; real target duration; generated WAV contains the target only.
- EMA weights, seed 0, CFG text/video 5/2, Euler with EPSS, sway -1, 32 NFE.
- The source comparison matches generation settings, decoder metadata,
  normalization, speaker metadata, test-list hash and latent-cache root.
- Inference uses this snapshot's
  `finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va.yaml`
  and its matching fixed-band training contract, with sigma 0.150 seconds.

Checkpoint root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Checkpoint: `model_150000.pt` (EMA, update 150000).

SHA256:
`6ddbee1cb390546bdb7542d3d0865de68298dfe672fb3433a5a051e02c129fdd`.

Output subdirectory:
`eval_s1_150000_speaker_avt_joint_fixed_band_window13_av_va_cfgv2.0`.

The source 9-frame results come from the corresponding subdirectory
`eval_s1_150000_speaker_avt_joint_fixed_band_av_va_cfgv2.0` under the source
fixed-band checkpoint root. Its checkpoint SHA256 is
`2a8040e0f8735e0e18b4afd643189e25823e0b4c80ba3e03faeb082818128db6`.

## Verification

The pipeline completed all stages on physical GPU 0. The audit checks:

- Exactly 213 WAVs and 213 generated AV-HuBERT features matching full test-list
  clip identities, including clips sharing a basename in different videos.
- Every WAV is finite, nonempty, 16-kHz mono and matches its inference receipt's
  sample count and SHA256.
- Generated feature shapes match GT features; all feature values are finite.
- Each of the four metric files has 213 ordered sample records. Summary lines
  are excluded from sample counts.
- SPKSIM/EMOSIM/AVSync are sample means; AVSync is also recomputed from feature
  arrays. WER is recomputed from summed word edit counts, rather than averaged
  sentence WER. Recomputed values agree with the metric log summaries.

The output directory contains `test/`, `inference_summary.json`,
`avhubert_feat/test/`, `_sim_results.jsonl`, `_wer_results.jsonl`,
`_emosim_results.jsonl`, `_avsync_results.jsonl`,
`evaluation_per_clip.jsonl` and `evaluation_audit.json`.

## Reproduction and run records

The existing evaluation entry was run from this project's root with:

```bash
source env.sh
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=0 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_window13_cfgv2.0_20261102.log 2>&1 < /dev/null &
```

For a new evaluation, set `OUTPUT_DIR` to a fresh directory; the entry refuses
to overwrite existing generated audio. To audit an existing completed output:

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/audit_avt_joint_s1.py --output-dir <completed-output-directory>
```

Pipeline PID was 327051; audit PID was 357210. Both completed. Original logs
and run metadata are retained in this snapshot's ignored `logs/` directory:

- `eval_s1_150000_window13_cfgv2.0_20261102.log`
- `audit_s1_150000_window13_20261102.log`
- `evaluation_s1_150k_window13_20261102_runs.json`

Model, configuration and evaluation implementation are unchanged in this
task. Only this results/reproducibility document is added to Git. Runtime
artifacts and checkpoint files are excluded from the commit.
