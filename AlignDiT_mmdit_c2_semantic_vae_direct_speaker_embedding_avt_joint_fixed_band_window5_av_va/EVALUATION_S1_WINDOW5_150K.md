# Window5 Gaussian AV/VA band: 150k EMA evaluation

Evaluation and complete coverage audit finished on 2026-11-02. This report
belongs to the five-frame Gaussian soft-band experiment: fixed sigma 0.050
seconds, nominal +/-2 aligned 40-Hz frames, with Gaussian tails retained.
It does not describe a hard-window experiment.

## Results and matched comparisons

All rows use 150k EMA weights and the same 213-clip CelebV-Dub Setting 1
protocol. Window5 was evaluated in this task. Window9 and window13 reuse
their previously completed, audited evaluations.

| Gaussian soft core | Sigma | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| **Current 5 frames** | **50 ms** | **0.64788** | **0.04794** | **0.77171** | **0.56360** |
| Source 9 frames | 100 ms | 0.64895 | 0.04668 | 0.74785 | 0.57954 |
| 13 frames | 150 ms | 0.65271 | 0.05088 | 0.75905 | 0.58568 |

Window5 has the highest EMOSIM of these three runs, but lower SPKSIM and
AVSync than window9 and window13. Window9 has the lowest WER. Window13 has
the highest SPKSIM and AVSync. These are descriptive comparisons of individual
trained runs, without a statistical significance claim.

Window5 corpus WER is **114 word errors / 2,378 reference words = 4.79394%**.
Window9 has 111 errors (4.66779%); window13 has 121 errors (5.08831%).

## Protocol and identity

- Full 213-clip test list; same-clip full GT reference prompt and CAM++ speaker
  conditioning; real target duration; output WAV includes only the target.
- EMA, seed 0, CFG text/video 5/2, Euler with EPSS, sway -1, 32 NFE.
- Both comparison runs match generation settings, test-list hash, decoder
  metadata, normalization, speaker metadata and latent-cache root.
- Configuration:
  `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window5_av_va.yaml`.
  The fixed-band training contract verifies sigma 0.050 seconds at inference.

Checkpoint root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window5_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Checkpoint: `model_150000.pt`, update 150000, EMA.

SHA256:
`4df40ed177cedc45d66471f26ab7e4aa43660b771a5aaee8fcdc68e184526831`.

Evaluation subdirectory:
`eval_s1_150000_speaker_avt_joint_fixed_band_window5_av_va_cfgv2.0`.

## Audit and artifacts

The complete pipeline ran on physical GPU 0 and passed all checks:

- Exactly 213 WAVs, 213 generated AV-HuBERT features and 213 records in each
  of the four metric files, checked against full test-list clip identities.
- All audio is nonempty, finite, 16-kHz mono and matches the inference receipt's
  sample count and SHA256. Duplicate clip basenames across videos remain distinct.
- Generated feature shapes match GT features, and all values are finite.
- Sample means and corpus WER are recomputed and match the reported results.
  AVSync is additionally recomputed from feature arrays. WER uses total word
  edits divided by total reference words, rather than mean sentence WER.

Artifacts under the evaluation subdirectory are `test/` WAVs,
`inference_summary.json`, `avhubert_feat/test/`, `_sim_results.jsonl`,
`_wer_results.jsonl`, `_emosim_results.jsonl`, `_avsync_results.jsonl`,
`evaluation_per_clip.jsonl` and `evaluation_audit.json`.

Inference took about 392 seconds. Pipeline PID was 577064 and audit PID was
596893; both completed successfully. Run metadata and original logs remain
in this snapshot's ignored `logs/`:

- `evaluation_s1_150k_window5_20261102_runs.json`
- `eval_s1_150000_window5_cfgv2.0_20261102.log`
- `audit_s1_150000_window5_20261102.log`

## Reproduction

Actual first-run invocation from this project directory:

```bash
source env.sh
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=0 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_window5_cfgv2.0_20261102.log 2>&1 < /dev/null &
```

For a rerun, set `OUTPUT_DIR` to a fresh directory. The existing entry refuses
to overwrite generated audio. Audit a completed directory with:

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/audit_avt_joint_s1.py --output-dir <completed-output-directory>
```

The window9 and window13 comparison artifacts are in their corresponding
fixed-band checkpoint roots; the run metadata records their exact audit paths.
Model/configuration/evaluation implementation is unchanged in this task.
Only this report is added to Git; checkpoints, logs and generated artifacts
are excluded.
