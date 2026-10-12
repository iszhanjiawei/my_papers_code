# GRID Setting2: real-dialogue dubbing, EMA 100k

The target text is **always the real GRID dialogue**. No lip-reading transcript
is used. This is Setting2 GTText dubbing, on the same 3,274 audited validation
targets as Setting1, with a different utterance from the **same speaker** as
reference. For each target, its own GT audio, latent and CAM++ vector are not
used by generation. Target video-only AV-HuBERT features and real target text
remain inputs; the independent reference provides both latent prompt and CAM++.

Reference assignments use only the complete validation partition. Within each
speaker, sort IDs by SHA256(`0:val/speaker/utterance`), then take the first cyclic
shift that makes every reference different in ID **and normalized transcript**.
This provides a deterministic seed-0 permutation: each validation clip is used
once as a reference for another clip. Freeze the 3,274 assignments and canonical
audio/latent/speaker hashes in `reference_pairs.json`; no selection depends on
model outputs or metric scores. Smoke runs use the same full assignment with
only their requested targets generated.

Reuse the verified reference caches: 64-D/40-Hz Semantic-VAE posterior samples,
CAM++ 192-D vectors, fixed LibriSpeech normalization and original vocabulary.
Prompt and target lengths can differ. Concatenate reference real dialogue,
two spaces, target real dialogue. Prompt video is zero, target video starts at
the independent prompt's end. Decode only target frames and crop to the target
video duration (`25 Hz frame count × 640 samples`) as 16-kHz mono PCM16.

Keep Setting1's EMA 100k, FP32, seed 0 per target, Euler/EPSS, 32 NFE, sway −1,
text/video CFG 5/2, and one utterance per process. Run four processes per GPU
on two RTX 4090s, eight independent workers total. If other workloads share the
GPUs, Whisper shards wait for at least 8,192 MiB of free memory and serialize
model allocation. Two model slots per GPU remain held until each ASR shard
finishes; all eight shards are retained without changing ASR parameters.

Metrics retain the existing implementations and Setting1 text normalization:

- SPKSIM compares generated speech with the **independent reference actually
  used by generation**, using WavLM-Large/ECAPA. Its reference differs from
  Setting1's same-clip GT, so the SPKSIM difference reflects both this metric
  pairing and the generation protocol.
- WER uses real target dialogue and local faster-whisper Large-v3 ASR; normalize
  both texts (NFKC, case, punctuation, whitespace, digits), preserve GRID letters
  including `a`, and aggregate word errors over all **19,644** reference words.
- EMOSIM compares emotion2vec classifier-score vectors with target canonical GT.
- AVSync compares joint 25-Hz AV-HuBERT features from the same target mouth video
  with generated and target GT audio. The video-only training cache is not the
  joint-feature reference.

All four result sets must cover every target ID; WAV receipts bind output hashes
to checkpoint, pair assignment, configuration and sampling. Partial smoke results
are explicitly marked and cannot be reported as full validation results.

```bash
OUT="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2"
mkdir -p "$OUT"
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_s2_100k.sh \
  --output-dir "$OUT" --gpus 0,1 --workers-per-gpu 4 \
  > "$OUT/pipeline.log" 2>&1 &
```

`pipeline_status.json`, worker logs and `active_workers.json` record progress.
Reuse the same output, pairing and worker count to resume verified artifacts;
`--start-stage sim/wer/emosim/avsync/merge` starts after verified generation.
The final `evaluation_summary.json`, four full-ID JSONL files, generated WAVs,
reference assignments and joint-feature caches stay outside Git.

## Full validation results, 2026-10-12

All generation and metric stages completed on **3,274/3,274** targets. The
checkpoint is EMA `model_100000.pt`, SHA256
`6394f7e7682f5d096c29a8769d3158f44949746fa84f0c1a66a09ea1dad37da7`.
Reference assignments have SHA256
`eb092eeda9294fa06a21d893c1a04ce09e20bf3d2bb0fdd99558ad0e5acbd99c`.

| Setting | Coverage | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| Setting1, same-clip prompt | 3274/3274 | 0.81041 | 0.12935 | 0.87838 | 0.78267 |
| Setting2, independent same-speaker prompt, GTText | 3274/3274 | 0.66391 | 0.47180 | 0.81068 | 0.70023 |

Setting2 corpus WER is **9268 / 19644 = 0.4717980045**, or **47.17980%**:
S=7820, D=819, I=629. All target and ASR texts use the normalization above;
no words are corrected, no utterances are dropped, and no per-shard WER means
are averaged. Setting1 WER is 2541 / 19644 (12.93525%). These are descriptive
results under different prompt protocols. SPKSIM also changes its metric
reference from same-clip GT to the independent prompt; its difference cannot
be attributed solely to generation. EMOSIM and AVSync retain target GT as
their comparison reference in both settings.

Independent verification passed with **3,274 successes and zero failures**.
It rechecked all WAV formats, durations and hashes, reference permutations and
cached reference hashes, prompt/target geometry, four full-ID result sets and
their aggregates. WER was recomputed with separate dynamic programming and
jiwer. All 6,548 joint features passed shape, FP32, finite-value, hash and
audio/mouth/model binding checks; independent NumPy AVSync recomputation had
maximum absolute difference `1.320251749e-7`. Eleven metric assets and the
EMA checkpoint were rehashed. The evaluation processes have finished.

Artifacts:

- [Full four-metric summary](/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2/evaluation_summary.json).
- [Independent verification](/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2/verification.json).
- `sim_results.jsonl`, `wer_results.jsonl`, `emosim_results.jsonl` and
  `avsync_results.jsonl` each contain 3,274 full IDs in validation-list order.
- `reference_pairs.json`, `protocol.json`, `metric_assets.json`, WAV receipts,
  `avhubert_joint/{gen,gt}/` and logs bind pairs, configuration and model assets.
  `metric_assets.json` preserves the initial source revision and the subsequent
  WER resume revision `3f18e82`, which changes GPU allocation only.

## WER diagnosis, 2026-10-12

The strongest actionable finding is sensitivity to the reference-plus-target
text input. In a fixed, output-independent subset of 132 targets (four per
speaker), retaining the actual reference latent, reference CAM++, target video,
P+T time axis, seed and CFG, but supplying **only the target text**, reduced WER
from 378/792 (47.72727%) to 122/792 (15.40404%). This is an exploratory input
intervention, not a replacement for the full 3,274-target Setting2 result above.
Removing the reference transcript also changes text length and token positions;
the experiment does not isolate semantic copying from text alignment effects.

### Full-set observations and ASR control

S1 and S2 use the same 3,274 targets, and train/validation contain the same 33
speaker identities. S2 is worse for all 33 speakers and for 2,640 individual
targets; 521 tie and 113 improve. Exact ASR sentences fall from 1,888 (57.67%)
to 233 (7.12%). Substitutions/deletions/insertions change from 2116/334/91 to
7820/819/629. Substitutions account for 84.79% of the 6,727 added edits.
Mean hypothesis length remains approximately six words (5.926 versus 5.942).

The identical frozen faster-whisper Large-v3/FP16/en/beam-5 evaluator was run
on all canonical GT recordings, with the same normalization:

| Targets | Count | GT WER | S1 WER | S2 WER |
|---|---:|---:|---:|---:|
| Full validation | 3274 | 11.80513% | 12.93525% | 47.17980% |
| GT recognized with zero errors | 2015 | 0.00000% | 5.03722% | 42.35732% |

GT has 2,319 edits (S=1956, D=299, I=64), including `bin→been` 315 times,
`red→read` 223, `place→play` 167 and `green→screen` 144. Absolute WER therefore
includes recognizer/transcription effects, but these do not explain the large
S2 degradation. GT WER must not be subtracted as a fixed additive noise floor.
The GT-correct subset is a selected diagnostic subset, not a new test set.

Reference-content association is substantial. With zero/one/two/three/four
matching target/reference word slots, S2 WER is respectively
54.04/48.40/41.97/35.14/24.24%; S1 is 12.57/12.76/13.55/13.49/11.74%.
Among 7,786 erroneous slots where target and reference differ, 1,010 (12.97%)
exactly match the actual reference word. Shuffling reference sentences within
speaker 2,000 times gives a mean 3.00% (central 95% permutation range
2.65–3.37%). This controls for frequent GRID vocabulary matches, but is an
exploratory association rather than a causal contribution estimate.

For `val/s34/sbbk4s`, target is `set blue by k four soon`, reference is
`place red in m zero again`, S1 ASR is the target, and S2 ASR is
`place red in m four soon`. Full exact reference copying occurs in only
3/3,274 cases; the widespread pattern is local mixing, not whole-sentence
repetition. Word-slot attribution uses edit alignment and excludes insertions.

### Fixed-subset interventions

Selection: within each speaker, the first four IDs ordered by
SHA256(`werdiag20261012:fullID`), independent of outputs/scores. Every row below
uses the same 132 targets and 792 target words. The diffusion rows retain EMA
100k, FP32, 32 NFE and seed 0. Other formal parameters remain fixed except for
the stated intervention.

| Condition | Word edits / 792 | WER | Exact sentences / 132 |
|---|---:|---:|---:|
| GT recording, ASR control | 101 | 12.75253% | 82 |
| GT latent directly decoded, VAE control | 107 | 13.51010% | 75 |
| Historical S1 on these targets | 105 | 13.25758% | 71 |
| Original S2, regenerated concurrently | 378 | 47.72727% | 8 |
| S2 with target text only | 122 | 15.40404% | 67 |
| S2 with normalized reference latent set to zero | 354 | 44.69697% | 15 |
| S2 with CFG text 1, video 2 | 188 | 23.73737% | 43 |
| Target-only T-frame sequence, no latent prompt, reference CAM++ retained | 122 | 15.40404% | 65 |

Target-text-only improves 104 targets, ties 25 and worsens three, with lower
aggregate WER for all 33 speakers in this subset. Among the 648 word slots
where target and reference differ, exact reference-word errors fall from 43
in the concurrent S2 baseline to zero with target-only text; latent-zeroing
leaves 23 and CFG=1 leaves 14. This supports reference-transcript interference
more specifically than a generic reduction in word errors.

The target-text-only intervention preserves the 6-second P+T geometry and
actual acoustic reference, so neither the doubled time axis nor the presence
of a reference waveform alone is sufficient to explain the high baseline.
Latent-zeroing gives only a small improvement under the original concatenated
text condition. Normalized zero is a mean latent, not a real silent recording
or ordinary no-prompt condition, so this does not establish acoustic prompts
as irrelevant. The separate T-frame condition changes several factors,
including target noise indices despite the same seed, and is a feasibility
probe rather than a single-factor mechanism test. Low VAE reconstruction WER
does not rule out decoder sensitivity to out-of-distribution generated latents.

Initial four-WAV reproduction checks failed a strict waveform tolerance:
maximum differences were 9–22 PCM16 units and RMS differences 0.75–1.41 units.
Source, input and model identities matched; the numerical source remains
unresolved. These failures are preserved in `ablation_archive_v1/`. Consequently
all 132 original S2 WAVs were regenerated in the same diagnostic run. Their
normalized ASR hypotheses and per-utterance edit counts match the old baseline
for **132/132** targets, and all intervention comparisons use this concurrent
baseline. No formal waveform, transcript or reported score was overwritten.

### Mechanistic interpretation and next checks

GRID fine-tuning takes one utterance's latent, video, text and speaker vector,
then masks 70–100% of its audio for inpainting. It does not construct independent
same-speaker reference/target sentence pairs. Its audio axis is 119/120 frames
and text is 20–31 characters; S2 uses 239/240 frames and 42–61 characters.
S1 also doubles the axis, but both prompt and target encode the same content.
Its low WER does not measure the same cross-utterance selection ability.

The text cross-attention in `dit_vt_mm.py` exposes all non-padding reference and
target characters to each target audio query. There is no explicit text-segment
mask. The `prompt_isolated_ca` option only suppresses CA updates at prompt audio
positions; enabling it alone would not hide reference text from target queries.
Together with the interventions and reference-word associations, this supports
a mismatch in the current concatenation/conditioning design. It does not prove
that a specific attention head or training component is the sole cause.

The actual CFG rule in `cfm_vt.py` is
`pred + cfg_video*(pred-tts) + cfg_text*(tts-null)`. The TTS branch retains audio,
text and speaker; the null branch drops all of them. Therefore `cfg_text=5`
amplifies a joint conditioning direction rather than target text alone. The
CFG=1 intervention supports a practical contribution of guidance strength,
without showing which condition inside that direction is responsible.

Verified output duration, crop boundaries, video offset and text-length limits
show no broad geometry fault. All S1/S2 WAVs match their GT sample counts;
mean RMS differs by only 0.40 dB, low-energy frame fraction is 4.11% versus
4.01%, and there are no empty ASR outputs. These checks cannot exclude every
word-level timing or phonetic error.

The next evaluation candidate is target-only text with the existing independent
reference audio and CAM++, followed by all four metrics on a fixed full set.
Training independent reference/target pairs or separating reference and target
text attention are hypotheses for subsequent experiments. The 132-target WER
improvement alone does not establish speaker, emotion or AVSync preservation,
and does not justify changing the formal 47.17980% result.

Diagnostic artifacts are under
`/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2/diagnostics_wer/`:
`reference_interference.{py,json}`, `distribution_audio.{py,json}`,
`gt_asr_analysis.{py,json}`, `asr_gt/`, `ablation_manifest.json`,
`ablation_contract.json`, `ablations/`, `asr_ablations/` and the independent
`ablation_analysis.{py,json}`. These are runtime artifacts outside Git.
The independent ablation audit passed all 792 WAV/receipt/ASR bindings,
528 source files, feature normalization/geometry hashes, fixed selection,
reference assignments and independent edit-count/summary recomputations.
