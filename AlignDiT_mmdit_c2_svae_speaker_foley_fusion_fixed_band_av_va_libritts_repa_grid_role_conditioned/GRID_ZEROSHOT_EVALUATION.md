# GRID targets with frozen CelebV-Dub references

This evaluation uses this snapshot's GRID Semantic-VAE/CAM++ C2 model and its
own EMA 100,000 and 150,000 checkpoints. It does not run, replace or import the
five model implementations/results shipped in `zero-shot-Grid/`.

## Frozen pairing and target selection

The authority for reference selection is
`zero-shot-Grid/GRID目标_CelebVDub参考_zero-shot复现说明.md`. Restore its frozen
tables directly: each GRID speaker has one fixed CelebV-Dub `test` reference,
with 33 distinct source video groups. Do not resample references. The full
3,281-row portable table must reproduce SHA256
`9f533801de330c4a1911ffbc7c56147d6e9638a760fa61166478b18aafe4de3d`.
All 33 source WAVs and prepared TXTs are checked against the document's hashes
and audio headers before extracting features. Original reference files are
preserved. Cross-dataset/source-video separation is established; the document
does not provide biometric identity deduplication or pretraining identity audits.

This model retains its existing audited **3,274-target** GRID validation set,
all of which belong to the frozen 3,281-row table. Preserve the frozen order
and `global_index` values, including gaps. The seven previously excluded IDs
are `s6/pbab6s`, `s29/srws4p`, `s28/sgwt7p`, `s13/brbl4n`, `s30/srak9n`,
`s23/sgag7a` and `s4/lgil4a`. The first has an invalid stream count; the others
failed MPEG decoding. Their original source-audit reasons are saved in
`shared/plan.json`. Do not invent replacement videos/features or describe this
as exact 3,281-target coverage. Both checkpoints use exactly the same 3,274
targets and reference caches.

## Current-model reference and inference inputs

Reference preprocessing uses the complete frozen WAV. Average channels to mono,
resample to 16 kHz with the existing torchaudio frontend, and write a separate
PCM16 canonical cache without RMS normalization or silence trimming. This
includes the one stereo reference. Both features come from this same canonical
reference audio:

- Semantic-VAE posterior samples, raw FP32 `[P,64]`, 40 Hz, right-zero-padded to
  a 400-sample multiple. Keep the training posterior logvar clamp `[-12,12]`.
  Use base seed 666 and the existing SHA256 formula with the full
  `celebvdub/test/video/clip` reference ID.
- CAM++ 192-D embedding from the full unmasked reference, using the existing
  circular 10-second padding, arithmetic mean and L2 normalization.

These are this model's feature types. The GE2E d-vectors and Tacotron mels used
by some of the five external implementations are not its inputs.

Only normalized **target** text is passed to `model.sample`; reference text is
retained solely as pairing provenance. The target's original cached six-word
dialogue equals the document's raw target text after its digit/silence cleanup
for every selected target. Raw target text remains separately available for
WER scoring. No target audio, latent or speaker vector is used as generation
conditioning; original target video duration supplies the requested time axis.

Keep the actual reference latent prompt and reference CAM++ on a P+T 40-Hz
grid. Prefix the video with P zero frames, then insert the target video features
at offset P. Pass `lens=P`, generate P+T frames, and decode only `[P:P+T]` with
the frozen Semantic-VAE 1000k decoder. Crop to exact original target samples
and save 16 kHz mono PCM16. Reference lengths can differ from target lengths.

Sampling matches the established current-model protocol: EMA weights, FP32,
high TF32 precision, per-target seed **0**, Euler/EPSS, 32 NFE, sway=-1 and
CFG text/video=5/2. Reference posterior seed 666 is separate from generation
seed 0; do not substitute another model's `666+global_index` generation rule.
Checkpoint update and all EMA state tensors are strictly validated.

## Metrics and three WER variants

One frozen faster-whisper Large-v3/FP16/English/beam-5 ASR result per generated
WAV supplies the same raw hypothesis to all three WER calculations. Preserve
the document's raw target truth, including digit strings and all `sp`/`sil`
tokens. Do not score the already cleaned generation text in their place.

| Field | Text processing on truth and hypothesis |
|---|---|
| `wer_original` | Historical Chinese/ASCII punctuation deletion, lowercasing, whitespace words; keep digits and silence labels |
| `wer_digits_only` | Supplied GRID regex cleanup, per-digit expansion including letter-digit tokens; keep silence labels |
| `wer_digits_silence` / main `wer` | Same digit expansion, then remove standalone `sp`/`sil` tokens; no waveform trimming |

All WERs are corpus `(S+D+I)/(H+S+D)`, with independently recomputed edit
distances. Never average sentence or shard WERs. Digit expansion is per
character (`12` becomes `one two`), not English integer reading. The 3,274
selected raw target strings retain 249 examples containing 276 silence tokens.
The main denominator is 19,644 target words; each other denominator is derived
from its actual normalized truth. Exact punctuation deletion follows the
supplied zero-shot evaluator, and differs from the earlier local S2 Unicode
punctuation-as-space protocol. Scores from the two WER protocols must be labeled.

The canonical-GRID GT audio control on these same 3,274 targets is
8608/19920 = **0.43213** (original), 2598/19920 = **0.13042** (digits only),
and 2322/19644 = **0.11820** (digits/silence). This reuses the prior verified
Large-v3/FP16/en/beam-5 raw GT ASR transcripts after checking all source audio
hashes, ASR checkpoint hashes and package versions, and re-scores them against
the document's original raw truth. It performs no new GT ASR inference and
does not subtract GT error from generation error. Provenance and all per-target
counts are in the run root's `score_gt_wer.py`, `gt_wer_three_protocols.json`
and `gt_wer_three_protocols_results.jsonl`.

SPKSIM_GT (`sim`) is WavLM/ECAPA cosine versus the GRID target GT, while
SPKSIM_REF (`sim_ref`) compares with the actual frozen CelebV-Dub reference.
For the latter, average original reference channels before 16-kHz resampling;
do not select one stereo channel. SPKSIM_REF is the direct external-voice
transfer diagnostic. EMOSIM remains emotion2vec classification-score cosine
versus target GRID GT. AVSync remains mean-frame cosine of joint AV-HuBERT
features: the same target mouth video paired with generated versus GT audio.
It is not SyncNet LSE. The non-WER metrics use equally weighted sample means.

## Execution and verification

From this experiment directory, use the existing environment and snapshot:

```bash
source env.sh
TASK_ROOT="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_CelebVDub_zeroshot_target_only_100k_150k_20261015"
mkdir -p "$TASK_ROOT"
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_zeroshot_100k_150k.sh \
  --root "$TASK_ROOT" --smoke > "$TASK_ROOT/smoke_master.log" 2>&1 < /dev/null &
# After both 33-reference smoke evaluations pass:
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_zeroshot_100k_150k.sh \
  --root "$TASK_ROOT" > "$TASK_ROOT/full_master.log" 2>&1 < /dev/null &
```

The full entry starts EMA 100k on physical GPU 0 and EMA 150k on physical GPU 1,
each with four independent shards. Each checkpoint runs inference, WAV binding
verification, asset audit, SPKSIM_GT/REF, WER, EMOSIM, joint feature extraction
and AVSync, merge, and a separate independent verification process. Whisper
retains the existing two-model-per-GPU allocation guard; the remaining ASR
shards wait rather than changing precision or evaluator settings.

`smoke_100000/` and `smoke_150000/` each select one target per external reference,
33 targets per model, and remain distinct from full `ema_100000/` and
`ema_150000/` outputs. Shared data under `shared/` include the full frozen
document portable table, source exclusions, current-model manifest and all
33 extracted reference caches. Verified immutable target-GT joint features
may be hard-linked from the previous local evaluation; the worker still
validates every audio/mouth/model binding and content hash.

Each output contains exact-ID WAVs, receipts, protocol and decoder contracts,
source snapshots, model-asset hashes, shard logs/results, four merged JSONLs,
three-protocol WER counts, `evaluation_summary.json` and `verification.json`.
The independent verifier does not import this evaluator. It restores the
original document with the independently supplied rebuilder, checks reference
audio/TXT hashes and channel conversion, latent/CAM++ shapes and seeds, all
target/output/reference bindings and durations, complete ID coverage, every
WER variant and corpus count, paired joint features with a NumPy cosine
recomputation, model/checkpoint hashes and unchanged launch source files.

For a failed stage, retain verified artifacts and resume the corresponding
checkpoint with `run_grid_zeroshot --start-stage <stage>` and the same shared
directory, output directory, checkpoint, update, physical GPU and shard count.
Do not switch checkpoints or smoke/full selection in an existing directory.

Runtime artifacts are outside Git. Pre-existing unrelated working-tree edits
remain separate; their initial evaluation/test diff is preserved in the run
root's `preexisting_worktree.patch`, and source snapshots identify exact loaded
files in addition to the Git base revision.

The first smoke audit exposed a verifier-only assumption that source references
were all 16 kHz. The frozen `--aqjaJyZLk/0_5` reference is 44.1-kHz stereo;
the inference frontend had already averaged/resampled it correctly. The
independent verifier was corrected to use each frozen source rate and check
the resampled channel mean against the canonical PCM16 audio. Original failure
logs/status are preserved. Smoke re-audits explicitly record this verifier-only
source change, while retaining strict hashes for all inference and metric
sources. Full runs require every source hash to remain unchanged.

## Full results, 2026-10-15

Both current-model checkpoints completed all 3,274 targets, with zero failures
and strict independent verification. Every non-checkpoint protocol field is
identical between the two runs, including external reference cache hashes,
target identities/text, generation geometry, seed and CFG. All values below
are sample means or corpus WER fractions, displayed to five decimal places.

| GRID EMA | Targets | SPKSIM_GT | SPKSIM_REF ↑ | Original WER ↓ | Digits-only WER ↓ | Digits/silence WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 3274 | 0.48523 | 0.13241 | 0.48981 | 0.20567 | 0.19512 | 0.60627 | 0.70233 |
| 150k | 3274 | 0.51334 | 0.13604 | 0.48218 | 0.19644 | 0.18565 | 0.65329 | 0.70901 |

The main WER is **3833/19644 = 19.51232%** at 100k and
**3647/19644 = 18.56547%** at 150k: **0.94685 percentage points** lower,
or **4.85260% relative reduction**. The 150k checkpoint improves 692 targets,
ties 1,996 and worsens 586 on main edit counts; 23 speaker/reference groups
improve and ten worsen. Exact main-WER sentences increase from 1,347 to 1,420.
These are paired descriptive comparisons, not a significance test or uniform
per-target improvement.

| EMA | WER normalization | S | D | I | Edits / reference words |
|---|---|---:|---:|---:|---:|
| 100k | Original | 5736 | 3608 | 413 | 9757 / 19920 |
| 100k | Digits only | 2911 | 717 | 469 | 4097 / 19920 |
| 100k | Digits/silence | 2891 | 457 | 485 | 3833 / 19644 |
| 150k | Original | 5631 | 3570 | 404 | 9605 / 19920 |
| 150k | Digits only | 2752 | 677 | 484 | 3913 / 19920 |
| 150k | Digits/silence | 2736 | 414 | 497 | 3647 / 19644 |

The original 48–49% WER values and the roughly 19% main values come from the
same generated WAVs and ASR hypotheses. Digit normalization has a large effect
on this score; it does not change model output. The matching GT-audio controls
are 0.43213 / 0.13042 / 0.11820. These are separate references for interpreting
the evaluator, not quantities to subtract from generation error.

All reported aggregate point estimates favor 150k. SPKSIM_REF increases by
only 0.00363, versus 0.02810 for SPKSIM_GT; the improvement in target content
does not establish strong external voice cloning. Preserve the distinction
between target-identity similarity and external-reference similarity.
The WER normalizer and seven-target exclusion difference also prevent treating
the older local S2 score or the document's five-model table as an identical
controlled comparison to this result.

Independent audits verified all 6,548 generated WAVs across both checkpoints,
the frozen 33-reference assignments and frontend, 13,096 logical joint feature
files/bindings (GT reused), complete metric coverage, raw target truth and all
three normalized corpus counts. NumPy AVSync recomputation differs by at most
1.4883993449e-7. Each audit rehashed 11 metric model assets and its generator
checkpoint and checked all launch/metric source hashes. Both full-run reports
have `source_hashes_unchanged=true`, no verifier-only source exceptions, and
`requested=succeeded=3274, failed=0, complete=true`. Their smoke raw hypotheses,
main normalized hypotheses and per-target main edit counts match the full run
for **33/33 targets per checkpoint**. No bitwise WAV equality is asserted.
The coordinator and workers have exited, and their GPU memory is released.

Checkpoint hashes:

- `model_100000.pt`: `6394f7e7682f5d096c29a8769d3158f44949746fa84f0c1a66a09ea1dad37da7`.
- `model_150000.pt`: `2230f55c1347d6f91d9b3ca7401de4de6060fa4d6f417d13ba0cb576b8531242`.

The evaluation entry, tests and protocol were committed as
`d76b8ee8989d5a565bd1d1527ea493da2aad1a9a` before the full launch. Exact
pre-existing working-tree source versions remain identified by the preserved
patch and launch hashes; those pre-existing edits were not mixed into the
evaluation commit.

Runtime root:
`/zjw524/projects/data/evaluations/GRID_CelebVDub_zeroshot_target_only_100k_150k_20261015/`.

- `ema_{100000,150000}/evaluation_summary.json`: all metric values and corpus counts.
- `ema_{100000,150000}/verification.json`: full strict independent verification.
- `checkpoint_comparison.json` and `compare_checkpoints.py`: matched-protocol
  checks, per-target/per-group WER changes and smoke replication.
- `listening_pairs.tsv`: one fixed target per reference group, linking target
  GT, original CelebV-Dub reference and both checkpoint outputs for listening.
- `gt_wer_three_protocols.json`, `gt_wer_three_protocols_results.jsonl`, shared
  reference caches, WAVs, receipts, joint features, source snapshots and all logs
  remain runtime artifacts outside Git.
