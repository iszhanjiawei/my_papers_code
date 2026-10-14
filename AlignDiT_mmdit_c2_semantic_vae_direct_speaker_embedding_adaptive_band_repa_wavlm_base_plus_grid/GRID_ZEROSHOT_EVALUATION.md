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
