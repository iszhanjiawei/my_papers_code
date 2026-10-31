# AVT hard-window9 + WavLM timestep-adaptive representation alignment

## Scope and provenance

Independent code copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va` (254 tracked files, byte-identical before edits). Neither that snapshot nor the WavLM donor is modified. Runtime copy hashes are in `logs/source_copy_manifest.json`.

The WavLM cache/dataset/projector/loss branch comes from `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus`. Only the REPA branch is ported: its adaptive attention band is not part of this experiment. TRA adapts the noise-dependent alignment strength in [TARO, section 4.3 and equation (7)](<../../papers/TARO- Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis/auto/TARO- Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis.md>).

This is TARO-inspired WavLM alignment for speech dubbing, not a full TARO reproduction or a claim of independently inventing TRA. TARO's EAT teacher, onset-aware conditioning, convolutional projector, coefficient 0.5, dataset and sampler are not adopted. The donor's WavLM teacher, alignment layer, MLP, interpolation, and coefficient 0.1 are retained to isolate timestep weighting.

## Objective and time direction

The existing flow path is `x_t = (1-t)*noise + t*clean`, where `t=0` is noise and `t=1` is clean audio. TARO uses clean coefficient alpha and noise coefficient sigma:

```
w(t) = sigmoid(log(alpha/(sigma+eps)))
     = alpha/(alpha+sigma+eps)
alpha=t, sigma=1-t -> w(t)=t/(1+eps), eps=1e-8
```

The stable rational form handles both endpoints without log(0). It is the coefficient ratio from TARO, not squared-SNR weighting. The same random timestep used to construct each CFM input determines that sample's weight; this is not a schedule over optimizer updates.

For valid generated-frame set G_b in sample b, with M=sum_b |G_b|:

```
L_TRA = sum_b w(t_b) * sum_{i in G_b} [1-cos(projector(h_bi), stopgrad(y_bi))] / M
L_total = L_CFM + lambda_CTC(update)*L_CTC + 0.1*L_TRA
```

Weights are applied before reduction, never normalized by their sum. The donor's frame-weighted denominator is unchanged. Prompt and padded frames never contribute. Cosine loss is evaluated in float32 and minimizes distance (`1-cos`), so gradient descent increases similarity. `repa_schedule=none` is supported by the low-level loss/CFM for regression testing; the production experiment contract requires `taro_clean_fraction`.

## Representation and architecture

- Frozen `microsoft/wavlm-base-plus`, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, checkpoint SHA256 `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.
- Final teacher layer 12, 768 dimensions, cached fp16 features from complete unmasked 16-kHz ground-truth utterances. Teacher is not placed on training GPUs and receives no gradients.
- Each unpadded 50-Hz teacher sequence is linearly interpolated to its valid 40-Hz latent length with `align_corners=False`, as in the REPA donor. This retains that interpolation approximation rather than reproducing TARO's convolutional resampling.
- Student is audio output of the tenth AVT block (zero-based index 9). Auxiliary projector is `768 -> 2048 -> 2048 -> 768`, SiLU, with bias.
- Original 12 AVT joint blocks and 6 audio-only blocks, CAM++ conditioning, CTC taps/warmup, Semantic-VAE and all attention edges retained.
- AV/VA attention is strictly zero beyond aligned-frame distance 4 (9 frames at interior positions); only existing keys are used at boundaries. At 40 Hz the radius is 100 ms. Other attention edges remain globally available.
- Projector adds 7,344,896 parameters and six state tensors. Inference loads them strictly but never executes the head, reads teacher features, or runs WavLM.

## Training and artifact isolation

Production config: `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus.yaml`.

The inherited recipe is unchanged: seed 666, all 79,613 CelebV-Dub training utterances, 4 GPUs, bf16, 3,600 frames/GPU and at most 32 samples/GPU, LR 5e-5, 20k LR warmup, inherited 200-epoch decay horizon, stop at 200k updates. CTC is zero through 10k and reaches 0.03 at 30k. Numbered checkpoints every 50k include 150k and 200k; `model_last.pt` every 5k.

Initialization uses the same pinned S2c 70k EMA parent as hard-window9, with fresh optimizer and update counter. This is an independent training comparison, not continuation from hard-window9 150k/200k. All inherited parameters use the same initialization; only the new projector is added at the end of construction.

Run (use the existing environment; do not change the shared editable package):

```bash
source env.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_4x4090.sh \
  > logs/train_wavlm_tra_20261031.log 2>&1 < /dev/null &
```

Checkpoint directory relative to `${ROOT_PREFIX}/zjw524/projects/data/ckpts/`:
`AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char`.

Each checkpoint embeds `wavlm_tra_contract`. Resume checks that contract before restoring tensors/optimizer. Inference also requires matching `speaker_training_contract.json`, its resolved config, the teacher receipt, and the strict hard-window contract. This prevents fixed REPA, another weighting rule, or plain hard9 weights from being silently treated as this experiment.

The existing S1 inference/evaluation shell entrypoints in this copy point to this new config and checkpoint directory. Historical evaluation documents copied from the source describe the source experiments, not TRA results.

## Diagnostics and verification

TensorBoard records `loss`, `diff_loss`, CTC weight/loss, `repa_unweighted_loss`, actual time-weighted `repa_loss`, `repa_weighted_loss` (after multiplying by 0.1), weight mean/min/max and frame-weighted mean, projector gradient/weight norms, global gradient norm, and hard-window radius/size.

The three-update real-data smoke uses `scripts/smoke_train_wavlm_tra_4x4090.sh`, its own checkpoint folder and TensorBoard run, and leaves the production LR schedule/batch size unchanged. Cache validation is reproducible with `PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python scripts/audit_wavlm_repa_cache.py --output logs/repa_cache_audit.json`.

Verified on 2026-10-31 before the production launch:

- 42 new tests passed: 14 loss/model/inference, 9 dataset/migration, 15 objective-contract, and 4 inherited trainer save/resume hook tests.
- Existing strict-window regression: 15 CPU tests passed and 2 CUDA-specific cases skipped; all 11 original window-contract tests passed. Actual bf16 training/backward was separately exercised on all four GPUs.
- Production 159-token model: 376,563,842 -> 383,908,738 parameters; 802 -> 808 state tensors. All inherited tensors are bitwise equal under seed 666; disabled REPA preserves RNG/state/output exactly. Only 14 expected resolved configuration fields differ.
- Full WavLM cache audit: 79,613/79,613 unique source IDs and arrays, 16,326,966 teacher frames, all finite fp16 [frames,768], no missing/extra/invalid files.
- Actual four-GPU bf16 production-batch smoke completed three optimizer updates and wrote a checkpoint; a second identical invocation restored its checkpoint and exited correctly at update 3.
- Smoke checkpoint: online update 3, EMA step 70,003, all 3,940 serialized tensors finite; the production inference builder strictly loaded all 808 EMA state entries, each bitwise equal to the saved EMA. TensorBoard confirms nonzero projector gradients, finite total/CFM/TRA losses, and total=CFM+0.1*TRA while CTC is still disabled.
- Full CFG Euler inference regression confirms the projector is never called and generated latent outputs match the baseline path given matching weights.
- New/supporting Python files pass Ruff and shell scripts pass syntax checks. Existing backbone/CFM/base trainer lint diagnostics are unchanged (no new diagnostics). Original hard9 source hashes remain unchanged for all 254 copied files.

Runtime evidence: `logs/repa_cache_audit.json`, `logs/tra_recipe_validation.json`, `logs/smoke_checkpoint_audit.json`, `logs/smoke_train_3updates_20261031.log`, `logs/smoke_resume_3updates_20261031.log`, and `logs/test_*.log`. These artifacts are intentionally excluded from Git. No TRA quality metric has been measured yet.


## Production launch receipt — 2026-10-31

Checked at `2026-10-31T17:59:26.978365+08:00`. Started `Sat Oct 31 17:57:12 2026` (server timezone Asia/Shanghai), independent PID/SID `23586`, TTY `?`; four workers `23669, 23671, 23672, 23675` use physical GPUs 0–3. Implementation commit: `9f4762e522640460764a95577572982f1d3796e0`.

At this check the live log reached update **95** and TensorBoard loss reached update **95**. All 26 scalar series are finite, projector gradients are nonzero, and the strict-window diagnostic remains 9. This is a timestamped launch receipt, not a future progress estimate; current state is in `logs/train_wavlm_tra_20261031.log`.

Run name: `AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`.

TensorBoard logdir: `/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va_tra_wavlm_base_plus/runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`. Service PID **12138**, port **6031**, verified HTTP 200 and live loss/scalar data at [the current internal address](http://10.248.62.197:6031/). In the IDE bottom panel, open **Ports**, find **6031**, then click that row's **Forwarded Address**. The UI forwarding URL is not exposed to the available tools; no public forwarding URL is claimed here.

The smoke run has its separate event directory ending `smoke3_20261031_semantic_vae_40hz_CelebVDub_char`, TensorBoard PID13714/port6032; production weights and events are separate. Runtime launch details are in `logs/formal_training_status.json`.
