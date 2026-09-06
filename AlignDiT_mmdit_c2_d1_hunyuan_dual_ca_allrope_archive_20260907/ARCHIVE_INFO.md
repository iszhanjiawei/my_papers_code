# Hunyuan C2/D1 archive before restoring the original architecture

Created on 2026-09-07 at the user's request.

- Source project: `../AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/`.
- Repository HEAD at capture: `e7ea610`.
- The source project's tracked code had no uncommitted changes at capture.
- This is an independent full filesystem copy, not a source-code symlink.
- All 6,935 regular files were checksum-compared before rollback; no file
  content was missing or different. Directory timestamps alone differed.
- Project-local logs, results and TensorBoard runs were copied too, but are
  runtime artifacts and are not committed. The existing `data` symlink was
  copied as a symlink; external datasets/checkpoints were not duplicated.

## Preserved experiment

This archive retains the Hunyuan-style dual Audio/Video text cross-attention,
CA Q/K RMSNorm and ordinal RoPE, and all-head self-attention RoPE. It includes
both C2 (12 MM + 6 audio) and D1 (6 MM + 12 audio) configurations, launchers,
inference/evaluation entry points and regression tests.

The original project is being restored to its tracked state at `3f4fe89`,
immediately before the Hunyuan architecture commit `20e57f8`. Hunyuan-specific
checkpoints require this archived implementation; do not load them into the
restored original C2/D1 architecture.

To inspect or evaluate the preserved implementation, work from this directory
and explicitly use `PYTHONPATH=src` with the existing `aligndit` environment.
Do not change the shared editable-package installation. The preserved Hunyuan
launchers resolve the project root relative to their own location. Checkpoint
paths in configs still refer to the existing external experiment directories;
do not launch another training run into them unintentionally.

Existing external checkpoints remain under `data/ckpts/` (the shared data
symlink), including:

- `AlignDiT_MMDiT_C2_HunyuanDualCA_AllRoPE_12MM6A_finetune_hifigan_16k_CelebVDub_char/`
- `AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_6MM12A_CTC6_12_finetune_hifigan_16k_CelebVDub_char/`

The experiment metrics remain in `../实验结果/实验结果总汇.md`, sections 14 and 18.
Neither checkpoint files nor experiment results are deleted by the rollback.
