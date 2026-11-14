# GRID fixed-13 LibriTTS + WavLM-Base+ REPA experiment

This independent snapshot targets GRID, as explicitly requested by the user.
The parent AGENTS.md rules for source isolation, Git, background jobs and
TensorBoard still apply. The original project must remain unchanged.

- Use `GRID_TRAINING.md`, `finetune_grid.yaml` and
  `src/aligndit/run/train/finetune_grid.sh` for this experiment. Inherited
  CelebV-Dub/LRS3 entries and evaluation reports describe other experiments.
- Preserve the copied AVT joint backbone, 12 multimodal + 6 audio-only blocks,
  fixed 13-frame soft AV/VA Gaussian prior, and full LibriTTS 50k EMA loading.
  Add only the reference project's WavLM-Base+ REPA supervision: student MM
  block 10, teacher layer 12, 768-D target, 2048-D projector, weight 0.1 from
  update 1. Do not introduce adaptive-band or change AVT joint attention.
  Preserve all 802 LibriTTS tensors and initialize only the new REPA projector.
- Reuse the complete audited GRID cache with its original train/val split,
  full MPG audio, 1024-D AV-HuBERT video, 64-D Semantic-VAE posterior samples
  and 192-D CAM++ embeddings, plus the existing pinned 50-Hz REPA cache.
  The validation split never enters training.
- Keep the pretrained character vocabulary and LibriSpeech normalization.
  Check GRID's actual extraction metadata, not the copied source VAE spec.
- Run with this snapshot's `PYTHONPATH=src`; keep outputs/checkpoints/events
  isolated from the source projects. Runtime artifacts are not Git content.
- Stop at 150,000 GRID updates; preserve the original 1,000-epoch LR schedule.
  Inherited evaluation reports describe the parent without REPA, not this run.
