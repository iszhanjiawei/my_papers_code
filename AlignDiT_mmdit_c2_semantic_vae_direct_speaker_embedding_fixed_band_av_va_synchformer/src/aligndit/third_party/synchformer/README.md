# Vendored Synchformer visual encoder

Source: https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley, commit
`df7b005b5023df2a9b73e1d66dd51d452799884e`, local
`papers_codes/HunyuanVideo-Foley/hunyuanvideo_foley/models/synchformer`.

The adapted visual-only implementation is copied from the existing
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_synchformer` snapshot.
All source, configuration and license files are independent regular files in
this experiment; only immutable external feature caches and checkpoints are reused.

Only the MotionFormer visual path used by `Synchformer.forward` is constructed. The official checkpoint's `vfeat_extractor.*` keys load strictly; audio AST and synchronization classification head are unused by Foley visual conditioning and are omitted.

Local adaptations: replace timm's `trunc_normal_` and `to_2tuple` helpers with their PyTorch/local equivalents, and fail explicitly for absent bundled configuration rather than downloading files implicitly. Model computations, structure and configuration are retained.

The accompanying upstream NOTICE identifies Synchformer as MIT, copyright (c) 2024 Vladimir Iashin; upstream file headers retain MotionFormer/Facebook and Ross Wightman attribution. Hunyuan upstream LICENSE and NOTICE are preserved alongside this code. This directory is isolated within this experiment.

Original licenses: [MotionFormer (Apache-2.0)](https://github.com/facebookresearch/Motionformer/blob/main/LICENSE) and [Synchformer (MIT)](https://github.com/v-iashin/Synchformer/blob/main/LICENSE), copied here as `LICENSE.motionformer` and `LICENSE.synchformer`.
