"""Real-schema smoke test for S2c migration and all S3 optimizer policies."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_c2_stage import (
    EXPECTED_LOADED_AUDIO_PARAMETER_TENSORS,
    EXPECTED_NEW_MULTIMODAL_PARAMETER_TENSORS,
    S3A_SOURCE_KIND,
    SINGLE_STAGE_S3,
    configure_s3_parameters,
    load_s3_parent_ema,
)
from aligndit.model.trainer_semantic_vae_c2 import SemanticVaeC2Trainer
from aligndit.script.misc.validate_semantic_vae_c2_checkpoint import (
    _semantic_digest,
    _validate_optimizer_scheduler,
)
from f5_tts.model.utils import get_tokenizer


def main() -> None:
    root_prefix = os.environ.get("ROOT_PREFIX", "")
    checkpoint_dir = Path(
        f"{root_prefix}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech"
    )
    parent_path = checkpoint_dir / "model_70000.pt"
    parent_contract = checkpoint_dir / "training_contract.json"
    vocab_path = Path(f"{root_prefix}/zjw524/projects/data/CelebVDub_char/vocab.txt")
    for path in (parent_path, parent_contract, vocab_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    transformer = DiT_VT_MMDiT(
        dim=768,
        depth=18,
        heads=12,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=True,
        qk_norm="rms_norm",
        conv_layers=4,
        pe_attn_head=1,
        attn_mask_enabled=True,
        always_use_attention_mask=True,
        mask_input_embeddings=True,
        strict_audio_video_alignment=True,
        checkpoint_activations=False,
        use_conformer=True,
        layer_indices_ctc=[6, 12],
        ctc_sampling_ratios=[1, 1],
        projector_dim=2048,
        n_mm_layers=12,
        n_text_layers=12,
        prompt_isolated_ca=False,
        audio_video_ratio=1,
        video_dim=1024,
        video_rope_scaled=False,
        text_num_embeds=vocab_size,
        mel_dim=64,
    )
    model = CFM_VT(
        transformer=transformer,
        mel_spec_module=PrecomputedAudioRepresentation(n_channels=64, target_sample_rate=16_000, hop_length=400),
        num_channels=64,
        vocab_char_map=vocab_char_map,
        ctc_lambda=0.1,
        audio_video_ratio=1,
        strict_audio_video_alignment=True,
    )
    report = load_s3_parent_ema(
        model,
        parent_path,
        parent_kind=S3A_SOURCE_KIND,
        expected_parent_stage="s2c",
        expected_parent_update=70_000,
        expected_parent_contract_sha256="3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d",
        expected_parent_sha256="02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4",
        expected_parent_size=2_762_690_094,
    )
    s3a_groups, s3a_parameters = configure_s3_parameters(
        model,
        stage="s3a",
        learning_rates={"multimodal_new": 5e-5},
        weight_decay=0.01,
    )
    if len(s3a_parameters.trainable_names) != EXPECTED_NEW_MULTIMODAL_PARAMETER_TENSORS:
        raise RuntimeError("S3a trainable tensor count changed")
    if len(s3a_parameters.frozen_names) != EXPECTED_LOADED_AUDIO_PARAMETER_TENSORS:
        raise RuntimeError("S3a frozen tensor count changed")
    s3b_groups, s3b_parameters = configure_s3_parameters(
        model,
        stage="s3b",
        learning_rates={
            "multimodal_new": 5e-5,
            "interface": 2e-5,
            "audio_blocks_0_5": 5e-6,
            "audio_backbone_rest": 1e-5,
        },
        weight_decay=0.01,
    )
    if s3b_parameters.frozen_names:
        raise RuntimeError("S3b must unfreeze every model parameter")

    single_stage_learning_rates = {
        "text_conditioner": 5e-6,
        "multimodal_core": 1e-5,
        "multimodal_gates": 1e-6,
        "ctc_heads": 1e-5,
        "interface": 5e-6,
        "audio_blocks_0_5": 2e-6,
        "audio_backbone_rest": 5e-6,
        "shared_conditioning": 1e-6,
    }
    single_stage_groups, single_stage_parameters = configure_s3_parameters(
        model,
        stage=SINGLE_STAGE_S3,
        learning_rates=single_stage_learning_rates,
        weight_decay=0.01,
    )
    if single_stage_parameters.frozen_names:
        raise RuntimeError("Single-stage S3 must jointly adapt every model parameter")
    if set(single_stage_parameters.category_names) != set(single_stage_learning_rates):
        raise RuntimeError(
            "Single-stage category mismatch: "
            f"expected={sorted(single_stage_learning_rates)}, "
            f"got={sorted(single_stage_parameters.category_names)}"
        )
    category_by_name = {
        name: category for category, names in single_stage_parameters.category_names.items() for name in names
    }
    expected_representatives = {
        "transformer.text_embed.": "text_conditioner",
        "transformer.video_embed.": "multimodal_core",
        "transformer.projectors_ctc.": "ctc_heads",
        "transformer.input_embed.proj.": "interface",
        "transformer.time_embed.": "shared_conditioning",
        "transformer.norm_out.": "shared_conditioning",
        "transformer.transformer_blocks.0.cross_attn.": "text_conditioner",
        "transformer.transformer_blocks.0.cross_attn_ada.": "multimodal_gates",
        "transformer.transformer_blocks.0.v_attn_norm.": "multimodal_gates",
        "transformer.transformer_blocks.0.v_attn.": "multimodal_core",
        "transformer.transformer_blocks.0.v_ff.": "multimodal_core",
        "transformer.transformer_blocks.0.attn.": "audio_blocks_0_5",
        "transformer.transformer_blocks.6.attn.": "audio_backbone_rest",
    }
    for prefix, expected_category in expected_representatives.items():
        matches = [name for name in category_by_name if name.startswith(prefix)]
        if not matches or any(category_by_name[name] != expected_category for name in matches):
            raise RuntimeError(f"Single-stage category {expected_category!r} does not own prefix {prefix!r}")
    for group in single_stage_groups:
        category = group["group_name"].rsplit(".", 1)[0]
        if group["lr"] != single_stage_learning_rates[category]:
            raise RuntimeError(f"Single-stage optimizer group has wrong LR: {group['group_name']}")
        expected_weight_decay = 0.0 if group["group_name"].endswith(".no_decay") else 0.01
        if group["weight_decay"] != expected_weight_decay:
            raise RuntimeError(f"Single-stage optimizer group has wrong decay: {group['group_name']}")

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = object.__new__(SemanticVaeC2Trainer)
        trainer.stage = "s3a"
        trainer.stage_start_update = 0
        trainer.max_stage_updates = 5_000
        trainer.save_per_updates = 5_000
        trainer.checkpoint_path = temp_dir
        calls = []

        def fake_save(stage_update, *, last=False):
            calls.append((stage_update, last))
            name = "model_last.pt" if last else f"model_{stage_update}.pt"
            (Path(temp_dir) / name).touch()

        trainer.save_checkpoint = fake_save
        trainer.reconcile_checkpoint_files(5_000)
        if calls != [(5_000, True), (5_000, False)]:
            raise RuntimeError(f"Interrupted checkpoint reconciliation changed: {calls}")

    synthetic_checkpoint = {
        "optimizer_state_dict": {
            "param_groups": [
                {
                    "group_name": "multimodal_new.decay",
                    "initial_lr": 5e-5,
                    "lr": 4e-5,
                    "params": [0, 1],
                    "weight_decay": 0.01,
                }
            ],
            "state": {},
        },
        "scheduler_state_dict": {"_last_lr": [4e-5], "last_epoch": 4},
    }
    _validate_optimizer_scheduler(
        synthetic_checkpoint,
        path=Path("synthetic.pt"),
        contract_groups=[
            {
                "group_name": "multimodal_new.decay",
                "lr": 5e-5,
                "parameter_names": ["a", "b"],
                "weight_decay": 0.01,
            }
        ],
        stage_update=1,
        world_size=4,
    )
    if _semantic_digest({"b": [1, 2], "a": 3.0}, field="left") != _semantic_digest(
        {"a": 3.0, "b": [1, 2]}, field="right"
    ):
        raise RuntimeError("Semantic resume-state digest depends on mapping insertion order")

    print(
        json.dumps(
            {
                "migration": report.to_dict(),
                "s3a": {
                    "trainable_tensors": len(s3a_parameters.trainable_names),
                    "frozen_tensors": len(s3a_parameters.frozen_names),
                    "optimizer_groups": [group["group_name"] for group in s3a_groups],
                },
                "s3b": {
                    "trainable_tensors": len(s3b_parameters.trainable_names),
                    "frozen_tensors": len(s3b_parameters.frozen_names),
                    "optimizer_groups": [group["group_name"] for group in s3b_groups],
                },
                "s3_single_stage": {
                    "trainable_tensors": len(single_stage_parameters.trainable_names),
                    "frozen_tensors": len(single_stage_parameters.frozen_names),
                    "category_tensors": {
                        name: len(parameters) for name, parameters in single_stage_parameters.category_names.items()
                    },
                    "optimizer_groups": [group["group_name"] for group in single_stage_groups],
                },
                "checkpoint_reconciliation": "passed",
                "validator_resume_schema": "passed",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
