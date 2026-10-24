"""End-to-end contract smoke test for the Semantic-VAE Direct-C2 experiment."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
from aligndit.model.semantic_vae_direct_migration import (
    EXPECTED_LOADED_KEYS,
    EXPECTED_NEW_TARGET_KEYS,
    EXPECTED_SOURCE_KEYS,
    EXPECTED_TARGET_KEYS,
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from f5_tts.model.utils import get_tokenizer


ROOT_PREFIX = os.environ.get("ROOT_PREFIX", "")
DEFAULT_PARENT_DIR = Path(
    f"{ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech"
)
DEFAULT_CACHE_ROOT = Path(f"{ROOT_PREFIX}/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32")
DEFAULT_NORMALIZATION = Path(
    f"{ROOT_PREFIX}/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/"
    "train_normalization.json"
)
DEFAULT_VOCAB = Path(f"{ROOT_PREFIX}/zjw524/projects/data/CelebVDub_char/vocab.txt")


def build_model(vocab_size: int, vocab_char_map: dict[str, int]) -> CFM_VT:
    transformer = DiT_VT_MMDiT(
        dim=768,
        depth=18,
        heads=12,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=False,
        qk_norm="rms_norm",
        conv_layers=4,
        pe_attn_head=1,
        attn_mask_enabled=True,
        checkpoint_activations=False,
        use_conformer=True,
        layer_indices_ctc=[6, 12],
        ctc_sampling_ratios=[1, 1],
        n_mm_layers=12,
        n_text_layers=12,
        prompt_isolated_ca=False,
        audio_video_ratio=1,
        video_dim=1024,
        video_rope_scaled=False,
        text_num_embeds=vocab_size,
        mel_dim=64,
    )
    return CFM_VT(
        transformer=transformer,
        mel_spec_module=PrecomputedAudioRepresentation(64, 16_000, 400),
        num_channels=64,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=1,
        ctc_lambda=0.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT_DIR / "model_70000.pt")
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT_DIR / "training_contract.json")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()

    validate_parent_artifacts(
        args.parent,
        args.parent_contract,
        expected_checkpoint_sha256="02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4",
        expected_checkpoint_size=2_762_690_094,
        expected_contract_sha256="3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d",
    )
    dataset = SemanticVaeCelebVDubDataset(
        manifest_path=args.cache_root / "manifests/train.jsonl",
        cache_root=args.cache_root,
        normalization_path=args.normalization,
        vocab_path=args.vocab,
        expected_manifest_sha256="0d16d5c8f00eb25ee51c7de604299a37cace1bc0e65b7127a45420c433b4d395",
        expected_inventory_sha256="a6478cce785748cbcefd87af54eafa9f654d735afa1c41b8f846e041cbc1286d",
        expected_normalization_sha256="65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6",
        expected_vocab_sha256="225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d",
    )
    infeasible_index = next(index for index, record in enumerate(dataset.records) if not record["ctc_feasible_40hz"])
    data_batch = dataset.collate_fn([dataset[0], dataset[infeasible_index]])
    assert data_batch["mel"].shape[1] == 64
    assert data_batch["mel"].shape[2] == data_batch["video"].shape[1]
    assert data_batch["ctc_feasible"].tolist() == [True, False]

    vocab_char_map, vocab_size = get_tokenizer(str(args.vocab), "custom")
    torch.manual_seed(1234)
    model = build_model(vocab_size, vocab_char_map)
    assert not hasattr(model.transformer, "text_context_norm")
    assert model.transformer.normalize_text_context is False
    assert model.transformer.layer_indices_ctc == (6, 12)
    assert model.transformer.ctc_sampling_ratios == (1, 1)
    assert all(projector.sampling_ratios == (1, 1) for projector in model.transformer.projectors_ctc)
    assert model.transformer.projectors_ctc[0].model[0].out_channels == 768

    source_state, parent_ema_step = load_s2c_ema_state(
        args.parent,
        expected_parent_contract_sha256="3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d",
    )
    report = migrate_s2c_ema_into_model(
        model,
        source_state,
        parent_path=args.parent,
        parent_sha256="02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4",
        parent_size=2_762_690_094,
        parent_contract_sha256="3d6fcf6649511a0f21546ca995ed047dfcca5ff58e9c2d3196d7c67b24e7633d",
        parent_ema_step=parent_ema_step,
    )
    assert report.source_key_count == EXPECTED_SOURCE_KEYS
    assert report.target_key_count == EXPECTED_TARGET_KEYS
    assert report.loaded_key_count == EXPECTED_LOADED_KEYS
    assert len(report.new_target_keys) == EXPECTED_NEW_TARGET_KEYS
    migrated_state = model.state_dict()
    for key in sorted(set(source_state) & set(migrated_state)):
        if not torch.equal(source_state[key], migrated_state[key].cpu()):
            raise RuntimeError(f"Migrated tensor is not bit-exact: {key}")

    parameters = list(model.parameters())
    # Original C2 keeps the video null embedding as a buffer, not a trainable parameter.
    assert len(parameters) == 701
    assert all(parameter.requires_grad for parameter in parameters)
    optimizer = AdamW(parameters, lr=5e-5)
    assert len(optimizer.param_groups) == 1
    assert math.isclose(optimizer.param_groups[0]["lr"], 5e-5)
    del optimizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Direct-C2 forward/backward smoke test")
    device = torch.device("cuda:0")
    model = model.to(device)
    model.train()
    latent = torch.randn(2, 48, 64, device=device)
    video = torch.randn(2, 48, 1024, device=device)
    lengths = torch.tensor([48, 41], device=device)
    texts = ["hello world", "direct c2"]
    text_lengths = torch.tensor([len(text) for text in texts], device=device)
    loss, components, _, prediction = model(
        latent,
        text=texts,
        lens=lengths,
        text_lens=text_lengths,
        video=video,
        video_lens=lengths.clone(),
    )
    if not torch.isfinite(loss) or not all(math.isfinite(value) for value in components.values()):
        raise RuntimeError(f"Non-finite Direct-C2 loss: total={loss}, components={components}")
    assert prediction.shape == latent.shape
    loss.backward()
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise RuntimeError("Direct-C2 backward produced no finite gradient")

    print(
        "Semantic-VAE Direct-C2 smoke passed: "
        f"dataset={len(dataset)}, migration={report.loaded_key_count}/{report.target_key_count}, "
        f"loss={loss.item():.6f}, diff={components['diff_loss']:.6f}, ctc={components['ctc_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
