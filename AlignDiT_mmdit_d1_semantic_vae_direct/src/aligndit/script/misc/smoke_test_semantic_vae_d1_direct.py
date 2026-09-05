"""D1 Semantic-VAE contract checks, with optional real-data integration checks.

Run from this snapshot with PYTHONPATH=src. The default is a short CPU check;
``--device cuda`` exercises the same assertions on an available GPU.
``--real-data --device cuda`` additionally checks the full-size pretrained
model on the pinned cache without optimizer updates or checkpoint writes.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.trainer_semantic_vae_direct_ctc_warmup import (
    SemanticVaeDirectD1CtcWarmupTrainer,
    ctc_lambda_for_update,
)
from aligndit.script.train.finetune_semantic_vae_d1_direct import build_model


def assert_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def check_schedule(model: CFM_VT) -> SimpleNamespace:
    # Values refer to the number of already completed child optimizer updates.
    expected = {0: 0.0, 9_999: 0.0, 10_000: 0.0000015, 19_999: 0.015, 29_999: 0.03, 70_000: 0.03}
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda value: value),
        ctc_target_lambda=0.03,
        ctc_warmup_start=10_000,
        ctc_warmup_end=30_000,
        current_ctc_lambda=0.0,
    )
    for completed, target in expected.items():
        value = ctc_lambda_for_update(completed, target=0.03, warmup_start=10_000, warmup_end=30_000)
        assert_close(value, target, name=f"CTC after {completed} completed updates")
        SemanticVaeDirectD1CtcWarmupTrainer._before_update(trainer, completed)
        assert_close(model.ctc_lambda, target, name="trainer applies schedule to unwrapped model")
        assert_close(trainer.current_ctc_lambda, target, name="trainer scalar weight")
    # A resumed child at 20k applies its next-update weight, not the S2c parent's 70k.
    SemanticVaeDirectD1CtcWarmupTrainer._before_update(trainer, 20_000)
    assert_close(model.ctc_lambda, 0.0150015, name="resumed child schedule")
    return trainer


def check_forward_backward(model: CFM_VT, device: torch.device, *, weight: float) -> dict[str, float]:
    torch.manual_seed(42)
    random.seed(42)
    model.train().to(device)
    model.zero_grad(set_to_none=True)
    model.ctc_lambda = weight
    backbone = model.transformer
    assert [type(block).__name__ for block in backbone.transformer_blocks] == ["MMDiTBlock_VT"] * 6 + ["DiTBlock"] * 12
    assert backbone.layer_indices_ctc == (5, 11)
    assert backbone.ctc_sampling_ratios == (1, 1)
    assert all(projector.sampling_ratios == (1, 1) for projector in backbone.projectors_ctc)
    latent = torch.randn(2, 24, 64, device=device)
    video = torch.randn(2, 24, 1024, device=device)
    lengths = torch.tensor([24, 20], device=device)
    text = torch.tensor([[1, 2, 3, 4], [3, 3, 4, -1]], device=device)
    text_lengths = torch.tensor([4, 3], device=device)
    projected = []
    handles = [
        projector.register_forward_hook(lambda _module, _inputs, output: projected.append(output))
        for projector in backbone.projectors_ctc
    ]
    try:
        loss, components, _, prediction = model(
            latent,
            text=text,
            lens=lengths,
            text_lens=text_lengths,
            video=video,
            video_lens=lengths.clone(),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert prediction.shape == latent.shape
    assert torch.isfinite(prediction).all() and torch.isfinite(loss)
    assert all(math.isfinite(value) for value in components.values())
    assert len(projected) == 2
    for logits, output_lengths in projected:
        assert logits.shape[:2] == (2, 24)
        assert torch.equal(output_lengths, lengths), "40-Hz CTC must not downsample the latent timeline"
    if weight:
        independent_losses = [
            F.ctc_loss(
                logits.transpose(0, 1).log_softmax(-1),
                text,
                output_lengths,
                text_lengths,
                blank=backbone.text_embed.text_embed.num_embeddings,
                reduction="mean",
                zero_infinity=True,
            )
            for logits, output_lengths in projected
        ]
        assert_close(components["ctc_loss"], torch.stack(independent_losses).mean().item(), name="dual-head CTC mean")
        assert_close(loss.item(), components["diff_loss"] + weight * components["ctc_loss"], name="weighted total loss")
    else:
        assert "ctc_loss" not in components, "CTC is inactive during the first 10k updates"
        assert_close(loss.item(), components["diff_loss"], name="zero-CTC total loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    for projector in backbone.projectors_ctc:
        gradient = projector.model[-1].weight.grad
        if weight:
            assert gradient is not None and torch.count_nonzero(gradient) > 0, "each CTC head must receive a gradient"
        else:
            assert gradient is None, "CTC heads must not train when lambda is zero"
    return {"total_loss": loss.item(), **components}


def check_real_data(config, device: torch.device) -> None:
    from accelerate.utils import set_seed

    from aligndit.model.semantic_vae_dataset import SemanticVaeCelebVDubDataset
    from aligndit.model.semantic_vae_direct_migration import (
        load_s2c_ema_state,
        migrate_s2c_ema_into_model,
        validate_parent_artifacts,
    )
    from f5_tts.model.utils import get_tokenizer

    if device.type != "cuda":
        raise ValueError("--real-data requires --device cuda for the full-size bf16 check")
    dataset_keys = (
        "manifest_path",
        "cache_root",
        "normalization_path",
        "vocab_path",
        "expected_manifest_sha256",
        "expected_inventory_sha256",
        "expected_normalization_sha256",
        "expected_vocab_sha256",
        "expected_record_count",
    )
    dataset = SemanticVaeCelebVDubDataset(**{key: config.datasets[key] for key in dataset_keys})
    print(
        f"Real manifest validated: {len(dataset)} records; CTC feasible={dataset.ctc_feasible_count}, "
        f"infeasible={dataset.ctc_infeasible_count}",
        flush=True,
    )
    indices = [
        min(
            (i for i, row in enumerate(dataset.records) if row["ctc_feasible_40hz"] == feasible),
            key=lambda i: dataset.records[i]["latent_frames"],
        )
        for feasible in (True, False)
    ]
    indices.append(
        next(
            i
            for i, row in enumerate(dataset.records)
            if 80 <= row["latent_frames"] <= 160 and row["ctc_min_input_frames"] * 2 <= row["latent_frames"]
        )
    )
    batch = dataset.collate_fn([dataset[i] for i in indices])
    print(
        f"Real batch: indices={indices}; latent lengths={batch['mel_lengths'].tolist()}; "
        f"text lengths={batch['text_lengths'].tolist()}",
        flush=True,
    )
    parent = config.ckpts
    validate_parent_artifacts(
        parent.pretrained_path,
        parent.parent_contract_path,
        expected_checkpoint_sha256=parent.expected_parent_sha256,
        expected_checkpoint_size=parent.expected_parent_size,
        expected_contract_sha256=parent.expected_parent_contract_sha256,
    )
    source, ema_step = load_s2c_ema_state(
        parent.pretrained_path,
        expected_parent_contract_sha256=parent.expected_parent_contract_sha256,
        expected_parent_update=parent.expected_parent_update,
    )
    set_seed(666)
    vocabulary, vocabulary_size = get_tokenizer(config.datasets.vocab_path, "custom")
    model = build_model(config, vocabulary, vocabulary_size)
    report = migrate_s2c_ema_into_model(
        model,
        source,
        parent_path=parent.pretrained_path,
        parent_sha256=parent.expected_parent_sha256,
        parent_size=parent.expected_parent_size,
        parent_contract_sha256=parent.expected_parent_contract_sha256,
        parent_ema_step=ema_step,
    )
    state = model.state_dict()
    common = set(state) & set(source)
    assert len(common) == 303 and all(torch.equal(state[key], source[key]) for key in common)
    print(
        f"S2c migration bit-exact: loaded={report.loaded_key_count}, "
        f"new={len(report.new_target_keys)}, target={report.target_key_count}",
        flush=True,
    )
    del source, state
    gc.collect()
    model.to(device).train()
    for weight in (0.0, 0.015, 0.03):
        set_seed(666)
        model.zero_grad(set_to_none=True)
        model.ctc_lambda = weight
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, prediction = model(
                batch["mel"].transpose(1, 2).to(device),
                text=batch["text"],
                lens=batch["mel_lengths"].to(device),
                text_lens=batch["text_lengths"].to(device),
                video=batch["video"].to(device),
                video_lens=batch["video_lengths"].to(device),
            )
        assert torch.isfinite(loss) and torch.isfinite(prediction).all()
        assert_close(
            loss.item(),
            components["diff_loss"] + weight * components.get("ctc_loss", 0.0),
            name="real bf16 weighted loss",
        )
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        for head in model.transformer.projectors_ctc:
            gradient = head.model[-1].weight.grad
            assert (gradient is not None and torch.count_nonzero(gradient) > 0) if weight else gradient is None
        print(f"Real bf16 forward/backward lambda={weight}: total={loss.item()}, {components}", flush=True)

    model.zero_grad(set_to_none=True)
    model.eval()
    item = dataset[indices[0]]
    frames = item["video"].shape[0]
    condition = item["mel_spec"].T[None].to(device)
    video = item["video"].to(device)
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=condition,
            text=[item["text"] + "  " + item["text"]],
            duration=torch.tensor([2 * frames], device=device),
            video=torch.cat([torch.zeros_like(video), video])[None],
            lens=torch.tensor([frames], device=device),
            steps=2,
            cfg_strength=5,
            cfg_strength_v=2,
            sway_sampling_coef=-1,
            seed=0,
            use_epss=True,
        )
    assert generated.shape == (1, 2 * frames, 64) and torch.isfinite(generated).all()
    assert torch.equal(generated[:, :frames], condition)
    print(
        f"Real latent sampling passed: {tuple(generated.shape)}, prompt preserved. "
        "Two-step plumbing check only; not a quality evaluation.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--real-data", action="store_true", help="Also check pinned data and full-size S2c migration")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(666)
    config_path = Path(__file__).parents[2] / "config/finetune_celebvdub_mm_d1_semantic_vae_direct.yaml"
    config = OmegaConf.load(config_path)
    # Keep all 18 blocks and D1 taps, reducing width and frontend size for speed.
    config.model.arch.update(dim=48, heads=3, dim_head=16, text_dim=32, conv_layers=0, use_conformer=False)
    vocabulary = {" ": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    model = build_model(config, vocabulary, 27)
    trainer = check_schedule(model)
    for weight in (0.0, 0.03):
        results = check_forward_backward(model, device, weight=weight)
        trainer.current_ctc_lambda = weight
        diagnostics = SemanticVaeDirectD1CtcWarmupTrainer._forward_diagnostics(trainer, None, results)
        assert_close(
            diagnostics["ctc_weighted_loss"], weight * results.get("ctc_loss", 0.0), name="logged weighted CTC"
        )
        print(f"audio_only lambda={weight}: {results}")

    arch = OmegaConf.to_container(config.model.arch, resolve=True)
    arch.pop("ctc_sampling_ratios")
    legacy = DiT_VT_MMDiT(**arch, text_num_embeds=27, mel_dim=64)
    assert legacy.ctc_sampling_ratios == (2, 1), "The inherited mel backbone default must remain unchanged"
    del legacy
    # Preserve the optional inherited Hunyuan CA; the D1 VAE config uses audio_only.
    dual = CFM_VT(
        transformer=DiT_VT_MMDiT(
            **arch, text_num_embeds=27, mel_dim=64, ctc_sampling_ratios=[1, 1], text_attention_mode="hunyuan_dual"
        ),
        mel_spec_module=PrecomputedAudioRepresentation(64, 16000, 400),
        num_channels=64,
        vocab_char_map=vocabulary,
        audio_video_ratio=1,
        ctc_lambda=0.03,
    )
    print(f"hunyuan_dual lambda=0.03: {check_forward_backward(dual, device, weight=0.03)}")
    print(
        f"D1 Semantic-VAE smoke passed on {device}: topology, 40-Hz shapes, both CTC gradients, averaged/weighted losses, warmup and resume hooks"
    )
    if args.real_data:
        del model, dual, trainer
        gc.collect()
        torch.cuda.empty_cache()
        check_real_data(OmegaConf.load(config_path), device)


if __name__ == "__main__":
    main()
