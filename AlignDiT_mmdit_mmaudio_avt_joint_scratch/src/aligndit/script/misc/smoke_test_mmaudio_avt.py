"""Regression canary for the MMAudio-style AVT Joint-DiT experiment.

CPU suite:
    PYTHONPATH=src python -u src/aligndit/script/misc/smoke_test_mmaudio_avt.py

Four-GPU DDP suite:
    PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
        src/aligndit/script/misc/smoke_test_mmaudio_avt.py --ddp

Full-width four-GPU DDP suite:
    PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
        src/aligndit/script/misc/smoke_test_mmaudio_avt.py --ddp-full
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import aligndit.model.cfm_vt as cfm_vt_module
from aligndit.model.backbone.dit_vt_mmaudio import DiT_VT_MMAudio, MMDiTBlockAVT
from aligndit.model.cfm_vt import CFM_VT
from f5_tts.model.modules import DiTBlock


torch.set_num_threads(1)


def _tiny_model(**overrides) -> DiT_VT_MMAudio:
    config = {
        "dim": 64,
        "depth": 4,
        "heads": 4,
        "dim_head": 16,
        "dropout": 0.0,
        "ff_mult": 2,
        "mel_dim": 8,
        "text_num_embeds": 32,
        "text_dim": 32,
        "text_mask_padding": True,
        "qk_norm": "rms_norm",
        "conv_layers": 1,
        "pe_attn_head": 1,
        "attn_backend": "torch",
        "attn_mask_enabled": True,
        "checkpoint_activations": False,
        "use_conformer": False,
        "layer_indices_ctc": [1, 3],
        "n_mm_layers": 2,
        "audio_video_ratio": 4,
        "video_dim": 16,
        "video_rope_scaled": True,
        "av_ff_kernel_size": 3,
        "text_ff_kernel_size": 1,
        "text_input_ff_mult": 2,
        "last_joint_pre_only": True,
    }
    config.update(overrides)
    return DiT_VT_MMAudio(**config)


def _full_model() -> DiT_VT_MMAudio:
    return DiT_VT_MMAudio(
        dim=768,
        depth=18,
        heads=12,
        ff_mult=2,
        mel_dim=80,
        text_num_embeds=159,
        text_dim=512,
        text_mask_padding=True,
        qk_norm="rms_norm",
        conv_layers=4,
        pe_attn_head=1,
        attn_backend="torch",
        attn_mask_enabled=True,
        checkpoint_activations=True,
        use_conformer=True,
        layer_indices_ctc=[5, 11],
        n_mm_layers=6,
        audio_video_ratio=4,
        video_dim=1024,
        video_rope_scaled=True,
        av_ff_kernel_size=3,
        text_ff_kernel_size=1,
        text_input_ff_mult=4,
        last_joint_pre_only=True,
    )


class _DummyMelSpec(nn.Module):
    n_mel_channels = 8

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the smoke canary supplies mel tensors directly")


def _batch(device: torch.device, *, poison_padding: bool = False) -> dict[str, torch.Tensor]:
    batch, audio_len, video_len, text_len = 2, 16, 4, 7
    generator = torch.Generator(device=device).manual_seed(1234)
    audio_mask = torch.tensor([[1] * audio_len, [1] * 13 + [0] * 3], dtype=torch.bool, device=device)
    video_mask = torch.tensor([[1] * video_len, [1] * 3 + [0]], dtype=torch.bool, device=device)
    text_mask = torch.tensor([[1] * text_len, [1] * 5 + [0] * 2], dtype=torch.bool, device=device)
    x = torch.randn(batch, audio_len, 8, generator=generator, device=device)
    cond = torch.randn(batch, audio_len, 8, generator=generator, device=device)
    video = torch.randn(batch, video_len, 16, generator=generator, device=device)
    text = torch.randint(0, 32, (batch, text_len), generator=generator, device=device)
    text[~text_mask] = -1
    if poison_padding:
        x[~audio_mask] = 1.0e4
        cond[~audio_mask] = -1.0e4
        video[~video_mask] = 1.0e4
        # The explicit mask is authoritative, even if padded token contents
        # are accidentally non-padding IDs.
        text[~text_mask] = 17

    complementary_mask = torch.zeros(batch, video_len, dtype=torch.bool, device=device)
    complementary_mask[0, 0] = True  # exercise the learned null-video parameter
    return {
        "x": x,
        "cond": cond,
        "text": text,
        "video": video,
        "time": torch.tensor([0.2, 0.8], device=device),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": video_mask,
        "complementary_mask": complementary_mask,
        "generation_mask": audio_mask.clone(),
        "cache": False,
    }


def _open_residual_gates(model: DiT_VT_MMAudio) -> None:
    """Open AdaLN-Zero branches so tests exercise their full gradients."""
    with torch.no_grad():
        for block in model.transformer_blocks:
            if isinstance(block, MMDiTBlockAVT):
                streams = (block.audio_block, block.video_block, block.text_block)
                for stream in streams:
                    if stream.pre_only:
                        continue
                    dim = stream.dim
                    stream.attn_norm.linear.bias[2 * dim : 3 * dim].fill_(0.25)
                    stream.attn_norm.linear.bias[5 * dim : 6 * dim].fill_(0.25)
            else:
                dim = model.dim
                block.attn_norm.linear.bias[2 * dim : 3 * dim].fill_(0.25)
                block.attn_norm.linear.bias[5 * dim : 6 * dim].fill_(0.25)
        model.proj_out.weight.normal_(std=0.02)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, msg=name)


def _assert_nonzero_grad(name: str, parameter: torch.nn.Parameter) -> None:
    grad = parameter.grad
    assert grad is not None, f"{name}: missing gradient"
    assert torch.isfinite(grad).all(), f"{name}: non-finite gradient"
    assert grad.abs().sum() > 0, f"{name}: zero gradient"


def test_scratch_config_contract() -> None:
    config_path = (
        Path(__file__).resolve().parents[2] / "config" / ("finetune_celebvdub_mmaudio_avt_joint_d1_scratch.yaml")
    )
    cfg = OmegaConf.load(config_path)
    assert cfg.datasets.name == "CelebVDub"
    assert cfg.model.backbone == "DiT_VT_MMAudio"
    assert cfg.model.arch.depth == 18 and cfg.model.arch.n_mm_layers == 6
    assert list(cfg.model.arch.layer_indices_ctc) == [5, 11]
    assert cfg.model.arch.checkpoint_activations is True
    assert cfg.ckpts.init_mode == "scratch"
    assert cfg.ckpts.pretrained_path is None
    assert cfg.ckpts.log_samples is False
    assert "MMAudioMMDiT_AVTJoint" in cfg.model.name and "Scratch" in cfg.model.name
    assert cfg.model.name in cfg.ckpts.save_dir
    print("[OK] scratch CelebVDub config cannot dispatch audio-pretrained initialization")


def test_structure_and_initialization() -> None:
    model = _tiny_model()
    blocks = model.transformer_blocks
    assert all(isinstance(block, MMDiTBlockAVT) for block in blocks[:2])
    assert all(type(block) is DiTBlock for block in blocks[2:])
    assert not any("cross_attn" in name for name, _ in model.named_modules())
    assert blocks[0].pre_only is False and blocks[1].pre_only is True
    assert blocks[1].audio_block.pre_only is False
    assert blocks[1].video_block.out_proj is None and blocks[1].video_block.ff is None
    assert blocks[1].text_block.out_proj is None and blocks[1].text_block.ff is None
    assert model.layer_indices_ctc == (1, 3)
    assert model.layer_map_ctc == {1: 0, 3: 1}

    for block in blocks:
        if isinstance(block, MMDiTBlockAVT):
            streams = (block.audio_block, block.video_block, block.text_block)
            for stream in streams:
                assert torch.count_nonzero(stream.attn_norm.linear.weight) == 0
                assert torch.count_nonzero(stream.attn_norm.linear.bias) == 0
                assert torch.count_nonzero(stream.qkv.weight) > 0
    assert torch.count_nonzero(model.proj_out.weight) == 0
    with torch.no_grad():
        for text_block in model.text_embed.text_blocks:
            text_block.grn.gamma.fill_(1.0)
    short_text = torch.tensor([[2, 4, 6]])
    mixed_text = torch.tensor(
        [
            [2, 4, 6, -1, -1, -1, -1, -1],
            [1, 3, 5, 7, 9, 11, 13, 15],
        ]
    )
    short_features = model.text_embed(short_text, 3)
    mixed_features = model.text_embed(mixed_text, 8)
    _assert_close("mask-aware character ConvNeXt", short_features[0], mixed_features[0, :3])
    try:
        _tiny_model(text_embedding_average_upsampling=True)
    except ValueError:
        pass
    else:
        raise AssertionError("native ordered-text stream accepted average upsampling")
    print("[OK] exact 2-joint/2-audio tiny analogue and AdaLN-Zero initialization")


def test_joint_masks_padding_and_pre_only() -> None:
    torch.manual_seed(7)
    block = MMDiTBlockAVT(
        dim=32,
        heads=4,
        dim_head=8,
        ff_mult=2,
        dropout=0.0,
        qk_norm="rms_norm",
        pe_attn_head=1,
        attn_mask_enabled=True,
        av_ff_kernel_size=3,
        text_ff_kernel_size=1,
        pre_only=False,
    )
    # Open all three residual streams without changing their random QKV/MLPs.
    with torch.no_grad():
        for stream in (block.audio_block, block.video_block, block.text_block):
            stream.attn_norm.linear.weight.zero_()
            stream.attn_norm.linear.bias.zero_()
            stream.attn_norm.linear.bias[64:96].fill_(0.3)
            stream.attn_norm.linear.bias[160:192].fill_(0.3)
    block.train()  # masks must remain active during training, not only eval

    audio_mask = torch.tensor(
        [
            [1] * 8,
            [1] * 6 + [0] * 2,
        ],
        dtype=torch.bool,
    )
    video_mask = torch.tensor([[1] * 3, [1] * 2 + [0]], dtype=torch.bool)
    text_mask = torch.tensor([[1] * 5, [1] * 3 + [0] * 2], dtype=torch.bool)
    marker = block.build_joint_key_mask(audio_mask, video_mask, text_mask)
    assert torch.equal(marker, torch.cat((audio_mask, video_mask, text_mask), dim=1))

    inputs = (torch.randn(2, 8, 32), torch.randn(2, 3, 32), torch.randn(2, 5, 32))
    poisoned = tuple(value.clone() for value in inputs)
    for value, mask in zip(poisoned, (audio_mask, video_mask, text_mask)):
        value[~mask] = 1.0e4
    time = torch.randn(2, 32)
    clean_out = block(*inputs, time, audio_mask, video_mask, text_mask)
    poison_out = block(*poisoned, time, audio_mask, video_mask, text_mask)
    for stream_name, clean, poison, mask in zip(
        ("audio", "video", "text"),
        clean_out,
        poison_out,
        (audio_mask, video_mask, text_mask),
    ):
        _assert_close(f"{stream_name} padding invariance", clean[mask], poison[mask])
        assert torch.count_nonzero(clean[~mask]) == 0
    assert not torch.allclose(clean_out[2][text_mask], inputs[2][text_mask])

    final_block = MMDiTBlockAVT(
        dim=32,
        heads=4,
        dim_head=8,
        ff_mult=2,
        dropout=0.0,
        qk_norm="rms_norm",
        pe_attn_head=1,
        attn_mask_enabled=True,
        av_ff_kernel_size=3,
        text_ff_kernel_size=1,
        pre_only=True,
    )
    with torch.no_grad():
        final_block.audio_block.attn_norm.linear.weight.zero_()
        final_block.audio_block.attn_norm.linear.bias.zero_()
        final_block.audio_block.attn_norm.linear.bias[64:96].fill_(0.3)
        final_block.audio_block.attn_norm.linear.bias[160:192].fill_(0.3)
        final_block.video_block.attn_norm.linear.weight.zero_()
        final_block.video_block.attn_norm.linear.bias.zero_()
        final_block.text_block.attn_norm.linear.weight.zero_()
        final_block.text_block.attn_norm.linear.bias.zero_()
    final_audio, final_video, final_text = final_block(*inputs, time, audio_mask, video_mask, text_mask)
    assert not torch.allclose(final_audio[audio_mask], inputs[0][audio_mask])
    assert torch.equal(final_video, inputs[1])
    assert torch.equal(final_text, inputs[2])
    print("[OK] [A,V,T] mask order, training padding invariance, text update, pre_only")


def test_full_forward_cfg_padding_and_gradients() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    _open_residual_gates(model)
    clean = _batch(torch.device("cpu"))
    poison = _batch(torch.device("cpu"), poison_padding=True)

    clean_out, clean_ctc = model(**clean)
    poison_out, poison_ctc = model(**poison)
    assert clean_out.shape == (2, 16, 8)
    assert list(clean_ctc) == [1, 3]
    _assert_close("full valid audio padding invariance", clean_out[clean["mask"]], poison_out[clean["mask"]])
    assert torch.count_nonzero(clean_out[~clean["mask"]]) == 0
    for layer_index in clean_ctc:
        clean_item = clean_ctc[layer_index]
        poison_item = poison_ctc[layer_index]
        for sample_index, valid_len in enumerate(clean_item["z_lens"].tolist()):
            _assert_close(
                f"CTC {layer_index} padding invariance sample {sample_index}",
                clean_item["z_tilde"][sample_index, :valid_len],
                poison_item["z_tilde"][sample_index, :valid_len],
            )

    packed, _ = model(**clean, cfg_infer=True)
    conditional, _ = model(**clean)
    video_dropped, _ = model(**clean, drop_video=True)
    unconditional, _ = model(**clean, drop_audio_cond=True, drop_text=True, drop_video=True)
    expected = torch.cat((conditional, video_dropped, unconditional), dim=0)
    _assert_close("batch-two branch-major packed CFG", packed, expected)

    scalar_time = {**clean, "time": torch.tensor(0.5)}
    singleton_time = {**clean, "time": torch.tensor([0.5])}
    scalar_packed, _ = model(**scalar_time, cfg_infer=True)
    singleton_packed, _ = model(**singleton_time, cfg_infer=True)
    _assert_close("scalar and singleton time expansion", scalar_packed, singleton_packed)
    try:
        model(**{**clean, "time": torch.tensor([0.1, 0.2, 0.3])})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid time batch shape was accepted")

    cache_kwargs = {
        "x": clean["x"],
        "cond": clean["cond"],
        "text": clean["text"],
        "video": clean["video"],
        "cache": True,
        "audio_mask": clean["mask"],
        "text_mask": clean["text_mask"],
        "video_mask": clean["video_mask"],
        "complementary_mask": clean["complementary_mask"],
    }
    cached_text = model.get_input_embed(**cache_kwargs)[1]
    changed_text = clean["text"].clone()
    changed_text[0, 0] = (changed_text[0, 0] + 1) % 32
    refreshed_text = model.get_input_embed(**{**cache_kwargs, "text": changed_text})[1]
    assert not torch.allclose(cached_text, refreshed_text)
    model.clear_cache()
    assert model.text_cond is None and model.text_uncond is None
    assert all(signature is None for signature in model._text_cache_signatures.values())

    model.train()
    model.zero_grad(set_to_none=True)
    prediction, ctc = model(**clean)
    target = torch.randn_like(prediction)
    loss = (prediction - target).square()[clean["mask"]].mean()
    loss = loss + sum(item["z_tilde"].square().mean() for item in ctc.values())
    loss.backward()
    first = model.transformer_blocks[0]
    last_joint = model.transformer_blocks[1]
    _assert_nonzero_grad("first audio QKV", first.audio_block.qkv.weight)
    _assert_nonzero_grad("first video QKV", first.video_block.qkv.weight)
    _assert_nonzero_grad("first text QKV", first.text_block.qkv.weight)
    _assert_nonzero_grad("first text FFN", first.text_block.ff.w2.weight)
    _assert_nonzero_grad("last-joint video pre-only QKV", last_joint.video_block.qkv.weight)
    _assert_nonzero_grad("last-joint text pre-only QKV", last_joint.text_block.qkv.weight)
    _assert_nonzero_grad("last audio-only attention", model.transformer_blocks[3].attn.to_q.weight)
    _assert_nonzero_grad("first CTC projector", model.projectors_ctc[0].model[0].weight)
    _assert_nonzero_grad("second CTC projector", model.projectors_ctc[1].model[0].weight)
    print("[OK] full forward, dual CTC, B>1 packed CFG, and tri-modal gradient paths")


def test_cfm_alignment_contract_and_exception_safe_cache() -> None:
    transformer = _tiny_model()
    cfm = CFM_VT(
        transformer=transformer,
        mel_spec_module=_DummyMelSpec(),
        vocab_char_map=None,
        ctc_lambda=0.0,
    )
    text = torch.tensor([[1, 2, 3]], dtype=torch.long)
    try:
        cfm(
            torch.randn(1, 16, 8),
            text=text,
            lens=torch.tensor([16]),
            text_lens=torch.tensor([3]),
            video=torch.randn(1, 3, 16),
            video_lens=torch.tensor([3]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("training accepted audio/video lengths that violate 4:1")

    try:
        cfm.sample(
            cond=torch.randn(1, 4, 8),
            text=text,
            duration=10,
            video=torch.randn(1, 3, 16),
            lens=torch.tensor([4]),
            steps=1,
            use_epss=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("sampling silently truncated a non-4:1 duration")

    original_odeint = cfm_vt_module.odeint

    def fail_after_one_evaluation(function, y0, timesteps, **_kwargs):
        function(timesteps[0], y0)
        assert transformer.text_cond is not None
        raise RuntimeError("injected ODE failure")

    cfm_vt_module.odeint = fail_after_one_evaluation
    try:
        try:
            cfm.sample(
                cond=torch.randn(1, 4, 8),
                text=text,
                duration=16,
                video=torch.randn(1, 4, 16),
                lens=torch.tensor([4]),
                steps=1,
                cfg_strength=0.0,
                cfg_strength_v=0.0,
                use_epss=False,
            )
        except RuntimeError as error:
            assert str(error) == "injected ODE failure"
        else:
            raise AssertionError("injected ODE failure did not propagate")
    finally:
        cfm_vt_module.odeint = original_odeint
    assert transformer.text_cond is None and transformer.text_uncond is None
    assert all(signature is None for signature in transformer._text_cache_signatures.values())
    print("[OK] strict 4:1 CFM contract and exception-safe text-cache lifecycle")


def run_cpu_suite() -> None:
    test_scratch_config_contract()
    test_structure_and_initialization()
    test_joint_masks_padding_and_pre_only()
    test_full_forward_cfg_padding_and_gradients()
    test_cfm_alignment_contract_and_exception_safe_cache()
    print("ALL MMAUDIO AVT CPU SMOKE TESTS PASSED")


def run_ddp_suite() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(2026)
    model = _tiny_model().to(device)
    _open_residual_gates(model)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1.0e-4)

    for step in range(2):
        torch.manual_seed(3000 + 10 * local_rank + step)
        batch = _batch(device)
        prediction, ctc = ddp(**batch)
        target = torch.randn_like(prediction)
        loss = (prediction - target).square()[batch["mask"]].mean()
        loss = loss + 0.1 * sum(item["z_tilde"].square().mean() for item in ctc.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    checksum = torch.stack([parameter.detach().float().sum() for parameter in ddp.module.parameters()]).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0], rtol=1e-5, atol=1e-4)
    if local_rank == 0:
        print(f"ALL MMAUDIO AVT 4-GPU DDP TESTS PASSED; checksum={checksum.item():.6f}")
    dist.destroy_process_group()


def run_full_ddp_suite() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(666)
    model = _full_model().to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=5.0e-5)
    batch_size, audio_len, video_len, text_len = 2, 400, 100, 48
    audio_mask = torch.ones(batch_size, audio_len, dtype=torch.bool, device=device)
    video_mask = torch.ones(batch_size, video_len, dtype=torch.bool, device=device)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)
    complementary_mask = torch.zeros(batch_size, video_len, dtype=torch.bool, device=device)
    complementary_mask[0, 0] = True
    torch.cuda.reset_peak_memory_stats()

    for step in range(2):
        generator = torch.Generator(device=device).manual_seed(8000 + 10 * local_rank + step)
        batch = {
            "x": torch.randn(batch_size, audio_len, 80, generator=generator, device=device),
            "cond": torch.randn(batch_size, audio_len, 80, generator=generator, device=device),
            "text": torch.randint(0, 159, (batch_size, text_len), generator=generator, device=device),
            "video": torch.randn(batch_size, video_len, 1024, generator=generator, device=device),
            "time": torch.rand(batch_size, generator=generator, device=device),
            "mask": audio_mask,
            "text_mask": text_mask,
            "video_mask": video_mask,
            "complementary_mask": complementary_mask,
            "generation_mask": audio_mask,
            "cache": False,
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, ctc = ddp(**batch)
            target = torch.randn(prediction.shape, generator=generator, device=device)
            loss = (prediction.float() - target).square().mean()
            loss = loss + 0.1 * sum(item["z_tilde"].float().square().mean() for item in ctc.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    checksum = torch.stack([parameter.detach().float().sum() for parameter in ddp.module.parameters()]).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0], rtol=1e-5, atol=1e-3)
    peak_gib = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
    dist.reduce(peak_gib, dst=0, op=dist.ReduceOp.MAX)
    if local_rank == 0:
        print(
            "ALL FULL-WIDTH MMAUDIO AVT 4-GPU DDP TESTS PASSED; "
            f"checksum={checksum.item():.6f}; max_peak={peak_gib.item():.3f} GiB"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ddp", action="store_true")
    mode.add_argument("--ddp-full", action="store_true")
    args = parser.parse_args()
    if args.ddp_full:
        run_full_ddp_suite()
    elif args.ddp:
        run_ddp_suite()
    else:
        run_cpu_suite()
