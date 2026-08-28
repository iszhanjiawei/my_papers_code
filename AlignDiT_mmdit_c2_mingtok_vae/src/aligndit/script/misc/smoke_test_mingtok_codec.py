"""GPU smoke test for the acoustic-only MingTok codec wrapper."""

from __future__ import annotations

import argparse
import os

import torch

from aligndit.model.mingtok_codec import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MODEL_SHA256,
    MINGTOK_HOP_SIZE,
    MINGTOK_LATENT_DIM,
    MingTokAcousticCodec,
    checkpoint_contract,
    stable_sample_seed,
)


def _rooted(path: str) -> str:
    prefix = os.environ.get("ROOT_PREFIX", "")
    return prefix.rstrip("/") + path if prefix else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-path",
        default=_rooted("/zjw524/projects/alignDiT_idea6/MingTok-VAE/paper_code/MingTok-Audio"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=_rooted("/zjw524/projects/alignDiT_idea6/MingTok-VAE/checkpoint/MingTok-Audio"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--component", choices=("encoder", "decoder", "both"), default="both")
    args = parser.parse_args()

    contract = checkpoint_contract(args.checkpoint_dir)
    assert contract == {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
    }
    print("[OK] exact local checkpoint SHA256 contract")
    assert stable_sample_seed("train/smoke/a.wav") == stable_sample_seed("train/smoke/a.wav", base_seed=666)
    print("[OK] default posterior base seed is explicitly 666")

    load_encoder = args.component in {"encoder", "both"}
    load_decoder = args.component in {"decoder", "both"}
    codec = MingTokAcousticCodec(
        repo_path=args.repo_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        load_encoder=load_encoder,
        load_decoder=load_decoder,
    )
    assert not codec.training
    assert all(not parameter.requires_grad for parameter in codec.parameters())
    codec.train(True)
    assert not codec.training
    print(f"[OK] strict {args.component} load; codec is permanently frozen/eval")

    encoded = None
    if load_encoder:
        # A padded two-item batch exercises ceil(length / 320): 320 -> 1,
        # 321 -> 2. Padding is on the right, as required by the causal encoder.
        time = torch.arange(321, dtype=torch.float32) / 16_000
        first = torch.sin(2 * torch.pi * 220 * time)
        second = torch.sin(2 * torch.pi * 330 * time)
        waveform = torch.stack((first, second), dim=0)
        lengths = torch.tensor([320, 321], dtype=torch.long)
        seeds = [
            stable_sample_seed("train/smoke/a.wav"),
            stable_sample_seed("train/smoke/b.wav"),
        ]
        sampled_a, frame_lengths_a = codec.encode(waveform, lengths, mode="sample", seeds=seeds)
        sampled_b, frame_lengths_b = codec.encode(waveform, lengths, mode="sample", seeds=seeds)
        mean_a, frame_lengths_mean = codec.encode(waveform, lengths, mode="mean")
        mean_b, _ = codec.encode(waveform, lengths, mode="mean")

        assert tuple(sampled_a.shape) == (2, 2, MINGTOK_LATENT_DIM)
        assert frame_lengths_a.tolist() == [1, 2]
        assert torch.equal(frame_lengths_a, frame_lengths_b)
        assert torch.equal(frame_lengths_a, frame_lengths_mean)
        assert torch.equal(sampled_a, sampled_b), "fixed posterior seeds must be bit-identical"
        assert torch.equal(mean_a, mean_b), "posterior mean must be deterministic"
        assert not torch.equal(sampled_a, mean_a), "sample mode unexpectedly equals posterior mean"
        assert torch.isfinite(sampled_a).all() and torch.isfinite(mean_a).all()
        encoded = mean_a
        print("[OK] raw [B,T,64] encode, 50-Hz ceil lengths, deterministic fixed posterior sample")

    if load_decoder:
        if encoded is None:
            generator = torch.Generator(device=args.device)
            generator.manual_seed(1234)
            encoded = torch.randn(
                (1, 2, MINGTOK_LATENT_DIM),
                generator=generator,
                device=args.device,
                dtype=torch.float32,
            )
        waveform = codec.decode(encoded)
        expected = (encoded.shape[0], 1, encoded.shape[1] * MINGTOK_HOP_SIZE)
        assert tuple(waveform.shape) == expected
        assert torch.isfinite(waveform).all()
        print(f"[OK] raw latent decode -> waveform {expected} at 16 kHz")

    print("MingTok acoustic codec smoke test passed.")


if __name__ == "__main__":
    main()
