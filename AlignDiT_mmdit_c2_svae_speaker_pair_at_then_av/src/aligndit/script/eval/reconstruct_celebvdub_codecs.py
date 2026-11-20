"""Reconstruct the CelebV-Dub Setting-1 test set with codec-only baselines.

This script measures the representation/decoder ceiling without running
AlignDiT.  All reconstructed waveforms are cropped or right-padded to the
exact 16 kHz ground-truth length so downstream AVSync comparisons use the
same time span.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from tqdm import tqdm

from aligndit.script.eval.utils import load_vocoder


SAMPLE_RATE = 16_000
MEL_CHANNELS = 80
VAE_CHECKPOINT_DIRS = {
    "acoustic_vae_dim64": "acoustic_vae_dim64",
    "semantic_vae_600k": "semantic_vae",
    "semantic_vae_1000k": "semantic_vae_1000k",
}
MINGTOK_CODEC = "mingtok_acoustic_64d"
CODEC_CHOICES = ("mel_hifigan", *VAE_CHECKPOINT_DIRS, MINGTOK_CODEC)


def get_args():
    root_prefix = os.environ.get("ROOT_PREFIX", "")
    workspace = Path(f"{root_prefix}/zjw524/projects/alignDiT_idea6")
    data_root = Path(f"{root_prefix}/zjw524/projects/data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=list(CODEC_CHOICES),
        choices=CODEC_CHOICES,
        help="Codec baselines to reconstruct. Models are loaded one at a time.",
    )
    parser.add_argument("--manifest", type=Path, default=data_root / "celebvdub_test_s1.lst")
    parser.add_argument("--dataset-root", type=Path, default=data_root / "CelebVDub")
    parser.add_argument("--output-root", type=Path, default=data_root / "codec_ceiling_celebvdub")
    parser.add_argument(
        "--hifigan-checkpoint",
        type=Path,
        default=workspace / "my_papers_code/hifigan_16k_LRS3/g_01000000",
    )
    parser.add_argument("--semantic-vae-repo", type=Path, default=workspace / "papers_codes/Semantic-VAE")
    parser.add_argument("--semantic-vae-weights-root", type=Path, default=workspace / "Semantic-VAE")
    parser.add_argument(
        "--mingtok-repo",
        type=Path,
        default=workspace / "MingTok-VAE/paper_code/MingTok-Audio",
    )
    parser.add_argument(
        "--mingtok-checkpoint",
        type=Path,
        default=workspace / "MingTok-VAE/checkpoint/MingTok-Audio",
    )
    parser.add_argument(
        "--mingtok-attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help="Qwen2 attention backend. Eager preserves MingTok's 32-frame sliding window explicitly.",
    )
    parser.add_argument(
        "--latent-mode",
        choices=("mean", "sample"),
        default="sample",
        help="Use posterior mean or a fixed per-utterance posterior sample for VAE reconstruction.",
    )
    parser.add_argument("--sample-seed", type=int, default=666)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N manifest entries.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None):
    utterances = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        utterances = utterances[:limit]
    if not utterances:
        raise ValueError(f"No utterances found in {path}")
    return utterances


def load_audio(path: Path):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform


def match_length(waveform: torch.Tensor, target_length: int):
    native_length = waveform.shape[-1]
    if native_length < target_length:
        waveform = F.pad(waveform, (0, target_length - native_length))
    else:
        waveform = waveform[..., :target_length]
    return waveform, native_length


def snr_db(reference: torch.Tensor, estimate: torch.Tensor):
    reference = reference.float()
    estimate = estimate.float()
    signal_power = reference.square().mean()
    noise_power = (reference - estimate).square().mean()
    return float(10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def stable_sample_seed(base_seed: int, utterance: str):
    digest = hashlib.sha256(f"{base_seed}:{utterance}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def load_semantic_vae(repo: Path, checkpoint_dir: Path, device: torch.device):
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from dac.model.dac import DAC

    metainfo = json.loads((checkpoint_dir / "metainfo.json").read_text())
    model_config = dict(metainfo["DAC"])
    bigvgan_config = Path(model_config["bigvgan_conf"])
    if not bigvgan_config.is_absolute():
        bigvgan_config = repo / bigvgan_config
    model_config["bigvgan_conf"] = str(bigvgan_config)

    model = DAC(**model_config)
    del model.projectors

    checkpoint = torch.load(
        checkpoint_dir / "dac/ema_state_dict.pth",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = checkpoint.get("state_dict", checkpoint)
    normalized = {}
    ignored = []
    model_state = model.state_dict()
    for raw_key, value in checkpoint.items():
        key = raw_key.removeprefix("ema_model.")
        if key in {"initted", "step"} or key.startswith("projectors."):
            continue
        if key not in model_state:
            if key.startswith("decoder_proj."):
                ignored.append(key)
                continue
            raise KeyError(f"Unexpected checkpoint key: {raw_key}")
        if model_state[key].shape != value.shape:
            raise ValueError(
                f"Checkpoint shape mismatch for {key}: expected {tuple(model_state[key].shape)}, "
                f"got {tuple(value.shape)}"
            )
        normalized[key] = value

    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Invalid VAE checkpoint: missing={missing}, unexpected={unexpected}")

    model = model.eval().requires_grad_(False).to(device)
    return model, len(ignored)


def load_mel_hifigan(checkpoint: Path, device: torch.device):
    model = load_vocoder(
        vocoder_name="hifigan_16k",
        is_local=True,
        local_path=str(checkpoint),
        device=device,
    )
    return model.eval().requires_grad_(False)


def _load_safetensors_prefix(module, checkpoint: Path, prefix: str):
    """Strictly load one released MingTok submodule without materializing 1.35B parameters."""

    expected = module.state_dict()
    loaded = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for key, target in expected.items():
            checkpoint_key = f"{prefix}.{key}"
            if checkpoint_key not in available:
                raise KeyError(f"Missing MingTok checkpoint tensor: {checkpoint_key}")
            value = handle.get_tensor(checkpoint_key)
            if value.shape != target.shape:
                raise ValueError(
                    f"MingTok shape mismatch for {checkpoint_key}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            loaded[key] = value

    missing, unexpected = module.load_state_dict(loaded, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Invalid MingTok {prefix} state: missing={missing}, unexpected={unexpected}")


def load_mingtok_acoustic(
    repo: Path,
    checkpoint_dir: Path,
    device: torch.device,
    attn_implementation: str,
):
    """Load only the 64D acoustic encoder and low-level decoder.

    The released AudioVAE constructor always creates the unused 629.6M semantic
    branch and imports its mandatory FlashAttention dependency.  The codec
    ceiling uses the public ``decode(latent)`` behavior, so constructing the
    encoder and low-level decoder directly is both equivalent and substantially
    lighter.  Eager attention replaces FlashAttention while preserving the
    released model's explicit 32-frame sliding-window mask.
    """

    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from audio_tokenizer.vae_modules import Decoder, Encoder

    config = json.loads((checkpoint_dir / "config.json").read_text())
    if (
        config["enc_kwargs"].get("input_dim") != 320
        or config["enc_kwargs"].get("hop_size") != 320
        or config["enc_kwargs"].get("latent_dim") != 64
        or config["dec_kwargs"].get("output_dim") != 320
        or config["dec_kwargs"].get("latent_dim") != 64
        or config.get("patch_size") != -1
    ):
        raise RuntimeError("The local MingTok checkpoint is not the expected 64D/50 Hz acoustic model")

    encoder_backbone = dict(config["enc_kwargs"]["backbone"])
    decoder_backbone = dict(config["dec_kwargs"]["backbone"])
    for backbone in (encoder_backbone, decoder_backbone):
        backbone["_attn_implementation"] = attn_implementation
        backbone["attn_implementation"] = attn_implementation

    encoder = Encoder(
        encoder_args=encoder_backbone,
        input_dim=320,
        hop_size=320,
        latent_dim=64,
        patch_size=-1,
    ).to(dtype=torch.bfloat16)
    decoder = Decoder(
        decoder_args=decoder_backbone,
        output_dim=320,
        latent_dim=64,
        semantic_model=None,
        patch_size=-1,
    ).to(dtype=torch.bfloat16)

    weights = checkpoint_dir / "model.safetensors"
    _load_safetensors_prefix(encoder, weights, "encoder")
    _load_safetensors_prefix(decoder, weights, "decoder")

    encoder = encoder.eval().requires_grad_(False).to(device)
    decoder = decoder.eval().requires_grad_(False).to(device)
    return (encoder, decoder)


@torch.inference_mode()
def reconstruct_mel(vocoder, mel_path: Path, device: torch.device):
    mel = torch.from_numpy(np.load(mel_path)).float()
    if mel.ndim != 2:
        raise ValueError(f"Expected a 2-D mel array, got {tuple(mel.shape)} from {mel_path}")
    if mel.shape[-1] == MEL_CHANNELS:
        mel = mel.transpose(0, 1)
    elif mel.shape[0] != MEL_CHANNELS:
        raise ValueError(f"Cannot identify the {MEL_CHANNELS}-channel axis in {tuple(mel.shape)}")
    reconstructed = vocoder(mel.unsqueeze(0).to(device)).squeeze(0)
    return reconstructed


@torch.inference_mode()
def reconstruct_vae(
    model,
    waveform: torch.Tensor,
    utterance: str,
    latent_mode: str,
    base_seed: int,
    device: torch.device,
):
    waveform_batch = waveform.unsqueeze(0).to(device)
    padded = model.preprocess(waveform_batch, SAMPLE_RATE)
    _, mu, log_var, _ = model.encode(padded)
    if latent_mode == "mean":
        latent = mu
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_sample_seed(base_seed, utterance))
        eps = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
        latent = mu + torch.exp(0.5 * log_var) * eps
    return model.decode(latent).squeeze(0)


@torch.inference_mode()
def reconstruct_mingtok(
    model,
    waveform: torch.Tensor,
    utterance: str,
    latent_mode: str,
    base_seed: int,
    device: torch.device,
):
    encoder, decoder = model
    waveform_batch = waveform.to(device=device, dtype=torch.bfloat16)
    encoded, _ = encoder(waveform_batch)
    parameters = encoded.transpose(1, 2)
    mean, raw_scale = parameters.chunk(2, dim=1)
    std = F.softplus(raw_scale) + 1e-4

    if latent_mode == "mean":
        latent = mean
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_sample_seed(base_seed, utterance))
        eps = torch.randn(mean.shape, dtype=mean.dtype, device=device, generator=generator)
        latent = mean + std * eps

    latent = latent.transpose(1, 2)
    expected_frames = (waveform.shape[-1] + 319) // 320
    if latent.shape != (1, expected_frames, 64):
        raise RuntimeError(
            f"Unexpected MingTok latent shape for {utterance}: "
            f"expected={(1, expected_frames, 64)}, got={tuple(latent.shape)}"
        )
    reconstructed = decoder.low_level_reconstruct(latent)
    if reconstructed.shape != (1, 1, expected_frames * 320):
        raise RuntimeError(
            f"Unexpected MingTok waveform shape for {utterance}: "
            f"expected={(1, 1, expected_frames * 320)}, got={tuple(reconstructed.shape)}"
        )
    if not torch.isfinite(reconstructed).all():
        raise FloatingPointError(f"MingTok produced non-finite waveform values for {utterance}")
    return reconstructed.squeeze(0)


def output_name(codec: str, latent_mode: str):
    return codec if codec == "mel_hifigan" else f"{codec}_{latent_mode}"


def read_existing_results(path: Path):
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            results[record["utterance"]] = record
    return results


def summarize_records(codec: str, records: list[dict], elapsed_seconds: float):
    snrs = [record["snr_db"] for record in records]
    native_deltas = [
        record["native_samples"] - record["target_samples"]
        for record in records
        if record["native_samples"] is not None
    ]
    return {
        "codec": codec,
        "samples": len(records),
        "snr_db_mean": float(np.mean(snrs)),
        "snr_db_min": float(np.min(snrs)),
        "snr_db_max": float(np.max(snrs)),
        "native_length_delta_mean": float(np.mean(native_deltas)) if native_deltas else None,
        "native_length_delta_abs_max": int(np.max(np.abs(native_deltas))) if native_deltas else None,
        "elapsed_seconds_this_run": elapsed_seconds,
    }


def reconstruct_codec(args, codec: str, utterances: list[str], device: torch.device):
    name = output_name(codec, args.latent_mode)
    output_dir = args.output_root / name
    results_path = output_dir / "_reconstruction_results.jsonl"
    summary_path = output_dir / "_reconstruction_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = {} if args.overwrite else read_existing_results(results_path)
    results_mode = "w" if args.overwrite else "a"

    if codec == "mel_hifigan":
        model = load_mel_hifigan(args.hifigan_checkpoint, device)
        ignored_checkpoint_keys = 0
    elif codec == MINGTOK_CODEC:
        model = load_mingtok_acoustic(
            args.mingtok_repo,
            args.mingtok_checkpoint,
            device,
            args.mingtok_attn_implementation,
        )
        ignored_checkpoint_keys = 0
    else:
        checkpoint_dir = args.semantic_vae_weights_root / VAE_CHECKPOINT_DIRS[codec]
        model, ignored_checkpoint_keys = load_semantic_vae(args.semantic_vae_repo, checkpoint_dir, device)

    records = []
    start_time = time.monotonic()
    with results_path.open(results_mode) as results_file:
        for utterance in tqdm(utterances, desc=name):
            output_path = output_dir / "test" / f"{utterance}.wav"
            if output_path.exists() and utterance in existing:
                records.append(existing[utterance])
                continue

            audio_path = args.dataset_root / "audio/test" / f"{utterance}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            reference = load_audio(audio_path)

            if codec == "mel_hifigan":
                mel_path = args.dataset_root / "mel_tacotron/test" / f"{utterance}.npy"
                if not mel_path.exists():
                    raise FileNotFoundError(mel_path)
                reconstructed = reconstruct_mel(model, mel_path, device)
            elif codec == MINGTOK_CODEC:
                reconstructed = reconstruct_mingtok(
                    model,
                    reference,
                    utterance,
                    args.latent_mode,
                    args.sample_seed,
                    device,
                )
            else:
                reconstructed = reconstruct_vae(
                    model,
                    reference,
                    utterance,
                    args.latent_mode,
                    args.sample_seed,
                    device,
                )

            reconstructed = reconstructed.detach().float().cpu()
            reconstructed, native_length = match_length(reconstructed, reference.shape[-1])
            reconstructed = reconstructed.clamp(-1.0, 1.0)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                reconstructed,
                SAMPLE_RATE,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            record = {
                "utterance": utterance,
                "codec": name,
                "target_samples": reference.shape[-1],
                "native_samples": native_length,
                "saved_samples": reconstructed.shape[-1],
                "snr_db": snr_db(reference, reconstructed),
            }
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()
            records.append(record)

    elapsed_seconds = time.monotonic() - start_time
    summary = summarize_records(name, records, elapsed_seconds)
    summary["ignored_legacy_checkpoint_keys"] = ignored_checkpoint_keys
    summary["latent_mode"] = None if codec == "mel_hifigan" else args.latent_mode
    summary["sample_seed"] = None if codec == "mel_hifigan" else args.sample_seed
    summary["mingtok_attn_implementation"] = (
        args.mingtok_attn_implementation if codec == MINGTOK_CODEC else None
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    args = get_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    for path in (
        args.manifest,
        args.dataset_root,
        args.hifigan_checkpoint,
        args.semantic_vae_repo,
        args.semantic_vae_weights_root,
        args.mingtok_repo,
        args.mingtok_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    utterances = read_manifest(args.manifest, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [reconstruct_codec(args, codec, utterances, device) for codec in args.codecs]
    combined_path = args.output_root / "_reconstruction_summaries.json"
    combined = {}
    if combined_path.exists():
        for record in json.loads(combined_path.read_text()):
            combined[record["codec"]] = record
    for record in summaries:
        combined[record["codec"]] = record
    combined_path.write_text(json.dumps(list(combined.values()), ensure_ascii=False, indent=2) + "\n")
    print(f"Combined reconstruction summary: {combined_path}")


if __name__ == "__main__":
    main()
