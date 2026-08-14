"""CelebV-Dub Setting 1 inference for the 64-D/40-Hz Semantic-VAE C2 model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH,
    LATENT_DIM,
    SAMPLE_RATE,
    load_semantic_vae_decoder,
    read_json_object,
    sha256_file,
)
from f5_tts.model.utils import get_tokenizer


EXPECTED_POLICY = "semantic-vae40-c2-one-stage-minimal-fix-v2"
EXPECTED_TEST_COUNT = 213


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def atomic_save_waveform(path: Path, waveform: torch.Tensor) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing waveform: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.wav")
    try:
        torchaudio.save(str(temporary), waveform, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_test_records(cache_root: Path, test_list: Path) -> list[dict[str, Any]]:
    manifest = cache_root / "manifests/test.jsonl"
    rows = read_jsonl(manifest)
    if len(rows) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TEST_COUNT} test records, found {len(rows)}")
    by_clip = {}
    for row in rows:
        key = row.get("utterance_key")
        if not isinstance(key, str) or not key.startswith("celebvdub/test/"):
            raise RuntimeError(f"Invalid test utterance key: {key!r}")
        clip = key.removeprefix("celebvdub/test/")
        if clip in by_clip:
            raise RuntimeError(f"Duplicate test clip: {clip}")
        by_clip[clip] = row
    clips = [line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(clips) != EXPECTED_TEST_COUNT or len(set(clips)) != EXPECTED_TEST_COUNT:
        raise RuntimeError("CelebV-Dub Setting 1 list must contain exactly 213 unique clips")
    if set(clips) != set(by_clip):
        raise RuntimeError("The Setting 1 list and Semantic-VAE test manifest select different clips")
    return [by_clip[clip] for clip in clips]


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = read_json_object(path)
    if (
        metadata.get("scope") != "train"
        or metadata.get("feature") != "semantic_vae_posterior_sample_v1"
        or metadata.get("channel_count") != LATENT_DIM
    ):
        raise RuntimeError("Invalid LibriSpeech-train Semantic-VAE normalization contract")
    mean = np.asarray(metadata.get("mean"), dtype=np.float32)
    std = np.asarray(metadata.get("std"), dtype=np.float32)
    if (
        mean.shape != (LATENT_DIM,)
        or std.shape != (LATENT_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise RuntimeError("Invalid Semantic-VAE normalization statistics")
    return mean, std, metadata


def build_model(config_path: Path, checkpoint_path: Path, expected_step: int, device: torch.device) -> CFM_VT:
    config = OmegaConf.load(config_path)
    arch = config.model.arch
    representation = config.model.audio_representation
    required_arch = {
        "depth": 18,
        "n_mm_layers": 12,
        "n_text_layers": 12,
        "audio_video_ratio": 1,
    }
    for key, expected in required_arch.items():
        if int(arch[key]) != expected:
            raise RuntimeError(f"Expected minimal-fix v2 {key}={expected}, got {arch[key]}")
    if not bool(arch.normalize_text_context) or bool(arch.prompt_isolated_ca) or bool(arch.video_rope_scaled):
        raise RuntimeError("The selected config is not the global-text minimal-fix v2 C2 architecture")
    if (
        int(representation.channels) != LATENT_DIM
        or int(representation.frame_rate) != 40
        or int(representation.sample_rate) != SAMPLE_RATE
        or int(representation.hop_length) != HOP_LENGTH
    ):
        raise RuntimeError("The selected config is not the fixed 64-D/40-Hz Semantic-VAE representation")

    vocab_char_map, vocab_size = get_tokenizer(config.datasets.vocab_path, "custom")
    model_cls = get_class(f"aligndit.model.{config.model.backbone}")
    model = CFM_VT(
        transformer=model_cls(**arch, text_num_embeds=vocab_size, mel_dim=LATENT_DIM),
        mel_spec_module=PrecomputedAudioRepresentation(LATENT_DIM, SAMPLE_RATE, HOP_LENGTH),
        num_channels=LATENT_DIM,
        vocab_char_map=vocab_char_map,
        audio_video_ratio=1,
        ctc_lambda=float(config.model.ctc_lambda),
        odeint_kwargs={"method": "euler"},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise RuntimeError("Unsupported AlignDiT checkpoint schema")
    if checkpoint.get("training_policy") != EXPECTED_POLICY:
        raise RuntimeError(f"Checkpoint is not minimal-fix v2: {checkpoint.get('training_policy')!r}")
    if checkpoint.get("update") != expected_step:
        raise RuntimeError(f"Checkpoint update mismatch: {checkpoint.get('update')} != {expected_step}")
    ema_state = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema_state, dict) or not bool(ema_state.get("initted")):
        raise RuntimeError("Checkpoint has no initialized EMA state")
    selected = {
        key.removeprefix("ema_model."): value for key, value in ema_state.items() if key not in {"initted", "step"}
    }
    model.load_state_dict(selected, strict=True)
    del checkpoint, ema_state, selected
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def validate_record_arrays(
    row: dict[str, Any], cache_root: Path, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frames = int(row["latent_frames"])
    original_samples = int(row["original_num_samples"])
    padded_samples = int(row["padded_num_samples"])
    if padded_samples != frames * HOP_LENGTH or not padded_samples - HOP_LENGTH < original_samples <= padded_samples:
        raise RuntimeError(f"Invalid exact-length metadata for {row['utterance_key']}")
    latent = np.load(cache_root / row["latent_relative_path"], allow_pickle=False)
    video = np.load(cache_root / row["video_40hz_relative_path"], allow_pickle=False)
    if latent.shape != (frames, LATENT_DIM) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid latent for {row['utterance_key']}: {latent.shape}/{latent.dtype}")
    if video.shape != (frames, 1024) or video.dtype != np.float32 or not np.isfinite(video).all():
        raise RuntimeError(f"Invalid 40-Hz video for {row['utterance_key']}: {video.shape}/{video.dtype}")
    normalized = ((latent - mean) / std).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise FloatingPointError(f"Non-finite normalized latent for {row['utterance_key']}")
    return normalized, video


def historical_setting1_text(text: str) -> str:
    """Reproduce the old prompt builder's exact English text concatenation."""

    prompt_text = text
    if prompt_text and len(prompt_text[-1].encode("utf-8")) == 1:
        prompt_text += " "
    return prompt_text + " " + text


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Semantic-VAE inference requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    cache_root = args.cache_root.resolve(strict=True)
    records = load_test_records(cache_root, args.test_list.resolve(strict=True))
    if args.max_items is not None:
        if not 0 < args.max_items <= len(records):
            raise ValueError("--max-items must be in [1, 213]")
        records = records[: args.max_items]
    mean, std, normalization = load_normalization(args.normalization.resolve(strict=True))
    cache_spec = read_json_object(cache_root / "state/latents/spec.json")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.rglob("*.wav")):
        raise FileExistsError(f"Output directory already contains wav files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.config.resolve(strict=True), args.checkpoint.resolve(strict=True), args.step, device)
    decoder, decoder_metadata = load_semantic_vae_decoder(
        repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint,
        cache_spec=cache_spec,
        device=device,
    )

    started_at = time.time()
    generated_rows = []
    with torch.inference_mode():
        for row in tqdm(records, desc=f"Semantic-VAE S1 {args.step}"):
            normalized, video = validate_record_arrays(row, cache_root, mean, std)
            frames = int(row["latent_frames"])
            cond = torch.from_numpy(normalized).unsqueeze(0).to(device)
            target_video = torch.from_numpy(video).to(device)
            total_video = torch.cat((torch.zeros_like(target_video), target_video), dim=0).unsqueeze(0)
            generated, _ = model.sample(
                cond=cond,
                text=[historical_setting1_text(str(row["text"]))],
                duration=torch.tensor([2 * frames], device=device),
                video=total_video,
                lens=torch.tensor([frames], device=device),
                steps=args.nfe,
                cfg_strength=args.cfg_text,
                cfg_strength_v=args.cfg_video,
                sway_sampling_coef=args.sway,
                seed=args.seed,
                use_epss=True,
            )
            generated_normalized = generated[:, frames : 2 * frames].float()
            if generated_normalized.shape != (1, frames, LATENT_DIM) or not torch.isfinite(generated_normalized).all():
                raise RuntimeError(f"Invalid generated latent for {row['utterance_key']}")
            mean_tensor = torch.from_numpy(mean).to(device)
            std_tensor = torch.from_numpy(std).to(device)
            generated_raw = generated_normalized * std_tensor + mean_tensor
            waveform = decoder(generated_raw.transpose(1, 2)).squeeze(0).float().cpu()
            padded_samples = int(row["padded_num_samples"])
            original_samples = int(row["original_num_samples"])
            if waveform.shape != (1, padded_samples) or not torch.isfinite(waveform).all():
                raise RuntimeError(
                    f"Semantic-VAE decoder output mismatch for {row['utterance_key']}: {tuple(waveform.shape)}"
                )
            waveform = waveform[:, :original_samples]
            clip = str(row["utterance_key"]).removeprefix("celebvdub/")
            output_path = output_dir / f"{clip}.wav"
            atomic_save_waveform(output_path, waveform)
            generated_rows.append(
                {
                    "utterance_key": row["utterance_key"],
                    "relative_path": output_path.relative_to(output_dir).as_posix(),
                    "samples": original_samples,
                    "sha256": sha256_file(output_path),
                }
            )

    summary = {
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "update": args.step,
            "weights": "EMA",
        },
        "dataset": {
            "cache_root": str(cache_root),
            "count": len(records),
            "normalization": str(args.normalization.resolve()),
            "normalization_sha256": sha256_file(args.normalization),
            "test_list": str(args.test_list.resolve()),
            "test_list_sha256": sha256_file(args.test_list),
        },
        "decoder": decoder_metadata,
        "elapsed_seconds": time.time() - started_at,
        "generation": {
            "cfg_text": args.cfg_text,
            "cfg_video": args.cfg_video,
            "nfe": args.nfe,
            "ode_method": "euler",
            "seed": args.seed,
            "sway": args.sway,
            "text_protocol": "historical CelebV-Dub Setting 1 prompt + two spaces + target",
            "use_epss": True,
        },
        "normalization_contract": {
            "count": normalization.get("count"),
            "feature": normalization.get("feature"),
            "scope": normalization.get("scope"),
        },
        "outputs": generated_rows,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "outputs"}, indent=2))


def parse_args() -> argparse.Namespace:
    workspace = Path(f"{os.environ.get('ROOT_PREFIX', '')}/zjw524/projects/alignDiT_idea6")
    data_root = workspace / "../data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True, choices=[150000, 200000])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[2] / "config/finetune_celebvdub_mm_c2_semantic_vae_minimal_fix.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=data_root / "CelebVDub_svae1000k_sample_seed666_fp32",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=data_root / "LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json",
    )
    parser.add_argument("--test-list", type=Path, default=data_root / "celebvdub_test_s1.lst")
    parser.add_argument("--semantic-vae-repo", type=Path, default=workspace / "papers_codes/Semantic-VAE")
    parser.add_argument(
        "--semantic-vae-checkpoint",
        type=Path,
        default=workspace / "Semantic-VAE/semantic_vae_1000k",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--cfg-text", type=float, default=5.0)
    parser.add_argument("--cfg-video", type=float, default=2.0)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
