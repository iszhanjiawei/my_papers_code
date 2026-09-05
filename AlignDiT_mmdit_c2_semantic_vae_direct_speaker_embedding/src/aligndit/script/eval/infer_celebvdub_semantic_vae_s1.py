"""CelebV-Dub Setting 1 inference for Semantic-VAE C2 with optional CAM++.

Setting 1 uses the same clip's full GT audio as its reference prompt. Speaker
conditioning, when configured, comes from that exact prompt waveform as well;
this protocol must not be described as evaluation with an independent prompt.
"""

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
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from tqdm import tqdm

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from aligndit.model.speaker_embedding import (
    load_speaker_embedding,
    speaker_embedding_path,
    validate_speaker_cache_metadata,
)
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH,
    LATENT_DIM,
    SAMPLE_RATE,
    load_semantic_vae_decoder,
    read_json_object,
    sha256_file,
)
from f5_tts.model.utils import get_tokenizer


EXPECTED_MINIMAL_FIX_POLICY = "semantic-vae40-c2-one-stage-minimal-fix-v2"
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


def load_composed_config(config_path: Path):
    """Load both standalone and defaults-based Hydra experiment configs."""

    with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
        return compose(config_name=config_path.stem)


def build_model(config_path: Path, checkpoint_path: Path, expected_step: int, device: torch.device) -> CFM_VT:
    config = load_composed_config(config_path)
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
            raise RuntimeError(f"Expected Semantic-VAE C2 {key}={expected}, got {arch[key]}")
    if bool(arch.prompt_isolated_ca) or bool(arch.video_rope_scaled):
        raise RuntimeError("The selected config is not the global-text, shared-40-Hz-RoPE C2 architecture")
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
    checkpoint_schema = checkpoint.get("checkpoint_schema_version")
    training_policy = checkpoint.get("training_policy")
    if checkpoint_schema is not None or training_policy is not None:
        if checkpoint_schema != 1 or training_policy != EXPECTED_MINIMAL_FIX_POLICY:
            raise RuntimeError(
                "Unsupported guarded checkpoint contract: "
                f"schema={checkpoint_schema!r}, policy={training_policy!r}"
            )
    else:
        expected_keys = {
            "ema_model_state_dict",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "update",
        }
        if set(checkpoint) != expected_keys:
            raise RuntimeError(
                f"Unsupported historical Direct-C2 checkpoint keys: {sorted(set(checkpoint) - expected_keys)}"
            )
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


def load_setting1_speaker_embeddings(
    config,
    records: list[dict[str, Any]],
) -> tuple[list[torch.Tensor | None], dict[str, Any] | None]:
    """Read CAM++ only from the waveform used by the existing S1 prompt.

    All selected vectors are validated before loading the large inference
    models, so an incomplete or mismatched cache fails without partial output.
    """
    speaker_dim = config.model.arch.get("speaker_dim")
    if speaker_dim is None:
        return [None] * len(records), None
    datasets = config.datasets
    expected_dim = int(datasets.get("speaker_embedding_dim", 192))
    if int(speaker_dim) != expected_dim:
        raise ValueError("model speaker_dim and dataset speaker_embedding_dim must agree")
    configured_cache = datasets.get("speaker_embedding_cache_dir")
    if not configured_cache:
        raise ValueError("speaker_embedding_cache_dir is required for speaker-conditioned inference")
    cache_dir = Path(configured_cache).resolve(strict=True)
    metadata = validate_speaker_cache_metadata(
        cache_dir,
        expected_dim=expected_dim,
        model_id=datasets.get("speaker_embedding_model_id"),
        checkpoint_sha256=datasets.get("speaker_embedding_checkpoint_sha256"),
    )
    audio_root = Path(
        datasets.get(
            "speaker_audio_root",
            f"{os.environ.get('ROOT_PREFIX', '')}/zjw524/projects/data/CelebVDub/audio",
        )
    ).resolve(strict=True)
    embeddings = []
    sources = []
    for row in records:
        audio_relative_path = Path(str(row["audio_relative_path"]))
        expected_relative = Path(str(row["utterance_key"]).removeprefix("celebvdub/") + ".wav")
        if (
            audio_relative_path != expected_relative
            or audio_relative_path.is_absolute()
            or ".." in audio_relative_path.parts
            or audio_relative_path.parts[0] != "test"
        ):
            raise ValueError(f"S1 reference audio does not match its prompt: {row['utterance_key']}")
        # The original waveform is not re-encoded from VAE reconstructions.
        prompt_waveform = audio_root / audio_relative_path
        embeddings.append(
            load_speaker_embedding(prompt_waveform, cache_dir, expected_dim=expected_dim, audio_root=audio_root)
        )
        sources.append(
            {
                "utterance_key": row["utterance_key"],
                "prompt_audio": str(prompt_waveform),
                "speaker_cache": str(speaker_embedding_path(prompt_waveform, cache_dir, audio_root=audio_root)),
            }
        )
    return embeddings, {
        "cache_dir": str(cache_dir),
        "metadata_sha256": sha256_file(cache_dir / "metadata.json"),
        "model_id": metadata["model_id"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "dim": expected_dim,
        "source_audio": metadata["source_audio"],
        "reference_protocol": "CelebV-Dub Setting 1: prompt and target are the same GT clip",
        "conditioning": "full/TTS branches keep prompt speaker; null branch drops prompt and speaker jointly",
        "sources": sources,
    }


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
    config = load_composed_config(args.config.resolve(strict=True))
    speaker_embeddings, speaker_metadata = load_setting1_speaker_embeddings(config, records)

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
        for row, speaker_embedding in tqdm(
            zip(records, speaker_embeddings), total=len(records), desc=f"Semantic-VAE S1 {args.step}"
        ):
            normalized, video = validate_record_arrays(row, cache_root, mean, std)
            frames = int(row["latent_frames"])
            cond = torch.from_numpy(normalized).unsqueeze(0).to(device)
            target_video = torch.from_numpy(video).to(device)
            total_video = torch.cat((torch.zeros_like(target_video), target_video), dim=0).unsqueeze(0)
            speaker_kwargs = {}
            if speaker_embedding is not None:
                speaker_kwargs["speaker_embedding"] = speaker_embedding.unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
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
                **speaker_kwargs,
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
        "speaker_embedding": speaker_metadata,
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
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[2]
        / "config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml",
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
