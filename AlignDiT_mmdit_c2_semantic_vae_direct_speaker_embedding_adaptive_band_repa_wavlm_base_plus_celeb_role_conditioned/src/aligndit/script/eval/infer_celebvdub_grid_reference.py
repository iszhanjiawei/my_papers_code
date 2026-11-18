"""Frozen-checkpoint CelebV-Dub inference with independent GRID audio prompts.

The target waveform and its acoustic latent are never loaded by this entry point.
Target lengths follow the existing, known-target-duration evaluation manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from aligndit.model.speaker_embedding import validate_speaker_embedding_array
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    atomic_save_waveform,
    build_model,
    load_composed_config,
    load_normalization,
    load_test_records,
    read_jsonl,
)
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH,
    LATENT_DIM,
    load_semantic_vae_decoder,
    read_json_object,
    sha256_file,
)


def reference_target_text(reference: str, target: str) -> str:
    """Use the historical English prompt separator with independent transcripts."""
    if not reference.strip() or not target.strip():
        raise ValueError("Both reference and target transcripts must be nonempty")
    if len(reference[-1].encode("utf-8")) == 1:
        reference += " "
    return reference + " " + target


def make_sampling_inputs(reference: np.ndarray, video: np.ndarray, text: str, mean, std):
    """Align both modalities on one 40-Hz grid, including any sampler text tail."""
    if reference.ndim != 2 or reference.shape[1] != LATENT_DIM or len(reference) == 0:
        raise ValueError("Expected nonempty GRID latent [P,64]")
    if video.ndim != 2 or video.shape[1] != 1024 or len(video) == 0:
        raise ValueError("Expected nonempty target video [T,1024]")
    if not np.isfinite(reference).all() or not np.isfinite(video).all():
        raise ValueError("Non-finite reference/video feature")
    prompt_frames, target_frames = len(reference), len(video)
    # The char tokenizer emits one token per character, including unknown chars.
    # Match CFM_VT.sample's max(text length, prompt length)+1 lower bound exactly.
    requested_duration = prompt_frames + target_frames
    effective_duration = max(requested_duration, len(text) + 1, prompt_frames + 1)
    if effective_duration > 4096:
        raise ValueError("The combined prompt/target exceeds the sampler's 4096-frame limit")
    combined_video = np.zeros((effective_duration, 1024), dtype=np.float32)
    combined_video[prompt_frames:requested_duration] = video
    normalized = ((reference - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized reference")
    return normalized, combined_video, prompt_frames, target_frames, effective_duration


def load_pairs(args, config):
    manifest = args.pair_manifest.resolve(strict=True)
    metadata = read_json_object(manifest.parent / "metadata.json")
    if metadata.get("status") != "complete" or metadata.get("count") != 213:
        raise ValueError("GRID reference cache is not complete for 213 pairs")
    required = {
        "manifest_sha256": sha256_file(manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "speaker_checkpoint_sha256": str(config.datasets.speaker_embedding_checkpoint_sha256),
    }
    cache_spec = read_json_object(args.cache_root / "state/latents/spec.json")
    required["semantic_vae_checkpoint_sha256"] = cache_spec["checkpoint"]["ema_sha256"]
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"GRID cache metadata mismatch: {key}")
    targets = load_test_records(args.cache_root, args.test_list)
    pairs = read_jsonl(manifest)
    if len(pairs) != len(targets) or len({p["pair_id"] for p in pairs}) != 213:
        raise ValueError("Expected exactly 213 unique GRID-target pairs")
    if len({p["target_id"] for p in pairs}) != 213:
        raise ValueError("Each target must appear exactly once")
    for pair, target in zip(pairs, targets):
        target_id = target["utterance_key"].removeprefix("celebvdub/")
        expected = {
            "target_id": target_id,
            "target_text": target["text"],
            "target_frames": target["latent_frames"],
            "target_num_samples": target["original_num_samples"],
        }
        for key, value in expected.items():
            if pair.get(key) != value:
                raise ValueError(f"Pair differs from the target manifest: {key}, {target_id}")
        if Path(target_id).is_absolute() or ".." in Path(target_id).parts:
            raise ValueError(f"Unsafe target identifier: {target_id}")
        expected_video = (args.cache_root / target["video_40hz_relative_path"]).resolve(strict=True)
        if Path(pair["target_video_path"]).resolve(strict=True) != expected_video:
            raise ValueError(f"Wrong target video: {target_id}")
        expected_audio = (args.target_audio_root / target["audio_relative_path"]).resolve(strict=True)
        if Path(pair["target_gt_audio"]).resolve(strict=True) != expected_audio:
            raise ValueError(f"Wrong evaluation-only target audio: {target_id}")
        if not pair["ref_id"].startswith("grid/") or not pair["ref_text"].strip():
            raise ValueError(f"Invalid GRID reference: {target_id}")
        ref_parts = pair["ref_id"].split("/")
        source = Path(pair["ref_source_audio"])
        if (len(ref_parts) != 3 or ref_parts[1] != pair["ref_speaker_id"]
                or source.parent.name != ref_parts[1] or source.stem != ref_parts[2]):
            raise ValueError(f"Inconsistent GRID identity: {pair['ref_id']}")
        if Path(pair["ref_audio"]).resolve() == Path(pair["target_gt_audio"]).resolve():
            raise ValueError("External GRID reference must differ from target audio")
        for path_key, hash_key in (
            ("ref_audio", "ref_audio_sha256"),
            ("ref_latent_path", "ref_latent_sha256"),
            ("ref_speaker_path", "ref_speaker_sha256"),
            ("target_video_path", "target_video_sha256"),
        ):
            if sha256_file(pair[path_key]) != pair[hash_key]:
                raise ValueError(f"Changed pair artifact: {pair[path_key]}")
        # Check all caches before loading the large generation model.
        latent = np.load(pair["ref_latent_path"], allow_pickle=False)
        video = np.load(pair["target_video_path"], allow_pickle=False)
        if latent.shape != (pair["ref_frames"], LATENT_DIM) or latent.dtype != np.float32:
            raise ValueError(f"Invalid GRID latent: {pair['ref_id']}")
        if video.shape != (pair["target_frames"], 1024) or video.dtype != np.float32:
            raise ValueError(f"Invalid target video: {target_id}")
        if not np.isfinite(latent).all() or not np.isfinite(video).all():
            raise ValueError(f"Non-finite features: {pair['pair_id']}")
        validate_speaker_embedding_array(np.load(pair["ref_speaker_path"], allow_pickle=False))
        samples = int(pair["target_num_samples"])
        if not (pair["target_frames"] - 1) * HOP_LENGTH < samples <= pair["target_frames"] * HOP_LENGTH:
            raise ValueError(f"Invalid target duration: {target_id}")
        combined_text = reference_target_text(pair["ref_text"], pair["target_text"])
        if max(pair["ref_frames"] + pair["target_frames"], len(combined_text) + 1) > 4096:
            raise ValueError(f"Pair exceeds the sampler limit: {pair['pair_id']}")
    return pairs, metadata, cache_spec


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Formal inference requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    config = load_composed_config(args.config.resolve(strict=True))
    mean, std, _ = load_normalization(args.normalization.resolve(strict=True))
    pairs, reference_metadata, cache_spec = load_pairs(args, config)
    extensions = [max(0, len(reference_target_text(p["ref_text"], p["target_text"])) + 1
                      - p["ref_frames"] - p["target_frames"]) for p in pairs]
    print(f"Preflight: 213 validated pairs; text-extension count={sum(x > 0 for x in extensions)}, "
          f"max={max(extensions)} frames", flush=True)
    if args.max_items is not None:
        if not 0 < args.max_items <= len(pairs):
            raise ValueError("--max-items must be in [1,213]")
        pairs = pairs[:args.max_items]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty inference output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.config.resolve(), args.checkpoint.resolve(), args.step, device)
    decoder, decoder_metadata = load_semantic_vae_decoder(
        repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint,
        cache_spec=cache_spec,
        device=device,
    )
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    started = time.time()
    rows = []
    with torch.inference_mode():
        for pair in tqdm(pairs, desc=f"GRID-reference EMA {args.step}"):
            reference = np.load(pair["ref_latent_path"], allow_pickle=False)
            video = np.load(pair["target_video_path"], allow_pickle=False)
            text = reference_target_text(pair["ref_text"], pair["target_text"])
            cond, total_video, p, t, total = make_sampling_inputs(reference, video, text, mean, std)
            speaker = np.load(pair["ref_speaker_path"], allow_pickle=False)
            generated, _ = model.sample(
                cond=torch.from_numpy(cond).unsqueeze(0).to(device),
                text=[text], duration=torch.tensor([total], device=device),
                video=torch.from_numpy(total_video).unsqueeze(0).to(device),
                lens=torch.tensor([p], device=device),
                speaker_embedding=torch.from_numpy(speaker).unsqueeze(0).to(device),
                steps=args.nfe, cfg_strength=args.cfg_text, cfg_strength_v=args.cfg_video,
                sway_sampling_coef=args.sway, seed=args.seed, use_epss=True,
            )
            if generated.shape != (1, total, LATENT_DIM):
                raise RuntimeError(f"Sampler changed validated duration: {pair['pair_id']}")
            target = generated[:, p:p + t].float()
            if target.shape != (1, t, LATENT_DIM) or not torch.isfinite(target).all():
                raise RuntimeError(f"Invalid target latent: {pair['pair_id']}")
            raw = target * std_tensor + mean_tensor
            wav = decoder(raw.transpose(1, 2)).squeeze(0).float().cpu()
            if wav.shape != (1, t * HOP_LENGTH) or not torch.isfinite(wav).all():
                raise RuntimeError(f"Invalid generated waveform: {pair['pair_id']}")
            wav = wav[:, :pair["target_num_samples"]]
            path = output / (pair["target_id"] + ".wav")
            atomic_save_waveform(path, wav)
            result = {
                "pair_id": pair["pair_id"], "target_id": pair["target_id"],
                "ref_id": pair["ref_id"], "ref_speaker_id": pair["ref_speaker_id"],
                "relative_path": path.relative_to(output).as_posix(),
                "samples": wav.shape[-1], "sha256": sha256_file(path),
                "prompt_frames": p, "target_frames": t,
                "sampler_frames": total, "text_extension_frames": total - p - t,
            }
            rows.append(result)
            with (output / "generation_records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "celebvdub_grid_reference_one_per_target_v1",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "update": args.step,
                       "weights": "EMA", "sha256": sha256_file(args.checkpoint)},
        "pair_manifest": str(args.pair_manifest.resolve()),
        "pair_manifest_sha256": sha256_file(args.pair_manifest),
        "reference_cache_metadata": reference_metadata,
        "decoder": decoder_metadata, "count": len(rows),
        "partial_smoke_test": len(rows) != 213,
        "generation": {"seed": args.seed, "nfe": args.nfe, "cfg_text": args.cfg_text,
                       "cfg_video": args.cfg_video, "sway": args.sway,
                       "ode_method": "euler", "use_epss": True,
                       "duration_protocol": "known target duration from historical S1 metadata",
                       "reference_protocol": "GRID waveform supplies both audio prompt and CAM++",
                       "text_protocol": "GRID reference transcript + historical separator + target transcript",
                       "target_acoustic_inputs": False},
        "elapsed_seconds": time.time() - started, "outputs": rows,
    }
    (output / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Inference complete: {len(rows)} GRID-reference outputs in {output}", flush=True)


def parse_args():
    root = Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=150000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml")
    parser.add_argument("--cache-root", type=Path, default=root / "data/CelebVDub_svae1000k_sample_seed666_fp32")
    parser.add_argument("--normalization", type=Path, default=root / "data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json")
    parser.add_argument("--test-list", type=Path, default=root / "data/celebvdub_test_s1.lst")
    parser.add_argument("--target-audio-root", type=Path, default=root / "data/CelebVDub/audio")
    parser.add_argument("--semantic-vae-repo", type=Path, default=root / "alignDiT_idea6/papers_codes/Semantic-VAE")
    parser.add_argument("--semantic-vae-checkpoint", type=Path, default=root / "alignDiT_idea6/Semantic-VAE/semantic_vae_1000k")
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
