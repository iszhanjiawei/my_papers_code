"""Leak-free Setting 2 inference for the Semantic-VAE / CAM++ / REPA model.

Target inputs are an explicit video-only feature and an independently cached
visual-recognition transcript. Reference VAE and CAM++ features are recomputed
from the exact different-utterance waveform. No target waveform, acoustic cache,
transcript, or historical S1 manifest is opened.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from f5_tts.model.utils import get_epss_timesteps

from aligndit.model.speaker_embedding import validate_speaker_embedding_array
from aligndit.script.eval.infer_celebvdub_grid_reference import make_sampling_inputs, reference_target_text
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    atomic_save_waveform, build_model, load_composed_config, load_normalization,
)
from aligndit.script.eval.prepare_grid_reference import posterior_stats
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH, LATENT_DIM, SAMPLE_RATE, load_semantic_vae, read_json_object, sha256_file,
)
from aligndit.script.misc.extract_campplus_speaker_embeddings import (
    AudioFeatureDataset, ExtractionItem, MODEL_ID, EXPECTED_CHECKPOINT_SHA256,
    atomic_save_json, atomic_save_npy, load_campplus,
)

FORBIDDEN = {"gt_audio", "gt_audio_path", "gt_text", "target_audio", "target_audio_path",
             "target_text", "target_text_path", "target_latent", "target_speaker_embedding"}


def safe_id(value):
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 3 or p.parts[0] != "test":
        raise ValueError(f"Unsafe sample ID: {value}")
    return p.as_posix()


def rows_from_manifest(args):
    rows, seen = [], set()
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if FORBIDDEN.intersection(raw):
            raise ValueError("Target GT fields are forbidden in inference manifest")
        ident, ref = safe_id(raw["id"]), safe_id(raw["reference_id"])
        if ident in seen or ident == ref or raw["speaker_id"] != raw["reference_speaker_id"]:
            raise ValueError(f"Invalid unique different-utterance same-speaker pair: {ident}")
        seen.add(ident)
        row = {k: raw[k] for k in ("id", "reference_id", "speaker_id", "reference_speaker_id",
                                  "reference_text", "identity_status", "protocol")}
        for field, expected_id in (("reference_audio", ref), ("video_feature", ident)):
            path = Path(raw[field]).expanduser()
            if not path.is_absolute():
                path = args.manifest.parent / path
            path = path.resolve(strict=True)
            if "/".join(path.with_suffix("").parts[-3:]) != expected_id:
                raise ValueError(f"Input path does not match explicit ID: {path}")
            if field == "video_feature" and "avhubert_video_feat" not in path.parts:
                raise ValueError(f"Require audited video-only features: {path}")
            row[field] = str(path)
        row["vsr_text_path"] = str((args.vsr_text_dir / (ident + ".txt")).resolve(strict=True))
        row["vsr_text"] = Path(row["vsr_text_path"]).read_text().strip().lower()
        row["reference_text"] = row["reference_text"].strip().lower()
        if not row["vsr_text"] or not row["reference_text"]:
            raise ValueError(f"Empty reference or VSR transcript: {ident}")
        video = np.load(row["video_feature"], allow_pickle=False)
        if video.ndim != 2 or video.shape[1] != 1024 or len(video) == 0 or not np.isfinite(video).all():
            raise ValueError(f"Invalid video-only feature: {ident}")
        row["video_frames_25hz"] = len(video)
        row["target_samples"] = len(video) * SAMPLE_RATE // 25
        row["target_frames"] = math.ceil(row["target_samples"] / HOP_LENGTH)
        row["input_sha256"] = {k: sha256_file(row[k]) for k in
                               ("reference_audio", "video_feature", "vsr_text_path")}
        rows.append(row)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty inference manifest")
    return rows


def posterior_seed(ref_id):
    digest = hashlib.sha256(f"666:celebvdub/{ref_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def source_hashes(project):
    paths = list((project / "src/aligndit/model").rglob("*.py"))
    paths += list((project / "src/f5_tts/model").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("infer_celebvdub_grid_reference.py"),
              Path(__file__).with_name("infer_celebvdub_semantic_vae_s1.py"),
              Path(__file__).with_name("semantic_vae_decoder.py"),
              Path(__file__).with_name("prepare_grid_reference.py"),
              project / "src/aligndit/script/misc/extract_campplus_speaker_embeddings.py"]
    return {str(p.relative_to(project)): sha256_file(p) for p in sorted(set(paths))}


def verify_reference(record, row, cache_contract):
    if record["reference_id"] != row["reference_id"] or record["reference_audio_sha256"] != row["input_sha256"]["reference_audio"]:
        raise ValueError("Reference cache ID or waveform changed")
    if record["contract"] != cache_contract:
        raise ValueError("Reference cache extraction contract changed")
    for key in ("latent", "speaker"):
        if sha256_file(record[key + "_path"]) != record[key + "_sha256"]:
            raise ValueError(f"Reference cache hash changed: {key}")
    latent = np.load(record["latent_path"], allow_pickle=False)
    if latent.dtype != np.float32 or latent.shape != (record["frames"], LATENT_DIM) or not np.isfinite(latent).all():
        raise ValueError("Invalid reference VAE latent")
    validate_speaker_embedding_array(np.load(record["speaker_path"], allow_pickle=False))


@torch.inference_mode()
def prepare_references(args, rows, vae, campplus, device, contract):
    records = {}
    for row in rows:
        ref = row["reference_id"]
        if ref in records:
            continue
        path = args.reference_cache / (ref + ".json")
        if path.exists():
            record = read_json_object(path)
            verify_reference(record, row, contract)
        else:
            wave, rate = torchaudio.load(row["reference_audio"])
            if rate != SAMPLE_RATE or wave.ndim != 2 or wave.shape[0] != 1:
                raise ValueError("Expected canonical mono 16 kHz CelebVDub reference")
            if not torch.isfinite(wave).all() or not wave.square().mean() > 0:
                raise ValueError("Nonfinite or silent reference")
            frames = math.ceil(wave.shape[-1] / HOP_LENGTH)
            padded = F.pad(wave.unsqueeze(0), (0, frames * HOP_LENGTH - wave.shape[-1]))
            mu, logvar = posterior_stats(vae, padded.to(device))
            generator = torch.Generator(device=device).manual_seed(posterior_seed(ref))
            noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
            latent = (mu + torch.exp(0.5 * logvar) * noise).squeeze(0).float().cpu().numpy()
            item = ExtractionItem(row["reference_audio"], "", "setting2_reference")
            _, features, error = AudioFeatureDataset([item])[0]
            if error:
                raise RuntimeError(error)
            speaker = campplus(features.to(device)).float().cpu().mean(0)
            speaker = F.normalize(speaker, dim=0).numpy()
            validate_speaker_embedding_array(speaker)
            latent_path = args.reference_cache / "latents" / (ref + ".npy")
            speaker_path = args.reference_cache / "speakers" / (ref + ".npy")
            atomic_save_npy(latent_path, latent)
            atomic_save_npy(speaker_path, speaker)
            record = {"reference_id": ref, "reference_audio": row["reference_audio"],
                      "reference_audio_sha256": row["input_sha256"]["reference_audio"],
                      "posterior_seed": posterior_seed(ref), "frames": frames,
                      "samples": wave.shape[-1], "latent_path": str(latent_path),
                      "speaker_path": str(speaker_path), "contract": contract,
                      "latent_sha256": sha256_file(latent_path), "speaker_sha256": sha256_file(speaker_path)}
            verify_reference(record, row, contract)
            atomic_save_json(path, record)
        records[ref] = record
        print(f"Reference {len(records)} prepared: {ref}", flush=True)
    return records


def run(args):
    project = Path(__file__).resolve().parents[4]
    cfg = load_composed_config(args.config)
    rows = rows_from_manifest(args)
    print(f"Preflight: {len(rows)} explicit Setting 2 pairs and visual-only transcripts", flush=True)
    if args.dry_run:
        return
    mean, std, _ = load_normalization(Path(cfg.datasets.normalization_path))
    normalization_sha = sha256_file(cfg.datasets.normalization_path)
    if normalization_sha != cfg.datasets.expected_normalization_sha256:
        raise ValueError("Training normalization hash mismatch")
    if sha256_file(cfg.datasets.vocab_path) != cfg.datasets.expected_vocab_sha256:
        raise ValueError("Training vocabulary hash mismatch")
    speaker_sha = sha256_file(args.speaker_checkpoint)
    if (speaker_sha != EXPECTED_CHECKPOINT_SHA256 or speaker_sha != cfg.datasets.speaker_embedding_checkpoint_sha256
            or cfg.datasets.speaker_embedding_model_id != MODEL_ID or int(cfg.model.arch.speaker_dim) != 192):
        raise ValueError("Training CAM++ contract mismatch")
    spec = read_json_object(Path(cfg.datasets.cache_root) / "state/latents/spec.json")
    if spec["extraction"]["protocol"] != "semantic_vae_posterior_sample_v1":
        raise ValueError("Unsupported Semantic-VAE posterior protocol")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    vae, vae_meta = load_semantic_vae(repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint, cache_spec=spec, device=device)
    campplus = load_campplus(args.speaker_checkpoint, device)
    contract = {"vae": vae_meta, "speaker_checkpoint_sha256": speaker_sha,
                "latent_protocol": "posterior_sample_v1_fp32_seed666_logvar_clamp_-12_12",
                "posterior_seed_formula": "SHA256(666:celebvdub/reference_id)[:8]_little_mod_2**63-1",
                "padding": "zero_right_to_400_samples", "speaker": "CAM++_10s_repeat_mean_L2",
                "torch": torch.__version__, "source_sha256": source_hashes(project)}
    with torch.inference_mode():
        posterior_stats(vae, torch.zeros(1, 1, HOP_LENGTH, device=device))
    references = prepare_references(args, rows, vae, campplus, device, contract)
    decoder = vae.decoder
    del vae, campplus
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    model = build_model(args.config, args.checkpoint, args.step, device)
    print(f"Strict initialized EMA model load passed, update={args.step}", flush=True)
    time_grid = get_epss_timesteps(args.nfe, device=device, dtype=torch.float32)
    time_grid = time_grid + args.sway * (torch.cos(torch.pi / 2 * time_grid) - 1 + time_grid)
    config = {"method": "Ours_150k", "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_update": args.step,
              "use_ema": True, "config": str(args.config), "config_sha256": sha256_file(args.config),
              "resolved_config": OmegaConf.to_container(cfg, resolve=True),
              "source_sha256": source_hashes(project), "decoder": vae_meta,
              "normalization_sha256": normalization_sha,
              "speaker_checkpoint_sha256": speaker_sha, "manifest_sha256": sha256_file(args.manifest),
              "manifest": str(args.manifest), "vsr_text_dir": str(args.vsr_text_dir),
              "steps": args.nfe, "seed": args.seed, "ode_method": "euler", "use_epss": True,
              "sway": args.sway, "cfg_t": args.cfg_text, "cfg_v": args.cfg_video,
              "dtype": "float32", "torch": torch.__version__,
              "sampling_time_grid": time_grid.cpu().tolist(), "sway_applied": True,
              "epss_32_step_behavior": "linspace fallback then sway; identical to baseline 32-step grid",
              "duration_source": "video-only 25Hz frames * 640 samples; ceil to 40Hz then crop",
              "video_interpolation": "linear_align_corners_false_25_to_40Hz",
              "text_source": "LipVoicer_visual_only_VSR; reference transcript from explicit reference",
              "reference_source": "exact explicit reference waveform for both VAE posterior and CAM++",
              "target_gt_audio_or_text_read": False, "target_acoustic_cache_read": False,
              "repa_teacher_inference": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and read_json_object(config_path) != config:
        raise ValueError("Existing output provenance differs; use a new output directory")
    atomic_save_json(config_path, config)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    outputs = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        ref = references[row["reference_id"]]
        path = args.output_dir / (row["id"] + ".wav")
        meta_path = path.with_suffix(".json")
        inputs = {"row": row, "reference": ref, "run_config_sha256": sha256_file(config_path)}
        if path.exists() or meta_path.exists():
            previous = read_json_object(meta_path)
            if previous["inputs"] != inputs or sha256_file(path) != previous["audio_sha256"]:
                raise ValueError(f"Existing sample provenance mismatch: {row['id']}")
            outputs.append(previous)
            print(f"[{index}/{len(rows)}] verified existing {row['id']}", flush=True)
        else:
            video25 = torch.from_numpy(np.load(row["video_feature"], allow_pickle=False)).float()
            video = F.interpolate(video25.T.unsqueeze(0), size=row["target_frames"],
                                  mode="linear", align_corners=False)[0].T.numpy()
            latent = np.load(ref["latent_path"], allow_pickle=False)
            speaker = np.load(ref["speaker_path"], allow_pickle=False)
            text = reference_target_text(row["reference_text"], row["vsr_text"])
            cond, full_video, p, t, total = make_sampling_inputs(latent, video, text, mean, std)
            with torch.inference_mode():
                generated, _ = model.sample(cond=torch.from_numpy(cond)[None].to(device), text=[text],
                    duration=torch.tensor([total], device=device), video=torch.from_numpy(full_video)[None].to(device),
                    lens=torch.tensor([p], device=device), speaker_embedding=torch.from_numpy(speaker)[None].to(device),
                    steps=args.nfe, cfg_strength=args.cfg_text, cfg_strength_v=args.cfg_video,
                    sway_sampling_coef=args.sway, seed=args.seed, use_epss=True)
                if generated.shape != (1, total, LATENT_DIM) or not torch.isfinite(generated).all():
                    raise ValueError("Invalid generated latent shape/content")
                raw = generated[:, p:p+t].float() * std_t + mean_t
                wave = decoder(raw.transpose(1, 2)).squeeze(0).float().cpu()
            if wave.shape != (1, t * HOP_LENGTH) or not torch.isfinite(wave).all() or not wave.square().mean() > 0:
                raise ValueError("Invalid decoded waveform shape/content")
            wave = wave[:, :row["target_samples"]]
            atomic_save_waveform(path, wave)
            result = {"id": row["id"], "inputs": inputs, "audio_sha256": sha256_file(path),
                      "samples": wave.shape[-1], "sample_rate": SAMPLE_RATE, "prompt_frames": p,
                      "target_frames": t, "sampler_frames": total, "text_extension_frames": total-p-t,
                      "rms": float(wave.square().mean().sqrt()), "peak": float(wave.abs().max()),
                      "generation_seconds": time.monotonic()-started}
            atomic_save_json(meta_path, result)
            outputs.append(result)
            print(f"[{index}/{len(rows)}] generated {row['id']} in {result['generation_seconds']:.1f}s", flush=True)
        atomic_save_json(args.output_dir / "progress.json", {"requested": len(rows), "complete": len(outputs), "failures": []})
    summary = {"requested": len(rows), "complete": len(outputs), "failures": [],
               "run_config_sha256": sha256_file(config_path),
               "outputs": [{k: item[k] for k in ("id", "audio_sha256", "samples", "sample_rate")} for item in outputs]}
    atomic_save_json(args.output_dir / "inference_summary.json", summary)
    print(f"Setting 2 generation complete: {len(outputs)}/{len(rows)}", flush=True)


def parse_args():
    root = Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects")
    project = Path(__file__).resolve().parents[4]
    bench = root / "alignDiT_idea6/Video-to-Speech-benchmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=bench / "setting2/inference.jsonl")
    parser.add_argument("--vsr-text-dir", type=Path, default=bench / "cache/lipvoicer_vsr_text")
    parser.add_argument("--output-dir", type=Path, default=bench / "results/Ours_150k")
    parser.add_argument("--reference-cache", type=Path, default=bench / "cache/ours_svae_campplus_setting2")
    parser.add_argument("--config", type=Path, default=project / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=150000)
    parser.add_argument("--semantic-vae-repo", type=Path, default=root / "alignDiT_idea6/papers_codes/Semantic-VAE")
    parser.add_argument("--semantic-vae-checkpoint", type=Path, default=root / "alignDiT_idea6/Semantic-VAE/semantic_vae_1000k")
    parser.add_argument("--speaker-checkpoint", type=Path, default=root / "data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt")
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/alignDiT_idea6_vts_gpu0.lock"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--sway", type=float, default=-1)
    parser.add_argument("--cfg-text", type=float, default=5)
    parser.add_argument("--cfg-video", type=float, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.nfe < 1:
        parser.error("Invalid --limit or --nfe")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.dry_run:
        run(args)
    else:
        with args.gpu_lock.open("a") as lock:
            print(f"Waiting for GPU lock: {args.gpu_lock}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("Acquired GPU lock", flush=True)
            run(args)
