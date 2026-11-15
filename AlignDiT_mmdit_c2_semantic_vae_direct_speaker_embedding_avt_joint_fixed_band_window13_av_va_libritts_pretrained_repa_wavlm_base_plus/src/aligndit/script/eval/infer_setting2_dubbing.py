"""Original 115-pair GT-dialogue Setting2 inference for the fixed13 AVT snapshot.

Uses the historical cross-utterance SVAE/CAM++ prompt cache after validating its
waveform, encoder contracts and feature hashes. Target acoustics are never read.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from aligndit.model.speaker_embedding import validate_speaker_embedding_array
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    atomic_save_waveform, build_model, load_composed_config, load_normalization,
)
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH, LATENT_DIM, SAMPLE_RATE, load_semantic_vae_decoder, sha256_file,
)


PROJECT = Path(__file__).resolve().parents[4]
PROTOCOL = "CelebVDub_Setting2_GTText_Dubbing_v1"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def model_text(reference, target):
    if not reference.strip() or not target.strip():
        raise ValueError("Empty reference or target transcript")
    if len(reference[-1].encode("utf-8")) == 1:
        reference += " "
    return reference + " " + target


def geometry(reference, video, text, mean, std):
    p, t = len(reference), len(video)
    total = max(p + t, len(text) + 1, p + 1)
    if total > 4096:
        raise ValueError("Prompt/target/text exceeds the sampler limit")
    full_video = np.zeros((total, 1024), dtype=np.float32)
    full_video[p:p + t] = video
    cond = ((reference - mean) / std).astype(np.float32)
    return cond, full_video, p, t, total


def prepare(args, cfg):
    bundle = args.bundle
    inventory = json.loads((bundle / "checksums.json").read_text())["files"]
    manifest = bundle / "aligndit_inference.jsonl"
    if sha256_file(manifest) != inventory[manifest.name]["sha256"]:
        raise ValueError("Inference manifest differs from package inventory")
    raw_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in raw_rows]
    require_ids = (bundle / "targets.lst").read_text().splitlines()
    if len(set(ids)) != 115 or ids != require_ids:
        raise ValueError("Require the complete original 115 targets in frozen order")
    spec = json.loads((Path(cfg.datasets.cache_root) / "state/latents/spec.json").read_text())
    if spec["extraction"]["protocol"] != "semantic_vae_posterior_sample_v1":
        raise ValueError("Unsupported training codec protocol")
    rows, references = [], {}
    for raw in raw_rows:
        row = dict(raw)
        ident, ref = row["id"], row["reference_id"]
        for value in (ident, ref):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 3 or path.parts[0] != "test":
                raise ValueError("Unsafe ID")
        if ident == ref or row["speaker_id"] != row["reference_speaker_id"] or row["protocol"] != PROTOCOL:
            raise ValueError("Wrong same-speaker cross-utterance pair")
        if any(key in row for key in ("target_audio", "gt_audio", "gt_av_feature", "target_latent")):
            raise ValueError("Target acoustic inputs are forbidden")
        row["target_text_path"] = "text/" + ident + ".txt"
        for field in ("reference_audio", "video_feature", "target_text_path"):
            path = (bundle / row[field]).resolve(strict=True)
            relative = path.relative_to(bundle).as_posix()
            if sha256_file(path) != inventory[relative]["sha256"]:
                raise ValueError("Input hash differs from frozen package: " + relative)
            row[field] = str(path)
        row["target_text"] = Path(row["target_text_path"]).read_text().strip().lower()
        row["reference_text"] = row["reference_text"].strip().lower()
        row["model_text"] = model_text(row["reference_text"], row["target_text"])
        video = np.load(row["video_feature"], allow_pickle=False)
        if video.ndim != 2 or video.shape[1] != 1024 or len(video) < 1 or not np.isfinite(video).all():
            raise ValueError("Invalid video-only feature")
        row["video_frames_25hz"] = len(video)
        row["target_samples"] = len(video) * 640
        row["target_frames"] = math.ceil(row["target_samples"] / HOP_LENGTH)
        row["input_sha256"] = {field: sha256_file(row[field]) for field in
                               ("reference_audio", "video_feature", "target_text_path")}
        if ref not in references:
            cache_path = args.reference_cache / (ref + ".json")
            record = json.loads(cache_path.read_text())
            contract = record["contract"]
            checks = [record["reference_id"] == ref,
                record["reference_audio_sha256"] == row["input_sha256"]["reference_audio"],
                contract["vae"]["checkpoint_sha256"] == spec["checkpoint"]["ema_sha256"],
                contract["vae"]["metainfo_sha256"] == spec["checkpoint"]["metainfo_sha256"],
                contract["vae"]["ema_step"] == spec["checkpoint"]["ema_step"],
                contract["vae"]["git_commit"] == spec["semantic_vae_source"]["commit"],
                contract["speaker_checkpoint_sha256"] == cfg.datasets.speaker_embedding_checkpoint_sha256,
                contract["latent_protocol"] == "posterior_sample_v1_fp32_seed666_logvar_clamp_-12_12",
                contract["padding"] == "zero_right_to_400_samples",
                contract["speaker"] == "CAM++_10s_repeat_mean_L2"]
            expected_seed = int.from_bytes(hashlib.sha256(f"666:celebvdub/{ref}".encode()).digest()[:8], "little") % (2**63 - 1)
            checks.append(record["posterior_seed"] == expected_seed)
            if not all(checks):
                raise ValueError("Reference cache contract differs from trained frontend")
            for field in ("latent", "speaker"):
                if sha256_file(record[field + "_path"]) != record[field + "_sha256"]:
                    raise ValueError("Cached reference feature hash changed")
            latent = np.load(record["latent_path"], allow_pickle=False)
            if latent.shape != (record["frames"], LATENT_DIM) or latent.dtype != np.float32 or not np.isfinite(latent).all():
                raise ValueError("Invalid reference latent")
            validate_speaker_embedding_array(np.load(record["speaker_path"], allow_pickle=False))
            if record["frames"] != math.ceil(record["samples"] / HOP_LENGTH):
                raise ValueError("Reference duration metadata differs")
            record = {**record, "cache_record_path": str(cache_path), "cache_record_sha256": sha256_file(cache_path)}
            references[ref] = record
        rows.append(row)
    if len(references) != 83 or sum(row["target_samples"] for row in rows) != 8501120:
        raise ValueError("Original test corpus changed")
    return rows, references, spec


def source_hashes():
    paths = list((PROJECT / "src/aligndit/model").rglob("*.py"))
    paths += list((PROJECT / "src/f5_tts/model").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("infer_celebvdub_semantic_vae_s1.py"),
              Path(__file__).with_name("semantic_vae_decoder.py")]
    return {str(path.relative_to(PROJECT)): sha256_file(path) for path in sorted(set(paths))}


def run(args):
    cfg = load_composed_config(args.config)
    mean, std, _ = load_normalization(Path(cfg.datasets.normalization_path))
    if sha256_file(cfg.datasets.normalization_path) != cfg.datasets.expected_normalization_sha256:
        raise ValueError("Wrong training normalization")
    if sha256_file(cfg.datasets.vocab_path) != cfg.datasets.expected_vocab_sha256:
        raise ValueError("Wrong training vocabulary")
    rows, references, spec = prepare(args, cfg)
    print("Preflight passed: 115 targets / 83 verified independent reference conditions", flush=True)
    if args.dry_run:
        return
    if args.limit:
        # Exercise the longest combined prompt/video first in the smoke test.
        rows = sorted(rows, key=lambda row: references[row["reference_id"]]["frames"] + row["target_frames"], reverse=True)[:args.limit]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    model = build_model(args.config, args.checkpoint, 150000, device)
    decoder, codec = load_semantic_vae_decoder(repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint, cache_spec=spec, device=device)
    print("Strict 150000 EMA + window13 fixed-band training contract + decoder load passed", flush=True)
    config = {"protocol": PROTOCOL, "task": "Setting2-GTText-Dubbing", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_update": 150000, "use_ema": True,
        "config": str(args.config), "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "training_contract_sha256": sha256_file(args.checkpoint.parent / "speaker_training_contract.json"),
        "source_sha256": source_hashes(), "decoder": codec, "seed": args.seed,
        "steps": 32, "ode_method": "euler", "use_epss": True, "sway": -1, "cfg_t": 5, "cfg_v": 2,
        "dtype": "float32", "manifest": str(args.bundle / "aligndit_inference.jsonl"),
        "manifest_sha256": sha256_file(args.bundle / "aligndit_inference.jsonl"),
        "normalization_sha256": sha256_file(cfg.datasets.normalization_path),
        "reference_cache": str(args.reference_cache), "reference_cache_reused": True,
        "reference_source": "exact different-utterance waveform SVAE posterior seed666 and CAM++",
        "text_source": "reference transcript + historical separator + supplied GT target dialogue",
        "target_gt_text_read": True, "target_gt_audio_read": False, "target_acoustic_cache_read": False,
        "vsr_used": False, "limit": args.limit, "gpu": args.gpu, "memory_fraction": args.memory_fraction,
        "duration_source": "25Hz video frames * 640 samples; ceil to 40Hz and crop target only"}
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output provenance differs; use a fresh directory")
    save_json(config_path, config)
    receipt_hash = sha256_file(config_path)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    outputs = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        ref = references[row["reference_id"]]
        path = args.output_dir / (row["id"] + ".wav")
        meta_path = path.with_suffix(".json")
        inputs = {"row": row, "reference": ref, "run_config_sha256": receipt_hash}
        if path.exists() or meta_path.exists():
            result = json.loads(meta_path.read_text())
            if result["inputs"] != inputs or sha256_file(path) != result["audio_sha256"]:
                raise ValueError("Existing sample changed")
        else:
            video25 = torch.from_numpy(np.load(row["video_feature"], allow_pickle=False)).float()
            video = F.interpolate(video25.T[None], size=row["target_frames"], mode="linear", align_corners=False)[0].T.numpy()
            latent = np.load(ref["latent_path"], allow_pickle=False)
            speaker = np.load(ref["speaker_path"], allow_pickle=False)
            cond, full_video, p, t, total = geometry(latent, video, row["model_text"], mean, std)
            with torch.inference_mode():
                generated, _ = model.sample(cond=torch.from_numpy(cond)[None].to(device), text=[row["model_text"]],
                    duration=torch.tensor([total], device=device), video=torch.from_numpy(full_video)[None].to(device),
                    lens=torch.tensor([p], device=device), speaker_embedding=torch.from_numpy(speaker)[None].to(device),
                    steps=32, cfg_strength=5, cfg_strength_v=2, sway_sampling_coef=-1, seed=args.seed, use_epss=True)
                if generated.shape != (1, total, 64) or not torch.isfinite(generated).all():
                    raise ValueError("Invalid generated latent")
                raw = generated[:, p:p+t].float() * std_t + mean_t
                wave = decoder(raw.transpose(1, 2)).squeeze(0).float().cpu()
            if wave.shape != (1, t * HOP_LENGTH) or not torch.isfinite(wave).all() or not wave.square().mean() > 0:
                raise ValueError("Invalid generated waveform")
            atomic_save_waveform(path, wave[:, :row["target_samples"]])
            result = {"id": row["id"], "inputs": inputs, "audio_sha256": sha256_file(path),
                "samples": row["target_samples"], "sample_rate": SAMPLE_RATE, "prompt_frames": p,
                "target_frames": t, "sampler_frames": total, "text_extension_frames": total-p-t,
                "generation_seconds": time.monotonic() - started}
            save_json(meta_path, result)
            del generated, raw, wave
        outputs.append({key: result[key] for key in ("id", "audio_sha256", "samples", "sample_rate")})
        save_json(args.output_dir / "progress.json", {"requested": len(rows), "complete": len(outputs), "failures": []})
        print(f"[{index}/{len(rows)}] {row['id']} {time.monotonic()-started:.1f}s", flush=True)
    save_json(args.output_dir / "inference_summary.json", {"requested": len(rows), "complete": len(outputs),
        "failures": [], "run_config_sha256": receipt_hash, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "outputs": outputs})
    print(f"Setting2 generation complete: {len(outputs)}/{len(rows)}", flush=True)


def main():
    root = Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "src/aligndit/config/finetune_celebvdub_repa.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=root / "alignDiT_idea6/my_papers_code/Setting2-Dubbing-test")
    parser.add_argument("--reference-cache", type=Path, default=root / "alignDiT_idea6/Video-to-Speech-benchmark/cache/ours_svae_campplus_setting2_gttext_dubbing")
    parser.add_argument("--semantic-vae-repo", type=Path, default=root / "alignDiT_idea6/papers_codes/Semantic-VAE")
    parser.add_argument("--semantic-vae-checkpoint", type=Path, default=root / "alignDiT_idea6/Semantic-VAE/semantic_vae_1000k")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--memory-fraction", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1 or args.limit < 0:
        parser.error("Invalid memory fraction or limit")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run(args)


if __name__ == "__main__":
    main()
