"""Frozen-checkpoint temporal retrieval and paired video interventions.

Run from this snapshot with PYTHONPATH=src. No optimizer or backward pass is
created. The diagnostic uses teacher-forced flow states, not ODE generation.
Teacher NPZs must be separately extracted with the original audio-only teacher.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligndit.model.audio_teacher_cache import teacher_frame_lengths
from aligndit.model.avhubert_infonce import sample_context_on_teacher_grid, temporal_context_infonce
from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
    build_model,
    load_composed_config,
    load_normalization,
    load_setting1_speaker_embeddings,
    validate_record_arrays,
)


CONDITIONS = (
    "correct", "advance_50ms", "delay_50ms", "advance_100ms", "delay_100ms",
    "advance_200ms", "delay_200ms", "shuffle", "zero", "learned_null",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sample_seed(utterance, seed, purpose):
    value = f"{utterance}|{seed}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def make_video_conditions(video, start, seed):
    """Change only the visible generation span; no circular time wrapping."""
    length = video.shape[0]
    position = torch.arange(length, device=video.device)
    result = {"correct": video}
    for milliseconds, frames in ((50, 2), (100, 4), (200, 8)):
        for direction, sign in (("advance", 1), ("delay", -1)):
            index = (position + sign * frames).clamp(start, length - 1)
            changed = video.clone()
            changed[start:] = video[index[start:]]
            result[f"{direction}_{milliseconds}ms"] = changed
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length - start, generator=generator).to(video.device) + start
    shuffled = video.clone()
    shuffled[start:] = video[permutation]
    result["shuffle"] = shuffled
    result["zero"] = torch.zeros_like(video)
    # The null intervention is executed separately using the model's learned
    # CFG null input. We still explicitly evaluate its auxiliary representation.
    result["learned_null"] = video
    return result


def retrieval_metrics(projected, teacher, generation, evaluation, valid_length):
    length = torch.tensor([projected.shape[1]], device=projected.device)
    sampled, eligible = sample_context_on_teacher_grid(projected.float(), length, generation, teacher.shape[0])
    _, interior = sample_context_on_teacher_grid(projected.float(), length, evaluation, teacher.shape[0])
    eligible = eligible[0, :valid_length] & interior[0, :valid_length]
    anchors = eligible.nonzero(as_tuple=False).flatten()
    if not len(anchors):
        return {"anchors": 0}
    keys = torch.arange(valid_length, device=projected.device)
    q = F.normalize(sampled[0, anchors].float(), dim=-1)
    k = F.normalize(teacher[:valid_length].float(), dim=-1)
    sim = q @ k.T
    distance = keys[None] - anchors[:, None]
    gap5 = (distance.abs() >= 5) | (distance == 0)
    has_negative = (distance.abs() >= 5).any(-1)
    anchors = anchors[has_negative]
    sim, distance, gap5 = sim[has_negative], distance[has_negative], gap5[has_negative]
    if not len(anchors):
        return {"anchors": 0}
    logits = sim / 0.07
    gap_logits = logits.masked_fill(~gap5, -torch.inf)
    full_best = sim.argmax(-1)
    gap_best = gap_logits.argmax(-1)
    error = (full_best - anchors).float()
    positive = sim.gather(1, anchors[:, None]).squeeze(1)
    near = (distance.abs() >= 1) & (distance.abs() <= 4)
    near_best = sim.masked_fill(~near, -torch.inf).max(-1).values
    result = {
        "anchors": len(anchors),
        "loss_gap5": F.cross_entropy(gap_logits, anchors).item(),
        "loss_full": F.cross_entropy(logits, anchors).item(),
        "top1_gap5": (gap_best == anchors).float().mean().item(),
        "top1_full": (full_best == anchors).float().mean().item(),
        "within_40ms": (error.abs() <= 1).float().mean().item(),
        "within_80ms": (error.abs() <= 2).float().mean().item(),
        "within_160ms": (error.abs() <= 4).float().mean().item(),
        "offset_signed_ms": (error.mean() * 40).item(),
        "offset_abs_ms": (error.abs().mean() * 40).item(),
        "offset_abs_median_ms": (error.abs().median() * 40).item(),
        "positive_cosine": positive.mean().item(),
        "near_margin": (positive - near_best).mean().item(),
        "near_beats_positive": (near_best > positive).float().mean().item(),
        "gap_correct_full_wrong": ((gap_best == anchors) & (full_best != anchors)).float().mean().item(),
    }
    for offset in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
        target = anchors + offset
        ok = (target >= 0) & (target < valid_length)
        result[f"teacher_offset_{offset:+d}_cos"] = sim[ok].gather(1, target[ok, None]).mean().item()
    return result


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    project = Path(__file__).resolve().parents[1]
    config_path = project / "src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_avhubert_infonce.yaml"
    config = load_composed_config(config_path)
    cache_root = Path(config.datasets.cache_root)
    manifest = cache_root / "manifests/test.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    if "correct" not in conditions or set(conditions) - set(CONDITIONS):
        raise ValueError("Conditions must include correct and use declared interventions")
    seeds = [int(x) for x in args.seeds.split(",")]
    times = [float(x) for x in args.times.split(",")]
    checkpoint = Path(config.ckpts.save_dir) / f"model_{args.step}.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    teacher_manifest = args.teacher_features.parent / "teacher_manifest.json"
    if not teacher_manifest.is_file():
        raise FileNotFoundError(f"Missing teacher provenance: {teacher_manifest}")
    contract = {
        "checkpoint": str(checkpoint), "checkpoint_step": args.step, "ema": True,
        "checkpoint_sha256": digest(checkpoint), "script_sha256": digest(__file__),
        "teacher_manifest_sha256": digest(teacher_manifest),
        "backbone_sha256": digest(project / "src/aligndit/model/backbone/dit_vt_mm.py"),
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "config_sha256": digest(config_path), "test_count": len(records),
        "noise_seeds": seeds, "flow_times": times, "conditions": conditions,
        "precision": args.precision,
        "generation": "last 80% of original clip; first 20% clean audio prompt; full GT speaker",
        "evaluation_anchors": "generation span excluding 8 latent frames at both ends; common across interventions",
        "video_shift": "only visible generation span, edge replication, no wrap; advance means V[t]=V_original[t+shift]",
        "teacher": "frozen original-GT audio-only AV-HuBERT, native25Hz, fp16 cached, valid preprocessing support only",
        "mode": "frozen eval teacher-forced flow states; no CFG combination, ODE, decoder, optimization or backward",
        "interpretation": "retrieval offset is an internal feature-index error, not generated waveform lip-sync delay",
    }
    contract_path = args.output / "protocol.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("Refusing to mix incompatible diagnostic protocols")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    mean, std, _ = load_normalization(Path(config.datasets.normalization_path))
    speakers, _ = load_setting1_speaker_embeddings(config, records)
    model = build_model(config_path, checkpoint, args.step, device)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    backbone = model.transformer
    assert not backbone.normalize_text_context
    vocab = model.vocab_char_map
    output_path = args.output / "per_clip.jsonl"
    existing = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            item = json.loads(line)
            existing.add((item["utterance_key"], item["noise_seed"], item["flow_t"], item["condition"]))
    print(json.dumps({"event": "model_loaded", "step": args.step, "resume_rows": len(existing)}), flush=True)
    started = time.monotonic()
    with output_path.open("a", buffering=1) as stream:
        for index, (record, speaker) in enumerate(zip(records, speakers)):
            latent_array, video_array = validate_record_arrays(record, cache_root, mean, std)
            latent = torch.from_numpy(latent_array).to(device)[None]
            video = torch.from_numpy(video_array).to(device)
            frames = latent.shape[1]
            start = int(frames * 0.2)
            generation = torch.zeros((1, frames), dtype=torch.bool, device=device)
            generation[:, start:] = True
            evaluation = torch.zeros_like(generation)
            evaluation[:, start + 8:frames - 8] = True
            cond = latent.masked_fill(generation[..., None], 0)
            text = torch.tensor([[vocab.get(character, 0) for character in record["text"]]], device=device)
            text_mask = torch.ones_like(text, dtype=torch.bool)
            frame_mask = torch.ones_like(generation)
            teacher_path = args.teacher_features / Path(record["audio_relative_path"]).with_suffix(".npz")
            with np.load(teacher_path, allow_pickle=False) as data:
                teacher = torch.from_numpy(data["features"].astype(np.float32)).to(device)
                valid = int(data["valid_length"])
                if str(data["utterance_key"].item()) != record["utterance_key"]:
                    raise RuntimeError(f"Teacher ID mismatch: {teacher_path}")
                expected_stored, expected_valid = teacher_frame_lengths(record["original_num_samples"])
                if teacher.shape != (expected_stored, 1024) or valid != expected_valid:
                    raise RuntimeError(f"Teacher length mismatch: {teacher_path}")
            speaker = speaker[None].to(device)
            for seed in seeds:
                noise_generator = torch.Generator().manual_seed(sample_seed(record["utterance_key"], seed, "noise"))
                noise = torch.randn(latent.shape, generator=noise_generator).to(device)
                target_flow = latent - noise
                videos = make_video_conditions(video, start, sample_seed(record["utterance_key"], seed, "shuffle"))
                for flow_t in times:
                    row_keys = [(record["utterance_key"], seed, flow_t, c) for c in conditions]
                    if all(k in existing for k in row_keys):
                        continue
                    noisy = (1 - flow_t) * noise + flow_t * latent
                    outputs = {}
                    normal = [c for c in conditions if c != "learned_null"]
                    groups = [normal[i:i + args.condition_batch] for i in range(0, len(normal), args.condition_batch)]
                    if "learned_null" in conditions:
                        groups.append(["learned_null"])
                    for group in groups:
                        count = len(group)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                            predicted, _, projected = backbone(
                                x=noisy.expand(count, -1, -1), cond=cond.expand(count, -1, -1),
                                text=text.expand(count, -1), video=torch.stack([videos[c] for c in group]),
                                time=torch.full((count,), flow_t, device=device),
                                mask=frame_mask.expand(count, -1), text_mask=text_mask.expand(count, -1),
                                video_mask=frame_mask.expand(count, -1),
                                complementary_mask=(~generation).expand(count, -1),
                                generation_mask=generation.expand(count, -1),
                                speaker_embedding=speaker.expand(count, -1),
                                drop_audio_cond=False, drop_text=False, drop_video=group == ["learned_null"],
                                drop_speaker=False, cfg_infer=False, cache=False, return_context_alignment=True,
                            )
                        for position, condition in enumerate(group):
                            outputs[condition] = (predicted[position:position + 1].float(), projected[position:position + 1].float())
                    reference_flow, reference_projection = outputs["correct"]
                    for condition in conditions:
                        key = (record["utterance_key"], seed, flow_t, condition)
                        if key in existing:
                            continue
                        predicted, projected = outputs[condition]
                        result = retrieval_metrics(projected, teacher, generation, evaluation, valid)
                        if index == 0 and seed == seeds[0] and flow_t == times[0] and condition == "correct":
                            length = torch.tensor([frames], device=device)
                            teacher_length = torch.tensor([len(teacher)], device=device)
                            reference_loss, reference_stats = temporal_context_infonce(
                                projected, teacher[None], length, teacher_length,
                                torch.tensor([valid], device=device), evaluation, enabled=True,
                            )
                            if reference_stats["infonce_valid_anchors"] != result["anchors"]:
                                raise RuntimeError("Diagnostic anchors disagree with training loss implementation")
                            if not np.isclose(reference_loss.item(), result["loss_gap5"], atol=1e-6):
                                raise RuntimeError("Diagnostic gap5 loss disagrees with training loss implementation")
                        selected = evaluation if evaluation.any() else generation
                        flow_selected = predicted[selected]
                        reference_selected = reference_flow[selected]
                        result.update(
                            utterance_key=record["utterance_key"], video_id=record["video_id"],
                            noise_seed=seed, flow_t=flow_t, condition=condition, latent_frames=frames,
                            flow_mse=F.mse_loss(flow_selected, target_flow[selected]).item(),
                            flow_change_rms=(flow_selected - reference_selected).square().mean().sqrt().item(),
                            reference_flow_rms=reference_selected.square().mean().sqrt().item(),
                            context_change_rms=(projected[selected] - reference_projection[selected]).square().mean().sqrt().item(),
                            reference_context_rms=reference_projection[selected].square().mean().sqrt().item(),
                        )
                        if result["anchors"] and not all(np.isfinite(v) for v in result.values() if isinstance(v, float)):
                            raise FloatingPointError(f"Nonfinite diagnostic: {key}")
                        stream.write(json.dumps(result) + "\n")
                        existing.add(key)
                    del outputs
            if (index + 1) % 5 == 0 or index == 0:
                print(json.dumps({"event": "progress", "clips": index + 1, "total": len(records),
                                  "seconds": round(time.monotonic() - started, 1), "rows": len(existing)}), flush=True)
    completion = {"clips": len(records), "rows": len(existing), "expected_rows": len(records) * len(seeds) * len(times) * len(conditions),
                  "seconds": time.monotonic() - started, "script_sha256": digest(__file__)}
    if completion["rows"] != completion["expected_rows"]:
        raise RuntimeError(f"Incomplete diagnostic: {completion}")
    (args.output / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps({"event": "complete", **completion}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=200000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--times", default="0.05,0.25,0.5,0.75,0.95")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--condition-batch", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
