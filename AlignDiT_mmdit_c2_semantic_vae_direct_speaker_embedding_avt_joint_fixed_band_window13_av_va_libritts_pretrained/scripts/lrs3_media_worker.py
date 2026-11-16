#!/usr/bin/env python3
"""Create auditable LRS3 mouth videos and frozen 25 Hz AV-HuBERT caches.

The input JSONL contains explicit paths and identities. Existing 96 px mouth
videos are linked; 224 px faces are aligned using the original 68-point tracks.
Segment offsets are established from the *complete* media inventory, including
segments absent from the training manifest. No whole-face resize fallback is
allowed. A failed item is reported and makes the final process exit nonzero.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import cv2
import numpy as np
from skimage.transform import estimate_transform


VERSION = "lrs3_landmarks_mouth_avhubert_v1"
PROJECT = Path(__file__).resolve().parents[1]
STABLE = [33, 36, 39, 42, 45]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def complete_segment_offsets(inventory, requested):
    groups = defaultdict(list)
    with open(inventory, newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            original = row["original_clip_id"]
            if original in requested:
                groups[original].append(row)
    offsets = {}
    for original, rows in groups.items():
        if len(rows) == 1 and rows[0]["segment_id"] == "":
            rows.sort(key=lambda row: 0)
        else:
            indices = sorted(int(row["segment_id"]) for row in rows)
            if indices != list(range(len(rows))):
                raise ValueError(f"Non-contiguous segment numbering: {original}: {indices}")
            rows.sort(key=lambda row: int(row["segment_id"]))
        total = sum(int(row["frames"]) for row in rows)
        offset = 0
        for row in rows:
            stem = Path(row["video_path"]).stem
            offsets[(original, stem)] = (offset, int(row["frames"]), total)
            offset += int(row["frames"])
    return offsets


def read_frames(path, expected=None, dimensions=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or abs(fps - 25) > 0.001:
        raise ValueError(f"Empty or non-25Hz video {path}: frames={len(frames)}, fps={fps}")
    if expected is not None and len(frames) != expected:
        raise ValueError(f"Decoded frame count mismatch {path}: {len(frames)} != {expected}")
    if dimensions is not None and any(frame.shape[:2] != dimensions for frame in frames):
        raise ValueError(f"Incorrect spatial dimensions: {path}")
    return frames


def interpolate_landmarks(raw):
    valid = [i for i, value in enumerate(raw)
             if value is not None and np.asarray(value).shape == (68, 2)
             and np.isfinite(value).all()]
    if not valid:
        raise ValueError("No valid landmark detections; full-face resize is prohibited")
    good = np.stack([raw[i] for i in valid]).astype(np.float32)
    times = np.arange(len(raw))
    tracks = np.empty((len(raw), 68, 2), dtype=np.float32)
    for point in range(68):
        for coord in range(2):
            tracks[:, point, coord] = np.interp(times, valid, good[:, point, coord])
    gaps = np.diff([-1] + valid + [len(raw)]) - 1
    metrics = {"landmarks_detected_frames": len(valid), "landmarks_total_frames": len(raw),
               "landmarks_max_missing_run": int(max(gaps)),
               "landmarks_missing_fraction": float(1 - len(valid) / len(raw))}
    # Long extrapolation/interpolation is unsuitable for a trusted training cache.
    if metrics["landmarks_missing_fraction"] > 0.2 or metrics["landmarks_max_missing_run"] > 25:
        raise ValueError(f"Landmark coverage requires review: {metrics}")
    return tracks, metrics


def crop_mouth(frames, tracks, mean_face):
    """Original AV-HuBERT/CelebV-Dub 12-frame smoothing and 96 px mouth ROI."""
    queue_frames, queue_tracks = deque(), deque()
    margin = min(12, len(frames))
    result, residuals = [], []
    transform = None

    def one(frame, landmarks, trans):
        transformed = trans(landmarks)
        x, y = np.rint(transformed[48:68].mean(axis=0)).astype(int)
        if not (48 <= x <= 208 and 48 <= y <= 208):
            raise ValueError(f"Mouth ROI outside canonical canvas: {(int(x), int(y))}")
        warped = cv2.warpAffine(frame, trans.params[:2, :3].astype(np.float32),
                               (256, 256), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        patch = warped[y - 48:y + 48, x - 48:x + 48]
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    for frame, track in zip(frames, tracks):
        queue_frames.append(frame)
        queue_tracks.append(track)
        if len(queue_frames) == margin:
            smooth = np.mean(queue_tracks, axis=0)
            transform = estimate_transform("similarity", smooth[STABLE], mean_face[STABLE])
            if not np.isfinite(transform.params).all() or not 0.25 < transform.scale < 4:
                raise ValueError("Invalid landmark alignment transform")
            residuals.append(float(np.linalg.norm(transform(smooth[STABLE]) - mean_face[STABLE], axis=1).mean()))
            result.append(one(queue_frames.popleft(), queue_tracks.popleft(), transform))
    while queue_frames:
        if transform is None:
            raise ValueError("No alignment transform; resize fallback is prohibited")
        result.append(one(queue_frames.popleft(), queue_tracks.popleft(), transform))
    if len(result) != len(frames):
        raise ValueError("Cropping changed the frame count")
    return result, {"alignment_residual_mean_px": float(np.mean(residuals)),
                    "alignment_residual_max_px": float(np.max(residuals))}


def verify_original_frame_mapping(original, frames, offset, total_count):
    """Check actual frame identity, not just equal aggregate durations."""
    if not original.exists():
        return {"landmark_mapping_evidence": "original_unavailable_frame_count_and_geometry"}
    capture = cv2.VideoCapture(str(original))
    comparisons = []
    try:
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != total_count:
            raise ValueError("Original video count differs from complete segment inventory")
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, offset + index)
            ok, reference = capture.read()
            if not ok or reference.shape != frames[index].shape:
                raise ValueError("Original/source frame geometry mismatch")
            difference = float(np.abs(reference.astype(np.float32) - frames[index]).mean())
            correlation = float(np.corrcoef(reference.ravel(), frames[index].ravel())[0, 1])
            comparisons.append({"segment_frame": index, "mean_absolute_pixel_error": difference,
                                "pixel_correlation": correlation})
            if difference > 8 or not np.isfinite(correlation) or correlation < 0.98:
                raise ValueError(f"Original landmark temporal mapping requires review: {comparisons[-1]}")
    finally:
        capture.release()
    return {"landmark_mapping_evidence": "decoded_original_first_middle_last_frames",
            "landmark_frame_comparisons": comparisons}


def write_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 96))
    if not writer.isOpened():
        raise ValueError("OpenCV mp4v writer did not open")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    read_frames(temporary, len(frames), (96, 96))
    os.replace(temporary, path)


def valid_array(path, frames):
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.shape != (frames, 1024) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid feature cache {path}: {array.shape}, {array.dtype}")


def load_reader(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    sys.path[:0] = [str(PROJECT / "src"), args.fairseq_root, str(Path(args.user_dir).parent), args.user_dir]
    import fairseq
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=args.user_dir))
    from aligndit.script.misc import avhubert_utils
    from aligndit.script.misc.extract_avhubert_from_only_video import AVHubertFeatureReader
    return AVHubertFeatureReader(args.checkpoint, custom_utils=avhubert_utils)


def media_protocol(checkpoint, mean_face):
    protocol = {"version": VERSION, "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": sha256(checkpoint), "mean_face_sha256": sha256(mean_face),
                "output_fps": 25, "feature_dimension": 1024,
                "crop": "96px_mouth_68point_similarity_12frame_smoothing",
                "landmarks": "original_tracks_cumulative_contiguous_segment_frames",
                "audio_input": None, "extract_finetune": True,
                "preprocess": "checkpoint_center_crop_and_normalization"}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, fingerprint


def main(args):
    cv2.setNumThreads(args.cpu_threads)
    root = Path(args.output_root)
    all_records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if len({r["relative_stem"] for r in all_records}) != len(all_records):
        raise ValueError("Duplicate output identities in manifest")
    records = all_records[args.shard::args.num_shards]
    if args.limit:
        records = records[:args.limit]
    requested = {r["original_clip_id"] for r in records if int(r["width"]) == 224}
    offsets = complete_segment_offsets(args.inventory, requested)
    mean_face = np.load(args.mean_face, allow_pickle=False)
    if mean_face.shape != (68, 2) or not np.isfinite(mean_face).all():
        raise ValueError("Invalid mean face")
    protocol, protocol_hash = media_protocol(args.checkpoint, args.mean_face)
    report = root / "reports" / f"media_{Path(args.manifest).stem}_{args.stage}_shard{args.shard}.jsonl"
    report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report.with_suffix(".protocol.json"), {**protocol, "protocol_sha256": protocol_hash,
                 "manifest_sha256": sha256(args.manifest), "num_shards": args.num_shards, "shard": args.shard})
    reader = None
    successes = failures = 0
    started = time.time()
    with report.open("a", buffering=1) as log:
        for index, row in enumerate(records):
            identity = row["relative_stem"]
            if Path(identity).is_absolute() or ".." in Path(identity).parts:
                raise ValueError(f"Unsafe relative identity {identity}")
            mouth = root / "video_mouth" / (identity + ".mp4")
            feature = root / "avhubert_video_25hz" / (identity + ".npy")
            receipt_path = root / "reports" / "media_receipts" / (identity + ".json")
            result = {"sample_id": row["sample_id"], "relative_stem": identity,
                      "original_clip_id": row["original_clip_id"], "stage": args.stage,
                      "video_mouth_path": str(mouth), "video_feature_path": str(feature),
                      "protocol_sha256": protocol_hash, "timestamp": time.time()}
            try:
                source = Path(row["video_path"])
                count = int(row["video_frames"])
                source_hash = sha256(source)
                repair_path = root / "repaired_landmarks" / (identity + ".npy")
                repair_hash = sha256(repair_path) if repair_path.exists() else None
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                matching = (receipt.get("protocol_sha256") == protocol_hash
                            and receipt.get("source_sha256") == source_hash
                            and receipt.get("landmark_repair_sha256") == repair_hash)
                if not matching or not mouth.exists():
                    frames = read_frames(source, count, (int(row["height"]), int(row["width"])))
                    crop_metrics = {}
                    if int(row["width"]) == 96 and int(row["height"]) == 96:
                        mouth.parent.mkdir(parents=True, exist_ok=True)
                        temporary = mouth.with_name(mouth.name + f".tmp.{os.getpid()}")
                        temporary.unlink(missing_ok=True)
                        temporary.symlink_to(source.resolve())
                        os.replace(temporary, mouth)
                        crop_kind = "existing_96px_mouth"
                    elif int(row["width"]) == 224 and int(row["height"]) == 224:
                        offset, segment_count, total_count = offsets[(row["original_clip_id"], source.stem)]
                        if segment_count != count:
                            raise ValueError("Manifest frames disagree with complete inventory")
                        landmark_path = Path(args.landmark_root) / (row["original_clip_id"] + ".pkl")
                        if repair_hash:
                            repair = json.loads(repair_path.with_suffix(".json").read_text())
                            if repair["source_sha256"] != source_hash or repair["landmark_sha256"] != repair_hash:
                                raise ValueError("FAN landmark repair/source checksum mismatch")
                            tracks = np.load(repair_path, allow_pickle=False)
                            if tracks.shape != (count, 68, 2) or not np.isfinite(tracks).all():
                                raise ValueError("Invalid repaired landmark shape/values")
                            crop_metrics = {"landmark_repair_sha256": repair_hash, "landmark_repair": repair}
                            crop_kind = "fan_repaired_aligned_96px_mouth"
                        else:
                            with landmark_path.open("rb") as stream:
                                tracks_all = pickle.load(stream)
                            if len(tracks_all) != total_count:
                                raise ValueError(f"Original landmark/media frame mismatch {len(tracks_all)} != {total_count}")
                            tracks, crop_metrics = interpolate_landmarks(tracks_all[offset:offset + count])
                            crop_metrics.update(verify_original_frame_mapping(
                                Path(args.original_root) / (row["original_clip_id"] + ".mp4"), frames, offset, total_count))
                            crop_kind = "landmark_aligned_96px_mouth"
                        if np.mean((tracks < -20) | (tracks > 244)) > 0.05:
                            raise ValueError("Landmarks outside the 224px source geometry")
                        patches, alignment = crop_mouth(frames, tracks, mean_face)
                        crop_metrics.update(alignment)
                        crop_metrics.update({"landmark_offset_frames": offset, "original_total_frames": total_count,
                                             "landmark_sha256": sha256(landmark_path) if landmark_path.exists() else None})
                        write_video(mouth, patches)
                    else:
                        raise ValueError(f"Unsupported video size: {row['width']}x{row['height']}")
                    receipt = {"relative_stem": identity, "source_path": str(source), "source_sha256": source_hash,
                               "protocol_sha256": protocol_hash, "crop_kind": crop_kind, "video_frames": count,
                               **crop_metrics, "features_complete": False}
                    atomic_json(receipt_path, receipt)
                read_frames(mouth, count, (96, 96))
                if args.stage != "crop":
                    reuse = receipt.get("features_complete", False) and feature.exists()
                    if reuse:
                        try:
                            valid_array(feature, count)
                            if sha256(feature) != receipt.get("feature_sha256"):
                                raise ValueError("Feature checksum mismatch")
                        except Exception:
                            reuse = False
                    if not reuse:
                        if reader is None:
                            reader = load_reader(args)
                            protocol["checkpoint_image_crop_size"] = int(reader.task.cfg.image_crop_size)
                            protocol["checkpoint_image_mean"] = float(reader.task.cfg.image_mean)
                            protocol["checkpoint_image_std"] = float(reader.task.cfg.image_std)
                            atomic_json(report.with_suffix(".protocol.json"), {**protocol, "protocol_sha256": protocol_hash,
                                         "manifest_sha256": sha256(args.manifest), "num_shards": args.num_shards, "shard": args.shard})
                        array = reader.get_feats((str(mouth), None)).detach().cpu().numpy().astype(np.float32)
                        feature.parent.mkdir(parents=True, exist_ok=True)
                        temporary = feature.with_name(feature.name + f".tmp.{os.getpid()}")
                        with temporary.open("wb") as stream:
                            np.save(stream, array, allow_pickle=False)
                        valid_array(temporary, count)
                        os.replace(temporary, feature)
                        receipt["features_complete"] = True
                        receipt["feature_sha256"] = sha256(feature)
                        atomic_json(receipt_path, receipt)
                    result["feature_shape"] = [count, 1024]
                    result["feature_cache_reused"] = reuse
                result.update({"status": "ok", "crop_kind": receipt["crop_kind"],
                               "video_frames": count, "crop_quality": {k: v for k, v in receipt.items()
                                   if k.startswith("landmark") or k.startswith("alignment")}})
                successes += 1
            except Exception as error:
                failures += 1
                result.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps({"index": index + 1, "total": len(records), "sample": identity,
                              "status": result["status"], "error": result.get("error"),
                              "elapsed_seconds": round(time.time() - started, 2)}), flush=True)
    summary = {"requested": len(records), "succeeded": successes, "failed": failures,
               "complete": successes == len(records), "elapsed_seconds": time.time() - started,
               "report": str(report), "protocol_sha256": protocol_hash}
    atomic_json(report.with_suffix(".summary.json"), summary)
    print(json.dumps(summary), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory", default=prefix + "/zjw524/datasets/LRS3/DARKNSU_extracted/audit/all_pairs_manifest.tsv")
    parser.add_argument("--landmark-root", default=prefix + "/zjw524/datasets/LRS3/landmarks/LRS3_landmarks")
    parser.add_argument("--original-root", default=prefix + "/zjw524/datasets/LRS3/ainncy")
    parser.add_argument("--checkpoint", default=prefix + "/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt")
    parser.add_argument("--fairseq-root", default=prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
    parser.add_argument("--user-dir", default=prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert")
    parser.add_argument("--mean-face", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--stage", choices=["all", "crop", "features"], default="all")
    arguments = parser.parse_args()
    if not 0 <= arguments.shard < arguments.num_shards:
        parser.error("shard must be in [0, num_shards)")
    sys.exit(main(arguments))
