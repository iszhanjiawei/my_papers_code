#!/usr/bin/env python3
"""Re-detect LRS3 landmarks with local FAN/S3FD weights, without resize fallback."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from lrs3_media_worker import atomic_json, interpolate_landmarks, read_frames, sha256


def main(args):
    root = Path(args.output_root)
    checkpoint_dir = root / "assets" / "torch" / "hub" / "checkpoints"
    names = ["2DFAN4-11f355bf06.pth.tar", "s3fd-619a316812.pth"]
    hashes = {}
    for name in names:
        path = checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required local FAN weight: {path}. Download using a direct network session first.")
        hashes[name] = sha256(path)
    torch.hub.set_dir(str(checkpoint_dir.parent))
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    import face_alignment
    options = dict(flip_input=False, device="cuda", face_detector="sfd",
                   face_detector_kwargs={"path_to_detector": str(checkpoint_dir / names[1])})
    if "compile" in inspect.signature(face_alignment.FaceAlignment).parameters:
        options["compile"] = False
    model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, **options)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows = rows[args.shard::args.num_shards]
    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    report = root / "reports" / f"media_fan_repair_{Path(args.manifest).stem}{suffix}.jsonl"
    report.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with report.open("a", buffering=1) as stream:
        for row in rows:
            result = {"sample_id": row["sample_id"], "relative_stem": row["relative_stem"]}
            try:
                existing_path = root / "repaired_landmarks" / (row["relative_stem"] + ".npy")
                existing_receipt = existing_path.with_suffix(".json")
                if existing_path.exists() and existing_receipt.exists():
                    previous = json.loads(existing_receipt.read_text())
                    tracks = np.load(existing_path, allow_pickle=False)
                    if (previous.get("models_sha256") == hashes
                            and previous.get("source_sha256") == sha256(row["video_path"])
                            and previous.get("landmark_sha256") == sha256(existing_path)
                            and previous.get("fan_mean_score_min", 0) >= args.min_landmark_score
                            and tracks.shape == (row["video_frames"], 68, 2)
                            and np.isfinite(tracks).all()):
                        result.update({**previous, "repair_cache_reused": True})
                        stream.write(json.dumps(result) + "\n")
                        print(json.dumps({"sample_id": row["sample_id"], "status": "ok", "repair_cache_reused": True}), flush=True)
                        continue
                frames = read_frames(row["video_path"], row["video_frames"], (224, 224))
                landmarks, scores, confidences = [], [], []
                rejected_low_scores = 0
                for frame in frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes = model.face_detector.detect_from_image(rgb.copy())
                    if len(boxes) == 0:
                        landmarks.append(None)
                        continue
                    # LRS3 is a tracked single-face crop. Choose the largest face
                    # consistently, without substituting the full-frame rectangle.
                    box = max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
                    predicted, score, _ = model.get_landmarks_from_image(rgb, detected_faces=[box], return_landmark_score=True)
                    if predicted is None:
                        landmarks.append(None)
                        continue
                    mean_score = float(np.mean(score[0]))
                    if mean_score < args.min_landmark_score:
                        landmarks.append(None)
                        rejected_low_scores += 1
                        continue
                    landmarks.append(predicted[0])
                    scores.append(mean_score)
                    confidences.append(float(box[-1]))
                tracks, quality = interpolate_landmarks(landmarks)
                if np.mean((tracks < -20) | (tracks > 244)) > 0.05:
                    raise ValueError("Re-detected landmarks leave the source geometry")
                path = root / "repaired_landmarks" / (row["relative_stem"] + ".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
                with temporary.open("wb") as output:
                    np.save(output, tracks.astype(np.float32), allow_pickle=False)
                os.replace(temporary, path)
                result.update({"status": "ok", "landmark_path": str(path),
                               "source_sha256": sha256(row["video_path"]), "landmark_sha256": sha256(path),
                               "models_sha256": hashes, "method": "FAN_2D_S3FD_every_frame_largest_face",
                               "face_alignment_version": getattr(face_alignment, "__version__", "unknown"),
                               "fan_mean_score_min": min(scores), "fan_mean_score_mean": float(np.mean(scores)),
                               "fan_mean_score_floor": args.min_landmark_score,
                               "fan_low_score_rejected_frames": rejected_low_scores,
                               "s3fd_min_confidence": min(confidences), "timestamp": time.time(), **quality})
                atomic_json(path.with_suffix(".json"), result)
            except Exception as error:
                failures += 1
                result.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
            stream.write(json.dumps(result) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ["traceback", "models_sha256"]}), flush=True)
    atomic_json(report.with_suffix(".summary.json"), {"requested": len(rows), "succeeded": len(rows) - failures,
                 "failed": failures, "complete": failures == 0, "shard": args.shard, "num_shards": args.num_shards})
    return int(failures > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-landmark-score", type=float, default=0.3,
                        help="Heuristic mean FAN heatmap-score floor; rejected frames count as missing detections.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    arguments = parser.parse_args()
    if not 0 <= arguments.shard < arguments.num_shards:
        parser.error("shard must be in [0, num_shards)")
    raise SystemExit(main(arguments))
