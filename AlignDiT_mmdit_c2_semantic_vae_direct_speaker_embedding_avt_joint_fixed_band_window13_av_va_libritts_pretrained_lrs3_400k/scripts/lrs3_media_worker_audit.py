#!/usr/bin/env python3
"""Validate cumulative LRS3 landmark frame offsets against decoded originals."""
import argparse
import json
from pathlib import Path
import pickle

import cv2
import numpy as np

from lrs3_media_worker import atomic_json, complete_segment_offsets


def frame_at(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Could not decode frame {frame}")
    return image


def main(args):
    cv2.setNumThreads(2)
    root = Path(args.output_root)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    rows = [row for row in rows if int(row["width"]) == 224]
    offsets = complete_segment_offsets(args.inventory, {row["original_clip_id"] for row in rows})
    results = []
    panels = []
    panel_originals = set()
    for row in rows:
        original = Path(args.original_root) / (row["original_clip_id"] + ".mp4")
        offset, count, total = offsets[(row["original_clip_id"], row["sample_id"])]
        result = {"sample_id": row["sample_id"], "original_clip_id": row["original_clip_id"],
                  "landmark_offset_frames": offset, "video_frames": count,
                  "original_total_frames": total, "original_available": original.exists()}
        source_cap = cv2.VideoCapture(row["video_path"])
        if original.exists():
            cap = cv2.VideoCapture(str(original))
            original_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result["decoded_original_frame_count"] = original_count
            comparisons = []
            for k in sorted({0, count // 2, count - 1}):
                source = frame_at(source_cap, k)
                reference = frame_at(cap, offset + k)
                if source.shape != reference.shape:
                    raise ValueError(f"Different original/source geometry: {row['sample_id']}")
                difference = np.abs(source.astype(np.float32) - reference.astype(np.float32))
                correlation = float(np.corrcoef(source.ravel(), reference.ravel())[0, 1])
                comparisons.append({"segment_frame": k, "original_frame": offset + k,
                                    "mean_absolute_pixel_error": float(difference.mean()),
                                    "pixel_correlation": correlation})
            cap.release()
            result["frame_comparisons"] = comparisons
            result["mapping_verified"] = original_count == total and all(
                c["mean_absolute_pixel_error"] <= 8 and c["pixel_correlation"] >= 0.98 for c in comparisons)
        # A deterministic contact sheet includes cropped outputs only when present.
        mouth = root / "video_mouth" / (row["relative_stem"] + ".mp4")
        if (mouth.exists() and row["original_clip_id"] not in panel_originals
                and (len(panels) < 6 or len(panels) < 12 and not original.exists())):
            panel_originals.add(row["original_clip_id"])
            with (Path(args.landmark_root) / (row["original_clip_id"] + ".pkl")).open("rb") as stream:
                landmarks = pickle.load(stream)
            repair = root / "repaired_landmarks" / (row["relative_stem"] + ".npy")
            repaired = np.load(repair, allow_pickle=False) if repair.exists() else None
            mouth_cap = cv2.VideoCapture(str(mouth))
            images = []
            for k in [0, count - 1]:
                source = frame_at(source_cap, k)
                points = repaired[k] if repaired is not None else landmarks[offset + k]
                if points is not None:
                    for x, y in points:
                        cv2.circle(source, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
                crop = frame_at(mouth_cap, k)
                images.append(np.concatenate([source, cv2.resize(crop, (224, 224))], axis=1))
            mouth_cap.release()
            panel = np.concatenate(images, axis=0)
            canvas = np.zeros((480, 448, 3), dtype=np.uint8)
            canvas[32:] = panel
            label = f"{row['sample_id']} offset={offset}"
            cv2.putText(canvas, label[:55], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
            panels.append(canvas)
        source_cap.release()
        results.append(result)
    report = root / "reports" / "media_pilot_frame_mapping.jsonl"
    report.write_text("".join(json.dumps(row) + "\n" for row in results))
    checked = [row for row in results if row["original_available"]]
    summary = {"checked_224px_samples": len(rows), "original_pixel_verified_samples": sum(r["mapping_verified"] for r in checked),
               "original_pixel_compared_samples": len(checked),
               "missing_original_samples": len(rows) - len(checked),
               "failed_original_comparisons": [r["sample_id"] for r in checked if not r["mapping_verified"]],
               "qualification": "Pixel comparison verifies cumulative offsets where originals exist. Other records require landmark geometry review; frame-count equality alone does not prove identity."}
    atomic_json(report.with_suffix(".summary.json"), summary)
    if panels:
        while len(panels) % 3:
            panels.append(np.zeros_like(panels[0]))
        contact = np.concatenate([np.concatenate(panels[i:i + 3], axis=1) for i in range(0, len(panels), 3)], axis=0)
        cv2.imwrite(str(root / "reports" / "media_pilot_landmark_contact_sheet.jpg"), contact)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    main(parser.parse_args())
