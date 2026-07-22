"""
Crop mouth ROI from full-face CelebV-Dub videos for AV-HuBERT feature extraction.

Pipeline (mirrors avhubert/preparation/):
  1. Detect 68 facial landmarks per frame using face_alignment (FAN).
  2. Interpolate missing detections.
  3. Warp each frame to a canonical 256x256 face using 5 stable landmarks
     aligned to the 20words_mean_face.npy template.
  4. Crop 96x96 mouth ROI (landmarks 48-68) from the warped frame.
  5. Convert to grayscale, write output .mp4 at 25 fps.

Output videos are suitable as direct input to extract_avhubert_from_only_video.py.
"""

import argparse
import glob
import logging
import math
import os
import subprocess
import sys
import tempfile
import shutil
from collections import deque

import cv2
import face_alignment
import numpy as np
import torch
from skimage import transform as tf
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("crop_mouth_celebvdub")

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
CROP_HEIGHT = 96
CROP_WIDTH = 96
WINDOW_MARGIN = 12
REDETECT_INTERVAL = 8  # re-run S3FD face detection every N frames; reuse bbox in between
DETECT_STRIDE = 2  # run FAN landmark prediction every N frames; interpolate the rest


def detect_landmarks_video(fa_model, frames_rgb, device):
    """68-landmark detection for a whole video with two cost-reduction tricks:

    1. S3FD bbox is re-detected only every REDETECT_INTERVAL frames and reused
       (faces barely move within these 1-2s clips).
    2. FAN landmarks are computed only every DETECT_STRIDE frames; the skipped
       frames are left as None and filled afterwards by landmarks_interpolate
       (landmarks change slowly at 25 fps).
    """
    n = len(frames_rgb)
    landmarks_list = [None] * n
    last_bbox = None
    for i in range(0, n, DETECT_STRIDE):
        rgb = frames_rgb[i]
        try:
            if i % REDETECT_INTERVAL == 0 or last_bbox is None:
                bboxes = fa_model.face_detector.detect_from_image(rgb.copy())
                if bboxes is not None and len(bboxes) > 0:
                    last_bbox = [bboxes[0]]
            if last_bbox is None:
                continue
            lms = fa_model.get_landmarks(rgb, detected_faces=last_bbox)
            if lms is not None and len(lms) > 0:
                landmarks_list[i] = lms[0]
        except Exception:
            pass
    # ensure the last frame has a landmark so interpolation can extend to the end
    if landmarks_list[-1] is None:
        for j in range(n - 1, -1, -1):
            if landmarks_list[j] is not None:
                landmarks_list[-1] = landmarks_list[j]
                break
    return landmarks_list


def read_video_frames(path):
    """Read video frames as BGR numpy arrays using OpenCV."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_lm = landmarks[start_idx]
    stop_lm = landmarks[stop_idx]
    delta = stop_lm - start_lm
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_lm + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def landmarks_interpolate(landmarks):
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        landmarks = linear_interpolate(
            landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
        )
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform("similarity", src, dst)
    warped = _cv2_warp(img, tform, std_size)
    return warped, tform


def apply_transform(transform, img, std_size):
    return _cv2_warp(img, transform, std_size)


def _cv2_warp(img, tform, std_size):
    # tform maps src -> dst (forward). cv2.warpAffine (without WARP_INVERSE_MAP)
    # forward-maps src into the dst canvas, matching skimage's tf.warp(inverse_map=tform.inverse).
    M = tform.params[:2, :3].astype(np.float32)
    warped = cv2.warpAffine(
        img,
        M,
        (std_size[1], std_size[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def cut_patch(img, landmarks, height, width):
    center_x = np.mean(landmarks[:, 0])
    center_y = np.mean(landmarks[:, 1])
    center_x = round(center_x)
    center_y = round(center_y)

    # Clip to image bounds
    half_h = height
    half_w = width
    y1, y2 = center_y - half_h, center_y + half_h
    x1, x2 = center_x - half_w, center_x + half_w
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img.shape[0], y2)
    x2 = min(img.shape[1], x2)
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] != height * 2 or patch.shape[1] != width * 2:
        patch = cv2.resize(patch, (width * 2, height * 2))
    return patch


def crop_patch(frames, landmarks, mean_face_landmarks, stable_pnts_ids, std_size,
               window_margin, start_idx, stop_idx, crop_height, crop_width):
    margin = min(len(frames), window_margin)
    q_frame, q_landmarks = deque(), deque()
    sequence = []
    transform = None

    for frame_idx, (frame, lm) in enumerate(zip(frames, landmarks)):
        q_landmarks.append(lm)
        q_frame.append(frame)

        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()

            trans_frame, transform = warp_img(
                smoothed_landmarks[stable_pnts_ids, :],
                mean_face_landmarks[stable_pnts_ids, :],
                cur_frame,
                std_size,
            )
            trans_landmarks = transform(cur_landmarks)
            patch = cut_patch(
                trans_frame,
                trans_landmarks[start_idx:stop_idx],
                crop_height // 2,
                crop_width // 2,
            )
            sequence.append(patch)

    # Flush remaining frames
    while q_frame:
        cur_frame = q_frame.popleft()
        if transform is not None:
            trans_frame = apply_transform(transform, cur_frame, std_size)
            trans_landmarks = transform(q_landmarks.popleft())
        else:
            # Fallback: just resize
            cur_frame = cv2.resize(cur_frame, (crop_width, crop_height))
            sequence.append(cur_frame)
            q_landmarks.popleft()
            continue
        patch = cut_patch(
            trans_frame,
            trans_landmarks[start_idx:stop_idx],
            crop_height // 2,
            crop_width // 2,
        )
        sequence.append(patch)

    return np.array(sequence) if sequence else None


def write_video_mp4(frames_bgr, output_path, fps=25):
    """Write frames to mp4 using OpenCV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(frames_bgr) == 0:
        return
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames_bgr:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()


def process_video(fa_model, video_path, output_path, mean_face_landmarks, device):
    frames = read_video_frames(video_path)
    if not frames:
        logger.warning(f"No frames: {video_path}")
        return False

    # Batched landmark detection over the whole video (see detect_landmarks_video).
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    landmarks_list = detect_landmarks_video(fa_model, frames_rgb, device)

    # Interpolate missing detections
    preprocessed_landmarks = landmarks_interpolate(landmarks_list)

    if preprocessed_landmarks is None:
        # Fallback: resize the entire frame
        logger.warning(f"No face detected in {video_path}, using resize fallback")
        resized = [cv2.cvtColor(cv2.resize(f, (CROP_WIDTH, CROP_HEIGHT)), cv2.COLOR_BGR2GRAY) for f in frames]
        write_video_mp4(resized, output_path)
        return True

    sequence = crop_patch(
        frames,
        preprocessed_landmarks,
        mean_face_landmarks,
        STABLE_PNTS_IDS,
        STD_SIZE,
        window_margin=WINDOW_MARGIN,
        start_idx=START_IDX,
        stop_idx=STOP_IDX,
        crop_height=CROP_HEIGHT,
        crop_width=CROP_WIDTH,
    )

    if sequence is None or len(sequence) == 0:
        logger.warning(f"Empty crop for {video_path}")
        return False

    # Convert to grayscale
    gray_frames = []
    for patch in sequence:
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray_frames.append(gray)

    write_video_mp4(gray_frames, output_path)
    return True


def main(args):
    mean_face_landmarks = np.load(args.mean_face)
    logger.info(f"mean_face shape: {mean_face_landmarks.shape}")

    device = "cuda" if args.device == "cuda" else "cpu"
    logger.info(f"Loading face_alignment model on {device} ...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=device,
    )
    logger.info("face_alignment model loaded.")

    video_files = sorted(glob.glob(f"{args.input_dir}/**/*.mp4", recursive=True))
    total = len(video_files)
    shard_size = math.ceil(total / args.nshard)
    start, end = args.rank * shard_size, min((args.rank + 1) * shard_size, total)
    video_files = video_files[start:end]
    logger.info(f"rank {args.rank} of {args.nshard}: processing {len(video_files)} / {total} videos")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    n_ok, n_skip, n_fail = 0, 0, 0
    CACHE_CLEAR_INTERVAL = 50  # release PyTorch CUDA cache every N videos
    for i, video_path in enumerate(tqdm(video_files, total=len(video_files))):
        rel = os.path.relpath(os.path.abspath(video_path), input_dir)
        output_path = os.path.join(output_dir, rel)
        if os.path.exists(output_path):
            n_skip += 1
            continue
        try:
            ok = process_video(fa, video_path, output_path, mean_face_landmarks, device)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.warning(f"FAILED {video_path}: {e}")
            n_fail += 1
        if device == "cuda" and i % CACHE_CLEAR_INTERVAL == 0:
            torch.cuda.empty_cache()

    logger.info(f"Done: ok={n_ok}, skip={n_skip}, fail={n_fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop mouth ROI from CelebV-Dub videos")
    parser.add_argument("--input-dir", type=str, required=True, help="Root dir of source .mp4 (symlinks OK)")
    parser.add_argument("--output-dir", type=str, required=True, help="Root dir for cropped mouth .mp4")
    parser.add_argument(
        "--mean-face",
        type=str,
        default="/home/zjw524/datasets/20words_mean_face.npy",
        help="Path to 20words_mean_face.npy",
    )
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args)
