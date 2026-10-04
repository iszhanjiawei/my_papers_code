#!/usr/bin/env python3
"""Evaluate generated speech against its original video using pretrained SyncNet.

No AlignDiT package is imported. Full-frame inputs go through the pinned official
S3FD/scene/track/crop pipeline; precomputed SyncNet crops can be scored explicitly.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from install import ASSETS, REVISION, sha256
from metrics import extract_embeddings, score_embeddings
from scipy.io import wavfile


def run(command, *, env=None, cwd=None, log=None):
    if log is not None:
        with log.open("a") as handle:
            handle.write(json.dumps([str(x) for x in command]) + "\n")
            handle.flush()
            result = subprocess.run(command, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        error_text = log.read_text()[-3000:] if result.returncode else ""
    else:
        result = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, check=False)
        error_text = result.stderr
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}\n{error_text[-3000:]}")
    return result.stdout


def executable(name):
    sibling = Path(sys.executable).parent / name
    candidate = str(sibling) if sibling.is_file() else shutil.which(name)
    if not candidate:
        raise FileNotFoundError(f"{name} not found; run install.py with this Python environment")
    run([candidate, "-version"])
    return candidate


def probe(path, stream_type, ffprobe):
    data = json.loads(run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = [stream for stream in data["streams"] if stream["codec_type"] == stream_type]
    if not streams:
        raise ValueError(f"No {stream_type} stream: {path}")
    stream = streams[0]
    duration = float(stream.get("duration", data.get("format", {}).get("duration", "nan")))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid {stream_type} duration: {path}")
    return {**stream, "duration_seconds": duration}


def safe_id(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Expected a nonempty relative sample ID: {value!r}")
    return str(path)


def load_samples(args):
    if args.manifest:
        base = args.manifest.resolve().parent
        samples = []
        for line in args.manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            # Missing audio must not silently substitute the video's ground truth.
            audio = entry["audio"]
            if audio is not None and (not isinstance(audio, str) or not audio.strip()):
                raise ValueError("Manifest audio must be a nonempty path string or explicit null for ground truth")
            samples.append(
                {
                    "id": safe_id(entry["id"]),
                    "video": (base / entry["video"]).resolve(),
                    "audio": (base / audio).resolve() if audio is not None else None,
                }
            )
    elif args.test_list:
        if not args.video_root or not args.audio_root:
            raise ValueError("--test-list requires --video-root and --audio-root")
        samples = []
        for line in args.test_list.read_text().splitlines():
            if not line.strip():
                continue
            clip = safe_id(line.strip())
            sample_id = clip if clip.startswith(args.split + "/") else f"{args.split}/{clip}"
            samples.append(
                {
                    "id": sample_id,
                    "video": (args.video_root / (sample_id + args.video_suffix)).resolve(),
                    "audio": (args.audio_root / (sample_id + ".wav")).resolve(),
                }
            )
    else:
        samples = [
            {
                "id": args.video.stem,
                "video": args.video.resolve(),
                "audio": args.audio.resolve() if args.audio else None,
            }
        ]
    ids = [sample["id"] for sample in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sample list must be nonempty and contain unique full relative IDs")
    return samples[: args.limit] if args.limit is not None else samples


def score_crop(model, video, work, args, ffmpeg, audio=None, log=None):
    work.mkdir(parents=True)
    run(
        [ffmpeg, "-y", "-v", "error", "-i", str(video), "-threads", "1", "-f", "image2", str(work / "%06d.jpg")],
        log=log,
    )
    run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(audio or video),
            "-async",
            "1",
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(work / "audio.wav"),
        ],
        log=log,
    )
    paths = sorted(work.glob("*.jpg"))
    if not paths:
        raise ValueError("Video has no decoded frames")
    frames = np.stack([cv2.imread(str(path)) for path in paths])
    rate, signal = wavfile.read(work / "audio.wav")
    if rate != 16000:
        raise ValueError("Audio conversion did not produce 16kHz")
    visual, acoustic = extract_embeddings(model, frames, signal, args.device, args.batch_size)
    scores = score_embeddings(visual, acoustic, args.vshift)
    scores.update(video_frames=len(frames), audio_samples=len(signal))
    return scores


def evaluate_sample(sample, model, args, repo, work, log, ffmpeg, ffprobe):
    video, audio = sample["video"], sample["audio"]
    for path in [video, audio]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    video_info = probe(video, "video", ffprobe)
    audio_info = probe(audio or video, "audio", ffprobe)
    delta = audio_info["duration_seconds"] - video_info["duration_seconds"]
    if args.duration_policy == "strict" and abs(delta) > args.duration_tolerance:
        raise ValueError(
            f"Audio/video duration mismatch {delta:+.4f}s exceeds {args.duration_tolerance}s; "
            "check pairing/prompt removal, or explicitly use --duration-policy overlap"
        )
    metadata = {
        "video_duration_seconds": video_info["duration_seconds"],
        "audio_duration_seconds": audio_info["duration_seconds"],
        "audio_minus_video_seconds": delta,
        "audio_source": "external_wav" if audio else "video_embedded_audio",
    }
    if args.input_kind == "syncnet-crop":
        num, den = video_info["r_frame_rate"].split("/")
        if (video_info["width"], video_info["height"]) != (224, 224) or abs(float(num) / float(den) - 25) > 1e-5:
            raise ValueError(
                "--input-kind syncnet-crop requires an official 224x224, 25fps face crop. "
                "AV-HuBERT mouth crops are not interchangeable."
            )
        tracks = [{"track": "input", **score_crop(model, video, work / "score", args, ffmpeg, audio, log)}]
    else:
        # Explicit stream mapping prevents accidentally scoring the video's GT audio.
        paired = work / "paired.mkv"
        run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-i",
                str(audio or video),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(paired),
            ],
            log=log,
        )
        env = dict(os.environ)
        env["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + env.get("PATH", "")
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["MKL_NUM_THREADS"] = str(args.threads)
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        run(
            [
                sys.executable,
                str(Path(__file__).with_name("pipeline.py")),
                str(repo / "run_pipeline.py"),
                "--videofile",
                str(paired),
                "--reference",
                "clip",
                "--data_dir",
                str(work / "pipeline"),
                "--min_track",
                str(args.min_track),
                "--min_face_size",
                str(args.min_face_size),
                "--facedet_scale",
                str(args.facedet_scale),
                "--crop_scale",
                str(args.crop_scale),
                "--num_failed_det",
                str(args.num_failed_det),
            ],
            cwd=repo,
            env=env,
            log=log,
        )
        crops = sorted((work / "pipeline" / "pycrop" / "clip").glob("*.avi"))
        if not crops:
            raise ValueError(
                "No valid face track. Check raw video, face size, and --min-track; "
                "do not substitute a mouth-only crop or silently discard this sample."
            )
        tracks = []
        for index, crop in enumerate(crops):
            tracks.append(
                {"track": crop.stem, **score_crop(model, crop, work / f"score_{index}", args, ffmpeg, log=log)}
            )
    if args.track_policy == "single" and len(tracks) != 1:
        raise ValueError(f"Expected one face track, found {len(tracks)}; review video or set an explicit track policy")
    # Video duration alone selects the face; do not select by generated audio's LSE.
    chosen = max(range(len(tracks)), key=lambda i: tracks[i]["video_frames"])
    selected = tracks if args.track_policy == "mean" else [tracks[chosen]]
    return {
        **metadata,
        "lse_d": float(np.mean([t["lse_d"] for t in selected])),
        "lse_c": float(np.mean([t["lse_c"] for t in selected])),
        "num_tracks": len(tracks),
        "selected_tracks": [t["track"] for t in selected],
        "tracks": tracks,
    }


def summarize(rows, requested, protocol):
    good = [row for row in rows if row["status"] == "ok"]
    return {
        "requested": requested,
        "processed": len(rows),
        "succeeded": len(good),
        "failed": len(rows) - len(good),
        "coverage": len(good) / requested,
        "complete": len(good) == requested,
        "lse_d": float(np.mean([row["lse_d"] for row in good])) if good else None,
        "lse_c": float(np.mean([row["lse_c"] for row in good])) if good else None,
        "aggregation": "equal mean over successful clips; incomplete coverage is not a full-test result",
        "protocol": protocol,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--manifest", type=Path, help="JSONL: id, video, audio; paths relative to manifest; null audio = embedded GT"
    )
    inputs.add_argument("--test-list", type=Path, help="CelebVDub list: folder/clip or test/folder/clip")
    inputs.add_argument("--video", type=Path, help="Single video; omit --audio only for embedded GT sanity checks")
    parser.add_argument("--audio", type=Path, help="Generated WAV for --video; must exclude prompt/reference audio")
    parser.add_argument("--video-root", type=Path, help="Root containing test/folder/clip.mp4 (full face/raw video)")
    parser.add_argument("--audio-root", "--gen-wav-dir", type=Path, help="Root containing test/folder/clip.wav")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-suffix", default=".mp4")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/syncnet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory; refuses to overwrite results"
    )
    parser.add_argument("--input-kind", choices=["full-frame", "syncnet-crop"], default="full-frame")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Select GPU via CUDA_VISIBLE_DEVICES; default cuda when available",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vshift", type=int, default=15, help="Search +/- frames at 25fps")
    parser.add_argument(
        "--min-track", type=int, default=10, help="Require > this many face detections (upstream default 100)"
    )
    parser.add_argument("--min-face-size", type=int, default=100)
    parser.add_argument("--facedet-scale", type=float, default=0.25)
    parser.add_argument("--crop-scale", type=float, default=0.40)
    parser.add_argument("--num-failed-det", type=int, default=25)
    parser.add_argument(
        "--track-policy",
        choices=["longest", "single", "mean"],
        default="longest",
        help="longest video track (ties: first), require single track, or equal track mean",
    )
    parser.add_argument("--duration-policy", choices=["strict", "overlap"], default="strict")
    parser.add_argument("--duration-tolerance", type=float, default=0.10)
    parser.add_argument("--limit", type=int, help="Smoke-test first N entries only; summary records this subset")
    parser.add_argument("--keep-work", action="store_true", help="Retain official crops/frames for inspection (large)")
    args = parser.parse_args()
    if args.audio and not args.video:
        parser.error("--audio is only valid with --video")
    for name in ["batch_size", "threads", "vshift", "min_track", "min_face_size", "facedet_scale", "num_failed_det"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.min_track < 5
        or args.duration_tolerance < 0
        or args.crop_scale < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Require min-track >= 5, tolerance/crop-scale >= 0 and limit >= 1")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    samples = load_samples(args)
    ffmpeg, ffprobe = executable("ffmpeg"), executable("ffprobe")
    repo = args.asset_dir.resolve() / "syncnet_python"
    checkpoint = args.asset_dir.resolve() / "syncnet_v2.model"
    if not checkpoint.is_file() or not (repo / "SyncNetModel.py").is_file():
        raise FileNotFoundError("Missing SyncNet installation; run evaluation/lse/install.py first")
    revision = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
    if revision != REVISION or run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise RuntimeError("SyncNet source differs from the pinned clean checkout; run install.py to diagnose")
    required = [checkpoint]
    if args.input_kind == "full-frame":
        required.append(repo / "detectors" / "s3fd" / "weights" / "sfd_face.pth")
    for path in required:
        if sha256(path) != ASSETS[path.name]:
            raise RuntimeError(f"Pretrained checkpoint checksum mismatch: {path}")
    sys.path.insert(0, str(repo))
    from SyncNetModel import S

    model = S().to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(
        {
            "metric": "SyncNet v2 / Wav2Lip LSE, mean-distance shift search, zero-padded audio embeddings",
            "fps": 25,
            "sample_rate": 16000,
            "checkpoint_sha256": ASSETS[checkpoint.name],
            "s3fd_sha256": ASSETS["sfd_face.pth"] if args.input_kind == "full-frame" else None,
            "upstream_revision": revision,
            "pipeline_audio_codec": "pcm_s16le (explicit lossless AVI audio; avoids default MP3 delay/padding)",
            "opencv_version": cv2.__version__,
            "ffmpeg": run([ffmpeg, "-version"]).splitlines()[0],
            "packages": {
                key: importlib.metadata.version(key)
                for key in ["torch", "numpy", "scipy", "python_speech_features", "scenedetect"]
            },
        }
    )
    installation = args.asset_dir / "installation.json"
    if installation.is_file():
        protocol["installation"] = json.loads(installation.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["results.jsonl", "summary.json", "protocol.json"]):
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "logs").mkdir(exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    rows = []
    with (output / "results.jsonl").open("x") as handle:
        for index, sample in enumerate(samples):
            key = f"{index:05d}_{hashlib.sha256(sample['id'].encode()).hexdigest()[:12]}"
            log = output / "logs" / f"{key}.log"
            temp = None
            if args.keep_work:
                work = output / "work" / key
                work.mkdir(parents=True)
            else:
                temp = tempfile.TemporaryDirectory(prefix=f"lse_{key}_")
                work = Path(temp.name)
            row = {key: str(value) if isinstance(value, Path) else value for key, value in sample.items()}
            try:
                result = evaluate_sample(sample, model, args, repo, work, log, ffmpeg, ffprobe)
                row.update(status="ok", **result)
                print(
                    f"[{index + 1}/{len(samples)}] {sample['id']} LSE-D={result['lse_d']:.5f} LSE-C={result['lse_c']:.5f}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 -- preserve failure coverage, then exit nonzero
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                print(f"[{index + 1}/{len(samples)}] ERROR {sample['id']}: {error}", file=sys.stderr, flush=True)
            finally:
                if temp is not None:
                    temp.cleanup()
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            rows.append(row)
            summary = summarize(rows, len(samples), protocol)
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
    print(json.dumps({key: value for key, value in summary.items() if key != "protocol"}, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
