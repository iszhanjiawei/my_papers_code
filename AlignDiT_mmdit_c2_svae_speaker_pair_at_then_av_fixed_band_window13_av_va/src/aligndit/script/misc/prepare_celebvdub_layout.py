"""
Reorganize the CelebV-Dub dataset into an LRS3-style layout expected by AlignDiT.

Original CelebV-Dub layout:
    <src>/<split>/<id>/<clip>.{mp4,wav,txt,npy}
    - .txt content is "Text: <UPPERCASE TRANSCRIPT>"

Target layout (under <dst>):
    <dst>/audio/<split>/<id>/<clip>.wav   -> symlink to original .wav
    <dst>/video/<split>/<id>/<clip>.mp4   -> symlink to original .mp4
    <dst>/text/<split>/<id>/<clip>.txt    -> cleaned transcript (prefix stripped, lower-cased)

The training pipeline locates mel / avhubert features by replacing
"/audio/" in the wav path with "/mel_tacotron/" and "/avhubert_video_feat/",
so the audio path must live under an ".../audio/..." directory.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


def clean_text(raw: str) -> str:
    """Strip the leading 'Text:' prefix and lower-case the transcript."""
    raw = raw.strip()
    # Files look like: "Text: SOME UPPERCASE SENTENCE"
    if raw.lower().startswith("text:"):
        raw = raw[raw.find(":") + 1 :]
    return raw.strip().lower()


def process_speaker(args):
    speaker_dir, split, dst_dir = args
    speaker = speaker_dir.name
    n_done = 0

    audio_out = Path(dst_dir) / "audio" / split / speaker
    video_out = Path(dst_dir) / "video" / split / speaker
    text_out = Path(dst_dir) / "text" / split / speaker
    audio_out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)

    for wav_path in speaker_dir.glob("*.wav"):
        stem = wav_path.stem
        mp4_path = speaker_dir / f"{stem}.mp4"
        txt_path = speaker_dir / f"{stem}.txt"

        # Require all three modalities to be present.
        if not (mp4_path.exists() and txt_path.exists()):
            continue

        try:
            text = clean_text(open(txt_path, "r", encoding="utf-8", errors="ignore").readline())
        except Exception:
            continue
        if len(text) == 0:
            continue

        # symlink audio
        audio_link = audio_out / f"{stem}.wav"
        if not audio_link.exists():
            os.symlink(wav_path.resolve(), audio_link)
        # symlink video
        video_link = video_out / f"{stem}.mp4"
        if not video_link.exists():
            os.symlink(mp4_path.resolve(), video_link)
        # write cleaned text
        with open(text_out / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")

        n_done += 1

    return n_done


def main():
    parser = argparse.ArgumentParser(description="Reorganize CelebV-Dub into LRS3-style layout.")
    parser.add_argument("--src", type=str, default=os.environ.get("ROOT_PREFIX", "") + "/zjw524/datasets/CelebV-Dub")
    parser.add_argument(
        "--dst",
        type=str,
        default="data/CelebVDub",
        help="Destination root (relative to project root or absolute).",
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_dir = args.dst

    for split in args.splits:
        split_dir = src_root / split
        if not split_dir.is_dir():
            print(f"[skip] split dir not found: {split_dir}")
            continue

        speaker_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        print(f"[{split}] {len(speaker_dirs)} speakers -> reorganizing ...")

        total = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            jobs = [(sd, split, dst_dir) for sd in speaker_dirs]
            for n in tqdm(ex.map(process_speaker, jobs), total=len(jobs)):
                total += n
        print(f"[{split}] linked {total} clips")


if __name__ == "__main__":
    main()
