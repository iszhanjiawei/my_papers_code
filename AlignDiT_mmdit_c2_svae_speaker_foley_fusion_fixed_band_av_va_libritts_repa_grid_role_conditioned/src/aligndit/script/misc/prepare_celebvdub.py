"""
Build AlignDiT training metadata for the CelebV-Dub dataset.

Expects the LRS3-style layout produced by ``prepare_celebvdub_layout.py``:
    data/CelebVDub/audio/<split>/<id>/<clip>.wav
    data/CelebVDub/text/<split>/<id>/<clip>.txt   (cleaned transcript)

Outputs (mirrors prepare_lrs3.py):
    data/CelebVDub_<tokenizer>/raw.arrow
    data/CelebVDub_<tokenizer>/duration.json
    data/CelebVDub_<tokenizer>/vocab.txt
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


sys.path.append(os.getcwd())


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    audio_lists = list(Path(audio_dir).rglob("*.wav"))
    vocab_set = set()
    for line in audio_lists:
        text_path = os.path.splitext(str(line).replace("/audio/", "/text/"))[0] + ".txt"
        if not os.path.exists(text_path):
            continue
        text = open(text_path, "r", encoding="utf-8").readline().strip().lower()
        if len(text) == 0:
            continue
        try:
            duration = sf.info(line).duration
        except Exception:
            continue
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main(args):
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    max_workers = args.max_workers

    result, duration_list, text_vocab_set = [], [], set()

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for subset in args.splits:
        subset_path = Path(os.path.join(dataset_dir, subset))
        if not subset_path.is_dir():
            print(f"[skip] {subset_path} not found")
            continue
        for audio_dir in subset_path.iterdir():
            if audio_dir.is_dir():
                futures.append(executor.submit(deal_with_audio_dir, audio_dir))

    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    with open(f"{save_dir}/vocab.txt", "w", encoding="utf-8") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nsample count: {len(result)}")
    print(f"total {sum(duration_list) / 3600:.2f} hours")
    print(f"vocab size: {len(text_vocab_set)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CelebV-Dub metadata for AlignDiT")
    parser.add_argument("--tokenizer", type=str, default="char")
    parser.add_argument("--dataset_dir", type=str, default="data/CelebVDub/audio")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--max_workers", type=int, default=36)
    args = parser.parse_args()

    dataset_name = f"CelebVDub_{args.tokenizer}"
    args.save_dir = str(files("aligndit").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {args.save_dir}\n")
    main(args)
