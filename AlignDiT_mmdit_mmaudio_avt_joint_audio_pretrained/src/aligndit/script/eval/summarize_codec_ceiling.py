"""Summarize codec reconstruction and CelebV-Dub metric outputs."""

import argparse
import json
import re
from pathlib import Path


CODEC_DIRS = (
    "mel_hifigan",
    "acoustic_vae_dim64_sample",
    "semantic_vae_600k_sample",
    "semantic_vae_1000k_sample",
)
METRICS = ("sim", "wer", "emosim", "avsync")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--codec-dirs", nargs="+", default=list(CODEC_DIRS))
    return parser.parse_args()


def read_metric(path: Path, metric: str):
    if not path.exists():
        return None
    match = re.search(rf"(?m)^{metric.upper()}:\s*([-+0-9.eE]+)\s*$", path.read_text())
    return float(match.group(1)) if match else None


def format_value(value):
    return "N/A" if value is None else f"{value:.5f}"


def main():
    args = get_args()
    rows = []
    for codec_dir in args.codec_dirs:
        directory = args.output_root / codec_dir
        reconstruction_path = directory / "_reconstruction_summary.json"
        reconstruction = json.loads(reconstruction_path.read_text()) if reconstruction_path.exists() else {}
        row = {
            "codec": codec_dir,
            "samples": reconstruction.get("samples"),
            "snr_db": reconstruction.get("snr_db_mean"),
            "native_length_delta_mean": reconstruction.get("native_length_delta_mean"),
        }
        for metric in METRICS:
            row[metric] = read_metric(directory / f"_{metric}_results.jsonl", metric)
        rows.append(row)

    summary_path = args.output_root / "_codec_ceiling_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    print("| Codec | N | SNR(dB) | SPKSIM↑ | WER↓ | EMOSIM↑ | AVSync↑ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['codec']} | {row['samples'] or 'N/A'} | {format_value(row['snr_db'])} | "
            f"{format_value(row['sim'])} | {format_value(row['wer'])} | "
            f"{format_value(row['emosim'])} | {format_value(row['avsync'])} |"
        )
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
