# Evaluate AlignDiT on CelebV-Dub test set (Setting 1)

import argparse
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np
from jiwer import compute_measures

from aligndit.script.eval.utils import get_celebvdub_test, run_asr_wer, run_avsync, run_emoembed, run_emosim
from f5_tts.eval.utils_eval import run_sim


rel_path = str(files("aligndit").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eval_task",
        type=str,
        default="sim",
        choices=["sim", "wer", "emosim", "emoembed", "avsync"],
    )
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument(
        "--wavlm_ckpt",
        type=str,
        default=os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth",
    )
    parser.add_argument(
        "--wavlm_base_ckpt",
        type=str,
        default=os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/wavlm_large_s3prl.pt",
    )
    parser.add_argument(
        "--asr_ckpt",
        type=str,
        default=os.environ.get("ROOT_PREFIX", "") + "/zjw524/alignDiT_pretrain_models/large-v3.pt",
    )
    parser.add_argument(
        "--emo_ckpt",
        type=str,
        default=os.environ.get("ROOT_PREFIX", "") + "/zjw524/projects/data/emotion2vec_plus_large",
    )
    parser.add_argument("--gt_av_feat", type=str, default="data/CelebVDub/avhubert_feat")
    parser.add_argument("--test-list", type=str, default=None, help="Override the CelebV-Dub Setting 1 list")
    parser.add_argument("--celebvdub-root", type=str, default=None, help="Override the CelebV-Dub dataset root")
    parser.add_argument("--eval_ground_truth", action="store_true", help="Evaluate GT audio (sanity check)")
    return parser.parse_args()


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir

    metalst = args.test_list or rel_path + "/data/celebvdub_test_s1.lst"
    celebvdub_path = args.celebvdub_root or rel_path + "/data/CelebVDub"

    gpus = list(range(args.gpu_nums))
    test_set = get_celebvdub_test(metalst, gen_wav_dir, gpus, celebvdub_path, eval_ground_truth=args.eval_ground_truth)

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"

    full_results = []
    metrics = []

    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            pool_args = [(rank, lang, sub_test_set, args.asr_ckpt) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, pool_args)
            for r in results:
                full_results.extend(r)

        refs = [r["truth"] for r in full_results]
        hypos = [r["hypo"] for r in full_results]
        metric = compute_measures(refs, hypos)["wer"]
        with open(result_path, "w") as f:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in full_results)
            metric = round(metric, 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            pool_args = [
                (rank, sub_test_set, args.wavlm_ckpt, args.wavlm_base_ckpt) for (rank, sub_test_set) in test_set
            ]
            results = pool.map(run_sim, pool_args)
            for r in results:
                full_results.extend(r)

        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line["sim"])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    elif eval_task == "emosim":
        with mp.Pool(processes=len(gpus)) as pool:
            pool_args = [(rank, sub_test_set, args.emo_ckpt) for (rank, sub_test_set) in test_set]
            results = pool.map(run_emosim, pool_args)
            for r in results:
                full_results.extend(r)

        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line["emosim"])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    elif eval_task == "emoembed":
        with mp.Pool(processes=len(gpus)) as pool:
            pool_args = [(rank, sub_test_set, args.emo_ckpt) for (rank, sub_test_set) in test_set]
            results = pool.map(run_emoembed, pool_args)
            for r in results:
                full_results.extend(r)

        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line["emoembed"])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    elif eval_task == "avsync":
        gen_av_feat = f"{gen_wav_dir}/avhubert_feat"
        with mp.Pool(processes=len(gpus)) as pool:
            pool_args = [(rank, sub_test_set, args.gt_av_feat, gen_av_feat) for (rank, sub_test_set) in test_set]
            results = pool.map(run_avsync, pool_args)
            for r in results:
                full_results.extend(r)

        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line["avsync"])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    else:
        raise ValueError(f"Unknown eval_task: {eval_task}")

    print(f"\nTotal {len(full_results)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
