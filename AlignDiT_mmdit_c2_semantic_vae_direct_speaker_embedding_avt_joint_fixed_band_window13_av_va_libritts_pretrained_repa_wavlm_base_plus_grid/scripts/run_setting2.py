"""Run Setting2 generation and shared metrics on one explicitly selected GPU."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def main():
    if "--tool" in sys.argv:
        index = sys.argv.index("--tool")
        wrapper = argparse.ArgumentParser()
        wrapper.add_argument("--memory-fraction", type=float, default=0.22)
        options = wrapper.parse_args(sys.argv[1:index])
        tool, arguments = sys.argv[index + 1], sys.argv[index + 2:]
        import torch
        torch.cuda.set_per_process_memory_fraction(options.memory_fraction, 0)
        sys.argv = [tool, *arguments]
        runpy.run_path(tool, run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--memory-fraction", type=float, default=0.22)
    args, extra = parser.parse_known_args()
    if extra or args.output_dir is None or args.checkpoint is None:
        parser.error("Require --checkpoint and --output-dir")
    prefix = os.environ.get("ROOT_PREFIX", "")
    python = prefix + "/zjw524/ENTER/envs/aligndit/bin/python"
    bench = PROJECT.parents[1] / "Video-to-Speech-benchmark"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "evaluation_logs"
    logs.mkdir(exist_ok=True)
    base_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")

    def execute(stage, command, infer=False):
        env = dict(base_env)
        if infer:
            env["PYTHONPATH"] = str(PROJECT / "src")
        else:
            env.pop("PYTHONPATH", None)
        print("Starting " + stage, flush=True)
        with (logs / (stage + ".log")).open("a") as stream:
            subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Completed " + stage, flush=True)

    execute("inference", [python, "-u", "-m", "aligndit.script.eval.infer_setting2_dubbing",
        "--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(output), "--gpu", args.gpu,
        "--memory-fraction", str(args.memory_fraction)], infer=True)
    for metric in ("spksim", "wer", "emosim", "avsync"):
        if (output / f"setting2_{metric}.json").exists():
            print("Retaining completed metric for final audit: " + metric, flush=True)
            continue
        if metric == "avsync":
            extraction = bench / "shared/aligndit_baseline/src/aligndit/script/misc/extract_avhubert.py"
            env = dict(base_env, PYTHONPATH=str(bench / "shared/aligndit_baseline/src") + ":" + prefix + "/zjw524/projects/data/av_hubert/fairseq/fairseq")
            command = [python, "-u", str(Path(__file__).resolve()), "--memory-fraction", str(args.memory_fraction),
                "--tool", str(extraction.resolve()), "--nshard", "1", "--rank", "0",
                "--v-input-dir", str(bench / "setting2/media/mouth_video"), "--a-input-dir", str(output),
                "--output-dir", str(output / "avhubert_feat"), "--ckpt-path", str(bench / "shared/large_vox_iter5.pt"),
                "--user_dir", prefix + "/zjw524/projects/data/av_hubert/avhubert/avhubert"]
            with (logs / "extract_avsync.log").open("a") as stream:
                subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        execute(metric, [python, "-u", str(Path(__file__).resolve()), "--memory-fraction", str(args.memory_fraction),
            "--tool", str(bench / "scripts/evaluate_setting2.py"), "--generated", str(output), "--metric", metric])
    execute("audit", [python, "-u", "scripts/audit_setting2.py", "--output-dir", str(output)], infer=True)
    print("Full original 115-target Setting2 evaluation and audit complete", flush=True)


if __name__ == "__main__":
    main()
