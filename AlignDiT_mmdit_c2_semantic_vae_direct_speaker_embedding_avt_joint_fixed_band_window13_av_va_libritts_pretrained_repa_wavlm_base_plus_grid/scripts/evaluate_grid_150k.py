"""Evaluate the isolated REPA experiment's 150k EMA in S1 and both S2 modes.

The checkpoint uses all four GPUs and four independent single-utterance workers
per GPU. Complete and verify each phase before the next. Existing artifacts can
be resumed with the same output directories and worker count.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = "AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char"
PHASES = (("s1", "reference-target", "s1"),
          ("s2", "reference-target", "s2_reference_target"),
          ("s2", "target-only", "s2_target_only"))


def publish(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def run(args):
    os.chdir(PROJECT)
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or args.workers_per_gpu < 1:
        raise ValueError("Supply four distinct GPU indices and positive workers-per-gpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = args.output_root / "GRID_avt_fixed13_libritts_repa_evaluation_batch.json"
    state = {"started_at": datetime.datetime.now().astimezone().isoformat(),
             "pid": os.getpid(), "project": str(PROJECT), "gpus": gpus,
             "workers_per_gpu": args.workers_per_gpu, "complete": False, "runs": []}
    children = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for setting, mode, label in PHASES:
            state["phase"] = label
            children = []
            current = []
            for step in (150000,):
                checkpoint = args.checkpoint_root / f"model_{step}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                output = args.output_root / f"GRID_avt_fixed13_libritts_repa_{label}_{step//1000}k_seed0_nfe32_cfg5_2"
                output.mkdir(parents=True, exist_ok=True)
                item = {"setting": setting, "text_conditioning": mode, "step": step,
                        "output_dir": str(output), "checkpoint": str(checkpoint), "verified": False}
                current.append(item)
                state["runs"].append(item)
                completed = output / "pipeline_status.json"
                already_complete = completed.exists() and json.loads(completed.read_text()).get("complete") is True
                if already_complete:
                    print(f"Reaudit completed pipeline: {output}", flush=True)
                    continue
                command = [sys.executable, "-u", "-m", "aligndit.script.eval.run_grid_s1",
                           "--setting", setting, "--text-conditioning", mode,
                           "--checkpoint", str(checkpoint), "--step", str(step),
                           "--output-dir", str(output), "--gpus", ",".join(gpus),
                           "--workers-per-gpu", str(args.workers_per_gpu)]
                with (output / "pipeline.log").open("a") as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
                item["pid"] = proc.pid
                children.append(proc)
                print(json.dumps(item), flush=True)
            publish(report, state)
            while any(proc.poll() is None for proc in children):
                failed = [proc for proc in children if proc.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Pipeline failed: {[proc.pid for proc in failed]}")
                time.sleep(2)
            if any(proc.returncode for proc in children):
                raise RuntimeError("A GRID pipeline failed; inspect pipeline.log")
            for item in current:
                output = Path(item["output_dir"])
                summary = json.loads((output / "evaluation_summary.json").read_text())
                if summary.get("count") != 3274 or summary.get("complete") is not True or summary["checkpoint"]["update"] != item["step"]:
                    raise RuntimeError(f"Incomplete or incorrect evaluation: {output}")
                subprocess.run([sys.executable, "-u", "scripts/verify_grid_evaluation.py",
                                "--output-dir", str(output)], check=True)
                audit = json.loads((output / "verification.json").read_text())
                if audit.get("status") != "PASS" or audit.get("complete") is not True:
                    raise RuntimeError(f"Independent verification failed: {output}")
                item.update(verified=True, metrics=summary["metrics"])
                publish(report, state)
            print(f"Completed and independently verified phase: {label}", flush=True)
        state.update(complete=True, phase="complete", finished_at=datetime.datetime.now().astimezone().isoformat())
        publish(report, state)
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        publish(report, state)
        raise
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    prefix = os.environ.get("ROOT_PREFIX", "")
    data = Path(prefix + "/zjw524/projects/data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=data / "evaluations")
    parser.add_argument("--checkpoint-root", type=Path, default=data / "ckpts" / EXPERIMENT)
    run(parser.parse_args())
