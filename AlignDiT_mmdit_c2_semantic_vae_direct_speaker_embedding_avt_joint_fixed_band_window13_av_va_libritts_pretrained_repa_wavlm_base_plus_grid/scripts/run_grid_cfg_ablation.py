"""Run the fixed GRID 100k CFG diagnostic on detached GPU workers."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


VARIANTS = (
    "cfg_t5_v2", "cfg_t3_v2", "cfg_t1_v2", "cfg_t5_v1",
    "cfg_t5_v0", "cfg_t0_v0", "drop_text_t0_v0", "drop_video_t0_v0",
)
STAGES = ("prepare", "infer", "sim", "wer", "emosim", "avsync", "merge", "verify")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--start-stage", choices=STAGES, default="prepare")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(g.isdigit() for g in gpus):
        parser.error("Specify four distinct GPU indices; two variant workers use each GPU")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    project = scripts.parent
    worker = scripts / "grid_cfg_ablation.py"
    verifier = scripts / "verify_grid_cfg_ablation.py"
    env_base = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    status_path = root / "pipeline_status.json"
    started = datetime.datetime.now().astimezone().isoformat()

    def status(stage, **fields):
        data = {"stage": stage, "complete": False, "pid": os.getpid(),
                "started": started, "updated": datetime.datetime.now().astimezone().isoformat(),
                "gpus": gpus, **fields}
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(status_path)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    jobs = []
    stage = "starting"
    try:
        for stage in STAGES[STAGES.index(args.start_stage):]:
            print(f"{datetime.datetime.now().astimezone().isoformat()} stage={stage}", flush=True)
            status(stage)
            jobs = []
            if stage in ("prepare", "merge", "verify"):
                commands = [(stage, None, [sys.executable, "-u", str(verifier if stage == "verify" else worker),
                                          "--output-dir", str(root)] +
                             ([] if stage == "verify" else ["--stage", stage]))]
            else:
                commands = []
                for index, variant in enumerate(VARIANTS):
                    command = [sys.executable, "-u", str(worker), "--output-dir", str(root),
                               "--stage", "infer" if stage == "infer" else "metrics", "--variant", variant]
                    if stage != "infer":
                        command += ["--task", stage]
                    commands.append((variant, gpus[index % 4], command))
            for name, gpu, command in commands:
                env = dict(env_base)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                logfile = root / "logs" / f"{stage}_{name}.log"
                logfile.parent.mkdir(parents=True, exist_ok=True)
                with logfile.open("a") as handle:
                    proc = subprocess.Popen(command, cwd=project, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((proc, name, gpu, logfile))
                print(f"  {name}: gpu={gpu} pid={proc.pid} log={logfile}", flush=True)
            status(stage, workers=[{"pid": p.pid, "variant": name, "gpu": gpu, "log": str(log)}
                                   for p, name, gpu, log in jobs])
            pending = list(jobs)
            while pending:
                for job in pending[:]:
                    proc, name, gpu, logfile = job
                    code = proc.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{stage}/{name} exited {code}; inspect {logfile}")
                        pending.remove(job)
                if pending:
                    time.sleep(1)
        status("complete", complete=True)
    except BaseException as error:
        status(stage, error=str(error))
        raise
    finally:
        for proc, *_ in jobs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, *_ in jobs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
