"""Native Setting 1 CFG sweep; partition workloads without changing protocols.

Run with this experiment's PYTHONPATH=src and its usual aligndit environment.
Generated files live outside Git. Failed stages are never reported as complete.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
PREFIX = os.environ.get("ROOT_PREFIX", "")
WORKSPACE = Path(PREFIX + "/zjw524/projects/alignDiT_idea6")
DATA = Path(PREFIX + "/zjw524/projects/data")
ASSETS = Path(PREFIX + "/zjw524/alignDiT_pretrain_models")
CKPTS = DATA / "ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
CONFIG = PROJECT / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml"
LIST = DATA / "celebvdub_test_s1.lst"
SCRIPT = PROJECT / "src/aligndit/script"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dispatch(jobs, logs, slots=8):
    """Each GPU has its own bounded queue; every failed child aborts the stage."""
    waiting = list(jobs)
    running = []
    failed = []
    last = 0
    logs.mkdir(parents=True, exist_ok=True)
    while waiting or running:
        for gpu in (0, 1):
            while sum(p[1] == gpu for p in running) < slots:
                index = next((i for i, j in enumerate(waiting) if j[1] == gpu), None)
                if index is None or failed:
                    break
                name, _, argv, extra = waiting.pop(index)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(PROJECT / "src"))
                env.update(extra)
                handle = (logs / (name + ".log")).open("w")
                process = subprocess.Popen([sys.executable, "-u", *map(str, argv)],
                                           cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                running.append((name, gpu, process, handle))
                print(f"START {name} GPU={gpu} PID={process.pid}", flush=True)
        for item in list(running):
            name, gpu, process, handle = item
            status = process.poll()
            if status is not None:
                handle.close()
                running.remove(item)
                print(f"END {name} returncode={status}", flush=True)
                if status:
                    failed.append(name)
        if failed and not running:
            raise RuntimeError(f"Failed workers: {failed}; see {logs}")
        if time.time() - last > 30:
            print(f"PROGRESS running={len(running)} queued={len(waiting)} failures={failed}", flush=True)
            last = time.time()
        time.sleep(1)


def inference_args(output, cfg, shard, nshard):
    return [SCRIPT / "eval/infer_celebvdub_semantic_vae_s1.py", "--checkpoint", CKPTS / "model_150000.pt",
            "--step", "150000", "--config", CONFIG, "--output-dir", output,
            "--cfg-text", "5", "--cfg-video", str(cfg), "--seed", "0", "--nfe", "32",
            "--shard", str(shard), "--nshard", str(nshard)]


def validate_audio(output, clips):
    import soundfile as sf
    manifests = [json.loads(x) for x in (DATA / "CelebVDub_svae1000k_sample_seed666_fp32/manifests/test.jsonl").read_text().splitlines() if x.strip()]
    samples = {r["utterance_key"].removeprefix("celebvdub/"): r["original_num_samples"] for r in manifests}
    wanted = {f"test/{c}.wav" for c in clips}
    actual = {p.relative_to(output).as_posix() for p in (output / "test").rglob("*.wav")}
    assert actual == wanted, (len(actual), len(wanted), actual ^ wanted)
    for relative in wanted:
        info = sf.info(str(output / relative))
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        assert info.frames == samples[relative.removesuffix(".wav")]


def merge_shards(root, output, cfg, nshard, clips):
    summaries = [json.loads((root / f"v{cfg}/shard{i}/inference_summary.json").read_text()) for i in range(nshard)]
    reference = summaries[0]
    for summary in summaries:
        for field in ("checkpoint", "decoder", "generation", "normalization_contract"):
            assert summary[field] == reference[field], field
        assert summary["dataset"]["test_list_sha256"] == digest(LIST)
    rows = [r for s in summaries for r in s["outputs"]]
    assert len(rows) == len({r["utterance_key"] for r in rows}) == 213
    by_key = {r["utterance_key"]: r for r in rows}
    for index, summary in enumerate(summaries):
        for row in summary["outputs"]:
            source = root / f"v{cfg}/shard{index}" / row["relative_path"]
            assert digest(source) == row["sha256"]
            destination = output / row["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
    reference["outputs"] = [by_key[f"celebvdub/test/{c}"] for c in clips]
    reference["dataset"].update(count=213, full_count=213, shard=None, nshard=nshard)
    reference["elapsed_seconds"] = sum(s["elapsed_seconds"] for s in summaries)
    reference["elapsed_semantics"] = "sum of worker compute time, not wall-clock"
    reference["peak_cuda_reserved_bytes"] = max(s["peak_cuda_reserved_bytes"] for s in summaries)
    sources = {r["utterance_key"]: r for s in summaries for r in s["speaker_embedding"]["sources"]}
    reference["speaker_embedding"]["sources"] = [sources[f"celebvdub/test/{c}"] for c in clips]
    save(output / "inference_summary.json", reference)
    validate_audio(output, clips)


def verify_metrics(output, clips):
    import numpy as np
    from jiwer import compute_measures
    scores = {}
    counts = {}
    for task in ("sim", "wer", "emosim", "emoembed", "avsync"):
        path = output / f"_{task}_results.jsonl"
        lines = path.read_text().splitlines()
        rows = [json.loads(x) for x in lines if x.startswith("{")]
        assert len(rows) == 213, (task, len(rows))
        for clip, row in zip(clips, rows):
            assert row["wav"] == Path(clip).name, (task, clip, row)
            row["utterance_key"] = f"celebvdub/test/{clip}"
        if task == "wer":
            detail = compute_measures([r["truth"] for r in rows], [r["hypo"] for r in rows])
            score = detail["wer"]
            counts = {k: detail[k] for k in ("hits", "substitutions", "deletions", "insertions")}
        else:
            values = np.array([r[task] for r in rows], dtype=float)
            assert np.isfinite(values).all(), task
            score = float(values.mean())
        logged = float(next(x.split(":", 1)[1] for x in lines if x.startswith(task.upper() + ":")))
        assert round(score, 5) == logged, (task, score, logged)
        scores[task] = score
        save(output / f"verified_{task}.json", rows)
    expected = {f"test/{c}.npy" for c in clips}
    found = {p.relative_to(output / "avhubert_feat").as_posix() for p in (output / "avhubert_feat").rglob("*.npy")}
    assert found == expected, (len(found), len(expected))
    for relative in expected:
        feature = np.load(output / "avhubert_feat" / relative)
        gt = np.load(DATA / "CelebVDub/avhubert_feat" / relative)
        assert feature.shape == gt.shape and np.isfinite(feature).all(), relative
    validate_audio(output, clips)
    return {"complete": True, "count": 213, "metrics": scores, "wer_edit_counts": counts,
            "output_dir": str(output), "checkpoint": str(CKPTS / "model_150000.pt"),
            "cfg_text": 5, "cfg_video": json.loads((output / "inference_summary.json").read_text())["generation"]["cfg_video"]}


def compare_waveforms(reference, candidate):
    """Native CUDA kernels need not be bitwise deterministic across processes.

    Do not change historical precision/kernel settings to force hash equality.
    Reject materially different signals; record both exact hashes and drift.
    """
    import numpy as np
    import soundfile as sf
    a, sr_a = sf.read(reference)
    b, sr_b = sf.read(candidate)
    assert sr_a == sr_b == 16000 and a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    rms = float(np.sqrt(np.mean((a - b)**2)))
    signal = float(np.sqrt(np.mean(a*a)))
    relative = rms / max(signal, 1e-6)
    corr = float(np.corrcoef(a, b)[0, 1]) if signal > 1e-6 else None
    result = {"reference_sha256": digest(reference), "candidate_sha256": digest(candidate),
              "max_abs_difference": float(np.max(np.abs(a-b))), "rms_difference": rms,
              "relative_rms_difference": relative, "correlation": corr,
              "criterion": "relative RMS <= 0.001 and correlation >= 0.99999 (non-silent)",
              "bitwise_equal": bool(np.array_equal(a, b))}
    assert relative <= 0.001 and (corr is None or corr >= 0.99999), result
    return result


def run(root, resume=False):
    root.mkdir(parents=True, exist_ok=resume)
    clips = [x.strip() for x in LIST.read_text().splitlines() if x.strip()]
    assert len(clips) == len(set(clips)) == 213
    manifests = [json.loads(x) for x in (DATA / "CelebVDub_svae1000k_sample_seed666_fp32/manifests/test.jsonl").read_text().splitlines() if x.strip()]
    lengths = {r["utterance_key"].removeprefix("celebvdub/test/"): r["latent_frames"] for r in manifests}
    longest = max(range(213), key=lambda i: lengths[clips[i]])
    print(f"Longest-clip memory pilot: {clips[longest]}, frames={lengths[clips[longest]]}", flush=True)
    if not (root / "pilot/inference_summary.json").exists():
        dispatch([("pilot", 0, inference_args(root / "pilot", 3, longest, 213), {})], root / "logs", slots=1)
    pilot = json.loads((root / "pilot/inference_summary.json").read_text())
    memory = pilot["peak_cuda_reserved_bytes"]
    slots = min(8, max(1, math.floor((23 * 1024**3) / (memory + 512 * 1024**2))))
    print(f"Pilot reserved={memory / 1024**3:.2f} GiB; native inference slots per GPU={slots}", flush=True)
    protocol = {"count_per_cfg": 213, "cfg_text": 5, "cfg_video": [3, 4, 5],
                                 "inference_slots_per_gpu": slots, "pilot_reserved_bytes": memory,
                                 "test_list_sha256": digest(LIST), "checkpoint_sha256": digest(CKPTS / "model_150000.pt"),
                                 "reference": "same clip full GT audio; historical Setting 1",
                                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()}
    if (root / "protocol.json").exists():
        previous = json.loads((root / "protocol.json").read_text())
        for key in ("count_per_cfg", "cfg_text", "cfg_video", "test_list_sha256", "checkpoint_sha256", "reference"):
            assert previous[key] == protocol[key], key
        previous.setdefault("resume_source_commits", []).append(protocol["source_commit"])
        protocol = previous
    save(root / "protocol.json", protocol)
    groups = [(3, 6), (4, 5), (5, 5)]
    outputs = {cfg: root / f"cfg_text5_video{cfg}" for cfg, _ in groups}
    jobs = []
    for cfg, nshard in groups:
        for shard in range(nshard):
            if (root / f"shards/v{cfg}/shard{shard}/inference_summary.json").exists():
                continue
            jobs.append((f"infer_v{cfg}_s{shard}", len(jobs) % 2,
                         inference_args(root / f"shards/v{cfg}/shard{shard}", cfg, shard, nshard), {}))
    dispatch(jobs, root / "logs", slots)
    for cfg, nshard in groups:
        if (outputs[cfg] / "inference_summary.json").exists():
            summary = json.loads((outputs[cfg] / "inference_summary.json").read_text())
            assert summary["generation"]["cfg_text"] == 5 and summary["generation"]["cfg_video"] == cfg
            assert summary["checkpoint"]["sha256"] == protocol["checkpoint_sha256"]
            assert summary["dataset"]["test_list_sha256"] == protocol["test_list_sha256"]
            assert len(summary["outputs"]) == 213
            validate_audio(outputs[cfg], clips)
            for row in summary["outputs"]:
                assert digest(outputs[cfg] / row["relative_path"]) == row["sha256"]
        else:
            merge_shards(root / "shards", outputs[cfg], cfg, nshard, clips)
    pilot_row = pilot["outputs"][0]
    drift = compare_waveforms(root / "pilot" / pilot_row["relative_path"], outputs[3] / pilot_row["relative_path"])
    save(root / "parallel_numeric_check.json", drift)
    print(f"All 639 WAVs verified; pilot numeric drift: {drift}", flush=True)
    jobs = []
    for cfg, nshard in groups:
        for rank in range(nshard):
            jobs.append((f"av_v{cfg}_s{rank}", len(jobs) % 2,
                         [SCRIPT / "misc/extract_avhubert.py", "--nshard", nshard, "--rank", rank,
                          "--v-input-dir", DATA / "CelebVDub/video_mouth/test/test",
                          "--a-input-dir", outputs[cfg] / "test", "--output-dir", outputs[cfg] / "avhubert_feat/test",
                          "--ckpt-path", ASSETS / "large_vox_iter5.pt", "--user_dir", DATA / "av_hubert/avhubert/avhubert"],
                         {"PYTHONPATH": str(PROJECT / "src") + ":" + str(DATA / "av_hubert/fairseq/fairseq")}))
    dispatch(jobs, root / "logs", slots=8)
    jobs = []
    for cfg, _ in groups:
        for task in ("sim", "wer", "emosim", "emoembed", "avsync"):
            jobs.append((f"metric_v{cfg}_{task}", len(jobs) % 2,
                         [SCRIPT / "eval/eval_celebvdub_test.py", "-e", task, "-g", outputs[cfg], "-n", "1",
                          "--test-list", LIST, "--celebvdub-root", DATA / "CelebVDub",
                          "--asr_ckpt", DATA / "faster-whisper-large-v3", "--gt_av_feat", DATA / "CelebVDub/avhubert_feat"], {}))
    dispatch(jobs, root / "logs", slots=8)
    results = [verify_metrics(outputs[cfg], clips) for cfg, _ in groups]
    save(root / "summary.json", {"complete": True, "groups": results})
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed inference, without replacing WAVs")
    args = parser.parse_args()
    run(args.output_root.resolve(), args.resume)
