"""Isolated, deterministic 132-clip GRID S1 CFG/condition diagnostic at REPA 100k.

This entry reuses the experiment's sampler, strict EMA loader, decoder and four
metric implementations. It never edits training or formal evaluation outputs.
Prepare once, infer each variant, evaluate each variant/task, then merge.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from aligndit.script.eval.grid_s1_common import (
    PROTOCOL, ensure_contract, identity, load_rows, metric_summary,
    normalize_text, output_record, publish, read, sha,
)
from aligndit.script.eval.grid_eval_inputs import artifact_path, validate_receipt, word_edit_counts


STUDY = "grid_s1_repa100k_cfg_ablation132_20261116_v1"
SALT = "cfgablation20261116:"
CHECKPOINT_SHA = "624b8f370a8e48060c382534d71dcc5a35065bcb10f516a2765ba4b6fe221984"
EXPERIMENT = "AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char"
TASKS = ("sim", "wer", "emosim", "avsync")
VARIANTS = {
    "cfg_t5_v2": {"cfg_text": 5, "cfg_video": 2, "ignore_modality": None},
    "cfg_t3_v2": {"cfg_text": 3, "cfg_video": 2, "ignore_modality": None},
    "cfg_t1_v2": {"cfg_text": 1, "cfg_video": 2, "ignore_modality": None},
    "cfg_t5_v1": {"cfg_text": 5, "cfg_video": 1, "ignore_modality": None},
    "cfg_t5_v0": {"cfg_text": 5, "cfg_video": 0, "ignore_modality": None},
    "cfg_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": None},
    "drop_text_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": "text"},
    "drop_video_t0_v0": {"cfg_text": 0, "cfg_video": 0, "ignore_modality": "video"},
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def bound_file(path):
    path = Path(path).resolve(strict=True)
    before = fingerprint(path)
    digest = sha(path)
    if fingerprint(path) != before:
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"path": str(path), "sha256": digest, "stat": before}


def verify_binding(binding, *, full_hash=False):
    path = Path(binding["path"])
    if fingerprint(path) != binding["stat"] or (full_hash and sha(path) != binding["sha256"]):
        raise RuntimeError(f"Bound file changed: {path}")


@contextmanager
def job_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker owns this task: {path}") from error
        yield


def select_rows(rows):
    """Select four IDs per speaker without reading outputs, ASR or any scores."""
    groups = defaultdict(list)
    scores = {identity(row): hashlib.sha256((SALT + identity(row)).encode()).hexdigest() for row in rows}
    for row in rows:
        groups[row["speaker"]].append(row)
    if len(groups) != 33 or any(len(group) < 4 for group in groups.values()):
        raise RuntimeError("Expected 33 GRID speakers with at least four validation clips each")
    chosen = {identity(row) for group in groups.values()
              for row in sorted(group, key=lambda row: (scores[identity(row)], identity(row)))[:4]}
    selected = [row for row in rows if identity(row) in chosen]
    if len(selected) != 132 or set(Counter(row["speaker"] for row in selected).values()) != {4}:
        raise RuntimeError("Invalid fixed 132-clip selection")
    return {"method": "per-speaker smallest SHA256(salt + full_id), retain original val manifest order",
            "salt": SALT, "selected_count": 132, "speaker_counts": dict(Counter(r["speaker"] for r in selected)),
            "ids": [identity(r) for r in selected], "rows": selected,
            "selection_scores": {identity(r): scores[identity(r)] for r in selected}}


def metric_args(root, task):
    """Use the unchanged metric parser's defaults, without patching its logic."""
    from aligndit.script.eval import eval_grid_s1
    previous = sys.argv
    try:
        sys.argv = [str(Path(eval_grid_s1.__file__)), "--output-dir", str(root), "--task", task]
        result = eval_grid_s1.args()
    finally:
        sys.argv = previous
    return result


def paths():
    prefix = os.environ.get("ROOT_PREFIX", "")
    project_root = Path(prefix + "/zjw524/projects")
    data = project_root / "data"
    return {"cache": data / "GRID_svae1000k_sample_seed666_fp32",
            "checkpoint": data / "ckpts" / EXPERIMENT / "model_100000.pt",
            "config": PROJECT / "src/aligndit/config/finetune_grid.yaml",
            "decoder_repo": project_root / "alignDiT_idea6/papers_codes/Semantic-VAE",
            "decoder_checkpoint": project_root / "alignDiT_idea6/Semantic-VAE/semantic_vae_1000k",
            "historical": data / "evaluations/GRID_avt_fixed13_libritts_repa_s1_100k_seed0_nfe32_cfg5_2"}


def variant_contract(root, contract, name):
    parameters = {"name": name, **VARIANTS[name]}
    return {"protocol": PROTOCOL, "study": STUDY, "ablation_contract_sha256": sha(root / "ablation_contract.json"),
            "selection_sha256": sha(root / "selection.json"), "variant": parameters,
            "requested_count": 132, "partial_smoke_test": True, "full_validation_count": 3274,
            "checkpoint": contract["checkpoint"], "config": contract["config"],
            "generation": {"seed": 0, "nfe": 32, "ode": "euler", "epss": True, "sway": -1,
                           "cfg_text": parameters["cfg_text"], "cfg_video": parameters["cfg_video"],
                           "ignore_modality": parameters["ignore_modality"], "batch_size": 1},
            "reference": "same target clip complete canonical MPG audio; GT latent prompt plus same-clip CAM++",
            "model_text": "same transcript + two spaces + same transcript; length retained when text is dropped",
            "output": "only target continuation; prompt removed; exact original duration; 16k mono PCM16",
            "condition_drop": "same internal null replacement used by training dropout; preserve sequence lengths, prompt, speaker and initial-noise seed"}


def prepare(a):
    from omegaconf import OmegaConf
    from aligndit.model.grid_dataset import NORMALIZATION_SHA256, VOCAB_SHA256
    from aligndit.script.eval import eval_grid_s1
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import load_composed_config
    p = paths()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with job_lock(a.output_dir / "prepare.lock"):
        if (a.output_dir / "ablation_contract.json").exists():
            load_contract(a.output_dir, verify_checkpoint=True)
            print("Reused verified prepared diagnostic contract", flush=True)
            return
        rows, done = load_rows(p["cache"])
        selection = select_rows(rows)
        selection["manifest_sha256"] = sha(p["cache"] / "manifests/val.jsonl")
        ensure_contract(a.output_dir / "selection.json", selection)
        config = load_composed_config(p["config"])
        if sha(config.datasets.normalization_path) != NORMALIZATION_SHA256 or sha(config.datasets.vocab_path) != VOCAB_SHA256:
            raise RuntimeError("Unexpected pretrained normalization or vocabulary")
        checkpoint = bound_file(p["checkpoint"])
        if checkpoint["sha256"] != CHECKPOINT_SHA:
            raise RuntimeError("This diagnostic only accepts the pinned current-project REPA 100k checkpoint")
        import torch
        checkpoint_value = torch.load(p["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
        ema = checkpoint_value["ema_model_state_dict"]
        if checkpoint_value["update"] != 100000 or not bool(ema["initted"]) or len(ema) - 2 != 808:
            raise RuntimeError("Expected initialized 808-tensor EMA at update 100000")
        del checkpoint_value, ema
        checkpoint.update(update=100000, weights="EMA", state_tensor_count=808)
        ma = metric_args(a.output_dir, "audit")
        eval_grid_s1.audit_assets(ma)
        metric_assets = read(a.output_dir / "metric_assets.json")
        asset_bindings = []
        for item in metric_assets["assets"]:
            binding = bound_file(item["path"])
            if binding["sha256"] != item["sha256"]:
                raise RuntimeError("Metric asset changed during prepare")
            asset_bindings.append(binding)
        # The unchanged decoder loader rechecks these hashes and the clean repo
        # revision at every inference startup; bind its resources here too.
        decoder_meta = p["decoder_checkpoint"] / "metainfo.json"
        decoder_conf = Path(read(decoder_meta)["DAC"]["bigvgan_conf"])
        if not decoder_conf.is_absolute():
            decoder_conf = p["decoder_repo"] / decoder_conf
        resource_paths = [p["cache"] / "complete.json", p["cache"] / "manifests/inventory_meta.json",
                          p["cache"] / "manifests/val.jsonl", p["cache"] / "manifests/train.jsonl",
                          p["cache"] / "state/latents/spec.json", Path(config.datasets.vocab_path),
                          Path(config.datasets.normalization_path), decoder_meta, decoder_conf,
                          p["decoder_checkpoint"] / "dac/ema_state_dict.pth"]
        resources = [bound_file(path) for path in resource_paths]
        sources = sorted(set((PROJECT / "src").rglob("*.py"))
                         | set((PROJECT / "src/aligndit/config").glob("*.yaml"))
                         | set((PROJECT / "scripts").glob("*.py"))
                         | set((PROJECT / "scripts").glob("*.sh"))
                         | set((PROJECT / "tests").rglob("*.py")))
        source_bindings = [bound_file(path) for path in sources]
        snapshot = []
        for binding in source_bindings:
            source = Path(binding["path"])
            target = a.output_dir / "source_snapshot" / source.relative_to(PROJECT)
            if target.exists():
                if sha(target) != binding["sha256"]:
                    raise RuntimeError("Existing diagnostic source snapshot differs")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            if sha(target) != binding["sha256"]:
                raise RuntimeError("Diagnostic source snapshot copy differs")
            snapshot.append({"source_path": str(source), "snapshot_path": str(target), "sha256": binding["sha256"]})
        ensure_contract(a.output_dir / "source_snapshot.json", {"files": snapshot})
        import subprocess
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
        ensure_contract(a.output_dir / "launch_source_snapshot.json", {
            "project_dir": str(PROJECT), "git_commit": revision,
            "source_files": [{"path": str(Path(item["path"]).relative_to(PROJECT)),
                              "bytes": item["stat"]["size_bytes"], "sha256": item["sha256"]}
                             for item in source_bindings]})
        input_bindings = {}
        for row in selection["rows"]:
            entry = {}
            for stage in ("audio", "latent", "video", "speaker"):
                if not validate_receipt(p["cache"], row, stage, done["extraction_sha256"]):
                    raise RuntimeError(f"Missing validated input: {identity(row)} / {stage}")
                entry[stage] = bound_file(artifact_path(p["cache"], row, stage))
            input_bindings[identity(row)] = entry
        historical = {name: bound_file(p["historical"] / name) for name in
                      ("protocol.json", "inference_summary.json", "evaluation_summary.json", "metric_assets.json",
                       *(f"{task}_results.jsonl" for task in TASKS))}
        old_protocol = read(p["historical"] / "protocol.json")
        if old_protocol["checkpoint"]["sha256"] != CHECKPOINT_SHA or old_protocol["checkpoint"]["update"] != 100000:
            raise RuntimeError("Historical baseline is not the same REPA 100k EMA")
        expected_generation = {"seed": 0, "nfe": 32, "ode": "euler", "epss": True, "sway": -1,
                               "cfg_text": 5, "cfg_video": 2, "batch_size": 1}
        if old_protocol["protocol"] != PROTOCOL or old_protocol["generation"] != expected_generation:
            raise RuntimeError("Historical baseline has different S1 generation settings")
        old_assets = read(p["historical"] / "metric_assets.json")
        if {item["path"]: item["sha256"] for item in old_assets["assets"]} != {
                item["path"]: item["sha256"] for item in metric_assets["assets"]}:
            raise RuntimeError("Metric models differ from the historical REPA 100k evaluation")
        contract = {"study": STUDY, "protocol": PROTOCOL, "diagnostic_subset_only": True,
                    "count": 132, "full_validation_count": 3274, "selection_sha256": sha(a.output_dir / "selection.json"),
                    "checkpoint": checkpoint, "config": OmegaConf.to_container(config, resolve=True),
                    "config_path": str(p["config"]), "cache_root": str(p["cache"]),
                    "decoder_repo": str(p["decoder_repo"]), "decoder_checkpoint": str(p["decoder_checkpoint"]),
                    "variants": VARIANTS, "metric_assets_sha256": sha(a.output_dir / "metric_assets.json"),
                    "asset_bindings": asset_bindings, "resource_bindings": resources, "source_files": source_bindings,
                    "source_snapshot_sha256": sha(a.output_dir / "source_snapshot.json"),
                    "launch_source_snapshot_sha256": sha(a.output_dir / "launch_source_snapshot.json"),
                    "input_bindings": input_bindings, "historical_root": str(p["historical"]),
                    "historical_bindings": historical,
                    "guidance_formula": "full + cfg_video*(full-tts) + cfg_text*(tts-null)",
                    "guidance_semantics": "tts retains text, reference audio and speaker; null drops all explicit conditions; CFG zero keeps full conditioning",
                    "condition_ablation_semantics": "both CFG values zero; internal text or video null; keep same prompt, speaker, masks, lengths, seed and duration",
                    "limitation": "132 validation clips selected without output scores; not the complete 3274-clip benchmark; S1 GT prompt contains target speech"}
        ensure_contract(a.output_dir / "ablation_contract.json", contract)
        for name in VARIANTS:
            ensure_contract(a.output_dir / name / "protocol.json", variant_contract(a.output_dir, contract, name))
        print(json.dumps({"prepared": True, "count": 132, "variants": list(VARIANTS),
                          "contract_sha256": sha(a.output_dir / "ablation_contract.json")}), flush=True)


def load_contract(root, *, verify_checkpoint=False):
    contract = read(root / "ablation_contract.json")
    if contract["study"] != STUDY or contract["variants"] != VARIANTS or contract["checkpoint"]["sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Wrong ablation study/checkpoint/variants")
    if sha(root / "selection.json") != contract["selection_sha256"] or sha(root / "metric_assets.json") != contract["metric_assets_sha256"]:
        raise RuntimeError("Selection or metric assets changed")
    if sha(root / "source_snapshot.json") != contract["source_snapshot_sha256"]:
        raise RuntimeError("Source snapshot inventory changed")
    if sha(root / "launch_source_snapshot.json") != contract["launch_source_snapshot_sha256"]:
        raise RuntimeError("Launch source snapshot changed")
    for item in contract["source_files"]:
        verify_binding(item, full_hash=True)
    for item in contract["resource_bindings"] + contract["asset_bindings"]:
        verify_binding(item)
    verify_binding(contract["checkpoint"], full_hash=verify_checkpoint)
    selection = read(root / "selection.json")
    rows, _ = load_rows(Path(contract["cache_root"]))
    expected = select_rows(rows)
    expected["manifest_sha256"] = sha(Path(contract["cache_root"]) / "manifests/val.jsonl")
    if selection != expected:
        raise RuntimeError("Selected rows are not the deterministic four-per-speaker subset")
    return contract, selection["rows"]


def verify_variant(root, contract, name, rows):
    folder = root / name
    if read(folder / "protocol.json") != variant_contract(root, contract, name):
        raise RuntimeError(f"Variant protocol changed: {name}")
    receipts = [output_record(folder, row) for row in rows]
    expected = {r["audio_relative_path"] for r in rows}
    if {p.relative_to(folder).as_posix() for p in folder.glob("val/*/*.wav")} != expected:
        raise RuntimeError("Variant WAV coverage differs from selected 132 clips")
    for row, receipt in zip(rows, receipts):
        if receipt["variant"] != {"name": name, **VARIANTS[name]} or receipt["inputs"] != contract["input_bindings"][identity(row)]:
            raise RuntimeError("Inference receipt condition/input mismatch")
    return {"protocol": PROTOCOL, "study": STUDY, "count": len(rows), "complete": False,
            "complete_subset": True, "partial_smoke_test": True, "outputs": receipts,
            "protocol_sha256": sha(folder / "protocol.json")}


def infer(a):
    import numpy as np
    import torch
    from tqdm import tqdm
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
        atomic_save_waveform, build_model, historical_setting1_text, load_normalization,
    )
    from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
    contract, rows = load_contract(a.output_dir, verify_checkpoint=True)
    folder = a.output_dir / a.variant
    with job_lock(folder / "infer.lock"):
        ensure_contract(folder / "protocol.json", variant_contract(a.output_dir, contract, a.variant))
        if (folder / "inference_summary.json").exists():
            if read(folder / "inference_summary.json") != verify_variant(a.output_dir, contract, a.variant, rows):
                raise RuntimeError("Stored inference summary differs")
            print(f"Reuse verified inference: {a.variant}", flush=True)
            return
        device = torch.device("cuda:0")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        model = build_model(Path(contract["config_path"]), Path(contract["checkpoint"]["path"]), 100000, device)
        if len(model.state_dict()) != 808:
            raise RuntimeError("Expected all 808 current-project EMA tensors")
        cache = Path(contract["cache_root"])
        decoder, decoder_meta = load_semantic_vae_decoder(repo=Path(contract["decoder_repo"]),
            checkpoint_root=Path(contract["decoder_checkpoint"]), cache_spec=read(cache / "state/latents/spec.json"), device=device)
        ensure_contract(folder / "decoder_shard0.json", decoder_meta)
        mean, std, _ = load_normalization(Path(contract["config"]["datasets"]["normalization_path"]))
        mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
        variant = VARIANTS[a.variant]
        with torch.inference_mode():
            for row in tqdm(rows, desc=a.variant, mininterval=5):
                output = folder / row["audio_relative_path"]
                receipt_path = folder / "receipts" / (identity(row) + ".json")
                if output.exists() or receipt_path.exists():
                    output_record(folder, row)
                    continue
                inputs = contract["input_bindings"][identity(row)]
                for binding in inputs.values():
                    verify_binding(binding, full_hash=True)
                latent = np.load(inputs["latent"]["path"], allow_pickle=False)
                video = torch.from_numpy(np.load(inputs["video"]["path"], allow_pickle=False)).to(device)
                speaker = torch.from_numpy(np.load(inputs["speaker"]["path"], allow_pickle=False)).to(device)
                frames = row["latent_frames"]
                model_text = historical_setting1_text(row["text"])
                generated, _ = model.sample(cond=torch.from_numpy((latent - mean) / std)[None].to(device),
                    text=[model_text], duration=torch.tensor([2 * frames], device=device),
                    video=torch.cat((torch.zeros_like(video), video))[None], lens=torch.tensor([frames], device=device),
                    steps=32, cfg_strength=variant["cfg_text"], cfg_strength_v=variant["cfg_video"],
                    ignore_modality=variant["ignore_modality"], sway_sampling_coef=-1, seed=0, use_epss=True,
                    speaker_embedding=speaker[None])
                target = generated[:, frames:2 * frames].float()
                if target.shape != (1, frames, 64) or not torch.isfinite(target).all():
                    raise RuntimeError("Invalid generated target latent")
                waveform = decoder((target * std_t + mean_t).transpose(1, 2)).squeeze(0).float().cpu()
                if waveform.shape != (1, frames * 400) or not torch.isfinite(waveform).all():
                    raise RuntimeError("Invalid decoded waveform")
                atomic_save_waveform(output, waveform[:, :row["original_num_samples"]])
                publish(receipt_path, {"id": identity(row), "relative_path": row["audio_relative_path"],
                    "sha256": sha(output), "samples": row["original_num_samples"],
                    "protocol_sha256": sha(folder / "protocol.json"), "variant": {"name": a.variant, **variant},
                    "model_text": model_text, "prompt_frames": frames, "target_frames": frames, "sampler_frames": 2 * frames,
                    "seed": 0, "nfe": 32, "inputs": inputs,
                    "video_mode": "zeros_like prefix then cached target video; internal null replacement iff ignore_modality=video",
                    "speaker_mode": "same-clip cached CAM++; retained in conditional branch"})
        publish(folder / "inference_summary.json", verify_variant(a.output_dir, contract, a.variant, rows))


def read_results(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def publish_jsonl(path, rows):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    os.replace(tmp, path)


def validate_results(results, task, rows):
    if [r["id"] for r in results] != [identity(r) for r in rows]:
        raise RuntimeError(f"Incomplete/out-of-order {task} IDs")
    summary = metric_summary(results, task)
    if task == "wer":
        import jiwer
        for row, result in zip(rows, results):
            truth, hypothesis = normalize_text(result["raw_truth"]), normalize_text(result["raw_hypo"])
            errors, words = word_edit_counts(truth, hypothesis)
            counts = jiwer.process_words(truth, hypothesis)
            if (truth != normalize_text(row["text"]) or truth != result["truth"] or hypothesis != result["hypo"]
                    or (errors, words) != (result["word_edit_distance"], result["reference_word_count"])
                    or (counts.substitutions, counts.deletions, counts.insertions) !=
                       (result["substitutions"], result["deletions"], result["insertions"])
                    or result["wer"] != errors / words):
                raise RuntimeError("Independent WER DP/alignment validation failed")
        if summary["reference_word_count"] != 6 * len(rows):
            raise RuntimeError("Unexpected selected GRID word denominator")
    return summary


def metric_binding(root, folder, contract, results, task):
    return {"inference_summary_sha256": sha(folder / "inference_summary.json"),
            "ablation_contract_sha256": sha(root / "ablation_contract.json"),
            "metric_assets_sha256": contract["metric_assets_sha256"],
            "results_sha256": sha(folder / f"{task}_results.jsonl"),
            "input_audio_hashes": {r["id"]: {"generated": r["generated_audio_sha256"],
                                                   "reference": r["reference_audio_sha256"]} for r in results}}


def metrics(a):
    import torch
    from aligndit.script.eval import eval_grid_s1
    contract, rows = load_contract(a.output_dir)
    folder = a.output_dir / a.variant
    with job_lock(folder / (a.task + ".lock")):
        if read(folder / "inference_summary.json") != verify_variant(a.output_dir, contract, a.variant, rows):
            raise RuntimeError("Metrics require verified completed 132-clip inference")
        result_path, summary_path = folder / f"{a.task}_results.jsonl", folder / f"{a.task}_summary.json"
        if result_path.exists() and summary_path.exists():
            results = read_results(result_path)
            summary = {**validate_results(results, a.task, rows), **metric_binding(a.output_dir, folder, contract, results, a.task)}
            if read(summary_path) != summary:
                raise RuntimeError("Stored metric results/binding differ")
            verify_result_audio(contract, rows, results, folder)
            print(f"Reuse verified {a.variant}/{a.task}", flush=True)
            return
        torch.set_num_threads(4)
        cache = Path(contract["cache_root"])
        tests = [(str(folder / row["audio_relative_path"]), str(cache / "audio" / row["audio_relative_path"]), row["text"]) for row in rows]
        ma = metric_args(folder, a.task)
        if a.task == "sim":
            from f5_tts.eval.utils_eval import run_sim
            values = run_sim((0, tests, str(ma.wavlm_ckpt), str(ma.wavlm_base_ckpt)))
        elif a.task == "emosim":
            from aligndit.script.eval.utils import run_emosim
            values = run_emosim((0, tests, str(ma.emo_ckpt)))
        elif a.task == "wer":
            values = eval_grid_s1.run_wer(ma, rows, tests)
        else:
            values = eval_grid_s1.run_avsync(ma, rows, tests)
        if len(values) != len(rows):
            raise RuntimeError("Incomplete metric results")
        results = []
        for row, value, test in zip(rows, values, tests):
            if value["wav"] != Path(row["audio_relative_path"]).stem:
                raise RuntimeError("Metric worker returned wrong sample order")
            results.append({**value, "wav": identity(row), "id": identity(row),
                            "generated_audio_sha256": sha(test[0]), "reference_audio_sha256": sha(test[1])})
        summary = validate_results(results, a.task, rows)
        verify_result_audio(contract, rows, results, folder)
        publish_jsonl(result_path, results)
        publish(summary_path, {**summary, **metric_binding(a.output_dir, folder, contract, results, a.task)})
        print(json.dumps({"variant": a.variant, "task": a.task, **summary}), flush=True)


def verify_result_audio(contract, rows, results, folder):
    for row, result in zip(rows, results):
        if result["generated_audio_sha256"] != sha(folder / row["audio_relative_path"]):
            raise RuntimeError("Metric result is bound to changed generated audio")
        audio = contract["input_bindings"][identity(row)]["audio"]
        verify_binding(audio, full_hash=True)
        if result["reference_audio_sha256"] != audio["sha256"]:
            raise RuntimeError("Metric reference is not the pinned same-clip GT audio")


def paired_changes(results, baseline):
    changes = [r["word_edit_distance"] - b["word_edit_distance"] for r, b in zip(results, baseline)]
    denominator = sum(r["reference_word_count"] for r in results)
    return {"delta_word_errors": sum(changes), "delta_WER_percentage_points": 100 * sum(changes) / denominator,
            "utterances_worse_equal_better": [sum(x > 0 for x in changes), sum(x == 0 for x in changes), sum(x < 0 for x in changes)],
            "by_id": {r["id"]: delta for r, delta in zip(results, changes)}}


def historical_comparison(root, contract, rows, baseline_metrics, baseline_results):
    import numpy as np
    import soundfile as sf
    historical = Path(contract["historical_root"])
    for binding in contract["historical_bindings"].values():
        verify_binding(binding, full_hash=True)
    current = root / "cfg_t5_v2"
    old = {}
    metrics = {}
    for task in TASKS:
        values = read_results(historical / f"{task}_results.jsonl")
        by_id = {r["id"]: r for r in values}
        if len(values) != len(by_id) or len(values) != 3274:
            raise RuntimeError("Historical metric coverage is not 3274 unique IDs")
        subset = [by_id[identity(r)] for r in rows]
        old[task] = subset
        metrics[task] = validate_results(subset, task, rows)
    waveform_comparisons = []
    asr_comparisons = []
    for row, new_wer, old_wer in zip(rows, baseline_results["wer"], old["wer"]):
        receipt = output_record(historical, row)
        new_path, old_path = current / row["audio_relative_path"], historical / row["audio_relative_path"]
        new_audio, nr = sf.read(new_path, dtype="float64")
        old_audio, oldr = sf.read(old_path, dtype="float64")
        if nr != oldr or new_audio.shape != old_audio.shape or not np.isfinite(old_audio).all():
            raise RuntimeError("Historical audio geometry/validity mismatch")
        difference = new_audio - old_audio
        waveform_comparisons.append({"id": identity(row), "current_sha256": sha(new_path),
            "historical_sha256": receipt["sha256"], "file_identical": sha(new_path) == receipt["sha256"],
            "pcm_identical": bool(np.array_equal(new_audio, old_audio)), "max_abs_difference": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference ** 2)))})
        asr_comparisons.append({"id": identity(row), "current_raw_hypo": new_wer["raw_hypo"],
            "historical_raw_hypo": old_wer["raw_hypo"], "current_hypo": new_wer["hypo"], "historical_hypo": old_wer["hypo"],
            "normalized_identical": new_wer["hypo"] == old_wer["hypo"],
            "delta_word_errors": new_wer["word_edit_distance"] - old_wer["word_edit_distance"]})
    return {"historical_root": str(historical), "subset_count": len(rows), "historical_subset_metrics": metrics,
            "current_subset_metrics": baseline_metrics,
            "metric_deltas_current_minus_historical": {t: baseline_metrics[t][t] - metrics[t][t] for t in TASKS},
            "paired_word_errors": paired_changes(baseline_results["wer"], old["wer"]),
            "waveform_file_identical_count": sum(r["file_identical"] for r in waveform_comparisons),
            "waveform_pcm_identical_count": sum(r["pcm_identical"] for r in waveform_comparisons),
            "ASR_normalized_identical_count": sum(r["normalized_identical"] for r in asr_comparisons),
            "waveforms": waveform_comparisons, "ASR": asr_comparisons,
            "note": "Differences are retained, not overwritten or forced to match historical outputs; use the concurrent baseline for all ablation comparisons."}


def merge(a):
    contract, rows = load_contract(a.output_dir)
    with job_lock(a.output_dir / "merge.lock"):
        summaries, all_results = {}, {}
        for name in VARIANTS:
            folder = a.output_dir / name
            if read(folder / "inference_summary.json") != verify_variant(a.output_dir, contract, name, rows):
                raise RuntimeError("Inference summary mismatch at merge")
            summaries[name], all_results[name] = {}, {}
            for task in TASKS:
                results = read_results(folder / f"{task}_results.jsonl")
                summary = validate_results(results, task, rows)
                if read(folder / f"{task}_summary.json") != {**summary, **metric_binding(a.output_dir, folder, contract, results, task)}:
                    raise RuntimeError("Metric summary/binding mismatch at merge")
                verify_result_audio(contract, rows, results, folder)
                summaries[name][task], all_results[name][task] = summary, results
        comparison = historical_comparison(a.output_dir, contract, rows, summaries["cfg_t5_v2"], all_results["cfg_t5_v2"])
        report = {"study": STUDY, "complete_subset": True, "full_validation_complete": False,
                  "diagnostic_subset_only": True, "selected_count": len(rows), "full_validation_count": 3274,
                  "variant_count": len(VARIANTS), "generated_wavs": len(rows) * len(VARIANTS),
                  "ablation_contract_sha256": sha(a.output_dir / "ablation_contract.json"),
                  "selection_sha256": sha(a.output_dir / "selection.json"), "checkpoint": contract["checkpoint"],
                  "variants": VARIANTS, "metrics": summaries,
                  "paired_word_error_changes": {name: {baseline: paired_changes(all_results[name]["wer"], all_results[baseline]["wer"])
                      for baseline in ("cfg_t5_v2", "cfg_t0_v0")} for name in VARIANTS},
                  "historical_baseline_comparison": comparison,
                  "limitations": ["Fixed 132-clip validation diagnostic, not full benchmark or independent test set.",
                      "S1 same-clip GT prompt contains the whole target transcript acoustically; dropping text is not removing all lexical information.",
                      "Guidance text coefficient also amplifies prompt/speaker conditioning; video coefficient zero retains video in full prediction.",
                      "No significance or architecture-causality claim; confirm candidate CFG on the full validation set separately."]}
        publish(a.output_dir / "ablation_summary.json", report)
        print(json.dumps({"complete_subset": True, "count": len(rows), "metrics": summaries}, indent=2), flush=True)


def args():
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "metrics", "merge"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(prefix + "/zjw524/projects/data/evaluations/GRID_avt_fixed13_libritts_repa_100k_cfg_ablation132_20261116"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--task", choices=TASKS)
    result = parser.parse_args()
    result.output_dir = result.output_dir.expanduser().resolve()
    if result.stage in ("infer", "metrics") and result.variant is None:
        parser.error("infer and metrics require --variant")
    if result.stage == "metrics" and result.task is None:
        parser.error("metrics requires --task")
    if result.stage != "metrics" and result.task is not None:
        parser.error("--task is only valid with --stage metrics")
    if result.stage in ("prepare", "merge") and result.variant is not None:
        parser.error("prepare and merge cover all variants; omit --variant")
    return result


if __name__ == "__main__":
    arguments = args()
    {"prepare": prepare, "infer": infer, "metrics": metrics, "merge": merge}[arguments.stage](arguments)
