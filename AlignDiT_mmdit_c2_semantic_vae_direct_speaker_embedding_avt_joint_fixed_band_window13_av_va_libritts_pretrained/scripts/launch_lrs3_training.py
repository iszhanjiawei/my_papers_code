"""Validate and launch the published LRS3 recipe on four GPUs in a new session.

Default: validate the full-source publication, report its accepted/excluded
scope, and write a pinned launch plan. Add --start to start actual training.
The producer's declared quality exclusions are preserved and reported; a
full-source publication is not advertised as complete restoration.

Examples (run from this project, using the aligndit environment):
  PYTHONPATH=src python scripts/launch_lrs3_training.py
  PYTHONPATH=src python scripts/launch_lrs3_training.py --start --gpus 0,1,2,3
  PYTHONPATH=src python scripts/launch_lrs3_training.py --variant pseudo

TensorBoard defaults to local port 6028 for strict or 6029 for pseudo; the
completed pilot uses 6027. No externally forwarded URL is generated.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import requests

from aligndit.model.lrs3_semantic_vae_dataset import SemanticVaeLRS3Dataset, _iter_jsonl
from aligndit.model.semantic_vae_dataset import sha256_file


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = {
    "strict": "finetune_lrs3_svae_speaker_window13_libritts",
    "pseudo": "finetune_lrs3_pseudo_svae_speaker_window13_libritts",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Artifact changed while validating: {path}")
    return {"path": str(path), "sha256": digest, "size_bytes": after.st_size}


def acquire_read_locks(root, stack):
    descriptors = []
    for name in ("audio", "video40", "media", "publication"):
        path = root / f"state/locks/{name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = stack.enter_context(path.open("a"))
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        descriptors.append(handle.fileno())
    for feature in ("latents", "video_25hz", "video_40hz", "speaker_embeddings"):
        guard = root / f"state/{feature}/WRITE_ACTIVE.json"
        if guard.exists() or guard.is_symlink():
            raise RuntimeError(f"Unfinished cache writer prevents launch: {guard}")
    return descriptors


def load_config(args):
    overrides = []
    if args.cache_root:
        overrides.append(f"datasets.cache_root={args.cache_root.resolve()}")
    if args.checkpoint_dir:
        overrides.append(f"ckpts.save_dir={args.checkpoint_dir.resolve()}")
    if args.run_until_update is not None:
        overrides.append(f"optim.run_until_update={args.run_until_update}")
    if args.epochs is not None:
        overrides.append(f"optim.epochs={args.epochs}")
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT / "src/aligndit/config")):
        config = compose(config_name=CONFIG_NAMES[args.variant], overrides=overrides)
    run_name = getattr(args, "run_name", None)
    if run_name is not None:
        from aligndit.model.fixed_temporal_band_contract import EXPERIMENT_MARKER
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name)
                or EXPERIMENT_MARKER not in run_name):
            raise ValueError(f"Run name must be a single safe filename containing {EXPERIMENT_MARKER}")
        config.model.name = run_name
    return config


def training_event_dir(config):
    # Match the training entry's experiment name, including isolated smoke runs.
    exp_name = (f"{config.model.name}_{config.model.audio_representation.name}_"
                f"{config.datasets.name}_{config.model.tokenizer}")
    return PROJECT / "runs" / exp_name


def validate_publication(config):
    root = Path(config.datasets.cache_root).resolve(strict=True)
    manifest = Path(config.datasets.manifest_path)
    meta_path = manifest.parent / "inventory_meta.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("selection") != {"mode": "full"}:
        raise RuntimeError("Full training requires a finalized full-source publication")
    source_path = root / "reports/source_contract.json"
    source = json.loads(source_path.read_text())
    source_contract_digest = sha256_file(source_path)
    readiness_path = root / f"reports/{manifest.parent.name}_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    if (meta["source_pairs"] != source["source_media_pairs"]
            or meta.get("source_original_ids") != source["original_clip_count"]
            or meta.get("source_contract_sha256") != source_contract_digest
            or readiness["source_pairs"] != source["source_media_pairs"]
            or readiness["original_ids"] != source["original_clip_count"]
            or readiness["ready_pairs"] != meta["training_ready_pairs"]
            or meta["training_ready_pairs"] + meta["excluded_pairs"] != meta["source_pairs"]
            or readiness["source_manifest_sha256"] != meta["source_manifest_sha256"]):
        raise RuntimeError("Published source coverage is inconsistent with its frozen source contract")
    source_inventory = root / "manifests/source_inventory.jsonl"
    source_inventory_digest = sha256_file(source_inventory)
    if (source_inventory_digest != meta["source_manifest_sha256"]
            or source_inventory_digest != source.get("source_inventory_sha256")):
        raise RuntimeError("Canonical source inventory differs from its contract or publication")
    config.datasets.expected_manifest_sha256 = meta["manifests"][manifest.name]["sha256"]
    config.datasets.expected_inventory_sha256 = meta["manifests"]["inventory.jsonl"]["sha256"]
    config.datasets.expected_inventory_meta_sha256 = sha256_file(meta_path)
    config.datasets.expected_record_count = meta["manifests"][manifest.name]["count"]
    values = OmegaConf.to_container(config.datasets, resolve=True)
    parameters = inspect.signature(SemanticVaeLRS3Dataset).parameters
    data = SemanticVaeLRS3Dataset(**{key: value for key, value in values.items() if key in parameters})
    if data.data_contract["split_counts"] != readiness["split_counts"]:
        raise RuntimeError("Readiness split counts differ from validated manifest contents")
    if readiness.get('media_quality') != data.data_contract.get('media_quality'):
        raise RuntimeError('Readiness report differs from the pinned media quality closure')
    quality_path = root / f"reports/{manifest.parent.name}_sample_quality.jsonl"
    source_ids = {}
    for row in _iter_jsonl(source_inventory):
        key = row["sample_id"]
        if key in source_ids:
            raise RuntimeError(f"Duplicate source-inventory sample: {key}")
        source_ids[key] = (row["original_clip_id"], row["split"])
    accepted_ids = {row["sample_id"] for row in _iter_jsonl(manifest.parent / "inventory.jsonl")}
    if len(accepted_ids) != meta["training_ready_pairs"]:
        raise RuntimeError("Accepted inventory contains duplicate sample IDs")
    source_originals = {original for original, _ in source_ids.values()}
    counts, excluded, seen, reasons = {"train": 0, "valid": 0, "test": 0}, 0, set(), {}
    for row in _iter_jsonl(quality_path):
        key = row["sample_id"]
        if key in seen:
            raise RuntimeError(f"Duplicate quality report sample ID: {key}")
        seen.add(key)
        if source_ids.pop(key, None) != (row["original_clip_id"], row["split"]):
            raise RuntimeError(f"Quality report source identity mismatch: {key}")
        if row["ready"] != (key in accepted_ids):
            raise RuntimeError(f"Quality report accepted membership mismatch: {key}")
        media_exclusion = data.media_quality_exclusions.get(key)
        if media_exclusion is not None and (row['ready'] or
                'media_quality:' + media_exclusion['reason'] not in row['reasons'] or
                row.get('media_quality_exclusion') != media_exclusion):
            raise RuntimeError('Quality report omits or changes a sealed media exclusion')
        if row["ready"]:
            if row["reasons"]:
                raise RuntimeError("Accepted quality record contains rejection reasons")
            counts[row["split"]] += 1
        else:
            excluded += 1
            if not row["reasons"]:
                raise RuntimeError("Excluded source sample has no declared quality reason")
            for reason in row["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
    if (source_ids or len(seen) != source["source_media_pairs"] or counts != readiness["split_counts"]
            or excluded != meta["excluded_pairs"] or reasons != readiness["pending_reasons"]):
        raise RuntimeError("Per-sample quality evidence differs from published coverage")
    coverage_path = root / f"reports/{manifest.parent.name}_original_coverage.jsonl"
    originals = set()
    source_pairs = ready_pairs = 0
    for row in _iter_jsonl(coverage_path):
        original = row["original_clip_id"]
        if original in originals:
            raise RuntimeError(f"Duplicate original-ID coverage record: {original}")
        originals.add(original)
        source_pairs += row["source_pairs"]
        ready_pairs += row["ready_pairs"]
    if (originals != source_originals or len(originals) != source["original_clip_count"] or source_pairs != source["source_media_pairs"]
            or ready_pairs != meta["training_ready_pairs"]):
        raise RuntimeError("Original-ID coverage report disagrees with published inventory")
    receipt = json.loads((manifest.parent / "cache_completion.json").read_text())
    speaker_dir = Path(receipt.get("speaker_metadata_dir", manifest.parent / "speaker_contract"))
    artifact_paths = [meta_path, source_path, source_inventory, readiness_path, quality_path, coverage_path,
                      manifest.parent / "cache_completion.json", speaker_dir / "metadata.json",
                      speaker_dir / "coverage_report.json", root / "state/latents/spec.json",
                      Path(config.datasets.vocab_path), Path(config.datasets.normalization_path)]
    artifact_paths.extend(manifest.parent / name for name in meta["manifests"])
    artifact_paths.extend(data.text_policy_artifact_paths)
    artifact_paths.extend(data.media_quality_artifact_paths)
    artifacts = [snapshot(path) for path in dict.fromkeys(artifact_paths)]
    artifacts.extend(data.text_evidence_artifacts)
    source_pins = {str(source_path.resolve()): source_contract_digest,
                   str(source_inventory.resolve()): source_inventory_digest}
    if any(artifact["sha256"] != source_pins[artifact["path"]]
           for artifact in artifacts if artifact["path"] in source_pins):
        raise RuntimeError("Canonical LRS3 source artifacts changed during launch validation")
    summary = {"source_pairs": source["source_media_pairs"], "source_original_ids": source["original_clip_count"],
               "source_contract_sha256": source_contract_digest, "source_inventory_sha256": source_inventory_digest,
               "text_policy": data.data_contract["text_policy"],
               "source_kind_counts": data.data_contract["source_kind_counts"],
               "strict_baseline": data.data_contract["strict_baseline"],
               "accepted_pairs": meta["training_ready_pairs"], "excluded_pairs": excluded,
               "accepted_split_counts": readiness["split_counts"], "exclusion_reasons": reasons,
               "accepted_audio_hours": readiness["audio_hours_by_split"],
               "scope": "All source records audited; training uses the explicitly accepted publication only"}
    if 'media_quality' in data.data_contract:summary['media_quality'] = data.data_contract['media_quality']
    return data, artifacts, summary


def verify_pins(artifacts):
    for expected in artifacts:
        actual = snapshot(expected["path"])
        if actual != expected:
            raise RuntimeError(f"Frozen training input changed: {expected['path']}")


def validate_devices(devices, port):
    selected = devices.split(",")
    if len(selected) != 4 or len(set(selected)) != 4 or any(not value.isdigit() for value in selected):
        raise ValueError("Specify exactly four distinct GPU indices, for example --gpus 0,1,2,3")
    for device in selected:
        result = subprocess.run(["nvidia-smi", f"--id={device}", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        if any(line.strip().isdigit() for line in result.stdout.splitlines()):
            raise RuntimeError(f"GPU {device} still has a compute process; training was not launched")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def ensure_tensorboard(output, event_dir, port):
    logdir = output / "tensorboard"
    logdir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    link = logdir / "lrs3_training"
    if not link.exists():
        link.symlink_to(event_dir, target_is_directory=True)
    elif link.resolve() != event_dir.resolve():
        raise RuntimeError("Existing TensorBoard link belongs to another run")
    session = requests.Session()
    session.trust_env = False
    url = f"http://127.0.0.1:{port}"
    try:
        response = session.get(url + "/data/environment", timeout=2)
    except requests.RequestException:
        response = None
    pid = None
    if response is not None:
        try:
            location = response.json()["data_location"]
        except (ValueError, KeyError) as error:
            raise RuntimeError(f"Port {port} is occupied by a different service") from error
        if Path(location).resolve() != logdir.resolve():
            raise RuntimeError(f"TensorBoard port {port} belongs to another log directory: {location}")
    else:
        with (output / "tensorboard.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "tensorboard.main", "--logdir", str(logdir),
                                        "--host", "0.0.0.0", "--port", str(port)], stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            pid = process.pid
        for _ in range(30):
            try:
                if session.get(url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("TensorBoard did not become HTTP-ready")
    result = {"pid": pid, "local_url": url, "port": port, "logdir": str(logdir),
              "http_status": session.get(url, timeout=3).status_code,
              "scalar_tags": session.get(url + "/data/plugin/scalars/tags", timeout=3).json(),
              "forwarded_url": None, "forwarding_note": "No external port-forwarding tool is available; URL is local to the server."}
    write_json(output / "tensorboard.json", result)
    return result


def worker(plan_path):
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output_dir"])
    status_path = output / "training_process.json"
    state = {"status": "validating", "pid": os.getpid(), "process_group": os.getpgrp(),
             "started_unix": time.time(), "plan": str(plan_path), "scope": plan["scope"]}
    wrote_status = False
    try:
        with ExitStack() as stack:
            launch_lock = stack.enter_context((output / "launch.lock").open("a"))
            fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(status_path, state)
            wrote_status = True
            descriptors = acquire_read_locks(Path(plan["cache_root"]), stack)
            verify_pins(plan["artifacts"])
            validate_devices(plan["gpus"], plan["master_port"])
            tb = ensure_tensorboard(output, Path(plan["event_dir"]), plan["tensorboard_port"])
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=plan["gpus"], ACCELERATE_MIXED_PRECISION="bf16",
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                               WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONPATH=str(PROJECT / "src"))
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                environment.pop(name, None)
            with (output / "training.log").open("ab") as log:
                process = subprocess.Popen(plan["command"], cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(descriptors + [launch_lock.fileno()]))
                state.update(status="running", torchrun_pid=process.pid, tensorboard=tb)
                write_json(status_path, state)
                session = requests.Session()
                session.trust_env = False
                while process.poll() is None:
                    state["heartbeat_unix"] = time.time()
                    try:
                        tags = session.get(tb["local_url"] + "/data/plugin/scalars/tags", timeout=3).json()
                        state["tensorboard_loss_visible"] = any("loss" in values for values in tags.values())
                    except (requests.RequestException, ValueError):
                        state["tensorboard_loss_visible"] = False
                    write_json(status_path, state)
                    time.sleep(10)
            state.update(status="completed" if process.returncode == 0 else "failed",
                         returncode=process.returncode, finished_unix=time.time())
            write_json(status_path, state)
            return process.returncode
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
        if wrote_status:
            write_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["strict", "pseudo"], default="strict",
                        help="Strict source-grounded publication by default; pseudo uses its separate pinned publication")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name", help="Separate model/run identity and TensorBoard events; retain the fixed-band marker")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--master-port", type=int, default=29587)
    parser.add_argument("--tensorboard-port", type=int)
    parser.add_argument("--run-until-update", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--audit-cache", action="store_true", help="Read every accepted training array before launch")
    parser.add_argument("--start", action="store_true", help="Start real four-GPU training in a detached session")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        raise SystemExit(worker(args.worker_plan.resolve()))
    config = load_config(args)
    if args.tensorboard_port is None:
        args.tensorboard_port = 6028 if args.variant == "strict" else 6029
    root = Path(config.datasets.cache_root).resolve()
    publication = "full" if args.variant == "strict" else "full_pseudo"
    output = (args.output_dir or root / f"reports/{publication}_training_launch").resolve()
    output.mkdir(parents=True, exist_ok=True)
    active_path = output / "training_process.json"
    if active_path.exists():
        previous = json.loads(active_path.read_text())
        if previous.get("status") in {"validating", "running"}:
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("This launch directory already has an active training process")
    with ExitStack() as stack:
        acquire_read_locks(root, stack)
        data, artifacts, scope = validate_publication(config)
        if args.audit_cache:
            write_json(output / "loader_cache_audit.json", data.audit_cache())
        from aligndit.model.fixed_temporal_band_contract import validate_fixed_band_training_config
        from aligndit.model.libritts_pretrained_migration import validate_libritts_checkpoint_artifact
        validate_fixed_band_training_config(config)
        parent = validate_libritts_checkpoint_artifact(config.ckpts.pretrained_path, config)
        artifacts.append({"path": parent["path"], "sha256": parent["sha256"], "size_bytes": parent["size"]})
        config_path = output / "resolved_config.json"
        write_json(config_path, OmegaConf.to_container(config, resolve=True))
        frozen_yaml = output / "resolved_training_config.yaml"
        frozen_yaml.write_text(OmegaConf.to_yaml(config, resolve=True))
        artifacts.extend([snapshot(config_path), snapshot(frozen_yaml)])
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--nproc_per_node=4",
                   f"--master_port={args.master_port}", "--module", "aligndit.script.train.finetune_lrs3_svae_speaker",
                   "--config-path", str(output), "--config-name", frozen_yaml.stem]
        plan = {"schema_version": 1, "variant": args.variant, "cache_root": str(root), "output_dir": str(output), "command": command,
                "gpus": args.gpus, "master_port": args.master_port, "tensorboard_port": args.tensorboard_port,
                "event_dir": str(training_event_dir(config)), "scope": scope, "artifacts": artifacts,
                "data_contract": data.data_contract, "config_sha256": sha256_file(config_path)}
        plan_path = output / "launch_plan.json"
        old_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
        if old_plan is not None and old_plan["artifacts"] != plan["artifacts"]:
            raise RuntimeError("Existing launch plan pins another publication; use a distinct output directory")
        write_json(plan_path, plan)
        print(json.dumps({"validated": True, "scope": scope, "plan": str(plan_path), "command": command}, indent=2))
        if args.start:
            existing_path = output / "training_process.json"
            if existing_path.exists():
                previous = json.loads(existing_path.read_text())
                if previous["status"] in {"validating", "running"}:
                    try:
                        os.kill(previous["pid"], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError("This training launcher is already active")
            with (output / "launcher.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                            "--worker-plan", str(plan_path)], cwd=PROJECT,
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            print(json.dumps({"launched_pid": process.pid, "process_group": process.pid,
                              "status_file": str(output / "training_process.json"),
                              "local_tensorboard_url": f"http://127.0.0.1:{args.tensorboard_port}",
                              "external_forwarded_url": None}, indent=2))


if __name__ == "__main__":
    main()
