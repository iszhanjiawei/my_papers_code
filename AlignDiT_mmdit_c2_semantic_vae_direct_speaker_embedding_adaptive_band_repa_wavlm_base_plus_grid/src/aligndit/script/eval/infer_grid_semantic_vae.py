"""Generate the complete GRID validation split from a frozen GRID EMA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from aligndit.model.trainer_grid_semantic_vae import contract_hash
from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder, sha256_file
from aligndit.script.train.finetune_grid_semantic_vae import build_dataset, build_model


SAMPLE_RATE = 16_000
EXPECTED_VAL_COUNT = 3_281


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects: {path}")
    return rows


def load_config(config_path: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent.resolve())):
        return compose(config_name=config_path.stem)


def grid_id(record: dict) -> str:
    path = Path(record["audio_relative_path"])
    if path.parts[:1] != ("val",) or len(path.parts) != 3 or path.suffix != ".wav":
        raise ValueError(f"Invalid GRID validation path: {path}")
    return f"{path.parent.name}/{path.stem}"


def validate_pairs(dataset, shared_manifest: Path, setting: int) -> list[tuple[int, int, dict]]:
    rows = read_jsonl(shared_manifest)
    if len(rows) != EXPECTED_VAL_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_VAL_COUNT} shared GRID rows, found {len(rows)}")
    by_id = {grid_id(record): index for index, record in enumerate(dataset.records)}
    if len(by_id) != EXPECTED_VAL_COUNT or set(by_id) != {row.get("id") for row in rows}:
        raise RuntimeError("Shared GRID evaluation manifest and immutable validation cache disagree")
    pairs = []
    for row in rows:
        target_id = row["id"]
        reference_id = target_id if setting == 1 else row.get("reference_id")
        if reference_id not in by_id:
            raise KeyError(f"Reference is absent from GRID validation cache: {reference_id}")
        target_index, reference_index = by_id[target_id], by_id[reference_id]
        target = dataset.records[target_index]
        reference = dataset.records[reference_index]
        if reference["speaker_id"] != target["speaker_id"] or (setting == 2 and reference_id == target_id):
            raise RuntimeError(f"Invalid setting-{setting} GRID pair: {reference_id} -> {target_id}")
        if Path(target["audio_path"]).resolve() != Path(row["gt_wav"]).resolve():
            raise RuntimeError(f"Ground-truth path disagreement for {target_id}")
        pairs.append((target_index, reference_index, row))
    return pairs


def load_ema(model, checkpoint_path: Path, step: int, checkpoint_dir: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    training_contract = read_json(checkpoint_dir / "grid_training_contract.json")
    expected_contract_hash = contract_hash(training_contract)
    if (
        checkpoint.get("grid_checkpoint_schema") != 1
        or int(checkpoint.get("update", -1)) != step
        or checkpoint.get("training_contract_sha256") != expected_contract_hash
    ):
        raise RuntimeError("GRID checkpoint does not match its immutable training contract")
    ema = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema, dict) or not bool(ema.get("initted")):
        raise RuntimeError("GRID checkpoint has no initialized EMA model")
    selected = {key.removeprefix("ema_model."): value for key, value in ema.items() if key not in {"initted", "step"}}
    model.load_state_dict(selected, strict=True)
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size": checkpoint_path.stat().st_size,
        "update": step,
        "weights": "EMA",
        "ema_step": int(ema["step"]),
        "training_contract_sha256": expected_contract_hash,
    }
    del checkpoint, ema, selected
    return metadata


def historical_text(reference: str, target: str) -> str:
    reference = reference.strip()
    if reference and len(reference[-1].encode("utf-8")) == 1:
        reference += " "
    return reference + " " + target.strip()


def write_waveform(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.wav")
    try:
        torchaudio.save(str(temporary), waveform, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_existing_waveform(path: Path, expected_samples: int) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        metadata = torchaudio.info(str(path))
    except Exception:  # noqa: BLE001 - a malformed partial output must be replaced
        return False
    return (
        metadata.sample_rate == SAMPLE_RATE and metadata.num_channels == 1 and metadata.num_frames == expected_samples
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal GRID inference requires CUDA")
    if args.nshard < 1 or not 0 <= args.rank < args.nshard or args.limit < 0:
        raise ValueError("Require nshard >= 1, 0 <= rank < nshard and limit >= 0")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    config = load_config(args.config)
    dataset = build_dataset(config, split="val")
    if dataset.is_subset or len(dataset) != EXPECTED_VAL_COUNT:
        raise RuntimeError("Formal GRID evaluation requires the complete 3,281-record validation cache")
    pairs = validate_pairs(dataset, args.shared_manifest.resolve(strict=True), args.setting)
    if args.limit:
        pairs = pairs[: args.limit]
    assigned = [(index, pair) for index, pair in enumerate(pairs) if index % args.nshard == args.rank]
    if not assigned:
        raise RuntimeError("This inference shard has no assigned validation records")

    model = build_model(config)
    model.odeint_kwargs = {"method": "euler"}
    checkpoint_metadata = load_ema(model, args.checkpoint.resolve(strict=True), args.step, args.checkpoint.parent)
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    contract = dataset.data_contract
    decoder, decoder_metadata = load_semantic_vae_decoder(
        repo=args.semantic_vae_repo,
        checkpoint_root=args.semantic_vae_checkpoint,
        cache_spec=contract["semantic_vae_contract"],
        device=device,
    )
    mean = torch.from_numpy(dataset.latent_mean).to(device)
    std = torch.from_numpy(dataset.latent_std).to(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    generated = []

    with torch.inference_mode():
        for global_index, (target_index, reference_index, shared_row) in tqdm(
            assigned, desc=f"GRID S{args.setting} EMA{args.step} shard {args.rank}/{args.nshard}"
        ):
            target = dataset[target_index]
            reference = dataset[reference_index]
            target_record, reference_record = dataset.records[target_index], dataset.records[reference_index]
            target_frames = int(target_record["latent_frames"])
            reference_frames = int(reference_record["latent_frames"])
            destination = output / "test" / f"{shared_row['id']}.wav"
            expected_samples = int(target_record["original_num_samples"])
            if valid_existing_waveform(destination, expected_samples):
                generated.append(
                    {
                        "id": shared_row["id"],
                        "reference_id": grid_id(reference_record),
                        "relative_path": destination.relative_to(output).as_posix(),
                        "samples": expected_samples,
                        "sha256": sha256_file(destination),
                        "seed": args.seed + global_index,
                        "reused": True,
                    }
                )
                continue

            condition = reference["mel_spec"].transpose(0, 1).unsqueeze(0).to(device)
            target_video = target["video"].to(device)
            full_video = torch.cat((torch.zeros(reference_frames, 1024, device=device), target_video), dim=0).unsqueeze(
                0
            )
            sampled, _ = model.sample(
                cond=condition,
                text=[historical_text(reference_record["text"], target_record["text"])],
                duration=torch.tensor([reference_frames + target_frames], device=device),
                video=full_video,
                lens=torch.tensor([reference_frames], device=device),
                speaker_embedding=reference["speaker_embedding"].unsqueeze(0).to(device),
                steps=args.nfe,
                cfg_strength=args.cfg_text,
                cfg_strength_v=args.cfg_video,
                sway_sampling_coef=args.sway,
                seed=args.seed + global_index,
                use_epss=True,
            )
            normalized = sampled[:, reference_frames : reference_frames + target_frames].float()
            if normalized.shape != (1, target_frames, 64) or not torch.isfinite(normalized).all():
                raise RuntimeError(f"Invalid sampled GRID latent: {shared_row['id']}")
            waveform = decoder((normalized * std + mean).transpose(1, 2)).squeeze(0).float().cpu()
            padded_samples = target_frames * 400
            if waveform.shape != (1, padded_samples) or not torch.isfinite(waveform).all():
                raise RuntimeError(f"Invalid decoded GRID waveform: {shared_row['id']}")
            waveform = waveform[:, :expected_samples]
            write_waveform(destination, waveform)
            generated.append(
                {
                    "id": shared_row["id"],
                    "reference_id": grid_id(reference_record),
                    "relative_path": destination.relative_to(output).as_posix(),
                    "samples": expected_samples,
                    "sha256": sha256_file(destination),
                    "seed": args.seed + global_index,
                    "reused": False,
                }
            )

    summary = {
        "schema_version": 1,
        "checkpoint": checkpoint_metadata,
        "dataset_contract_sha256": dataset.contract_sha256,
        "shared_manifest": str(args.shared_manifest.resolve()),
        "shared_manifest_sha256": sha256_file(args.shared_manifest),
        "setting": args.setting,
        "protocol": (
            "target utterance itself supplies audio prompt and CAM++"
            if args.setting == 1
            else "same-speaker distinct validation utterance supplies audio prompt and CAM++; target audio is not read"
        ),
        "rank": args.rank,
        "nshard": args.nshard,
        "expected_full_count": len(pairs),
        "generated_count": len(generated),
        "generation": {
            "nfe": args.nfe,
            "ode_method": "euler",
            "use_epss": True,
            "sway": args.sway,
            "cfg_text": args.cfg_text,
            "cfg_video": args.cfg_video,
            "base_seed": args.seed,
            "per_sample_seed": "base_seed + global manifest index",
            "duration": "target ground-truth video length",
        },
        "decoder": decoder_metadata,
        "elapsed_seconds": time.time() - started,
        "outputs": generated,
    }
    summary_path = output / f"inference_summary.rank{args.rank}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "outputs"}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    project = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project / "src/aligndit/config/finetune_grid_mmdit.yaml")
    parser.add_argument(
        "--semantic-vae-repo",
        type=Path,
        default=Path(f"{prefix}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE"),
    )
    parser.add_argument(
        "--semantic-vae-checkpoint",
        type=Path,
        default=Path(f"{prefix}/zjw524/projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k"),
    )
    parser.add_argument("--setting", type=int, choices=(1, 2), default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--cfg-text", type=float, default=5.0)
    parser.add_argument("--cfg-video", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
