"""Compare native mel and Semantic-VAE audio inpainting on identical waveforms.

The two representations' MSE values are deliberately never compared. Generated
regions are scored against original speech, their own codec reconstruction, and
the observed reference context. Seed repetitions are averaged within utterance
before paired, subset-stratified bootstrap resampling.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import soundfile as sf
import torch

from aligndit.script.eval.compare_native_audio_pretrains import SVAE_CHECKPOINTS
from aligndit.script.misc.svae_cache_utils import atomic_write_json, atomic_write_jsonl, safe_join, sha256_file


SAMPLE_RATE = 16000
BRANCHES = ("mel", "svae")
SEEDS = (666, 667, 668)
CHECKPOINTS = {
    "mel": "4a9fc0e526ce47745aee839348406ca99597d32f5ed028bda42a3de3ec900fcd",
    "svae": "02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4",
}
COMMON_FIELDS = (
    "utterance_key",
    "subset",
    "speaker_id",
    "original_num_samples",
    "mask_start_sample",
    "mask_end_sample",
    "source_audio_sha256",
    "reference_input",
)
WARNINGS = [
    "This is no-text, no-video audio inpainting; the missing words are not supplied to either model.",
    "STOI and SI-SDR measure agreement with the missing target waveform, not unconstrained generation quality.",
    "All model scores use only the missing region. Observed audio cannot inflate them.",
    "SPKSIM is WavLM-Large/ECAPA cosine; it is not a calibrated percentage of speaker identity.",
    "Legacy EMOSIM is emotion-class score cosine; true emotion embedding cosine is reported separately.",
    "No human listening scores, WER, or AVSync were measured by this evaluator.",
    "Codec reconstructions are a representation control, not a numerical bound on every similarity metric.",
    "Both context encoders see the original waveform with the missing interval zeroed; codec oracles alone see full speech.",
    "Native encoders have unequal receptive fields, and zeroed mask boundaries may affect adjacent context features.",
    "SI-SDR uses an epsilon of 1e-12 and a -120 dB sentinel for a silent generated crop; RMS is also reported.",
    "Seed repetitions are dependent: average 3 seeds per utterance, then bootstrap utterances within subset.",
    "The 50 selected utterances are a pilot, not a full-development or downstream dubbing benchmark.",
]


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def bound_manifest(folder: Path, complete: dict[str, Any], field: str) -> tuple[list[dict], str]:
    binding = complete[field]
    path = safe_join(folder, binding["path"])
    digest = sha256_file(path)
    if digest != binding["sha256"]:
        raise ValueError(f"Changed manifest: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != complete["count"] or not rows or path.stat().st_size != binding["size_bytes"]:
        raise ValueError(f"Wrong manifest count: {path}")
    return rows, digest


def checked_wave(root: Path, info: dict[str, Any], expected_samples: int, cache: dict) -> np.ndarray:
    path = safe_join(root, info["path"])
    key = (str(path), info["sha256"], expected_samples)
    if key not in cache:
        if not path.is_file() or path.stat().st_size != info["size_bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError(f"Missing or changed waveform: {path}")
        wave, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE or wave.ndim != 1 or len(wave) != expected_samples:
            raise ValueError(f"Unexpected waveform geometry at {path}: {sample_rate}, {wave.shape}")
        if not np.isfinite(wave).all():
            raise ValueError(f"Non-finite waveform: {path}")
        cache[key] = wave
    return cache[key]


def load_run(
    root: Path, canary_limit: int | None = None, svae_update: int = 70000
) -> tuple[list[dict], dict[str, list[dict]], dict, dict]:
    """Reject partial runs, duplicate draws, stale WAVs, or mismatched physical spans."""
    common_complete = read_object(root / "common/complete.json")
    # The immutable common fixture records the ORIGINAL mel/70k comparison.
    # A longitudinal candidate changes only its explicit branch identity.
    expected_checkpoints = {**CHECKPOINTS, "svae": SVAE_CHECKPOINTS[svae_update]}
    common, common_sha = bound_manifest(root, common_complete, "manifest")
    if common_complete.get("schema_version") != 1 or common_complete.get("checkpoints") != CHECKPOINTS:
        raise ValueError("Unexpected source checkpoint identities")
    if common_complete.get("protocol", {}).get("sampling_seeds") != list(SEEDS):
        raise ValueError("Unexpected sampling protocol")
    protocol = common_complete["protocol"]
    if (
        protocol.get("name") != "librispeech-native-mel500k-svae70k-waveform-masked-inpainting-v2"
        or protocol.get("conditioning")
        != "same source waveform zeroed inside physical mask BEFORE native encoding; no text/video/HuBERT"
        or protocol.get("context_latent")
        != "fixed keyed posterior sample of zero-masked waveform; clean cached latents are oracle only"
    ):
        raise ValueError("Only the waveform-masked, leakage-controlled v2 protocol is supported")
    if len(common) != 50 or {row["subset"] for row in common} != {"dev-clean", "dev-other"}:
        raise ValueError("Formal comparison requires the fixed 50-item clean/other selection")
    by_key = {row["utterance_key"]: row for row in common}
    if len(by_key) != len(common):
        raise ValueError("Duplicate common utterance key")
    if any(sum(row["subset"] == subset for row in common) != 25 for subset in ("dev-clean", "dev-other")):
        raise ValueError("Formal comparison requires 25 items per subset")
    if len({row["speaker_id"] for row in common}) != 50:
        raise ValueError("Pilot selection must contain one utterance per speaker")
    provenance: dict[str, Any] = {"common": common_complete, "common_manifest_sha256": common_sha}
    waves: dict = {}
    for row in common:
        start, end, total = row["mask_start_sample"], row["mask_end_sample"], row["original_num_samples"]
        if not 0 <= start < end <= total or start % 800 or end % 800:
            raise ValueError("Mask must occupy the same exact 50 ms sample grid in both representations")
        source = Path(row["source_audio_path"])
        if not source.is_file() or sha256_file(source) != row["source_audio_sha256"]:
            raise ValueError(f"Changed source audio: {source}")
        full = checked_wave(root, row["reference_full"], total, waves)
        missing = checked_wave(root, row["reference_masked"], end - start, waves)
        context = checked_wave(root, row["reference_context"], total - (end - start), waves)
        observed_input = checked_wave(root, row["reference_input"], total, waves)
        if not np.array_equal(missing, full[start:end]):
            raise ValueError("Reference masked waveform does not match its full-wave sample span")
        if not np.array_equal(context, np.concatenate((full[:start], full[end:]))):
            raise ValueError("Reference context does not contain exactly the observed samples")
        expected_input = full.copy()
        expected_input[start:end] = 0
        if not np.array_equal(observed_input, expected_input):
            raise ValueError("Encoder input must replace exactly the missing raw-wave interval with zeros")
    if canary_limit is not None:
        if not 1 <= canary_limit < 50:
            raise ValueError("Canary limit must be between 1 and 49")
        common = common[:canary_limit]
        by_key = {row["utterance_key"]: row for row in common}
    provenance["canary_limit"] = canary_limit
    branches = {}
    for branch in BRANCHES:
        folder = branch if canary_limit is None else f"{branch}_canary{canary_limit}"
        complete = read_object(root / folder / "generation_complete.json")
        rows, digest = bound_manifest(root, complete, "generation_manifest")
        if (
            complete.get("branch") != branch
            or complete.get("common_manifest_sha256") != common_sha
            or complete.get("common_complete_sha256") != sha256_file(root / "common/complete.json")
            or complete.get("protocol") != common_complete["protocol"]
            or complete.get("waveform_complete") is not True
            or complete.get("canary_limit") != canary_limit
            or complete.get("checkpoint", {}).get("sha256") != expected_checkpoints[branch]
        ):
            raise ValueError(f"Unbound branch completion: {branch}")
        expected = {(key, seed) for key in by_key for seed in SEEDS}
        observed = [(row["utterance_key"], row["sampling_seed"]) for row in rows]
        if len(rows) != len(common) * 3 or len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError(f"Branch {branch} must contain exactly {len(common)} utterances x 3 seeds")
        if len({row["generated_masked"]["path"] for row in rows}) != len(rows):
            raise ValueError("Generated draws must bind distinct output files")
        oracle_bindings = {}
        for row in rows:
            source = by_key[row["utterance_key"]]
            if row.get("branch") != branch or any(row.get(name) != source[name] for name in COMMON_FIELDS):
                raise ValueError(f"Generation metadata does not match shared selection: {row['utterance_key']}")
            start, end, total = source["mask_start_sample"], source["mask_end_sample"], source["original_num_samples"]
            for kind in ("generated", "oracle"):
                full = checked_wave(root, row[f"{kind}_full"], total, waves)
                masked = checked_wave(root, row[f"{kind}_masked"], end - start, waves)
                if not np.array_equal(masked, full[start:end]):
                    raise ValueError(f"Wrong {kind} missing-region crop: {row['utterance_key']}")
            oracle = (row["oracle_full"], row["oracle_masked"])
            if row["utterance_key"] in oracle_bindings and oracle_bindings[row["utterance_key"]] != oracle:
                raise ValueError("Codec oracle must be identical across sampling seeds")
            oracle_bindings[row["utterance_key"]] = oracle
        branches[branch] = rows
        provenance[branch] = complete
        provenance[f"{branch}_manifest_sha256"] = digest
    return common, branches, provenance, waves


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Incompatible or non-finite embeddings")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        raise ValueError("Zero-norm embedding")
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))


def si_sdr(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = reference.astype(np.float64) - np.mean(reference, dtype=np.float64)
    prediction = prediction.astype(np.float64) - np.mean(prediction, dtype=np.float64)
    energy = float(reference @ reference)
    if energy < 1e-12:
        raise ValueError("Reference region is silent; SI-SDR is undefined")
    if float(prediction @ prediction) < 1e-12:
        return -120.0
    projected = reference * float(prediction @ reference) / energy
    # Epsilon prevents infinity for exact reconstructions or zero target projection.
    return float(
        10 * np.log10((projected @ projected + 1e-12) / ((prediction - projected) @ (prediction - projected) + 1e-12))
    )


def waveform_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    from pystoi import stoi

    if reference.shape != generated.shape:
        raise ValueError("Waveform metrics require exact equal-length crops")
    scores = {
        "stoi": float(stoi(reference, generated, SAMPLE_RATE, extended=False)),
        "si_sdr_db": si_sdr(reference, generated),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Non-finite waveform metric")
    return scores


def metric_audio_items(common: list[dict], branches: dict[str, list[dict]]) -> dict[str, dict]:
    """Unique physical crops; reference context is concatenated observed speech."""
    items = {}
    for row in common:
        for name in ("reference_masked", "reference_context"):
            info = row[name]
            if (
                name == "reference_context"
                and row["original_num_samples"] - (row["mask_end_sample"] - row["mask_start_sample"]) < SAMPLE_RATE
            ):
                continue
            items[info["path"]] = info
    for rows in branches.values():
        for row in rows:
            for name in ("generated_masked", "oracle_masked"):
                items[row[name]["path"]] = row[name]
    return items


def model_file_provenance(paths: list[Path]) -> list[dict]:
    return [{"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)} for path in paths]


def embedding_metrics(root: Path, items: dict[str, dict], args: argparse.Namespace) -> tuple[dict, dict]:
    """Load one metric model at a time, never download assets implicitly."""
    from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

    device = torch.device(args.device)
    result: dict[str, dict] = {path: {} for path in items}
    provenance = {"speaker": model_file_provenance([args.speaker_model, args.speaker_base])}
    speaker = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", wavlm_ckpt_path=str(args.speaker_base))
    checkpoint = torch.load(args.speaker_model, map_location="cpu", weights_only=True)
    state = checkpoint["model"].copy()
    del state["loss_calculator.projection.weight"]
    speaker.load_state_dict(state, strict=True)
    speaker = speaker.eval().requires_grad_(False).to(device)
    del checkpoint, state
    with torch.inference_mode():
        for index, relative in enumerate(items):
            wave, _ = sf.read(safe_join(root, relative), dtype="float32")
            result[relative]["speaker"] = (
                speaker(torch.from_numpy(wave).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            )
            if index % 50 == 0:
                print(f"speaker embeddings {index}/{len(items)}", flush=True)
    del speaker
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.emotion_model is not None:
        from funasr import AutoModel

        assets = [args.emotion_model / name for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt")]
        provenance["emotion"] = model_file_provenance(assets)
        model = AutoModel(model=str(args.emotion_model.resolve(strict=True)), disable_update=True, device=str(device))
        with torch.inference_mode():
            for index, relative in enumerate(items):
                outputs = model.generate(
                    str(safe_join(root, relative)),
                    output_dir=None,
                    granularity="utterance",
                    extract_embedding=True,
                    disable_pbar=True,
                )
                if len(outputs) != 1:
                    raise ValueError("Emotion model returned unexpected batch size")
                result[relative]["emotion_embedding"] = np.asarray(outputs[0]["feats"])
                result[relative]["emotion_scores"] = np.asarray(outputs[0]["scores"])
                if index % 50 == 0:
                    print(f"emotion embeddings {index}/{len(items)}", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result, provenance


def scores_for_pair(generated: dict, reference: dict, embeddings: dict) -> dict[str, float]:
    left, right = embeddings[generated["path"]], embeddings[reference["path"]]
    names = {
        "speaker": "spksim",
        "emotion_embedding": "emotion_embedding_cosine",
        "emotion_scores": "legacy_emosim_scores_cosine",
    }
    if left.keys() != right.keys():
        raise ValueError("Missing metric embedding in pair")
    return {names[name]: cosine(left[name], right[name]) for name in left}


def measure_rows(
    root: Path, common: list[dict], branches: dict, waves: dict, embeddings: dict
) -> tuple[list[dict], list[dict]]:
    by_key = {row["utterance_key"]: row for row in common}
    generated_results, oracle_results = [], []
    seen_oracles = set()
    for branch, rows in branches.items():
        for row in rows:
            key = row["utterance_key"]
            shared = by_key[key]
            samples = shared["mask_end_sample"] - shared["mask_start_sample"]
            original = checked_wave(root, shared["reference_masked"], samples, waves)
            oracle = checked_wave(root, row["oracle_masked"], samples, waves)
            generated = checked_wave(root, row["generated_masked"], samples, waves)
            result = {
                "utterance_key": key,
                "subset": row["subset"],
                "speaker_id": row["speaker_id"],
                "branch": branch,
                "sampling_seed": row["sampling_seed"],
                "missing_seconds": samples / SAMPLE_RATE,
            }
            scores = waveform_metrics(original, generated) | scores_for_pair(
                row["generated_masked"], shared["reference_masked"], embeddings
            )
            result.update({f"{name}_vs_original": value for name, value in scores.items()})
            result.update(
                {f"{name}_vs_own_oracle": value for name, value in waveform_metrics(oracle, generated).items()}
            )
            context_info = shared["reference_context"]
            result["spksim_vs_context"] = (
                cosine(
                    embeddings[row["generated_masked"]["path"]]["speaker"], embeddings[context_info["path"]]["speaker"]
                )
                if context_info["path"] in embeddings
                else None
            )
            result["rms"] = float(np.sqrt(np.mean(generated.astype(np.float64) ** 2)))
            result["abs_ge_1_fraction"] = float(np.mean(np.abs(generated) >= 1))
            generated_results.append(result)
            if (branch, key) not in seen_oracles:
                seen_oracles.add((branch, key))
                control = {
                    "utterance_key": key,
                    "subset": row["subset"],
                    "speaker_id": row["speaker_id"],
                    "branch": branch,
                }
                scores = waveform_metrics(original, oracle) | scores_for_pair(
                    row["oracle_masked"], shared["reference_masked"], embeddings
                )
                control.update({f"{name}_vs_original": value for name, value in scores.items()})
                control["spksim_vs_context"] = (
                    cosine(
                        embeddings[row["oracle_masked"]["path"]]["speaker"], embeddings[context_info["path"]]["speaker"]
                    )
                    if context_info["path"] in embeddings
                    else None
                )
                oracle_results.append(control)
    return generated_results, oracle_results


def summarize_paired(rows: list[dict], *, generated: bool, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("Result rows must have one nonempty, consistent metric schema")
    metric_names = sorted(
        set(rows[0]) - {"utterance_key", "subset", "speaker_id", "branch", "sampling_seed", "missing_seconds"}
    )
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["branch"], row["utterance_key"]), []).append(row)
    reduced = {}
    for (branch, key), draws in grouped.items():
        if branch not in BRANCHES or len({(row["subset"], row["speaker_id"]) for row in draws}) != 1:
            raise ValueError("Inconsistent metric identity across seeds")
        if generated and {row["sampling_seed"] for row in draws} != set(SEEDS):
            raise ValueError("Incomplete seed replication in results")
        expected_count = 3 if generated else 1
        if len(draws) != expected_count:
            raise ValueError("Duplicate or missing metric rows")
        values = {}
        for name in metric_names:
            entries = [draw[name] for draw in draws]
            if all(value is None for value in entries):
                values[name] = None
            elif any(value is None or not np.isfinite(value) for value in entries):
                raise ValueError(f"Partial/nonfinite metric: {branch}, {key}, {name}")
            else:
                values[name] = float(np.mean(entries))
        reduced[(branch, key)] = {"subset": draws[0]["subset"], **values}
    keys = sorted(key for branch, key in reduced if branch == "mel")
    if set(keys) != {key for branch, key in reduced if branch == "svae"}:
        raise ValueError("Metric branches have unequal utterance sets")
    if any(reduced[("mel", key)]["subset"] != reduced[("svae", key)]["subset"] for key in keys):
        raise ValueError("Paired metric branches disagree on subset")
    result = {}
    for subset in ("overall", "dev-clean", "dev-other"):
        selected = [key for key in keys if subset == "overall" or reduced[("mel", key)]["subset"] == subset]
        if not selected:
            result[subset] = {"utterances": 0, "metrics": {}}
            continue
        group = {"utterances": len(selected), "metrics": {}}
        for name in metric_names:
            eligible = [
                key
                for key in selected
                if reduced[("mel", key)][name] is not None and reduced[("svae", key)][name] is not None
            ]
            if not eligible:
                group["metrics"][name] = {"paired_utterances": 0}
                continue
            mel = np.asarray([reduced[("mel", key)][name] for key in eligible])
            svae = np.asarray([reduced[("svae", key)][name] for key in eligible])
            delta = svae - mel
            digest = hashlib.sha256(f"{bootstrap_seed}:{subset}:{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            strata = [
                np.asarray([i for i, key in enumerate(eligible) if reduced[("mel", key)]["subset"] == sub])
                for sub in ("dev-clean", "dev-other")
            ]
            strata = [indices for indices in strata if len(indices)]
            draws = np.concatenate(
                [rng.choice(indices, size=(bootstrap_samples, len(indices)), replace=True) for indices in strata],
                axis=1,
            )
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            group["metrics"][name] = {
                "paired_utterances": len(eligible),
                "mel_mean": float(mel.mean()),
                "svae_mean": float(svae.mean()),
                "svae_minus_mel": float(delta.mean()),
                "delta_ci95": interval.tolist(),
            }
        result[subset] = group
    return result


def write_listening_page(
    output: Path, root: Path, common: list[dict], branches: dict, svae_update: int = 70000
) -> None:
    indexed = {
        branch: {(row["utterance_key"], row["sampling_seed"]): row for row in rows} for branch, rows in branches.items()
    }
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Native audio pretrained model comparison</title><style>body{font-family:sans-serif;margin:2rem;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}audio{width:250px}summary{cursor:pointer}p{line-height:1.5}</style>",
        f"<h1>Original mel500k vs Semantic-VAE S2c{svae_update // 1000}k</h1><p>Named, non-blind listening page. No human listening scores have been collected. Players below contain only the missing region. No transcript was supplied to either model. Compare codec controls before attributing differences to the flow model.</p>",
    ]
    if len(common) != 50:
        parts.append(
            f"<p><strong>CANARY ONLY: {len(common)} utterances. This is a pipeline check, not a formal comparison.</strong></p>"
        )

    def player(info: dict) -> str:
        relative = os.path.relpath(safe_join(root, info["path"]), output)
        return f'<audio controls preload="none" src="{html.escape(quote(relative, safe="/"))}"></audio>'

    for row in common:
        key = row["utterance_key"]
        mel, svae = indexed["mel"][(key, SEEDS[0])], indexed["svae"][(key, SEEDS[0])]
        parts.append(
            f"<h2>{html.escape(key)}</h2><p>{html.escape(row['subset'])}; hidden [{row['mask_start_sample'] / SAMPLE_RATE:.2f}, {row['mask_end_sample'] / SAMPLE_RATE:.2f}) s</p>"
        )
        parts.append(
            "<table><tr><th>Original missing region</th><th>Mel codec oracle</th><th>Semantic-VAE codec oracle</th><th>Observed context (joined)</th></tr><tr>"
            + "".join(
                f"<td>{player(info)}</td>"
                for info in (
                    row["reference_masked"],
                    mel["oracle_masked"],
                    svae["oracle_masked"],
                    row["reference_context"],
                )
            )
            + "</tr></table>"
        )
        parts.append(
            f"<table><tr><th>Seed</th><th>Original mel500k generated</th><th>Semantic-VAE S2c{svae_update // 1000}k generated</th></tr>"
        )
        for seed in SEEDS:
            parts.append(
                f"<tr><td>{seed}</td><td>{player(indexed['mel'][(key, seed)]['generated_masked'])}</td><td>{player(indexed['svae'][(key, seed)]['generated_masked'])}</td></tr>"
            )
        parts.append(
            "</table><details><summary>Original transcript (not model input)</summary><p>"
            + html.escape(row.get("text", ""))
            + "</p></details>"
        )
    (output / "listening.html").write_text("\n".join(parts), encoding="utf-8")


def write_markdown(output: Path, summary: dict) -> None:
    count = summary["generated"]["overall"]["utterances"]
    lines = [
        "# Native audio pretrained model comparison",
        "",
        f"{count} LibriSpeech dev utterances, one per speaker, 3 sampling seeds per model. No text or video conditioning.",
        f"Evaluation scope: {summary['scope']}.",
        "",
        f"Each utterance is equally weighted after averaging its 3 seed results. Delta = Semantic-VAE S2c{summary.get('svae_update', 70000) // 1000}k minus original mel500k. 95% intervals use utterance-paired, subset-stratified bootstrap.",
        "",
    ]
    for section in ("generated", "codec_controls"):
        lines.extend([f"## {section}", ""])
        for subset, group in summary[section].items():
            lines.extend(
                [
                    f"### {subset} (N={group['utterances']})",
                    "",
                    "| Metric | Paired N | Mel | Semantic-VAE | Delta | Delta 95% CI |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for name, item in group["metrics"].items():
                if not item["paired_utterances"]:
                    lines.append(f"| {name} | 0 | — | — | — | — |")
                    continue
                lo, hi = item["delta_ci95"]
                lines.append(
                    f"| {name} | {item['paired_utterances']} | {item['mel_mean']:.6f} | {item['svae_mean']:.6f} | {item['svae_minus_mel']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            *[f"- {warning}" for warning in WARNINGS],
            "",
            "Listen locally using [listening.html](listening.html). Paths are relative to this report; keep the run directory intact.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prefix = os.environ.get("ROOT_PREFIX", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-name", default="metrics")
    parser.add_argument("--canary-limit", type=int, help="Evaluate only matching *_canaryN branches as a smoke check")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svae-update", type=int, choices=tuple(SVAE_CHECKPOINTS), default=70000)
    parser.add_argument(
        "--speaker-model", type=Path, default=Path(f"{prefix}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth")
    )
    parser.add_argument(
        "--speaker-base", type=Path, default=Path(f"{prefix}/zjw524/alignDiT_pretrain_models/wavlm_large_s3prl.pt")
    )
    parser.add_argument(
        "--emotion-model",
        type=Path,
        help="Optional local emotion2vec_plus_large directory; reports embedding and legacy score cosine",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap samples")
    if args.canary_limit is not None and not 1 <= args.canary_limit < 50:
        parser.error("--canary-limit must be in [1,49]")
    if args.canary_limit is not None and "canary" not in args.output_name:
        parser.error("Canary output directory name must contain 'canary'")
    started = time.time()
    root = args.run_root.resolve(strict=True)
    common, branches, provenance, waves = load_run(root, args.canary_limit, args.svae_update)
    print(
        f"Validated {len(common)} shared utterances, {len(common) * 3} draws per branch, and all source/WAV hashes",
        flush=True,
    )
    if args.validate_only:
        return
    if Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        parser.error("--output-name must be one directory name")
    output = safe_join(root, args.output_name)
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(666)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    items = metric_audio_items(common, branches)
    embeddings, model_provenance = embedding_metrics(root, items, args)
    generated, oracles = measure_rows(root, common, branches, waves, embeddings)
    atomic_write_jsonl(output / "generated_metrics.jsonl", generated)
    atomic_write_jsonl(output / "codec_metrics.jsonl", oracles)
    bootstrap = {"bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed}
    summary = {
        "schema_version": 1,
        "svae_update": args.svae_update,
        "scope": "formal_50_utterance_pilot" if args.canary_limit is None else "canary_only_not_experimental_evidence",
        "run_root": str(root),
        "generated": summarize_paired(generated, generated=True, **bootstrap),
        "codec_controls": summarize_paired(oracles, generated=False, **bootstrap),
        "bootstrap": {
            **bootstrap,
            "unit": "utterance_after_seed_mean",
            "stratified_by": "subset",
            "delta": "svae_minus_mel",
        },
        "provenance": provenance,
        "metric_models": model_provenance,
        "warnings": WARNINGS,
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "soundfile", "pystoi")},
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
    }
    atomic_write_json(output / "summary.json", summary)
    write_markdown(output, summary)
    write_listening_page(output, root, common, branches, args.svae_update)
    atomic_write_json(
        output / "complete.json",
        {
            "schema_version": 1,
            "common_manifest_sha256": provenance["common_manifest_sha256"],
            "files": {
                name: sha256_file(output / name)
                for name in (
                    "summary.json",
                    "summary.md",
                    "listening.html",
                    "generated_metrics.jsonl",
                    "codec_metrics.jsonl",
                )
            },
        },
    )
    print(json.dumps({"output": str(output), "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
