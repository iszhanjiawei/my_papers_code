"""Reuse the immutable native v2 fixture for S2c 50k/60k/70k longitudinal evaluation.

No model, mask, seed, normalization, context encoding or solver changes are
allowed. Existing 70k outputs stay untouched; a six-draw latent replay checks
that the extended generator reproduces them bitwise before new evaluations.
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np

from aligndit.script.eval.compare_native_audio_pretrains import (
    SEEDS,
    SVAE_CHECKPOINTS,
    read_object,
    validate_artifact,
)
from aligndit.script.eval.evaluate_native_audio_pretrains import load_run, summarize_paired
from aligndit.script.misc.svae_cache_utils import atomic_write_json, read_jsonl, sha256_file


UPDATES = (50000, 60000, 70000)
METRICS = (
    "spksim_vs_context",
    "spksim_vs_original",
    "emotion_embedding_cosine_vs_original",
    "legacy_emosim_scores_cosine_vs_original",
    "stoi_vs_original",
)


def completed_metrics(root: Path, update: int) -> tuple[dict, list[dict]]:
    """Require completed formal metrics, then validate their bound native WAVs."""
    marker = read_object(root / "metrics/complete.json")
    for name, digest in marker["files"].items():
        if Path(name).name != name or sha256_file(root / "metrics" / name) != digest:
            raise ValueError(f"Changed metric artifact: {root}/{name}")
    summary = read_object(root / "metrics/summary.json")
    if summary["scope"] != "formal_50_utterance_pilot" or summary.get("svae_update", 70000) != update:
        raise ValueError("Wrong checkpoint or canary metric report")
    _, _, provenance, _ = load_run(root, svae_update=update)
    if (
        provenance["common_manifest_sha256"] != marker["common_manifest_sha256"]
        or summary["provenance"]["svae"]["checkpoint"]["sha256"] != SVAE_CHECKPOINTS[update]
        or summary["provenance"]["svae"] != provenance["svae"]
        or summary["provenance"]["mel"] != provenance["mel"]
    ):
        raise ValueError("Metric report is not bound to these generated waveforms")
    rows = list(read_jsonl(root / "metrics/generated_metrics.jsonl"))
    recalculated = summarize_paired(rows, generated=True, bootstrap_samples=10000, bootstrap_seed=20260906)
    if recalculated != summary["generated"]:
        raise ValueError("Generated statistics do not reproduce from bound per-draw metrics")
    return summary, rows


def clone_fixtures(reference: Path, target: Path, *, replay: bool = False) -> None:
    target.mkdir(parents=True, exist_ok=False)
    names = ("common", "svae_context_canary2", "mel_canary2")
    if not replay:
        names += ("svae_context", "mel")
    for name in names:
        # Separate copies, never writable hardlinks/symlinks to prior results.
        shutil.copytree(reference / name, target / name)


def prepare(reference: Path, study: Path) -> None:
    if study.exists():
        raise FileExistsError(f"Refusing to overwrite longitudinal study: {study}")
    completed_metrics(reference, 70000)
    load_run(reference, canary_limit=2)
    study.mkdir(parents=True, exist_ok=False)
    for update in (50000, 60000):
        clone_fixtures(reference, study / f"s2c_{update // 1000}k")
    clone_fixtures(reference, study / "replay_70k", replay=True)
    atomic_write_json(
        study / "study_contract.json",
        {
            "reference_root": str(reference),
            "updates": list(UPDATES),
            "checkpoint_sha256": SVAE_CHECKPOINTS,
            "common_complete_sha256": sha256_file(reference / "common/complete.json"),
            "context_complete_sha256": sha256_file(reference / "svae_context/context_complete.json"),
            "reference_metrics_complete_sha256": sha256_file(reference / "metrics/complete.json"),
            "reference_generation_sha256": sha256_file(reference / "svae/generation_complete.json"),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )
    print(f"Prepared immutable paired fixtures at {study}", flush=True)


def check_study(reference: Path, study: Path) -> None:
    contract = read_object(study / "study_contract.json")
    expected = {
        "common_complete_sha256": reference / "common/complete.json",
        "context_complete_sha256": reference / "svae_context/context_complete.json",
        "reference_metrics_complete_sha256": reference / "metrics/complete.json",
        "reference_generation_sha256": reference / "svae/generation_complete.json",
    }
    if contract["reference_root"] != str(reference) or contract["updates"] != list(UPDATES):
        raise ValueError("Different source study supplied")
    for key, path in expected.items():
        if contract[key] != sha256_file(path):
            raise ValueError(f"Reference study changed: {key}")


def check_replay(reference: Path, study: Path) -> None:
    check_study(reference, study)
    replay = study / "replay_70k"
    manifests = []
    for root in (reference, replay):
        complete = read_object(root / "svae_latents_canary2/latent_generation_complete.json")
        if complete["count"] != 6 or complete["checkpoint"]["sha256"] != SVAE_CHECKPOINTS[70000]:
            raise ValueError("Replay must be exactly six 70k draws")
        rows = list(read_jsonl(validate_artifact(root, complete["generation_manifest"])))
        manifests.append({(r["utterance_key"], r["sampling_seed"]): r for r in rows})
    old, new = manifests
    if len(old) != 6 or old.keys() != new.keys():
        raise ValueError("Replay selection differs")
    for key, original in old.items():
        candidate = new[key]
        for field in ("ode_seed", "context_posterior_seed", "context_latent", "reference_input"):
            if candidate[field] != original[field]:
                raise ValueError(f"Replay condition changed: {key}/{field}")
        a = np.load(validate_artifact(reference, original["generated_latent"]), allow_pickle=False)
        b = np.load(validate_artifact(replay, candidate["generated_latent"]), allow_pickle=False)
        if not np.array_equal(a, b):
            raise ValueError(f"70k replay is not bitwise identical: {key}")
    atomic_write_json(study / "replay_verified.json", {"draws": 6, "bitwise_equal": True})
    print("70k replay passed: all six latent outputs are bitwise identical", flush=True)


def temporal_pair(earlier: list[dict], later: list[dict]) -> dict:
    rows = []
    for branch, values in (("mel", earlier), ("svae", later)):
        for row in values:
            if row["branch"] == "svae":
                rows.append({**row, "branch": branch})
    result = summarize_paired(rows, generated=True, bootstrap_samples=10000, bootstrap_seed=20260906)
    for group in result.values():
        for score in group["metrics"].values():
            score["earlier_mean"] = score.pop("mel_mean")
            score["later_mean"] = score.pop("svae_mean")
            score["later_minus_earlier"] = score.pop("svae_minus_mel")
    return result


def verify_reused_controls(reference: Path, root: Path, baseline: list[dict], rows: list[dict]) -> None:
    for file in ("common/complete.json", "svae_context/context_complete.json", "mel/generation_complete.json"):
        if sha256_file(reference / file) != sha256_file(root / file):
            raise ValueError(f"Reused fixture changed: {file}")
    key = lambda r: (r["utterance_key"], r["sampling_seed"])
    original_mel = {key(r): r for r in baseline if r["branch"] == "mel"}
    for row in rows:
        if row["branch"] == "mel":
            for field, value in original_mel[key(row)].items():
                actual = row[field]
                if isinstance(value, float):
                    if not np.isclose(value, actual, rtol=0, atol=1e-6):
                        raise ValueError(f"Metric model drift in fixed mel control: {key(row)}/{field}")
                elif actual != value:
                    raise ValueError("Fixed mel control metadata differs")
    original = read_object(reference / "svae/generation_complete.json")
    candidate = read_object(root / "svae/generation_complete.json")
    a = {key(r): r for r in read_jsonl(validate_artifact(reference, original["generation_manifest"]))}
    b = {key(r): r for r in read_jsonl(validate_artifact(root, candidate["generation_manifest"]))}
    for k, row in a.items():
        for field in ("context_latent", "context_posterior_seed", "ode_seed", "oracle_full", "oracle_masked"):
            if row[field] != b[k][field]:
                raise ValueError(f"Non-checkpoint variable changed: {k}/{field}")


def summarize(reference: Path, study: Path) -> None:
    check_study(reference, study)
    if read_object(study / "replay_verified.json") != {"draws": 6, "bitwise_equal": True}:
        raise ValueError("A successful unchanged 70k replay is required")
    output = study / "comparison"
    if output.exists():
        raise FileExistsError(output)
    roots = {u: reference if u == 70000 else study / f"s2c_{u // 1000}k" for u in UPDATES}
    summaries, rows = {}, {}
    for update, root in roots.items():
        summaries[update], rows[update] = completed_metrics(root, update)
    for update in (50000, 60000):
        verify_reused_controls(reference, roots[update], rows[70000], rows[update])
        if summaries[update]["metric_models"] != summaries[70000]["metric_models"]:
            raise ValueError("Different metric models or assets")
    pairs = {
        f"{a // 1000}k_to_{b // 1000}k": temporal_pair(rows[a], rows[b])
        for a, b in ((50000, 60000), (60000, 70000), (50000, 70000))
    }
    output.mkdir()
    overview = {
        str(u): {m: summaries[u]["generated"]["overall"]["metrics"][m]["svae_mean"] for m in METRICS} for u in UPDATES
    }
    atomic_write_json(
        output / "summary.json",
        {
            "updates": list(UPDATES),
            "utterances": 50,
            "seeds": list(SEEDS),
            "overview": overview,
            "paired_changes": pairs,
            "metric_summary_sha256": {str(u): sha256_file(r / "metrics/summary.json") for u, r in roots.items()},
            "study_contract_sha256": sha256_file(study / "study_contract.json"),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )
    lines = [
        "# S2c 50k / 60k / 70k: unchanged native waveform-masked generation protocol",
        "",
        "50 paired utterances, 3 seeds per checkpoint; 70k is reused. Average seeds within utterance before stratified paired bootstrap (10,000 replicates). No training was performed.",
        "",
        "| Metric ↑ | 50k | 60k | 70k |",
        "|---|---:|---:|---:|",
    ]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{overview[str(u)][m]:.6f}" for u in UPDATES) + " |")
    for name, groups in pairs.items():
        lines += ["", f"## {name}", "", "Delta = later minus earlier; 95% CI. No multiplicity correction.", ""]
        lines += ["| Metric | Delta | 95% CI |", "|---|---:|---:|"]
        for m in METRICS:
            s = groups["overall"]["metrics"][m]
            lo, hi = s["delta_ci95"]
            lines.append(f"| {m} | {s['later_minus_earlier']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += [
        "",
        "Only checkpoint changes. Common masks, posterior context, ODE seeds, EMA selection, FP32, Euler EPSS32, CFG, codec and metric assets are fixed. Six 70k replay latents match the original bitwise; copied mel control scores match within 1e-6; all codec oracle WAVs match exactly.",
        "",
        "This pilot measures no-text/no-video completion, not WER/AVSync. It does not prove that additional updates under a new learning-rate schedule will help. The old 70k schedule has nearly zero terminal learning rate. Conditions are encoded from zero-masked waveforms, unlike full-latent-then-mask training.",
        "",
        "[Listening page](listening.html)",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifests = {}
    for update, root in roots.items():
        c = read_object(root / "svae/generation_complete.json")
        manifests[update] = {
            (r["utterance_key"], r["sampling_seed"]): r
            for r in read_jsonl(validate_artifact(root, c["generation_manifest"]))
        }
    page = [
        "<!doctype html><meta charset='utf-8'><title>S2c checkpoint comparison</title>",
        "<h1>S2c 50k / 60k / 70k</h1><p>Named, non-blind listening; missing regions only. No human scores collected.</p>",
    ]
    for key in sorted({k[0] for k in manifests[70000]}):
        page.append(f"<h2>{html.escape(key)}</h2><table><tr><th>Seed</th><th>50k</th><th>60k</th><th>70k</th></tr>")
        for seed in SEEDS:
            page.append(f"<tr><td>{seed}</td>")
            for update in UPDATES:
                info = manifests[update][(key, seed)]["generated_masked"]
                rel = os.path.relpath(roots[update] / info["path"], output)
                page.append(
                    f'<td><audio controls preload="none" src="{html.escape(quote(rel, safe="/"))}"></audio></td>'
                )
            page.append("</tr>")
        page.append("</table>")
    (output / "listening.html").write_text("\n".join(page), encoding="utf-8")
    atomic_write_json(
        output / "complete.json",
        {"files": {n: sha256_file(output / n) for n in ("summary.json", "summary.md", "listening.html")}},
    )
    print(f"Completed longitudinal comparison: {output}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "check-replay", "summarize"), required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--study-root", type=Path, required=True)
    args = p.parse_args()
    reference, study = args.reference_root.resolve(strict=True), args.study_root.resolve()
    if reference == study or reference in study.parents or study in reference.parents:
        raise ValueError("Study and reference must be separate non-nested directories")
    {"prepare": prepare, "check-replay": check_replay, "summarize": summarize}[args.mode](reference, study)


if __name__ == "__main__":
    main()
