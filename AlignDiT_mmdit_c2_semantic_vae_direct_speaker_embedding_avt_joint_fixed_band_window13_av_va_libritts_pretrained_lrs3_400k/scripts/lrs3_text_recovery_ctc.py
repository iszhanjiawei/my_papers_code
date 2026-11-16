#!/usr/bin/env python3
"""Independent local wav2vec2 CTC checks for LRS3 transcript recovery.

This stage produces evidence and explicitly unapproved correction candidates.
Its agreement counters never silently convert ASR-only text into source labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

from lrs3_text_recovery import atomic_json, atomic_jsonl, digest, edit_distance, read_bound_audio, read_jsonl, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--mfa-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--controls", type=int, default=32)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
    torch.set_num_threads(4)
    rows = {r["sample_id"]: r for r in read_jsonl(args.manifest)}
    whisper, current_labels = {}, {}
    for path in sorted((args.evidence_dir / "groups").glob("*.json")):
        group = json.loads(path.read_text())
        if group.get("status") != "complete":
            continue
        whisper.update({r["sample_id"]: r for r in group["segment_asr"]})
        current_labels.update({r["sample_id"]: r for r in group["labels"]})
    controls = []
    for path in sorted((args.evidence_dir / "controls").glob("*.json"))[:args.controls]:
        control = json.loads(path.read_text())
        if control.get("status") == "complete":
            whisper[control["sample_id"]] = control["asr"]
            controls.append(control["sample_id"])
    mfa = {r["sample_id"]: r for r in read_jsonl(args.mfa_dir / "boundary_diagnostics.jsonl")}
    wanted = sorted(set(current_labels) | set(controls))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "segments").mkdir(exist_ok=True)
    identity = json.loads((args.model / "download_info.json").read_text())
    protocol = {"model_identity": identity, "physical_gpu": args.gpu, "compute_dtype": "float16", "decoder": "greedy CTC; no language model or text prompts",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "samples": wanted, "controls": controls}
    atomic_json(args.output / "run.json", protocol)
    config = Wav2Vec2Config.from_pretrained(str(args.model), local_files_only=True)
    # The released inference checkpoint omits the training-only SpecAugment
    # embedding. Disable masking before construction instead of silently
    # accepting arbitrary missing state keys. Evaluation already never masks.
    config.mask_time_prob = 0.0
    config.mask_feature_prob = 0.0
    config.apply_spec_augment = False
    processor = Wav2Vec2Processor.from_pretrained(str(args.model), local_files_only=True)
    model = Wav2Vec2ForCTC(config)
    state = torch.load(args.model / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device="cuda", dtype=torch.float16).eval()
    blank = model.config.pad_token_id
    vocabulary = {value: key for key, value in processor.tokenizer.get_vocab().items()}
    delimiter = processor.tokenizer.word_delimiter_token_id
    stride = math.prod(model.config.conv_stride)
    results = []
    for index, sample_id in enumerate(wanted):
        row = rows[sample_id]
        path = args.output / "segments" / (digest(sample_id)[:24] + ".json")
        fingerprint = digest({"sample_id": sample_id, "audio_path": row["audio_path"], "model": identity,
                              "audio_sha256": hashlib.sha256(Path(row["audio_path"]).read_bytes()).hexdigest(),
                              "whisper_text": whisper[sample_id]["text"], "mfa_source_text": mfa.get(sample_id, {}).get("mfa_source_assigned_text"),
                              "normalization": "waveform mean/std, greedy CTC"})
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") == fingerprint:
                results.append(saved)
                continue
        started = time.time()
        audio, sr, source_identity = read_bound_audio(row["audio_path"])
        if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid CTC input: {sample_id}")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        values = inputs.input_values.to(device="cuda", dtype=torch.float16)
        attention_mask = inputs.attention_mask.to("cuda") if "attention_mask" in inputs else None
        with torch.inference_mode():
            logits = model(values, attention_mask=attention_mask).logits[0].float()
            probabilities = torch.softmax(logits, dim=-1)
            ids = probabilities.argmax(dim=-1)
            chosen = probabilities.gather(-1, ids[:, None])[:, 0]
            ids_np, scores = ids.cpu().numpy(), chosen.cpu().numpy()
        text = processor.batch_decode(ids_np[None, :])[0].strip().lower()
        emissions = []
        offset = 0
        for token, frames in itertools.groupby(ids_np.tolist()):
            length = len(list(frames))
            if token != blank:
                emissions.append({"token_id": token, "token": vocabulary[token], "start_frame": offset,
                                  "end_frame": offset + length, "probability": float(np.max(scores[offset:offset + length]))})
            offset += length
        lexical = [e for e in emissions if e["token_id"] != delimiter]
        words, current = [], []
        for emission in emissions + [{"token_id": delimiter}]:
            if emission["token_id"] == delimiter:
                if current:
                    words.append({"text": "".join(e["token"] for e in current).lower(),
                        "start": current[0]["start_frame"] * stride / sr,
                        "end": min(len(audio) / sr, current[-1]["end_frame"] * stride / sr),
                        "min_character_probability": min(e["probability"] for e in current),
                        "mean_character_probability": float(np.mean([e["probability"] for e in current]))})
                    current = []
            else:
                current.append(emission)
        ctc_words = tokens(text)
        whisper_words = tokens(whisper[sample_id]["text"])
        mfa_words = tokens(mfa.get(sample_id, {}).get("mfa_source_assigned_text", ""))
        reference = tokens(row.get("text", "")) if sample_id in controls else None
        result = {"sample_id": sample_id, "original_clip_id": row["original_clip_id"], "fingerprint": fingerprint,
            "audio_source_binding": source_identity,
            "ctc_text": text, "ctc_words": words, "whisper_text": whisper[sample_id]["text"],
            "mfa_source_text": mfa.get(sample_id, {}).get("mfa_source_assigned_text"),
            "ctc_whisper_agree": ctc_words == whisper_words,
            "ctc_mfa_source_agree": bool(mfa_words) and ctc_words == mfa_words,
            "min_character_probability": min((e["probability"] for e in lexical), default=None),
            "mean_character_probability": float(np.mean([e["probability"] for e in lexical])) if lexical else None,
            "current_quality": current_labels.get(sample_id, {}).get("quality", "known_label_control"),
            "reference_text": row.get("text") if reference is not None else None,
            "reference_word_count": len(reference) if reference is not None else None,
            "reference_edit_distance": edit_distance(reference, ctc_words) if reference is not None else None,
            "reference_exact": reference == ctc_words if reference is not None else None,
            "duration": len(audio) / sr, "elapsed_seconds": time.time() - started,
            "training_approved": False, "status": "independent_asr_evidence_only"}
        atomic_json(path, result)
        results.append(result)
        print(json.dumps({"sample_id": sample_id, "index": index + 1, "total": len(wanted),
                          "ctc_whisper_agree": result["ctc_whisper_agree"], "ctc_mfa_source_agree": result["ctc_mfa_source_agree"]}), flush=True)
    atomic_jsonl(args.output / "results.jsonl", results)
    known = [r for r in results if r["reference_text"] is not None]
    unknown = [r for r in results if r["reference_text"] is None]
    report = {"samples": len(results), "known_controls": len(known),
        "controls_ctc_exact": sum(r["reference_exact"] for r in known),
        "controls_ctc_wer": sum(r["reference_edit_distance"] for r in known) / max(1, sum(r["reference_word_count"] for r in known)),
        "controls_dual_asr_agree": sum(r["ctc_whisper_agree"] for r in known),
        "controls_dual_asr_agree_but_reference_differs": sum(r["ctc_whisper_agree"] and not r["reference_exact"] for r in known),
        "recovery_segments": len(unknown), "recovery_dual_asr_agree": sum(r["ctc_whisper_agree"] for r in unknown),
        "recovery_ctc_mfa_source_agree": sum(r["ctc_mfa_source_agree"] for r in unknown),
        "previously_review_ctc_mfa_source_agree": sum(r["ctc_mfa_source_agree"] and r["current_quality"] == "review_required" for r in unknown),
        "labels_automatically_approved": 0,
        "limitations": ["Greedy wav2vec2 decoding is an independent check, not ground truth.", "Agreement of two recognizers can still share errors.", "Reference disagreements may reflect ASR errors, source transcription errors, or crop-boundary label differences; listening is needed.", "CTC character probabilities are not calibrated transcript accuracy."]}
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
