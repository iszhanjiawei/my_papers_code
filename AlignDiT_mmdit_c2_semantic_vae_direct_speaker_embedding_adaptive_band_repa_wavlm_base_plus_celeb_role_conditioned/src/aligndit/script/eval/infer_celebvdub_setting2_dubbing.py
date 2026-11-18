"""Setting 2 dubbing with supplied GT dialogue and an independent audio prompt.

Reuse the frozen VTS sampling implementation with explicit, audited input and
provenance substitutions. This entry never reads the target waveform. The
original VTS entry is unchanged and still requires visual-recognition text.
"""
from pathlib import Path
import hashlib


def select_model_text(reference_text, target_text, mode):
    """Change only the text condition; retain the independent audio prompt."""
    if mode == "target-only":
        return target_text
    if mode == "reference-target":
        from aligndit.script.eval.infer_celebvdub_grid_reference import reference_target_text
        return reference_target_text(reference_text, target_text)
    raise ValueError(f"Unknown text input mode: {mode}")


def build_source():
    original = Path(__file__).with_name("infer_celebvdub_setting2.py")
    expected = "895e471d00fcbba631f5527058dcd0c38dbf9f0ec0324331cc7bc8a91528229e"
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Shared inference implementation changed; audit the dubbing adapter before use")
    source = original.read_text()
    substitutions = [
        ('"""Leak-free Setting 2 inference for the Semantic-VAE / CAM++ / REPA model.\n\nTarget inputs are an explicit video-only feature and an independently cached\nvisual-recognition transcript. Reference VAE and CAM++ features are recomputed\nfrom the exact different-utterance waveform. No target waveform, acoustic cache,\ntranscript, or historical S1 manifest is opened.\n"""',
         '"""Setting2-GTText-Dubbing: target video, supplied real dialogue, cross-utterance reference. No target waveform or target acoustic cache is read."""'),
        ('if field == "video_feature" and "avhubert_video_feat" not in path.parts:\n                raise ValueError(f"Require audited video-only features: {path}")',
         'if field == "video_feature":\n                package = args.manifest.parent\n                inventory = json.loads((package / "checksums.json").read_text())["files"]\n                relative = path.relative_to(package).as_posix()\n                if not relative.startswith("features/video/") or inventory[relative]["sha256"] != sha256_file(path):\n                    raise ValueError(f"Require verified video-only package feature: {path}")'),
        ('"target_gt_audio_or_text_read": False, "target_acoustic_cache_read": False,',
         '"task": "Setting2-GTText-Dubbing", "target_gt_audio_or_text_read": True, "target_gt_text_read": True, "target_gt_audio_read": False, "target_acoustic_cache_read": False, "vsr_used": False,'),
        ('"source_sha256": source_hashes(project)}',
         '"source_sha256": source_hashes(project), "shared_implementation_sha256": "' + expected + '"}'),
        ('paths += [Path(__file__), Path(__file__).with_name("infer_celebvdub_grid_reference.py"),',
         'paths += [Path(__file__), Path(__file__).with_name("infer_celebvdub_setting2.py"), Path(__file__).with_name("infer_celebvdub_grid_reference.py"),'),
        ('default=bench / "setting2/inference.jsonl"',
         'default=root / "alignDiT_idea6/my_papers_code/celebvdub_setting2/inference.jsonl"'),
        ('default=bench / "cache/lipvoicer_vsr_text"',
         'default=root / "alignDiT_idea6/my_papers_code/celebvdub_setting2/text"'),
        ('default=bench / "results/Ours_150k"',
         'default=bench / "results/setting2_gttext_dubbing/Ours_150k"'),
        ('default=bench / "cache/ours_svae_campplus_setting2"',
         'default=bench / "cache/ours_svae_campplus_setting2_gttext_dubbing"'),
        ('parser.add_argument("--vsr-text-dir",',
         'parser.add_argument("--text-input-mode", choices=["reference-target", "target-only"], default="reference-target", help="Text passed to model.sample; audio prompt and speaker conditioning are unchanged")\n    parser.add_argument("--vsr-text-dir",'),
        ('text = reference_target_text(row["reference_text"], row["vsr_text"])',
         'text = select_model_text(row["reference_text"], row["vsr_text"], args.text_input_mode)'),
        ('"text_source": "LipVoicer_visual_only_VSR; reference transcript from explicit reference",',
         '"text_source": ("supplied_target_GT_dialogue_only" if args.text_input_mode == "target-only" else "supplied_target_GT_dialogue; reference transcript from explicit different utterance"),\n              "text_input_mode": args.text_input_mode, "reference_text_conditioned": args.text_input_mode == "reference-target",'),
        ('"generation_seconds": time.monotonic()-started}',
         '"generation_seconds": time.monotonic()-started, "model_text": text, "text_input_mode": args.text_input_mode}'),
    ]
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError(f"Expected exactly one audited substitution: {before[:100]}")
        source = source.replace(before, after)
    source = source.replace("vsr_text", "target_text")
    source = source.replace("--vsr-text-dir", "--target-text-dir")
    source = source.replace("reference or VSR transcript", "reference or supplied GT transcript")
    source = source.replace("and visual-only transcripts", "and supplied GT dialogue")
    source = source.replace('f"Ours_{args.step // 1000}k"', 'f"Ours_GTText_Dubbing_{args.step // 1000}k"')
    return source


def main():
    original = Path(__file__).with_name("infer_celebvdub_setting2.py")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__",
                 "select_model_text": select_model_text}
    exec(compile(build_source(), str(original), "exec"), namespace)


if __name__ == "__main__":
    main()
