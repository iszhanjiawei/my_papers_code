"""Setting 2 dubbing with supplied GT dialogue and an independent audio prompt.

Reuse the frozen VTS sampling implementation with explicit, audited input and
provenance substitutions. This entry never reads the target waveform. The
original VTS entry is unchanged and still requires visual-recognition text.
"""
from pathlib import Path
import hashlib


def main():
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
        ('"LipVoicer_visual_only_VSR; reference transcript from explicit reference"',
         '"supplied_target_GT_dialogue; reference transcript from explicit different utterance"'),
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
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
    exec(compile(source, str(original), "exec"), namespace)


if __name__ == "__main__":
    main()
