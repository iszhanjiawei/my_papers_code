"""Run the repository's CelebV-Dub cache extractor with its A40 numerical reference.

The reference uses the original 16-kHz waveform, seed, FP32 posterior and
1,000,014-step codec EMA. Its bytes were reproduced independently using the
LibriTTS encoder and a SoundFile decode on an A40.
"""

import hashlib
import sys
from pathlib import Path

import soundfile as sf
import torch

# This snapshot uses the upstream cache writer, including its integrity checks
# and atomic resume protocol. The cache spec stores digests of both programs.
upstream = Path(__file__).resolve().parents[5] / "AlignDiT_mmdit_c2_semantic_vae" / "src"
sys.path.insert(0, str(upstream))
from aligndit.script.misc import extract_celebvdub_svae_latents as celeb


A40_LATENT_SHA256 = "2e6577fc48f3a283a0ee5e23c93372ca97ec4965a286ba44843eae2873846896"
WAV_SHA256 = "2b17a9600c396fad7c0ad7bc7808dfdf0caa810fd6ab7c1731bb87c2f85d8314"
PCM_SHA256 = "e72d28b4abe746276db8f35de523ce98ca741d60f1010b9a8d3145f4b97710bd"
upstream_build = celeb.build_and_verify_spec
upstream_validate = celeb.validate_stored_spec_resources


def reference_identity(dataset_root):
    path = dataset_root / celeb.GOLDEN_RECORD["audio_relative_path"]
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000 or pcm.shape != (378579, 1):
        raise ValueError(f"Unexpected reference waveform at {path}")
    if (celeb.base.sha256_file(path) != WAV_SHA256
            or hashlib.sha256(pcm.astype("<f4", copy=False).tobytes()).hexdigest() != PCM_SHA256):
        raise ValueError(f"A40 reference input differs from the verified recording: {path}")
    return {
        "device": "NVIDIA A40",
        "wav_sha256": WAV_SHA256,
        "pcm_f32le_sha256": PCM_SHA256,
        "posterior_seed": celeb.GOLDEN_POSTERIOR_SEED,
        "raw_latent_sha256": A40_LATENT_SHA256,
        "adapter_sha256": celeb.base.sha256_file(Path(__file__)),
    }


def build(args, context):
    if torch.cuda.get_device_name(context.local_rank) != "NVIDIA A40":
        raise RuntimeError("This extractor uses the NVIDIA A40 numerical reference")
    spec = upstream_build(args, context)
    spec["extraction"]["a40_reference"] = reference_identity(args.dataset_root)
    return spec


def validate(args, spec):
    upstream_validate(args, spec)
    if spec["extraction"].get("a40_reference") != reference_identity(args.dataset_root):
        raise RuntimeError("The saved A40 cache reference differs from this extractor")


if __name__ == "__main__":
    celeb.GOLDEN_RAW_LATENT_SHA256 = A40_LATENT_SHA256
    celeb.build_and_verify_spec = build
    celeb.validate_stored_spec_resources = validate
    celeb.main()
