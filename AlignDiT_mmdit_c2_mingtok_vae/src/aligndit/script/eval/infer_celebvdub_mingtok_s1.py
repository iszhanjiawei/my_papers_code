"""CelebV-Dub Setting-1 inference for the strict C2 MingTok experiment."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aligndit.model.cfm_mingtok import CFM_MingTok
from aligndit.model.dataset import cut_or_pad
from aligndit.model.mingtok_codec import (
    MINGTOK_HOP_SIZE,
    MINGTOK_LATENT_DIM,
    MINGTOK_LATENT_FPS,
    MINGTOK_SAMPLE_RATE,
    MingTokAcousticCodec,
    stable_sample_seed,
)
from aligndit.script.eval.utils import get_celebvdub_test_metainfo_s1
from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model.utils import get_tokenizer


AUDIO_VIDEO_RATIO = 2


def _historical_setting1_text(prompt_text: str, target_text: str) -> str:
    """Reproduce the original C2 Setting-1 prompt concatenation exactly."""

    if not prompt_text:
        raise ValueError("CelebV-Dub Setting-1 reference text must not be empty")
    if len(prompt_text[-1].encode("utf-8")) == 1:
        prompt_text += " "
    return prompt_text + target_text


def _load_video(path: Path, expected_dim: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"AV-HuBERT feature not found: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != expected_dim or array.shape[0] <= 0:
        raise RuntimeError(f"Invalid video feature shape {array.shape}, expected [T,{expected_dim}]: {path}")
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError(f"Non-finite video feature: {path}")
    return torch.from_numpy(array.astype(np.float32, copy=False))


def _load_mono_16k(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[-1] <= 5_000:
        raise RuntimeError(f"Invalid or empty reference waveform {tuple(waveform.shape)}: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != MINGTOK_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, MINGTOK_SAMPLE_RATE)
    return waveform.to(dtype=torch.float32).contiguous()


def _validate_config(config: DictConfig) -> None:
    arch = config.model.arch
    required_arch = {
        "depth": 18,
        "n_mm_layers": 12,
        "n_text_layers": 12,
        "audio_video_ratio": AUDIO_VIDEO_RATIO,
    }
    for key, expected in required_arch.items():
        actual = int(arch[key])
        if actual != expected:
            raise RuntimeError(f"MingTok C2 inference requires {key}={expected}, got {actual}")
    if bool(arch.prompt_isolated_ca):
        raise RuntimeError("MingTok C2 inference requires the original global text context")
    if list(arch.ctc_sampling_ratios) != [1, 1]:
        raise RuntimeError("MingTok C2 inference requires 50-Hz CTC strides [1,1]")
    if int(config.model.latent.dim) != MINGTOK_LATENT_DIM or int(config.model.latent.fps) != MINGTOK_LATENT_FPS:
        raise RuntimeError("The selected config is not the raw 64-D/50-Hz MingTok representation")


def _tokenizer_from_config(config: DictConfig) -> tuple[dict[str, int] | None, int, Path]:
    data_dir = Path(str(config.datasets.data_dir)).expanduser().resolve()
    tokenizer = str(config.model.tokenizer)
    dataset_name = str(config.datasets.name)
    if tokenizer in {"char", "pinyin"}:
        vocab_path = data_dir / f"{dataset_name}_{tokenizer}" / "vocab.txt"
        vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    elif tokenizer == "custom":
        vocab_path = Path(str(config.model.tokenizer_path)).expanduser().resolve()
        vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    elif tokenizer == "byte":
        vocab_path = data_dir / f"{dataset_name}_byte" / "vocab.txt"
        vocab_char_map, vocab_size = get_tokenizer(dataset_name, "byte")
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}")
    if tokenizer != "byte":
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Tokenizer vocabulary not found: {vocab_path}")
        if vocab_char_map is None or vocab_char_map.get(" ") != 0:
            raise RuntimeError(f"Tokenizer space token must have index 0: {vocab_path}")
    return vocab_char_map, vocab_size, vocab_path


def _build_model(
    config: DictConfig,
    checkpoint_path: Path,
    device: torch.device,
    ode_method: str,
) -> tuple[CFM_MingTok, Path]:
    _validate_config(config)
    vocab_char_map, vocab_size, vocab_path = _tokenizer_from_config(config)
    model_cls = get_class(f"aligndit.model.{config.model.backbone}")
    model = CFM_MingTok(
        transformer=model_cls(
            **config.model.arch,
            text_num_embeds=vocab_size,
            mel_dim=MINGTOK_LATENT_DIM,
        ),
        num_channels=MINGTOK_LATENT_DIM,
        audio_video_ratio=AUDIO_VIDEO_RATIO,
        vocab_char_map=vocab_char_map,
        ctc_lambda=float(config.model.ctc_lambda),
        odeint_kwargs={"method": ode_method},
    )
    # The original C2 inference path uses EMA weights and a float32 AlignDiT.
    model = load_checkpoint(
        model,
        str(checkpoint_path),
        str(device),
        dtype=torch.float32,
        use_ema=True,
    )
    return model.eval().requires_grad_(False), vocab_path


def _prepare_prompt(
    record: tuple[str, str, str, str, str],
    codec: MingTokAcousticCodec,
    dataset_root: Path,
    video_dim: int,
    posterior_seed: int,
) -> tuple[str, torch.Tensor, int, int, str, torch.Tensor]:
    utterance, prompt_text, prompt_wav_string, target_text, target_wav_string = record
    # Preserve the logical path below CelebVDub/audio.  These wav files may be
    # symlinks whose targets live outside dataset_root.
    prompt_wav = Path(prompt_wav_string).expanduser().absolute()
    target_wav = Path(target_wav_string).expanduser().absolute()
    if not prompt_wav.is_file():
        raise FileNotFoundError(f"Reference waveform not found: {prompt_wav}")
    if not target_wav.is_file():
        raise FileNotFoundError(f"Target waveform not found: {target_wav}")

    audio_root = (dataset_root / "audio").absolute()
    try:
        prompt_relative = prompt_wav.relative_to(audio_root)
        target_relative = target_wav.relative_to(audio_root)
    except ValueError as error:
        raise RuntimeError(f"Setting-1 audio must be below {audio_root}: {prompt_wav}/{target_wav}") from error

    ref_video_path = dataset_root / "avhubert_video_feat" / prompt_relative.with_suffix(".npy")
    target_video_path = dataset_root / "avhubert_video_feat" / target_relative.with_suffix(".npy")
    ref_video = _load_video(ref_video_path, video_dim)
    target_video = _load_video(target_video_path, video_dim)

    waveform = _load_mono_16k(prompt_wav)
    num_samples = int(waveform.shape[-1])
    sample_seed = stable_sample_seed(prompt_relative.as_posix(), base_seed=posterior_seed)
    encoded, encoded_lengths = codec.encode(
        waveform,
        torch.tensor([num_samples], dtype=torch.long),
        mode="sample",
        seeds=[sample_seed],
    )
    encoded_frames = int(encoded_lengths[0].item())
    encoded = encoded[0, :encoded_frames].float()

    # Training targets are raw sampled latents.  There is deliberately no
    # channel-wise mean/std, global scaling, or prompt RMS normalization here.
    ref_frames = int(ref_video.shape[0]) * AUDIO_VIDEO_RATIO
    prompt_latent = cut_or_pad(encoded, ref_frames, dim=0, mode="replicate").unsqueeze(0)
    total_video = torch.cat((torch.zeros_like(ref_video), target_video), dim=0).unsqueeze(0)
    total_frames = int(total_video.shape[1]) * AUDIO_VIDEO_RATIO
    final_text = _historical_setting1_text(prompt_text, target_text)
    return utterance, prompt_latent, ref_frames, total_frames, final_text, total_video


def _default_output_dir(args: argparse.Namespace, project_root: Path) -> Path:
    modality = "avt" if args.ignore_modality is None else f"no-{args.ignore_modality}"
    epss = "epss" if not args.disable_epss else "uniform"
    sampling = (
        f"seed{args.seed}_{args.ode_method}_nfe{args.nfe}_{epss}"
        f"_cfgt{args.cfg_text}_cfgv{args.cfg_video}_{modality}_gt-dur"
    )
    if args.max_items is not None:
        sampling += f"_first{args.max_items}"
    return project_root / "results" / f"{args.config.stem}_{args.checkpoint.stem}" / "celebvdub_test_s1" / sampling


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MingTok Setting-1 inference requires CUDA")
    if args.nfe <= 0:
        raise ValueError("--nfe must be positive")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("--max-items must be positive")

    accelerator = Accelerator()
    device = accelerator.device
    torch.cuda.set_device(device)

    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config = OmegaConf.load(config_path)
    data_dir = Path(str(config.datasets.data_dir)).expanduser().resolve(strict=True)
    dataset_root = data_dir / str(config.datasets.name)
    test_list = (args.test_list or data_dir / "celebvdub_test_s1.lst").expanduser().resolve(strict=True)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CelebV-Dub root not found: {dataset_root}")

    metainfo = get_celebvdub_test_metainfo_s1(str(test_list), str(dataset_root))
    if not metainfo:
        raise RuntimeError(f"No valid CelebV-Dub Setting-1 records found in {test_list}")
    if args.max_items is not None:
        if args.max_items > len(metainfo):
            raise ValueError(f"--max-items={args.max_items} exceeds the {len(metainfo)} valid records")
        metainfo = metainfo[: args.max_items]
    # get_inference_prompt_vt in the original C2 applies this exact shuffle.
    random.Random(666).shuffle(metainfo)

    project_root = Path(__file__).resolve().parents[4]
    output_dir = (args.output_dir or _default_output_dir(args, project_root)).expanduser().resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    model, vocab_path = _build_model(config, checkpoint_path, device, args.ode_method)
    codec_repo = Path(args.mingtok_repo or str(config.model.codec.repo_path)).expanduser().resolve(strict=True)
    codec_checkpoint = (
        Path(args.mingtok_checkpoint or str(config.model.codec.checkpoint_dir)).expanduser().resolve(strict=True)
    )
    codec = MingTokAcousticCodec(
        repo_path=codec_repo,
        checkpoint_dir=codec_checkpoint,
        device=device,
        dtype=args.codec_dtype,
        backend=args.codec_backend,
        load_encoder=True,
        load_decoder=True,
    )

    if accelerator.is_main_process:
        print(f"checkpoint (EMA): {checkpoint_path}")
        print(f"tokenizer: {vocab_path}")
        print(f"MingTok: raw sampled 64D/50Hz, posterior seed={args.posterior_seed}, no normalization")
        print(f"output: {output_dir}")

    started_at = time.time()
    local_count = 0
    with accelerator.split_between_processes(metainfo) as local_records:
        iterator = tqdm(
            local_records,
            desc=f"MingTok S1 rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
        for record in iterator:
            utterance, cond, ref_frames, total_frames, final_text, total_video = _prepare_prompt(
                record,
                codec,
                dataset_root,
                int(config.model.arch.video_dim),
                args.posterior_seed,
            )
            cond = cond.to(device=device, dtype=torch.float32)
            total_video = total_video.to(device=device, dtype=torch.float32)
            lens = torch.tensor([ref_frames], device=device, dtype=torch.long)
            duration = torch.tensor([total_frames], device=device, dtype=torch.long)

            generated, trajectory = model.sample(
                cond=cond,
                text=[final_text],
                duration=duration,
                video=total_video,
                lens=lens,
                steps=args.nfe,
                cfg_strength=args.cfg_text,
                cfg_strength_v=args.cfg_video,
                sway_sampling_coef=args.sway,
                seed=args.seed,
                use_epss=not args.disable_epss,
                ignore_modality=args.ignore_modality,
            )
            del trajectory
            if generated.ndim != 3 or generated.shape[0] != 1 or generated.shape[-1] != MINGTOK_LATENT_DIM:
                raise RuntimeError(f"Invalid generated latent shape for {utterance}: {tuple(generated.shape)}")
            if generated.shape[1] < total_frames:
                raise RuntimeError(
                    f"Generated latent is shorter than GT duration for {utterance}: "
                    f"{generated.shape[1]} < {total_frames}"
                )
            full_latent = generated[:, :total_frames]
            if not bool(torch.isfinite(full_latent).all()):
                raise FloatingPointError(f"Non-finite generated latent for {utterance}")

            # Match original C2 inference: remove the reference frames first,
            # then pass only the generated target representation to the decoder.
            target_latent = full_latent[:, ref_frames:total_frames]
            waveform = codec.decode(target_latent)[0].float().cpu()
            expected_samples = (total_frames - ref_frames) * MINGTOK_HOP_SIZE
            if tuple(waveform.shape) != (1, expected_samples) or not bool(torch.isfinite(waveform).all()):
                raise RuntimeError(
                    f"Invalid decoded waveform for {utterance}: {tuple(waveform.shape)}, "
                    f"expected (1,{expected_samples})"
                )

            save_path = output_dir / f"{utterance}.wav"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(save_path),
                waveform,
                MINGTOK_SAMPLE_RATE,
            )
            local_count += 1

    accelerator.wait_for_everyone()
    processed = torch.tensor([local_count], device=device, dtype=torch.long)
    processed = accelerator.reduce(processed, reduction="sum")
    if accelerator.is_main_process:
        elapsed_minutes = (time.time() - started_at) / 60
        print(f"Done: generated {int(processed.item())} waveforms in {elapsed_minutes:.2f} minutes.")


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="AlignDiT training checkpoint; EMA is loaded.")
    parser.add_argument(
        "--config",
        type=Path,
        default=package_root / "config/train_celebvdub_mingtok_c2.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--mingtok-repo", type=Path)
    parser.add_argument("--mingtok-checkpoint", type=Path)
    parser.add_argument("--codec-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--codec-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--posterior-seed", type=int, default=666)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--ode-method", default="euler")
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--cfg-text", type=float, default=5.0)
    parser.add_argument("--cfg-video", type=float, default=2.0)
    parser.add_argument("--ignore-modality", choices=("text", "video"))
    parser.add_argument("--disable-epss", action="store_true")
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
