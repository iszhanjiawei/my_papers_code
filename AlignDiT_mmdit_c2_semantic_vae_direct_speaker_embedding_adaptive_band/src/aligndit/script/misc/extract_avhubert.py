import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from python_speech_features import logfbank
from scipy.io import wavfile


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_avhubert_feature")


def select_audio_region(wav_data, region):
    """Select a metric window in memory without rewriting the complete WAV."""
    start, count = region
    if start < 0 or count <= 0 or start + count > len(wav_data):
        raise ValueError(f"Invalid AVSync audio window {region} for {len(wav_data)} samples")
    return wav_data[start : start + count]


def load_audio_regions(summary_path, audio_root):
    summary_path = Path(summary_path).resolve(strict=True)
    root = Path(audio_root).resolve(strict=True).parent
    if summary_path.parent != root:
        raise ValueError("Audio-region summary must belong to the generated WAV root")
    summary = json.loads(summary_path.read_text())
    if summary["generation"].get("output_region") != "raw-full":
        raise ValueError("Audio-region summary requires raw-full inference")
    regions = {}
    for row in summary["outputs"]:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "test":
            raise ValueError("Invalid generated audio path")
        path = str((root / relative).resolve(strict=True))
        if path in regions:
            raise ValueError("Duplicate generated audio in region summary")
        region = (int(row["target_audio_start_samples"]), int(row["target_audio_num_samples"]))
        if region[0] != int(row["prompt_frames"]) * 400 or region[0] + region[1] > int(row["samples"]):
            raise ValueError("Target audio window differs from the original inference timeline")
        regions[path] = region
    return regions


class AVHubertFeatureReader(object):
    def __init__(self, ckpt_path, max_chunk=1600000, custom_utils=None, audio_regions=None):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.max_chunk = max_chunk
        self.stack_order_audio = self.task.cfg.stack_order_audio
        image_crop_size, image_mean, image_std = (
            self.task.cfg.image_crop_size,
            self.task.cfg.image_mean,
            self.task.cfg.image_std,
        )
        self.transform = custom_utils.Compose(
            [
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std),
            ]
        )

        self.custom_utils = custom_utils
        self.audio_regions = audio_regions
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f"Transform: {self.transform}")

    def load_feature(self, mix_name, ref_len=None):
        def stacker(feats, stack_order):
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        video_fn, audio_fn = mix_name
        video_feats = self.load_image(video_fn)

        sample_rate, wav_data = wavfile.read(audio_fn)
        if self.audio_regions is not None:
            if sample_rate != 16_000 or wav_data.ndim != 1:
                raise ValueError("Raw-full AVSync windows require mono 16-kHz WAV")
            wav_data = select_audio_region(wav_data, self.audio_regions[str(Path(audio_fn).resolve(strict=True))])
        if len(wav_data.shape) > 1:
            wav_data = wav_data.mean(axis=1)
        if sample_rate != 16_000:
            import librosa

            wav_data = librosa.resample(
                wav_data.astype(np.float32), orig_sr=sample_rate, target_sr=16_000
            )
            sample_rate = 16_000
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
        audio_feats = stacker(audio_feats, self.stack_order_audio)

        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate(
                [audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)]
            )
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_image(self, audio_name):
        feats = self.custom_utils.load_video(audio_name)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def get_feats(self, path, ref_len=None):
        video_feats, audio_feats = self.load_feature(path, ref_len)
        with torch.no_grad():
            audio_feats, video_feats = (
                torch.from_numpy(audio_feats.astype(np.float32)).cuda(),
                torch.from_numpy(video_feats.astype(np.float32)).cuda(),
            )
            if self.task.cfg.normalize:
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
            video_feats = video_feats.unsqueeze(dim=0).permute((0, 4, 1, 2, 3)).contiguous()
            audio_feats = audio_feats.unsqueeze(dim=0).transpose(1, 2)
            source = {"audio": audio_feats, "video": video_feats}
            feat, _ = self.model.extract_finetune(
                source=source,
                padding_mask=None,
                mask=False,
            )
            return feat.squeeze(dim=0)


def get_path_iterator(a_root, v_root, nshard, rank, a_file_extension, v_file_extension):
    lines = sorted(glob.glob(f"{v_root}/**/*{v_file_extension}", recursive=True))

    tot = len(lines)
    shard_size = math.ceil(tot / nshard)
    start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
    assert start < end, "start={start}, end={end}"
    logger.info(f"rank {rank} of {nshard}, process {end - start} ({start}-{end}) out of {tot}")

    lines = lines[start:end]

    def iterate():
        for line in lines:
            video_path = line
            audio_path = line.replace(v_root, a_root).replace(v_file_extension, a_file_extension)
            yield (video_path, audio_path)

    return iterate, len(lines)


def main(args, custom_utils=None):
    nshard = args.nshard
    rank = args.rank
    a_input_dir = args.a_input_dir
    v_input_dir = args.v_input_dir
    output_dir = args.output_dir
    a_file_extension = args.a_file_extension
    v_file_extension = args.v_file_extension

    ckpt_path = args.ckpt_path
    max_chunk = args.max_chunk
    audio_regions = load_audio_regions(args.audio_region_summary, a_input_dir) if args.audio_region_summary else None
    reader = AVHubertFeatureReader(ckpt_path, max_chunk, custom_utils=custom_utils, audio_regions=audio_regions)

    generator, num = get_path_iterator(a_input_dir, v_input_dir, nshard, rank, a_file_extension, v_file_extension)
    iterator = generator()

    processed = []
    for path in tqdm.tqdm(iterator, total=num):
        try:
            feature = reader.get_feats(path)
        except Exception:
            if audio_regions is not None:
                raise  # An incomplete diagnostic must not silently drop clips.
            print(f"error in processing {path}")
            continue

        video_path, audio_path = path
        assert v_input_dir in video_path
        save_path = video_path.replace(v_input_dir, output_dir)
        assert save_path != video_path
        save_path = os.path.splitext(save_path)[0] + ".npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.save(save_path, feature.detach().cpu().numpy())
        if audio_regions is not None:
            processed.append({
                "audio": str(Path(audio_path).resolve()),
                "video": str(Path(video_path).resolve()),
                "feature": str(Path(save_path).resolve()),
                "audio_start_samples": audio_regions[str(Path(audio_path).resolve())][0],
                "audio_num_samples": audio_regions[str(Path(audio_path).resolve())][1],
            })
        del feature

    if audio_regions is not None:
        if len(processed) != num:
            raise RuntimeError("AVSync video coverage differs from the raw-full audio summary")
        (Path(output_dir).parent / "audio_region_protocol.json").write_text(json.dumps({
            "metric": "AVSync target-window diagnostic; full WAV remains unchanged",
            "inference_summary": str(Path(args.audio_region_summary).resolve()),
            "count": len(processed), "bindings": processed,
        }, indent=2) + "\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--v-input-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/video",
    )
    parser.add_argument(
        "--a-input-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/audio",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/avhubert_feat",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="ckpts/av_hubert/lrs3_vox/clean-pretrain/large_vox_iter5.pt",
    )
    parser.add_argument("--v-file-extension", type=str, default=".mp4", help="Video file extension.")
    parser.add_argument("--a-file-extension", type=str, default=".wav", help="Audio file extension.")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--user_dir", type=str, default=None)
    parser.add_argument("--audio-region-summary", type=str, default=None,
                        help="Raw-full inference summary selecting the target window for AVSync only")
    args = parser.parse_args()

    fairseq.utils.import_user_module(args)
    sys.path.append(args.user_dir)

    from aligndit.script.misc import avhubert_utils

    logger.info(args)
    main(args, custom_utils=avhubert_utils)
