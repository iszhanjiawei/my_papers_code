"""Check full-size D1 forward/backward on real training clips; no optimizer step.

Run from this isolated snapshot with PYTHONPATH=src and the aligndit Python.
Requires the same existing data and pretrained audio-teacher resources as training.
"""

import argparse
import json
from pathlib import Path

import torch
from accelerate.utils import set_seed
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model import DiT_VT_MMDiT
from aligndit.model.avhubert_teacher import AVHubertAudioTeacher
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.dataset import load_dataset_mel
from aligndit.model.modules import MelSpec_tacotron
from f5_tts.model.utils import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    set_seed(666)
    with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base="1.3"):
        cfg = compose(config_name="finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope_avhubert_dual_role")
    vocab, vocab_size = get_tokenizer(str(Path(cfg.datasets.data_dir) / "CelebVDub_char/vocab.txt"), "custom")
    dataset = load_dataset_mel(
        cfg.datasets.name, cfg.model.tokenizer,
        mel_spec_module=MelSpec_tacotron(**cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in cfg.model.mel_spec.items() if k != "mel_spec_type"},
        dataset_type="CustomDataset_mel_video_teacher", data_dir=cfg.datasets.data_dir,
    )
    indices = [i for i, duration in enumerate(dataset.durations) if 3.0 <= duration <= 5.0][:2]
    assert len(indices) == 2
    batch = dataset.collate_fn([dataset[i] for i in indices])
    device = torch.device(args.device)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    teacher = AVHubertAudioTeacher(**OmegaConf.to_container(cfg.audio_teacher, resolve=True), device=device)
    targets, target_lengths = teacher.encode(batch["audio_paths"])
    model = CFM_VT(
        transformer=DiT_VT_MMDiT(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=80),
        mel_spec_module=MelSpec_tacotron(**cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in cfg.model.mel_spec.items() if k != "mel_spec_type"},
        vocab_char_map=vocab, ctc_lambda=cfg.model.ctc_lambda,
        avhubert_rep_lambda=cfg.model.avhubert_rep_lambda,
    ).to(device).train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss, components, _, prediction = model(
            batch["mel"].transpose(1, 2), text=batch["text"], video=batch["video"],
            lens=batch["mel_lengths"], text_lens=batch["text_lengths"],
            video_lens=batch["video_lengths"], audio_teacher_features=targets,
            audio_teacher_lengths=target_lengths,
        )
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction).all()
    loss.backward()
    projector = model.transformer.avhubert_rep_projector
    projector_grad = sum(float(p.grad.float().abs().sum()) for p in projector.parameters() if p.grad is not None)
    assert projector_grad > 0
    assert all(p.grad is None and not p.requires_grad for p in teacher.model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert components["avhubert_rep_valid_frames"] > 0
    assert abs(float(loss) - (
        components["diff_loss"] + cfg.model.ctc_lambda * components["ctc_loss"]
        + components["avhubert_rep_weighted_loss"]
    )) < 1e-4
    report = dict(
        status="passed", purpose="full D1 real training batch gradient check, no optimizer update",
        dataset_indices=indices, audio_paths=batch["audio_paths"],
        student_mel_lengths=batch["mel_lengths"].tolist(), teacher_lengths=target_lengths.tolist(),
        loss=float(loss), components=components, projector_absolute_gradient=projector_grad,
        teacher_frozen=True, generator_parameters=sum(p.numel() for p in model.parameters()),
        peak_gpu_gib=(torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
