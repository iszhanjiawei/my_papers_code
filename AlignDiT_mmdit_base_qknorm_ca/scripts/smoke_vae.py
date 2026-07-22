"""
Semantic-VAE 冒烟测试脚本
验证：
  1. 权重可正常加载
  2. CelebVDub 音频 encode→decode 不报错
  3. 潜空间维度和帧率符合预期（[T, 64] @ 40 Hz）
  4. 重建 SNR 是否合理
  5. 保存一条重建音频供人工聆听
"""

import sys
import glob
import random
import math
from pathlib import Path

import torch
import torchaudio
import librosa
import numpy as np

# ---------- 路径配置 ----------
SVAE_REPO   = "/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE"
SVAE_CKPT   = "/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k"
CELEBVDUB_WAV = "/zjw524/projects/data/CelebVDub/audio/train"
OUTPUT_DIR  = "/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT_mmdit_base/scripts/smoke_vae_output"
N_TEST      = 10     # 抽测音频条数
EXPECTED_SR = 16000
EXPECTED_DIM = 64
EXPECTED_HZ  = 40    # 潜空间帧率
# --------------------------------

sys.path.insert(0, SVAE_REPO)
from dac.model.dac import DAC                  # noqa: E402
from dac.model.utils import read_json_file     # noqa: E402

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ---- 1. 加载模型 ----
def load_model(save_path: str) -> DAC:
    metainfo = read_json_file(Path(save_path) / "metainfo.json")
    ckpt = torch.load(
        Path(save_path) / "dac" / "ema_state_dict.pth", map_location="cpu"
    )
    ckpt = {k.replace("ema_model.", ""): v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("projectors")}
    model = DAC(**metainfo["DAC"])
    del model.projectors
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


print("=" * 60)
print("[1] 加载 Semantic-VAE 权重 ...")
model = load_model(SVAE_CKPT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"    模型已加载，sample_rate={model.sample_rate}, device={device}")


# ---- 2. 收集测试音频 ----
all_wavs = glob.glob(f"{CELEBVDUB_WAV}/**/*.wav", recursive=True)
random.seed(42)
test_wavs = random.sample(all_wavs, min(N_TEST, len(all_wavs)))
print(f"\n[2] 从 CelebVDub 随机抽取 {len(test_wavs)} 条音频测试")


# ---- 3. 逐条 encode → decode ----
def snr_db(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """信噪比（dB），对齐长度后计算"""
    min_len = min(orig.shape[-1], recon.shape[-1])
    orig  = orig[..., :min_len].float()
    recon = recon[..., :min_len].float()
    noise_power = ((orig - recon) ** 2).mean()
    if noise_power < 1e-12:
        return float("inf")
    sig_power = (orig ** 2).mean()
    return 10 * math.log10((sig_power / noise_power).item() + 1e-12)


print(f"\n[3] encode → decode 测试 (期望潜空间: [{EXPECTED_DIM}d @ {EXPECTED_HZ}Hz])\n")
snr_list = []
saved = False

valid = 0
i = 0
wav_iter = iter(random.sample(all_wavs, len(all_wavs)))  # shuffle all

while valid < N_TEST:
    try:
        wav_path = next(wav_iter)
    except StopIteration:
        break

    try:
        wav_t, sr = torchaudio.load(wav_path)               # [C, T]
    except Exception:
        continue  # 跳过损坏/不可读文件

    if wav_t.shape[1] < EXPECTED_SR * 0.2:                  # 跳过 < 0.2s 极短片段
        continue

    if sr != EXPECTED_SR:
        wav_t = torchaudio.functional.resample(wav_t, sr, EXPECTED_SR)

    wav_t = wav_t.mean(0, keepdim=True).to(device)          # [1, T] mono
    wav_t = model.preprocess(wav_t, EXPECTED_SR)             # pad to model stride
    i = valid

    with torch.no_grad():
        # encode：z_hat 是投影后的潜空间（用于 decode），mu 是 VAE 均值
        z_hat, mu, log_var, _ = model.encode(wav_t.unsqueeze(0))  # wav: [1,T] → [1,1,T]
        # reparameterize → 这是 DiT 将来操作的 pre-proj latent
        latent = model.reparameterize(mu, log_var)  # [1, 64, T_lat]
        # decode（从 z_hat 重建，z_hat 已投影）
        recon = model.decode(z_hat)                  # [1, 1, T_wav]

    # ---- 形状校验 ----
    lat_dim  = latent.shape[1]         # 期望 64
    lat_T    = latent.shape[2]
    wav_T    = wav_t.shape[-1]
    actual_hz = EXPECTED_SR / (wav_T / lat_T) if lat_T > 0 else 0

    dim_ok = lat_dim == EXPECTED_DIM
    hz_ok  = abs(actual_hz - EXPECTED_HZ) < 1.0   # 允许 ±1 Hz 误差

    # ---- SNR ----
    orig_cpu  = wav_t.squeeze().cpu()
    recon_cpu = recon.squeeze().cpu()
    snr = snr_db(orig_cpu, recon_cpu)
    snr_list.append(snr)

    status = "✅" if (dim_ok and hz_ok) else "⚠️"
    print(f"  [{valid+1:02d}] {Path(wav_path).name:30s} | "
          f"latent [{lat_dim}d, T={lat_T}] @ {actual_hz:.1f}Hz {status} | "
          f"SNR={snr:+.1f}dB | wav_len={wav_T/EXPECTED_SR:.2f}s")

    # 保存第一条重建音频
    if not saved:
        orig_out  = Path(OUTPUT_DIR) / "orig.wav"
        recon_out = Path(OUTPUT_DIR) / "recon.wav"
        torchaudio.save(str(orig_out),  orig_cpu.unsqueeze(0), EXPECTED_SR)
        torchaudio.save(str(recon_out), recon_cpu.unsqueeze(0).clamp(-1, 1), EXPECTED_SR)
        print(f"\n    *** 原始音频已保存: {orig_out}")
        print(f"    *** 重建音频已保存: {recon_out}\n")
        saved = True

    valid += 1


# ---- 4. 汇总 ----
print("\n" + "=" * 60)
print("[4] 汇总")
print(f"    平均 SNR : {np.mean(snr_list):.2f} dB")
print(f"    最低 SNR : {np.min(snr_list):.2f} dB")
print(f"    最高 SNR : {np.max(snr_list):.2f} dB")
print(f"\n    [参考] 论文 (Semantic-VAE Table 3):")
print(f"      Vocos+mel UTMOS=3.24; Semantic-VAE UTMOS=3.56")
print(f"      SNR 参考值: 通常 >15dB 表示重建质量良好")

if np.mean(snr_list) >= 15:
    print("\n  ✅ 冒烟测试通过：VAE 重建质量良好，可继续集成到 MM-DiT")
elif np.mean(snr_list) >= 10:
    print("\n  ⚠️ 重建质量一般（可能存在域偏移），建议微调 VAE 后再集成")
else:
    print("\n  ❌ 重建质量较差，需检查权重/采样率或在 CelebVDub 上微调 VAE")
print("=" * 60)
