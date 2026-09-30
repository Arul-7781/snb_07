from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .checkpoint import load_checkpoint
from .metrics import si_sdr_db, snr_db
from .model import AudioINN
from .utils import write_json


def load_audio(path, sr, length):
    x, source_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if source_sr != sr:
        g = math.gcd(source_sr, sr)
        x = resample_poly(x, sr // g, source_sr // g).astype(np.float32)
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    else:
        x = x[:length]
    return torch.from_numpy(x).view(1, 1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cover", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--out-dir", default="demo_output")
    ap.add_argument("--seed", type=int, default=None, help="Random-latent seed; defaults to the training seed")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioINN(**cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    sr = cfg["data"]["sample_rate"]
    length = int(sr * cfg["data"]["segment_seconds"])
    cover = load_audio(args.cover, sr, length).to(device)
    secret = load_audio(args.secret, sr, length).to(device)

    with torch.no_grad():
        y, r = model.encode(cover, secret)
        seed = cfg["seed"] if args.seed is None else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(r.shape, device=device, dtype=r.dtype, generator=generator)
        _, s_random = model.decode(y, z)
        c_oracle, s_oracle = model.decode(y, r)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in {
        "cover.wav": cover,
        "secret.wav": secret,
        "stego.wav": y,
        "recovered_random_z.wav": s_random,
        "recovered_cover_true_r.wav": c_oracle,
        "recovered_true_r_oracle.wav": s_oracle,
    }.items():
        sf.write(out / name, tensor[0, 0].detach().cpu().float().numpy(), sr)

    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "random_z_seed": seed,
        "cover_stego_snr_db": float(snr_db(cover.float(), y.float()).item()),
        "secret_random_z_sisdr_db": float(si_sdr_db(secret.float(), s_random.float()).item()),
        "secret_true_r_sisdr_db": float(si_sdr_db(secret.float(), s_oracle.float()).item()),
        "cover_cycle_max_abs": float((cover.float() - c_oracle.float()).abs().max()),
        "secret_cycle_max_abs": float((secret.float() - s_oracle.float()).abs().max()),
        "stego_peak_abs": float(y.float().abs().max()),
        "stego_clip_fraction": float((y.float().abs() >= 1.0).float().mean()),
    }
    write_json(out / "metrics.json", metrics)
    print(f"Wrote demo files and metrics to {out}")
    print(metrics)

if __name__ == "__main__":
    main()
