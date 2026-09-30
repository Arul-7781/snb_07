import argparse
import math
from pathlib import Path

import torch

from audio_inn.data import discover_audio_files
from audio_inn.model import AudioINN
from audio_inn.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_4090.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    errors = []

    print("=== GPU ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch CUDA:", torch.version.cuda)
    minimum_vram_gb = float(cfg["training"].get("minimum_vram_gb", 0))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print("gpu:", props.name)
        print("vram GB:", round(vram_gb, 2))
        if vram_gb < minimum_vram_gb:
            errors.append(
                f"Config requires at least {minimum_vram_gb:g} GB VRAM, but {vram_gb:.2f} GB is available"
            )
        if cfg["training"].get("amp") and cfg["training"].get("amp_dtype") == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                errors.append("Config requests bfloat16 AMP, but this GPU does not support bfloat16")
    elif minimum_vram_gb:
        errors.append(f"Config requires a CUDA GPU with at least {minimum_vram_gb:g} GB VRAM")

    print("\n=== Dataset ===")
    root = Path(cfg["data"]["root"])
    splits = [cfg["data"]["train_split"], cfg["data"]["val_split"], cfg["data"]["test_split"]]
    for split in splits:
        files = discover_audio_files(root / split)
        print(f"{split}: {len(files)} audio files")
        if len(files) < 2:
            errors.append(f"{root / split} needs at least 2 .flac or .wav files")

    print("\n=== Configuration ===")
    sample_rate = int(cfg["data"]["sample_rate"])
    segment_seconds = float(cfg["data"]["segment_seconds"])
    length = int(round(sample_rate * segment_seconds))
    squeeze_factor = int(cfg["model"]["squeeze_factor"])
    fft_sizes = cfg["stft"]["fft_sizes"]
    hop_factors = cfg["stft"]["hop_factors"]
    print("segment samples:", length)
    print("squeeze factor:", squeeze_factor)
    if sample_rate <= 0 or segment_seconds <= 0:
        errors.append("sample_rate and segment_seconds must be positive")
    if squeeze_factor <= 0 or length % max(squeeze_factor, 1):
        errors.append(f"Segment length {length} must be divisible by squeeze factor {squeeze_factor}")
    if not fft_sizes or len(fft_sizes) != len(hop_factors):
        errors.append("fft_sizes and hop_factors must be non-empty and have equal lengths")
    elif any(n <= 0 or h <= 0 or n // h == 0 for n, h in zip(fft_sizes, hop_factors)):
        errors.append("FFT sizes and hop factors must be positive and produce a nonzero hop length")
    for key in ("train_pairs_per_epoch", "val_pairs"):
        if int(cfg["data"][key]) <= 0:
            errors.append(f"data.{key} must be positive")
    for key in ("epochs", "batch_size", "log_every_steps", "validate_every_epochs", "save_every_epochs", "sample_every_epochs"):
        if int(cfg["training"][key]) <= 0:
            errors.append(f"training.{key} must be positive")

    print("\n=== Model synthetic round-trip ===")
    if length > 0 and squeeze_factor > 0 and length % squeeze_factor == 0:
        model = AudioINN(**cfg["model"])
        generator = torch.Generator().manual_seed(int(cfg["seed"]))
        cover = torch.randn(1, 1, length, generator=generator) * 0.05
        secret = torch.randn(1, 1, length, generator=generator) * 0.05
        with torch.no_grad():
            stego, latent = model.encode(cover, secret)
            cover_recovered, secret_recovered = model.decode(stego, latent)
        cover_error = float((cover - cover_recovered).abs().max())
        secret_error = float((secret - secret_recovered).abs().max())
        print("max cover cycle error:", cover_error)
        print("max secret cycle error:", secret_error)
        if not math.isfinite(cover_error + secret_error) or max(cover_error, secret_error) >= 2e-5:
            errors.append("Synthetic forward/reverse cycle error must be finite and below 2e-5")
    else:
        print("skipped because the segment length is incompatible with the model")

    if errors:
        print("\n=== FAILED ===")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    print("\npreflight passed")


if __name__ == "__main__":
    main()
