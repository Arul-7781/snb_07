from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from spectra_inn.config import load_config, segment_samples
from spectra_inn.data.dataset import PairedAudioDataset
from spectra_inn.metrics.audio import si_sdr_db, snr_db
from spectra_inn.models.clean_inn import CleanAudioINN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SPECTRA-INN checkpoint and export listening examples"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=None, help="Override the config stored in the checkpoint"
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Override data.manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--examples", type=int, default=5, help="Number of WAV examples to export")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the Gaussian latent")
    return parser.parse_args()


def quantize_pcm16(waveform: torch.Tensor) -> torch.Tensor:
    """Simulate saving the stego as a 16-bit WAV file and reading it back."""
    return torch.round(waveform.clamp(-1.0, 1.0) * 32767.0) / 32767.0


@torch.no_grad()
def evaluate_batch(
    model: CleanAudioINN,
    cover: torch.Tensor,
    secret: torch.Tensor,
    generator: torch.Generator,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    stego, residual, _ = model(cover, secret)
    stego_pcm16 = quantize_pcm16(stego)
    recovered = {
        "true_residual": model.inverse(stego, residual)[1],
        "gaussian": model.blind_inverse(stego, generator=generator)[1],
    }
    if model.predictor is not None:
        recovered["predicted"] = model.predicted_inverse(stego)[1]
        recovered["predicted_pcm16"] = model.predicted_inverse(stego_pcm16)[1]
    else:
        recovered["gaussian_pcm16"] = model.blind_inverse(stego_pcm16, generator=generator)[1]

    metrics = {
        "cover_snr_db": snr_db(cover, stego),
        "cover_si_sdr_db": si_sdr_db(cover, stego),
        "stego_peak": stego.abs().amax(),
        "stego_clip_fraction": (stego.abs() > 1.0).float().mean(),
    }
    for name, estimate in recovered.items():
        metrics[f"secret_snr_db_{name}"] = snr_db(secret, estimate)
        metrics[f"secret_si_sdr_db_{name}"] = si_sdr_db(secret, estimate)
    signals = {"cover": cover, "secret": secret, "stego": stego}
    signals.update({f"recovered_secret_{name}": value for name, value in recovered.items()})
    return {name: float(value.float().cpu()) for name, value in metrics.items()}, signals


def export_examples(
    signals: dict[str, torch.Tensor],
    output_dir: Path,
    start_index: int,
    count: int,
    sample_rate: int,
) -> int:
    written = 0
    batch = next(iter(signals.values())).shape[0]
    for item in range(min(batch, count)):
        example_dir = output_dir / f"example_{start_index + item:03d}"
        example_dir.mkdir(parents=True, exist_ok=True)
        for name, waveform in signals.items():
            audio = waveform[item].detach().float().cpu().clamp(-1.0, 1.0).numpy().T
            sf.write(example_dir / f"{name}.wav", np.ascontiguousarray(audio), sample_rate)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint["config"]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = CleanAudioINN.from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = config["data"]
    dataset = PairedAudioDataset(
        manifest=args.manifest or data["manifest"],
        split=args.split,
        sample_rate=int(data["sample_rate"]),
        channels=int(data["channels"]),
        segment_samples=segment_samples(config),
        min_rms=float(data["min_rms"]),
        random_gain_db=0.0,
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_dir = args.output_dir or (
        checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}_{args.split}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    totals: dict[str, float] = {}
    items = 0
    exported = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)
        metrics, signals = evaluate_batch(model, cover, secret, generator)
        size = cover.shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * size
        items += size
        if exported < args.examples:
            exported += export_examples(
                signals, output_dir, exported, args.examples - exported, int(data["sample_rate"])
            )
    if items == 0:
        raise RuntimeError(f"Split '{args.split}' produced no evaluation batches")

    summary = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", -1)),
        "latent_mode": model.latent_mode,
        "split": args.split,
        "num_items": items,
        "metrics": {name: value / items for name, value in totals.items()},
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Evaluated {items} items from split '{args.split}' at step {summary['step']}")
    print(f"Decoder latent: {model.latent_mode}")
    for name, value in sorted(summary["metrics"].items()):
        print(f"  {name:40s} {value:10.4f}")
    print(f"Wrote metrics.json and {exported} listening examples to {output_dir}")


if __name__ == "__main__":
    main()
