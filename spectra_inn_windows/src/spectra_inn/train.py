from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from spectra_inn.config import load_config, segment_samples
from spectra_inn.data.dataset import PairedAudioDataset, SyntheticPairedAudioDataset
from spectra_inn.losses.audio import StageOneLoss
from spectra_inn.metrics.audio import snr_db
from spectra_inn.models.clean_inn import CleanAudioINN
from spectra_inn.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage-1 clean SPECTRA-INN baseline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic audio and a short run")
    return parser.parse_args()


def resolve_device(requested: str, smoke: bool) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "cpu":
        return torch.device("cpu")
    if cuda_available:
        return torch.device("cuda")
    if not smoke:
        raise RuntimeError(
            "No CUDA GPU is visible. Refusing a real training run; "
            "use --device cpu only if intentional."
        )
    return torch.device("cpu")


def create_loaders(
    config: dict[str, Any], smoke: bool
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    data = config["data"]
    training = config["training"]
    samples = segment_samples(config)
    if smoke:
        train_dataset = SyntheticPairedAudioDataset(
            length=64,
            channels=int(data["channels"]),
            samples=samples,
            sample_rate=int(data["sample_rate"]),
        )
        val_dataset = SyntheticPairedAudioDataset(
            length=16,
            channels=int(data["channels"]),
            samples=samples,
            sample_rate=int(data["sample_rate"]),
        )
        workers = 0
        pin_memory = False
    else:
        common = {
            "manifest": data["manifest"],
            "sample_rate": int(data["sample_rate"]),
            "channels": int(data["channels"]),
            "segment_samples": samples,
            "min_rms": float(data["min_rms"]),
        }
        train_dataset = PairedAudioDataset(
            split=str(data["train_split"]),
            random_gain_db=float(data["random_gain_db"]),
            random_crop=True,
            **common,
        )
        val_dataset = PairedAudioDataset(
            split=str(data["val_split"]),
            random_gain_db=0.0,
            random_crop=False,
            **common,
        )
        workers = int(data["num_workers"])
        pin_memory = bool(data["pin_memory"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=max(0, min(workers, 2)),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def infinite_batches(
    loader: DataLoader[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["cover"].to(device, non_blocking=True),
        batch["secret"].to(device, non_blocking=True),
    )


def compute_batch(
    model: CleanAudioINN,
    objective: StageOneLoss,
    cover: torch.Tensor,
    secret: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stego, residual, diagnostics = model(cover, secret)
    recovered_cover, recovered_secret, _ = model.blind_inverse(stego)
    total, components = objective(
        cover=cover,
        secret=secret,
        stego=stego,
        residual=residual,
        recovered_cover=recovered_cover,
        recovered_secret=recovered_secret,
        diagnostics=diagnostics,
    )
    metrics = {
        "cover_snr_db": snr_db(cover, stego),
        "secret_snr_db": snr_db(secret, recovered_secret),
        "max_abs_stego": stego.abs().amax(),
        "mean_abs_log_scale": diagnostics["mean_abs_log_scale"],
        "max_abs_log_scale": diagnostics["max_abs_log_scale"],
    }
    return total, components, metrics


@torch.no_grad()
def exact_cycle_error(
    model: CleanAudioINN,
    cover: torch.Tensor,
    secret: torch.Tensor,
) -> torch.Tensor:
    stego, residual, _ = model(cover, secret)
    exact_cover, exact_secret = model.inverse(stego, residual)
    return torch.maximum((cover - exact_cover).abs().amax(), (secret - exact_secret).abs().amax())


@torch.no_grad()
def validate(
    model: CleanAudioINN,
    objective: StageOneLoss,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        cover, secret = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            total, components, metrics = compute_batch(model, objective, cover, secret)
        values = {"total": total, **components, **metrics}
        values["exact_cycle_max"] = exact_cycle_error(model, cover, secret)
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().float().cpu())
        count += 1
    model.train()
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    keep: int,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step:08d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "torch_version": torch.__version__,
        },
        temporary,
    )
    temporary.replace(path)
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for old_path in checkpoints[:-max(1, keep)]:
        old_path.unlink()
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["step"])


def write_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = {key: value for key, value in config.items() if not key.startswith("_")}
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_config, handle, sort_keys=False)
    metadata = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["training"]["max_steps"] = args.max_steps
    if args.max_hours is not None:
        if args.max_hours <= 0:
            raise ValueError("--max-hours must be positive")
        config["training"]["max_hours"] = args.max_hours

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(args.device, args.smoke)
    training = config["training"]
    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")

    model = CleanAudioINN.from_config(config).to(device)
    objective = StageOneLoss(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_loader, val_loader = create_loaders(config, args.smoke)
    write_run_metadata(output_dir, config, model, device)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    resume_value = args.resume or training.get("resume")
    step = 0
    if resume_value:
        step = load_checkpoint(Path(resume_value), model, optimizer, scaler, device)
        print(f"Resumed from step {step}: {resume_value}")

    max_steps = int(training["max_steps"])
    max_hours_value = training.get("max_hours")
    max_hours = float(max_hours_value) if max_hours_value is not None else None
    deadline = time.perf_counter() + max_hours * 3600 if max_hours is not None else None
    accumulation_steps = int(training["accumulation_steps"])
    log_every = int(training["log_every"])
    validate_every = int(training["validate_every"])
    checkpoint_every = int(training["checkpoint_every"])
    batches = infinite_batches(train_loader)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    progress = tqdm(total=max_steps, initial=step, unit="step")
    last_log_time = time.perf_counter()
    while step < max_steps:
        aggregate_total = 0.0
        last_components: dict[str, torch.Tensor] = {}
        last_metrics: dict[str, torch.Tensor] = {}
        last_cover: torch.Tensor | None = None
        last_secret: torch.Tensor | None = None

        for _ in range(accumulation_steps):
            cover, secret = to_device(next(batches), device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                total, components, metrics = compute_batch(model, objective, cover, secret)
                scaled_loss = total / accumulation_steps
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {float(total)}")
            scaler.scale(scaled_loss).backward()
            aggregate_total += float(total.detach().float().cpu()) / accumulation_steps
            last_components = components
            last_metrics = metrics
            last_cover = cover
            last_secret = secret

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["grad_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)
        time_limit_reached = deadline is not None and time.perf_counter() >= deadline
        final_step = step == max_steps or time_limit_reached

        if step == 1 or step % log_every == 0:
            elapsed = max(time.perf_counter() - last_log_time, 1e-6)
            last_log_time = time.perf_counter()
            writer.add_scalar("train/total", aggregate_total, step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
            writer.add_scalar("train/steps_per_second", log_every / elapsed, step)
            for name, value in last_components.items():
                writer.add_scalar(f"train/loss_{name}", float(value.detach().float().cpu()), step)
            for name, value in last_metrics.items():
                writer.add_scalar(f"train/{name}", float(value.detach().float().cpu()), step)
            if last_cover is not None and last_secret is not None:
                cycle = exact_cycle_error(model, last_cover, last_secret)
                writer.add_scalar("train/exact_cycle_max", float(cycle.cpu()), step)
                progress.set_postfix(loss=f"{aggregate_total:.4f}", cycle=f"{float(cycle):.2e}")

        if step % validate_every == 0 or final_step:
            validation = validate(
                model=model,
                objective=objective,
                loader=val_loader,
                device=device,
                amp=amp,
                max_batches=int(training["validation_batches"]),
            )
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, step)
            print(
                f"\nvalidation step={step} total={validation['total']:.4f} "
                f"cover_snr={validation['cover_snr_db']:.2f} "
                f"secret_snr={validation['secret_snr_db']:.2f} "
                f"cycle={validation['exact_cycle_max']:.3e}"
            )

        if step % checkpoint_every == 0 or final_step:
            path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                keep=int(training["keep_checkpoints"]),
            )
            print(f"\ncheckpoint: {path}")
            writer.flush()

        if time_limit_reached:
            print(f"\nReached the configured {max_hours:.2f}-hour wall-clock limit at step {step}.")
            break

    progress.close()
    writer.close()


if __name__ == "__main__":
    main()
