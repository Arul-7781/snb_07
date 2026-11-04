from __future__ import annotations

import argparse
import json
import random
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from spectra_inn.config import load_config, segment_samples
from spectra_inn.data.dataset import PairedAudioDataset, SyntheticPairedAudioDataset
from spectra_inn.data.evaluation import EvaluationPairedAudioDataset, balanced_indices
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
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Explicitly override optimizer LR AFTER loading a checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic audio and a short run")
    parser.add_argument(
        "--overfit-pairs", type=int, default=None,
        help="Cache this many real training pairs for a separate _overfit diagnostic run",
    )
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
        if data.get("validation_pairing", "legacy") == "legacy":
            val_dataset = PairedAudioDataset(
                split=str(data["val_split"]), random_gain_db=0.0, random_crop=False, **common
            )
        else:
            val_dataset = EvaluationPairedAudioDataset(
                split=str(data["val_split"]), pairing=data["validation_pairing"],
                pair_seed=int(data.get("evaluation_pair_seed", 0)), **common,
            )
        if data.get("validation_subset_size") is not None:
            indices = balanced_indices(
                val_dataset.records, int(data["validation_subset_size"]),
                int(data.get("evaluation_pair_seed", 0)),
            )
            val_dataset = Subset(val_dataset, indices)
        workers = int(data["num_workers"])
        pin_memory = bool(data["pin_memory"])

    overfit_pairs = data.get("overfit_pairs")
    if overfit_pairs is not None:
        if smoke:
            raise ValueError("--overfit-pairs uses real audio and cannot be combined with --smoke")
        count = int(overfit_pairs)
        if count < int(training["batch_size"]) or count > len(train_dataset):
            raise ValueError("overfit_pairs must be between batch_size and training dataset size")
        indices = random.Random(int(config["experiment"]["seed"])).sample(
            range(len(train_dataset)), count
        )
        # Decode/crop/augment once, not on every epoch. Validation remains held-out audio.
        train_dataset = [train_dataset[index] for index in indices]
        workers = 0
        pin_memory = False

    train_generator = val_generator = None
    if data.get("loader_seed") is not None:
        # Model initialization must not change the matched arms' data order/worker seeds.
        train_generator = torch.Generator().manual_seed(int(data["loader_seed"]))
        val_generator = torch.Generator().manual_seed(int(data["loader_seed"]) + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=workers > 0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=max(0, min(workers, 2)),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=workers > 0,
        generator=val_generator,
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
    recovered_cover, recovered_secret, predicted = model.decode(stego)
    total, components = objective(
        cover=cover,
        secret=secret,
        stego=stego,
        residual=residual,
        recovered_cover=recovered_cover,
        recovered_secret=recovered_secret,
        diagnostics=diagnostics,
        predicted_residual=predicted if model.predictor is not None else None,
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
def evaluate_fixed_pairs(
    model: CleanAudioINN,
    objective: StageOneLoss,
    pairs: list[dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Score the entire overfit cache in FP32 without perturbing training RNG or mode."""
    if not pairs or batch_size < 1:
        raise ValueError("Fixed-pair evaluation needs pairs and a positive batch_size")
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    was_training = model.training
    totals: dict[str, float] = {}
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            with torch.autocast(device_type=device.type, enabled=False):
                for start in range(0, len(pairs), batch_size):
                    chunk = pairs[start : start + batch_size]
                    batch = {
                        key: torch.stack([pair[key] for pair in chunk])
                        for key in ("cover", "secret")
                    }
                    cover, secret = to_device(batch, device)
                    total, components, metrics = compute_batch(model, objective, cover, secret)
                    for name, value in {"total": total, **components, **metrics}.items():
                        weighted = float(value.float().cpu()) * len(chunk)
                        totals[name] = totals.get(name, 0.0) + weighted
    finally:
        model.train(was_training)
    return {name: value / len(pairs) for name, value in totals.items()}


@torch.no_grad()
def latent_comparison(
    model: CleanAudioINN,
    cover: torch.Tensor,
    secret: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Secret recovery under each decoder-side latent, evaluated in float32."""
    stego, residual, _ = model(cover, secret)
    _, gaussian_secret, _ = model.blind_inverse(stego)
    _, oracle_secret = model.inverse(stego, residual)
    values = {
        "secret_snr_db_gaussian": snr_db(secret, gaussian_secret),
        "secret_snr_db_true_residual": snr_db(secret, oracle_secret),
    }
    if model.predictor is not None:
        _, predicted_secret, predicted = model.predicted_inverse(stego)
        values["secret_snr_db_predicted"] = snr_db(secret, predicted_secret)
        values["latent_prediction_snr_db"] = snr_db(residual, predicted)
    return values


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
        values.update(latent_comparison(model, cover, secret))
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
    saved_predictor = checkpoint.get("config", {}).get("model", {}).get("predictor", {})
    if getattr(model, "predictor", None) is not None:
        if saved_predictor.get("architecture", "flat") != model.predictor_architecture:
            raise ValueError("Checkpoint predictor architecture differs; start a fresh run")
        saved_dilations = saved_predictor.get("dilations")
        if saved_dilations is None:
            count = int(saved_predictor.get("num_blocks", 8))
            if model.predictor_architecture == "flat":
                saved_dilations = [1] * count
            else:
                counts = [count // 3 + int(index < count % 3) for index in range(3)]
                saved_dilations = [2 ** (index % 3 + level) for level, n in enumerate(counts)
                                   for index in range(n)]
        if tuple(saved_dilations) != model.predictor.dilations:
            raise ValueError("Checkpoint predictor dilations differ; start a fresh run")
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
    predictor = getattr(model, "predictor", None)
    serializable_config = {key: value for key, value in config.items() if not key.startswith("_")}
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_config, handle, sort_keys=False)
    metadata = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latent_mode": getattr(model, "latent_mode", "gaussian"),
        "waveform_mode": config["loss"].get("waveform_mode", "charbonnier"),
        "predictor_parameters": (
            sum(parameter.numel() for parameter in predictor.parameters())
            if predictor is not None
            else 0
        ),
        "predictor_dilations": list(predictor.dilations) if predictor is not None else None,
        "predictor_architecture": getattr(model, "predictor_architecture", "flat"),
        "predictor_context_kind": getattr(predictor, "receptive_field_kind", "exact"),
        "predictor_context_ms": (
            predictor.receptive_field
            * (2 ** int(config["model"]["wavelet_levels"]))
            / int(config["data"]["sample_rate"])
            * 1000
            if predictor is not None
            else None
        ),
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
    run_started = time.perf_counter()
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
    if args.learning_rate is not None:
        if not 0 < args.learning_rate < float("inf"):
            raise ValueError("--learning-rate must be finite and positive")
        config["training"]["learning_rate"] = args.learning_rate
    if args.overfit_pairs is not None:
        if args.smoke:
            raise ValueError("--overfit-pairs cannot be combined with --smoke")
        if args.overfit_pairs < int(config["training"]["batch_size"]):
            raise ValueError("--overfit-pairs must be at least batch_size")
        config["data"]["overfit_pairs"] = args.overfit_pairs

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(args.device, args.smoke)
    training = config["training"]
    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    elif config["data"].get("overfit_pairs") is not None:
        output_dir = output_dir.with_name(output_dir.name + "_overfit")
        print(
            f"Overfit diagnostic: {config['data']['overfit_pairs']} fixed training pairs; "
            "validation still uses held-out audio. Do not use this as a quality benchmark."
        )

    model = CleanAudioINN.from_config(config).to(device)
    objective = StageOneLoss(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp, init_scale=float(training.get("amp_init_scale", 65536.0))
    )
    train_loader, val_loader = create_loaders(config, args.smoke)
    write_run_metadata(output_dir, config, model, device)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    resume_value = args.resume or training.get("resume")
    step = 0
    if resume_value:
        step = load_checkpoint(Path(resume_value), model, optimizer, scaler, device)
        print(f"Resumed from step {step}: {resume_value}")
    if args.learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"Applied explicit optimizer LR: {args.learning_rate}")

    initial_step = step
    skipped_updates = 0
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("Stop requested; finishing the current step, validation, and atomic checkpoint.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

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
    last_log_step = step
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
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_updates += int(scaler.get_scale() < old_scale)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)
        time_limit_reached = stop_requested or (
            deadline is not None and time.perf_counter() >= deadline
        )
        final_step = step == max_steps or time_limit_reached

        if step == 1 or step % log_every == 0:
            elapsed = max(time.perf_counter() - last_log_time, 1e-6)
            last_log_time = time.perf_counter()
            writer.add_scalar("train/total", aggregate_total, step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
            writer.add_scalar("train/steps_per_second", (step - last_log_step) / elapsed, step)
            writer.add_scalar("train/amp_skipped_updates_session", skipped_updates, step)
            writer.add_scalar("train/amp_scale", scaler.get_scale(), step)
            last_log_step = step
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
            predicted_snr = validation.get("secret_snr_db_predicted")
            predicted_text = f"predicted={predicted_snr:.2f} " if predicted_snr is not None else ""
            print(
                f"\nvalidation step={step} total={validation['total']:.4f} "
                f"cover_snr={validation['cover_snr_db']:.2f} "
                f"secret_snr={validation['secret_snr_db']:.2f} "
                f"[{predicted_text}gaussian={validation['secret_snr_db_gaussian']:.2f} "
                f"true_residual={validation['secret_snr_db_true_residual']:.2f}] "
                f"cycle={validation['exact_cycle_max']:.3e}"
            )
            if config["data"].get("overfit_pairs") is not None:
                fixed_training = evaluate_fixed_pairs(
                    model, objective, train_loader.dataset, device, int(training["batch_size"])
                )
                for name, value in fixed_training.items():
                    writer.add_scalar(f"overfit_train/{name}", value, step)
                print(
                    f"fixed_train step={step} pairs={len(train_loader.dataset)} "
                    f"total={fixed_training['total']:.4f} "
                    f"cover_snr={fixed_training['cover_snr_db']:.2f} "
                    f"secret_snr={fixed_training['secret_snr_db']:.2f}"
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
            print(f"\nStopped at step {step}; requested stop or configured time budget reached.")
            break

    progress.close()
    writer.close()
    summary = {
        "initial_step": initial_step, "step": step, "target_steps": max_steps,
        "target_reached": step >= max_steps, "elapsed_seconds": time.perf_counter() - run_started,
        "steps_attempted_session": step - initial_step,
        "amp_skipped_updates_session": skipped_updates,
        "optimizer_updates_session": step - initial_step - skipped_updates,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "stopped_by_signal": stop_requested,
    }
    temporary = output_dir / "training_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
