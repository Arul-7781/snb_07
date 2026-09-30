from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .data import LibriSpeechPairDataset
from .losses import compute_losses
from .metrics import si_sdr_db, snr_db
from .model import AudioINN
from .utils import append_csv, configure_torch, device_summary, load_config, make_logger, save_config, set_seed, worker_init_fn, write_json


def build_model(cfg):
    return AudioINN(**cfg["model"])


def build_loader(cfg, split: str, pairs: int, batch_size: int, train: bool):
    dcfg = cfg["data"]
    ds = LibriSpeechPairDataset(
        root=dcfg["root"],
        split=split,
        sample_rate=dcfg["sample_rate"],
        segment_seconds=dcfg["segment_seconds"],
        num_pairs=pairs,
        seed=cfg["seed"] + (0 if train else 100000),
    )
    g = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=dcfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
        generator=g,
    )


def autocast_ctx(cfg, device):
    if device.type != "cuda" or not cfg["training"].get("amp", True):
        return torch.autocast(device_type=device.type, enabled=False)
    dtype_name = cfg["training"].get("amp_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    sums = {"loss": 0.0, "cover_snr": 0.0, "secret_sisdr": 0.0, "secret_sisdr_true_r": 0.0}
    count = 0
    cycle_max_abs = 0.0
    stego_peak_abs = 0.0
    stego_clipped = 0
    stego_samples = 0
    val_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 999)
    for cover, secret in loader:
        cover = cover.to(device, non_blocking=True)
        secret = secret.to(device, non_blocking=True)
        with autocast_ctx(cfg, device):
            stego, r = model.encode(cover, secret)
            z = torch.randn(r.shape, device=device, dtype=r.dtype, generator=val_gen)
            _, secret_hat = model.decode(stego, z)
            cover_oracle, secret_oracle = model.decode(stego, r)
            total, _ = compute_losses(cover, secret, stego, r, secret_hat, cfg)
        b = cover.shape[0]
        sums["loss"] += float(total) * b
        sums["cover_snr"] += float(snr_db(cover.float(), stego.float()).mean()) * b
        sums["secret_sisdr"] += float(si_sdr_db(secret.float(), secret_hat.float()).mean()) * b
        sums["secret_sisdr_true_r"] += float(si_sdr_db(secret.float(), secret_oracle.float()).mean()) * b
        cycle_max_abs = max(
            cycle_max_abs,
            float((cover.float() - cover_oracle.float()).abs().max()),
            float((secret.float() - secret_oracle.float()).abs().max()),
        )
        stego_peak_abs = max(stego_peak_abs, float(stego.float().abs().max()))
        stego_clipped += int((stego.float().abs() >= 1.0).sum())
        stego_samples += stego.numel()
        count += b
    metrics = {k: v / max(count, 1) for k, v in sums.items()}
    metrics.update({
        "cycle_max_abs": cycle_max_abs,
        "stego_peak_abs": stego_peak_abs,
        "stego_clip_fraction": stego_clipped / max(stego_samples, 1),
    })
    return metrics


def save_samples(model, batch, cfg, device, out_dir: Path, epoch: int):
    model.eval()
    cover, secret = batch
    cover = cover[:1].to(device)
    secret = secret[:1].to(device)
    with torch.no_grad(), autocast_ctx(cfg, device):
        stego, r = model.encode(cover, secret)
        z = torch.randn_like(r)
        _, secret_hat = model.decode(stego, z)
        cover_oracle, secret_oracle = model.decode(stego, r)
    d = out_dir / f"epoch_{epoch:03d}"
    d.mkdir(parents=True, exist_ok=True)
    sr = cfg["data"]["sample_rate"]
    for name, x in {
        "cover": cover,
        "secret": secret,
        "stego": stego,
        "secret_random_z": secret_hat,
        "cover_true_r_oracle": cover_oracle,
        "secret_true_r_oracle": secret_oracle,
    }.items():
        sf.write(d / f"{name}.wav", x[0, 0].detach().float().cpu().numpy(), sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_4090.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Allow a fresh run to replace an existing run directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.resume:
        cfg["training"]["resume"] = args.resume

    set_seed(cfg["seed"])
    configure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg["output"]["root"]) / cfg["run_name"]
    if run_dir.exists() and any(run_dir.iterdir()) and not cfg["training"].get("resume"):
        if not args.overwrite:
            raise RuntimeError(f"Run directory {run_dir} already exists and is non-empty. Use a new --run-name, --resume, or --overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")
    write_json(run_dir / "environment.json", device_summary())
    logger = make_logger(run_dir)
    if SummaryWriter is not None:
        writer = SummaryWriter(run_dir / "tensorboard")
    else:
        writer = None
        logger.warning("TensorBoard is unavailable; continuing with CSV/file logging only.")
    logger.info("Environment: %s", device_summary())

    train_loader = build_loader(cfg, cfg["data"]["train_split"], cfg["data"]["train_pairs_per_epoch"], cfg["training"]["batch_size"], True)
    val_loader = build_loader(cfg, cfg["data"]["val_split"], cfg["data"]["val_pairs"], cfg["training"]["batch_size"], False)
    sample_batch = next(iter(val_loader))

    model = build_model(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    logger.info(
        "Data: train_split=%s files=%d pairs=%d val_split=%s files=%d pairs=%d",
        cfg["data"]["train_split"], len(train_loader.dataset.files), len(train_loader.dataset),
        cfg["data"]["val_split"], len(val_loader.dataset.files), len(val_loader.dataset),
    )
    logger.info("Model: parameters=%d config=%s", sum(p.numel() for p in model.parameters()), cfg["model"])

    start_epoch = 1
    step = 0
    best_val = float("inf")
    resume = cfg["training"].get("resume")
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        logger.info("Resumed from %s at epoch=%s step=%s", resume, start_epoch, step)

    use_scaler = device.type == "cuda" and cfg["training"].get("amp", True) and cfg["training"].get("amp_dtype") == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for cover, secret in pbar:
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(cfg, device):
                stego, r = model.encode(cover, secret)
                z = torch.randn_like(r)
                _, secret_hat = model.decode(stego, z)
                total, parts = compute_losses(cover, secret, stego, r, secret_hat, cfg)

            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step + 1}: {float(total.detach())}")

            if use_scaler:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                optimizer.step()

            step += 1
            if step % cfg["training"]["log_every_steps"] == 0:
                metrics = {k: float(v) for k, v in parts.items()}
                metrics["cover_snr"] = float(snr_db(cover.float(), stego.float()).mean().detach())
                metrics["secret_sisdr"] = float(si_sdr_db(secret.float(), secret_hat.float()).mean().detach())
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}", sisdr=f"{metrics['secret_sisdr']:.2f}")
                for k, v in metrics.items():
                    if writer is not None:
                        writer.add_scalar(f"train/{k}", v, step)
                append_csv(run_dir / "train_metrics.csv", {"step": step, "epoch": epoch, **metrics})

        if epoch % cfg["training"]["validate_every_epochs"] == 0:
            vm = validate(model, val_loader, cfg, device)
            logger.info("epoch=%d val=%s", epoch, vm)
            for k, v in vm.items():
                if writer is not None:
                    writer.add_scalar(f"val/{k}", v, step)
            append_csv(run_dir / "val_metrics.csv", {"step": step, "epoch": epoch, **vm})
            if vm["loss"] < best_val:
                best_val = vm["loss"]
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, step, best_val, cfg)

        if epoch % cfg["training"]["save_every_epochs"] == 0:
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, step, best_val, cfg)
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, step, best_val, cfg)

        if epoch % cfg["training"]["sample_every_epochs"] == 0:
            save_samples(model, sample_batch, cfg, device, run_dir / "samples", epoch)

        logger.info("epoch=%d duration_sec=%.1f best_val=%.6f", epoch, time.time() - t0, best_val)

    if writer is not None:
        writer.close()
    logger.info("Training complete. Best checkpoint: %s", run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()
