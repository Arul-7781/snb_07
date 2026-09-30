from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def command_first_line(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else f"exit={result.returncode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the SPECTRA-INN training environment")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDA is unavailable",
    )
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    version = sys.version_info
    python_ok = version.major == 3 and version.minor in (11, 12)
    print(f"Python: {sys.version.split()[0]} {'OK' if python_ok else 'UNSUPPORTED'}")
    if not python_ok:
        failures.append("Use an isolated Python 3.11 environment (3.12 is secondary)")

    for executable in ("ffmpeg", "ffprobe"):
        ok, detail = command_first_line([executable, "-version"])
        print(f"{executable}: {detail} {'OK' if ok else 'MISSING'}")
        if not ok:
            failures.append(f"{executable} is required")

    disk = shutil.disk_usage(Path.cwd())
    free_gb = disk.free / 1024**3
    print(f"Free disk: {free_gb:.1f} GiB")
    if free_gb < args.min_free_gb:
        warnings.append(f"Less than the recommended {args.min_free_gb:.0f} GiB is free")

    for package in ("numpy", "yaml", "soundfile", "scipy", "tensorboard"):
        try:
            module = importlib.import_module(package)
            detail = getattr(module, "__version__", "installed")
            print(f"{package}: {detail} OK")
        except Exception as exc:
            print(f"{package}: MISSING ({type(exc).__name__})")
            failures.append(f"Python package '{package}' is not installed")

    try:
        import torch

        print(f"torch: {torch.__version__} OK")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")
        if cuda_ok:
            properties = torch.cuda.get_device_properties(0)
            memory_gb = properties.total_memory / 1024**3
            print(
                f"GPU: {properties.name}; VRAM: {memory_gb:.1f} GiB; "
                f"CUDA runtime: {torch.version.cuda}"
            )
            if memory_gb < 18:
                warnings.append("Less than 18 GiB VRAM; reduce the baseline micro-batch")
        elif not args.allow_cpu:
            failures.append("PyTorch cannot see a CUDA GPU")
    except Exception as exc:
        print(f"torch: MISSING ({type(exc).__name__})")
        failures.append("Install a CUDA-enabled PyTorch build before the project dependencies")

    smi_ok, smi_detail = command_first_line(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    print(f"nvidia-smi: {smi_detail} {'OK' if smi_ok else 'FAILED'}")
    if not smi_ok and not args.allow_cpu:
        failures.append("nvidia-smi cannot communicate with the NVIDIA driver")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if failures:
        print("\nEnvironment is not ready:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nEnvironment is ready for the requested mode.")


if __name__ == "__main__":
    main()
