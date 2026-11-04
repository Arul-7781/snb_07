from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates a required invariant."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = deepcopy(loaded)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "model", "loss", "training")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    loss = config["loss"]

    if loss.get("waveform_mode", "charbonnier") not in ("charbonnier", "normalized_mse"):
        raise ConfigError("loss.waveform_mode must be 'charbonnier' or 'normalized_mse'")
    try:
        power_floor = float(loss.get("nmse_power_floor", 1e-6))
    except (TypeError, ValueError):
        raise ConfigError("loss.nmse_power_floor must be finite and positive") from None
    if not math.isfinite(power_floor) or power_floor <= 0:
        raise ConfigError("loss.nmse_power_floor must be finite and positive")
    auxiliary_weight = float(loss.get("latent_prediction", 0.0))
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ConfigError("loss.latent_prediction must be finite and nonnegative")

    sample_rate = int(data["sample_rate"])
    channels = int(data["channels"])
    segment_samples = round(float(data["segment_seconds"]) * sample_rate)
    levels = int(model["wavelet_levels"])
    divisor = 2**levels

    if sample_rate <= 0 or channels <= 0 or segment_samples <= 0:
        raise ConfigError("Sample rate, channels, and segment duration must be positive")
    if segment_samples % divisor:
        raise ConfigError(
            f"Segment length {segment_samples} must be divisible by 2**wavelet_levels={divisor}"
        )
    if int(model["num_blocks"]) < 1:
        raise ConfigError("model.num_blocks must be positive")
    if float(model["clamp"]) <= 0:
        raise ConfigError("model.clamp must be positive")
    latent_mode = model.get("latent_mode", "gaussian")
    if latent_mode not in ("gaussian", "predicted"):
        raise ConfigError("model.latent_mode must be 'gaussian' or 'predicted'")
    predictor = model.get("predictor") or {}
    if predictor.get("architecture", "flat") not in ("flat", "unet"):
        raise ConfigError("model.predictor.architecture must be 'flat' or 'unet'")
    if predictor.get("architecture") == "unet" and int(predictor.get("num_blocks", 8)) < 3:
        raise ConfigError("U-Net RPM needs at least three blocks")
    if predictor.get("architecture") == "unet" and segment_samples // divisor < 4:
        raise ConfigError("U-Net RPM needs at least four Haar positions")
    if auxiliary_weight and latent_mode != "predicted":
        raise ConfigError("latent_prediction requires predicted latent mode")
    if latent_mode == "predicted":
        if int(predictor.get("num_blocks", 8)) < 1:
            raise ConfigError("model.predictor.num_blocks must be positive")
        if int(predictor.get("hidden_channels", 64)) < 1:
            raise ConfigError("model.predictor.hidden_channels must be positive")
        if int(predictor.get("kernel_size", 3)) % 2 == 0:
            raise ConfigError("model.predictor.kernel_size must be odd")
        dilations = predictor.get("dilations")
        if dilations is not None and (
            not isinstance(dilations, (list, tuple))
            or len(dilations) != int(predictor.get("num_blocks", 8))
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in dilations
            )
        ):
            raise ConfigError(
                "model.predictor.dilations must contain one positive integer per predictor block"
            )
    overfit_pairs = data.get("overfit_pairs")
    if overfit_pairs is not None and (
        not isinstance(overfit_pairs, int)
        or isinstance(overfit_pairs, bool)
        or overfit_pairs < int(training["batch_size"])
    ):
        raise ConfigError("data.overfit_pairs must be an integer at least as large as batch_size")
    if int(training["batch_size"]) < 1 or int(training["accumulation_steps"]) < 1:
        raise ConfigError("Batch size and accumulation steps must be positive")
    if data.get("validation_pairing", "legacy") not in ("legacy", "cross_speaker"):
        raise ConfigError("data.validation_pairing must be legacy or cross_speaker")
    if data.get("validation_subset_size") is not None and int(data["validation_subset_size"]) < 1:
        raise ConfigError("data.validation_subset_size must be positive")
    if int(training["max_steps"]) < 1:
        raise ConfigError("training.max_steps must be positive")
    max_hours = training.get("max_hours")
    if max_hours is not None and float(max_hours) <= 0:
        raise ConfigError("training.max_hours must be positive or null")


def segment_samples(config: dict[str, Any]) -> int:
    return round(float(config["data"]["segment_seconds"]) * int(config["data"]["sample_rate"]))
