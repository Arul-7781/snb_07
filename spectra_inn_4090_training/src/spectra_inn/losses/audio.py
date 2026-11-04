from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def charbonnier(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((first - second).square() + epsilon**2).mean()


def normalized_mse(
    reference: torch.Tensor, estimate: torch.Tensor, power_floor: float = 1e-6
) -> torch.Tensor:
    """Per-example waveform error relative to reference power, computed in float32.

    This is amplitude-sensitive NMSE, not scale-invariant SDR. A power floor prevents
    near-silent examples from producing unbounded weights. Mean NMSE and mean dB SNR
    are related objectives, but are not identical averages.
    """
    if not math.isfinite(power_floor) or power_floor <= 0:
        raise ValueError("power_floor must be finite and positive")
    if reference.ndim != 3 or reference.shape != estimate.shape:
        raise ValueError("normalized_mse expects matching [batch, channels, samples] tensors")
    reference = reference.float()
    estimate = estimate.float()
    power = reference.square().mean(dim=(-2, -1)).clamp_min(power_floor)
    error = (reference - estimate).square().mean(dim=(-2, -1))
    return (error / power).mean()


def multi_resolution_stft_loss(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    windows: tuple[int, ...],
) -> torch.Tensor:
    # cuFFT and log magnitudes are more stable in float32 than under AMP float16.
    reference = reference.float()
    estimate = estimate.float()
    batch, channels, samples = reference.shape
    reference_flat = reference.reshape(batch * channels, samples)
    estimate_flat = estimate.reshape(batch * channels, samples)
    total = reference.new_zeros(())
    for window_length in windows:
        if window_length > samples:
            continue
        hop_length = window_length // 4
        window = torch.hann_window(window_length, device=reference.device, dtype=reference.dtype)
        reference_stft = torch.stft(
            reference_flat,
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
        )
        estimate_stft = torch.stft(
            estimate_flat,
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
        )
        reference_magnitude = reference_stft.abs().clamp_min(1e-7)
        estimate_magnitude = estimate_stft.abs().clamp_min(1e-7)
        log_magnitude = F.l1_loss(estimate_magnitude.log(), reference_magnitude.log())
        difference_norm = torch.linalg.vector_norm(estimate_magnitude - reference_magnitude)
        reference_norm = torch.linalg.vector_norm(reference_magnitude)
        spectral_convergence = difference_norm / (reference_norm + 1e-7)
        total = total + log_magnitude + spectral_convergence
    return total / max(1, len(windows))


def latent_prior_loss(latent: torch.Tensor) -> torch.Tensor:
    mean = latent.mean(dim=(-2, -1))
    variance = latent.var(dim=(-2, -1), unbiased=False)
    return mean.square().mean() + (variance - 1.0).square().mean()


class StageOneLoss(nn.Module):
    """Clean Stage-1 loss; robust and psychoacoustic terms are deliberately absent."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        loss = config["loss"]
        self.waveform_mode = str(loss.get("waveform_mode", "charbonnier"))
        if self.waveform_mode not in ("charbonnier", "normalized_mse"):
            raise ValueError("waveform_mode must be 'charbonnier' or 'normalized_mse'")
        self.nmse_power_floor = float(loss.get("nmse_power_floor", 1e-6))
        if not math.isfinite(self.nmse_power_floor) or self.nmse_power_floor <= 0:
            raise ValueError("nmse_power_floor must be finite and positive")
        self.weights = {
            "stego_time": float(loss["stego_time"]),
            "stego_stft": float(loss["stego_stft"]),
            "secret_time": float(loss["secret_time"]),
            "secret_stft": float(loss["secret_stft"]),
            "recovered_cover": float(loss["recovered_cover"]),
            "latent_prior": float(loss["latent_prior"]),
            "scale_stability": float(loss["scale_stability"]),
            "latent_prediction": float(loss.get("latent_prediction", 0.0)),
        }
        self.stft_windows = tuple(int(value) for value in loss["stft_windows"])

    def waveform_loss(self, reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if self.waveform_mode == "normalized_mse":
            return normalized_mse(reference, estimate, self.nmse_power_floor)
        return charbonnier(reference, estimate)

    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor,
        stego: torch.Tensor,
        residual: torch.Tensor,
        recovered_cover: torch.Tensor,
        recovered_secret: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
        predicted_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = {
            "stego_time": self.waveform_loss(cover, stego),
            "stego_stft": multi_resolution_stft_loss(cover, stego, self.stft_windows),
            "secret_time": self.waveform_loss(secret, recovered_secret),
            "secret_stft": multi_resolution_stft_loss(secret, recovered_secret, self.stft_windows),
            "recovered_cover": self.waveform_loss(cover, recovered_cover),
            "latent_prior": latent_prior_loss(residual),
            "scale_stability": diagnostics["mean_abs_log_scale"],
        }
        if self.weights["latent_prediction"]:
            if predicted_residual is None:
                raise ValueError("latent_prediction requires a predicted residual")
            components["latent_prediction"] = normalized_mse(
                residual.detach(), predicted_residual, self.nmse_power_floor
            )
        else:
            components["latent_prediction"] = stego.new_zeros(())
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components
