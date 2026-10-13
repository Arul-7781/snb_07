from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectra_inn.models.coupling import AffineCouplingBlock1d
from spectra_inn.models.redundancy import RedundancyPredictor1d
from spectra_inn.transforms.haar import HaarPacket1d

LATENT_MODES = ("gaussian", "predicted")


class CleanAudioINN(nn.Module):
    """Stage-1 audio-in-audio INN with explicit residual output."""

    def __init__(
        self,
        audio_channels: int,
        wavelet_levels: int,
        num_blocks: int,
        hidden_channels: int,
        subnet_layers: int,
        clamp: float,
        kernel_size: int,
        latent_mode: str = "gaussian",
        predictor_hidden_channels: int = 64,
        predictor_blocks: int = 8,
        predictor_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode must be one of {LATENT_MODES}, got {latent_mode!r}")
        self.latent_mode = latent_mode
        self.audio_channels = int(audio_channels)
        self.wavelet_levels = int(wavelet_levels)
        self.transform = HaarPacket1d(wavelet_levels)
        coefficient_channels = audio_channels * (2**wavelet_levels)
        self.blocks = nn.ModuleList(
            AffineCouplingBlock1d(
                channels=coefficient_channels,
                hidden_channels=hidden_channels,
                subnet_layers=subnet_layers,
                clamp=clamp,
                kernel_size=kernel_size,
            )
            for _ in range(num_blocks)
        )
        self.predictor: RedundancyPredictor1d | None = None
        if latent_mode == "predicted":
            self.predictor = RedundancyPredictor1d(
                channels=coefficient_channels,
                hidden_channels=predictor_hidden_channels,
                num_blocks=predictor_blocks,
                kernel_size=predictor_kernel_size,
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CleanAudioINN:
        model = config["model"]
        predictor = model.get("predictor") or {}
        return cls(
            audio_channels=int(config["data"]["channels"]),
            wavelet_levels=int(model["wavelet_levels"]),
            num_blocks=int(model["num_blocks"]),
            hidden_channels=int(model["hidden_channels"]),
            subnet_layers=int(model["subnet_layers"]),
            clamp=float(model["clamp"]),
            kernel_size=int(model["kernel_size"]),
            latent_mode=str(model.get("latent_mode", "gaussian")),
            predictor_hidden_channels=int(predictor.get("hidden_channels", 64)),
            predictor_blocks=int(predictor.get("num_blocks", 8)),
            predictor_kernel_size=int(predictor.get("kernel_size", 3)),
        )

    def forward(
        self, cover: torch.Tensor, secret: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if cover.shape != secret.shape:
            raise ValueError(
                f"Stage-1 cover and secret shapes must match, got {cover.shape} and {secret.shape}"
            )
        first = self.transform.analysis(cover)
        second = self.transform.analysis(secret)
        log_scales: list[torch.Tensor] = []
        for block in self.blocks:
            first, second, log_scale = block(first, second)
            log_scales.append(log_scale)
        stego = self.transform.synthesis(first)
        residual = self.transform.synthesis(second)
        diagnostics = {
            "mean_abs_log_scale": torch.stack([value.abs().mean() for value in log_scales]).mean(),
            "max_abs_log_scale": torch.stack([value.abs().amax() for value in log_scales]).amax(),
        }
        return stego, residual, diagnostics

    def inverse(
        self, stego: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stego.shape != latent.shape:
            raise ValueError(
                f"Stego and latent shapes must match, got {stego.shape} and {latent.shape}"
            )
        return self._inverse_coefficients(
            self.transform.analysis(stego), self.transform.analysis(latent)
        )

    def _inverse_coefficients(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in reversed(self.blocks):
            first, second = block.inverse(first, second)
        cover = self.transform.synthesis(first)
        secret = self.transform.synthesis(second)
        return cover, secret

    def blind_inverse(
        self,
        stego: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.randn(
            stego.shape,
            device=stego.device,
            dtype=stego.dtype,
            generator=generator,
        )
        cover, secret = self.inverse(stego, latent)
        return cover, secret, latent

    def predicted_inverse(
        self, stego: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.predictor is None:
            raise RuntimeError("predicted_inverse requires model.latent_mode='predicted'")
        first = self.transform.analysis(stego)
        second = self.predictor(first)
        cover, secret = self._inverse_coefficients(first, second)
        return cover, secret, self.transform.synthesis(second)

    def decode(
        self,
        stego: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover (cover, secret, latent) from stego alone using the configured latent."""
        if self.latent_mode == "predicted":
            return self.predicted_inverse(stego)
        return self.blind_inverse(stego, generator=generator)
