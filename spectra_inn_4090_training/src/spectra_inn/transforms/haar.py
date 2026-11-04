from __future__ import annotations

import math

import torch
from torch import nn


class HaarPacket1d(nn.Module):
    """Multi-level orthonormal Haar packet transform with an analytic inverse."""

    def __init__(self, levels: int) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = int(levels)
        self.scale = 1.0 / math.sqrt(2.0)

    def analysis(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(waveform.shape)}")
        coefficients = waveform
        for _ in range(self.levels):
            if coefficients.shape[-1] % 2:
                raise ValueError("Time dimension must be divisible by 2**levels")
            even = coefficients[..., 0::2]
            odd = coefficients[..., 1::2]
            low = (even + odd) * self.scale
            high = (even - odd) * self.scale
            coefficients = torch.cat((low, high), dim=1)
        return coefficients

    def synthesis(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(coefficients.shape)}")
        waveform = coefficients
        for _ in range(self.levels):
            if waveform.shape[1] % 2:
                raise ValueError("Channel dimension must be divisible by 2**levels")
            low, high = waveform.chunk(2, dim=1)
            even = (low + high) * self.scale
            odd = (low - high) * self.scale
            waveform = torch.stack((even, odd), dim=-1).flatten(start_dim=-2)
        return waveform

    def forward(self, waveform: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return self.synthesis(waveform) if reverse else self.analysis(waveform)

