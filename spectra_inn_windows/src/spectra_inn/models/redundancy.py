from __future__ import annotations

import torch
from torch import nn


class ResidualBlockNoNorm1d(nn.Module):
    """LF-VSN residual block (conv, LeakyReLU, conv, skip) without normalization."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_normal_(conv.weight)
            conv.weight.data.mul_(0.1)
            nn.init.zeros_(conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.conv2(self.activation(self.conv1(inputs)))


class RedundancyPredictor1d(nn.Module):
    """Predicts the discarded INN residual from stego Haar coefficients.

    Audio port of the LF-VSN redundancy prediction module (RPM). It replaces the
    Gaussian latent at decode time and is trained only through the recovery losses.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        padding = kernel_size // 2
        self.input = nn.Conv1d(channels, hidden_channels, kernel_size, padding=padding)
        self.blocks = nn.Sequential(
            *(ResidualBlockNoNorm1d(hidden_channels, kernel_size) for _ in range(num_blocks))
        )
        # LF-VSN needs no projection because its feature width equals the latent width.
        # Zero-initializing ours makes training start from a zero latent.
        self.output = nn.Conv1d(hidden_channels, channels, kernel_size, padding=padding)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, stego_coefficients: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(stego_coefficients)))
