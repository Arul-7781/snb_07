from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlockNoNorm1d(nn.Module):
    """Dilated residual block for multi-scale temporal context."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_normal_(conv.weight)
            conv.weight.data.mul_(0.1)
            nn.init.zeros_(conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.conv2(self.activation(self.conv1(inputs)))


class RedundancyPredictor1d(nn.Module):
    """Predicts the discarded INN residual with a multi-scale temporal U-Net.

    The encoder gives the predictor a long temporal receptive field while the skip
    connections preserve fine waveform detail needed by the INN inverse.
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
        block_count = max(1, num_blocks // 3)
        self.input = nn.Conv1d(channels, hidden_channels, kernel_size, padding=padding)
        self.encoder = nn.Sequential(
            *(
                ResidualBlockNoNorm1d(
                    hidden_channels,
                    kernel_size,
                    dilation=2 ** (index % 6),
                )
                for index in range(block_count)
            )
        )
        self.down1 = nn.Conv1d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        self.encoder2 = nn.Sequential(
            *(
                ResidualBlockNoNorm1d(
                    hidden_channels * 2,
                    kernel_size,
                    dilation=2 ** (index % 5),
                )
                for index in range(block_count)
            )
        )
        self.down2 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            *(
                ResidualBlockNoNorm1d(
                    hidden_channels * 4,
                    kernel_size,
                    dilation=2 ** (index % 4),
                )
                for index in range(block_count)
            )
        )
        self.fuse2 = nn.Conv1d(
            hidden_channels * 6, hidden_channels * 2, kernel_size, padding=padding
        )
        self.fuse1 = nn.Conv1d(
            hidden_channels * 3, hidden_channels, kernel_size, padding=padding
        )
        # LF-VSN needs no projection because its feature width equals the latent width.
        # Zero-initializing ours makes training start from a zero latent.
        self.output = nn.Conv1d(hidden_channels, channels, kernel_size, padding=padding)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, stego_coefficients: torch.Tensor) -> torch.Tensor:
        first = self.encoder(self.input(stego_coefficients))
        second = self.encoder2(self.down1(first))
        bottleneck = self.bottleneck(self.down2(second))
        up_second = F.interpolate(
            bottleneck, size=second.shape[-1], mode="linear", align_corners=False
        )
        up_second = self.fuse2(torch.cat((up_second, second), dim=1))
        up_first = F.interpolate(
            up_second, size=first.shape[-1], mode="linear", align_corners=False
        )
        up_first = self.fuse1(torch.cat((up_first, first), dim=1))
        return self.output(up_first)


class SecretRecoveryDecoder1d(nn.Module):
    """Direct stego-to-secret decoder used when predicted residual inversion is unstable."""

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
        self.input = nn.Conv1d(1, hidden_channels, kernel_size, padding=padding)
        self.blocks = nn.Sequential(
            *(
                ResidualBlockNoNorm1d(
                    hidden_channels,
                    kernel_size,
                    dilation=2 ** (index % 10),
                )
                for index in range(num_blocks)
            )
        )
        self.output = nn.Conv1d(hidden_channels, channels, kernel_size, padding=padding)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(stego)))
