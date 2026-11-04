from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlockNoNorm1d(nn.Module):
    """LF-VSN-style residual block with optional dilation and no normalization."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation < 1:
            raise ValueError("dilation must be a positive integer")
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
        dilations: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.dilations = tuple(dilations) if dilations is not None else (1,) * num_blocks
        if len(self.dilations) != num_blocks or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.dilations
        ):
            raise ValueError("dilations must contain one positive integer per predictor block")
        # Input/output convolutions are undilated; each residual block has two convolutions.
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (1 + sum(self.dilations))
        padding = kernel_size // 2
        self.input = nn.Conv1d(channels, hidden_channels, kernel_size, padding=padding)
        self.blocks = nn.Sequential(
            *(
                ResidualBlockNoNorm1d(hidden_channels, kernel_size, dilation)
                for dilation in self.dilations
            )
        )
        # LF-VSN needs no projection because its feature width equals the latent width.
        # Zero-initializing ours makes training start from a zero latent.
        self.output = nn.Conv1d(hidden_channels, channels, kernel_size, padding=padding)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, stego_coefficients: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(stego_coefficients)))


class MultiScaleRedundancyPredictor1d(nn.Module):
    """Three-scale RPM; still predicts only the latent, never the secret directly.

    This is a local implementation of the U-Net idea in the external iteration notes,
    not a claim to reproduce that branch's exact unpublished implementation/weights.
    Width stays constant across scales to limit capacity and VRAM growth.
    """

    def __init__(
        self, channels: int, hidden_channels: int, num_blocks: int, kernel_size: int,
        dilations: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0 or num_blocks < 3:
            raise ValueError("U-Net RPM needs a positive odd kernel and at least three blocks")
        counts = [num_blocks // 3 + int(index < num_blocks % 3) for index in range(3)]
        default = [2 ** (index % 3 + level) for level, count in enumerate(counts)
                   for index in range(count)]
        self.dilations = tuple(default if dilations is None else dilations)
        if len(self.dilations) != num_blocks or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.dilations
        ):
            raise ValueError("dilations must contain one positive integer per predictor block")
        padding = kernel_size // 2
        self.input = nn.Conv1d(channels, hidden_channels, kernel_size, padding=padding)
        stages = []
        offset = 0
        receptive_field = kernel_size
        for level, count in enumerate(counts):
            ds = self.dilations[offset : offset + count]
            stages.append(nn.Sequential(*(
                ResidualBlockNoNorm1d(hidden_channels, kernel_size, dilation) for dilation in ds
            )))
            receptive_field += 2 * (kernel_size - 1) * sum(ds) * 2**level
            if level < 2:
                receptive_field += 3 * 2**level  # kernel-4 stride-2 downsampling
            offset += count
        self.stages = nn.ModuleList(stages)
        self.down = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 4, stride=2, padding=1) for _ in range(2)
        ])
        self.fuse = nn.ModuleList([
            nn.Conv1d(2 * hidden_channels, hidden_channels, kernel_size, padding=padding)
            for _ in range(2)
        ])
        self.activation = nn.LeakyReLU(0.2)
        self.output = nn.Conv1d(hidden_channels, channels, kernel_size, padding=padding)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Conservative bound: interpolation can involve two coarse neighboring positions.
        self.receptive_field = receptive_field + 6 + 4 * (kernel_size - 1)
        self.receptive_field_kind = "upper_bound"

    def forward(self, stego_coefficients: torch.Tensor) -> torch.Tensor:
        if stego_coefficients.shape[-1] < 4:
            raise ValueError("U-Net RPM needs at least four coefficient positions")
        fine = self.stages[0](self.input(stego_coefficients))
        middle = self.stages[1](self.activation(self.down[0](fine)))
        coarse = self.stages[2](self.activation(self.down[1](middle)))
        up = F.interpolate(coarse, size=middle.shape[-1], mode="linear", align_corners=False)
        up = self.activation(self.fuse[0](torch.cat((up, middle), dim=1)))
        up = F.interpolate(up, size=fine.shape[-1], mode="linear", align_corners=False)
        return self.output(self.activation(self.fuse[1](torch.cat((up, fine), dim=1))))
