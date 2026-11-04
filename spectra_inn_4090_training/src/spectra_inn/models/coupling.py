from __future__ import annotations

import torch
from torch import nn


class ResidualSubNet1d(nn.Module):
    """Compact non-invertible conditioner used inside an invertible block."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if layers < 2:
            raise ValueError("layers must be at least two")

        self.input = nn.Conv1d(channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList()
        for index in range(layers - 2):
            dilation = 2 ** (index % 4)
            padding = dilation * (kernel_size // 2)
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(1, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                )
            )
        self.output = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size, padding=kernel_size // 2),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class AffineCouplingBlock1d(nn.Module):
    """Additive/affine two-stream coupling with bounded log scale."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        subnet_layers: int,
        clamp: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.clamp = float(clamp)
        kwargs = {
            "channels": channels,
            "hidden_channels": hidden_channels,
            "layers": subnet_layers,
            "kernel_size": kernel_size,
        }
        self.phi = ResidualSubNet1d(**kwargs)
        self.rho = ResidualSubNet1d(**kwargs)
        self.eta = ResidualSubNet1d(**kwargs)

    def _log_scale(self, first: torch.Tensor) -> torch.Tensor:
        return self.clamp * torch.tanh(self.rho(first))

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_out = first + self.phi(second)
        log_scale = self._log_scale(first_out)
        second_out = second * torch.exp(log_scale) + self.eta(first_out)
        return first_out, second_out, log_scale

    def inverse(
        self, first_out: torch.Tensor, second_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = self._log_scale(first_out)
        second = (second_out - self.eta(first_out)) * torch.exp(-log_scale)
        first = first_out - self.phi(second)
        return first, second

