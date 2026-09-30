from __future__ import annotations

import torch
from torch import nn


def squeeze1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, t = x.shape
    if t % factor != 0:
        raise ValueError(f"Length {t} must be divisible by squeeze factor {factor}")
    x = x.view(b, c, t // factor, factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.view(b, c * factor, t // factor)


def unsqueeze1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, t = x.shape
    if c % factor != 0:
        raise ValueError(f"Channels {c} must be divisible by squeeze factor {factor}")
    x = x.view(b, c // factor, factor, t)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.view(b, c // factor, t * factor)


class CouplingSubnet(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels * 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.net(x).chunk(2, dim=1)
        return s, t


class AffineCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, clamp: float = 1.5):
        super().__init__()
        if channels % 2:
            raise ValueError("Coupling block needs an even channel count")
        self.half = channels // 2
        self.subnet = CouplingSubnet(self.half, hidden)
        self.clamp = clamp

    def _st(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.subnet(a)
        s = self.clamp * torch.tanh(s / max(self.clamp, 1e-6))
        return s, t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[:, : self.half], x[:, self.half :]
        s, t = self._st(a)
        b = b * torch.exp(s) + t
        y = torch.cat([a, b], dim=1)
        return torch.flip(y, dims=[1])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.flip(y, dims=[1])
        a, b = y[:, : self.half], y[:, self.half :]
        s, t = self._st(a)
        b = (b - t) * torch.exp(-s)
        return torch.cat([a, b], dim=1)


class AudioINN(nn.Module):
    def __init__(self, squeeze_factor: int = 8, num_blocks: int = 8, hidden_channels: int = 64, clamp: float = 1.5):
        super().__init__()
        self.squeeze_factor = squeeze_factor
        channels = 2 * squeeze_factor
        self.blocks = nn.ModuleList([
            AffineCouplingBlock(channels, hidden_channels, clamp=clamp)
            for _ in range(num_blocks)
        ])

    def encode(self, cover: torch.Tensor, secret: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cover, secret], dim=1)
        h = squeeze1d(x, self.squeeze_factor)
        for block in self.blocks:
            h = block(h)
        out = unsqueeze1d(h, self.squeeze_factor)
        return out[:, 0:1], out[:, 1:2]

    def decode(self, stego: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([stego, latent], dim=1)
        h = squeeze1d(x, self.squeeze_factor)
        for block in reversed(self.blocks):
            h = block.inverse(h)
        out = unsqueeze1d(h, self.squeeze_factor)
        return out[:, 0:1], out[:, 1:2]
