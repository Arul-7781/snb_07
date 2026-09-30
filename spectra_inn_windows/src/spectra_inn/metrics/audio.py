from __future__ import annotations

import torch


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    signal = reference.square().mean(dim=(-2, -1))
    noise = (reference - estimate).square().mean(dim=(-2, -1))
    return (10.0 * torch.log10((signal + epsilon) / (noise + epsilon))).mean()

