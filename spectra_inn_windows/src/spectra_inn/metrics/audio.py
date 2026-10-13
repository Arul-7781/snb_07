from __future__ import annotations

import torch


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    signal = reference.square().mean(dim=(-2, -1))
    noise = (reference - estimate).square().mean(dim=(-2, -1))
    return (10.0 * torch.log10((signal + epsilon) / (noise + epsilon))).mean()



def si_sdr_db(
    reference: torch.Tensor, estimate: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    energy = reference.square().sum(dim=-1, keepdim=True)
    target = (estimate * reference).sum(dim=-1, keepdim=True) / (energy + epsilon) * reference
    noise = estimate - target
    ratio = target.square().sum(dim=-1) / (noise.square().sum(dim=-1) + epsilon)
    return (10.0 * torch.log10(ratio + epsilon)).mean()
