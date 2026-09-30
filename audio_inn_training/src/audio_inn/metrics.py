from __future__ import annotations

import torch


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = reference - estimate
    p_signal = reference.pow(2).mean(dim=(-1, -2))
    p_noise = noise.pow(2).mean(dim=(-1, -2))
    return 10.0 * torch.log10((p_signal + eps) / (p_noise + eps))


def si_sdr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ref = reference.flatten(1)
    est = estimate.flatten(1)
    ref = ref - ref.mean(dim=1, keepdim=True)
    est = est - est.mean(dim=1, keepdim=True)
    alpha = (est * ref).sum(dim=1, keepdim=True) / (ref.pow(2).sum(dim=1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10.0 * torch.log10((target.pow(2).sum(dim=1) + eps) / (noise.pow(2).sum(dim=1) + eps))
