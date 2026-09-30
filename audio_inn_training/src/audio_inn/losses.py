from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_resolution_stft_loss(x: torch.Tensor, y: torch.Tensor, fft_sizes: list[int], hop_factors: list[int]) -> torch.Tensor:
    if not fft_sizes or len(fft_sizes) != len(hop_factors):
        raise ValueError("fft_sizes and hop_factors must be non-empty and have equal lengths")
    # FFT/STFT kernels are most reliable in float32 even when AMP is enabled.
    x = x.squeeze(1).float()
    y = y.squeeze(1).float()
    total = x.new_tensor(0.0)
    for n_fft, hop_factor in zip(fft_sizes, hop_factors):
        if n_fft <= 0 or hop_factor <= 0:
            raise ValueError("FFT sizes and hop factors must be positive")
        hop = n_fft // hop_factor
        if hop == 0:
            raise ValueError(f"hop factor {hop_factor} is too large for FFT size {n_fft}")
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
        total = total + F.l1_loss(torch.abs(X), torch.abs(Y))
    return total / max(len(fft_sizes), 1)


def latent_prior_loss(r: torch.Tensor) -> torch.Tensor:
    mean = r.mean()
    std = r.std(unbiased=False)
    return mean.square() + (std - 1.0).square()


def compute_losses(cover, secret, stego, latent, recovered_secret, cfg):
    stft_cfg = cfg["stft"]
    w = cfg["loss"]
    stego_l1 = F.l1_loss(stego, cover)
    stego_stft = multi_resolution_stft_loss(cover, stego, stft_cfg["fft_sizes"], stft_cfg["hop_factors"])
    secret_l1 = F.l1_loss(recovered_secret, secret)
    secret_stft = multi_resolution_stft_loss(secret, recovered_secret, stft_cfg["fft_sizes"], stft_cfg["hop_factors"])
    prior = latent_prior_loss(latent)
    total = (
        w["stego_l1"] * stego_l1
        + w["stego_stft"] * stego_stft
        + w["secret_l1"] * secret_l1
        + w["secret_stft"] * secret_stft
        + w["latent_prior"] * prior
    )
    return total, {
        "loss": total.detach(),
        "stego_l1": stego_l1.detach(),
        "stego_stft": stego_stft.detach(),
        "secret_l1": secret_l1.detach(),
        "secret_stft": secret_stft.detach(),
        "latent_prior": prior.detach(),
    }
