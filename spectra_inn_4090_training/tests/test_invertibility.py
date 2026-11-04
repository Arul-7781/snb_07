from __future__ import annotations

import torch

from spectra_inn.models.clean_inn import CleanAudioINN


def test_clean_inn_exact_residual_round_trip() -> None:
    model = CleanAudioINN(
        audio_channels=1,
        wavelet_levels=3,
        num_blocks=2,
        hidden_channels=16,
        subnet_layers=3,
        clamp=1.2,
        kernel_size=3,
    )
    cover = torch.randn(2, 1, 4096) * 0.1
    secret = torch.randn(2, 1, 4096) * 0.1
    stego, residual, diagnostics = model(cover, secret)
    recovered_cover, recovered_secret = model.inverse(stego, residual)
    assert stego.shape == cover.shape
    assert residual.shape == secret.shape
    assert diagnostics["max_abs_log_scale"].ndim == 0
    assert torch.allclose(cover, recovered_cover, atol=3e-6, rtol=1e-5)
    assert torch.allclose(secret, recovered_secret, atol=3e-6, rtol=1e-5)


def test_blind_inverse_preserves_shapes() -> None:
    model = CleanAudioINN(
        audio_channels=1,
        wavelet_levels=2,
        num_blocks=1,
        hidden_channels=8,
        subnet_layers=2,
        clamp=1.0,
        kernel_size=3,
    )
    cover = torch.randn(1, 1, 1024)
    secret = torch.randn(1, 1, 1024)
    stego, _, _ = model(cover, secret)
    recovered_cover, recovered_secret, latent = model.blind_inverse(stego)
    assert recovered_cover.shape == cover.shape
    assert recovered_secret.shape == secret.shape
    assert latent.shape == secret.shape

