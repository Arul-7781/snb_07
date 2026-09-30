import torch
import pytest
from audio_inn.losses import multi_resolution_stft_loss, latent_prior_loss


def test_stft_zero_for_identical():
    x = torch.randn(2,1,16000)
    loss = multi_resolution_stft_loss(x, x, [256,512], [4,4])
    assert float(loss) < 1e-6


def test_prior_finite():
    r = torch.randn(4,1,16000)
    assert torch.isfinite(latent_prior_loss(r))


def test_stft_config_lengths_must_match():
    x = torch.randn(1, 1, 1024)
    with pytest.raises(ValueError, match="equal lengths"):
        multi_resolution_stft_loss(x, x, [256, 512], [4])
