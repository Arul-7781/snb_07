import torch
from audio_inn.model import AudioINN, squeeze1d, unsqueeze1d


def test_squeeze_roundtrip():
    x = torch.randn(2, 2, 16000)
    y = squeeze1d(x, 8)
    z = unsqueeze1d(y, 8)
    assert torch.equal(x, z)


def test_true_latent_invertibility():
    torch.manual_seed(0)
    model = AudioINN(squeeze_factor=8, num_blocks=3, hidden_channels=16)
    c = torch.randn(2, 1, 16000) * 0.1
    s = torch.randn(2, 1, 16000) * 0.1
    y, r = model.encode(c, s)
    c2, s2 = model.decode(y, r)
    assert torch.max(torch.abs(c - c2)).item() < 2e-5
    assert torch.max(torch.abs(s - s2)).item() < 2e-5
