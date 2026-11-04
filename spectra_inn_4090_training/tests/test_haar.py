from __future__ import annotations

import torch

from spectra_inn.transforms.haar import HaarPacket1d


def test_haar_packet_round_trip() -> None:
    transform = HaarPacket1d(levels=3)
    waveform = torch.randn(2, 2, 4096)
    coefficients = transform.analysis(waveform)
    reconstructed = transform.synthesis(coefficients)
    assert coefficients.shape == (2, 16, 512)
    assert torch.allclose(waveform, reconstructed, atol=2e-6, rtol=1e-6)

