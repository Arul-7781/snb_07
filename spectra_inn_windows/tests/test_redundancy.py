from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from spectra_inn import evaluate
from spectra_inn.config import ConfigError, load_config, validate_config
from spectra_inn.data.manifest import build_manifest
from spectra_inn.losses.audio import StageOneLoss
from spectra_inn.models.clean_inn import CleanAudioINN
from spectra_inn.models.redundancy import RedundancyPredictor1d
from spectra_inn.train import compute_batch, latent_comparison

ROOT = Path(__file__).resolve().parents[1]


def _predicted_model() -> CleanAudioINN:
    return CleanAudioINN(
        audio_channels=1,
        wavelet_levels=3,
        num_blocks=2,
        hidden_channels=16,
        subnet_layers=3,
        clamp=1.2,
        kernel_size=3,
        latent_mode="predicted",
        predictor_hidden_channels=16,
        predictor_blocks=2,
        predictor_kernel_size=3,
    )


def test_predictor_starts_from_zero_latent() -> None:
    predictor = RedundancyPredictor1d(channels=8, hidden_channels=16, num_blocks=2, kernel_size=3)
    coefficients = torch.randn(2, 8, 128)
    latent = predictor(coefficients)
    assert latent.shape == coefficients.shape
    assert torch.count_nonzero(latent) == 0


def test_predicted_decode_shapes_and_exact_cycle() -> None:
    model = _predicted_model()
    cover = torch.randn(2, 1, 4096) * 0.1
    secret = torch.randn(2, 1, 4096) * 0.1
    stego, residual, _ = model(cover, secret)
    recovered_cover, recovered_secret, latent = model.decode(stego)
    assert recovered_cover.shape == cover.shape
    assert recovered_secret.shape == secret.shape
    assert latent.shape == residual.shape
    exact_cover, exact_secret = model.inverse(stego, residual)
    assert torch.allclose(cover, exact_cover, atol=3e-6, rtol=1e-5)
    assert torch.allclose(secret, exact_secret, atol=3e-6, rtol=1e-5)


def test_gaussian_model_has_no_predictor() -> None:
    model = CleanAudioINN(
        audio_channels=1,
        wavelet_levels=2,
        num_blocks=1,
        hidden_channels=8,
        subnet_layers=2,
        clamp=1.0,
        kernel_size=3,
    )
    assert model.latent_mode == "gaussian"
    assert model.predictor is None
    with pytest.raises(RuntimeError):
        model.predicted_inverse(torch.randn(1, 1, 1024))


def test_secret_loss_trains_predictor() -> None:
    config = load_config(ROOT / "configs" / "rpm_one_day_librispeech.yaml")
    model = _predicted_model()
    objective = StageOneLoss(config)
    cover = torch.randn(2, 1, 4096) * 0.1
    secret = torch.randn(2, 1, 4096) * 0.1
    total, _, _ = compute_batch(model, objective, cover, secret)
    total.backward()
    assert model.predictor is not None
    output_gradient = model.predictor.output.weight.grad
    assert output_gradient is not None
    assert output_gradient.abs().sum() > 0

    values = latent_comparison(model, cover, secret)
    assert set(values) == {
        "secret_snr_db_gaussian",
        "secret_snr_db_true_residual",
        "secret_snr_db_predicted",
        "latent_prediction_snr_db",
    }


def test_rpm_config_builds_predicted_model() -> None:
    config = load_config(ROOT / "configs" / "rpm_one_day_librispeech.yaml")
    assert config["model"]["latent_mode"] == "predicted"
    assert config["loss"]["latent_prior"] == 0.0
    model = CleanAudioINN.from_config(config)
    assert model.predictor is not None


def test_config_rejects_unknown_latent_mode() -> None:
    config = load_config(ROOT / "configs" / "baseline_16k.yaml")
    config["model"]["latent_mode"] = "uniform"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_evaluate_exports_metrics_and_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    time = np.arange(16000, dtype=np.float32) / 16000
    for index, frequency in enumerate((220.0, 330.0, 440.0)):
        waveform = 0.1 * np.sin(2 * np.pi * frequency * time)
        sf.write(audio_root / f"tone_{index}.wav", waveform, 16000)
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(
        root=audio_root,
        output=manifest,
        train_ratio=0.8,
        val_ratio=0.1,
        source="test",
        forced_split="test",
    )

    config = load_config(ROOT / "configs" / "rpm_one_day_librispeech.yaml")
    config["data"]["segment_seconds"] = 0.512
    config["data"]["manifest"] = str(manifest)
    config["model"].update({"num_blocks": 1, "hidden_channels": 8, "subnet_layers": 2})
    config["model"]["predictor"] = {"hidden_channels": 8, "num_blocks": 1, "kernel_size": 3}
    model = CleanAudioINN.from_config(config)
    checkpoint = tmp_path / "run" / "checkpoints" / "step_00000001.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"step": 1, "model": model.state_dict(), "config": config}, checkpoint)

    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectra-evaluate",
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "2",
            "--examples",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
    )
    evaluate.main()

    summary = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["latent_mode"] == "predicted"
    assert summary["num_items"] == 3
    assert "secret_si_sdr_db_predicted_pcm16" in summary["metrics"]
    assert (output_dir / "example_001" / "recovered_secret_predicted.wav").is_file()
