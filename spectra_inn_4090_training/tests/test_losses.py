from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectra_inn.config import ConfigError, load_config, validate_config
from spectra_inn.losses.audio import StageOneLoss, charbonnier, normalized_mse
from spectra_inn.models.clean_inn import CleanAudioINN
from spectra_inn.train import compute_batch

ROOT = Path(__file__).resolve().parents[1]


def test_normalized_mse_is_zero_for_exact_reconstruction() -> None:
    reference = torch.randn(2, 1, 128)
    assert float(normalized_mse(reference, reference)) == 0.0


def test_normalized_mse_penalizes_gain_errors_and_normalizes_each_example() -> None:
    reference = torch.ones(2, 1, 128)
    reference[0] *= 0.01
    estimate = reference.clone()
    estimate[0] = 0
    # A zero prediction has unit NMSE, independent of reference amplitude above the floor.
    assert torch.allclose(normalized_mse(reference, estimate), torch.tensor(0.5))
    assert torch.allclose(normalized_mse(reference, 0.5 * reference), torch.tensor(0.25))
    assert torch.allclose(
        normalized_mse(reference, estimate), normalized_mse(4 * reference, 4 * estimate)
    )


def test_normalized_mse_is_finite_for_silence_and_has_finite_gradients() -> None:
    reference = torch.zeros(2, 1, 128)
    estimate = torch.full_like(reference, 0.01, requires_grad=True)
    loss = normalized_mse(reference, estimate)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(estimate.grad).all()
    assert torch.count_nonzero(estimate.grad) == estimate.numel()


def test_normalized_mse_avoids_half_precision_square_underflow() -> None:
    reference = torch.full((2, 1, 128), 0.001, dtype=torch.float16)
    estimate = torch.full_like(reference, 0.0009)
    loss = normalized_mse(reference, estimate)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(0.01, rel=0.02)


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan"), float("inf")])
def test_normalized_mse_rejects_invalid_floor(floor: float) -> None:
    reference = torch.ones(2, 1, 128)
    with pytest.raises(ValueError, match="power_floor"):
        normalized_mse(reference, reference, floor)


def test_normalized_mse_rejects_broadcasting_and_missing_batch_dimensions() -> None:
    reference = torch.ones(2, 1, 128)
    with pytest.raises(ValueError, match="matching"):
        normalized_mse(reference, reference[:1])
    with pytest.raises(ValueError, match="matching"):
        normalized_mse(reference[0], reference[0])


def test_original_configs_keep_charbonnier_waveform_loss() -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_librispeech.yaml")
    objective = StageOneLoss(config)
    reference = torch.randn(2, 1, 128)
    estimate = torch.randn_like(reference)
    assert objective.waveform_mode == "charbonnier"
    assert torch.equal(
        objective.waveform_loss(reference, estimate), charbonnier(reference, estimate)
    )


def test_nmse_trial_changes_only_the_waveform_objective() -> None:
    trial = load_config(ROOT / "configs" / "rpm_3060_nmse_10k.yaml")
    control = load_config(ROOT / "configs" / "rpm_3060_control_10k.yaml")
    assert StageOneLoss(trial).waveform_mode == "normalized_mse"
    assert trial["loss"].pop("waveform_mode") == "normalized_mse"
    assert trial["loss"].pop("nmse_power_floor") == 1e-6
    for config in (trial, control):
        config.pop("experiment")
        config.pop("_config_path")
    assert trial == control


@pytest.mark.parametrize("mode", ["mse_typo", "", None])
def test_config_rejects_unknown_waveform_mode(mode: object) -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_nmse_10k.yaml")
    config["loss"]["waveform_mode"] = mode
    with pytest.raises(ConfigError, match="waveform_mode"):
        validate_config(config)


@pytest.mark.parametrize("floor", [0, -1, float("nan"), float("inf"), None, "invalid"])
def test_config_rejects_invalid_nmse_power_floor(floor: object) -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_nmse_10k.yaml")
    config["loss"]["nmse_power_floor"] = floor
    with pytest.raises(ConfigError, match="power_floor"):
        validate_config(config)


def test_nmse_loss_backpropagates_through_predicted_inverse() -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_nmse_10k.yaml")
    config["model"].update({"num_blocks": 1, "hidden_channels": 8, "subnet_layers": 2})
    config["model"]["predictor"] = {"hidden_channels": 8, "num_blocks": 2, "kernel_size": 3}
    config["loss"]["stft_windows"] = [64, 128]
    model = CleanAudioINN.from_config(config)
    objective = StageOneLoss(config)
    cover = torch.randn(2, 1, 512) * 0.1
    secret = torch.randn_like(cover) * 0.1
    loss, _, _ = compute_batch(model, objective, cover, secret)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
