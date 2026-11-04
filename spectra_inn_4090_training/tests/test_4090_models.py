from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from spectra_inn.config import ConfigError, load_config, validate_config
from spectra_inn.losses.audio import StageOneLoss, normalized_mse
from spectra_inn.models.clean_inn import CleanAudioINN
from spectra_inn.models.redundancy import MultiScaleRedundancyPredictor1d
from spectra_inn.train import compute_batch, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> dict:
    config = load_config(ROOT / "configs/rpm_3060_nmse_10k.yaml")
    config["model"].update(num_blocks=1, hidden_channels=8, subnet_layers=2)
    config["model"]["predictor"] = {
        "architecture": "unet", "hidden_channels": 8, "num_blocks": 3, "kernel_size": 3,
    }
    config["loss"]["stft_windows"] = [64, 128]
    return config


@pytest.mark.parametrize("length", [4, 17, 32, 33])
def test_unet_predictor_shape_and_zero_start(length: int) -> None:
    predictor = MultiScaleRedundancyPredictor1d(8, 8, 3, 3)
    output = predictor(torch.randn(2, 8, length))
    assert output.shape == (2, 8, length)
    assert torch.count_nonzero(output) == 0
    assert predictor.receptive_field_kind == "upper_bound"


def test_unet_auxiliary_keeps_blind_metrics_and_exact_inverse() -> None:
    config = tiny_config()
    config["loss"]["latent_prediction"] = 0.05
    validate_config(config)
    model = CleanAudioINN.from_config(config)
    cover, secret = torch.randn(2, 1, 512) * 0.1, torch.randn(2, 1, 512) * 0.1
    loss, components, metrics = compute_batch(model, StageOneLoss(config), cover, secret)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(components["latent_prediction"].detach()) > 0
    # True-residual reconstruction is excellent at identity, but blind recovery is zero.
    assert abs(float(metrics["secret_snr_db"].detach())) < 1e-3
    stego, residual, _ = model(cover, secret)
    restored_cover, restored_secret = model.inverse(stego, residual)
    assert torch.allclose(restored_secret, secret, atol=3e-6)
    assert torch.allclose(restored_cover, cover, atol=3e-6)
    assert model.predictor.output.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_auxiliary_target_is_detached() -> None:
    config = tiny_config()
    config["loss"]["latent_prediction"] = 0.05
    reference = torch.randn(2, 1, 128, requires_grad=True)
    predicted = torch.randn(2, 1, 128, requires_grad=True)
    sample = torch.randn(2, 1, 128)
    _, components = StageOneLoss(config)(
        sample, sample, sample, reference, sample, sample,
        {"mean_abs_log_scale": torch.tensor(0.0)}, predicted_residual=predicted,
    )
    components["latent_prediction"].backward()
    assert reference.grad is None
    assert predicted.grad.abs().sum() > 0
    assert torch.allclose(components["latent_prediction"], normalized_mse(reference, predicted))


@pytest.mark.parametrize("changes", [
    {"architecture": "typo"}, {"architecture": "unet", "num_blocks": 2},
])
def test_config_rejects_invalid_new_predictors(changes: dict) -> None:
    config = tiny_config()
    config["model"]["predictor"].update(changes)
    with pytest.raises(ConfigError):
        validate_config(config)


def test_checkpoint_rejects_silent_dilation_change(tmp_path: Path) -> None:
    config = tiny_config()
    config["model"]["predictor"].update(architecture="flat", num_blocks=3, dilations=[1, 1, 1])
    model = CleanAudioINN.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = save_checkpoint(tmp_path, 1, model, optimizer, scaler, config, 3)
    changed = deepcopy(config)
    changed["model"]["predictor"]["dilations"] = [1, 2, 4]
    other = CleanAudioINN.from_config(changed)
    with pytest.raises(ValueError, match="dilations"):
        load_checkpoint(path, other, optimizer, scaler, torch.device("cpu"))


def test_unet_old_style_config_checkpoint_round_trip(tmp_path: Path) -> None:
    config = tiny_config()
    model = CleanAudioINN.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = save_checkpoint(tmp_path, 2, model, optimizer, scaler, config, 3)
    assert load_checkpoint(path, model, optimizer, scaler, torch.device("cpu")) == 2


def test_cpu_cli_training_summary_and_explicit_resume_lr(tmp_path: Path) -> None:
    config = tiny_config()
    config["experiment"]["output_dir"] = str(tmp_path / "tiny")
    config["data"].update(segment_seconds=0.032, num_workers=0)
    config["training"].update(
        batch_size=2, accumulation_steps=1, amp=False,
        log_every=1, validate_every=1, checkpoint_every=1, validation_batches=1,
    )
    config["loss"]["latent_prediction"] = 0.05
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config))
    command = [sys.executable, "-m", "spectra_inn.train", "--config", str(path),
               "--smoke", "--device", "cpu"]
    subprocess.run(command + ["--max-steps", "2"], cwd=ROOT,
                   capture_output=True, text=True, check=True, timeout=45)
    output = tmp_path / "tiny_smoke"
    subprocess.run(
        command + ["--max-steps", "3", "--resume", str(output / "checkpoints/step_00000002.pt"),
                   "--learning-rate", "0.00003"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=45,
    )
    checkpoint = torch.load(output / "checkpoints/step_00000003.pt", weights_only=False)
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 3e-5
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["target_reached"]
    assert summary["initial_step"] == 2 and summary["step"] == 3
    assert summary["optimizer_updates_session"] == 1
    assert summary["peak_cuda_allocated_bytes"] == 0
