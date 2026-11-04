from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectra_inn import train
from spectra_inn.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


class CountingDataset:
    def __init__(self, **kwargs: object) -> None:
        self.split = kwargs["split"]
        self.calls = 0

    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"cover": torch.randn(1, 16), "secret": torch.randn(1, 16)}


def test_overfit_caches_training_pairs_but_keeps_validation_held_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train, "PairedAudioDataset", CountingDataset)
    config = load_config(ROOT / "configs" / "rpm_3060_context_10k.yaml")
    config["data"]["overfit_pairs"] = 16
    train_loader, val_loader = train.create_loaders(config, smoke=False)
    assert len(train_loader.dataset) == 16
    assert isinstance(train_loader.dataset, list)
    assert isinstance(val_loader.dataset, CountingDataset)
    assert val_loader.dataset.split == "val"
    assert val_loader.dataset.calls == 0
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    cached = [pair["cover"].clone() for pair in train_loader.dataset]
    for _ in range(2):
        assert sum(batch["cover"].shape[0] for batch in train_loader) == 16
    assert all(
        torch.equal(pair["cover"], original)
        for pair, original in zip(train_loader.dataset, cached, strict=True)
    )


@pytest.mark.parametrize("count", [0, -1, 2, True, 4.5])
def test_config_rejects_invalid_overfit_pair_counts(count: object) -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_context_10k.yaml")
    config["data"]["overfit_pairs"] = count
    with pytest.raises(ConfigError, match="overfit_pairs"):
        validate_config(config)


def test_overfit_rejects_request_larger_than_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "PairedAudioDataset", CountingDataset)
    config = load_config(ROOT / "configs" / "rpm_3060_context_10k.yaml")
    config["data"]["overfit_pairs"] = 101
    with pytest.raises(ValueError, match="overfit_pairs"):
        train.create_loaders(config, smoke=False)


def test_overfit_cannot_use_synthetic_smoke_audio() -> None:
    config = load_config(ROOT / "configs" / "rpm_3060_context_10k.yaml")
    config["data"]["overfit_pairs"] = 16
    with pytest.raises(ValueError, match="--smoke"):
        train.create_loaders(config, smoke=True)


@pytest.mark.parametrize("training", [True, False])
def test_fixed_pair_metrics_cover_all_pairs_and_preserve_rng_and_mode(
    monkeypatch: pytest.MonkeyPatch, training: bool,
) -> None:
    model = torch.nn.Identity().train(training)
    objective = torch.nn.Identity()
    pairs = [
        {"cover": torch.full((1, 8), float(index)), "secret": torch.zeros(1, 8)}
        for index in range(5)
    ]
    sizes = []

    def fake_compute(model, objective, cover, secret):
        assert not torch.is_grad_enabled()
        assert not model.training
        # A diagnostic must not change later training RNG, even for Gaussian decoding.
        torch.rand(3)
        sizes.append(len(cover))
        value = cover.mean()
        return value, {"component": value * 2}, {"cover_snr_db": value, "secret_snr_db": value}

    monkeypatch.setattr(train, "compute_batch", fake_compute)
    state = torch.get_rng_state().clone()
    result = train.evaluate_fixed_pairs(model, objective, pairs, torch.device("cpu"), 2)
    assert sizes == [2, 2, 1]
    assert result == {
        "total": 2.0, "component": 4.0, "cover_snr_db": 2.0, "secret_snr_db": 2.0,
    }
    assert torch.equal(state, torch.get_rng_state())
    assert model.training is training


def test_fixed_pair_metric_failure_restores_rng_and_model_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Identity().train()
    pairs = [{"cover": torch.zeros(1, 8), "secret": torch.zeros(1, 8)}]

    def fail(*args):
        torch.rand(3)
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(train, "compute_batch", fail)
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        train.evaluate_fixed_pairs(model, torch.nn.Identity(), pairs, torch.device("cpu"), 1)
    assert torch.equal(state, torch.get_rng_state())
    assert model.training


def test_explicit_loader_seed_decouples_data_order_from_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Indices(CountingDataset):
        def __getitem__(self, index):
            value = torch.full((1, 16), float(index))
            return {"cover": value, "secret": value}

    monkeypatch.setattr(train, "PairedAudioDataset", Indices)
    config = load_config(ROOT / "configs/rpm_3060_context_10k.yaml")
    config["data"].update(loader_seed=3407, num_workers=0, pin_memory=False)
    orders = []
    for initialization_seed in (1, 99):
        torch.manual_seed(initialization_seed)
        torch.randn(123)  # Different models may consume different amounts of initialization RNG.
        loader, _ = train.create_loaders(config, smoke=False)
        orders.append([batch["cover"][:, 0, 0].tolist() for batch in loader])
    assert orders[0] == orders[1]
