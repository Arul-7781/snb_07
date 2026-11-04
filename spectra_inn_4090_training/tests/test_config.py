from __future__ import annotations

from pathlib import Path

from spectra_inn.config import load_config, segment_samples


def test_baseline_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "baseline_16k.yaml")
    assert segment_samples(config) == 32000
    assert config["model"]["wavelet_levels"] == 3


def test_one_day_config_has_wall_clock_limit() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "one_day_librispeech.yaml")
    assert config["training"]["max_hours"] == 20.0
    assert config["data"]["sample_rate"] == 16000
