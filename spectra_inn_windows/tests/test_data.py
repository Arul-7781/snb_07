from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from spectra_inn.data.dataset import PairedAudioDataset
from spectra_inn.data.manifest import build_manifest


def _write_tone(path: Path, sample_rate: int, frequency: float) -> None:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.1 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, waveform, sample_rate)


def test_manifest_and_dataset_contract(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_tone(first, 16000, 220.0)
    _write_tone(second, 16000, 440.0)

    indexed_manifest = tmp_path / "indexed.jsonl"
    count, failures = build_manifest(
        root=tmp_path,
        output=indexed_manifest,
        train_ratio=0.8,
        val_ratio=0.1,
        source="test",
    )
    assert count == 2
    assert failures == []

    official_manifest = tmp_path / "official_splits.jsonl"
    build_manifest(
        root=tmp_path,
        output=official_manifest,
        train_ratio=0.8,
        val_ratio=0.1,
        source="train_source",
        forced_split="train",
    )
    build_manifest(
        root=tmp_path,
        output=official_manifest,
        train_ratio=0.8,
        val_ratio=0.1,
        source="val_source",
        forced_split="val",
        append=True,
    )
    official_records = [
        json.loads(line) for line in official_manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["split"] for record in official_records} == {"train", "val"}

    paired_manifest = tmp_path / "paired.jsonl"
    records = [
        {
            "path": str(path),
            "sample_rate": 16000,
            "channels": 1,
            "frames": 16000,
            "duration": 1.0,
            "split": "train",
            "source": "test",
        }
        for path in (first, second)
    ]
    paired_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    dataset = PairedAudioDataset(
        manifest=paired_manifest,
        split="train",
        sample_rate=16000,
        channels=1,
        segment_samples=8000,
        min_rms=0.001,
        random_gain_db=0.0,
        random_crop=False,
    )
    item = dataset[0]
    assert item["cover"].shape == (1, 8000)
    assert item["secret"].shape == (1, 8000)
