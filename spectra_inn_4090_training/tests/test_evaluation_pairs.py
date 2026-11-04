from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spectra_inn.data.evaluation import EvaluationPairedAudioDataset, balanced_indices, speaker_id


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    for speaker in ("10", "20", "30"):
        for index in range(3):
            path = tmp_path / speaker / "chapter" / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.full(512, 0.01 * (int(speaker) + index), dtype=np.float32), 16000)
            rows.append({"path": str(path), "source": "librispeech_test", "split": "val"})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def dataset(manifest: Path, **kwargs) -> EvaluationPairedAudioDataset:
    return EvaluationPairedAudioDataset(
        manifest=manifest, split="val", sample_rate=16000, channels=1, segment_samples=256,
        **kwargs,
    )


def test_cross_speaker_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    first = dataset(manifest, pairing="cross_speaker", pair_seed=7)
    second = dataset(manifest, pairing="cross_speaker", pair_seed=7)
    assert first.secret_indices == second.secret_indices
    for index, secret_index in enumerate(first.secret_indices):
        assert speaker_id(first.records[index]) != speaker_id(first.records[secret_index])
        assert first[index]["cover"].shape == (1, 256)


def test_balanced_covers_are_spread_over_speakers(tmp_path: Path) -> None:
    data = dataset(make_manifest(tmp_path))
    indices = balanced_indices(data.records, 6, 7)
    assert indices == balanced_indices(data.records, 6, 7)
    assert len(set(indices)) == 6
    assert {speaker_id(data.records[index]) for index in indices} == {"10", "20", "30"}


def test_cross_speaker_silence_retry_does_not_cross_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = dataset(make_manifest(tmp_path), pairing="cross_speaker", min_rms=0.003)
    original = data._load
    tried = []

    def silent(record):
        tried.append(speaker_id(record))
        return np.zeros_like(original(record))

    monkeypatch.setattr(data, "_load", silent)
    with pytest.raises(RuntimeError, match="pairing not relaxed"):
        data[0]
    assert set(tried) == {speaker_id(data.records[0])}


def test_legacy_pairing_remains_neighbor_based(tmp_path: Path) -> None:
    data = dataset(make_manifest(tmp_path))
    assert data.secret_indices == []
    pair = data[0]
    assert np.isclose(float(pair["cover"].mean()), 0.1, atol=1e-4)
    assert np.isclose(float(pair["secret"].mean()), 0.11, atol=1e-4)


def test_unknown_speaker_does_not_get_guessed_from_arbitrary_folders() -> None:
    with pytest.raises(ValueError, match="speaker_id"):
        speaker_id({"source": "music", "path": "/a/b/c.wav"})
