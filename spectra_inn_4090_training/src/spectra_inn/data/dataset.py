from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


def _read_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {path}. Build it with spectra-build-manifest first."
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") == split:
                record["_line"] = line_number
                records.append(record)
    if len(records) < 2:
        raise ValueError(f"Split '{split}' needs at least two audio records; found {len(records)}")
    return records


def _convert_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if channels == 1:
        return audio.mean(axis=1, keepdims=True)
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, channels, axis=1)
    return audio[:, :channels]


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float32, copy=False)


def _crop_or_pad(audio: np.ndarray, samples: int, random_crop: bool) -> np.ndarray:
    length = audio.shape[0]
    if length >= samples:
        start = random.randint(0, length - samples) if random_crop and length > samples else 0
        return audio[start : start + samples]
    padded = np.zeros((samples, audio.shape[1]), dtype=np.float32)
    padded[:length] = audio
    return padded


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


class PairedAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Pairs independent cover/secret tracks and returns fixed-length waveform crops."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        sample_rate: int,
        channels: int,
        segment_samples: int,
        min_rms: float = 0.0,
        random_gain_db: float = 0.0,
        random_crop: bool = True,
    ) -> None:
        self.records = _read_manifest(Path(manifest).expanduser().resolve(), split)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.segment_samples = int(segment_samples)
        self.min_rms = float(min_rms)
        self.random_gain_db = float(random_gain_db)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, record: dict[str, Any]) -> np.ndarray:
        audio, source_rate = sf.read(record["path"], dtype="float32", always_2d=True)
        audio = _convert_channels(audio, self.channels)
        audio = _resample(audio, int(source_rate), self.sample_rate)
        audio = _crop_or_pad(audio, self.segment_samples, self.random_crop)
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite samples in {record['path']}")
        return np.clip(audio, -1.0, 1.0)

    def _load_with_retry(self, index: int) -> np.ndarray:
        for offset in range(min(8, len(self.records))):
            audio = self._load(self.records[(index + offset) % len(self.records)])
            if _rms(audio) >= self.min_rms:
                return audio
        raise RuntimeError("Could not find a non-silent crop after eight attempts")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cover_index = index % len(self.records)
        if self.random_crop:
            secret_offset = random.randint(1, len(self.records) - 1)
        else:
            secret_offset = 1
        secret_index = (cover_index + secret_offset) % len(self.records)
        cover = self._load_with_retry(cover_index)
        secret = self._load_with_retry(secret_index)

        if self.random_gain_db > 0:
            cover_gain = 10 ** (random.uniform(-self.random_gain_db, self.random_gain_db) / 20)
            secret_gain = 10 ** (random.uniform(-self.random_gain_db, self.random_gain_db) / 20)
            cover = np.clip(cover * cover_gain, -1.0, 1.0)
            secret = np.clip(secret * secret_gain, -1.0, 1.0)

        return {
            "cover": torch.from_numpy(np.ascontiguousarray(cover.T)),
            "secret": torch.from_numpy(np.ascontiguousarray(secret.T)),
        }


class SyntheticPairedAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic-enough structured audio used only to validate training plumbing."""

    def __init__(self, length: int, channels: int, samples: int, sample_rate: int) -> None:
        self.length = length
        self.channels = channels
        self.samples = samples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return self.length

    def _signal(self, index: int, offset: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(index * 7919 + offset)
        time = torch.arange(self.samples, dtype=torch.float32) / self.sample_rate
        base_frequency = 90.0 + float((index * 37 + offset) % 900)
        phase = torch.rand((), generator=generator) * (2 * math.pi)
        signal = 0.18 * torch.sin(2 * math.pi * base_frequency * time + phase)
        signal += 0.06 * torch.sin(2 * math.pi * base_frequency * 2.03 * time)
        signal += 0.01 * torch.randn(self.samples, generator=generator)
        return signal.clamp(-1, 1).repeat(self.channels, 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"cover": self._signal(index, 17), "secret": self._signal(index, 43)}
