from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

AUDIO_EXTS = {".flac", ".wav"}


def discover_audio_files(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _crop_or_pad(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) >= length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start:start + length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(audio)] = audio
    return out


class LibriSpeechPairDataset(Dataset):
    """Deterministic pseudo-random cover/secret pairing for a LibriSpeech split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        num_pairs: int = 10000,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.files = discover_audio_files(self.split_dir)
        if len(self.files) < 2:
            raise RuntimeError(
                f"Need at least 2 audio files in {self.split_dir}. "
                "See docs/SETUP.md for the expected LibriSpeech layout."
            )
        self.sample_rate = sample_rate
        self.length = int(round(sample_rate * segment_seconds))
        self.num_pairs = num_pairs
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx * 104729 + self.epoch * 1000003)
        n = len(self.files)
        c_idx = int(rng.integers(0, n))
        s_idx = int(rng.integers(0, n - 1))
        if s_idx >= c_idx:
            s_idx += 1

        cover = _crop_or_pad(_load_mono(self.files[c_idx], self.sample_rate), self.length, rng)
        secret = _crop_or_pad(_load_mono(self.files[s_idx], self.sample_rate), self.length, rng)

        cover_t = torch.from_numpy(cover).unsqueeze(0)
        secret_t = torch.from_numpy(secret).unsqueeze(0)
        return cover_t, secret_t
