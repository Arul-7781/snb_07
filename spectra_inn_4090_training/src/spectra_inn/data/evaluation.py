"""Deterministic evaluation pairings; legacy behavior remains the default."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from spectra_inn.data.dataset import PairedAudioDataset, _rms


def speaker_id(record: dict[str, Any]) -> str:
    explicit = record.get("speaker_id")
    if explicit is not None:
        return str(explicit)
    if str(record.get("source", "")).startswith("librispeech_"):
        return Path(record["path"]).parent.parent.name
    raise ValueError("Speaker-aware evaluation requires speaker_id or a LibriSpeech source")


def balanced_indices(records: list[dict[str, Any]], limit: int, seed: int) -> list[int]:
    """Spread cover selection over speakers, instead of taking the first sorted files."""
    if limit < 1:
        raise ValueError("Pair limit must be positive")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[speaker_id(record)].append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < min(limit, len(records)):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == min(limit, len(records)):
                    break
    rng.shuffle(selected)
    return selected


class EvaluationPairedAudioDataset(PairedAudioDataset):
    def __init__(self, *, pairing: str = "legacy", pair_seed: int = 0, **kwargs: Any) -> None:
        super().__init__(random_crop=False, random_gain_db=0.0, **kwargs)
        if pairing not in ("legacy", "cross_speaker"):
            raise ValueError("pairing must be legacy or cross_speaker")
        self.pairing = pairing
        self.groups: dict[str, list[int]] = defaultdict(list)
        self.group_positions: dict[int, int] = {}
        self.secret_indices: list[int] = []
        if pairing == "cross_speaker":
            for index, record in enumerate(self.records):
                key = speaker_id(record)
                self.group_positions[index] = len(self.groups[key])
                self.groups[key].append(index)
            speakers = sorted(self.groups)
            if len(speakers) < 2:
                raise ValueError("Cross-speaker evaluation needs at least two speakers")
            positions = {key: index for index, key in enumerate(speakers)}
            rng = random.Random(pair_seed)
            for record in self.records:
                current = positions[speaker_id(record)]
                other = rng.randrange(len(speakers) - 1)
                other += int(other >= current)
                self.secret_indices.append(rng.choice(self.groups[speakers[other]]))

    def _same_speaker_load(self, index: int):
        pool = self.groups[speaker_id(self.records[index])]
        start = self.group_positions[index]
        for offset in range(min(8, len(pool))):
            audio = self._load(self.records[pool[(start + offset) % len(pool)]])
            if _rms(audio) >= self.min_rms:
                return audio
        raise RuntimeError("No non-silent crop within the selected speaker; pairing not relaxed")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.pairing == "legacy":
            return super().__getitem__(index)
        # Retry within each selected speaker, never across a speaker boundary.
        cover = self._same_speaker_load(index)
        secret = self._same_speaker_load(self.secret_indices[index])
        return {
            "cover": torch.from_numpy(cover.T.copy()),
            "secret": torch.from_numpy(secret.T.copy()),
        }
