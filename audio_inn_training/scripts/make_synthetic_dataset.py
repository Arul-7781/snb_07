from pathlib import Path
import numpy as np
import soundfile as sf

sr = 16000
root = Path("data/LibriSpeech")
for split, count in [("train-clean-100", 12), ("dev-clean", 6), ("test-clean", 6)]:
    d = root / split / "synthetic" / "0001"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(1000 + i + len(split))
        duration = 1.2 + 0.05 * (i % 4)
        t = np.arange(int(sr * duration)) / sr
        freq = 140 + 17 * i
        x = 0.08 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.shape)
        sf.write(d / f"synthetic-{i:04d}.wav", x.astype(np.float32), sr)
print("Synthetic dataset written under data/LibriSpeech")
