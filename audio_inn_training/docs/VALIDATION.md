# Scaffold Validation

Before packaging, the scaffold was exercised with a generated synthetic 16 kHz dataset to verify the engineering path independently of LibriSpeech availability.

Validated:

- Python source compilation.
- `pytest` unit tests: squeeze/unsqueeze round trip, exact INN inversion with the true latent, STFT loss sanity, latent-prior loss sanity.
- One complete smoke-training epoch.
- Validation pass.
- `best.pt`, `last.pt`, and epoch checkpoint creation.
- WAV sample export.
- Evaluation script loading `best.pt` and comparing random-`z` recovery against the true-`r` oracle.
- Demo export with deterministic random `z`, true-`r` reverse-pass WAVs, and machine-readable metrics.
- Strict preflight failure for absent datasets and undersized hardware configs.
- Atomic checkpoint replacement to protect the previous checkpoint if a save is interrupted.

The synthetic data and validation run outputs are intentionally **not included** in the ZIP. Generate a local synthetic dataset with `python scripts/make_synthetic_dataset.py` only if you want to test the full pipeline before downloading LibriSpeech.

This validates that the scaffold runs; it does **not** validate research quality or reproduce published WavInWav results. The real baseline still needs to be trained and evaluated on LibriSpeech.
