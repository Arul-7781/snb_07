# Codex Review Checklist

This scaffold is deliberately optimized for getting a viable Audio-INN prototype training quickly. Before a long final run, review these items against the exact paper/architecture you decide to reproduce.

## Architecture

- Verify the affine-coupling equations against the intended WavInWav implementation/paper.
- Decide whether the simple channel reversal between blocks should be replaced by a learned/invertible 1x1 mixing layer.
- Confirm the squeeze factor, block count, hidden width, dilation/receptive field, and subnet architecture.
- Check whether the final paper baseline requires additional preprocessing, scrambling, or conditioning not present here.

## Training objective

- Verify the relative weights of:
  - stego waveform loss;
  - stego STFT loss;
  - secret waveform loss;
  - secret STFT loss;
  - latent prior regularization.
- Check whether the published baseline uses additional perceptual or adversarial losses.
- Confirm that Gaussian latent substitution during training matches the intended blind-decoding experiment.

## Data

- Confirm the chosen LibriSpeech subsets and train/validation/test separation.
- Verify that cover and secret pair generation cannot accidentally select the same file.
- Consider speaker-disjoint cover/secret pairing if required by the experimental design.
- For final paper-quality experiments, persist a fixed test-pair manifest rather than regenerating pairs from only a seed.

## Engineering

- Run `pytest -q` after every structural change.
- Run `python scripts/preflight.py --config <training-config>` before long training.
- Check resume behavior from `last.pt`.
- Check TensorBoard and CSV metrics agree.
- Confirm WAV samples are not clipping.
- Profile peak VRAM and throughput on the RTX 4090 before increasing model size.

## Important limitation

This repository is **not claimed to be an exact reproduction of WavInWav**. It is a compact invertible audio-hiding baseline with the same high-level structure, built to establish the project's training/evaluation infrastructure quickly.

Do not report comparisons against published WavInWav numbers until the architecture and experimental protocol have been matched carefully.
