# Research Plan

## Problem

Hide a full secret speech waveform inside a cover speech waveform while keeping the resulting stego audio perceptually close to the cover and allowing the secret to be recovered.

The invertible core maps:

```text
F(cover, secret) = (stego, redundancy)
```

Exact inversion is possible when both outputs are retained:

```text
F^-1(stego, redundancy) = (cover, secret)
```

In a real steganography setting, only the stego waveform is transmitted, so the auxiliary redundancy is unavailable. The baseline replaces it with Gaussian `z` during decoding.

## Stage 1 — two-week prototype

Build a stable WavInWav-style baseline:

- LibriSpeech 16 kHz mono.
- 1-second cover/secret segments.
- 1D invertible affine coupling network.
- waveform + multi-resolution STFT objectives.
- Gaussian latent substitution during blind decoding.
- true-redundancy oracle for diagnosis.
- TensorBoard/CSV logging.
- best/last/epoch checkpoints.
- reproducible WAV examples.

## Stage 2 — after prototype: diagnose robustness

Run the four-case oracle study:

```text
A: clean stego + true r
B: clean stego + random z
C: attacked stego + true r
D: attacked stego + random z
```

This tells us whether the bottleneck is the missing redundancy, corrupted stego, or both.

## Stage 3 — deterministic redundancy prediction

Adapt the LF-VSN idea to audio:

```text
stego -> Audio-RPM -> r_hat -> inverse INN
```

First train clean, then attack-aware.

## Stage 4 — conditional latent diffusion

Use attacked-stego features as multi-scale conditioning and progressively refine a redundancy estimate:

```text
attacked stego
     ↓
condition encoder
     ↓
RPM coarse estimate
     ↓
conditional diffusion refinement
     ↓
r_hat
     ↓
inverse INN
```

Use FiLM/adaptive GroupNorm-style conditioning rather than literal AdaIN.

## Stage 5 — stego repair, only if required

If attacked stego + true `r` still fails, predict a correction residual:

```text
y_corr = y_attacked + delta_y
```

and decode using:

```text
F^-1(y_corr, r_hat)
```

## Stage 6 — multimodal extension

Combine the future Audio-INN robustness stack with the existing Video-INN using a shared AV representation. Prefer one shared conditioning representation with separate audio/video redundancy heads rather than forcing both modalities into one literal seed.

The strongest experiment is cross-modal rescue: corrupt one modality and test whether the cleaner modality improves recovery of the damaged modality's missing reversible state.
