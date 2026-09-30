# Current Goals: Two-Week Prototype

## What we are building now

For the next two weeks, the goal is **not** the full robustness/diffusion research system.

The goal is a viable, reproducible **Audio-INN audio-in-audio steganography prototype** that can be demonstrated end-to-end:

```text
cover speech + secret speech
          ↓
       Audio-INN
          ↓
stego speech + auxiliary latent r
          ↓
transmit/use stego speech
          ↓
Audio-INN inverse with Gaussian z
          ↓
recovered secret speech
```

The prototype must demonstrate:

1. Training on LibriSpeech.
2. Stego audio that remains close to the cover.
3. Secret recovery from stego using the current Gaussian-latent baseline.
4. The true-`r` oracle path, which verifies the invertible core.
5. Repeatable evaluation, logging, checkpoints, and generated WAV examples.

## What is explicitly deferred

These are research extensions, **not two-week blockers**:

- MP3/AAC/Opus robustness training.
- Attack-aware redundancy prediction (RPM).
- Conditional diffusion over the redundancy latent.
- Stego correction/restoration.
- WER/PESQ/STOI integration.
- Full A/V multimodal integration.
- Cross-modal shared latent / AV Transformer.

## Success criteria for the presentation

A presentation-ready result should include:

- a trained checkpoint;
- a plot of training/validation loss;
- cover vs stego SNR;
- secret recovery SI-SDR using random `z`;
- true-`r` oracle SI-SDR;
- several listening examples:
  - cover.wav
  - secret.wav
  - stego.wav
  - recovered_random_z.wav
  - recovered_true_r_oracle.wav

The prototype is successful even if random-`z` recovery is not yet robust to compression. Robustness is the next research phase.
