# SPECTRA-INN RPM Experiment History

## Purpose

This document records the audio-pipeline changes made while investigating poor blind secret recovery. It is intended to give the team one reproducible explanation of:

- What the original INN did.
- Why the first RPM result was poor.
- What changed in each iteration.
- Which architecture is currently considered the primary experiment.
- Which results are valid and which are only smoke tests.
- How to reproduce the current run.

The work described here is on branch `audio-rpm-improve`.

## Executive Summary

The original SPECTRA-INN Stage 1 model was numerically invertible, but blind decoding replaced the discarded INN residual with random Gaussian noise. The first LF-VSN-style RPM replaced that random noise with a learned residual prediction, but validation secret SNR remained around 10-11 dB.

The important diagnostic was:

```text
true residual secret SNR: approximately 82-102 dB
RPM predicted residual secret SNR: approximately 10-11 dB
```

This proves that the INN coupling blocks can invert accurately when given the true residual. It also proves that the RPM is not predicting the missing residual well enough. The problem is therefore not basic numerical invertibility; it is the blind residual-prediction formulation and training objective.

The current primary architecture is a pure INN/RPM path with a multi-scale U-Net-style RPM. The optional direct secret decoder remains in the code for ablation, but it is disabled as the primary secret path.

## Baseline Before RPM

### Original model behavior

The original model implemented:

```text
cover + secret
    -> INN
    -> stego + residual
```

The stego waveform is available to the receiver. The residual is discarded. During blind decoding, the model used:

```text
stego + random Gaussian latent
    -> INN inverse
    -> recovered cover + recovered secret
```

The baseline therefore did not have a learned redundancy predictor.

### Original loss

The baseline objective was `StageOneLoss` in `src/spectra_inn/losses/audio.py`:

```text
2.0 * stego waveform Charbonnier loss
1.0 * stego multi-resolution STFT loss
1.0 * recovered secret waveform Charbonnier loss
1.0 * recovered secret multi-resolution STFT loss
0.25 * recovered cover waveform loss
0.1  * Gaussian latent prior loss
0.001 * coupling-scale stability loss
```

The Gaussian latent prior encouraged the discarded residual to look like a standard normal distribution. That made sense for Gaussian blind decoding, but it was not appropriate once the residual was predicted by an RPM.

### Baseline limitations

The baseline was clean-audio only. It did not model:

- Codec compression.
- 16-bit PCM quantization during training.
- Resampling.
- Noise or filtering attacks.
- Synchronization shifts.
- Robust extraction.
- Error correction.

## Dataset Pipeline Changes

The repository originally downloaded:

- `train-clean-100` for training.
- `dev-clean` for validation.
- `test-clean` for testing.

The launcher was changed so normal mode now uses:

- `train-clean-360` for training.
- `dev-clean` for validation.
- `test-clean` for testing.

The launcher still has an opt-in `-FullLibriSpeech` mode for all seven official splits.

### Download behavior

`train_windows.ps1` now:

1. Uses OpenSLR FLAC archives compatible with the PyTorch loader.
2. Uses `aria2c` with parallel segmented downloads.
3. Verifies each archive with its official MD5 checksum.
4. Extracts the official split directories.
5. Rebuilds `data/manifests/all.jsonl`.
6. Preserves official train/validation/test boundaries.

The current manifest was verified with:

```text
train-clean-360: 104,014 records
 dev-clean:       2,703 records
 test-clean:      2,620 records
 total:         109,337 records
```

The dataset is 16 kHz audio. The training config uses mono 2-second random crops.

## Iteration 1: Initial LF-VSN-Style RPM

### Goal

Replace random Gaussian latent decoding with a learned residual prediction module.

### Architecture

`RedundancyPredictor1d` initially used:

- Input convolution.
- Eight residual convolution blocks.
- LeakyReLU activation.
- Output convolution initialized to zero.

The path was:

```text
stego Haar coefficients
    -> RPM
    -> predicted residual
    -> INN inverse
    -> recovered secret
```

### Training behavior

The RPM was trained only indirectly through the final recovered-cover and recovered-secret losses. There was no direct comparison between the predicted residual and the true residual.

The Gaussian prior loss was disabled:

```yaml
latent_prior: 0.0
```

### Result

The run reached 200,000 steps. The important final validation result was approximately:

```text
cover_snr:        14.57 dB
predicted secret: 11.05 dB
Gaussian secret:   0.36 dB
true residual:    82.50 dB
```

### Interpretation

The true residual result was excellent because it represents near-exact numerical inversion. The predicted result was poor because the RPM did not reconstruct the missing residual accurately.

This was the first evidence that the main bottleneck was blind residual prediction, not INN numerical stability.

## Iteration 2: Direct Residual-Prediction Loss

### Change

A new loss component was added to the full INN objective:

```yaml
latent_prediction: 0.1
```

It compares:

```text
true residual vs RPM predicted residual
```

using a Charbonnier loss. The true residual is detached from the gradient so the INN cannot change the target to reduce this auxiliary loss.

### Full objective

The RPM trial objective became:

```text
2.0 * stego waveform loss
1.0 * stego STFT loss
1.0 * secret waveform loss
1.0 * secret STFT loss
0.25 * recovered cover loss
0.1  * residual prediction loss
0.0  * Gaussian latent prior loss
0.001 * scale stability loss
```

### Validation

All 13 tests, Ruff, and smoke training passed. However, the subsequent real run still produced approximately 10.46 dB after 16,000 steps. The direct residual loss alone was not enough.

## Iteration 3: Larger Dilated RPM

### Changes

The RPM was enlarged and given dilated convolutions:

- Hidden channels: `64 -> 128`.
- Residual blocks: `8 -> 10`.
- Dilations expanded temporal context.
- RPM learning rate changed to `2e-4`.
- Main INN learning rate remained `1e-4`.
- A 20,000-step true-residual warm-up was added.

### Warm-up idea

During warm-up, the INN used the true residual for reconstruction while the RPM learned to predict the residual. After warm-up, the RPM was used for blind decoding.

The purpose was to avoid forcing a randomly initialized RPM into the main inverse path immediately.

### Limitation

This used a hard transition at the end of warm-up. It was later replaced because a hard switch can create a sudden loss discontinuity.

## Iteration 4: Optional Direct Secret Decoder

### Change

A direct stego-to-secret decoder was added as an experiment:

```text
stego waveform
    -> direct secret decoder
    -> recovered secret
```

This gave the model a direct supervised secret-recovery path.

### Why it was not accepted as the primary architecture

A direct decoder bypasses the INN inverse for secret recovery. That makes the system a hybrid model rather than a pure INN/RPM system.

The direct decoder remains available as an optional ablation, but the primary config uses:

```yaml
secret_decode_mode: predicted
```

Therefore the primary path remains:

```text
stego -> RPM -> predicted residual -> INN inverse -> secret
```

## Iteration 5: Gradual INN/RPM Curriculum

### Change

The hard oracle-to-RPM switch was replaced with a gradual blend.

The training path starts as:

```text
100% true residual, 0% predicted residual
```

and gradually becomes:

```text
0% true residual, 100% predicted residual
```

over the configured warm-up period.

The effective blend is:

```text
reconstruction output =
    (1 - alpha) * true-residual output
    + alpha * predicted-residual output
```

where `alpha` increases with training step.

The direct residual loss still trains the RPM from the beginning. The curriculum only controls how strongly the predicted inverse influences the main recovery losses.

## Iteration 6: Secret-Focused Loss and Larger RPM

### Changes

To target a secret SNR around 30 dB, the loss and model capacity were increased:

```yaml
secret_time: 4.0
secret_stft: 3.0
latent_prediction: 0.5
predictor_learning_rate: 0.0003
oracle_warmup_steps: 5000
```

The RPM was temporarily increased to 256 hidden channels and 12 blocks. This was later revised because a more structured architecture was preferable to simply increasing width.

## Iteration 7: Current Multi-Scale U-Net RPM

### Current architecture

The current RPM is a 1D multi-scale encoder-decoder with skip connections:

```text
stego Haar coefficients
        |
        v
fine-scale encoder
        |
        v
2x downsampled encoder
        |
        v
4x downsampled bottleneck
        |
        v
upsample + skip connection
        |
        v
upsample + skip connection
        |
        v
predicted residual
```

The current RPM configuration is:

```yaml
predictor:
  hidden_channels: 128
  num_blocks: 9
  kernel_size: 3
```

The architecture uses:

- Dilated residual blocks at multiple scales.
- Strided convolutions for downsampling.
- A bottleneck with wider temporal context.
- Linear interpolation for upsampling.
- Skip connections to preserve local detail.
- A zero-initialized output layer for stable startup.

### Why this is a real architectural change

The previous predictor mostly processed audio at one temporal resolution. The new design can model both:

- Long-range structure such as slowly changing audio envelopes.
- Short-range detail needed to reconstruct the INN residual accurately.

### Current primary config

The current experiment is:

```text
name: rpm_unet_secret30_one_day_librispeech
output: runs/rpm_unet_secret30_one_day_librispeech
```

Important values:

```yaml
model:
  latent_mode: predicted
  secret_decode_mode: predicted
  predictor:
    hidden_channels: 128
    num_blocks: 9
  secret_decoder:
    hidden_channels: 128
    num_blocks: 10

loss:
  secret_time: 4.0
  secret_stft: 3.0
  latent_prediction: 0.5
  latent_prior: 0.0

training:
  max_steps: 200000
  max_hours: 24.0
  learning_rate: 0.0001
  predictor_learning_rate: 0.0003
  oracle_warmup_steps: 5000
```

The `secret_decoder` remains in the model for optional ablation, but `secret_decode_mode: predicted` ensures the primary result comes from the INN inverse and RPM.

## Validation and Test Status

The following checks passed after the current architecture was implemented:

- 13 unit tests passed.
- Ruff lint passed.
- Python configuration loading passed.
- CPU smoke training passed.
- Real GPU one-step training passed on an RTX 4090.
- Exact INN cycle error remained near machine precision in smoke tests.

A one-step run is not a quality result. It only confirms that the implementation executes.

The previous multi-scale experiment reached at least:

```text
step_00005000.pt
step_00010000.pt
step_00015000.pt
```

before the launcher returned exit code `1`. The exact Python traceback was not captured in the final report, so the failure cause still needs to be identified from the terminal output immediately before the wrapper error.

## Reproduction Command

The current run should start from scratch with:

```powershell
cd C:\Users\student6\Desktop\sbn_07\spectra_inn_windows

.\train_windows.ps1 `
  -DataDirectory ".\datasets" `
  -ConfigPath configs\rpm_one_day_librispeech.yaml `
  -MaxSteps 200000 `
  -TrainingHours 24 `
  -SkipInstall -SkipDownload -SkipTests -SkipSmoke
```

Do not pass `-Resume` when starting the current architecture for the first time. Older checkpoints have incompatible model shapes or represent earlier experiments.

Checkpoints are saved every 5,000 steps under:

```text
runs\rpm_unet_secret30_one_day_librispeech\checkpoints\
```

## Metrics to Monitor

The most important metrics are:

```text
validation/secret_snr_db_predicted
validation/latent_prediction_snr_db
validation/secret_snr_db_true_residual
validation/cover_snr_db
validation/exact_cycle_max
```

Interpretation:

- `secret_snr_db_predicted`: the real blind-decoding result using RPM plus INN inverse.
- `latent_prediction_snr_db`: how close the RPM prediction is to the true residual.
- `secret_snr_db_true_residual`: the oracle ceiling when the true residual is supplied.
- `cover_snr_db`: stego fidelity.
- `exact_cycle_max`: numerical invertibility check.

A large gap between true-residual SNR and predicted SNR means the RPM is still not predicting the discarded residual well enough.

## Current Technical Risk

The central risk is not necessarily insufficient RPM capacity. The INN can learn to place a large amount of secret information in the discarded residual. If that information is not encoded predictably in the stego waveform, no RPM can recover it reliably just by becoming larger.

If the current U-Net RPM remains below the target after 20k-50k steps, the next research change should be structural rather than another width increase. Possible directions are:

- Make the discarded latent lower-dimensional.
- Add an explicit rate or energy constraint on the residual.
- Add a dedicated predictable latent bottleneck.
- Train the INN to maximize secret recovery through predicted residuals from the beginning.
- Use a progressive residual representation instead of predicting the complete waveform residual.

The direct decoder can be used as an ablation to determine whether the stego contains enough secret information, but it should not be used to claim pure INN/RPM recovery.

## Bottom Line

The current architecture is substantially better motivated than the original flat RPM:

```text
multi-scale RPM
+ direct residual supervision
+ gradual predicted-latent curriculum
+ stronger secret objective
+ separate RPM learning rate
+ pure INN/RPM primary decode path
```

It is a stronger experiment, not a guaranteed 30 dB solution. The next decision must be based on the predicted-vs-true residual gap at meaningful checkpoints, not on the oracle score alone.
