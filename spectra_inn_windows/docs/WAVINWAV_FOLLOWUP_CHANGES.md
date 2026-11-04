# SPECTRA-INN Changes After the WavInWav Review

## Scope

This document records the changes made after reviewing the public `odunola499/wavinwav` repository. The goal was to bring over the useful architectural ideas while preserving the original SPECTRA-INN requirement:

```text
stego -> RPM -> predicted residual -> INN inverse -> recovered secret
```

The direct secret decoder remains available as an optional ablation, but it is not the primary path in the current configuration.

The changes are on branch:

```text
audio-wavinwav-inspired
```

## Starting Point

Before the WavInWav-inspired changes, the audio pipeline had:

- Haar packet analysis/synthesis.
- Eight affine INN coupling blocks.
- Simple dilated residual conditioner subnetworks.
- An LF-VSN-style residual prediction module.
- Direct residual prediction loss.
- A gradual true-residual-to-predicted-residual curriculum.
- 16 kHz mono LibriSpeech audio.
- `train-clean-360` for training.
- `dev-clean` for validation.
- `test-clean` for testing.

The previous results were poor:

```text
predicted secret SNR: approximately 10-11 dB
true-residual secret SNR: approximately 80-100 dB
```

That gap showed that the INN itself could invert accurately, but the RPM could not predict the discarded residual accurately enough.

## WavInWav Ideas Reviewed

The WavInWav repository uses several relevant ideas:

- Direct waveform-domain coupling blocks.
- Dense residual conditioner subnetworks.
- Multiple temporal scales through dilated convolutions.
- Random channel/noise perturbation during training.
- Optional resampling augmentation.
- Optional STFT adversarial training.
- Optional WavLM embedding distillation.

Not all ideas were copied directly. The reference repository also has risks, including potentially detached STFT and adversarial gradients, random-latent blind decoding, and incomplete public result reporting.

## Change 1: New Experiment Branch

A separate branch was created:

```text
audio-wavinwav-inspired
```

This protects the previous `audio-rpm-improve` work and allows the WavInWav-inspired architecture to be evaluated independently.

No existing dataset or previous checkpoints were deleted.

## Change 2: Multi-Scale INN Coupling Conditioners

### Previous design

Each affine INN block used a conditioner with:

- One input convolution.
- A sequence of dilated residual convolutions.
- Group normalization.
- SiLU activation.
- One output convolution.

All processing occurred at one temporal resolution.

### New design

`ResidualSubNet1d` in `src/spectra_inn/models/coupling.py` now has multiple parallel dilated branches.

The default configuration uses four branches with dilation values:

```text
1, 2, 4, 8
```

Each branch processes the hidden representation at a different temporal scale:

```text
input
  -> branch dilation 1
  -> branch dilation 2
  -> branch dilation 4
  -> branch dilation 8
  -> concatenate branches
  -> 1x1 fusion convolution
  -> residual refinement blocks
  -> output projection
```

The fused output is used by all three subnetworks inside every affine coupling block:

```text
phi: updates the cover stream
rho: predicts the affine log scale
eta: updates the secret/residual stream
```

### Why this was added

The discarded residual is audio with both short- and long-range structure. A single local conditioner may not see enough context to predict it. The parallel dilated branches allow the INN to model local waveform details and broader temporal patterns simultaneously.

### Configuration

```yaml
model:
  subnet_scales: 4
```

This changes the reversible core itself, not just the RPM.

## Change 3: Existing RPM Kept as the Primary Decoder

The RPM remains the main blind decoder:

```text
stego Haar coefficients
    -> multi-scale RPM
    -> predicted residual
    -> INN inverse
    -> recovered secret
```

The WavInWav-inspired changes do not replace the RPM with a random latent. Random Gaussian decoding remains available only as a baseline comparison.

The primary configuration remains:

```yaml
model:
  latent_mode: predicted
  secret_decode_mode: predicted
```

`secret_decode_mode: direct` exists only for ablation and is not used by the current experiment.

## Change 4: Channel and Noise Augmentation

The training loop now optionally corrupts the stego waveform before RPM decoding.

The augmentation is applied only during training, not validation:

```text
clean stego from INN
    -> optional channel corruption
    -> RPM
    -> predicted residual
    -> INN inverse
```

### Supported perturbations

#### Additive Gaussian noise

```text
stego + random Gaussian noise
```

#### Multiplicative speckle noise

```text
stego + stego * random noise
```

#### Resampling perturbation

The stego is temporarily resampled by a random factor and then returned to its original length. The current range is approximately:

```text
0.9x to 1.1x
```

### Current stable settings

```yaml
training:
  channel_augment_probability: 0.25
  channel_noise_std: 0.001
  channel_speckle_std: 0.005
  channel_resample_probability: 0.1
  channel_resample_min: 0.9
  channel_resample_max: 1.1
```

The augmentation was intentionally reduced after the first version became numerically unstable. The purpose is to improve decoder robustness without overwhelming the clean hiding task.

## Change 5: Numerical Stability Improvements

The first WavInWav-inspired run reached approximately step 9,789 and then failed with:

```text
FloatingPointError: Non-finite loss at step 9790: nan
```

The failure occurred after a loss increase and was associated with numerical instability in the larger multi-scale model under float16 mixed precision.

### Stability changes

#### bfloat16 AMP

The training loop now supports an explicit AMP dtype and the stable config uses:

```yaml
amp: true
amp_dtype: bfloat16
```

The RTX 4090 supports bfloat16, which provides a larger exponent range than float16 and reduces overflow risk.

#### Lower INN clamp

```yaml
clamp: 1.5 -> 0.75
```

The affine scale is bounded more conservatively, reducing exponential amplification inside coupling blocks.

#### Lower learning rates

```yaml
learning_rate: 0.0001 -> 0.00005
predictor_learning_rate: 0.0003 -> 0.0001
```

The larger multi-scale conditioner and RPM receive smaller updates to reduce parameter explosion.

#### Stronger gradient clipping

```yaml
grad_clip_norm: 5.0 -> 2.0
```

This limits unusually large update norms.

#### Non-finite batch handling

The training loop now:

1. Checks the loss before backpropagation.
2. Discards the entire gradient accumulation if the loss is non-finite.
3. Checks the gradient norm after unscaling.
4. Discards updates with non-finite gradients.
5. Allows a small number of transient bad batches.
6. Stops with a clear error if non-finite values persist.

This prevents one corrupt batch from silently updating the model with NaN values.

## Change 6: Fresh Experiment Output

The stable configuration uses a fresh experiment name:

```yaml
experiment:
  name: rpm_wavinwav_stable_one_day_librispeech
  output_dir: runs/rpm_wavinwav_stable_one_day_librispeech
```

This prevents the new stable architecture from mixing with checkpoints from:

- `rpm_one_day_librispeech`
- `rpm_improve_one_day_librispeech`
- `rpm_arch2_one_day_librispeech`
- `rpm_arch3_one_day_librispeech`
- `rpm_arch4_one_day_librispeech`
- `rpm_arch5_secret30_one_day_librispeech`
- `rpm_unet_secret30_one_day_librispeech`

Old checkpoints should not be resumed because their architecture or optimizer settings differ.

## Current Stable Configuration

The current primary configuration contains:

```yaml
model:
  wavelet_levels: 3
  num_blocks: 8
  hidden_channels: 64
  subnet_layers: 4
  subnet_scales: 4
  clamp: 0.75
  kernel_size: 3
  latent_mode: predicted
  secret_decode_mode: predicted
  predictor:
    hidden_channels: 128
    num_blocks: 9
    kernel_size: 3

loss:
  stego_time: 2.0
  stego_stft: 1.0
  secret_time: 4.0
  secret_stft: 3.0
  recovered_cover: 0.25
  latent_prediction: 0.5
  latent_prior: 0.0
  scale_stability: 0.001

training:
  max_steps: 200000
  max_hours: 24.0
  learning_rate: 0.00005
  predictor_learning_rate: 0.0001
  oracle_warmup_steps: 5000
  amp: true
  amp_dtype: bfloat16
  grad_clip_norm: 2.0
```

## Validation Performed

After the architecture and stability changes:

- 13 unit tests passed.
- Ruff lint passed.
- Configuration loading passed.
- CPU smoke training passed.
- Multi-scale coupling forward/inverse execution passed.
- Channel augmentation execution passed.
- Exact INN cycle error remained near machine precision in smoke tests.

The smoke result is only a plumbing check. It is not evidence of final secret-recovery quality.

## Training Command

The dataset is already installed and the manifest is present. Start the stable experiment with:

```powershell
cd C:\Users\student6\Desktop\sbn_07\spectra_inn_windows

.\train_windows.ps1 `
  -DataDirectory ".\datasets" `
  -ConfigPath configs\rpm_one_day_librispeech.yaml `
  -MaxSteps 200000 `
  -TrainingHours 24 `
  -SkipInstall -SkipDownload -SkipTests -SkipSmoke
```

The launcher reuses the existing dataset because `-SkipDownload` is supplied.

## Output Locations

Checkpoints:

```text
runs\rpm_wavinwav_stable_one_day_librispeech\checkpoints\
```

TensorBoard logs:

```text
runs\rpm_wavinwav_stable_one_day_librispeech\tensorboard\
```

Run metadata:

```text
runs\rpm_wavinwav_stable_one_day_librispeech\config.resolved.yaml
runs\rpm_wavinwav_stable_one_day_librispeech\run_metadata.json
```

## Metrics to Monitor

The most important metrics remain:

```text
validation/secret_snr_db_predicted
validation/latent_prediction_snr_db
validation/secret_snr_db_true_residual
validation/cover_snr_db
validation/exact_cycle_max
```

Interpretation:

- `secret_snr_db_predicted`: actual pure INN/RPM secret recovery.
- `latent_prediction_snr_db`: RPM similarity to the true residual.
- `secret_snr_db_true_residual`: oracle numerical-inversion ceiling.
- `cover_snr_db`: stego fidelity.
- `exact_cycle_max`: reversible-core correctness.

## What Was Deliberately Not Added

The following WavInWav ideas were reviewed but not enabled in the primary experiment:

### WavLM distillation

The reference repository includes optional WavLM embedding losses. They were not added yet because the immediate problem is residual prediction and INN stability. A perceptual embedding loss should be added only after the waveform path is stable and its gradients are verified.

### Adversarial discriminator

The reference repository includes a multi-scale STFT discriminator. It was not added to this branch because adversarial training would add another unstable optimization loop while the generator itself is still producing NaNs.

### Random-latent decoding

The reference repository uses random latent replacement. That remains a baseline in SPECTRA-INN but is not used by the primary RPM experiment.

## Expected Outcome and Risk

These changes should make the architecture more expressive and the training process more stable. They do not guarantee a 30 dB secret SNR.

The largest remaining risk is structural: the INN may still place too much secret information in the discarded residual, making full residual prediction impossible. If the stable experiment remains near 10-11 dB after 20k-50k steps, the next change should be a compact predictable latent representation rather than another increase in RPM capacity.

The stable experiment should be judged by measured predicted secret SNR, not by the true-residual oracle score.
