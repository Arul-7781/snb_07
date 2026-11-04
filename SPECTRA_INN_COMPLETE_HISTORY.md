# SPECTRA-INN Project History, Architectural Changes, and Results

**Report date:** 4 November 2026  
**Repository:** `Arul-7781/snb_07`  
**Coverage:** From the original LF-VSN video prototype through the SPECTRA-INN audio
baseline, RPM experiments, WavInWav-inspired changes, branch organization, and the
prepared RTX 4090 study.

> **Evidence rule.** This report distinguishes completed measurements from software
> checks, proposed work, partial runs, and expectations. A high oracle score is not
> reported as ordinary blind-recovery quality. A one-step GPU run is not reported as a
> training result. The numerical results below are taken from the repository's experiment
> records and should be read with their stated protocols and limitations.

## 1. Executive summary

The project began as a video steganography implementation based on LF-VSN, an invertible
neural-network system that hides one or more secret videos inside a cover video. The
repository was then extended into an audio-focused SPECTRA-INN research scaffold. The
audio system uses an invertible network to map a cover/secret pair into a stego waveform
and a residual. Because the residual is not transmitted in blind decoding, a learned
Redundancy Prediction Module (RPM) predicts it from the stego signal.

The main progression was:

1. **LF-VSN video prototype:** single-secret Vimeo-90K training, deterministic data
   handling, PSNR/SSIM reporting, subset preparation, result plotting, and a Gradio
   encode/decode demo.
2. **SPECTRA-INN Stage 1 audio baseline:** 16 kHz mono speech, three-level Haar
   analysis/synthesis, eight affine invertible coupling blocks, and blind decoding
   using either a Gaussian latent or a learned RPM.
3. **RPM investigation:** a learned residual predictor replaced random Gaussian
   decoding. The INN was shown to invert almost exactly with the true residual, while
   predicted-residual recovery remained the bottleneck.
4. **Objective and context experiments:** direct residual supervision, larger/dilated
   predictors, curriculum ideas, a longer-context screen, and normalized MSE (NMSE)
   waveform loss.
5. **Measured 3060 result:** the fresh 10,000-step NMSE checkpoint reached **15.76 dB
   cover SNR, 7.90 dB secret SNR, and 6.57 dB secret SI-SDR** on 2,620 fixed test pairs.
   The historical 81,000-step Charbonnier reference reached **10.19 dB cover SNR,
   6.99 dB secret SNR, and 5.39 dB secret SI-SDR** under the same test protocol.
6. **WavInWav-inspired branch:** multi-scale coupling conditioners, channel
   perturbation, resampling augmentation, and numerical-stability safeguards were
   implemented. The primary path remained `stego -> RPM -> INN inverse`; no direct
   secret decoder was promoted to the main architecture.
7. **RTX 4090 study:** a reproducible sequential experiment bundle was prepared with
   matched controls, a multi-scale U-Net RPM candidate, validation gates, paired
   seeds, refinement, resume support, and a 24–30-hour budget. **The 4090 study has
   not been run as of this report.**

The strongest supported scientific conclusion is that the Stage 1 prototype works as an
experimental audio hiding system, and that the normalized waveform objective produced a
better measured checkpoint than the historical Charbonnier reference. That comparison is
not yet an equal-budget, multi-seed ablation. The central unresolved problem remains blind
prediction of the discarded residual.

## 2. Repository and branch timeline

| Date | Milestone | Main result or status |
|---|---|---|
| 27 Aug 2026 | LF-VSN video prototype added | End-to-end video train/test/demo workflow established. |
| 28 Aug 2026 | Video checkpoints added | Best/latest trained video checkpoints preserved. |
| 31 Aug 2026 | README organization updated | README versions renamed and consolidated. |
| 30 Sep 2026 | Audio research projects and checkpoints added | SPECTRA-INN audio scaffold and historical artifacts established. |
| 13 Oct 2026 | LF-VSN-style RPM added to SPECTRA-INN | Learned residual prediction replaced random latent as the main blind-decoding experiment. |
| 15 Oct 2026 | Audio experiment record completed | 3060 results, fixed-pair diagnostics, and limitations documented. |
| 29–30 Oct 2026 | 4090 study designed | Matched screen/refinement protocol and fresh training bundle prepared. |
| 4 Nov 2026 | Bundle committed and branches pushed | `feature/4090-training-bundle`, `audio-rpm-improve`, and `audio-wavinwav-inspired` are on `origin`. |

Important branches:

- `main` / `audio-training-results`: initial research projects and trained checkpoints.
- `audio-trial2`: LF-VSN-style RPM implementation.
- `audio-rpm-improve`: residual-prediction improvement work and iteration history.
- `audio-wavinwav-inspired`: multi-scale coupling and robustness/stability work.
- `feature/4090-training-bundle`: fresh RTX 4090 experiment suite and tests.
- `audio_pipeline`: original LF-VSN video prototype.

The branch names preserve experimental separation. Older checkpoints should not be resumed
with configurations whose architecture or optimizer settings differ.

## 3. Starting point: LF-VSN video prototype

### 3.1 Original purpose and architecture

The first project line implemented LF-VSN-style video steganography. Its conceptual flow
was:

```text
cover video + one or more secret videos
              |
              v
       invertible video network
              |
              +--> stego video
              +--> latent/residual information

stego video + decoding key/latent
              |
              v
       inverse network
              |
              v
       recovered secret video(s)
```

The upstream LF-VSN design targets large capacity and flexible recovery, including
multiple secrets and key-controlled extraction. This repository's first implemented
prototype narrowed the task to one secret video so it could be trained and debugged on
an RTX 3060.

### 3.2 Architecture and configuration modifications

The video prototype added or changed:

- A one-secret training configuration with `num_video: 1`, `gop: 3`,
  `single_secret_only: true`, and a 20,000-iteration prototype budget.
- A configuration guard in `LFVSN.py` so an incompatible secret count fails early.
- Configurable affine coupling clamp plumbing, with the prototype using `clamp: 0.9`.
- Safer test-offset handling instead of assuming only `[-1, 0, 1]`.
- Deterministic sorted frame ordering.
- A guard against selecting the same cover and secret sequence index.
- Dataset size and list-divisibility checks.
- Optional validation/test item limits for fast prototyping.
- PSNR and SSIM computation, TensorBoard logging, CSV export, and JSON summaries.
- A Vimeo-90K subset builder for constrained experiments.
- A plotting/report script for metric curves and comparison panels.
- A Gradio application for encode/decode and decode-existing-stego workflows.
- Robust video preprocessing for arbitrary upload lengths, frame pairing, resizing,
  even dimensions, reflection-padded temporal context, and FFmpeg fallback.
- Device-safe tensor allocation and removal of fragile BasicSR/torchvision imports.
- Compatibility handling for old and new Gradio queue parameter names.

### 3.3 Dataset and operational behavior

The prototype used Vimeo-90K triplets. The documented prototype subset contained:

- 3,000 training sequences.
- 400 test-list entries interpreted as 200 cover and 200 secret entries.
- Random 128-pixel crops.
- Shared crop coordinates for cover and secret.
- RGB conversion and standard flip/rotation augmentation.

For arbitrary demo videos, the app used the shorter of the two input lengths, optionally
subsampled frames, resized the encode inputs consistently, and created valid three-frame
contexts at sequence boundaries.

### 3.4 Results and limitations of the video phase

The repository preserved trained checkpoints and metric artifacts, but the available
project record does not provide a single consolidated numerical table comparable to the
later audio table. Therefore this report does not invent a video quality number.

The meaningful result of this phase was engineering completeness: a train -> test ->
analyze -> interactive-demo workflow, with persistent PSNR/SSIM artifacts and safer
handling of short or variable-length videos.

Operationally, the demo documented that compressed preview video can damage extraction.
The lossless stego export must be used for decode. This is an important architectural
and deployment constraint: the invertible signal is sensitive to channel distortion.

## 4. SPECTRA-INN Stage 1 audio baseline

### 4.1 Scope

SPECTRA-INN was created as a research scaffold for a larger progressive audio-INN
architecture, but intentionally implemented Stage 1 only:

- 16 kHz mono clean speech.
- Two-second cover and secret segments.
- Three-level perfect-reconstruction Haar transform.
- Eight bounded affine coupling blocks.
- Clean-audio training without codec simulation.
- Optional Gaussian-latent or predicted-latent blind decoding.

The planned router, secret codec, robust embedder/extractor, channel simulator,
steganalyzer, error correction, and A/V synchronization layers were documented but not
claimed as implemented.

### 4.2 Baseline architecture

The forward and inverse paths were:

```text
Encoding:
    cover + secret
        -> Haar analysis
        -> eight-block affine INN
        -> stego + true residual

Oracle decoding:
    stego + true residual
        -> inverse INN
        -> recovered cover + recovered secret

Blind baseline decoding:
    stego + Gaussian latent
        -> inverse INN
        -> recovered cover + recovered secret
```

The true residual is required for exact numerical inversion, but it is discarded in the
blind setting. The Gaussian path was therefore a deliberately weak baseline, not a
claim that the residual was recoverable from a random prior.

### 4.3 Baseline losses

The original Stage 1 objective combined:

| Loss term | Weight |
|---|---:|
| Stego waveform Charbonnier | 2.0 |
| Stego multi-resolution STFT | 1.0 |
| Recovered secret waveform Charbonnier | 1.0 |
| Recovered secret multi-resolution STFT | 1.0 |
| Recovered cover waveform | 0.25 |
| Gaussian latent prior in RPM runs | 0.0 |
| Coupling-scale stability | 0.001 |

Training used AdamW, mixed precision, gradient accumulation, clipping, checkpointing,
validation, and resume support. The common model had approximately 878,344 trainable
parameters, including approximately 200,776 RPM parameters.

### 4.4 Dataset protocol

The final documented LibriSpeech manifest contained:

| Split | Records | Hours | Speakers |
|---|---:|---:|---:|
| `train-clean-100` | 28,539 | 100.591 | 251 |
| `train-clean-360` | 104,014 | 363.606 | 921 |
| Combined training | 132,553 | 464.196 | 1,172 |
| `dev-clean` | 2,703 | 5.388 | 40 |
| `test-clean` | 2,620 | 5.403 | 40 |

Speaker intersections between training, validation, and test were reported as zero.
Training used random two-second crops; evaluation used deterministic fixed crops and
neighbor pairing. “Full validation” and “full test” mean all split records under that
protocol, not every possible crop or pair.

## 5. RPM evolution and architectural changes

### 5.1 Why a learned RPM was needed

The first audio blind decoder sampled a Gaussian latent. The LF-VSN-style change added
an RPM that predicts the missing residual from stego features:

```text
stego Haar coefficients
        -> RPM
        -> predicted residual
        -> inverse INN
        -> recovered cover + recovered secret
```

The first RPM was a one-dimensional residual convolution network with:

- Input convolution.
- Eight residual convolution blocks.
- LeakyReLU activation.
- Zero-initialized output projection.

The Gaussian latent-prior penalty was disabled because the predictor was no longer meant
to match an independent Gaussian sample.

### 5.2 Initial RPM result

The initial LF-VSN-style run reached approximately 200,000 steps. Its reported validation
measurements were:

| Decode mode | Cover SNR | Secret SNR |
|---|---:|---:|
| Predicted residual | 14.57 dB | 11.05 dB |
| Gaussian latent | — | 0.36 dB |
| True residual/oracle | — | 82.50 dB |

The true-residual score near 82.50 dB demonstrated that the reversible core could invert
accurately. The large gap between oracle and predicted residual scores isolated the main
problem: blind residual prediction, not basic INN numerical invertibility.

### 5.3 Direct residual-prediction supervision

An auxiliary Charbonnier loss was added:

```text
0.1 * Charbonnier(true residual, predicted residual)
```

The target was detached so the INN could not alter the target through this auxiliary
term. Tests, lint, and smoke training passed. A subsequent real run still reported
approximately 10.46 dB secret SNR after 16,000 steps. Direct residual matching alone
did not solve the blind-recovery bottleneck.

### 5.4 Larger and dilated RPM

The predictor was then increased from width 64 to 128 and from 8 to 10 residual blocks,
with expanded dilation and a separate RPM learning rate of `2e-4`. A 20,000-step
true-residual warm-up was introduced so the INN could learn reconstruction before the
predicted residual controlled the inverse path.

The hard oracle-to-predicted transition was recognized as risky because it can create a
loss discontinuity. This led to the gradual curriculum described next.

### 5.5 Gradual true-residual-to-predicted curriculum

Instead of switching abruptly, the reconstruction output was blended:

```text
output = (1 - alpha) * true-residual output
       + alpha * predicted-residual output
```

`alpha` increased from zero to one over warm-up. The RPM was trained from the beginning,
but its predicted inverse was introduced gradually into the main recovery losses.

This was a training-path modification, not a change to the final blind decoding
definition. The final score still had to use a predicted residual.

### 5.6 Secret-focused loss and capacity increase

An attempt was made to target stronger secret recovery by increasing:

```yaml
secret_time: 4.0
secret_stft: 3.0
latent_prediction: 0.5
predictor_learning_rate: 0.0003
oracle_warmup_steps: 5000
```

The RPM was temporarily increased to width 256 and 12 blocks. This was subsequently
reframed as a need for more structured multi-scale context rather than only increasing
width.

### 5.7 Multi-scale U-Net RPM

The proposed structured predictor became a one-dimensional encoder-decoder:

```text
fine-scale encoder
       -> 2x downsampled encoder
       -> 4x downsampled bottleneck
       -> upsample + skip connection
       -> upsample + skip connection
       -> predicted residual
```

Its intended benefits were:

- Long-range envelope and temporal structure from the bottleneck.
- Short-range residual detail through skip connections.
- More useful context than a flat local predictor.
- Zero-initialized output for stable startup.

The proposed configuration used width 128 and 9 blocks. A prior partial multi-scale run
produced 5k, 10k, and 15k checkpoints before a wrapper returned exit code 1, but the
final traceback was not captured. Therefore this is an implementation/proposal status,
not a validated quality result.

## 6. Measured 3060 experiments

### 6.1 Experiment inventory

The documented experiments were:

| ID | Experiment | What changed | Status |
|---|---|---|---|
| E0 | Synthetic and CPU smoke checks | Invertibility and execution checks | Completed; not a quality result. |
| E1 | Original RPM pilot | Initial Charbonnier RPM on real data | Completed. |
| E2 | Original RPM continued | Expanded corpus and longer reference | Completed; 81k reference checkpoint. |
| E3 | Longer-context RPM | Flat dilations through 128 | Small fixed-pair screen. |
| E4 | Matched short-context control | Same fixed-pair protocol as E3 | Completed control. |
| E5 | Loss-gradient inspection | No model updates; measured gradient contributions | Diagnostic only. |
| E6 | NMSE objective screen | Energy-normalized waveform error | Small fixed-pair screen. |
| E7 | Fresh real-data NMSE run | New 10k-step NMSE checkpoint | Completed and evaluated. |

### 6.2 Longer context

The original flat RPM had a direct receptive field of approximately 18.5 ms of stego
audio. The context trial used dilations:

```text
[1, 2, 4, 8, 16, 32, 64, 128]
```

for an approximate RPM receptive field of 512.5 ms. In the matched 16-pair, 1,000-step
diagnostic, longer context did not show a clear advantage. This does **not** prove that
all longer-context predictors are worse; it only says this particular one-seed screen
did not justify a full run.

### 6.3 NMSE objective

The NMSE experiment retained the short-context RPM and replaced the waveform terms for
stego, secret, and recovered cover with per-example energy-normalized MSE:

```text
NMSE = mean((estimate - reference)^2)
       / max(mean(reference^2), 0.000001)
```

The diagnostic was motivated by the observation that the old spectral secret gradient
could dominate the absolute waveform term. The NMSE objective gives quiet and loud
examples a more comparable relative-error contribution.

### 6.4 Main held-out results

The final comparison used 2,620 fixed two-second pairs from `test-clean`:

| Metric | Original RPM + Charbonnier, 81k | RPM + NMSE, 10k | Difference |
|---|---:|---:|---:|
| Cover SNR | 10.19 dB | **15.76 dB** | **+5.56 dB** |
| Secret SNR, predicted residual | 6.99 dB | **7.90 dB** | **+0.91 dB** |
| Secret SI-SDR, predicted residual | 5.39 dB | **6.57 dB** | **+1.17 dB** |
| Secret SNR after PCM16 simulation | 6.99 dB | **7.90 dB** | **+0.91 dB** |

The NMSE checkpoint preserved the cover better and recovered the secret better under
this evaluation protocol. PCM16 rounding had negligible additional effect at the
reported precision.

### 6.5 Correct interpretation

The 10k versus 81k comparison is a checkpoint comparison with unequal training histories,
not a fully controlled equal-budget ablation. The small matched diagnostics support the
objective change, but a fresh equal-step Charbonnier control is still needed.

The oracle score is not the blind score:

```text
true residual -> near-exact inverse check
predicted residual -> actual recipient/decoder condition
```

The measured 7.90 dB predicted-residual secret SNR is therefore the relevant blind
recovery value, not the 80–100 dB oracle values seen in numerical inversion tests.

## 7. WavInWav-inspired architectural branch

### 7.1 Motivation

The public WavInWav design was reviewed for useful ideas including direct waveform-domain
coupling, dense residual conditioners, multi-scale temporal context, channel/noise
perturbations, resampling augmentation, adversarial STFT training, and embedding
distillation.

Only ideas compatible with the SPECTRA-INN research goal were implemented. The primary
requirement remained:

```text
stego -> RPM -> predicted residual -> INN inverse -> recovered secret
```

### 7.2 Multi-scale coupling conditioners

The previous coupling conditioner processed one temporal resolution. The new
`ResidualSubNet1d` uses parallel dilated branches:

```text
input
  -> dilation 1 branch
  -> dilation 2 branch
  -> dilation 4 branch
  -> dilation 8 branch
  -> concatenate and 1x1 fuse
  -> residual refinement
  -> output
```

The fused conditioner is used by the affine coupling subnets that update the cover
stream, predict affine scale, and update the secret/residual stream. This modifies the
reversible core itself, rather than only changing the RPM.

### 7.3 Channel and resampling augmentation

During training only, the stego waveform may be corrupted before RPM decoding:

```text
clean stego
  -> additive Gaussian noise and/or multiplicative speckle
  -> optional 0.9x–1.1x resample and restore
  -> RPM
  -> predicted residual
  -> inverse INN
```

The stable configuration used reduced probabilities and amplitudes after stronger
augmentation became unstable. Validation remained clean so that the effect measured
decoder robustness rather than training-time corruption of the evaluation.

### 7.4 Numerical-stability changes

The first WavInWav-inspired run reached approximately step 9,789 and then failed with a
non-finite loss. The branch added:

- Explicit bfloat16 AMP support for the RTX 4090.
- Lower affine clamp, from 1.5 to 0.75.
- Lower main learning rate, from `1e-4` to `5e-5`.
- Lower RPM learning rate, from `3e-4` to `1e-4`.
- Stronger gradient clipping, from 5.0 to 2.0.
- Pre-backpropagation loss finiteness checks.
- Post-unscale gradient finiteness checks.
- Discarding invalid accumulation/update batches.
- A bounded number of transient bad batches followed by an explicit failure.
- A fresh output directory so incompatible old checkpoints are not mixed.

These are implementation and reliability improvements. The available record does not
contain a completed held-out quality table for this branch, so no quality gain is
claimed.

### 7.5 Deliberately excluded ideas

The following remain unpromoted or unimplemented in the primary path:

- Direct secret decoder as the main result.
- Random-latent decoding as the main method.
- WavLM embedding distillation.
- STFT adversarial discriminator.
- Claims of MP3/AAC/Opus robustness.
- Cryptographic encryption or key management.
- Formal steganalysis resistance.

## 8. Prepared RTX 4090 study

### 8.1 Purpose

The 4090 bundle is a new, sequential study rather than a continuation of old runs. It
was designed to answer whether structured RPM context helps under controlled budgets
while preserving the successful NMSE recipe and avoiding an oracle-training shortcut.

No 4090 quality run had started when the plan was written, and no completed 4090 result
is present in this repository snapshot.

### 8.2 Matched experiment matrix

| Arm | Predictor | Approximate parameters | Question |
|---|---|---:|---|
| A `a_nmse_control` | Original flat RPM, width 64, 8 blocks | 878,344 total | Contemporary NMSE control. |
| B `b_flat_capacity` | Flat RPM, width 152, 9 blocks | 1,935,376 total | Does added capacity alone help? |
| C `c_unet` | Three-scale U-Net, width 128, 9 blocks | 1,899,080 total | Does multi-scale context help? |
| D `d_unet_latent` | Same U-Net plus light residual NMSE | 1,899,080 total | Does detached-target residual supervision help? |

The flat-capacity and U-Net arms are approximately parameter matched. All arms share
data, seed, crop duration, effective batch, objective, INN, learning rate, monitoring,
and promotion rules as far as the study permits.

### 8.3 Sequential phases

The runner is designed to:

1. Preflight CUDA, GPU identity, disk space, versions, and the local dataset.
2. Profile actual 4090 throughput and choose safe equal-step budgets.
3. Run four matched screen arms.
4. Promote candidates using validation gates rather than test results.
5. Run a second paired seed.
6. Continue finalists for equal-step refinement.
7. Freeze final evaluation and write a report.

The normal total budget is 28 hours, with a configurable 24–30-hour machine reservation.
The suite supports checkpoint retention, per-job logs, explicit failure states, and resume
from a saved suite state.

### 8.4 What success would and would not mean

A 4090 run could provide a better controlled architecture comparison and a throughput
measurement. It would not automatically prove 30 dB recovery, codec robustness,
imperceptibility, or security. The central residual-information bottleneck remains a
scientific risk.

## 9. Validation and engineering work

The audio repository accumulated software checks alongside model experiments:

- Exact Haar analysis/synthesis tests.
- INN invertibility tests.
- Coupling and redundancy predictor tests.
- Loss and configuration tests.
- Dataset/manifest and evaluation-pair tests.
- Overfit and smoke-training checks.
- LibriSpeech preparation tests.
- Training-bundle integrity tests.
- Ruff lint checks.
- CPU smoke training.
- One-step RTX 4090 execution checks.

The documented test counts grew from 25 to 47 and then to 71 as the implementation
expanded. These checks establish software behavior and numerical invariants; they do not
establish audio quality. The 4090 bundle includes a fresh test suite and a `pyproject.toml`
with reproducible setup instructions.

## 10. What has not been demonstrated

The following claims must not be made from the current evidence:

- Exact blind secret recovery.
- Imperceptible or transparent stego audio.
- MP3, AAC, Opus, filtering, resampling, or synchronization robustness as a completed
  result.
- Security against trained steganalysis.
- Cryptographic confidentiality.
- Formal speech intelligibility or human listening-study performance.
- A fair equal-budget, multi-seed superiority claim for NMSE over Charbonnier.
- A completed 4090 architecture result.
- Integration of the Stage 1 audio system with the original video demo.
- A complete 48 kHz stereo progressive payload system.

## 11. Recommended next steps

1. Run the prepared 4090 suite in a persistent terminal and preserve its generated
   `REPORT.md`, `suite_state.json`, logs, checkpoints, and evaluation artifacts.
2. Run a fresh equal-step Charbonnier control using the same data, seed, and monitoring
   protocol as the NMSE candidate.
3. Repeat the best architecture with a second seed and report confidence intervals or
   per-speaker distributions where possible.
4. Evaluate multiple crops and independent cover/secret pairing policies.
5. Add listening and intelligibility evaluation before making perceptual claims.
6. Test controlled channel distortions one at a time: PCM conversion, resampling,
   filtering, and codecs.
7. Only after the clean baseline is understood, add robust payload coding, error
   correction, steganalysis evaluation, and eventual audio/video integration.

## 12. Source records and reproducibility map

The detailed source documents are:

- [Audio experiment history and numerical results](spectra_inn_4090_training/docs/experiment_history_and_results.md)
- [RPM iteration history](spectra_inn_windows/docs/RPM_ITERATION_HISTORY.md)
- [WavInWav-inspired changes](spectra_inn_windows/docs/WAVINWAV_FOLLOWUP_CHANGES.md)
- [4090 experimental plan](spectra_inn_4090_training/docs/4090_experiment_plan.md)
- [4090 training instructions](spectra_inn_4090_training/TRAINING_README.md)
- [Audio mentor briefing](spectra_inn_4090_training/docs/mentor_results_briefing.md)
- [LF-VSN implementation summary](sbn_07_video/IMPLEMENTATION_SUMMARY.md)
- [SPECTRA-INN Windows README](spectra_inn_windows/README.md)

The current branch organization and pushed branches are available on the repository
remote. The 4090 bundle commit is `6a36473` (`Add RTX 4090 training bundle`).

## 13. Final conclusion

The project has progressed from a video invertible-steganography prototype into a
carefully instrumented audio-INN research program. The most important architectural
lesson is that reversible numerical reconstruction and blind residual prediction are
different problems. The INN can reconstruct almost exactly when its residual is known;
the practical decoder must infer that residual from stego audio, and that inference
remains imperfect.

The best completed measured audio checkpoint is the 10,000-step NMSE model, with
15.76 dB cover SNR, 7.90 dB predicted-residual secret SNR, and 6.57 dB secret SI-SDR
on the documented 2,620-pair test protocol. This is a meaningful prototype result, but
not a finished robust or secure steganography system. The prepared 4090 study is the
next controlled experiment intended to determine whether additional RPM capacity,
multi-scale context, or residual supervision improves blind recovery without relying on
oracle residuals.
