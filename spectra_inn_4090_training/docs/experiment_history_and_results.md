# SPECTRA-INN: experiment history and results

**Record date:** 15 October 2026

**Project scope:** `spectra_inn_windows` only

**Current status:** Training is paused. The final 10,000-step NMSE model has been evaluated on validation and held-out test crops.

This document explains what was actually run, why each experiment was tried, what the results mean, and what is still unproven. It covers user-run training and the supporting checks and evaluations performed during the same workflow.

The main evidence is the saved training logs, TensorBoard events, resolved configurations, checkpoints, and evaluation `metrics.json` files. Numerical tables are rounded for readability. Where a measurement came from an offline diagnostic rather than a saved evaluation JSON, that is stated explicitly. Proposed experiments are not counted as completed experiments.

For a shorter version to use in a meeting, see [the mentor briefing](mentor_results_briefing.md).

### Contents

1. [Result summary](#1-the-result-in-one-paragraph)
2. [Task and terminology](#2-what-the-system-is-trying-to-do)
3. [Implemented scope](#3-what-was-implemented-and-what-was-not)
4. [Hardware, software, and setup](#4-hardware-software-and-setup-work)
5. [Dataset and splits](#5-dataset-and-split-protocol)
6. [Model and training settings](#6-common-model-and-training-settings)
7. [Actual experiment inventory](#7-inventory-what-actually-ran)
8. [Chronological record](#8-chronological-experiment-record)
9. [Final numerical results](#9-final-validation-and-test-results)
10. [How to interpret the metrics](#10-how-to-read-the-numbers-correctly)
11. [Supported conclusions](#11-conclusions-supported-by-the-experiments)
12. [Limitations](#12-limitations-and-unresolved-issues)
13. [Evidence and artifacts](#13-evidence-and-artifacts-to-keep)
14. [Proposed but unrun experiments](#14-what-was-proposed-but-not-run)
15. [Future work](#15-sensible-next-work-when-training-resumes)
16. [Plain-language conclusion](#16-final-plain-language-conclusion)

## 1. The result in one paragraph

We built and ran a working speech-in-speech hiding prototype on a Ryzen 5 3600 and an RTX 3060 12 GB. The original RPM model was trained to a saved checkpoint at 81,000 steps. Its secret recovery improved early but then showed limited further improvement. We tested a longer-context predictor on a small, fixed training set; that particular change did not help in the screening comparison. We then replaced the waveform loss with energy-normalized mean squared error, keeping the short-context model. This worked better in the small diagnostic and was followed by a fresh, real-data 10,000-step run. On the same held-out test-crop protocol, the NMSE checkpoint achieved **15.76 dB cover SNR and 7.90 dB secret SNR**, compared with **10.19 dB and 6.99 dB** for the original 81k checkpoint.

### Main test results

These results cover **2,620 test records**, evaluated as fixed two-second cover/secret pairs.

| Test measurement | Original RPM + Charbonnier, 81k steps | RPM + normalized MSE, 10k steps | Difference |
|---|---:|---:|---:|
| Cover SNR | 10.19 dB | **15.76 dB** | **+5.56 dB** |
| Secret SNR, predicted residual | 6.99 dB | **7.90 dB** | **+0.91 dB** |
| Secret SI-SDR, predicted residual | 5.39 dB | **6.57 dB** | **+1.17 dB** |
| Secret SNR after simulated PCM16 rounding | 6.99 dB | **7.90 dB** | **+0.91 dB** |

**What this supports:** the final NMSE checkpoint preserved the cover better and recovered the secret better under this evaluation protocol.

**What it does not establish:** an equal-budget comparison, statistical significance across seeds, imperceptible hiding, lossless recovery, resistance to compression, or a finished steganography system. The two main checkpoints had different training histories and numbers of steps.

## 2. What the system is trying to do

The task is to hide one speech signal inside another speech signal.

| Term | Simple meaning |
|---|---|
| Cover | The ordinary audio that carries the hidden audio. |
| Secret | The audio that we want to hide and later recover. |
| Stego | The modified cover audio containing the hidden information. |
| Residual or latent | A second output produced by the invertible network. It contains information needed for exact numerical inversion. |
| INN | An invertible neural network: its internal forward operation has a corresponding reverse operation when all required outputs are available. |
| RPM | Redundancy Prediction Module. It tries to predict the missing residual from stego audio. |
| Blind recovery | Recovering the secret using stego and the learned model, without supplying the original cover or the true residual. |
| Checkpoint | A saved model and training state that can be evaluated or resumed. |
| Step | One training-loop step, including the configured accumulation of smaller batches. It is not an epoch. AMP can skip an optimizer update when gradients overflow, so the displayed counter is not a guarantee that every update was applied. |

The implemented mapping is:

```text
Encoding:
    cover + secret -> invertible network -> stego + true residual

Real, stego-only decoding:
    stego -> RPM -> predicted residual
    stego + predicted residual -> reverse network -> recovered cover + recovered secret

Oracle check:
    stego + true residual -> reverse network -> almost exact original signals
```

The oracle has extra information: the true residual. Its high score is not the score available to a recipient who only receives stego audio.

Also, cover SNR and oracle secret SNR are different measurements. Cover SNR measures how much the transmitted cover was changed. Oracle secret SNR measures how accurately the secret is reconstructed when the decoder is given the missing residual. They should not be treated as interchangeable targets.

## 3. What was implemented, and what was not

### Implemented and exercised

- Two-second, 16 kHz, mono cover and secret signals.
- A three-level Haar transform and an eight-block invertible core.
- A learned RPM for stego-only decoding.
- Original Charbonnier waveform supervision with multi-resolution STFT losses.
- An optional dilated RPM for the context experiment.
- An optional normalized-MSE waveform objective.
- Real-data training, mixed precision, gradient accumulation, clipping, checkpointing, and resume.
- Fixed-pair training diagnostics that reuse the same cached audio tensors.
- Validation/test scoring, simulated PCM16 quantization checks, and WAV example export.

### Not implemented or not demonstrated by these experiments

- The complete progressive, codec-layered architecture described in the research plan.
- MP3/AAC/Opus attack training or demonstrated resistance to those codecs.
- A psychoacoustic router, error-correction pipeline, or robust payload extractor.
- 48 kHz stereo cover integration or integration with video.
- A cryptographic encryption or key-management boundary.
- Security against a trained steganalysis detector.
- Exact secret recovery from stego alone.
- A formal listening study or a measured speech-intelligibility study.

The work is therefore a **Stage-1 audio hiding prototype and experimental baseline**, not the completed larger system.

## 4. Hardware, software, and setup work

### Machine and environment used

| Item | Recorded setting |
|---|---|
| CPU | AMD Ryzen 5 3600, 6 cores / 12 threads |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB VRAM |
| System memory | Approximately 16 GB |
| Operating environment | Linux |
| Python | 3.11.15 in the project `.venv` |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| NVIDIA driver | 595.91.07 |
| Experiment seed | 3407 |

The directory name contains `windows`, but the experiments in this record ran on Linux, not through the PowerShell launcher.

### Setup problems that were fixed

1. The initial active Python was 3.13 and did not have the required training packages. A Python 3.11 environment was created.
2. The first test attempt failed during collection because PyTorch was missing. A successful Ruff check at that point only meant the style check passed; it did not mean the tests passed.
3. The CUDA PyTorch installation timed out while downloading `nvidia-cusparselt-cu12==0.7.1` from the NVIDIA package endpoint.
4. That exact dependency was installed from official PyPI. The PyTorch installation was retried with a longer HTTP timeout and fewer concurrent downloads.
5. PyTorch then reported the RTX 3060 correctly, and the initial **25 tests passed**.
6. The test suite grew with the implementation: **47 tests passed** after the context/fixed-pair work, and **71 passed** after the NMSE and full-cache reporting work. Ruff also passed at those checkpoints.

These were environment and software-validation steps, not model-quality experiments. Passing tests does not establish good audio hiding quality.

### Storage strategy and observed usage

The user could allocate around 70 GB, but the inspected project filesystem initially had about **50.8 GiB free**. Dataset preparation streamed each archive into extraction, checked its published MD5, and did not retain the compressed archive afterward.

Rounded disk usage observed after the experiments:

| Item | Approximate allocated disk usage |
|---|---:|
| Python environment | 6.8 GiB |
| Extracted `train-clean-100` | 6.3 GiB |
| Extracted `train-clean-360` | 23 GiB |
| Extracted `dev-clean` | 349 MiB |
| Extracted `test-clean` | 357 MiB |
| All current run outputs | 193 MiB |

These are rounded `du -sh` readings, not exact byte counts. A complete training checkpoint is approximately **11 MB**. The normal run configurations retain only three checkpoints per run. No full decoded-waveform cache was created.

An interrupted `train-clean-360` download left about 70 MB in `.partial-train-clean-360`. That folder was moved aside as a backup, leaving verified datasets intact, and the split was downloaded again from the beginning. Streaming reduced peak disk use but did not support resuming the compressed download. The successful final dataset fit with approximately 13 GiB of free space remaining.

## 5. Dataset and split protocol

The dataset is LibriSpeech clean English read speech, distributed under CC BY 4.0. Both cover and secret came from speech; these were not music-cover experiments.

The final manifest was audited directly:

| Official subset | Role | Audio records | Actual audio hours | Speaker IDs |
|---|---|---:|---:|---:|
| `train-clean-100` | Training | 28,539 | 100.591 | 251 |
| `train-clean-360` | Training | 104,014 | 363.606 | 921 |
| **Combined training** | **Training** | **132,553** | **464.196** | **1,172** |
| `dev-clean` | Validation | 2,703 | 5.388 | 40 |
| `test-clean` | Held-out test | 2,620 | 5.403 | 40 |

The familiar “100 + 360 = 460 hours” description uses dataset names. The actual manifest contains approximately **464.2 training hours**.

Speaker IDs were checked from the LibriSpeech paths. The observed speaker-ID intersections were **zero** for train/validation, train/test, and validation/test.

### How audio was sampled

- The pilot used `train-clean-100`; the original run then continued on the expanded training manifest.
- The final NMSE real-data run used the expanded training set from the beginning.
- Normal training drew random two-second crops and paired a cover record with another record from the same training split. Different records do not necessarily mean different speakers.
- Normal training included random gain changes of up to ±3 dB and a minimum-RMS check of 0.003.
- In fixed-pair diagnostics, 16 pairs were selected with the seed, then decoded, cropped, and augmented once. The same tensors were reused afterward.
- Evaluation used deterministic crops from the beginning of records and paired a record with its neighbor. Neighboring records can belong to the same speaker.
- The loader can retry another record when a crop is too quiet. Therefore, the reported number of processed records should not be interpreted as proof that every evaluated pair is unique and independent.

### Important meaning of “full validation” and “full test”

“Full” means the evaluator processed all records in that split under this crop/pairing protocol. It does **not** mean every second of every recording, every possible pair, or multiple random crops per recording were evaluated.

The evaluation commands used batch size 4 and a maximum of 1,000 batches, which is enough to cover the 2,703 validation records or the 2,620 test records.

### Dataset evidence

- Manifest: `data/manifests/rpm_librispeech.jsonl`.
- Summary: `data/manifests/rpm_librispeech.summary.json`.
- Manifest SHA256 at documentation time:

```text
94c8a742b7282f28ea913460aa97bb08e60ba0c195f61a3faa55adde8f29e5b2
```

The 464-hour figure describes available source audio. It is not a claim that each experiment trained on every audio second, or that a step corresponds to one complete pass through the corpus.

## 6. Common model and training settings

Unless stated otherwise, the real-data experiments used:

| Setting | Value |
|---|---|
| Cover / secret format | 16 kHz, mono, two seconds each |
| Samples per signal | 32,000 |
| Haar levels | 3 |
| Invertible coupling blocks | 8 |
| Hidden width | 64 |
| Conditioner depth setting | 4 |
| Kernel size | 3 |
| Affine log-scale clamp setting | 1.5 |
| Decoder latent | Predicted by RPM |
| RPM blocks / width | 8 / 64 |
| Total trainable parameters | **878,344** |
| RPM parameters | **200,776** |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.000001 |
| Micro-batch | 4 cover/secret pairs |
| Gradient accumulation | 4 micro-batches |
| Effective batch | **16 pairs** |
| Gradient clipping threshold | 5.0 |
| Mixed precision | Enabled for training |
| Normal data workers | 4 |
| Training log interval | 25 steps |
| Normal validation / checkpoint interval | 1,000 steps |
| Training-time validation | First 20 batches: normally 80 pairs |
| STFT windows | 256, 512, 1,024 samples |

Fixed-pair caching used zero loader workers after the cache was built. Offline evaluation used float32 inference. All tested RPM variants had the same parameter count; changing dilation changed where convolutions looked, not the number of weights.

### Original loss

The original waveform term was Charbonnier loss, a smooth form of absolute sample error. Other terms compared multi-resolution STFT magnitudes, recovered cover, and affine scales.

The shared weights were:

| Term | Weight |
|---|---:|
| Stego versus cover waveform | 2.0 |
| Stego versus cover STFT | 1.0 |
| Recovered secret waveform | 1.0 |
| Recovered secret STFT | 1.0 |
| Recovered original cover waveform | 0.25 |
| Gaussian latent-prior penalty | 0.0 in RPM runs |
| Scale stability | 0.001 |

No direct residual-matching loss was used. The predictor originally learned through the recovered-audio objectives.

### Longer-context change

The original RPM used dilation 1 in every residual block. Its direct receptive field was 37 Haar positions, corresponding to approximately **18.5 ms** of stego audio.

The context trial used block dilations:

```text
[1, 2, 4, 8, 16, 32, 64, 128]
```

Its direct receptive field was 1,025 Haar positions, approximately **512.5 ms**. This describes the RPM itself, not a complete receptive-field analysis of the entire encoder/decoder.

### NMSE change

The final loss trial kept the original short-context RPM. It replaced the waveform term for stego, secret, and recovered cover with per-example normalized mean squared error:

```text
NMSE = average over examples of:
       mean((estimate - reference)^2) / max(mean(reference^2), 0.000001)
```

In simple terms, it asks how large the waveform error is relative to the signal's own energy. This gives quiet and loud examples a more comparable role than an unnormalized absolute-error term.

The calculation runs in float32, including during mixed-precision training. The reference-power floor prevents very quiet clips from receiving unlimited weight. It still penalizes incorrect volume: this is not a scale-invariant loss.

The model architecture, loss weights, STFT windows, learning rate, effective batch, and seed were otherwise unchanged relative to the short-context control. Mean NMSE is related to SNR, but minimizing average NMSE is not mathematically identical to maximizing average SNR measured in dB.

## 7. Inventory: what actually ran

There are **eight persistent run directories** in this workspace:

| Run directory | Actual work completed | What it is not |
|---|---|---|
| `rpm_3060_librispeech_smoke` | 10 synthetic steps, original RPM | A real-data quality result |
| `rpm_3060_librispeech` | 1k pilot, then continuation to an 81k saved checkpoint | A completed 150k run |
| `rpm_3060_context_10k_smoke` | 10 synthetic steps, dilated RPM | A 10k real-data context experiment |
| `rpm_3060_context_10k_overfit` | 200 steps, then continuation to 1k on 16 fixed pairs | A full-data context benchmark |
| `rpm_3060_control_10k_overfit` | Same 200→1k diagnostic, short-context RPM | A fresh full-data 10k control |
| `rpm_3060_nmse_10k_smoke` | 10 synthetic steps, NMSE loss | A generalization result |
| `rpm_3060_nmse_10k_overfit` | Same 200→1k diagnostic, short-context RPM + NMSE | The initialization for the final model |
| `rpm_3060_nmse_10k` | Fresh real-data run completed to 10k; validation and test evaluated | A continuation of the overfit weights |

All directory names above are under `runs/`.

**A filename containing `10k` is not evidence that 10,000 steps were completed.** In particular, the full-data `rpm_3060_context_10k` and `rpm_3060_control_10k` runs were prepared as possible next steps but were not run. Only their `_overfit` diagnostics were completed.

The original 1k pilot and 81k model belong to one continuous training history. The later NMSE 10k model started from scratch and was not initialized from either an 81k checkpoint or the 16-pair diagnostic.

## 8. Chronological experiment record

### E0. Synthetic smoke checks

**Question:** Do the forward pass, backward pass, reverse pass, CUDA setup, validation, and checkpoint code work together?

Each implementation was checked with ten synthetic steps. The available saved results are:

| Variant | Synthetic validation loss | Cover SNR | Predicted secret SNR | Reported cycle error |
|---|---:|---:|---:|---:|
| Original short-context RPM | 3.1494 | 15.49 dB | -0.57 dB | 5.12e-6 |
| Dilated RPM | 3.0402 | 17.41 dB | -0.86 dB | 3.67e-6 |
| Short-context RPM + NMSE | 4.1955 | 18.01 dB | -0.59 dB | 4.06e-6 |

These checks completed and saved checkpoints. They established that the implementations could run, not that any variant was better at hiding real speech. Their loss values should not be compared as an architecture ranking, especially across different objectives.

### E1. Original RPM pilot: 1,000 real-data steps

**Question:** Can the target machine run the actual model at a usable speed, and does it learn on real audio?

- Config: `configs/rpm_3060_librispeech.yaml`.
- Training data at this point: `train-clean-100`, approximately 100.59 hours.
- Original short-context RPM and Charbonnier waveform loss.
- Started from scratch and ran to step 1,000.
- Progress-bar elapsed time: **8 minutes 11 seconds**.
- User-reported whole-command time: approximately **8 minutes 16 seconds**.
- Average progress rate at the end: approximately **2.04 steps/second**.

The training loss fell from approximately **13.0223 to 2.7281**.

| Evaluation scope | Cover SNR | Secret SNR | Secret SI-SDR | Oracle secret SNR |
|---|---:|---:|---:|---:|
| Training-time validation: first 20 batches | 14.07 dB | 9.11 dB | Not printed | 88.96 dB |
| All 2,703 validation records, FP32 crops | 11.50 dB | 6.06 dB | 4.33 dB | 89.78 dB |

The full-validation predicted PCM16 secret SNR was also approximately **6.06 dB**. The training-time reported cycle error was **4.792e-5**.

**Decision:** Hardware feasibility was confirmed. The experiment was continued with a larger training corpus. The early “20–60 minute pilot” estimate was replaced by the measured eight-minute result.

**Evidence:** `runs/rpm_3060_librispeech/profile.log`, its TensorBoard events, and `evaluation/step_00001000_val/metrics.json`.

The old 1k checkpoint itself is no longer retained because the normal run keeps only its latest three checkpoints. Its evaluation JSON and logs remain.

### E2. Original RPM continued on the expanded corpus

**Question:** Will substantially more training improve both cover preservation and secret recovery?

- Added `train-clean-360`, bringing available training audio to 464.196 hours.
- Resumed the 1k model; the step counter was cumulative, not restarted at zero.
- The configured step target was 150,000. The run was manually stopped after an 81,000-step checkpoint instead of completing that target.
- The last log progress is around step 81,007; the retained evaluated checkpoint is **step 81,000**.
- Continuation progress-bar elapsed time: approximately **10 hours 43 minutes**.

Selected small-validation measurements from the saved TensorBoard history:

| Step | Validation loss | Cover SNR | Secret SNR |
|---:|---:|---:|---:|
| 1,000 | 2.3543 | 14.07 dB | 9.11 dB |
| 10,000 | 1.8115 | 15.00 dB | 9.76 dB |
| 20,000 | 1.8119 | 15.24 dB | 9.64 dB |
| 40,000 | 1.7265 | 15.26 dB | 9.43 dB |
| 60,000 | 1.6540 | 15.50 dB | 9.69 dB |
| 78,000 | 1.6227 | 15.67 dB | 9.89 dB |
| 81,000 | 1.6578 | 15.62 dB | 9.66 dB |

The main observation was diminishing secret-SNR improvement, not a steady rise toward the oracle score. The cycle metric remained small rather than showing a catastrophic numerical failure. That alone did not establish good blind recovery.

At the saved 81k checkpoint:

| Evaluation scope | Cover SNR | Secret SNR | Secret SI-SDR |
|---|---:|---:|---:|
| Training-time validation: first 20 batches | 15.62 dB | 9.66 dB | Not printed |
| All validation records, FP32 crops | **10.74 dB** | **7.78 dB** | **6.62 dB** |
| All test records, FP32 crops; evaluated later | **10.19 dB** | **6.99 dB** | **5.39 dB** |

On full validation, continuing from 1k to 81k improved secret SNR by approximately 1.72 dB, but cover SNR decreased by approximately 0.76 dB. The small validation subset did not tell the whole story. This is a measured trade-off, not sufficient evidence by itself to diagnose overfitting.

**Decision:** Preserve this checkpoint as the original-loss reference and investigate targeted changes instead of automatically spending the remaining hours on the same setup.

**Evidence:** `training.log`, TensorBoard events, and the `step_00081000_val` and `step_00081000_test` evaluation folders under `runs/rpm_3060_librispeech/`.

### E3. Longer-context RPM on 16 fixed pairs

**Question:** Was the original predictor's short temporal context limiting recovery?

- Config: `configs/rpm_3060_context_10k.yaml`.
- Changed only predictor dilation geometry relative to the prepared short-context control, apart from run names/paths.
- Increased direct RPM context from 18.5 ms to 512.5 ms without increasing parameter count.
- Started fresh on 16 cached training pairs.
- Ran 200 steps, then resumed that diagnostic to 1,000 total steps.
- These cached pairs were not randomly recropped every epoch.

| Stage | Training loss shown in log | Small-validation cover SNR | Small-validation secret SNR | Stage elapsed time |
|---|---:|---:|---:|---:|
| Fresh 0→200 | 3.5807 | 17.39 dB | -0.15 dB | 1:53 |
| Resume 200→1,000 | 2.9290 | 18.83 dB | 0.30 dB | 6:50 |

Separate FP32 inference checks reconstructed the seeded cache and scored all 16 pairs:

| Checkpoint | Cached-pair cover SNR | Cached-pair secret SNR | Cached-pair secret SI-SDR |
|---|---:|---:|---:|
| 200 | 13.02 dB | -0.24 dB | -18.03 dB |
| 1,000 | 15.37 dB | 1.05 dB | -6.34 dB |

These cache-wide values came from offline inference checks, not from the original `validation` log lines. They are recorded here and in the trial notes; the early version of the trainer did not yet print `fixed_train` summaries.

**Interpretation:** Training loss decreased, but secret waveform recovery remained weak. The 200-step result was not treated as enough evidence to reject the idea; it was extended and then compared with a matched short-context diagnostic.

**Evidence:** `runs/rpm_3060_context_10k_overfit/overfit.log`, `overfit_1000.log`, TensorBoard, retained 200/1k checkpoints, and the offline diagnostic record in [the context/NMSE trial notes](rpm_nmse_trial.md).

### E4. Matched short-context control on the same 16 pairs

**Question:** Did the larger context improve anything compared with the existing predictor under a similar short training schedule?

- Config: `configs/rpm_3060_control_10k.yaml`.
- Original all-one dilations, 18.5 ms direct context.
- Same seed, model size, losses, effective batch, and selected cached audio.
- Same fresh 200 steps followed by a resume to 1,000 steps.

| Stage | Training loss shown in log | Small-validation cover SNR | Small-validation secret SNR | Stage elapsed time |
|---|---:|---:|---:|---:|
| Fresh 0→200 | 3.4277 | 20.58 dB | -0.19 dB | 1:47 |
| Resume 200→1,000 | 2.8432 | 19.36 dB | 0.51 dB | 6:43 |

Offline FP32 scoring over the whole cache at step 1,000 gave:

- Cover SNR: **16.11 dB**.
- Secret SNR: **1.22 dB**.
- Secret SI-SDR: **-5.69 dB**.

The reconstructed cached tensors for context and control were checked to be byte-identical. Their SHA256 was:

```text
7f1af6fbc0d421f19a4c881c762dc1debbb9d2bb8f2c54f145e9899c204550cc
```

The short-context control was slightly better on both cover and secret SNR in this diagnostic. The differences were small and came from one seed, so they do not prove that dilation is generally worse. They did mean that this particular dilation schedule had not earned a longer training run.

**Decision:** Keep the short-context architecture for the next experiment. Investigate the objective rather than adding context or training hours.

**Evidence:** `runs/rpm_3060_control_10k_overfit/overfit_200.log`, `overfit_1000.log`, TensorBoard, retained checkpoints, and the offline FP32 comparison.

### E5. Loss-gradient inspection: no training updates

**Question:** Was reducing the existing loss necessarily improving the waveform accuracy measured by SNR?

An offline gradient check was performed at the short-context 1k diagnostic checkpoint. Gradients were computed in float32 and averaged across the 16 cached pairs. No optimizer updates were applied.

| Loss component | Value at this check | Parameter-gradient norm |
|---|---:|---:|
| Secret Charbonnier, weight 1 | 0.03697 | **0.3768** |
| Secret STFT, weight 1 | 2.20462 | **38.3760** |
| Cover Charbonnier, including weight 2 | 0.01232 | 3.6404 |
| Cover STFT, weight 1 | 0.56614 | 207.4047 |
| Candidate secret NMSE, weight 1 | 0.78819 | **19.8346** |
| Candidate cover NMSE, including weight 2 | 0.12133 | 42.1064 |

The secret STFT gradient norm was approximately **101.8 times** the secret Charbonnier gradient norm at this checkpoint. The waveform term was giving a much weaker parameter-level signal in this measurement.

Magnitude-STFT losses can improve spectral shape without guaranteeing accurate sample-by-sample timing and phase. SNR, however, directly penalizes waveform error. This supported testing a waveform objective with a more appropriate scale.

**Important limit:** Gradient norms at one checkpoint do not prove the root cause, and their relative sizes are not a complete description of optimization. Directions, parameterization, clipping, and interactions also matter. This was a diagnostic reason to run an ablation, not proof that changing the loss must succeed.

### E6. NMSE objective on 16 fixed pairs

**Question:** Does energy-normalized waveform supervision improve the actual reconstruction metrics without changing the architecture?

- Config: `configs/rpm_3060_nmse_10k.yaml`.
- Original short-context RPM.
- Replaced the three waveform objectives with normalized MSE; retained the existing weights and STFT terms.
- Started from scratch, not from the original or dilated checkpoints.
- Used the same 16-pair diagnostic schedule: fresh 200 steps, then resume to 1,000.

The trainer was also extended to print an FP32 `fixed_train` summary covering all cached pairs. It preserves model mode and CPU/CUDA random-number state, so the reporting pass does not intentionally change later training sampling.

| Stage | Training loss shown in log | Fixed-pair cover SNR | Fixed-pair secret SNR | Small-validation cover SNR | Small-validation secret SNR | Stage time |
|---|---:|---:|---:|---:|---:|---:|
| Fresh 0→200 | 3.3277 | **15.75 dB** | **6.15 dB** | 16.30 dB | 8.21 dB | 1:47 |
| Resume 200→1,000 | 2.7232 | **18.34 dB** | **7.44 dB** | 18.53 dB | 9.97 dB | 6:35 |

At 1,000 steps, the direct cached-pair comparison was:

| Variant | Cover SNR | Secret SNR |
|---|---:|---:|
| Short-context Charbonnier | 16.11 dB | 1.22 dB |
| Dilated Charbonnier | 15.37 dB | 1.05 dB |
| Short-context NMSE | **18.34 dB** | **7.44 dB** |

Relative to the short-context Charbonnier diagnostic, NMSE improved cover SNR by approximately **2.23 dB** and secret SNR by approximately **6.22 dB** on this small training cache.

This was the clearest positive screening result. It did not yet establish generalization because the model repeatedly saw only 16 fixed training pairs. The small validation scores were useful additional observations, not substitutes for a real-data run and full evaluation.

**Do not compare raw loss totals between Charbonnier and NMSE as if they had the same units.** Compare reconstruction metrics, clipping, and examples instead.

**Decision:** Run a fresh, bounded real-data NMSE experiment before considering another overnight job.

**Evidence:** `runs/rpm_3060_nmse_10k_overfit/overfit_200.log`, `overfit_1000.log`, `overfit_train/*` TensorBoard tags, and the retained checkpoints.

### E7. Fresh real-data NMSE run to 10,000 steps

**Question:** Does the improvement survive beyond the 16-pair diagnostic when trained on the expanded corpus?

- Config: `configs/rpm_3060_nmse_10k.yaml`.
- Fresh model and optimizer; no overfit checkpoint was used as initialization.
- Normal random-crop training on the 464.196-hour manifest.
- No `--overfit-pairs` flag.
- Original short-context RPM with NMSE waveform supervision.
- Initial target: 10,000 steps with a three-hour safety budget.

#### Interruption and recovery

The assistant-launched background process stopped around **step 7,470**. Its last complete checkpoint was **step 7,000**. Initially there was no traceback explaining the stop. A later shell notification confirmed that the job was cancelled because the **server restarted**.

This was not evidence of a model NaN failure. The originally chained evaluation command did not run at that time.

The run was resumed in the user's terminal from step 7,000 with a cumulative target of 10,000 and a fresh one-hour safety budget. Approximately 470 unsaved steps from the interrupted process had to be repeated.

| Invocation | Progress | Logged elapsed time |
|---|---|---:|
| Initial background invocation | 0→approximately 7,470; checkpoint at 7,000 | 1:00:37 |
| Terminal resume | Saved 7,000→10,000 | 24:57 |
| Combined displayed time | Includes repeated unsaved work | Approximately **1:25:34** |

These times come from progress bars. They are not a precise GPU-busy-time or energy measurement, and they exclude waiting between invocations.

#### Training-time validation trend

These are the first 20 validation batches, not the final full-split measurements:

| Step | Cover SNR | Secret SNR | Reported cycle error |
|---:|---:|---:|---:|
| 1,000 | 18.33 dB | 10.26 dB | 5.79e-5 |
| 5,000 | 18.91 dB | 10.58 dB | 8.25e-5 |
| 7,000 | 20.12 dB | 10.76 dB | 9.89e-5 |
| 8,000 | 20.45 dB | 10.68 dB | 1.23e-4 |
| 9,000 | 21.12 dB | 10.80 dB | 1.36e-4 |
| 10,000 | **21.16 dB** | **10.85 dB** | **1.40e-4** |

Final displayed training loss was **2.2211** and small-validation loss was **1.8564**. These numbers belong to the NMSE objective and are not directly comparable to original-loss totals.

The numerical cycle error was small, but it did rise during training. This should be monitored if training resumes; it should not be described as exact bit-for-bit recovery.

#### Full validation and final test evaluation

After the 10k checkpoint was verified, it was evaluated on all validation records. Then both the fixed 81k reference and fixed 10k NMSE checkpoint were evaluated on the held-out test split. The test set was not used to choose between the earlier context diagnostics.

The complete numerical comparison is in the next section.

## 9. Final validation and test results

### All validation records: 2,703 fixed-crop pairs

| Measurement | Original RPM, 81k | NMSE RPM, 10k | NMSE minus original |
|---|---:|---:|---:|
| Cover SNR | 10.7372 dB | **16.9571 dB** | **+6.2198 dB** |
| Cover SI-SDR | 10.1734 dB | **16.8278 dB** | **+6.6544 dB** |
| Secret SNR, predicted residual | 7.7846 dB | **8.6747 dB** | **+0.8901 dB** |
| Secret SI-SDR, predicted residual | 6.6206 dB | **7.7004 dB** | **+1.0799 dB** |
| Secret SNR, predicted residual after PCM16 | 7.7836 dB | **8.6745 dB** | **+0.8909 dB** |
| Secret SNR, true residual: oracle only | 85.4857 dB | 82.6610 dB | -2.8247 dB |
| Clipped stego samples | 0.000490% | 0.000106% | Lower in this validation measurement |

### All test records: 2,620 fixed-crop pairs

| Measurement | Original RPM, 81k | NMSE RPM, 10k | NMSE minus original |
|---|---:|---:|---:|
| Cover SNR | 10.1949 dB | **15.7563 dB** | **+5.5613 dB** |
| Cover SI-SDR | 9.5856 dB | **15.5699 dB** | **+5.9843 dB** |
| Secret SNR, predicted residual | 6.9903 dB | **7.8979 dB** | **+0.9076 dB** |
| Secret SI-SDR, predicted residual | 5.3947 dB | **6.5666 dB** | **+1.1719 dB** |
| Secret SNR, predicted residual after PCM16 | 6.9901 dB | **7.8976 dB** | **+0.9076 dB** |
| Secret SI-SDR after PCM16 | 5.3944 dB | **6.5662 dB** | **+1.1719 dB** |
| Secret SNR, true residual: oracle only | 85.1457 dB | 82.3750 dB | -2.7707 dB |
| Clipped stego samples | 0.000252% | 0.000295% | Slightly higher, but still very rare |

The test comparison shows a substantial cover-SNR improvement and a smaller, positive secret-recovery improvement. It does not show an improvement in every possible property: oracle numerical reconstruction was somewhat lower, and the tiny test clipping fraction was slightly higher.

### What PCM16 testing showed

For the NMSE test checkpoint, predicted secret SNR changed from **7.8979046 dB** to **7.8976229 dB** after simulated PCM16 quantization: a drop of approximately **0.0002818 dB**.

Therefore, 16-bit rounding was not a major source of the measured recovery error at this operating point. This was a simulated clamp-and-round quantization check in the evaluator, not a complete codec/compression robustness test or a verified bit-identical file save/reload pipeline.

The exported WAVs are separately useful for listening. They are clipped to the valid output range before writing, so the unclipped floating-point clipping fraction must still be reported honestly.

## 10. How to read the numbers correctly

### SNR

SNR compares original signal energy with reconstruction-error energy. Higher is better. The implementation calculates SNR for each example and averages the dB values.

- **Cover SNR:** compare original cover with stego. Higher means the carrier was changed less in waveform terms.
- **Secret SNR:** compare original secret with recovered secret. Higher means better waveform recovery.
- A negative secret SNR indicates very poor reconstruction; producing approximately the right spectral shape is not enough to score well.
- dB is logarithmic. Do not describe a 5.56 dB gain as “5.56 percent better.”

### SI-SDR

SI-SDR checks reconstructed waveform agreement while allowing an overall gain adjustment. It complements SNR but does not replace it: correct carrier volume still matters.

### True-residual or oracle SNR

The oracle score uses the actual residual produced during encoding. It is a numerical-inversion diagnostic. It is not the performance of the blind RPM decoder, and that residual was not hidden/transmitted as a separate recovered payload in these experiments.

Matching an 82–90 dB oracle score through stego-only prediction is not an ordinary “train longer” target. The oracle receives an extra audio-sized output. Exact recovery of two arbitrary full-rate inputs from one same-sized fixed-precision output cannot generally be assumed. Speech structure can support approximate learned recovery, but does not remove the need to measure the information/quality trade-off.

### Gaussian scores

The evaluator also substitutes Gaussian noise for the missing residual. These RPM models were trained to use a predicted residual, not Gaussian noise. Poor Gaussian scores are not a fair comparison with a separately trained Gaussian-latent model.

For example, the final test Gaussian secret SNR was -8.15 dB for the original RPM checkpoint and -15.62 dB for NMSE. Those numbers must not be presented as a Gaussian-baseline experiment that RPM defeated. No separate Gaussian training run was completed here.

### Different meanings of “training” and “validation” outputs

| Output | What it measures |
|---|---|
| `train/total` | Loss averaged over the accumulated micro-batches for that step. |
| Most `train/*` component/SNR logs | The last micro-batch, not the entire training set or all 16 cached pairs. |
| `fixed_train` / `overfit_train/*` | FP32 score across the full cached fixed-pair set, added during the NMSE work. |
| Console `validation` | The configured first 20 validation batches, normally 80 pairs. |
| Full-validation `metrics.json` | All 2,703 validation records under the deterministic crop protocol. |
| Final test `metrics.json` | All 2,620 held-out test records under that same protocol. |

This is why the last NMSE console values, **21.16 dB cover / 10.85 dB secret**, must not be used as the final test headline. The actual test values are **15.76 / 7.90 dB**.

Two additional reporting details:

- The evaluation JSON's `stego_peak` is an average of batch maxima, not the largest sample across the entire split. A value below 1 there does not prove zero clipping.
- The validation `exact_cycle_max` summary averages batchwise maximum errors. It is not a globally certified worst-case error over every record.

## 11. Conclusions supported by the experiments

### Supported

1. **The implemented Stage-1 model is feasible on the available machine.** Measured throughput was approximately two training steps per second at the chosen batch and crop sizes.
2. **More of the same original-loss training had diminishing secret-SNR returns.** This was visible in the recorded small-validation trend and motivated stopping at the 81k checkpoint.
3. **The tested dilation schedule did not improve the 1k fixed-pair screen.** It was not pursued as a full-data run.
4. **The waveform objective mattered.** NMSE clearly improved both SNRs in the small, matched diagnostic, then produced a better real-data checkpoint on full validation and test crops.
5. **The largest final gain was cover preservation.** Test cover SNR improved by 5.56 dB; secret SNR improved by a more modest 0.91 dB.
6. **Simulated PCM16 rounding was not the main bottleneck for these checkpoints.** The secret-SNR drop was very small.

### Not supported

- That the model is transparent to listeners or that recovered speech is consistently intelligible.
- That NMSE always wins across seeds, corpora, pairing strategies, and training budgets.
- That the final model is lossless, codec-robust, secure, encrypted, or state of the art.
- That longer context can never help; only one schedule and one small screening setup were tested.
- That ten thousand steps are intrinsically sufficient, or that a guaranteed improvement would follow from another twenty hours.
- That the NMSE run was “8.1 times more training-efficient.” The step counts differ by that ratio, but the experiment does not isolate a rigorous efficiency claim.

## 12. Limitations and unresolved issues

1. **Unequal main training budgets.** The reference used 81k cumulative steps; the NMSE model used 10k. A fresh original-loss 10k full-data control was not run.
2. **Different early data histories.** The reference used the smaller corpus for its initial 1k steps; NMSE used the expanded manifest from the start.
3. **One seed.** There are no repeated-seed means, confidence intervals, or significance tests.
4. **Fixed crops and neighbor pairing.** The final evaluation is not a broad random-pair, independently cross-speaker benchmark. Speaker-disjoint dataset splits do not mean each evaluated cover/secret pair uses different speakers.
5. **No comprehensive perceptual evaluation.** WAVs were exported, but this record contains no formal listening-study scores, ASR word-error results, STOI, PESQ, or ViSQOL results.
6. **No attack/security evaluation.** No evidence here establishes robustness to compression, resampling, noise, cropping, or steganalysis.
7. **Residual gap remains large.** The final test predicted secret SNR is 7.90 dB, far below its oracle SNR. The reconstruction is not close to numerical inversion from stego alone.
8. **Clipping is not exactly zero.** It is very rare in the evaluated crops, but cannot be called absent.
9. **AMP skipped updates occurred.** Initial gradient norms were non-finite in earlier logs, and saved scaler reductions indicate overflow handling. Later sampled logs were finite, but the logs do not give a complete count of every skipped update. Finite loss alone does not rule out gradient overflows.
10. **Resume is not a bitwise-continuous replay.** The checkpoint contains model, optimizer, scaler, config, and step. It does not restore complete Python/NumPy/Torch RNG and sampler positions. Recreated fixed pairs were verified to match, but general resumed batch order is not guaranteed to be identical to an uninterrupted run.
11. **Source version was not frozen as an experiment commit.** The working tree contains implementation changes. Reproduction requires preserving that source version alongside the weights and configs, not only writing down a repository name.
12. **Test results are now known.** Further tuning should use validation, not repeatedly optimize decisions against these same reported test scores.

These limitations should be stated plainly. They do not erase the observed checkpoint improvements, but they limit how broadly those improvements can be claimed.

## 13. Evidence and artifacts to keep

All paths below are relative to `spectra_inn_windows`.

### Main checkpoints

| Model | Path | Size |
|---|---|---:|
| Original RPM reference | `runs/rpm_3060_librispeech/checkpoints/step_00081000.pt` | 11,001,419 bytes |
| Final NMSE model | `runs/rpm_3060_nmse_10k/checkpoints/step_00010000.pt` | 11,001,547 bytes |

SHA256 values:

```text
Original 81k:
f9bf0b68f8f764ab79fc97feb48107c54b36fe8cef71708cbb11aef774c0a17f

NMSE 10k:
f74979b8447b6fc25abcda2cf0ea0dc710b08383bec4b56065b360fd11a14972
```

The original run currently retains steps 79k, 80k, and 81k. The NMSE real-data run retains 8k, 9k, and 10k. Each fixed-pair diagnostic retains its 200- and 1,000-step checkpoints. Smoke folders retain their 10-step checkpoint.

### Main numerical reports

| Purpose | File |
|---|---|
| Original pilot, full validation | `runs/rpm_3060_librispeech/evaluation/step_00001000_val/metrics.json` |
| Original 81k, full validation | `runs/rpm_3060_librispeech/evaluation/step_00081000_val/metrics.json` |
| Original 81k, final test | `runs/rpm_3060_librispeech/evaluation/step_00081000_test/metrics.json` |
| NMSE 10k, full validation | `runs/rpm_3060_nmse_10k/evaluation/step_00010000_val/metrics.json` |
| NMSE 10k, final test | `runs/rpm_3060_nmse_10k/evaluation/step_00010000_test/metrics.json` |

### Logs and configuration

- Original pilot: `runs/rpm_3060_librispeech/profile.log`.
- Original continuation: `runs/rpm_3060_librispeech/training.log`.
- Context diagnostic: `runs/rpm_3060_context_10k_overfit/overfit.log` and `overfit_1000.log`.
- Short-context diagnostic: `runs/rpm_3060_control_10k_overfit/overfit_200.log` and `overfit_1000.log`.
- NMSE diagnostic: `runs/rpm_3060_nmse_10k_overfit/overfit_200.log` and `overfit_1000.log`.
- NMSE full-data run: `runs/rpm_3060_nmse_10k/training.log` and `training_resume.log`.
- Every run's `run_metadata.json`, `config.resolved.yaml`, and `tensorboard/` directory.
- Configurations under `configs/`, including the four `rpm_3060_*` configurations.

The last `config.resolved.yaml` in a resumed directory describes the latest invocation's flags. For example, the final NMSE invocation shows a one-hour safety budget, while the initial config allowed three hours. Use the checkpoint plus the logs to reconstruct the full history rather than assuming the last YAML describes every earlier invocation.

### Audio examples

The main test exports are in:

```text
runs/rpm_3060_librispeech/evaluation/step_00081000_test/
runs/rpm_3060_nmse_10k/evaluation/step_00010000_test/
```

Each contains ten example directories, starting at `example_000/`, with files including:

```text
cover.wav
secret.wav
stego.wav
recovered_secret_predicted.wav
recovered_secret_predicted_pcm16.wav
recovered_secret_true_residual.wav
recovered_secret_gaussian.wav
```

For a practical demo, compare cover with stego and secret with **predicted** recovery. Do not play the true-residual recovery as if it were the recipient's result.

Listen to all examples before selecting presentation samples. Show representative behavior and, if useful, a weaker case rather than presenting a best-case clip as average performance. No listening quality is asserted by this document itself.

Run outputs and most data artifacts are ignored by Git. Keeping this Markdown file does not automatically back up its checkpoints, audio, or JSON evidence. Archive those separately before moving machines or cleaning the workspace.

## 14. What was proposed but not run

The following ideas appeared in planning but have no completed experimental result in this record:

- A fresh full-data 10k short-context Charbonnier control.
- A full-data 10k dilated-context model.
- A 150k completed original model or a long NMSE overnight run.
- Lower-learning-rate fine-tuning or a learning-rate scheduler.
- Direct supervision of the predicted residual against the true residual.
- Removing the recovered-cover loss as an ablation.
- A combined dilated-RPM plus NMSE real-data experiment.
- Compressed-secret payloads, extra side-information channels, and codec-robust training.

These must not appear in a presentation as completed experiments merely because a config or discussion exists.

## 15. Sensible next work, when training resumes

The immediate decision was to pause and document rather than automatically start another overnight run.

For a stronger scientific comparison, the first useful next experiment is the prepared **fresh full-data 10k Charbonnier control**, using the same expanded corpus and settings as the NMSE run. That would reduce the largest training-budget/data-history confound.

After that, useful priorities are:

1. Repeat the controlled comparison across more seeds.
2. Evaluate multiple crops and explicitly different-speaker cover/secret pairs.
3. Add a reproducible listening and speech-intelligibility evaluation.
4. Investigate clipping, numerical conditioning, and loss balance before scaling training time.
5. Treat compression robustness and the larger architecture as separate, later milestones.

The present result is worth presenting as an engineering prototype with a promising, measured objective change. It is not a reason to promise that more training will reach the oracle score.

## 16. Final plain-language conclusion

We got audio-in-audio training working on the available consumer hardware. We did not simply run one model and report the nicest console number. We checked the small-validation bias, tried a context change, ran a matched small control, inspected the objective, and tested a normalized waveform loss. The final NMSE checkpoint improved cover preservation substantially and secret recovery modestly on held-out test crops. The work demonstrates progress and a useful lesson about training objectives, while leaving a clear gap to high-quality, secure, robust, stego-only recovery.
