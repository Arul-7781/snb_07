# Mentor briefing: what to say and present

**Status as of 15 October 2026:** Experiments are paused. We have a working Stage-1 speech-in-speech prototype, saved checkpoints, held-out test results, and audio examples.

This is the short meeting version. The detailed evidence and chronology are in [experiment history and results](experiment_history_and_results.md).

**If the meeting is soon:** use the pitch in Section 1 and the table in Section 2. The rest is preparation material for slides, the demo, and questions.

## 1. A short pitch you can say directly

> We have a working audio-in-audio prototype on our RTX 3060. It hides two-second speech segments inside other speech and uses a learned predictor to recover the missing information at decoding. We trained an initial reference to 81,000 steps, then tested longer predictor context and a revised waveform loss. Longer context did not help in our small screening experiment. The energy-normalized loss did, so we trained a fresh model for 10,000 steps. On 2,620 held-out test crops, cover SNR improved from 10.19 to 15.76 dB, and secret SNR from 6.99 to 7.90 dB. These are promising prototype results, not lossless recovery. The main comparison still needs an equal-budget control.

**One-line takeaway:**

> The prototype works, and changing the waveform objective produced a better tested checkpoint; adding context alone did not show a benefit in our small diagnostic.

## 2. The table to put on the main results slide

**LibriSpeech `test-clean`: 2,620 fixed two-second pairs. Higher is better.**

| Metric | Original RPM reference, 81k steps | NMSE RPM, 10k steps | Gain |
|---|---:|---:|---:|
| Cover SNR | 10.19 dB | **15.76 dB** | **+5.56 dB** |
| Secret SNR | 6.99 dB | **7.90 dB** | **+0.91 dB** |
| Secret SI-SDR | 5.39 dB | **6.57 dB** | **+1.17 dB** |

Put this footnote on the slide:

> One seed; fixed-crop/neighbor-pair evaluation; unequal training budgets. This is a checkpoint comparison, not yet a matched-budget claim.

Useful explanation:

- **Cover SNR:** how little we changed the carrier audio.
- **Secret SNR:** how accurately we recovered the hidden waveform.
- **Secret SI-SDR:** a complementary recovery score that allows an overall volume adjustment.
- The largest gain was cover preservation; the secret-recovery gain was smaller but positive.

Use these **test** numbers, not the 21.16/10.85 dB console numbers. Those console values came from only the first 80 validation pairs.

## 3. A simple four-slide presentation

### Slide 1 — What we built

Show this diagram:

```text
Cover + secret -> invertible encoder -> stego + residual

Stego -> learned residual predictor -> predicted residual
Stego + predicted residual -> reverse network -> recovered secret
```

Say: “This is the clean 16 kHz mono speech baseline, not the entire proposed robust architecture.”

Keep the hardware note small: Ryzen 5 3600, RTX 3060 12 GB, about 16 GB RAM, approximately 878k model parameters.

### Slide 2 — What we tried

| Experiment | Honest outcome |
|---|---|
| Original RPM, 81k saved steps | Working reference; secret improvement became slow. |
| Longer-context RPM, 16 fixed pairs | No clear advantage over the matched short-context diagnostic at 1k steps. |
| Normalized-MSE loss, 16 fixed pairs | Stronger reconstruction scores in the small diagnostic. |
| Fresh NMSE model, full-data training to 10k | Better final validation and test checkpoint scores. |

Say: “We used the small diagnostics to decide what was worth training. We did not run the longer-context model as a full 10k data experiment.”

### Slide 3 — Final test results

Use the table above. Add two short points:

- Simulated 16-bit rounding changed the NMSE secret SNR by only about **0.0003 dB** in this test.
- Recovery remains imperfect; the result does not establish inaudibility, security, or compression robustness.

### Slide 4 — Audio demo, limits, and next step

Play the same example's:

1. Original cover, then stego.
2. Original secret, then **predicted-residual** recovered secret.

Say: “The next scientific check is an equal-budget full-data control and broader pairing/listening evaluation, not simply more overnight training.”

## 4. Where to get the demo and numerical evidence

All paths are relative to `spectra_inn_windows`.

**NMSE test results and ten exported examples:**

```text
runs/rpm_3060_nmse_10k/evaluation/step_00010000_test/
```

Inside `example_000/` and the other example folders, use:

- `cover.wav` and `stego.wav` for carrier distortion.
- `secret.wav` and `recovered_secret_predicted.wav` for actual blind recovery.
- `recovered_secret_predicted_pcm16.wav` if discussing the quantization check.

**Original reference examples and metrics:**

```text
runs/rpm_3060_librispeech/evaluation/step_00081000_test/
```

Each evaluation directory has `metrics.json`. Those files are the source of the results slide.

Listen before the meeting. Choose representative examples and be ready to show a weaker one too. No formal human listening score has been measured yet. Do not play `recovered_secret_true_residual.wav` and call it ordinary stego-only recovery: it uses information the recipient would not have.

## 5. Likely mentor questions: short, honest answers

**“Did you get results, or only run setup?”**

“We completed an 81k original-loss reference, three small fixed-pair diagnostics, and a fresh 10k NMSE real-data model. Both main checkpoints have final held-out test measurements and exported audio.”

**“Why did you change the loss?”**

“The spectral loss could improve while waveform recovery stayed weak. At one diagnostic checkpoint, its secret gradient norm was about 100 times the original waveform term's. That motivated a normalized waveform-error ablation. The measurement suggested a test; it did not prove the cause by itself.”

**“Did the larger predictor help?”**

“Not in the one-seed, 16-pair, 1k-step screen we ran. We did not spend a full training budget on it, and we are not claiming that all longer-context models are worse.”

**“Why does the log also show an 80-plus dB recovery score?”**

“That is the oracle decoder using the true residual. Ordinary decoding predicts that missing residual. The actual final blind test secret SNR is 7.90 dB, not the oracle score.”

**“Is 10k versus 81k a fair ablation?”**

“Not a fully controlled main comparison. The final NMSE checkpoint scores better, but the budgets and early data histories differ. We also have smaller matched diagnostics supporting the objective change. A fresh 10k original-loss full-data control is still needed.”

**“How much data did you use?”**

“The expanded training manifest contains 464.2 hours of clean speech, with separate validation and test speakers. Training samples two-second crops. The final test covers all 2,620 test records under one fixed-crop protocol, not every second or every possible pairing.”

**“Can the recovered secret be understood, and is the carrier unchanged?”**

“We have examples and waveform metrics, but no formal intelligibility or listening-study result. Reconstruction errors remain, so we should demonstrate the audio rather than promise perceptual transparency.”

**“Is it secure or robust to MP3?”**

“Those properties have not been tested. This milestone is the clean audio baseline; steganalysis, encryption, and codec robustness are separate work.”

**“What happened to the interrupted training?”**

“The job server restarted. We resumed the 10k experiment from its 7k checkpoint in a terminal and completed it. It was not an abandoned experiment or evidence of a model crash.”

## 6. What to ask the mentor

> We have a reproducible prototype checkpoint and a promising objective change. Would you prefer the next milestone to focus on a rigorous equal-budget comparison and listening evaluation, or on extending the system toward codec robustness? Our recommendation is to validate the current clean baseline more rigorously first.

If a specific next experiment is requested, propose the **prepared fresh 10k short-context Charbonnier control** on the same expanded dataset, followed by additional seeds and different-speaker pairings. Do not imply that this control has already been run at full-data scale.

## 7. Avoid these claims

- “We achieved 82–90 dB blind recovery.” Those are oracle scores.
- “The final result is 21.16 dB cover and 10.85 dB secret.” Those are small-subset console scores, not final test scores.
- “We improved test secret SNR by 6.22 dB.” That larger gain belongs to the 16-pair training diagnostic; the final test gain is **0.91 dB**.
- “We trained the longer-context model for 10k full-data steps.” Only its 1k fixed-pair diagnostic was completed.
- “Our model is lossless, inaudible, encrypted, secure, MP3-robust, or state of the art.” The experiments do not establish those claims.
- “Ten thousand steps are proven to be eight times more efficient.” Unequal step counts alone do not prove training efficiency.

## 8. Meeting checklist

- [ ] Bring the final **test** result table with its limitations footnote.
- [ ] Keep both `metrics.json` files available if the numbers are questioned.
- [ ] Listen to the selected examples on the actual presentation device.
- [ ] Prepare a cover/stego pair and a secret/predicted-recovery pair.
- [ ] Explain the oracle-versus-blind distinction in one sentence.
- [ ] State that this is a Stage-1 prototype and a promising checkpoint comparison.
- [ ] End with one concrete request for the next milestone, not a promise of near-lossless recovery.
