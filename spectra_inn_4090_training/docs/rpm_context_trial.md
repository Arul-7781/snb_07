# Next experiment: longer-context RPM, without changing the stopped baseline

The original run stopped at step 81,000. Its full-validation crop evaluation is saved at:

```text
runs/rpm_3060_librispeech/evaluation/step_00081000_val/
```

Across 2,703 validation records, cover SNR was **10.737 dB**, predicted secret SNR **7.785 dB**,
secret SI-SDR **6.621 dB**, and predicted PCM16 secret SNR **7.784 dB**. This is a broader
diagnostic than the training loop's first 20 batches, which looked considerably better.
These are fixed two-second crops paired with neighboring records, not exhaustive audio or
independent cross-speaker pairings. The test split remains untouched.

## What changes, and what does not

`configs/rpm_3060_context_10k.yaml` uses predictor block dilations
`[1, 2, 4, 8, 16, 32, 64, 128]`. The predictor's direct receptive field grows from 37 to
1,025 Haar positions: approximately **18.5 ms to 512.5 ms** at 16 kHz / three Haar levels.
Both have **878,344 parameters**. Larger context is a hypothesis to test, not a promised SNR gain.

The INN, width, losses, learning rate, effective batch, and dataset stay the same. Old configs
default to all-one dilations and old checkpoints still load with their original behavior.
No learning-rate scheduling, residual supervision, or SNR loss has been added in this trial.

New outputs are isolated:

| Run | Directory |
|---|---|
| Synthetic integration test | `runs/rpm_3060_context_10k_smoke/` |
| Fixed-pair diagnostic | `runs/rpm_3060_context_10k_overfit/` |
| Longer-context trial | `runs/rpm_3060_context_10k/` |
| Matched short-context control | `runs/rpm_3060_control_10k/` |

**Start the new trial from scratch. Do not pass `--resume` from the 81k run or the overfit run.**
The changed dilation geometry makes old weights a different function even though tensor
shapes match. Reusing them would be a warm-start experiment, not this controlled comparison.
Each real-data diagnostic/trial should be run once per output directory; ask before overwriting an
existing run. No additional dataset download or large cache is needed. Disk currently has
about 13 GiB free; retain the 10 GiB safety margin.

## 1. Check the implementation

From `spectra_inn_windows`:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m spectra_inn.train \
  --config configs/rpm_3060_context_10k.yaml --device cuda --smoke --max-steps 10
```

## 2. Fixed-pair overfit diagnostic first

```bash
mkdir -p runs/rpm_3060_context_10k_overfit
set -o pipefail
.venv/bin/python -u -m spectra_inn.train \
  --config configs/rpm_3060_context_10k.yaml --device cuda \
  --overfit-pairs 16 --max-steps 200 --max-hours 0.25 \
  2>&1 | tee runs/rpm_3060_context_10k_overfit/overfit.log
```

This picks 16 seeded training pairs, decodes/crops/augments them **once**, and cycles the
same cached tensors. It is not a manifest containing 16 files that gets randomly recropped
every step. Validation still uses separate held-out audio. The trainer adds `_overfit` to
the output directory and records `data.overfit_pairs` in its resolved config.

Allow a few minutes; the 15-minute training budget is a guard, not a runtime prediction.
The initial encoder is nearly identity, so cover SNR starts very high and can fall as hiding
is learned. Look for falling **training loss** and improving **training secret SNR**, finite
gradients, and small float32 cycle errors. Good held-out validation is not expected from
16 memorized pairs. This is only a plumbing/optimization check; it cannot prove capacity
or generalization. Failure to fit in 200 steps needs investigation, not an automatic verdict.

Report the final output before committing to the short trial if losses are non-finite,
cycle errors grow rapidly, or training shows no useful improvement. Do not use these
overfit-trained weights as initialization for the next stage.

## 3. Run the 10,000-step longer-context trial

```bash
mkdir -p runs/rpm_3060_context_10k
set -o pipefail
time .venv/bin/python -u -m spectra_inn.train \
  --config configs/rpm_3060_context_10k.yaml --device cuda \
  2>&1 | tee runs/rpm_3060_context_10k/training.log
```

The config stops at 10,000 steps or three training hours, saving every 1,000 steps. The
old short-context rate suggests around 80 minutes for 10k, but dilation may change speed;
use the new run's ETA rather than assuming identical throughput. Keep the terminal alive.
If interrupted, Ctrl+C does not save a fresh checkpoint: use only a completed `.pt` and
resume only with this same experiment/config. Preserve logs with a new filename on resume.

## 4. Evaluate on full validation, not the test set

After reaching 10,000 steps:

```bash
.venv/bin/python -m spectra_inn.evaluate \
  --checkpoint runs/rpm_3060_context_10k/checkpoints/step_00010000.pt \
  --split val --device cuda --batch-size 4 --max-batches 1000 --examples 10
```

The 1,000-batch allowance covers all 2,703 validation records at batch size 4. Inspect
`metrics.json` and listen to the WAVs in `evaluation/step_00010000_val/`. Compare cover
distortion, blind predicted recovery, PCM16 degradation, and clipping together. An oracle
or this model's Gaussian decode is not a separately trained baseline.

## 5. If promising, run the matched fresh control

Historical results from the 81k model can show whether the new trial looks practically
useful, but cannot establish a fair architecture comparison: training duration differs,
and the original run used the 100-hour corpus for its first 1,000 steps.

For a controlled comparison, `configs/rpm_3060_control_10k.yaml` uses the current expanded
dataset from step 1, the same seed, settings, and 10k budget, but all-one predictor dilations:

```bash
mkdir -p runs/rpm_3060_control_10k
time .venv/bin/python -u -m spectra_inn.train \
  --config configs/rpm_3060_control_10k.yaml --device cuda \
  2>&1 | tee runs/rpm_3060_control_10k/training.log
.venv/bin/python -m spectra_inn.evaluate \
  --checkpoint runs/rpm_3060_control_10k/checkpoints/step_00010000.pt \
  --split val --device cuda --batch-size 4 --max-batches 1000 --examples 10
```

Run sequentially on the one GPU. Compare equal completed step counts; if either hits its
time limit early, do not compare unequal budgets as evidence that dilation helped.
An equal-step comparison tests learning at the same number of updates; record elapsed time
too, because the two models need not have equal compute cost. A single seed is exploratory.

As a practical screening rule, look for a meaningful secret-SNR/SI-SDR improvement without
a substantial cover-SNR loss, increased clipping, or worse PCM16 recovery. For example,
0.5 dB more secret SNR with no more than 0.5 dB cover loss is a possible initial threshold,
not statistical proof or a claim of imperceptibility. Listen before accepting it.
If the new trial clearly underperforms, stop rather than automatically starting another
overnight run. If promising, review both results before extending training or changing losses.
