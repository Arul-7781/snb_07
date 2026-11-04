# Loss-only RPM diagnostic: normalized waveform MSE

## Why test this next?

After the same 200 + 800-step fixed-pair schedule, FP32 scores across the entire 16-pair
cache were:

| Predictor | Cover SNR | Secret SNR | Secret SI-SDR |
|---|---:|---:|---:|
| Original short context | 16.11 dB | 1.22 dB | -5.69 dB |
| Dilated longer context | 15.37 dB | 1.05 dB | -6.34 dB |

The reconstructed pair tensors were byte-identical (SHA256
`7f1af6fbc0d421f19a4c881c762dc1debbb9d2bb8f2c54f145e9899c204550cc`).
This one-seed diagnostic does not show a benefit for the tested dilation schedule. Both
models have weak waveform recovery. It does not establish that longer context can never help.

An offline FP32 gradient check at the short-context 1k checkpoint, averaging across the
16 pairs, found parameter-gradient norms of about **0.377 for secret Charbonnier loss**
versus **38.38 for secret STFT loss** (both weights 1). The normalized-MSE candidate gave
about **19.83**. Cover STFT gradients were also much stronger than cover Charbonnier
gradients. No optimizer steps were taken in this analysis.

Magnitude-STFT improvements do not establish accurate waveform/phase recovery. These
measurements motivate an objective ablation, but gradient norms at one checkpoint do not
prove the cause or guarantee that another loss will improve SNR.

## The single change

`configs/rpm_3060_nmse_10k.yaml` keeps the **short-context** model and all existing loss
weights, STFT windows, learning rate, AMP, and data settings. Only the waveform objective
changes from Charbonnier to per-example normalized MSE for stego, secret, and recovered cover:

```text
NMSE = mean_over_examples(mean((estimate - reference)^2) / max(mean(reference^2), floor))
```

The reference-power floor is `1e-6` (RMS 0.001 in normalized waveform units). Computation
is float32 even with AMP, and gain errors are penalized. This is not SI-SDR. Mean NMSE is
related to SNR but is not exactly equivalent to maximizing mean dB SNR. The floor prevents
near-silent clips from getting unbounded weights.

All old configs still use Charbonnier by default. Their checkpoints, dataset, and outputs
remain unchanged. No new dataset or large cache is required.

## 1. Check code and synthetic integration

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m spectra_inn.train \
  --config configs/rpm_3060_nmse_10k.yaml --device cuda --smoke --max-steps 10
```

## 2. Run the same fixed-pair diagnostic, starting fresh

Do not resume old Charbonnier weights or reuse a previous experiment directory. The
commands below repeat the earlier **200 then 800 steps** pattern, so the optimizer, scaler,
and data-loader restart schedule are matched as well as the selected cached waveforms.

```bash
RUN=runs/rpm_3060_nmse_10k_overfit
CONFIG=configs/rpm_3060_nmse_10k.yaml
mkdir -p "$RUN"
set -o pipefail

.venv/bin/python -u -m spectra_inn.train \
  --config "$CONFIG" --device cuda \
  --overfit-pairs 16 --max-steps 200 --max-hours 0.25 \
  2>&1 | tee "$RUN/overfit_200.log" &&
.venv/bin/python -u -m spectra_inn.train \
  --config "$CONFIG" --device cuda \
  --overfit-pairs 16 \
  --resume "$RUN/checkpoints/step_00000200.pt" \
  --max-steps 1000 --max-hours 0.25 \
  2>&1 | tee "$RUN/overfit_1000.log"
```

Allow approximately 8–10 minutes total at the measured speed. Each invocation has a
15-minute training guard. Do not start a 10k or overnight run yet.

## 3. Judge the right measurements

The trainer now prints a separate line after overfit validation:

```text
fixed_train step=1000 pairs=16 total=... cover_snr=... secret_snr=...
```

These metrics cover **all cached training pairs in FP32**, not just the last four-pair
micro-batch. They are also recorded under `overfit_train/*` in TensorBoard. This diagnostic
preserves the model mode and CPU/CUDA RNG state; it does not alter the next training batch.
The usual `validation ...` line still refers to separate held-out audio, not the cache.

Compare final fixed-pair SNR against the short-context Charbonnier control (**16.11 dB
cover, 1.22 dB secret**). Look for a clear secret improvement without a large cover-quality
loss, then check SI-SDR, PCM16, and examples before claiming progress. Do not compare raw
total losses across modes: the units and scale have changed. A small gain on 16 memorized
pairs is a screening signal, not evidence of held-out improvement.

There were AMP overflow/skipped-update indications in the earlier runs: non-finite initial
gradient norms and reduced scaler values. The scaler handles these by skipping updates;
finite loss alone does not rule them out. Persistent non-finite gradients need investigation,
not a longer training run. AMP settings are intentionally unchanged for this loss-only test.

Return the `fixed_train` and validation lines at step 1,000 for review. If the objective
improves cached-pair recovery, a separate fresh real-data trial is the next test. If it does
not, inspect gradients/conditioning and a reconstruction-only diagnostic rather than
continuing repeated overnight runs. Near-oracle stego-only recovery is still not guaranteed.
