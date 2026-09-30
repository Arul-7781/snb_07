# Setup Guide — RTX 4090 + LibriSpeech

This scaffold is designed for Linux and an NVIDIA RTX 4090. The same code can run on an RTX 3060 12 GB with a smaller batch.

## 1. Verify the NVIDIA driver

Run:

```bash
nvidia-smi
```

You should see your RTX 4090 and a working NVIDIA driver.

You generally do **not** need to install the full CUDA toolkit separately just to use the PyTorch pip wheel. PyTorch ships the required CUDA runtime libraries with its CUDA-enabled packages; the NVIDIA driver must still be working.

## 2. Recommended Python

Use Python 3.10–3.12 if possible for the least packaging friction.

Check:

```bash
python3 --version
```

## 3. Create the virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install dependencies

The simplest route is:

```bash
pip install -e ".[dev]"
```

Then verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If `CUDA available` is `False`, install the current CUDA-enabled PyTorch build using the selector at:

https://pytorch.org/get-started/locally/

Then rerun the check above.

If `soundfile` reports a system library error on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y libsndfile1
```

## 5. Dataset location

The default config expects this exact structure:

```text
data/
  LibriSpeech/
    train-clean-100/
      ...speaker/chapter/*.flac
    dev-clean/
      ...speaker/chapter/*.flac
    test-clean/
      ...speaker/chapter/*.flac
```

The loader recursively discovers `.flac` and `.wav`, so the normal LibriSpeech speaker/chapter structure is fine.

### Option A — automatic download

From the project root:

```bash
./scripts/download_librispeech.sh
```

This downloads and extracts:

- `train-clean-100`
- `dev-clean`
- `test-clean`

under `data/LibriSpeech/`.

Each archive is removed after successful extraction to avoid retaining a
second multi-gigabyte copy of the dataset.

### Option B — manual download

Download the same LibriSpeech subsets from OpenSLR and extract the resulting `LibriSpeech` directory under `data/`.

## 6. Preflight check

Run:

```bash
python scripts/preflight.py --config configs/baseline_4090.yaml
```

It checks:

- PyTorch/CUDA visibility;
- GPU name and VRAM;
- file counts for train/validation/test;
- segment/model/STFT configuration consistency;
- configured minimum GPU memory;
- exact synthetic INN forward/reverse consistency.

Preflight exits nonzero when a required split has fewer than two audio files,
the selected hardware is too small for the config, or a consistency check
fails. Do not start a long training run until it succeeds for the exact config
you plan to train.

## 7. Run tests

```bash
pytest -q
```

The most important initial test verifies that:

```text
(cover, secret) -> INN -> (stego, r) -> INN^-1 -> (cover, secret)
```

is numerically invertible when the true `r` is retained.

## 8. Smoke training

Run:

```bash
python -m audio_inn.train --config configs/smoke.yaml --run-name smoke
```

This is intentionally tiny. Confirm it creates:

```text
runs/smoke/
  train.log
  train_metrics.csv
  val_metrics.csv
  environment.json
  config_resolved.yaml
  checkpoints/
    best.pt
    last.pt
    epoch_001.pt
  samples/
    epoch_001/
      cover.wav
      secret.wav
      stego.wav
      secret_random_z.wav
      cover_true_r_oracle.wav
      secret_true_r_oracle.wav
  tensorboard/
```

## 9. Start the RTX 4090 training run

```bash
python -m audio_inn.train \
  --config configs/baseline_4090.yaml \
  --run-name baseline_4090_v1
```

The default starts with:

```text
16 kHz
1-second chunks
8 invertible blocks
hidden width 64
batch size 16
bfloat16 AMP
15 epochs
12,000 cover-secret pairs per epoch
```

These are starting values, not sacred hyperparameters.

## 10. TensorBoard

In another terminal:

```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006
```

Then open:

```text
http://localhost:6006
```

Track at minimum:

- train/loss
- train/cover_snr
- train/secret_sisdr
- val/loss
- val/cover_snr
- val/secret_sisdr
- val/secret_sisdr_true_r
- val/cycle_max_abs
- val/stego_peak_abs
- val/stego_clip_fraction

## 11. Resume after interruption

Use `last.pt`:

```bash
python -m audio_inn.train \
  --config configs/baseline_4090.yaml \
  --run-name baseline_4090_v1 \
  --resume runs/baseline_4090_v1/checkpoints/last.pt
```

Checkpoint contents include:

- model weights;
- optimizer state;
- epoch;
- global step;
- best validation loss;
- resolved config.

## 12. Evaluate the best checkpoint

```bash
python -m audio_inn.evaluate \
  --checkpoint runs/baseline_4090_v1/checkpoints/best.pt \
  --output runs/baseline_4090_v1/test_clean.json
```

It reports:

- cover/stego SNR;
- secret SI-SDR using random Gaussian `z`;
- secret SI-SDR using true redundancy `r` as an oracle.

## 13. Generate a demo pair

Choose any two LibriSpeech `.flac` files:

```bash
python -m audio_inn.demo \
  --checkpoint runs/baseline_4090_v1/checkpoints/best.pt \
  --cover /path/to/cover.flac \
  --secret /path/to/secret.flac \
  --out-dir demo_output
```

Listen to:

```text
demo_output/cover.wav
demo_output/secret.wav
demo_output/stego.wav
demo_output/recovered_random_z.wav
demo_output/recovered_cover_true_r.wav
demo_output/recovered_true_r_oracle.wav
demo_output/metrics.json
```

`metrics.json` records the random-`z` seed, cover/stego SNR, random-`z` and
true-`r` secret SI-SDR, exact-cycle maximum errors, stego peak, and clipping
fraction. The demo seed defaults to the checkpoint's training seed and can be
overridden with `--seed`.

## 14. RTX 3060 12 GB fallback

The baseline is feasible on a 3060 12 GB. A ready-to-use config is included:

```bash
python scripts/preflight.py --config configs/baseline_3060.yaml
python -m audio_inn.train \
  --config configs/baseline_3060.yaml \
  --run-name baseline_3060_v1
```

If necessary also reduce:

```yaml
model:
  hidden_channels: 48
  num_blocks: 6
```

The 3060 should be used mainly for development, tests, evaluation, and smaller runs. Use the 4090 for the final training iteration when available.

## 15. OOM troubleshooting on the 4090

If CUDA OOM occurs:

1. reduce batch `16 -> 8`;
2. keep bfloat16 AMP enabled;
3. reduce `hidden_channels: 64 -> 48`;
4. reduce `num_blocks: 8 -> 6`;
5. verify no other GPU process is consuming VRAM with `nvidia-smi`.

Do not shrink the 1-second segment first; the project is designed around that time unit.
