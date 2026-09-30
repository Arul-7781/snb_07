# Audio-INN Two-Week Prototype Scaffold

This repository is a **minimal, trainable Audio-INN baseline** for the capstone's audio-in-audio steganography direction.

The immediate target is a stable prototype in two weeks. Robustness, RPM, diffusion, stego repair, and multimodal AV work are intentionally deferred until the baseline is working.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

./scripts/download_librispeech.sh
python scripts/preflight.py --config configs/baseline_4090.yaml
pytest -q

python -m audio_inn.train \
  --config configs/baseline_4090.yaml \
  --run-name baseline_4090_v1
```

The checked machine has an RTX 3060 12 GB. On that GPU, use
`configs/baseline_3060.yaml` for both preflight and training instead of the
4090 config.

TensorBoard:

```bash
tensorboard --logdir runs --port 6006
```

## Read these first

1. [`docs/CURRENT_GOALS.md`](docs/CURRENT_GOALS.md) — what must be finished for the two-week prototype.
2. [`docs/SETUP.md`](docs/SETUP.md) — complete environment, dataset, training, logging, resume, and evaluation instructions.
3. [`docs/TWO_WEEK_SPRINT.md`](docs/TWO_WEEK_SPRINT.md) — recommended execution order.
4. [`docs/PLAN.md`](docs/PLAN.md) — the broader research roadmap after the prototype.

## What the trainer automatically saves

Every run gets its own directory:

```text
runs/<run_name>/
  train.log
  train_metrics.csv
  val_metrics.csv
  environment.json
  config_resolved.yaml
  tensorboard/
  checkpoints/
    best.pt
    last.pt
    epoch_XXX.pt
  samples/
      epoch_XXX/*.wav
```

Validation logs include random-`z` secret SI-SDR, true-`r` oracle SI-SDR,
forward/reverse cycle error, stego peak level, and clipping fraction. The demo
also writes these checks to `metrics.json` so the one-pass encode/reverse-decode
result can be presented without reconstructing it from console output.

This is intentionally designed so a long RTX 4090 run can be interrupted, inspected, and resumed without losing the experiment configuration.

## Important research note

This scaffold implements a compact WavInWav-style **research baseline**, not an exact reproduction of any published repository. Treat the architecture and hyperparameters as a starting point to be verified/refined before making paper-comparison claims.
