# Two-Week Sprint

## Days 1–2 — setup and smoke test

- Create venv and install package.
- Download/extract LibriSpeech `train-clean-100`, `dev-clean`, `test-clean`.
- Run `python scripts/preflight.py`.
- Run `pytest -q`.
- Run smoke training with `configs/smoke.yaml`.

Deliverable: one checkpoint and WAV samples from a tiny run.

## Days 3–5 — baseline training

- Start `configs/baseline_4090.yaml` on RTX 4090.
- Watch TensorBoard.
- Verify:
  - training loss decreases;
  - cover/stego SNR trends upward;
  - secret SI-SDR improves;
  - no NaNs/clipping explosion.
- Tune only if necessary:
  - learning rate;
  - loss weights;
  - number of blocks;
  - hidden width.

Deliverable: stable run with resumable checkpoint.

## Days 6–8 — evaluate and listen

- Evaluate `best.pt` on `test-clean`.
- Generate multiple demo pairs.
- Compare random-`z` recovery with true-`r` oracle.
- Save TensorBoard plots/screenshots for presentation.

Deliverable: metrics table + listening examples.

## Days 9–11 — one controlled improvement

Only if baseline is stable, improve one thing at a time:

- better loss weights;
- slightly wider/deeper INN;
- longer training;
- better latent prior regularization.

Do **not** begin diffusion unless the baseline is already presentation-ready.

## Days 12–14 — freeze prototype

- rerun evaluation on fixed test manifest;
- prepare final demo WAVs;
- archive best checkpoint/config;
- document results and known limitations;
- optionally run the clean four-case oracle scaffold as a teaser for the next phase.

The priority is a demonstrable, reproducible baseline, not maximum novelty within two weeks.
