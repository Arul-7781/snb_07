# SPECTRA-INN

Research scaffold for the progressive audio-INN architecture described in
[the final plan](docs/final_audio_inn_plan.md).

The current code intentionally implements **Stage 1 only**: a 16 kHz mono clean audio-in-audio
baseline with a perfect-reconstruction Haar analysis transform and bounded affine coupling.
It distinguishes:

- exact numerical inversion when the true residual is retained;
- blind clean recovery when the residual is replaced by a sampled prior.

It does not yet claim codec robustness, progressive codec payloads, exact blind recovery, or
48 kHz stereo integration. Those enter only after the baseline passes its stop/go tests.

## Windows one-script start

Open PowerShell in the project directory. The launcher creates the Python 3.11 environment, installs
CUDA PyTorch and project dependencies, checks the NVIDIA driver and FFmpeg, runs tests and a GPU
smoke train, downloads and verifies LibriSpeech `train-clean-100`, `dev-clean`, and `test-clean`,
builds the official-split manifest, and starts a wall-clock-limited training run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\train_windows.ps1 -DataDirectory "D:\datasets"
```

On later runs, avoid reinstalling dependencies:

```powershell
.\train_windows.ps1 -DataDirectory "D:\datasets" -SkipInstall -SkipDownload -SkipTests -SkipSmoke
```

Use `-SetupOnly` to stop after environment verification, `-MaxSteps 1000` for a bounded profiling
run, `-TrainingHours 20` to set the wall-clock budget, or
`-Resume "runs\one_day_librispeech\checkpoints\step_00005000.pt"` to resume. Supplying
`-AudioRoot "D:\my_audio"` bypasses automatic dataset download and uses local audio instead.

## Trial 2: predicted latent (LF-VSN RPM)

`configs/rpm_one_day_librispeech.yaml` replaces the decoder's Gaussian latent with a learned
redundancy predictor, as in LF-VSN. See [the trial guide](docs/rpm_trial.md) for the exact steps
from download to evaluation.

## Quick start

Read [training prerequisites](docs/training_prerequisites.md) first. In a Python 3.11 environment:

```bash
# Install the CUDA build selected at https://pytorch.org/get-started/locally/
python -m pip install torch torchaudio --index-url <PYTORCH_CUDA_INDEX_URL>

python -m pip install -e '.[dev,metrics]'
python scripts/check_env.py
python -m pytest
python -m spectra_inn.train --config configs/baseline_16k.yaml --smoke --max-steps 10
```

Index local audio and start real training:

```bash
python -m spectra_inn.data.manifest \
  --root /absolute/path/to/audio \
  --output data/manifests/all.jsonl

python -m spectra_inn.train --config configs/baseline_16k.yaml
```

The manifest builder creates deterministic track-level train/validation/test splits. Inspect the
manifest before training and keep speakers, songs, and source recordings from leaking across splits.

## Current package layout

```text
configs/                 Reproducible experiment configurations
data/manifests/          JSONL indexes only; audio remains outside Git
docs/                    Architecture and training documentation
scripts/                 Environment and operational checks
src/spectra_inn/
  data/                  Manifest and paired segment dataset
  losses/                Waveform, STFT, and latent objectives
  metrics/               Lightweight training-time metrics
  models/                Invertible baseline and coupling blocks
  transforms/            Exact Haar analysis/synthesis
  train.py               AMP, accumulation, validation, checkpoint/resume
tests/                   Transform and model invariants
```

The planned router, secret codec, robust embedder/extractor, channel simulator, steganalyzer, and
A/V synchronization packages are documented but should be added stage-by-stage, not as empty
training claims.
