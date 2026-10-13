# Trial 2: LF-VSN redundancy prediction for audio

Branch: `audio-trial2`

## What changed

The baseline throws away the INN's second output (the residual `r`) and, at decode time, puts
random Gaussian noise `z` in its place. LF-VSN instead predicts that missing piece from the stego
with a small network, the redundancy prediction module (RPM). This trial ports that idea to audio.

| Piece | Where |
|---|---|
| `RedundancyPredictor1d`: conv in, 8 residual blocks (conv, LeakyReLU 0.2, conv, skip, no normalization), conv out | `src/spectra_inn/models/redundancy.py` |
| `model.latent_mode: gaussian \| predicted`, `CleanAudioINN.predicted_inverse()` and `decode()` | `src/spectra_inn/models/clean_inn.py` |
| Training decodes with the configured latent; validation also logs the Gaussian and true-residual decodes | `src/spectra_inn/train.py` |
| Trial config: `one_day_librispeech.yaml` plus the predictor, with `latent_prior: 0.0` | `configs/rpm_one_day_librispeech.yaml` |
| Test-split scoring and WAV export | `src/spectra_inn/evaluate.py` |

How it maps to LF-VSN (`sbn_07_video/code/models/modules/Inv_arch.py`):

- **Input.** The predictor sees the stego's Haar coefficients, the same tensor the inverse INN
  receives as its first stream.
- **Output.** It produces the second stream, in place of noise.
- **Training signal.** It is trained only through the secret and cover recovery losses. Nothing
  compares its output to the true residual.
- **Prior loss off.** `latent_prior` is set to 0 because the residual no longer has to look like
  noise.
- **Extra output layer.** One addition beyond LF-VSN: a zero-initialized output convolution. It
  is needed because the feature width (64) differs from the latent width (8). It also means
  training starts from a zero latent.
- **Not included.** LF-VSN's key-controlled RPM variant is not part of this trial.

The existing baseline configs still use `latent_mode: gaussian`, so their behaviour is unchanged.

## Running it on Windows

### 0. Prerequisites (one time)

- NVIDIA GPU with a current driver. `nvidia-smi` must work in PowerShell.
- 64-bit Python 3.11 from python.org, with the `py` launcher.
- FFmpeg with `ffmpeg` and `ffprobe` on `PATH`.
- Git for Windows.
- About 30 GB of free disk for LibriSpeech (archives plus extracted audio) and run outputs.
  `check_env.py` warns below 100 GB, but that warning is advisory.

### 1. Get the branch

```powershell
git clone https://github.com/Arul-7781/snb_07.git
cd snb_07
git checkout audio-trial2
cd spectra_inn_windows
```

If the repository is already cloned, use `git fetch origin` then `git checkout audio-trial2`.

### 2. Install, download LibriSpeech, and train the predictor model

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\train_windows.ps1 -DataDirectory "D:\datasets" -ConfigPath configs\rpm_one_day_librispeech.yaml
```

In order, this one command:

1. Creates `.venv` and installs CUDA PyTorch and the project.
2. Checks the GPU and FFmpeg.
3. Runs `pytest`, `ruff`, and a 10-step smoke train.
4. Downloads and MD5-verifies the LibriSpeech `train-clean-100`, `dev-clean`, and `test-clean`
   splits into `D:\datasets`.
5. Builds `data\manifests\all.jsonl`.
6. Trains for up to 20 hours (`-TrainingHours` changes this).

Outputs go to `runs\rpm_one_day_librispeech\`.

### 3. Train the Gaussian baseline for comparison

The comparison only means something against a baseline trained the same way. After step 2
finishes, run the baseline. Run the two jobs one after the other, because sharing one GPU slows
both.

```powershell
.\train_windows.ps1 -DataDirectory "D:\datasets" -ConfigPath configs\one_day_librispeech.yaml `
  -SkipInstall -SkipDownload -SkipTests -SkipSmoke
```

Outputs go to `runs\one_day_librispeech\`.

### 4. Watch training live

In a second PowerShell window:

```powershell
cd <path>\snb_07\spectra_inn_windows
.\.venv\Scripts\Activate.ps1
tensorboard --logdir runs --port 6006
```

Open http://localhost:6006. Both runs appear side by side. The scalars that matter:

| Scalar | Meaning | Want |
|---|---|---|
| `validation/cover_snr_db` | how close the stego is to the cover | high, and similar across runs |
| `validation/secret_snr_db_predicted` | secret recovered with the predictor (trial run) | higher than the baseline's Gaussian number |
| `validation/secret_snr_db_gaussian` | secret recovered with random noise (meaningful for the baseline run) | baseline reference |
| `validation/secret_snr_db_true_residual` | secret recovered with the real residual, the ceiling | very high; this is the exact-inverse check |
| `validation/latent_prediction_snr_db` | how close the predicted latent is to the real residual | diagnostic only |

The console also prints one line per validation, for example:
`secret_snr=... [predicted=... gaussian=... true_residual=...]`.

In the predictor run, `secret_snr_db_gaussian` is expected to be poor, because that model was
never trained to accept noise. Compare the predictor run's `predicted` value against the
baseline run's `gaussian` value.

### 5. Resume after an interruption

Checkpoints are saved every 5000 steps, and the last 3 are kept.

```powershell
dir runs\rpm_one_day_librispeech\checkpoints
.\train_windows.ps1 -DataDirectory "D:\datasets" -ConfigPath configs\rpm_one_day_librispeech.yaml `
  -SkipInstall -SkipDownload -SkipTests -SkipSmoke `
  -Resume runs\rpm_one_day_librispeech\checkpoints\step_00050000.pt
```

A resumed run gets a fresh `-TrainingHours` budget.

### 6. Score on the test split and listen

Use the newest checkpoint of each run. Run `dir runs\<run>\checkpoints` to see the names. A
run stopped by the 20-hour limit ends before step 150000, so replace `step_00150000.pt` with the
newest file listed.

```powershell
.\.venv\Scripts\python.exe -m spectra_inn.evaluate `
  --checkpoint runs\rpm_one_day_librispeech\checkpoints\step_00150000.pt
.\.venv\Scripts\python.exe -m spectra_inn.evaluate `
  --checkpoint runs\one_day_librispeech\checkpoints\step_00150000.pt
```

Each command writes to `runs\<run>\evaluation\<checkpoint>_test\`:

- `metrics.json`, which is also printed. It contains cover SNR and SI-SDR, and secret SNR and
  SI-SDR for each decoder latent: `predicted` or `gaussian`, `true_residual`, and `*_pcm16`.
  `*_pcm16` is the same decode after the stego is rounded to 16-bit WAV samples, which is what
  happens when it is saved to disk.
- `example_000\` and onwards, each holding `cover.wav`, `secret.wav`, `stego.wav`, and
  `recovered_secret_<latent>.wav`.

The defaults are 50 batches of 8 test pairs and 5 exported examples. Change them with
`--max-batches`, `--batch-size`, and `--examples`.

### 7. Reading the result

The trial helps if both of these hold:

- the predictor run's `secret_si_sdr_db_predicted` is above the baseline run's
  `secret_si_sdr_db_gaussian`;
- `cover_snr_db` is about the same in both runs.

For reference, LF-VSN reports about +0.6 dB secret and +0.3 dB stego from its RPM on video.

The gap between `predicted` and `true_residual` is how much room is left for a better predictor.
`*_pcm16` numbers far below their float counterparts mean that saving the stego to WAV hurts
recovery. Neither model is trained with that rounding yet.

`runs\` is git-ignored. To share results, commit the `metrics.json` files and the TensorBoard
event files, not the checkpoints or WAVs.
