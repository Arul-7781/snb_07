# Start here: RTX 4090 training bundle

This folder contains the code, configurations, tests and instructions for the sequential
RPM experiment suite. **It does not contain a dataset, trained weights, old runs, a Python
environment, or CUDA installers.** All models in the new study start fresh.

The complete protocol is in [docs/4090_experiment_plan.md](docs/4090_experiment_plan.md).
The older notes under `docs/` are research context; use the commands here for this bundle.

## 1. Requirements

- One RTX 4090 with 24 GB VRAM and a compatible NVIDIA driver.
- 64-bit Python **3.11**; 32 GB RAM recommended.
- Local SSD storage, preferably 70–90 GB free before setup/data preparation.
- FFmpeg and FFprobe on `PATH` for the environment checker.
- Internet for package installation and, if necessary, LibriSpeech download.

PyTorch wheels carry their CUDA runtime. You do not need to install a separate CUDA toolkit.
Linux is the tested development platform; Windows commands are provided but the suite's
Windows process-control path has not been exercised on this development machine.

Extract the entire ZIP first and open a terminal **inside the extracted project folder**.
Do not run Python files from inside a ZIP viewer.

## 2. Install the environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 600 -c constraints-training.txt nvidia-cusparselt-cu12 --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 600 -c constraints-training.txt torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -c constraints-training.txt -e '.[dev,metrics]'
.\.venv\Scripts\python.exe scripts\check_env.py --min-free-gb 40
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

### Linux Bash

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --timeout 600 -c constraints-training.txt \
  nvidia-cusparselt-cu12 --index-url https://pypi.org/simple
python -m pip install --no-cache-dir --timeout 600 -c constraints-training.txt \
  torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -c constraints-training.txt -e '.[dev,metrics]'
python scripts/check_env.py --min-free-gb 40
python -m pytest
python -m ruff check .
```

Stop if installation, CUDA detection, or tests fail. The constraints pin the directly tested
packages, not every platform-dependent transitive dependency. No training quality or 4090
throughput is implied by a software test pass.

## 3. Provide the dataset separately

Copy the complete verified LibriSpeech folders to a fast SSD, for example:

```text
D:\datasets\LibriSpeech\
  train-clean-100\
  train-clean-360\
  dev-clean\
  test-clean\
```

On Linux the corresponding example is `/fastssd/LibriSpeech/`. The root argument must point
to the folder containing these four split directories. Old absolute-path manifests are
not included: the suite builds a local manifest on the new machine.

If you need to download the dataset instead, run **before the timed study**:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_librispeech.py --data-dir "D:\datasets" --download-only
```

Or on Linux:

```bash
.venv/bin/python scripts/prepare_librispeech.py --data-dir /fastssd --download-only
```

These examples produce `D:\datasets\LibriSpeech` or `/fastssd/LibriSpeech` respectively.
The helper verifies archive MD5 and streams extraction without retaining the archive.
It reserves free space and refuses to overwrite incomplete/unverified folders. Follow its
specific recovery message if a download was interrupted; do not delete verified datasets.

## 4. Run the complete experiment set with one command

Preview without starting training:

```powershell
.\.venv\Scripts\python.exe scripts\run_4090_suite.py --plan
```

Windows actual run:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_4090_suite.py --data-root "D:\datasets\LibriSpeech" --output runs\4090_suite_v1 --budget-hours 28 --workers 4
```

Linux actual run:

```bash
.venv/bin/python -u scripts/run_4090_suite.py \
  --data-root /fastssd/LibriSpeech \
  --output runs/4090_suite_v1 --budget-hours 28 --workers 4
```

**Do not create `runs/4090_suite_v1` beforehand**, and do not start another training copy.
Keep the computer awake and use a persistent terminal/session. Do not launch this through
a chat-managed background process that may disappear when its server restarts.

The runner profiles the real hardware and executes matched screens, validation-based
selection, paired second-seed checks, matched continuations and final evaluation in order.
The normal runtime ceiling is **28 hours**, including indexing/profiling/evaluation but
excluding package installation and dataset transfer/download. Choose 24 hours if you need
more setup margin within a 30-hour machine reservation. Early negative screening can stop
the study sooner. A fixed runtime cannot guarantee a quality improvement or convergence.

## 5. Results, stopping and resume

Look at:

```text
runs/4090_suite_v1/REPORT.md
runs/4090_suite_v1/suite_state.json
runs/4090_suite_v1/logs/
```

The report and state distinguish completed, failed, timed-out and incomplete jobs. Full
Python errors are retained in the named logs. Training artifacts, metrics and listening
examples stay under this new suite directory.

To stop, press Ctrl+C once and let the supervisor finish its checkpoint grace period.
Forced termination may preserve only the previous complete checkpoint.

To resume, repeat your original command with the **same arguments** and append:

```text
--resume-suite
```

Resume reuses completed jobs and the original deadline; it does not grant another 28 hours.
Do not change code or configuration midway through a suite. If a recorded child is still
alive, the runner refuses to launch a duplicate; inspect it first.

## 6. Bundle integrity

`BUNDLE_MANIFEST.json` lists each payload file's SHA256, size, original source path and
the source Git commit when available. A separate `.zip.sha256` accompanies the download.
The source repository also contains `scripts/build_training_bundle.py` to regenerate the
allow-listed ZIP. It excludes datasets, existing results/runs, weights, environments,
archives, bytecode and caches. The ZIP itself is intentionally not committed to Git.
