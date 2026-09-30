# Training prerequisites and bootstrap

This document is the operational companion to the
[SPECTRA-INN research plan](final_audio_inn_plan.md). It covers Stage 0 and Stage 1: environment,
data, baseline training, and the gates that must pass before progressive/robust training begins.

## 1. Assumptions

- Linux host, one NVIDIA RTX A4500 20 GB or RTX 4500 Ada 24 GB.
- Baseline: 16 kHz mono cover and secret, two-second random crops.
- Integration target: 48 kHz stereo cover and 16 kHz mono secret.
- Primary high-capacity channel: lossless WAV/FLAC or unchanged PCM.
- Robust decoding is a later, lower-rate graceful-degradation mode.
- Exact original-cover recovery is not assumed when the forward residual is discarded.

## 2. Hardware

Minimum to run the baseline:

- RTX 4500-class GPU with 20 GB VRAM.
- 32 GB system RAM; 64 GB recommended for preprocessing and codec evaluation.
- 100 GB free storage for a small speech-first experiment.
- SSD/NVMe storage; random crop loading from a slow disk will starve the GPU.

Recommended for the complete data and experiment matrix:

- 250–500 GB free storage for original audio, normalized caches, codec caches, checkpoints,
  real-codec outputs, and steganalysis datasets.
- Separate checkpoint and dataset directories if possible.

The current workspace has about 108 GB free, which is sufficient for the baseline but not a broad
music/speech/environmental corpus plus all generated artifacts. Do not duplicate full waveforms into
the repository.

## 3. System software

Required:

- Ubuntu or another recent Linux distribution.
- A working NVIDIA driver visible through `nvidia-smi`.
- Python 3.11; Python 3.12 may work, but 3.11 is the compatibility baseline.
- FFmpeg and FFprobe with MP3, AAC, Opus, OGG, and FLAC support.
- `libsndfile`, a C/C++ build toolchain, Git, and Git LFS.

Ubuntu packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential ffmpeg git git-lfs libsndfile1 libsndfile1-dev pkg-config
```

The local host already has FFmpeg 7.1 with the required common codecs. At inspection time,
`nvidia-smi` could not communicate with the driver, so CUDA training must not begin until that is
fixed and the GPU name/VRAM are visible.

## 4. Python and CUDA environment

Do not use the workspace's current Python 3.13 environment for the initial research stack. Create an
isolated Python 3.11 environment:

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the stable PyTorch build selected for the installed NVIDIA driver from the official selector:

```text
https://pytorch.org/get-started/locally/
```

Example shape only—replace the index URL with the selector's current recommendation:

```bash
python -m pip install torch torchaudio --index-url <PYTORCH_CUDA_INDEX_URL>
```

Do not install a separate CUDA toolkit merely because a tutorial says so. PyTorch wheels carry the
needed CUDA runtime; the host driver must be compatible with the selected wheel.

Install this project:

```bash
python -m pip install -e '.[dev,metrics]'
python scripts/check_env.py
```

The environment check must report:

- Python 3.11.x.
- PyTorch import succeeds.
- `torch.cuda.is_available()` is true.
- Expected RTX GPU and approximately 20 or 24 GB VRAM.
- FFmpeg/FFprobe found.
- At least 100 GB free disk for baseline work.

## 5. Optional research dependencies

Install these only when their stage begins:

### Frozen secret codec

EnCodec supports a 24 kHz mono model and a 48 kHz stereo model, but its software stack is older than
the core scaffold. Put it in a separate preprocessing environment if it conflicts with the training
environment. Precompute codebooks/embeddings rather than running a frozen codec in every training
step.

### ViSQOL

ViSQOL has its own Bazel/build dependencies. Keep it outside the training loop and run it on exported
validation WAV files. Use 16 kHz speech mode for the speech baseline and 48 kHz audio mode for the
integration target.

### PESQ/STOI and ASR

PESQ and STOI apply primarily to speech and have sample-rate/domain restrictions. Install and run them
as milestone evaluation jobs, not per-step differentiable losses. ASR WER and speaker embedding
models should also live in the evaluation environment to preserve training VRAM.

### Experiment tracking

TensorBoard is installed by default. Add Weights & Biases or MLflow only if the project needs a remote
tracker; never make network access a prerequisite for resuming training.

## 6. Data prerequisites

Start with speech secrets and mixed cover domains.

### One-day Windows default

`train_windows.ps1` automatically downloads the official OpenSLR LibriSpeech `train-clean-100`,
`dev-clean`, and `test-clean` archives. It verifies their published MD5 checksums, extracts them,
and records the official train/validation/test assignments in the manifest. This is the recommended
one-day baseline because the audio is already 16 kHz, the training archive is about 6.3 GB for 100
hours of clean multi-speaker speech, and no training-time resampling is required.

The default `configs/one_day_librispeech.yaml` run uses an effective batch of 16 two-second pairs,
three STFT resolutions, a maximum of 150,000 optimizer steps, and a 20-hour wall-clock limit. It
always validates and saves a final checkpoint when the time limit is reached.

Baseline minimum:

- 50–100 hours of training audio.
- At least 5 hours each for validation and held-out testing.
- Multiple speakers and recording conditions.
- Enough independent tracks that splits occur at track/speaker level, not crop level.

Recommended research set:

- Secret speech: VCTK plus a LibriSpeech subset.
- Cover speech: disjoint speakers/recordings.
- Cover music: legally usable full tracks across genres.
- Environmental covers: a licensed diverse sound-event corpus.

Keep raw audio outside the repository. The manifest stores absolute paths and metadata.

Data rules:

1. Decode once to PCM WAV or FLAC for training; do not repeatedly invoke FFmpeg in DataLoader workers.
2. Preserve source files and record every resampling/downmix operation.
3. Reject corrupt, clipped, near-silent, and too-short files during validation or record them in an
   explicit exclusion manifest.
4. Split before segmenting. All chunks from one speaker/session/song/recording stay in one split.
5. Do not peak-normalize every crop. It erases natural level statistics and makes the task easier in
   an unrealistic way.
6. Record licenses, dataset version, checksums, source domain, speaker/track identifiers, original
   sample rate, and channel count.

## 7. Manifest format

Each line is one JSON object:

```json
{"path":"/abs/audio.wav","sample_rate":48000,"channels":2,"frames":240000,"duration":5.0,"split":"train","source":"music_corpus"}
```

Build a starter manifest:

```bash
python -m spectra_inn.data.manifest \
  --root /absolute/path/to/audio \
  --output data/manifests/all.jsonl \
  --train-ratio 0.90 \
  --val-ratio 0.05 \
  --source local_corpus
```

The builder hashes relative file paths for deterministic splits. For VCTK/LibriSpeech/music, replace
this generic split with group-aware manifests based on speaker or track ID before reporting results.

## 8. Pre-training validation

Run in order:

```bash
python scripts/check_env.py
python -m pytest
python -m spectra_inn.train \
  --config configs/baseline_16k.yaml \
  --smoke \
  --max-steps 10
```

Expected smoke-test behavior:

- No NaNs or infinities.
- Exact-cycle maximum absolute error near float32 numerical precision.
- Checkpoint and TensorBoard event files appear under `runs/baseline_16k_clean/`.
- Loss changes are not a quality claim; ten synthetic steps only validate plumbing.

Then run an overfit test on 16–64 real segments. The model must drive training secret/cover losses
down and retain a small exact-cycle error. Do not start a multi-day run before this succeeds.

## 9. Stage-1 training

Start:

```bash
python -m spectra_inn.train --config configs/baseline_16k.yaml
```

Resume by setting `training.resume` in a copied YAML file or passing:

```bash
python -m spectra_inn.train \
  --config configs/baseline_16k.yaml \
  --resume runs/baseline_16k_clean/checkpoints/step_00005000.pt
```

Before the full run, profile 1,000 steps and record:

- examples/second and audio-seconds/second;
- peak allocated/reserved VRAM;
- GPU utilization;
- DataLoader wait time;
- checkpoint size;
- validation wall time.

Tune in this order when memory is insufficient:

1. Lower micro-batch size.
2. Increase gradient accumulation to preserve effective batch.
3. Reduce hidden width from 64 to 48.
4. Enable/check activation checkpointing when added.
5. Reduce crop duration only for profiling; restore the target duration for the reported model.

## 10. Required Stage-1 gates

Do not proceed to codec attacks or the progressive router until all are true:

1. Haar analysis/synthesis and model forward/reverse tests pass on CPU and GPU.
2. Exact residual cycle has stable maximum error without exploding affine scales.
3. Blind clean recovery works on held-out speakers/tracks, not just training pairs.
4. Cover waveform has no clipping and preserves sample count.
5. Cover/stego and secret/recovery metrics are exported separately.
6. A matched non-invertible baseline configuration is planned for comparison.
7. A fixed listening subset has been reserved before extensive tuning.

## 11. Staged files still to add

After Stage 1 passes, add in this order:

```text
src/spectra_inn/
  codecs/
    base_enhancement.py       Frozen codec adapter and cached representation format
  routing/
    masking.py                Auditory masking headroom
    survival.py               Empirical/learned attack-survival score
    router.py                 Budgeted base/enhancement allocation
  robust/
    packet.py                 Framing, CRC, interleaving, ECC boundary
    embedder.py               Robust base residual
    extractor.py              Soft-bit/base-token decoder
  channel/
    differentiable.py         Training-time attacks
    ffmpeg_roundtrip.py       Real validation codecs
  sync/
    pilot.py                  Keyed pilot and rolling counter
    clock_fit.py              Drift/offset estimation
  steganalysis/
    detector.py               Separately trained security baseline
  av/
    segment.py                Shared integer PTS contract
```

Each addition needs a unit test, a config switch, an ablation with the feature disabled, and a payload
or distortion budget. Empty modules do not count as completed architecture.

## 12. Reproducibility record

Every reported run should archive:

- Git commit or source snapshot.
- Complete YAML configuration.
- `pip freeze` and PyTorch/CUDA/driver versions.
- Dataset manifest checksum and exclusion list.
- Random seed and exact resume checkpoint.
- Model parameter count and peak VRAM.
- Training/validation curves.
- Clean and attacked evaluation tables.
- Five representative cover/stego/secret/recovered audio examples.

Do not select only the best checkpoint by training loss. Select using a predefined validation score
that constrains cover quality first and secret recovery second.
