# RPM training on Ryzen 5 3600 / RTX 3060 12 GB

These commands target the inspected **Ubuntu/Linux** host. Run them from
`spectra_inn_windows` in Bash. Do not use `train_windows.ps1` for this workflow: it is a
Windows launcher and retains dataset archives.

## Dataset and storage choice

- Pilot: LibriSpeech `train-clean-100`, `dev-clean`, `test-clean`.
- Main experiment: add `train-clean-360`, for approximately **460 training hours**.
- Official validation/test splits stay separate. Both cover and secret are drawn from the
  relevant split; this is speech-in-speech, not a music-cover experiment.
- All four extracted splits should occupy roughly 31–34 GB. Allow another 5–10 GB for
  the environment and about 1 GB for outputs. No decoded waveform cache is generated.
- The helper streams archives into a staging directory, checks the published archive MD5,
  and only then promotes the split. It retains **no compressed archive**, avoiding a second
  24 GB copy while extracting `train-clean-360`.
- Its conservative storage estimate requires **10 GiB to remain free** and checks the
  reserve during extraction. Do not add the 500-hour split on this space budget.
- This is an economical use of the space, not a reason to fill the disk. A bigger dataset
  improves diversity, but does not make a fixed 20-hour run converge faster. Stop at 100 hours
  if the pilot fails or storage planning rejects the larger corpus.
- LibriSpeech is CC BY 4.0: <https://www.openslr.org/12/>.

## 1. Create the environment

The inspected default Python was 3.13 and lacked PyTorch. `uv` is already installed.
Do not overwrite an existing environment; inspect it first if `.venv` already exists.

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache -e '.[dev,metrics]'
python scripts/check_env.py --min-free-gb 40
python -m pytest
python -m ruff check .
```

The CUDA 12.8 wheel is compatible with the inspected NVIDIA 595 driver and RTX 3060.
A separate CUDA toolkit is not required. The implemented loader uses SoundFile, not
torchaudio. FFmpeg/FFprobe were already present. Recheck the environment if training on
another machine. The low-VRAM warning is advisory, but CUDA must be available.

## 2. Run the synthetic smoke check

```bash
python -m spectra_inn.train \
  --config configs/rpm_3060_librispeech.yaml \
  --device cuda --smoke --max-steps 10
```

Check for finite losses and a small float32 exact-cycle error. Synthetic smoke results are
not evidence of real secret recovery. Outputs use the separate `_smoke` run directory.

## 3. Prepare the 100-hour pilot

```bash
python scripts/prepare_librispeech.py --train-splits train-clean-100 --plan
python scripts/prepare_librispeech.py --train-splits train-clean-100
```

Defaults place FLAC files in `datasets/librispeech/LibriSpeech/` and write the new, separate
manifest `data/manifests/rpm_librispeech.jsonl`. Its `.summary.json` contains counts and hours.
Use `--data-dir /path/on/another/ssd` on both preparation commands if desired; the manifest
records absolute paths, so the training config does not need changing.

The helper refuses to overwrite an existing unverified dataset. Existing verified splits
are reused using their completion marker; it does not re-hash every audio file on reruns.
Do not modify those files after verification.

Streaming saves disk but **cannot resume a download**. On interruption/checksum failure,
the helper prints and retains only that split's `.partial-<split>` staging directory for
inspection. Delete only the printed partial directory before retrying; do not delete the
whole dataset directory or use unverified partial audio for training. Checksum verification
happens after extraction, and the partial directory is never indexed.

## 4. Profile on real data before a long run

```bash
mkdir -p runs/rpm_3060_librispeech
set -o pipefail
time python -u -m spectra_inn.train \
  --config configs/rpm_3060_librispeech.yaml \
  --device cuda --max-steps 1000 --max-hours 1 \
  2>&1 | tee runs/rpm_3060_librispeech/profile.log
```

The pilot stops at 1,000 total optimizer steps or one hour, whichever comes first, and saves
a final checkpoint. Watch `nvidia-smi` in another terminal. Record elapsed time, completed
steps, GPU memory, loss, and validation metrics. Disregard the first TensorBoard throughput
point: the current trainer divides by `log_every` even at step 1.

If memory is insufficient, change batch size to 2 and accumulation to 8 (effective batch
still 16). If CPU loading/RAM is the bottleneck, try two workers instead of four. Do not
reduce model width or crop duration before trying the smaller micro-batch.

## 5. Add 360 hours, then resume for up to 20 hours

Only after the pilot is finite and exact-cycle diagnostics are stable:

```bash
python scripts/prepare_librispeech.py --plan
python scripts/prepare_librispeech.py
CHECKPOINT="$(find runs/rpm_3060_librispeech/checkpoints -maxdepth 1 -name 'step_*.pt' | sort | tail -n 1)"
test -n "$CHECKPOINT"
python -u -m spectra_inn.train \
  --config configs/rpm_3060_librispeech.yaml --device cuda \
  --resume "$CHECKPOINT" --max-hours 20 \
  2>&1 | tee runs/rpm_3060_librispeech/training.log
```

Preparation reuses the verified pilot splits and atomically rebuilds the manifest with
both training subsets, not duplicated records. It must finish while training is stopped.
Resume loads the pilot's model/optimizer; the run then draws from the expanded manifest.

The config keeps the original RPM architecture, uses AMP and micro-batch 4 with accumulation
4, validates every 1,000 steps, saves every 1,000 steps, and retains three checkpoints.
The 20-hour limit starts after setup/resume and includes periodic training diagnostics;
final validation/save can extend it. Each resumed invocation gets a fresh time budget.
The 150,000-step cap is cumulative across resumes, not 150,000 new steps per invocation.
This time budget is an experiment, not a quality guarantee.

Keep the terminal/session alive. Ctrl+C or a power interruption does **not** guarantee a
fresh checkpoint; resume from the newest completed `.pt`. For another day's training,
repeat the checkpoint-selection and training commands (use a new log filename).

## 6. Watch training

In a second terminal, activate `.venv`, then:

```bash
tensorboard --logdir runs/rpm_3060_librispeech --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>. Watch cover SNR, predicted-secret SNR, exact-cycle error,
stego peak, and validation loss together. Poor Gaussian decoding in an RPM run is expected:
that decoder is not the one being trained. The true-residual result is an inversion ceiling,
not the result available to a recipient who only has stego audio.

## 7. Evaluate and listen

After training stops, select the latest checkpoint again (not the old pilot checkpoint):

```bash
CHECKPOINT="$(find runs/rpm_3060_librispeech/checkpoints -maxdepth 1 -name 'step_*.pt' | sort | tail -n 1)"
test -n "$CHECKPOINT"
python -m spectra_inn.evaluate \
  --checkpoint "$CHECKPOINT" --device cuda \
  --batch-size 4 --max-batches 1000 --examples 10
```

At batch size 4, 1,000 batches are enough to cover the official `test-clean` records. This
evaluates fixed two-second crops, **not every second of every recording**. The current
evaluator pairs neighboring records, which may be from the same speaker; it is a first
diagnostic, not a comprehensive cross-speaker/random-pair benchmark. Training-time
validation only checks 20 batches. Do not select a checkpoint based on repeated test-set tuning.

Results appear under `runs/rpm_3060_librispeech/evaluation/<checkpoint>_test/`:

- `metrics.json`: cover SNR/SI-SDR, clipping, predicted-secret SNR/SI-SDR, true-residual
  diagnostics, and predicted decoding after PCM16 quantization.
- `example_000/` through `example_009/`: cover, secret, stego, and recovered-secret WAVs.

Listen to `stego.wav` against `cover.wav`, and `recovered_secret_predicted.wav` against
`secret.wav`. Check both ordinary predicted metrics and `predicted_pcm16`: large degradation
after PCM16 rounding means saved audio is hurting recovery. Exported WAVs are clipped to
[-1, 1] and normally PCM16, so float-domain scores alone do not establish delivered quality.

To have the assistant review real results later, provide the evaluation directory or return
to this workspace and ask it to read `metrics.json`, `training.log`, and the dataset summary.
No results can be reported until training/evaluation have actually run. A fair RPM-versus-
Gaussian comparison additionally requires a separate matched baseline with the same dataset,
effective batch, and training budget; the RPM model's own Gaussian decode is not that baseline.
