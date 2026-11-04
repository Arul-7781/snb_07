# Next RPM study: one RTX 4090, a 24–30-hour budget

**Prepared:** 29 October 2026. **Default suite deadline:** 28 hours.

This is a new, sequential study, not another unbounded overnight run. The runner profiles
the actual machine, chooses equal-step budgets, applies predeclared validation gates,
and writes a report. It may finish early if a challenger fails the gates or a step cap
is reached. It never pads a weak experiment merely to consume 24 hours.

**No 4090 training has been started during preparation of this plan.** The code has CPU
unit/integration checks, but 4090 throughput is intentionally measured on the destination
machine. Neither a speedup over the 3060 nor a 30 dB recovery score is assumed.

## 1. What the plan carries forward

### Locally verified evidence

From [our complete experiment record](experiment_history_and_results.md):

| Finding | Consequence for this study |
|---|---|
| NMSE 10k test checkpoint: 15.76 dB cover, 7.90 dB secret, 6.57 dB secret SI-SDR | Keep energy-normalized waveform supervision. |
| Original Charbonnier RPM 81k: 10.19 dB cover, 6.99 dB secret on the same test-crop protocol | Train a new matched control, rather than call unequal historical budgets a clean ablation. |
| Flat dilation did not help the matched 16-pair diagnostic | Do not repeat the same flat-dilation trial as the principal experiment. |
| Secret STFT gradients dominated the old absolute waveform objective | Do not revert to the old Charbonnier setup or simultaneously raise all STFT weights. |
| PCM16 rounding barely changed recovery at the current quality level | Measure it throughout, but do not spend a main arm on quantization training yet. |
| First-80-pair validation overstated whole-split scores | Use speaker-spread monitoring and full validation for promotion. |
| A background job was cancelled when the server restarted | Run the suite in a persistent user terminal, with per-job logs and resume state. |

The old test numbers remain historical references. New candidates are selected using
validation, not by repeatedly testing against those known test scores.

### Additional evidence supplied by the team

The following document was read in full:

```text
/media/hunt/5CC89FE6C89FBD28/pc_download/RPM_ITERATION_HISTORY.md
SHA256: cb0f145fa82525d6aa254ae4b5e7fab59de185e0bc0dc7fbbf5eb1df7df260f9
```

A [provenance note](references/README.md) preserves the reviewed source identity. The original
was unavailable at packaging time on 30 October 2026, so no verbatim copy is claimed. The
summary here remains available without the mounted drive.

The document describes branch `audio-rpm-improve`. Its key reported observations are:

- An initial RPM reached 200k steps with approximately 14.57 dB cover / 11.05 dB secret
  **validation** SNR. The exact evaluation coverage is not established by that document.
- Charbonnier residual supervision at weight 0.1 alone still gave approximately 10.46 dB
  secret SNR after 16k steps.
- Wider/dilated predictors, separate predictor learning rates, an oracle warm-up and a
  gradual oracle/predicted output blend were tried or proposed.
- A direct secret decoder was retained as an ablation, not the primary INN/RPM path.
- The current proposed primary predictor is a multiscale U-Net RPM, width 128 / nine blocks.
- A 4090 one-step execution check passed. That is not a model-quality result.
- A previous multiscale run retained 5k/10k/15k checkpoints before a wrapper exit code 1;
  no final traceback or verified complete quality report is supplied there.

Those are **reported external results**, not locally reproduced checkpoint measurements.
They must not be mixed into the same test-results table as our local runs.

The described newer branch's implementation was not present in this directory. The
multiscale module added here is a documented implementation of that idea, not an assertion
that we reproduced its exact architecture, weights, curriculum or optimizer recipe.

### What we deliberately avoid

- No direct secret decoder in the primary result: decoding remains stego → RPM → INN inverse.
- No true-residual/output blending in training or scoring. A warm-up's oracle contribution
  could hide a failure of genuine blind recovery. All reported secret scores use predicted
  residuals from the first training step.
- No simultaneous change to architecture, secret/STFT weights, learning rate groups and
  curriculum. That would make a gain hard to explain.
- No claim that a high oracle SNR proves that a bigger RPM can reconstruct information
  absent from stego. The residual-information bottleneck remains a fundamental risk.

## 2. Experiment matrix

All four screen arms start fresh. They use the same corpus, seed, data-order seed, crop
duration, effective batch, NMSE objective, INN, learning rate and monitoring protocol.

| Arm | Predictor | Total parameters | Training change being tested |
|---|---|---:|---|
| A: `a_nmse_control` | Original flat RPM, width 64, eight blocks | 878,344 | Matched contemporary control using our successful NMSE recipe. |
| B: `b_flat_capacity` | Flat RPM, width 152, nine blocks | 1,935,376 | Whether extra capacity alone helps. |
| C: `c_unet` | Three-scale U-Net RPM, width 128, nine blocks | 1,899,080 | Multiscale context and skip connections instead of a larger flat predictor. |
| D: `d_unet_latent` | Same U-Net as C | 1,899,080 | Add light, detached-target normalized residual supervision. |

B is within approximately **2% of C's total parameter count**, making it a more useful
capacity control than comparing a large U-Net only against the tiny original RPM.
It is approximately parameter-matched, not exactly architecture- or compute-matched.

The U-Net uses three resolutions, two stride-2 downsampling layers, interpolation back
to the skip sizes, and a zero-initialized output projection. Its configured dilations are:

```yaml
[1, 2, 4, 2, 4, 8, 4, 8, 16]
```

The predictor's estimated receptive-field **upper bound** is approximately 307 ms. Unlike
the flat predictor's exact 18.5 ms field, interpolation/position can affect the precise
support. This is an RPM-context description, not a whole-encoder receptive-field claim.

D alone adds:

```text
0.05 * NMSE(stop_gradient(true_residual), predicted_residual)
```

The previous auxiliary result used a different loss scale and broader training history.
This normalized auxiliary term is a hypothesis, not a guaranteed repair. Detaching the
target stops gradients through that target in the current backward pass; it does not
freeze the INN or stop targets changing between steps. The INN can still receive gradients
through stego and the other losses.

### Shared recipe

- 16 kHz mono, two-second crops, three Haar levels, eight original INN blocks.
- Effective batch **16 pairs**. Start at micro-batch 8 / accumulation 2 on a 24 GB 4090.
- If profiling hits an OOM, reprofile **all** arms at 4/4, then 2/8 or 1/16 if necessary.
- Initial learning rate `1e-4`, AdamW, gradient clipping 5, FP16 AMP.
- Initial AMP scaler 1,024 for all new arms to reduce startup overflow. This differs from
  the historical default and is part of the shared new recipe, not a hidden advantage for C/D.
- Existing NMSE/STFT weights stay unchanged; latent prior remains off.
- Stage-two refinement applies an explicit `3e-5` learning rate to **both** finalists after
  checkpoint load. The historical YAML-only resume-LR trap is handled explicitly.
- Checkpoint every 1k attempted steps, retaining three per phase. Screen checkpoints live
  separately and are not pruned away by refinement.
- Screen seed 3407; paired confirmation seed 7391; evaluation pair seed 20261029.
- Independent data-loader generators keep initialization size from silently changing the
  arms' data order/worker seeds. Different seeds intentionally change training data order.

## 3. What the one command executes

### Phase 0 — Preflight and real-data throughput profiling

1. Check CUDA, GPU identity, output-disk headroom and the installed PyTorch/CUDA versions.
2. Rebuild a manifest for this machine from complete official split folders. Check record
   counts, 16 kHz mono format and speaker-ID separation.
3. Run 200 profiling steps for each arm. These weights are discarded as initializations;
   actual screens start fresh.
4. Record end-to-end seconds per attempted step, peak allocated GPU memory and AMP skips.
5. Choose a common micro-batch and common screening step budget.

The profiler uses the real loader and losses, not synthetic throughput or a TFLOPS ratio.
No automatic CPU fallback is allowed. An explicit `--allow-other-gpu` exists for development,
but it does not make another device's runtime a verified 4090 result.

### Phase 1 — Four matched screens

Run A, B, C, then D sequentially to the **same number of attempted training steps**.
The calibrated screen is between **20k and 50k steps per arm**. If measured throughput
cannot safely fit the minimum under the requested study budget, stop after profiling and
report that fact rather than silently turn a 1k smoke test into a quality benchmark.

After each screen, score all validation records in two protocols:

- `legacy`: the original neighbor-pair protocol, for continuity with prior results.
- `cross_speaker`: a reproducible different-speaker secret for each cover. Silence retries
  remain within each selected speaker; the rule is never silently relaxed.

Training-time monitoring uses **320 speaker-spread validation covers**, not the first 80
sorted pairs. Full offline validation, not the monitor loss, drives promotion.

### Phase 2 — Validation-only promotion

Rank B/C/D by cross-speaker predicted-secret SNR, but accept a challenger only if:

- Cross-speaker secret SNR improves by at least **0.5 dB** over A.
- Cover SNR falls by no more than **0.5 dB** in either evaluation protocol.
- Cross-speaker secret SI-SDR falls by no more than **0.25 dB**.
- Clipped stego fraction is at most **0.0001**, meaning **0.01%**, in both protocols.
- Predicted-secret SNR loses no more than **0.5 dB** after simulated PCM16 rounding.
- Oracle secret SNR stays at least **60 dB** as a numerical-health gate.
- The global worst-case float32 cycle error stays at most **0.005**; a healthy average
  oracle score must not conceal a badly conditioned outlier.
- Metrics are finite and training has not exceeded the AMP skip guard.

These are screening rules, not definitions of transparency/security or statistical proof.
If none qualifies, the suite ends successfully as `completed_screen_no_promotion`. It does
not blindly spend the remaining hours on a failed hypothesis.
This is budgeted triage: a slower-learning architecture could improve later and still fail
this screen. A negative result does not prove that every U-Net or residual-supervision
configuration is ineffective.

### Phase 3 — A second matched seed

Train **both A and the shortlisted challenger**, fresh at seed 7391, to the same screen
step budget. Score full cross-speaker validation and require the same promotion gates.
If the gain does not pass this check, stop as `completed_screen_not_replicated`.

This checks direction/size at the screen budget. It does not give two independent seeds
for the later long-refinement result, and it is not a significance test.

### Phase 4 — Equal-step refinement

Continue A and the confirmed challenger from their original seed-3407 screen checkpoints.
Allocate a common additional step count from the remaining measured budget. Apply `3e-5`
to both after loading their optimizers. Their cumulative final step targets are identical,
with a **150k cumulative-step cap per finalist**.

This answers whether the new architecture/auxiliary remains helpful when the control
receives the same continuation opportunity. It is not “train only the winner longer.”

### Phase 5 — Frozen final evaluation and report

Once training targets are fixed and completed, score both finalists on all validation and
test records, with both pairing protocols, PCM16 diagnostics and five test examples per
protocol. Test measurements are never inputs to promotion or refinement allocation.

The historical test set has already been inspected in earlier work. This avoids new
within-suite test-driven selection, but does not turn that dataset into a brand-new unseen
benchmark. Broader independent data remains future work.

## 4. How the 24–30-hour bound works

The default is a **28-hour suite-wide wall-clock deadline**, configurable only between
24 and 30 hours. It covers manifest indexing, profiling, training, monitoring, checkpoint
saves, offline evaluation and orchestration. Package installation and downloading/copying
the corpus must happen **before** the timed suite.

If the entire machine reservation includes fresh provisioning, prefer a **24-hour suite**
and prestage the dataset, leaving time for setup within a 30-hour reservation. Download
duration cannot be guaranteed without knowing network speed; no script can promise that
fetching roughly 31 GB over an unknown connection will fit a fixed setup window.

### Allocation, rather than a guessed 4090 speed

Let `c_A ... c_D` be measured seconds per attempted step. The common screen budget uses
up to 45% of the requested time and accounts for **all four screens plus two seed repeats**:

```text
S ≈ round_down_to_1000(
      0.75 * screen_and_repeat_seconds
      / (c_A + c_B + c_C + c_D + c_A + max(c_B, c_C, c_D))
    )
```

`S` is capped at 50k and must reach the 20k minimum. The extra `c_A + max(...)` reserves
both paired replicas, not just the challenger. Startup/final validation are already present
in profiling time. The 0.75 factor adds a conservative throughput margin.

After actual screening and replication time is known, the common refinement count is
computed from the **sum** of the two finalists' per-step costs, with a **three-hour offline
evaluation reserve** and extra orchestration margin. Unused early-phase time can therefore
be available to refinement; the script does not assume that all architectures run equally fast.

### Enforcement and caveats

- Each child process is supervised with a timeout tied to the remaining global deadline.
- Each training child also receives its own smaller internal `--max-hours` guard.
- A first stop signal asks the trainer to finish a step and save atomically. If it cannot
  finish within the supervisor's grace window, the job is terminated and older complete
  checkpoints are retained. A fresh checkpoint is not guaranteed after a forced kill.
- A failed job, missing artifact, unequal achieved step target or invalid numerical result
  is recorded as failed/incomplete. The suite does not publish it as a completed comparison.
- The original deadline is persisted. `--resume-suite` does **not** award another 28 hours.
- The script can finish before 24 hours because of failed promotion, step caps or conservative
  reserves. The hours are a ceiling, not a requirement to waste compute.

This is a runtime bound under normal OS/process control, not a guarantee against power
loss, an unresponsive device/filesystem, insufficient hardware throughput, or a promise that
every candidate converges. The 28-hour default leaves margin below a 30-hour booking.

## 5. Prepare a fresh machine before starting the timer

Recommended: one exclusive RTX 4090 24 GB, Python 3.11, 32 GB RAM or more, local SSD/NVMe,
and approximately 70–90 GB free for environment, dataset, outputs and headroom. Keep other
GPU jobs off the machine. Do not run from a slow network audio share.

The runner does not need `uv`, `aria2c`, Bash, the mounted history file, or old checkpoints.
The commands below use standard Python/pip. FFmpeg and FFprobe are required by the existing
environment checker; install them through the machine's normal OS setup first.

### Linux setup

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --timeout 600 \
  nvidia-cusparselt-cu12==0.7.1 --index-url https://pypi.org/simple
python -m pip install --no-cache-dir --timeout 600 \
  'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[dev,metrics]'
python scripts/check_env.py --min-free-gb 40
python -m pytest
python -m ruff check .
```

This pins the PyTorch/CUDA wheel used for the local software tests. A compatible NVIDIA
driver is required; a separate CUDA toolkit is not. The explicit PyPI NVIDIA dependency
step avoids the download endpoint that previously timed out.

### Windows setup

Use 64-bit Python 3.11 and a compatible NVIDIA driver. Run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 600 nvidia-cusparselt-cu12==0.7.1 --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 600 torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e '.[dev,metrics]'
.\.venv\Scripts\python.exe scripts\check_env.py --min-free-gb 40
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

The orchestration uses platform-specific process groups and OS locks, but Windows behavior
has not been executed on this Linux development host. Linux is the locally tested path.
Do not invoke `train_windows.ps1` for this matrix; it is a separate older workflow.

### Dataset: reuse or download

Preferred: copy the already verified complete LibriSpeech folders onto the new machine.
The required layout is:

```text
<audio-root>/
  train-clean-100/
  train-clean-360/
  dev-clean/
  test-clean/
```

`<audio-root>` means the **LibriSpeech directory itself**, not its parent `datasets` folder.
The suite reindexes absolute paths locally and checks official file counts:
28,539 / 104,014 / 2,703 / 2,620 respectively. It rejects incomplete/duplicated splits.
This is not a full re-check of downloaded archive MD5 or every audio sample after a copy;
keep the original verified data and copy it reliably.

If download is necessary, run this before the timed suite:

```bash
python scripts/prepare_librispeech.py \
  --data-dir datasets/librispeech --download-only
```

The helper keeps no compressed archives. On Windows use the environment's `python.exe`
and the same script arguments on one line. If an interrupted `.partial-*` directory remains,
inspect/move aside that one partial directory before retrying; do not delete verified data.

The default study keeps our earlier 100+360 corpus: approximately 464.2 hours. To use only
the external history's `train-clean-360` corpus, consistently add
`--train-splits train-clean-360` to download and suite commands. All new arms will then use
that corpus, and historical comparisons must acknowledge the different data budget.

## 6. One command runs the experiment set sequentially

### Preview: no experiment directory, GPU work, or training

```bash
.venv/bin/python scripts/run_4090_suite.py --plan --budget-hours 28
```

### Linux: run the whole study

Keep the terminal/session alive and disable sleep. Do not create the output directory in
advance: a new suite refuses to overwrite an existing directory.

```bash
.venv/bin/python -u scripts/run_4090_suite.py \
  --data-root datasets/librispeech/LibriSpeech \
  --output runs/4090_suite_v1 \
  --budget-hours 28 \
  --workers 4
```

Replace `--data-root` with the SSD path if needed. Use a persistent terminal, `tmux`, or the
machine's normal job manager—not an OpenCode/chat-managed background command.

### Windows: the same sequential runner

```powershell
.\.venv\Scripts\python.exe -u scripts\run_4090_suite.py --data-root "D:\datasets\LibriSpeech" --output runs\4090_suite_v1 --budget-hours 28 --workers 4
```

Choose `--budget-hours 24` or `30` before a fresh run if that better fits the reservation.
Do not edit the budget afterward to make an old run appear to have stayed under its cap.

### Monitoring

The console prints which child job is running and its log path. Every child's stdout and
stderr go into a distinct file under `logs/`, including the full Python traceback if it fails.
Small clipping fractions print with useful precision instead of misleading `0.0000` rounding.

In a second activated terminal:

```bash
tensorboard --logdir runs/4090_suite_v1/training --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006`. `REPORT.md` is refreshed after jobs, not on every training step.
Do not launch the training commands manually in parallel with the suite.

## 7. Pause, error handling, and resume

Press Ctrl+C once and allow the supervisor to stop the current child. Avoid repeated
interrupts unless a forced stop is necessary. Check the last completed checkpoint afterward.

Resume with the same output, data root, budget, workers and requested micro-batch:

```bash
.venv/bin/python -u scripts/run_4090_suite.py \
  --data-root datasets/librispeech/LibriSpeech \
  --output runs/4090_suite_v1 \
  --budget-hours 28 \
  --workers 4 \
  --resume-suite
```

On Windows add `--resume-suite` to the matching PowerShell invocation.

- Completed jobs are reused, not trained again.
- A partial training phase resumes its latest atomic checkpoint if available.
- The runner refuses a duplicate when the recorded child PID is still alive. It does not
  automatically kill an unknown/live process. Check that process/log before retrying.
- OS locking prevents two suite controllers from writing the same study simultaneously.
- Source/study changes, a different GPU/PyTorch environment, changed data root/manifest,
  or reset budget are refused. Do not update the code halfway through a comparison.
- A restart does not reproduce sampler/worker state bit-for-bit. Stage boundaries restart
  the independently seeded loaders consistently; unplanned interruption histories must be
  disclosed if they differ between arms. Complete replay is not claimed.
- A profile checkpoint without a reliable timing summary needs inspection, not an invented
  throughput estimate. An interrupted manifest leaves a `.tmp` for inspection.
- If the original deadline has passed, the suite stops rather than grants more GPU time.
  Existing reports/checkpoints remain available even if the complete plan did not finish.

There is no silent “exit code 1, carry on” behavior. Inspect the named raw log and
`stop_reason` in the state/report. Resolve the actual error before deciding on a new run.

## 8. Outputs and success criteria

Everything new stays under the chosen suite directory:

```text
runs/4090_suite_v1/
  suite_state.json       # original deadline, hashes, commands, attempts, statuses, results
  REPORT.md              # readable result table and promotion decisions
  .suite.lock
  manifest.jsonl         # local-machine paths, not the old machine's manifest
  manifest.summary.json
  configs/               # generated, explicit per-phase configurations
  logs/                  # separate attempt logs, including errors
  training/              # profiles, screens, repeats, matched refinements
  evaluation/            # validation/test JSON and final listening examples
```

The normal completed path has four screen runs, two second-seed runs and two matched
continuations, in addition to four profiles. Extra profiling can occur after an OOM fallback.
There are full-split metrics in both pairing protocols, plus explicitly named per-session
AMP skip counts, target-step completion, runtime and peak allocated CUDA memory summaries.

Primary success is **better predicted-secret SNR/SI-SDR at comparable cover quality on
cross-speaker validation**, repeated at the paired second seed and then checked in the
frozen final test. Check PCM16, clipping, numerical conditioning and listening samples too.

Do not select by mixed total loss: D has an additional objective term, and the oracle is
not blind recovery. Do not label a failed/incomplete or early negative screen a completed
30-hour performance experiment. A two-seed screen is useful evidence, not a final multi-seed
benchmark or a proof of statistical significance.

If the U-Net and auxiliary fail these tests, the next research question should be a more
predictable/lower-rate secret or latent representation—not another unexamined width increase
or an oracle-assisted score. A structural payload change needs its own future protocol.

## 9. Preparing for the later training-only ZIP

No ZIP is created by this task. The runner intentionally depends only on portable source
and destination-machine paths. A later bundle should include:

- `pyproject.toml` and the README it references.
- `src/spectra_inn/`, including the new `experiments` package.
- `configs/4090_suite.yaml` and its base `configs/rpm_3060_nmse_10k.yaml`.
- `scripts/run_4090_suite.py`, `scripts/prepare_librispeech.py`, and `scripts/check_env.py`.
- Tests and these setup/run instructions, so the fresh machine can be checked first.
- A code/dependency version record and the relevant history summary/hash in this document.

Exclude `.venv`, old `runs`, `.codegraph`, caches, downloaded audio/archives, old checkpoints,
and old absolute-path manifests. Copy/download the corpus separately and let the runner
build its local manifest. The external mounted Markdown file is not needed at runtime.
The new runner/configs/tests are currently working-tree additions: a plain archive of an
older Git commit would omit them. The later ZIP must contain the actual reviewed source.
