# LF-VSN_V2: Paper-Aligned Training Setup

This folder is an isolated V2 workspace for LF-VSN, prepared for full training with Vimeo-90K and a 4090.

## What was improved in V2 (paper-driven)

The following updates were implemented based on the LF-VSN paper and its ablation/implementation details:

1. **Scalable embedding behavior**
   - Added dynamic channel truncation in `DenseBlock_v2` (`code/models/modules/Subnet_constructor.py`) so aggregation can run with variable active secret counts.
2. **Scalable secret-count training support**
   - Added optional `scalable_training` with configurable active secret candidates, random subset selection, and active-count-aware reverse pass (`code/models/LFVSN.py`, `code/models/modules/Inv_arch.py`).
3. **Temporal window usage improvement**
   - Added configurable train-time frame offsets (`train_frame_offsets`) so 7-frame sequences can sample multiple triplet windows.
4. **Training preprocessing improvements**
   - Added synchronized augmentations across cover+secret videos (flip/rotation) in training dataloader (`code/data/Vimeo90K_dataset.py`).
   - Added min-frame checks in train/val datasets.
5. **Performance and stability for modern GPUs**
   - Added AMP mixed precision support and scaler state resume (`code/models/LFVSN.py`, `code/models/base_model.py`).
   - Added optional TF32 enablement for Ada GPUs (4090) in train/test scripts.
   - Improved DataLoader settings support (`pin_memory`, `persistent_workers`, `prefetch_factor`) in `code/data/__init__.py`.
6. **Safer model loading**
   - Updated checkpoint loading to use `weights_only=True` when supported.

## Which Vimeo-90K subset to download?

For **real LF-VSN full training**, use **Vimeo-90K Septuplet**.

- ✅ Use: **septuplet** (`vimeo_septuplet.zip`, 7-frame sequences, 448x256)
- ❌ Do not use triplet for full LF-VSN training here

Reason:
- The paper implementation details and this training setup rely on 7-frame sequences and 3-frame sliding windows within them.

## Fresh Windows prerequisites

On a brand-new Windows machine, install these first:
- NVIDIA Game Ready/Studio driver for RTX 4090.
- Git for Windows.
- Python 3.10+ (64-bit) with PATH enabled.

Then open PowerShell in `LF-VSN_V2\\code` and run the setup commands below.

## Windows 4090 environment setup (recommended)

From PowerShell in `LF-VSN_V2\\code`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4090_env.ps1
```

This creates `.venv`, installs CUDA PyTorch, and installs training dependencies from:
- `code/requirements/windows-train.txt`

Optional plotting/demo dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4090_env.ps1 -InstallOptional
```

## One-command dataset setup

### Windows (PowerShell)

From `LF-VSN_V2\\code`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_vimeo90k_septuplet_windows.ps1 -PythonExe .\.venv\Scripts\python.exe
```

Custom destination and val pair count:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_vimeo90k_septuplet_windows.ps1 -VimeoRoot D:\datasets\vimeo90k -NumValPairs 200 -PythonExe .\.venv\Scripts\python.exe
```

### Linux/macOS

From `LF-VSN_V2/code`:

```bash
chmod +x scripts/setup_vimeo90k_septuplet.sh
./scripts/setup_vimeo90k_septuplet.sh
```

What this does:
- Downloads `vimeo_septuplet.zip` from TOFlow dataset server.
- Extracts `sequences` to `code/vimeo90k/sequences`.
- Builds LF-VSN-compatible list files:
  - `code/vimeo90k/sep_trainlist.txt`
  - `code/vimeo90k/sep_testlist.txt` (for 1-secret eval)
  - `code/vimeo90k/sep_testlist_7video.txt` (for 7-secret eval)

## One-command training run on Windows

From PowerShell in `LF-VSN_V2\\code`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1
```

What this runner does:
- Creates/repairs environment if needed.
- Prepares Vimeo-90K septuplet if missing.
- Auto-resumes from latest `experiments/<name>/training_state/*.state`.
- Starts training with generated runtime config under `options/train/runtime_generated/`.

If you want a stricter 2-day budget cap (conservative niter):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1 -Strict2Day
```

If you want to train from scratch (ignore resume state):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1 -NoResume
```

## One-command evaluation run on Windows

From PowerShell in `LF-VSN_V2\\code`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_windows.ps1
```

What this runner does:
- Creates/repairs environment if needed.
- Prepares Vimeo-90K septuplet if missing.
- Auto-selects checkpoint in `experiments/<name>/models/`:
  - `latest_G.pth` if available, otherwise highest-iteration `*_G.pth`.
- Runs `test.py` with the selected checkpoint.

Evaluate the scalable 7-secret config:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_windows.ps1 -Config options/train/train_LF-VSN_v2_4090_2day_scalable7.yml
```

## Training configs provided

In `code/options/train/`:

- `train_LF-VSN_v2_fullpaper_1video.yml`
  - Full paper-style schedule (`250k` iterations).
- `train_LF-VSN_v2_4090_2day_1video.yml`
  - Practical 4090 schedule for ~2-day budget (`180k` iterations).
- `train_LF-VSN_v2_4090_2day_scalable7.yml`
  - Scalable training for `1..7` secret videos in a single run.

## Paper setup and reported training time

From Sec. 4.1 (Implementation Details) of the LF-VSN paper:
- Dataset: Vimeo-90K training split, 448x256 sequences.
- Train crop/aug: random crop `144x144`, horizontal + vertical flip.
- Optimizer: Adam, `beta1=0.9`, `beta2=0.5`.
- Batch size: `16`.
- Weight decay: `1e-12`.
- Initial LR: `1e-4`, halved every `30K` iterations.
- Total iterations: `250K`.
- Reported wall time: about `3 days` on one NVIDIA Tesla V100 GPU.

## Recommended command for your 2-day 4090 run

From PowerShell in `LF-VSN_V2\\code`:

```powershell
.\.venv\Scripts\python.exe train.py -opt options/train/train_LF-VSN_v2_4090_2day_1video.yml
```

Then test latest checkpoint:

```powershell
.\.venv\Scripts\python.exe test.py -opt options/train/train_LF-VSN_v2_4090_2day_1video.yml
```

## Optional: scalable 1..7-secret training

```powershell
.\.venv\Scripts\python.exe train.py -opt options/train/train_LF-VSN_v2_4090_2day_scalable7.yml
```

Evaluate that model on 7-secret setting:

```powershell
.\.venv\Scripts\python.exe test.py -opt options/train/train_LF-VSN_v2_4090_2day_scalable7.yml
```

## Notes

- This V2 setup is isolated from your original `LF-VSN-main` folder.
- If throughput is lower than expected, reduce `batch_size` first.
- If memory allows, keep AMP + TF32 enabled for best 4090 throughput.
