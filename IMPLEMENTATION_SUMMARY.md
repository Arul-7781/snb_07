# LF-VSN Prototype Implementation Summary

This document summarizes exactly what was implemented for your LF-VSN prototype, how it differs from the original repository behavior, how variable-length videos are handled, and which dataset/preprocessing pipeline was used.

---

## 1) What You Implemented

You implemented a **single-secret LF-VSN prototype system** with:

- A custom **1-secret training configuration** for Vimeo-90K 3-frame clips.
- A **subset generation pipeline** so training can run on RTX 3060 in a few hours.
- Extended **metrics logging/export** (PSNR + SSIM, per-frame and summary).
- A **results analysis script** that plots curves and builds comparison panels.
- A full **web demo app** for encode/decode with robust video preprocessing.

In short: this is no longer just the base LF-VSN training repo; it is now an end-to-end prototype workflow (train -> test -> analyze -> interactive demo).

---

## 2) All Changes vs Original LF-VSN Repo

### A. Prototype configuration and training behavior

1. **New prototype option file**
   - Added `code/options/train/train_LF-VSN_1video_proto.yml`.
   - Key differences from standard configs:
     - `num_video: 1` (single secret)
     - `gop: 3`
     - `single_secret_only: true`
     - `test_frame_offsets: [0]`
     - `niter: 20000` (prototype length)
     - `batch_size: 2`
     - `max_items: 200` for val set
     - `clamp: 0.9` in network settings

2. **Single-secret safety check in model**
   - File: `code/models/LFVSN.py`
   - Added a guard so config mismatch is caught early:
     - if `single_secret_only=true` and `num_video != 1`, it raises an error.

3. **Robust test offset logic**
   - File: `code/models/LFVSN.py`
   - Replaced rigid `[-1,0,1]` assumptions with `_get_test_offsets(...)` so short triplets are handled safely.

4. **Configurable invertible block clamp**
   - File: `code/models/modules/Inv_arch.py`
   - Added configurable `clamp` plumbing into `InvNN`/`InvBlock`.
   - Prototype uses `clamp: 0.9`.

### B. Dataset/data-loader robustness

5. **Training dataset controls and determinism improvements**
   - File: `code/data/Vimeo90K_dataset.py`
   - Added optional `max_samples` limiter.
   - Sorted frame filenames for deterministic frame order.
   - Added check to avoid selecting identical cover/secret sequence index.
   - Added minimum-size sanity check for list length vs `num_video`.

6. **Validation/test list integrity checks**
   - File: `code/data/video_test_dataset.py`
   - Added optional `max_items` for fast val/test.
   - Enforces list divisibility by `(num_video + 1)`.
   - Raises explicit error on empty split.

### C. Metrics and experiment artifacts

7. **SSIM + persistent metric export during training validation**
   - File: `code/train.py`
   - Added SSIM computation alongside PSNR.
   - Exports per-validation artifacts:
     - `experiments/<exp>/metrics/val_iter_XXXXXXXX.csv`
     - `experiments/<exp>/metrics/val_iter_XXXXXXXX_summary.json`
   - Added TensorBoard SSIM scalar logging.

8. **SSIM + persistent metric export during test**
   - File: `code/test.py`
   - Added SSIM computation and test metric exports:
     - `code/results/<exp>/metrics_test.csv`
     - `code/results/<exp>/metrics_test_summary.json`

### D. New scripts for prototype workflow

9. **Subset preparation script (new)**
   - File: `code/scripts/prepare_vimeo90k_subset.py`
   - Can build prototype lists from:
     - train+test lists
     - test-only list
     - train-only list
     - or auto-scan split
   - Added fallback for your case where local data overlapped entirely with test list.

10. **Results plotting + panel script (new)**
    - File: `code/scripts/plot_prototype_results.py`
    - Aggregates validation JSON summaries.
    - Plots PSNR/SSIM curves.
    - Creates side-by-side visual panels.
    - Writes `prototype_report.json`.

11. **Interactive demo app (new)**
    - File: `code/scripts/lfvsn_demo_app.py`
    - Gradio app with:
      - Encode + Decode tab
      - Decode Existing Stego tab
      - robust upload parsing
      - preprocessing for arbitrary videos
      - duration-safe output FPS
      - decode-safe stego export
      - size presets (High Fidelity / Balanced / Compact)

### E. Compatibility/runtime fixes

12. **Removed fragile BasicSR import dependency in model modules**
    - Files:
      - `code/models/modules/Subnet_constructor.py`
      - `code/models/modules/Inv_arch.py`
    - This avoids the `torchvision.transforms.functional_tensor` mismatch issue from newer torchvision environments.

13. **Device-safe IWT tensor allocation**
    - File: `code/models/modules/common.py`
    - Changed hardcoded CUDA allocation to device-aware allocation:
      - from `.cuda()` style to `x.new_zeros(...)`

14. **Gradio API compatibility fix**
    - File: `code/scripts/lfvsn_demo_app.py`
    - Queue launch handles both old/new Gradio parameter names.

---

## 3) How Different Cover/Secret Lengths Are Handled

There are two contexts:

### A. Training/evaluation on Vimeo-90K subset

- Data is sequence-based (`im1/im2/im3`), so each sample is a fixed triplet from dataset folders.
- Cover and secret are selected as sequence IDs from list files and are naturally aligned by sequence format.

### B. Demo app uploads (arbitrary user videos)

In `code/scripts/lfvsn_demo_app.py`, the app does:

1. Read both videos into frame arrays.
2. Compute `paired = min(len(cover_frames), len(secret_frames))`.
3. Use only the paired timeline; extra frames from longer video are ignored.
4. Optionally subsample uniformly if max frame cap is active.
5. Resize both streams to the same target size (encode path).
6. For each output frame index, build 3-frame temporal context via reflection padding (`reflect_index`, `gather_triplet`) so boundaries are valid.

Result: variable-length and boundary cases are handled deterministically and safely.

---

## 4) Dataset Used and Preprocessing Performed

## Dataset used

- Base dataset: **Vimeo-90K sequences** (triplets).
- Data root used by config/scripts:
  - `code/vimeo90k/sequences`

- Prototype lists generated:
  - `code/vimeo90k/sep_trainlist_proto.txt` -> **3000** training sequences
  - `code/vimeo90k/sep_testlist_proto.txt` -> **400** entries total
    - interpreted as 200 cover + 200 secret entries (for `num_video=1`)

## Preprocessing used for training pipeline

From `code/data/Vimeo90K_dataset.py` and prototype YAML:

- Frame order is sorted by filename.
- Random crop to `GT_size=128`.
- Same crop coordinates are applied to cover and secret sequences.
- BGR -> RGB conversion.
- NHWC -> NCHW tensor conversion.
- Data augmentation from config:
  - flip enabled
  - rotation enabled

## Preprocessing used in app pipeline (inference)

- Reads uploaded videos with OpenCV.
- If OpenCV fails, tries ffmpeg transcode fallback for readability.
- Converts grayscale/BGRA to RGB.
- Aligns cover/secret lengths via minimum common length.
- Optional frame subsampling (`Max Frames`, plus size presets).
- Enforces DWT-safe even dimensions.
- For decode-only path, avoids resizing stego content (only minimal even crop if needed).

---

## 5) Important Operational Note (Decode Reliability)

The hidden signal is highly sensitive to compression.

- `stego_preview.mp4` is for viewing and may be lossy.
- `stego_lossless.mp4` is decode-safe and should be used for extraction.

In the app, this is exposed as:

- **Download Stego (lossless for decode)** -> use this for decode tab.

---

## 6) Current Prototype Outputs and Artifacts

- Training checkpoints:
  - `experiments/train_LF-VSN_1video_proto/models/*.pth`

- Validation metrics over time:
  - `experiments/train_LF-VSN_1video_proto/metrics/val_iter_*.csv`
  - `experiments/train_LF-VSN_1video_proto/metrics/val_iter_*_summary.json`

- Test outputs:
  - `code/results/train_LF-VSN_1video_proto/*.png`
  - `code/results/train_LF-VSN_1video_proto/metrics_test.csv`
  - `code/results/train_LF-VSN_1video_proto/metrics_test_summary.json`

- Analysis outputs:
  - `code/results/train_LF-VSN_1video_proto/analysis/prototype_report.json`
  - `code/results/train_LF-VSN_1video_proto/analysis/sample_*.png`
  - `code/results/train_LF-VSN_1video_proto/analysis/prototype_val_curves.png` (when matplotlib installed)

- Demo app run artifacts:
  - `code/results/app_demo_runs/<run_id>/...`

---

## 7) One-Line Summary

You now have a **single-secret, RTX-3060-friendly LF-VSN prototype stack** with custom training subset logic, expanded metric tracking, robust test/analysis tooling, and an interactive encode/decode app that handles real-world video input preprocessing and decode-safe output paths.

---

## 8) Training Results Summary (Prototype Run)

This section summarizes the metrics from the completed prototype training run.

### Run setup used

- Config: `code/options/train/train_LF-VSN_1video_proto.yml`
- Train subset: `3000` sequences (`code/vimeo90k/sep_trainlist_proto.txt`)
- Validation protocol: `200` cover samples + `1` secret each (`400` entries in `code/vimeo90k/sep_testlist_proto.txt`)
- Total training budget: `20000` iterations
- Validation frequency: every `500` iterations -> `40` validation points
- Effective epochs: about `14` (3000 samples / batch size 2 -> ~1500 iters/epoch)

### Validation trend (first vs final)

- First validation @ `iter 500`:
  - PSNR Cover: `22.4059`
  - PSNR Secret: `21.9767`
  - PSNR Stego: `22.0815`
  - SSIM Cover: `0.6564`
  - SSIM Secret: `0.6338`
  - SSIM Stego: `0.5771`

- Final validation @ `iter 20000`:
  - PSNR Cover: `22.6180` (`+0.2120`)
  - PSNR Secret: `22.5862` (`+0.6095`)
  - PSNR Stego: `22.5353` (`+0.4539`)
  - SSIM Cover: `0.7105` (`+0.0542`)
  - SSIM Secret: `0.7078` (`+0.0739`)
  - SSIM Stego: `0.6835` (`+0.1064`)

### Best checkpoints observed on validation

- Best Cover PSNR: `iter 16000` -> PSNR `22.6225`, SSIM `0.7091`
- Best Secret PSNR: `iter 17000` -> PSNR `22.5981`, SSIM `0.7075`
- Best Stego PSNR: `iter 19000` -> PSNR `22.5418`, SSIM `0.6857`
- Practical balanced pick: around `iter 19000` (best stego and strong overall tradeoff)

### Test run summary

- Latest test summary file: `code/results/train_LF-VSN_1video_proto/metrics_test_summary.json`
- Checkpoint used in that test: `.../models/19000_G.pth`
- Test sample count: `200`
- Test metrics:
  - PSNR Cover: `22.6197`
  - PSNR Secret: `22.5737`
  - PSNR Stego: `22.5418`
  - SSIM Cover: `0.7103`
  - SSIM Secret: `0.7069`
  - SSIM Stego: `0.6857`

### Interpretation for prototype quality

- Training is stable and convergent for this lightweight run.
- Metrics improve from early to late iterations and plateau in the `16k-20k` region.
- Results are appropriate for a prototype subset run on RTX 3060; full-paper performance would require larger data and longer training.

---

## 9) Data Preprocessing (Current vs Full Implementation)

This section summarizes what preprocessing is active now in your prototype and what you should keep/extend for full-scale training.

### A. Current preprocessing in your prototype

#### 1) Dataset/list preparation

- Input root: `code/vimeo90k/sequences`
- List generation script: `code/scripts/prepare_vimeo90k_subset.py`
- Actions performed:
  - validates sequence folders and minimum frame count (`--min-frames 3`)
  - supports train+test list, test-only, train-only, or auto-scan splitting
  - for your case, split from test-only data into:
    - `code/vimeo90k/sep_trainlist_proto.txt` (3000 train)
    - `code/vimeo90k/sep_testlist_proto.txt` (200 cover + 200 secret entries)

#### 2) Training loader preprocessing (`code/data/Vimeo90K_dataset.py`)

- Loads frame PNGs in sorted filename order.
- Reads images as float32 in `[0,1]`, then BGR -> RGB.
- Randomly samples secret sequences, explicitly ensuring secret index != cover index.
- Applies one random crop shared across cover and secret (`GT_size=128` for prototype).
- Converts to tensor format `(T,C,H,W)` and stacks secret tensors.
- Supports optional `max_samples` truncation.

#### 3) Validation/test loader preprocessing (`code/data/video_test_dataset.py`)

- Sorts and validates sequence list structure.
- Enforces list divisibility by `(num_video + 1)`.
- Supports optional `max_items` truncation.
- Loads full sequence tensors from folder (`read_img_seq`), RGB format, normalized `[0,1]`.

#### 4) Model-level preprocessing/constraints

- Wavelet transforms (DWT/IWT) are inside model path.
- Spatial dimensions must be even.
- No globally fixed resolution is required, but cover and secret in a sample must match shape.

#### 5) Demo app preprocessing (`code/scripts/lfvsn_demo_app.py`)

- Robust video decode (OpenCV + ffmpeg fallback).
- Converts grayscale/BGRA to RGB.
- Aligns cover/secret by `min(length_cover, length_secret)`.
- Optional frame subsampling (`Max Frames`) and size presets.
- Encode path resizes both videos to a common target shape (cover-driven) and enforces even dimensions.
- Decode-only path avoids arbitrary resize (only minimal even crop if needed).
- Uses lossless stego export for reliable decode.

### B. What to do for full implementation

#### 1) Dataset scale

- Move from prototype subset to full Vimeo-90K train split/list.
- Keep deterministic sequence parsing and list integrity checks.

#### 2) Augmentation behavior

- Your YAML includes `use_flip` and `use_rot`, but current dataset code does not call augmentation utilities.
- For full training, integrate explicit augmentation calls in dataloader if you want those options active.

#### 3) Resolution strategy

- Keep random crop training (for memory and generalization).
- Consider larger crop size and longer schedule to approach paper-level quality.

#### 4) Aspect ratio handling in app

- Current encode path resizes secret to cover target shape (can warp AR when cover/secret AR differ).
- For production-like behavior, consider letterbox/pad-to-match instead of geometric warp.

#### 5) Stego transport policy

- Continue using lossless stego for decode-critical paths.
- If compressed stego must be used, expect major degradation in recovered secret quality.
