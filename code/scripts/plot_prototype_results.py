import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def metric_scalar(value):
    if isinstance(value, list):
        if not value:
            return float('nan')
        return float(np.mean(value))
    return float(value)


def collect_val_points(metrics_dir):
    points = []
    for path in sorted(metrics_dir.glob('val_iter_*_summary.json')):
        with open(path, 'r') as f:
            data = json.load(f)
        points.append({
            'iter': int(data['iter']),
            'epoch': int(data['epoch']),
            'psnr_cover': metric_scalar(data['psnr_cover']),
            'psnr_secret': metric_scalar(data['psnr_secret']),
            'psnr_stego': metric_scalar(data['psnr_stego']),
            'ssim_cover': metric_scalar(data['ssim_cover']),
            'ssim_secret': metric_scalar(data['ssim_secret']),
            'ssim_stego': metric_scalar(data['ssim_stego']),
            'path': str(path),
        })
    points.sort(key=lambda x: x['iter'])
    return points


def collect_test_summaries(results_dir):
    items = []
    for path in sorted(results_dir.glob('metrics_test_summary*.json')):
        with open(path, 'r') as f:
            data = json.load(f)
        items.append({
            'path': str(path),
            'checkpoint': data.get('checkpoint', ''),
            'num_samples': int(data.get('num_samples', 0)),
            'psnr_cover': metric_scalar(data['psnr_cover']),
            'psnr_secret': metric_scalar(data['psnr_secret']),
            'psnr_stego': metric_scalar(data['psnr_stego']),
            'ssim_cover': metric_scalar(data['ssim_cover']),
            'ssim_secret': metric_scalar(data['ssim_secret']),
            'ssim_stego': metric_scalar(data['ssim_stego']),
        })
    return items


def best_point(points, key):
    return max(points, key=lambda x: x[key])


def save_plot(points, test_summaries, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required for plotting. Install it with: pip install matplotlib'
        ) from exc

    iters = [p['iter'] for p in points]
    psnr_cover = [p['psnr_cover'] for p in points]
    psnr_secret = [p['psnr_secret'] for p in points]
    psnr_stego = [p['psnr_stego'] for p in points]

    ssim_cover = [p['ssim_cover'] for p in points]
    ssim_secret = [p['ssim_secret'] for p in points]
    ssim_stego = [p['ssim_stego'] for p in points]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(iters, psnr_cover, label='Cover PSNR', linewidth=2)
    axes[0].plot(iters, psnr_secret, label='Secret PSNR', linewidth=2)
    axes[0].plot(iters, psnr_stego, label='Stego PSNR', linewidth=2)
    axes[0].set_title('Validation PSNR vs Iteration')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='best')

    axes[1].plot(iters, ssim_cover, label='Cover SSIM', linewidth=2)
    axes[1].plot(iters, ssim_secret, label='Secret SSIM', linewidth=2)
    axes[1].plot(iters, ssim_stego, label='Stego SSIM', linewidth=2)
    axes[1].set_title('Validation SSIM vs Iteration')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('SSIM')
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='best')

    if test_summaries:
        latest_test = test_summaries[-1]
        axes[0].axhline(latest_test['psnr_cover'], linestyle='--', linewidth=1.2,
                        color='tab:blue', alpha=0.5)
        axes[0].axhline(latest_test['psnr_secret'], linestyle='--', linewidth=1.2,
                        color='tab:orange', alpha=0.5)
        axes[0].axhline(latest_test['psnr_stego'], linestyle='--', linewidth=1.2,
                        color='tab:green', alpha=0.5)
        axes[1].axhline(latest_test['ssim_cover'], linestyle='--', linewidth=1.2,
                        color='tab:blue', alpha=0.5)
        axes[1].axhline(latest_test['ssim_secret'], linestyle='--', linewidth=1.2,
                        color='tab:orange', alpha=0.5)
        axes[1].axhline(latest_test['ssim_stego'], linestyle='--', linewidth=1.2,
                        color='tab:green', alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def add_label(image, label):
    h, w = image.shape[:2]
    label_h = 30
    canvas = np.full((h + label_h, w, 3), 255, dtype=np.uint8)
    canvas[label_h:, :, :] = image
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def build_sample_panel(results_dir, output_dir, key, secret_idx=0):
    cover_gt = results_dir / f'{key}_GT.png'
    stego = results_dir / f'{key}_LR.png'
    cover_rec = results_dir / f'{key}_SR.png'
    secret_gt = results_dir / f'{key}_{secret_idx}_LRGT.png'
    secret_rec = results_dir / f'{key}_{secret_idx}_SR_h.png'

    required = [cover_gt, stego, cover_rec, secret_gt, secret_rec]
    if any(not p.exists() for p in required):
        return None

    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in required]
    if any(img is None for img in imgs):
        return None

    labels = ['Cover GT', 'Stego (Container)', 'Recovered Cover', 'Secret GT', 'Recovered Secret']
    tiles = [add_label(img, label) for img, label in zip(imgs, labels)]
    panel = np.concatenate(tiles, axis=1)

    out_path = output_dir / f'sample_{key}.png'
    cv2.imwrite(str(out_path), panel)
    return out_path


def create_visual_panels(results_dir, output_dir, max_samples=12, secret_idx=0):
    gt_paths = sorted(results_dir.glob('*_GT.png'))
    keys = []
    for path in gt_paths:
        stem = path.stem  # e.g. 10_0_GT
        if not stem.endswith('_GT'):
            continue
        key = stem[:-3]
        keys.append(key)

    panel_paths = []
    for key in keys[:max_samples]:
        panel_path = build_sample_panel(results_dir, output_dir, key, secret_idx=secret_idx)
        if panel_path is not None:
            panel_paths.append(panel_path)

    return panel_paths


def save_report(points, test_summaries, output_path):
    report = {
        'num_validation_points': len(points),
        'latest_validation': points[-1] if points else None,
        'best_validation': {
            'cover_psnr': best_point(points, 'psnr_cover') if points else None,
            'secret_psnr': best_point(points, 'psnr_secret') if points else None,
            'stego_psnr': best_point(points, 'psnr_stego') if points else None,
            'cover_ssim': best_point(points, 'ssim_cover') if points else None,
            'secret_ssim': best_point(points, 'ssim_secret') if points else None,
            'stego_ssim': best_point(points, 'ssim_stego') if points else None,
        },
        'test_summaries': test_summaries,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    code_root = Path(__file__).resolve().parents[1]
    repo_root = code_root.parent
    default_experiment_root = repo_root / 'experiments' / 'train_LF-VSN_1video_proto'
    default_results_root = code_root / 'results' / 'train_LF-VSN_1video_proto'
    default_output_dir = default_results_root / 'analysis'

    parser = argparse.ArgumentParser(description='Plot LF-VSN prototype metrics and build sample panels.')
    parser.add_argument('--experiment-root',
                        default=str(default_experiment_root),
                        help='Experiment root containing metrics/ from training.')
    parser.add_argument('--results-root',
                        default=str(default_results_root),
                        help='Results directory containing test outputs and metrics_test_summary.json.')
    parser.add_argument('--output-dir',
                        default=str(default_output_dir),
                        help='Directory to save plots and comparison panels.')
    parser.add_argument('--max-samples', type=int, default=12,
                        help='Number of sample comparison panels to save.')
    parser.add_argument('--secret-idx', type=int, default=0,
                        help='Secret index to visualize (for multi-secret runs).')
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = experiment_root / 'metrics'
    if not metrics_dir.exists():
        raise FileNotFoundError('Metrics directory not found: {}'.format(metrics_dir))
    if not results_root.exists():
        raise FileNotFoundError('Results directory not found: {}'.format(results_root))

    val_points = collect_val_points(metrics_dir)
    if not val_points:
        raise ValueError('No validation summaries found in {}'.format(metrics_dir))
    test_summaries = collect_test_summaries(results_root)

    plot_path = output_dir / 'prototype_val_curves.png'
    plot_saved = False
    try:
        save_plot(val_points, test_summaries, plot_path)
        plot_saved = True
    except RuntimeError as err:
        print('Warning: {}'.format(err))

    panel_paths = create_visual_panels(results_root, output_dir,
                                       max_samples=args.max_samples,
                                       secret_idx=args.secret_idx)

    report_path = output_dir / 'prototype_report.json'
    save_report(val_points, test_summaries, report_path)

    if plot_saved:
        print('Saved metric curves to {}'.format(plot_path))
    else:
        print('Metric curves were not generated (install matplotlib to enable plotting).')
    print('Saved {} sample panels to {}'.format(len(panel_paths), output_dir))
    print('Saved report to {}'.format(report_path))
    if panel_paths:
        print('Example panel: {}'.format(panel_paths[0]))


if __name__ == '__main__':
    main()
