import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np
import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models import networks  # noqa: E402
from models.modules.Quantization import Quantization  # noqa: E402
from models.modules.common import DWT, IWT  # noqa: E402


def load_option_for_inference(opt_path):
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)
    opt['is_train'] = False
    opt['dist'] = False
    if opt.get('distortion') == 'sr':
        opt['network_G']['scale'] = opt['scale']
    if 'datasets' not in opt:
        opt['datasets'] = {}
    return opt


def strip_module_prefix(state_dict):
    clean = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            clean[key[7:]] = value
        else:
            clean[key] = value
    return clean


def load_checkpoint(ckpt_path, device):
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return strip_module_prefix(state_dict)


def ensure_rgb(frame_bgr):
    if frame_bgr is None:
        return None
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    if frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def resolve_video_path(video_input):
    if isinstance(video_input, (str, Path)):
        return str(video_input)
    if isinstance(video_input, dict):
        for key in ('path', 'video', 'name'):
            value = video_input.get(key)
            if isinstance(value, str) and value:
                return value
    raise ValueError('Unsupported video input type: {} ({})'.format(type(video_input), video_input))


def _read_video_frames_cv2(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 24.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = ensure_rgb(frame)
        if rgb is not None:
            frames.append(rgb)
    cap.release()

    if not frames:
        return [], 0.0
    return frames, fps


def _transcode_for_opencv(video_path):
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        return None
    out_dir = Path(tempfile.mkdtemp(prefix='lfvsn_ffmpeg_'))
    out_path = out_dir / 'transcoded.mp4'
    cmd = [
        ffmpeg_path,
        '-y',
        '-i',
        str(video_path),
        '-an',
        '-pix_fmt',
        'yuv420p',
        '-vcodec',
        'libx264',
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return str(out_path)


def read_video_frames(video_input):
    video_path = resolve_video_path(video_input)
    frames, fps = _read_video_frames_cv2(video_path)
    if frames:
        return frames, fps

    transcoded = _transcode_for_opencv(video_path)
    if transcoded is not None:
        frames, fps = _read_video_frames_cv2(transcoded)
        if frames:
            return frames, fps

    raise ValueError('No readable frames found in {}. Please use a standard MP4/H264 file.'.format(video_path))


def sample_indices(length, max_frames):
    if max_frames <= 0 or length <= max_frames:
        return list(range(length))
    if max_frames == 1:
        return [length // 2]
    idxs = np.linspace(0, length - 1, num=max_frames)
    return [int(round(v)) for v in idxs]


def make_even(value):
    value = int(value)
    if value < 2:
        value = 2
    if value % 2 != 0:
        value -= 1
    return max(value, 2)


def choose_target_size(height, width, max_side):
    out_h, out_w = height, width
    if max_side > 0 and max(height, width) > max_side:
        scale = float(max_side) / float(max(height, width))
        out_h = int(round(height * scale))
        out_w = int(round(width * scale))
    out_h = make_even(out_h)
    out_w = make_even(out_w)
    return out_h, out_w


def resize_frame(frame_rgb, target_h, target_w):
    h, w = frame_rgb.shape[:2]
    if h == target_h and w == target_w:
        return frame_rgb
    interpolation = cv2.INTER_AREA if (target_h < h or target_w < w) else cv2.INTER_LINEAR
    return cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)


def preprocess_pair(cover_frames, secret_frames, max_frames, max_side):
    paired = min(len(cover_frames), len(secret_frames))
    if paired <= 0:
        raise ValueError('Cover and secret videos must both have at least one frame.')

    idxs = sample_indices(paired, max_frames)
    cover = [cover_frames[i] for i in idxs]
    secret = [secret_frames[i] for i in idxs]

    target_h, target_w = choose_target_size(cover[0].shape[0], cover[0].shape[1], max_side)
    cover = [resize_frame(frame, target_h, target_w) for frame in cover]
    secret = [resize_frame(frame, target_h, target_w) for frame in secret]

    return cover, secret, {
        'paired_frames': paired,
        'used_frames': len(cover),
        'target_size': [target_h, target_w],
    }


def preprocess_single(frames, max_frames, max_side):
    idxs = sample_indices(len(frames), max_frames)
    out = [frames[i] for i in idxs]
    target_h, target_w = choose_target_size(out[0].shape[0], out[0].shape[1], max_side)
    out = [resize_frame(frame, target_h, target_w) for frame in out]
    return out, {
        'input_frames': len(frames),
        'used_frames': len(out),
        'target_size': [target_h, target_w],
    }


def preprocess_stego_for_decode(frames, max_frames):
    idxs = sample_indices(len(frames), max_frames)
    out = [frames[i] for i in idxs]

    in_h, in_w = out[0].shape[:2]
    target_h = make_even(in_h)
    target_w = make_even(in_w)
    cropped = target_h != in_h or target_w != in_w
    if cropped:
        out = [frame[:target_h, :target_w, :] for frame in out]

    return out, {
        'input_frames': len(frames),
        'used_frames': len(out),
        'target_size': [target_h, target_w],
        'resized': False,
        'cropped_to_even': cropped,
    }


def apply_size_preset(size_preset, max_frames, max_side):
    preset = str(size_preset).strip() if size_preset is not None else 'High Fidelity (largest file)'
    max_frames = max(0, int(max_frames))
    max_side = max(128, int(max_side))

    if preset == 'Balanced (recommended)':
        max_side = min(max_side, 384)
        max_frames = 180 if max_frames == 0 else min(max_frames, 180)
    elif preset == 'Compact (smallest file)':
        max_side = min(max_side, 320)
        max_frames = 120 if max_frames == 0 else min(max_frames, 120)

    return preset, max_frames, max_side


def reflect_index(idx, length):
    if length == 1:
        return 0
    out = idx
    while out < 0 or out >= length:
        if out < 0:
            out = -out
        elif out >= length:
            out = (2 * length) - out - 2
    return out


def gather_triplet(frames, center_idx):
    n = len(frames)
    ids = [
        reflect_index(center_idx - 1, n),
        reflect_index(center_idx, n),
        reflect_index(center_idx + 1, n),
    ]
    return np.stack([frames[i] for i in ids], axis=0)


def tensor_to_rgb_uint8(tensor):
    frame = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (frame * 255.0 + 0.5).astype(np.uint8)


def write_video(frames_rgb, output_path, fps):
    if not frames_rgb:
        raise ValueError('No frames to write for {}'.format(output_path))
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise ValueError('Unable to create output video {}'.format(output_path))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_video_lossless(frames_rgb, output_path, fps):
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix='lfvsn_lossless_'))
    try:
        for i, frame in enumerate(frames_rgb):
            frame_path = temp_dir / '{:06d}.png'.format(i)
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cmd = [
            ffmpeg_path,
            '-y',
            '-framerate',
            str(float(fps)),
            '-i',
            str(temp_dir / '%06d.png'),
            '-an',
            '-c:v',
            'libx264rgb',
            '-crf',
            '0',
            '-preset',
            'veryslow',
            '-pix_fmt',
            'rgb24',
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)


def choose_output_fps(user_fps, source_fps, input_count, used_count, keep_duration=True):
    user_fps = float(user_fps)
    if user_fps > 0:
        return user_fps
    if source_fps <= 0:
        source_fps = 24.0
    if keep_duration and input_count > 0 and used_count > 0:
        adjusted = source_fps * (float(used_count) / float(input_count))
        return max(1.0, adjusted)
    return source_fps


def psnr_uint8(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def add_label(frame_rgb, label):
    h, w = frame_rgb.shape[:2]
    canvas = np.full((h + 34, w, 3), 255, dtype=np.uint8)
    canvas[34:, :, :] = frame_rgb
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_preview_panel(frames_with_labels, output_path):
    tiles = [add_label(frame, label) for label, frame in frames_with_labels]
    panel = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


class LFVSNEngine:
    def __init__(self, opt_path, ckpt_path, device):
        self.opt_path = str(Path(opt_path).resolve())
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.device = torch.device(device)

        self.opt = load_option_for_inference(self.opt_path)
        self.gop = int(self.opt['gop'])
        self.num_video = int(self.opt['num_video'])
        if self.num_video != 1:
            raise ValueError('This demo app supports num_video=1 only.')
        if self.gop != 3:
            raise ValueError('This demo app is implemented for gop=3.')

        self.net = networks.define_G_v2(self.opt).to(self.device)
        state_dict = load_checkpoint(self.ckpt_path, self.device)
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()

        self.dwt = DWT().to(self.device)
        self.iwt = IWT().to(self.device)
        self.quant = Quantization().to(self.device)

    def _triplet_to_tensor(self, triplet_rgb):
        tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(triplet_rgb, (0, 3, 1, 2)))
        ).float()
        tensor = tensor.unsqueeze(0).to(self.device) / 255.0
        return tensor

    @torch.inference_mode()
    def encode_decode_triplet(self, cover_triplet_rgb, secret_triplet_rgb):
        cover_t = self._triplet_to_tensor(cover_triplet_rgb)
        secret_t = self._triplet_to_tensor(secret_triplet_rgb).unsqueeze(1)

        _, _, _, h, w = cover_t.shape

        host_dwt = self.dwt(cover_t.reshape(1, -1, h, w))
        secret_dwt = [self.dwt(secret_t[:, 0].reshape(1, -1, h, w))]

        output, _ = self.net(x=host_dwt, x_h=secret_dwt)
        output = self.iwt(output)

        stego = output[:, :3 * self.gop, :, :].reshape(1, self.gop, 3, h, w)[:, self.gop // 2]
        stego_q = self.quant(stego)

        y = self.quant(
            stego_q.unsqueeze(1).repeat(1, self.gop, 1, 1, 1).reshape(1, -1, h, w)
        )
        out_x, out_x_h, _ = self.net(x=self.dwt(y), rev=True)
        out_x = self.iwt(out_x)
        out_x_h = [self.iwt(out_x_h_i) for out_x_h_i in out_x_h]

        rec_cover = out_x.reshape(1, self.gop, 3, h, w)[:, self.gop // 2]
        rec_secret = torch.stack(out_x_h, dim=1).reshape(1, 1, self.gop, 3, h, w)[:, 0, self.gop // 2]

        stego_frame = tensor_to_rgb_uint8(stego_q)
        rec_cover_frame = tensor_to_rgb_uint8(rec_cover)
        rec_secret_frame = tensor_to_rgb_uint8(rec_secret)

        return stego_frame, rec_cover_frame, rec_secret_frame

    @torch.inference_mode()
    def decode_stego_frame(self, stego_rgb):
        stego_t = torch.from_numpy(
            np.ascontiguousarray(np.transpose(stego_rgb, (2, 0, 1)))
        ).float().unsqueeze(0).to(self.device) / 255.0

        _, _, h, w = stego_t.shape
        y = self.quant(stego_t.unsqueeze(1).repeat(1, self.gop, 1, 1, 1).reshape(1, -1, h, w))
        out_x, out_x_h, _ = self.net(x=self.dwt(y), rev=True)
        out_x = self.iwt(out_x)
        out_x_h = [self.iwt(out_x_h_i) for out_x_h_i in out_x_h]

        rec_cover = out_x.reshape(1, self.gop, 3, h, w)[:, self.gop // 2]
        rec_secret = torch.stack(out_x_h, dim=1).reshape(1, 1, self.gop, 3, h, w)[:, 0, self.gop // 2]

        return tensor_to_rgb_uint8(rec_cover), tensor_to_rgb_uint8(rec_secret)


class EngineCache:
    def __init__(self):
        self.cache = {}

    def get(self, opt_path, ckpt_path, device):
        key = (str(Path(opt_path).resolve()), str(Path(ckpt_path).resolve()), str(device))
        if key not in self.cache:
            self.cache[key] = LFVSNEngine(opt_path, ckpt_path, device)
        return self.cache[key]


def make_run_dir(output_root):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_root) / '{}_{}'.format(ts, uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_app(default_opt, default_ckpt, output_root, default_device):
    cache = EngineCache()

    def encode_and_decode(
        cover_video,
        secret_video,
        opt_path,
        ckpt_path,
        size_preset,
        max_frames,
        max_side,
        output_fps,
        keep_duration,
        progress=gr.Progress(track_tqdm=False),
    ):
        if cover_video is None or secret_video is None:
            raise gr.Error('Please upload both cover and secret videos.')
        opt_path = str(opt_path).strip() if opt_path else default_opt
        ckpt_path = str(ckpt_path).strip() if ckpt_path else default_ckpt
        if not Path(opt_path).exists():
            raise gr.Error('Option file not found: {}'.format(opt_path))
        if not Path(ckpt_path).exists():
            raise gr.Error('Checkpoint not found: {}'.format(ckpt_path))

        device = default_device
        engine = cache.get(opt_path, ckpt_path, device)

        try:
            cover_frames, cover_fps = read_video_frames(cover_video)
            secret_frames, _ = read_video_frames(secret_video)
        except Exception as err:
            raise gr.Error(str(err))

        size_preset, max_frames, max_side = apply_size_preset(size_preset, max_frames, max_side)
        cover_frames, secret_frames, prep = preprocess_pair(
            cover_frames, secret_frames, max_frames, max_side
        )

        n = len(cover_frames)
        stego_frames = []
        rec_cover_frames = []
        rec_secret_frames = []

        for i in range(n):
            progress((i + 1) / max(n, 1), desc='Encoding/decoding frame {}/{}'.format(i + 1, n))
            cover_triplet = gather_triplet(cover_frames, i)
            secret_triplet = gather_triplet(secret_frames, i)
            stego, rec_cover, rec_secret = engine.encode_decode_triplet(cover_triplet, secret_triplet)
            stego_frames.append(stego)
            rec_cover_frames.append(rec_cover)
            rec_secret_frames.append(rec_secret)

        fps = choose_output_fps(
            output_fps,
            cover_fps,
            prep['paired_frames'],
            prep['used_frames'],
            keep_duration=bool(keep_duration),
        )
        run_dir = make_run_dir(output_root)
        stego_path_preview = run_dir / 'stego_preview.mp4'
        stego_path_lossless = run_dir / 'stego_lossless.mp4'
        cover_rec_path = run_dir / 'decoded_cover.mp4'
        secret_rec_path = run_dir / 'decoded_secret.mp4'
        preview_path = run_dir / 'preview.png'

        write_video(stego_frames, stego_path_preview, fps)
        stego_lossless_ok = write_video_lossless(stego_frames, stego_path_lossless, fps)
        write_video(rec_cover_frames, cover_rec_path, fps)
        write_video(rec_secret_frames, secret_rec_path, fps)

        save_preview_panel([
            ('Cover GT', cover_frames[0]),
            ('Stego', stego_frames[0]),
            ('Decoded Cover', rec_cover_frames[0]),
            ('Secret GT', secret_frames[0]),
            ('Decoded Secret', rec_secret_frames[0]),
        ], preview_path)

        report = {
            'run_dir': str(run_dir),
            'device': str(device),
            'option_path': str(Path(opt_path).resolve()),
            'checkpoint_path': str(Path(ckpt_path).resolve()),
            'size_preset': size_preset,
            'preprocess': prep,
            'keep_duration': bool(keep_duration),
            'fps': fps,
            'stego_preview_path': str(stego_path_preview),
            'stego_lossless_path': str(stego_path_lossless) if stego_lossless_ok else None,
            'avg_psnr_cover': float(np.mean([psnr_uint8(a, b) for a, b in zip(rec_cover_frames, cover_frames)])),
            'avg_psnr_secret': float(np.mean([psnr_uint8(a, b) for a, b in zip(rec_secret_frames, secret_frames)])),
            'avg_psnr_stego_vs_cover': float(np.mean([psnr_uint8(a, b) for a, b in zip(stego_frames, cover_frames)])),
        }
        report_path = run_dir / 'run_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        report['run_report_path'] = str(report_path)

        stego_download = str(stego_path_lossless) if stego_lossless_ok else str(stego_path_preview)
        return (
            str(stego_path_preview),
            str(cover_rec_path),
            str(secret_rec_path),
            str(preview_path),
            report,
            stego_download,
        )

    def decode_only(
        stego_video,
        stego_file,
        opt_path,
        ckpt_path,
        max_frames,
        output_fps,
        keep_duration,
        progress=gr.Progress(track_tqdm=False),
    ):
        if stego_video is None and stego_file is None:
            raise gr.Error('Please upload a stego video/file.')
        opt_path = str(opt_path).strip() if opt_path else default_opt
        ckpt_path = str(ckpt_path).strip() if ckpt_path else default_ckpt
        if not Path(opt_path).exists():
            raise gr.Error('Option file not found: {}'.format(opt_path))
        if not Path(ckpt_path).exists():
            raise gr.Error('Checkpoint not found: {}'.format(ckpt_path))

        device = default_device
        engine = cache.get(opt_path, ckpt_path, device)

        source = stego_file if stego_file is not None else stego_video

        try:
            stego_frames, stego_fps = read_video_frames(source)
        except Exception as err:
            raise gr.Error(str(err))

        max_frames = max(0, int(max_frames))
        stego_frames, prep = preprocess_stego_for_decode(stego_frames, max_frames)

        n = len(stego_frames)
        rec_cover_frames = []
        rec_secret_frames = []
        for i, stego in enumerate(stego_frames):
            progress((i + 1) / max(n, 1), desc='Decoding frame {}/{}'.format(i + 1, n))
            rec_cover, rec_secret = engine.decode_stego_frame(stego)
            rec_cover_frames.append(rec_cover)
            rec_secret_frames.append(rec_secret)

        fps = choose_output_fps(
            output_fps,
            stego_fps,
            prep['input_frames'],
            prep['used_frames'],
            keep_duration=bool(keep_duration),
        )
        run_dir = make_run_dir(output_root)
        cover_rec_path = run_dir / 'decoded_cover.mp4'
        secret_rec_path = run_dir / 'decoded_secret.mp4'
        preview_path = run_dir / 'preview_decode.png'

        write_video(rec_cover_frames, cover_rec_path, fps)
        write_video(rec_secret_frames, secret_rec_path, fps)
        save_preview_panel([
            ('Stego', stego_frames[0]),
            ('Decoded Cover', rec_cover_frames[0]),
            ('Decoded Secret', rec_secret_frames[0]),
        ], preview_path)

        report = {
            'run_dir': str(run_dir),
            'device': str(device),
            'option_path': str(Path(opt_path).resolve()),
            'checkpoint_path': str(Path(ckpt_path).resolve()),
            'preprocess': prep,
            'keep_duration': bool(keep_duration),
            'fps': fps,
        }
        report_path = run_dir / 'run_report_decode.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        report['run_report_path'] = str(report_path)

        return str(cover_rec_path), str(secret_rec_path), str(preview_path), report

    with gr.Blocks(title='LF-VSN Prototype Demo') as demo:
        gr.Markdown(
            '# LF-VSN Prototype App\n'
            'Upload a cover video and a secret video to generate stego output, then decode recovered cover/secret.\n'
            'The app automatically aligns frame count, resizes to a valid shape, and enforces even dimensions for DWT/IWT.\n'
            'Tip: set Max Frames = 0 to process full video without trimming.\n'
            'Important: use the **lossless stego file** output for decode; the preview video is lossy and can break secret recovery.\n'
            'Use Size Preset to trade off stego size vs quality while keeping decode safe.'
        )

        with gr.Tab('Encode + Decode'):
            with gr.Row():
                cover_video = gr.Video(label='Cover Video')
                secret_video = gr.Video(label='Secret Video')
            with gr.Row():
                opt_in = gr.Textbox(value=default_opt, label='Option YAML Path')
                ckpt_in = gr.Textbox(value=default_ckpt, label='Checkpoint Path')
            with gr.Row():
                size_preset = gr.Dropdown(
                    choices=[
                        'High Fidelity (largest file)',
                        'Balanced (recommended)',
                        'Compact (smallest file)',
                    ],
                    value='Balanced (recommended)',
                    label='Size Preset',
                )
                max_frames = gr.Number(value=0, precision=0, label='Max Frames to Process (0 = all)')
                max_side = gr.Slider(128, 1280, value=512, step=2, label='Max Frame Side (px)')
                out_fps = gr.Number(value=0, label='Output FPS (0 = source FPS)')
                keep_duration = gr.Checkbox(value=True, label='Keep Original Duration')

            run_btn = gr.Button('Run Encode + Decode', variant='primary')

            with gr.Row():
                stego_video = gr.Video(label='Stego Video')
                decoded_cover_video = gr.Video(label='Decoded Cover Video')
                decoded_secret_video = gr.Video(label='Decoded Secret Video')
            preview_img = gr.Image(type='filepath', label='Preview Panel')
            report_json = gr.JSON(label='Run Report')
            stego_file = gr.File(label='Download Stego (lossless for decode)', file_count='single')

            run_btn.click(
                fn=encode_and_decode,
                inputs=[cover_video, secret_video, opt_in, ckpt_in, size_preset, max_frames, max_side, out_fps, keep_duration],
                outputs=[
                    stego_video,
                    decoded_cover_video,
                    decoded_secret_video,
                    preview_img,
                    report_json,
                    stego_file,
                ],
            )

        with gr.Tab('Decode Existing Stego'):
            stego_input = gr.Video(label='Stego Video')
            stego_input_file = gr.File(label='Stego File (recommended)', file_count='single')
            with gr.Row():
                opt_in2 = gr.Textbox(value=default_opt, label='Option YAML Path')
                ckpt_in2 = gr.Textbox(value=default_ckpt, label='Checkpoint Path')
            with gr.Row():
                max_frames2 = gr.Number(value=0, precision=0, label='Max Frames to Process (0 = all)')
                out_fps2 = gr.Number(value=0, label='Output FPS (0 = source FPS)')
                keep_duration2 = gr.Checkbox(value=True, label='Keep Original Duration')

            run_btn2 = gr.Button('Run Decode', variant='primary')

            with gr.Row():
                decoded_cover_video2 = gr.Video(label='Decoded Cover Video')
                decoded_secret_video2 = gr.Video(label='Decoded Secret Video')
            preview_img2 = gr.Image(type='filepath', label='Preview Panel')
            report_json2 = gr.JSON(label='Run Report')

            run_btn2.click(
                fn=decode_only,
                inputs=[stego_input, stego_input_file, opt_in2, ckpt_in2, max_frames2, out_fps2, keep_duration2],
                outputs=[decoded_cover_video2, decoded_secret_video2, preview_img2, report_json2],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description='LF-VSN web demo app')
    parser.add_argument(
        '--opt',
        default=str(CODE_ROOT / 'options/train/train_LF-VSN_1video_proto.yml'),
        help='Path to YAML option file.',
    )
    parser.add_argument(
        '--ckpt',
        default=str(REPO_ROOT / 'experiments/train_LF-VSN_1video_proto/models/latest_G.pth'),
        help='Path to model checkpoint.',
    )
    parser.add_argument(
        '--output-dir',
        default=str(CODE_ROOT / 'results/app_demo_runs'),
        help='Directory for generated videos and reports.',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Host for Gradio app.')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio app.')
    parser.add_argument('--share', action='store_true', help='Enable Gradio public share link.')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Inference device.',
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    demo = build_app(
        default_opt=str(Path(args.opt).resolve()),
        default_ckpt=str(Path(args.ckpt).resolve()),
        output_root=output_root,
        default_device=args.device,
    )
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue(concurrency_count=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
