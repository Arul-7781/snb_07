from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint
from .data import LibriSpeechPairDataset
from .metrics import si_sdr_db, snr_db
from .model import AudioINN
from .utils import load_config, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--pairs", type=int, default=512)
    ap.add_argument("--output", default="evaluation.json")
    args = ap.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = raw["config"]
    split = args.split or cfg["data"]["test_split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioINN(**cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ds = LibriSpeechPairDataset(
        cfg["data"]["root"], split,
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        num_pairs=args.pairs,
        seed=cfg["seed"] + 200000,
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    sums = {"cover_snr": 0.0, "secret_sisdr_random_z": 0.0, "secret_sisdr_true_r": 0.0}
    n = 0
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 1234)
    with torch.no_grad():
        for cover, secret in tqdm(loader):
            cover, secret = cover.to(device), secret.to(device)
            y, r = model.encode(cover, secret)
            z = torch.randn(r.shape, device=device, dtype=r.dtype, generator=gen)
            _, s_random = model.decode(y, z)
            _, s_oracle = model.decode(y, r)
            b = cover.shape[0]
            sums["cover_snr"] += float(snr_db(cover, y).mean()) * b
            sums["secret_sisdr_random_z"] += float(si_sdr_db(secret, s_random).mean()) * b
            sums["secret_sisdr_true_r"] += float(si_sdr_db(secret, s_oracle).mean()) * b
            n += b

    result = {k: v / max(n, 1) for k, v in sums.items()}
    result.update({"checkpoint": args.checkpoint, "split": split, "pairs": n})
    write_json(Path(args.output), result)
    print(result)


if __name__ == "__main__":
    main()
