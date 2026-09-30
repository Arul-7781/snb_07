from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def deterministic_split(relative_path: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def inspect_audio(
    path: Path,
    root: Path,
    train_ratio: float,
    val_ratio: float,
    source: str,
    forced_split: str | None = None,
    group_by_parent: bool = False,
) -> dict[str, Any]:
    info = sf.info(path)
    relative = path.relative_to(root).as_posix()
    split_key = path.parent.name if group_by_parent else relative
    return {
        "path": str(path.resolve()),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": float(info.duration),
        "split": forced_split or deterministic_split(split_key, train_ratio, val_ratio),
        "source": source,
        "group": path.parent.name if group_by_parent else None,
    }


def build_manifest(
    root: Path,
    output: Path,
    train_ratio: float,
    val_ratio: float,
    source: str,
    pattern: str = "**/*",
    forced_split: str | None = None,
    append: bool = False,
    group_by_parent: bool = False,
) -> tuple[int, list[tuple[str, str]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train>0, val>=0, and train+val<1")
    if not root.is_dir():
        raise FileNotFoundError(f"Audio root does not exist: {root}")

    paths = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    count = 0
    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for path in paths:
            try:
                record = inspect_audio(
                    path,
                    root,
                    train_ratio,
                    val_ratio,
                    source,
                    forced_split=forced_split,
                    group_by_parent=group_by_parent,
                )
            except Exception as exc:  # corrupt third-party media should not abort indexing
                failures.append((str(path), str(exc)))
                continue
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic JSONL audio manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--source", default="local")
    parser.add_argument("--pattern", default="**/*")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--group-by-parent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, failures = build_manifest(
        root=args.root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        source=args.source,
        pattern=args.pattern,
        forced_split=args.split,
        append=args.append,
        group_by_parent=args.group_by_parent,
    )
    print(f"Wrote {count} records to {args.output}")
    if failures:
        print(f"Skipped {len(failures)} unreadable files")
        for path, reason in failures[:20]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
