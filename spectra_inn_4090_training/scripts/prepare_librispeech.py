"""Stream verified LibriSpeech splits to disk without retaining tar.gz archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO

GIB = 1024**3
CHUNK = 1024**2
CHECK_INTERVAL = 32 * CHUNK
MARKER = ".spectra_verified.json"
# Published at https://www.openslr.org/resources/12/md5sum.txt.
# Extracted sizes are conservative planning estimates, not exact measurements.
SPLITS = {
    "train-clean-100": ("2a93770f6d5c6c964bc36631d331a522", 6.5, "train"),
    "train-clean-360": ("c0e676e450a7ff2f54aeade5171606fa", 24.0, "train"),
    "dev-clean": ("42e2234ba48799c1f50f24a7926300a1", 0.4, "val"),
    "test-clean": ("32fa31d27d2e1cad72775fee3f4849a9", 0.4, "test"),
}


class HashingReader:
    def __init__(self, source: BinaryIO, name: str) -> None:
        self.source = source
        self.name = name
        self.digest = hashlib.md5(usedforsecurity=False)
        self.bytes_read = 0
        self.next_report = 256 * CHUNK

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        if self.bytes_read >= self.next_report:
            print(f"{self.name}: downloaded {self.bytes_read / GIB:.2f} GiB", flush=True)
            self.next_report = self.bytes_read + 256 * CHUNK
        return data


def free_bytes(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def require_reserve(path: Path, reserve_bytes: int) -> None:
    if free_bytes(path) < reserve_bytes:
        raise RuntimeError("Free-space reserve reached; stopping extraction before filling disk")


def verified_split(data_dir: Path, name: str) -> bool:
    root = data_dir / "LibriSpeech" / name
    if not root.exists():
        return False
    marker = root / MARKER
    if root.is_symlink() or not marker.is_file():
        raise RuntimeError(
            f"Existing dataset has no trusted completion marker: {root}. "
            "Use a different --data-dir; existing files will not be overwritten."
        )
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("split") != name or record.get("md5") != SPLITS[name][0]:
        raise RuntimeError(f"Invalid completion marker: {marker}")
    return True


def extract_stream(
    reader: HashingReader, staging: Path, name: str, reserve_bytes: int
) -> None:
    written_since_check = CHECK_INTERVAL
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported archive member: {member.name}")
            # Ignore shared README/license/speaker metadata. Keep only this official split.
            if path.parts[:2] != ("LibriSpeech", name):
                continue
            target = staging.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with source, target.open("xb") as output:
                while chunk := source.read(CHUNK):
                    if written_since_check >= CHECK_INTERVAL:
                        require_reserve(staging, reserve_bytes)
                        written_since_check = 0
                    output.write(chunk)
                    written_since_check += len(chunk)
    # tar can stop at its end marker before the HTTP body is exhausted.
    while reader.read(CHUNK):
        pass
    require_reserve(staging, reserve_bytes)


def download_split(data_dir: Path, name: str, base_url: str, reserve_bytes: int) -> None:
    if verified_split(data_dir, name):
        print(f"Using previously verified {name}")
        return
    staging = data_dir / f".partial-{name}"
    if staging.exists():
        raise RuntimeError(
            f"Incomplete download exists: {staging}. Inspect/remove only that partial directory "
            "before retrying. Streaming downloads cannot resume, and it will not be overwritten."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    require_reserve(data_dir, reserve_bytes)
    staging.mkdir()
    url = f"{base_url.rstrip('/')}/{name}.tar.gz"
    print(f"Downloading {url}; no archive will be retained", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SPECTRA-INN/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            reader = HashingReader(response, name)
            extract_stream(reader, staging, name, reserve_bytes)
        actual = reader.digest.hexdigest()
        if actual != SPLITS[name][0]:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual}")
        split_root = staging / "LibriSpeech" / name
        if not split_root.is_dir() or not any(split_root.rglob("*.flac")):
            raise RuntimeError(f"No FLAC audio extracted for {name}")
        (split_root / MARKER).write_text(
            json.dumps({"split": name, "md5": actual, "url": url}, indent=2) + "\n",
            encoding="utf-8",
        )
        destination = data_dir / "LibriSpeech" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing dataset: {destination}")
        split_root.rename(destination)
        # Only remove empty staging directories belonging to this download.
        (staging / "LibriSpeech").rmdir()
        staging.rmdir()
        print(f"Checksum verified: {destination}", flush=True)
    except BaseException:
        print(f"Download did not finish. Partial files retained at {staging}", flush=True)
        raise


def write_manifest(data_dir: Path, names: list[str], manifest: Path) -> None:
    # Import only when indexing; --plan and --download-only need only the standard library.
    from spectra_inn.data.manifest import build_manifest

    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Inspect the incomplete manifest before retrying: {temporary}")
    for index, name in enumerate(names):
        count, failures = build_manifest(
            root=data_dir / "LibriSpeech" / name,
            output=temporary,
            train_ratio=0.90,
            val_ratio=0.05,
            source=f"librispeech_{name.replace('-', '_')}",
            forced_split=SPLITS[name][2],
            append=index > 0,
        )
        if failures or count < 2:
            raise RuntimeError(f"Indexing {name} failed: {count} records; {failures[:5]}")
    totals: dict[str, dict[str, float | int]] = {}
    with temporary.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            values = totals.setdefault(record["split"], {"tracks": 0, "hours": 0.0})
            values["tracks"] += 1
            values["hours"] += record["duration"] / 3600
    temporary.replace(manifest)
    summary = {"manifest": str(manifest), "official_splits": names, "totals": totals}
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/librispeech"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/rpm_librispeech.jsonl")
    )
    parser.add_argument(
        "--train-splits", nargs="+", choices=("train-clean-100", "train-clean-360"),
        default=["train-clean-100", "train-clean-360"],
    )
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--base-url", default="https://www.openslr.org/resources/12")
    parser.add_argument("--plan", action="store_true", help="Show storage plan without downloading")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must not be negative")
    data_dir = args.data_dir.expanduser().resolve()
    names = list(dict.fromkeys([*args.train_splits, "dev-clean", "test-clean"]))
    missing = [name for name in names if not verified_split(data_dir, name)]
    estimated_gib = sum(SPLITS[name][1] for name in missing)
    available_gib = free_bytes(data_dir) / GIB
    print(f"Splits: {', '.join(names)}")
    print(f"Estimated additional extracted data: {estimated_gib:.1f} GiB")
    print(f"Free: {available_gib:.1f} GiB; reserve: {args.reserve_gib:.1f} GiB")
    if available_gib < estimated_gib + args.reserve_gib:
        raise SystemExit(
            "Not enough free space for this plan. Use --train-splits train-clean-100, "
            "or choose --data-dir on a larger filesystem. Do not fill the system disk."
        )
    if args.plan:
        return
    if not args.download_only:
        # Fail before downloading gigabytes if the indexing environment is not ready.
        import spectra_inn.data.manifest  # noqa: F401

    for name in names:
        download_split(data_dir, name, args.base_url, int(args.reserve_gib * GIB))
    if not args.download_only:
        write_manifest(data_dir, names, args.manifest.expanduser().resolve())


if __name__ == "__main__":
    main()
