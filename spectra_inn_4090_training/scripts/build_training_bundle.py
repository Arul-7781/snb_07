"""Build a small, allow-listed training ZIP; never crawl datasets or old runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

PREFIX = "spectra_inn_4090_training"
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv", ".codegraph"}


def payload_files(root: Path) -> dict[str, Path]:
    root = root.resolve()
    required = ["pyproject.toml", "TRAINING_README.md", "constraints-training.txt", ".gitignore"]
    files = {name: root / name for name in required}
    # The bundle opens with the training guide, not the repository's legacy launcher README.
    files["README.md"] = root / "TRAINING_README.md"
    for folder, pattern in [
        ("src/spectra_inn", "*.py"), ("scripts", "*.py"), ("tests", "*.py"),
        ("configs", "*.yaml"), ("docs", "*.md"),
    ]:
        for path in sorted((root / folder).rglob(pattern)):
            relative = path.relative_to(root)
            if any(part in CACHE_PARTS or part.startswith(".") for part in relative.parts):
                continue
            files[relative.as_posix()] = path
    for name in ["scripts/run_4090_suite.py", "scripts/prepare_librispeech.py",
                 "scripts/check_env.py", "configs/4090_suite.yaml",
                 "configs/rpm_3060_nmse_10k.yaml", "src/spectra_inn/experiments/runner.py",
                 "src/spectra_inn/train.py", "src/spectra_inn/evaluate.py",
                 "src/spectra_inn/data/dataset.py", "src/spectra_inn/data/evaluation.py",
                 "src/spectra_inn/losses/audio.py"]:
        if name not in files:
            raise FileNotFoundError(f"Required training payload is missing: {name}")
    for name, path in files.items():
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Payload must be a local file, not an external link: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Unexpectedly large file in training payload: {name}")
        path.read_bytes().decode("utf-8")  # Only source/configuration/documentation text.
    return dict(sorted(files.items()))


def git_source(root: Path) -> dict:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all", "--", ".")
    return {"git_commit": commit, "working_tree_dirty": None if status is None else bool(status)}


def build_bundle(root: Path, output: Path, *, force: bool = False) -> dict:
    root = root.resolve()
    output = output.expanduser().resolve()
    checksum = output.with_suffix(output.suffix + ".sha256")
    if not force and (output.exists() or checksum.exists()):
        raise FileExistsError(f"Bundle/checksum exists; inspect it or use --force: {output}")
    files = payload_files(root)
    payload = {name: path.read_bytes() for name, path in files.items()}
    total = sum(len(data) for data in payload.values())
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Training-only payload exceeds4MiB; inspect unexpected additions")
    manifest = {
        "format": 1, "archive_root": PREFIX, "source": git_source(root),
        "entrypoint": "scripts/run_4090_suite.py", "dataset_included": False,
        "trained_weights_included": False, "payload_bytes": total,
        "timestamps": "normalized for reproducible builds",
        "files": [{"path": name, "source_path": path.relative_to(root).as_posix(),
                   "bytes": len(payload[name]), "sha256": hashlib.sha256(payload[name]).hexdigest()}
                  for name, path in files.items()],
    }
    payload["BUNDLE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(f"{PREFIX}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"zip": str(output), "compressed_bytes": output.stat().st_size,
            "payload_files": len(files), "sha256": digest, "source": manifest["source"]}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist" / f"{PREFIX}.zip")
    parser.add_argument(
        "--force", action="store_true", help="Explicitly replace an existing bundle"
    )
    args = parser.parse_args()
    print(json.dumps(build_bundle(root, args.output, force=args.force), indent=2))


if __name__ == "__main__":
    main()
