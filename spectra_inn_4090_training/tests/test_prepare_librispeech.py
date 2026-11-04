from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_librispeech as prepare


def archive_bytes(name: str = "LibriSpeech/train-clean-100/1/2/example.flac") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        contents = b"test audio placeholder"
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))
        readme = tarfile.TarInfo("LibriSpeech/README.TXT")
        readme.size = 4
        archive.addfile(readme, io.BytesIO(b"info"))
    return buffer.getvalue()


class PrepareLibriSpeechTests(unittest.TestCase):
    def test_hash_includes_bytes_after_tar_end(self) -> None:
        # Real gzip archives can include padding beyond the end of the tar stream.
        payload = archive_bytes() + b"\x00" * 40000
        with tempfile.TemporaryDirectory() as directory:
            reader = prepare.HashingReader(io.BytesIO(payload), "test")
            prepare.extract_stream(reader, Path(directory), "train-clean-100", 0)
            self.assertEqual(reader.bytes_read, len(payload))
            self.assertEqual(reader.digest.hexdigest(), hashlib.md5(payload).hexdigest())
            self.assertTrue(
                (Path(directory) / "LibriSpeech/train-clean-100/1/2/example.flac").is_file()
            )
            self.assertFalse((Path(directory) / "LibriSpeech/README.TXT").exists())

    def test_path_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "../../escape.flac",
                "/absolute.flac",
                "LibriSpeech/train-clean-100/../../escape.flac",
                "LibriSpeech/train-clean-100/..\\escape.flac",
            ):
                with self.subTest(name=name):
                    reader = prepare.HashingReader(io.BytesIO(archive_bytes(name)), "test")
                    with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                        prepare.extract_stream(reader, Path(directory), "train-clean-100", 0)

    def test_symlink_rejected(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("LibriSpeech/train-clean-100/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside"
            archive.addfile(info)
        with tempfile.TemporaryDirectory() as directory:
            reader = prepare.HashingReader(io.BytesIO(buffer.getvalue()), "test")
            with self.assertRaisesRegex(ValueError, "Unsupported archive member"):
                prepare.extract_stream(reader, Path(directory), "train-clean-100", 0)

    def test_free_space_reserve_stops_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader = prepare.HashingReader(io.BytesIO(archive_bytes()), "test")
            with patch.object(prepare, "free_bytes", return_value=0):
                with self.assertRaisesRegex(RuntimeError, "reserve reached"):
                    prepare.extract_stream(reader, Path(directory), "train-clean-100", 1)

    def test_unverified_existing_dataset_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "LibriSpeech/train-clean-100"
            root.mkdir(parents=True)
            original = root / "user-file.txt"
            original.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no trusted completion marker"):
                prepare.download_split(Path(directory), "train-clean-100", "https://test", 0)
            self.assertEqual(original.read_text(encoding="utf-8"), "keep")

    def test_verified_existing_split_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "LibriSpeech/train-clean-100"
            root.mkdir(parents=True)
            (root / prepare.MARKER).write_text(
                json.dumps(
                    {"split": "train-clean-100", "md5": prepare.SPLITS["train-clean-100"][0]}
                ),
                encoding="utf-8",
            )
            with patch.object(prepare.urllib.request, "urlopen") as request:
                prepare.download_split(Path(directory), "train-clean-100", "https://test", 0)
            request.assert_not_called()

    def test_successful_download_promoted_after_verification(self) -> None:
        payload = archive_bytes()
        md5 = hashlib.md5(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with (
                patch.dict(prepare.SPLITS, {"train-clean-100": (md5, 0.01, "train")}),
                patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(payload)),
            ):
                prepare.download_split(data_dir, "train-clean-100", "https://test", 0)
                self.assertTrue(prepare.verified_split(data_dir, "train-clean-100"))
            self.assertFalse((data_dir / ".partial-train-clean-100").exists())
            self.assertEqual(list(data_dir.rglob("*.tar.gz")), [])

    def test_checksum_failure_never_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with patch.object(
                prepare.urllib.request, "urlopen", return_value=io.BytesIO(archive_bytes())
            ):
                with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                    prepare.download_split(data_dir, "train-clean-100", "https://test", 0)
            self.assertFalse((data_dir / "LibriSpeech/train-clean-100").exists())
            self.assertTrue((data_dir / ".partial-train-clean-100").is_dir())

    def test_existing_partial_download_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / ".partial-train-clean-100").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Incomplete download exists"):
                prepare.download_split(data_dir, "train-clean-100", "https://test", 0)

    def test_free_space_lookup_handles_nonexistent_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertGreater(prepare.free_bytes(Path(directory) / "not-created/yet"), 0)

    def test_manifest_preserves_official_splits_without_duplicate_records(self) -> None:
        module = types.ModuleType("spectra_inn.data.manifest")

        def fake_build(**kwargs: object) -> tuple[int, list[tuple[str, str]]]:
            output = Path(str(kwargs["output"]))
            with output.open("a" if kwargs["append"] else "w", encoding="utf-8") as handle:
                for index in range(2):
                    record = {
                        "path": f"{kwargs['root']}/{index}.flac",
                        "duration": 3600,
                        "split": kwargs["forced_split"],
                    }
                    handle.write(json.dumps(record) + "\n")
            return 2, []

        module.build_manifest = fake_build
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            names = list(prepare.SPLITS)
            with patch.dict(sys.modules, {"spectra_inn.data.manifest": module}):
                prepare.write_manifest(root, names, manifest)
                prepare.write_manifest(root, names, manifest)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(records), 8)
            self.assertEqual(len({record["path"] for record in records}), 8)
            self.assertEqual({record["split"] for record in records}, {"train", "val", "test"})
            summary = json.loads(manifest.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["totals"]["train"], {"tracks": 4, "hours": 4.0})
            self.assertFalse(manifest.with_suffix(".jsonl.tmp").exists())

    def test_manifest_failure_preserves_previous_manifest(self) -> None:
        module = types.ModuleType("spectra_inn.data.manifest")
        module.build_manifest = lambda **kwargs: (0, [("bad.flac", "unreadable")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("previous manifest\n", encoding="utf-8")
            with patch.dict(sys.modules, {"spectra_inn.data.manifest": module}):
                with self.assertRaisesRegex(RuntimeError, "Indexing .* failed"):
                    prepare.write_manifest(root, ["train-clean-100"], manifest)
            self.assertEqual(manifest.read_text(), "previous manifest\n")


if __name__ == "__main__":
    unittest.main()
