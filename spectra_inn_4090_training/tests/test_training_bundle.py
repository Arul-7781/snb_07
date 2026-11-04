from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_training_bundle import MAX_FILE_BYTES, PREFIX, build_bundle, payload_files


def fixture_project(root: Path) -> Path:
    files = {
        "pyproject.toml": "[project]\nname = 'test-project'\n",
        "README.md": "Legacy repository README; not the bundle entry point.",
        "TRAINING_README.md": "# Training instructions\n",
        "constraints-training.txt": "pytest==8.4.2\n",
        ".gitignore": "dist/\n",
        "scripts/run_4090_suite.py": "# runner\n",
        "scripts/prepare_librispeech.py": "# downloader\n",
        "scripts/check_env.py": "# checker\n",
        "src/spectra_inn/experiments/runner.py": "# orchestration\n",
        "src/spectra_inn/train.py": "# trainer\n",
        "src/spectra_inn/evaluate.py": "# evaluator\n",
        "src/spectra_inn/data/dataset.py": "# audio loader code, not an audio dataset\n",
        "src/spectra_inn/data/evaluation.py": "# pairing code\n",
        "src/spectra_inn/losses/audio.py": "# loss functions\n",
        "configs/4090_suite.yaml": "name: test\n",
        "configs/rpm_3060_nmse_10k.yaml": "name: base\n",
        "tests/test_example.py": "# test\n",
        "docs/4090_experiment_plan.md": "# plan\n",
        "datasets/audio.flac": "DO NOT INCLUDE",
        "data/manifests/private.jsonl": "DO NOT INCLUDE",
        "runs/checkpoints/model.pt": "DO NOT INCLUDE",
        "results/private_result.json": "DO NOT INCLUDE",
        "dist/old.zip": "DO NOT INCLUDE",
        ".venv/private.py": "DO NOT INCLUDE",
        "src/spectra_inn/__pycache__/fake.py": "DO NOT INCLUDE",
        "scripts/.private.py": "DO NOT INCLUDE",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_only_source_and_training_material_are_selected(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    files = payload_files(root)
    assert files["README.md"] == root / "TRAINING_README.md"
    assert "src/spectra_inn/data/dataset.py" in files
    assert not any(name.startswith(("datasets/", "data/", "runs/", "results/", "dist/"))
                   for name in files)
    assert all("__pycache__" not in name and ".private" not in name for name in files)


def test_zip_hashes_checksums_and_contents_are_valid(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    output = tmp_path / "delivery.zip"
    result = build_bundle(root, output)
    assert result["compressed_bytes"] < 100000
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_suffix(".zip.sha256").read_text().startswith(result["sha256"])
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read(f"{PREFIX}/BUNDLE_MANIFEST.json"))
        assert not manifest["dataset_included"]
        assert not manifest["trained_weights_included"]
        assert archive.read(f"{PREFIX}/README.md") == (root / "TRAINING_README.md").read_bytes()
        for item in manifest["files"]:
            data = archive.read(f"{PREFIX}/{item['path']}")
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert len(archive.namelist()) == len(manifest["files"]) + 1


def test_repeated_builds_of_the_same_source_are_identical(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    first = build_bundle(root, tmp_path / "first.zip")
    second = build_bundle(root, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]


def test_existing_delivery_is_not_overwritten_without_explicit_force(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    output = tmp_path / "delivery.zip"
    first = build_bundle(root, output)
    with pytest.raises(FileExistsError):
        build_bundle(root, output)
    assert build_bundle(root, output, force=True)["sha256"] == first["sha256"]


def test_external_symlink_is_not_packaged(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    external = tmp_path / "external.py"
    external.write_text("private external material")
    link = root / "scripts/external.py"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="external link"):
        payload_files(root)


def test_unexpected_large_payload_is_rejected(tmp_path: Path) -> None:
    root = fixture_project(tmp_path / "project")
    (root / "scripts/large.py").write_text("#" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="large file"):
        payload_files(root)
