from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from spectra_inn.experiments.planning import choose_challenger, common_steps, validate_study
from spectra_inn.experiments.runner import Suite, pid_alive, prepare_manifest, suite_lock

ROOT = Path(__file__).resolve().parents[1]


def args_for(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=tmp_path / "new_suite", data_root=tmp_path / "LibriSpeech",
        resume_suite=False, budget_hours=28.0, train_splits=["train-clean-100"],
        workers=0, micro_batch=8, min_free_gib=0, allow_other_gpu=False,
    )


def make_suite(tmp_path: Path) -> Suite:
    study = yaml.safe_load((ROOT / "configs/4090_suite.yaml").read_text())
    base = yaml.safe_load((ROOT / "configs/rpm_3060_nmse_10k.yaml").read_text())
    return Suite(args_for(tmp_path), ROOT, study, base, "fixed_source_signature")


def score(secret: float = 10, cover: float = 18) -> dict:
    return {"metrics": {
        "cover_snr_db": cover, "secret_snr_db_predicted": secret,
        "secret_snr_db_predicted_pcm16": secret - 0.01,
        "secret_si_sdr_db_predicted": secret - 1,
        "stego_clip_fraction": 1e-6, "secret_snr_db_true_residual": 85,
        "exact_cycle_max_global": 1e-4,
    }}


def protocols(secret=10, cover=18) -> dict:
    return {name: score(secret, cover) for name in ("legacy", "cross_speaker")}


def test_equal_steps_accounts_for_sum_of_all_sequential_costs() -> None:
    assert common_steps(10000, [0.2, 0.3, 0.5], 0.75, 1000, 50000) == 7000
    assert common_steps(100000, [0.2], 0.75, 1000, 50000) == 50000
    assert common_steps(100, [1], 0.75, 1000, 50000) == 0


@pytest.mark.parametrize("field,value", [
    ("throughput_safety", 0), ("repeat_seed", 3407), ("evaluation_reserve_hours", float("nan")),
    ("min_screen_steps", 100), ("max_refine_steps", 500),
])
def test_study_misconfiguration_fails_before_spending_gpu_time(
    tmp_path: Path, field: str, value: object,
) -> None:
    suite = make_suite(tmp_path)
    study = deepcopy(suite.study)
    study[field] = value
    with pytest.raises(ValueError):
        validate_study(study)


@pytest.mark.parametrize("costs", [[], [0], [float("nan")], [-1]])
def test_invalid_throughput_is_not_treated_as_free_training(costs: list) -> None:
    with pytest.raises(ValueError):
        common_steps(1000, costs, 0.75, 1000, 50000)


def test_selection_uses_validation_constraints_not_oracle_scores(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    results = {"a": protocols(), "b": protocols(10.3), "c": protocols(11),
               "d": protocols(12, cover=16)}
    winner, reasons = choose_challenger(results, "a", suite.study["selection"])
    assert winner == "c"
    assert reasons["b"] and reasons["d"]
    assert reasons["c"] == []


def test_no_qualifying_candidate_is_not_forced_to_win(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    winner, _ = choose_challenger({"a": protocols(), "b": protocols(10.1)}, "a",
                                  suite.study["selection"])
    assert winner is None


def test_control_nan_invalidates_selection(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    with pytest.raises(ValueError, match="Control"):
        choose_challenger({"a": protocols(float("nan")), "b": protocols(12)}, "a",
                          suite.study["selection"])


def test_resume_preserves_original_deadline_and_rejects_source_changes(tmp_path: Path) -> None:
    first = make_suite(tmp_path)
    original = first.state["deadline_utc_epoch"]
    args = args_for(tmp_path)
    args.resume_suite = True
    second = Suite(args, ROOT, first.study, first.base, "fixed_source_signature")
    assert second.state["deadline_utc_epoch"] == original
    assert second.remaining() <= 28 * 3600
    with pytest.raises(ValueError, match="Code/study"):
        Suite(args, ROOT, first.study, first.base, "changed_source")


def test_existing_directory_is_not_overwritten(tmp_path: Path) -> None:
    first = make_suite(tmp_path)
    with pytest.raises(FileExistsError):
        Suite(args_for(tmp_path), ROOT, first.study, first.base, "fixed_source_signature")


def test_os_lock_cannot_be_acquired_twice(tmp_path: Path) -> None:
    with suite_lock(tmp_path / "lock"):
        with pytest.raises(OSError):
            with suite_lock(tmp_path / "lock"):
                pass
    with suite_lock(tmp_path / "lock"):
        pass


def test_failed_child_keeps_traceback_and_does_not_continue(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    command = [sys.executable, "-c", "raise RuntimeError('specific_original_failure')"]
    with pytest.raises(RuntimeError, match="failed"):
        suite.job("bad_child", command, 30, suite.root / "never_written.json")
    entry = suite.state["jobs"]["bad_child"]
    assert entry["status"] == "failed"
    assert "specific_original_failure" in entry["error_tail"]
    assert "Traceback" in Path(entry["attempts"][0]["log"]).read_text()


def test_successful_child_is_not_rerun_on_resume(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    result = suite.root / "result.json"
    code = f"from pathlib import Path; Path({str(result)!r}).write_text('{{}}')"
    command = [sys.executable, "-c", code]
    assert suite.job("child", command, 30, result) == {}
    assert suite.job("child", ["does-not-exist"], 30, result) == {}
    assert len(suite.state["jobs"]["child"]["attempts"]) == 1


def test_expired_budget_does_not_launch_a_child(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    suite.monotonic_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        suite.job("not_launched", ["does-not-exist"], 30, suite.root / "missing.json")


def test_resume_does_not_duplicate_a_still_running_child(tmp_path: Path) -> None:
    suite = make_suite(tmp_path)
    suite.state["jobs"]["live"] = {
        "status": "running", "attempts": [{"pid": os.getpid()}],
    }
    assert pid_alive(os.getpid())
    assert not pid_alive(-1)
    with pytest.raises(RuntimeError, match="duplicate GPU"):
        suite.job("live", ["does-not-exist"], 30, suite.root / "missing.json")
    assert len(suite.state["jobs"]["live"]["attempts"]) == 1


def test_plan_never_creates_an_experiment_directory(tmp_path: Path) -> None:
    output = tmp_path / "no_creation"
    result = subprocess.run(
        [sys.executable, "scripts/run_4090_suite.py", "--plan", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=30,
    )
    assert "budget_hours: 28" in result.stdout
    assert not output.exists()


def test_manifest_rebuilds_local_absolute_paths_and_checks_speaker_leaks(tmp_path: Path) -> None:
    root = tmp_path / "LibriSpeech"
    for split, speaker in [("train-clean-100", "1"), ("dev-clean", "2"), ("test-clean", "3")]:
        folder = root / split / speaker / "chapter"
        folder.mkdir(parents=True)
        for index in range(2):
            sf.write(folder / f"{index}.flac", np.full(256, 0.1, dtype=np.float32), 16000)
    result = prepare_manifest(
        root, ["train-clean-100"], tmp_path / "manifest.jsonl", strict_counts=False
    )
    assert result["counts"] == {"train": 2, "val": 2, "test": 2}
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert all(Path(row["path"]).is_absolute() and Path(row["path"]).is_file() for row in rows)
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        prepare_manifest(root, ["train-clean-100"], tmp_path / "not_official.jsonl")
    (root / "test-clean/3").rename(root / "test-clean/1")
    with pytest.raises(ValueError, match="leak"):
        prepare_manifest(root, ["train-clean-100"], tmp_path / "leaky.jsonl", strict_counts=False)


def test_full_orchestration_is_sequential_matched_and_test_is_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = make_suite(tmp_path)
    suite.manifest.write_text("[]")
    import hashlib

    suite.state.update(
        dataset={"sha256": hashlib.sha256(b"[]").hexdigest(), "counts": {"val": 4, "test": 4}},
        micro_batch=8,
        seconds_per_step={arm["name"]: 0.05 for arm in suite.study["arms"]},
    )
    gpu = json.dumps({"name": "RTX 4090", "vram_bytes": 24 * 1024**3, "torch": "x", "cuda": "x"})
    monkeypatch.setattr(
        "spectra_inn.experiments.runner.subprocess.check_output", lambda *a, **k: gpu
    )
    calls = []

    def fake_train(arm, name, steps, seconds, **kwargs):
        calls.append(("train", name, steps, kwargs))
        return suite.root / name / f"step_{steps:08d}.pt"

    def fake_evaluate(name, checkpoint, split, pairing):
        calls.append(("eval", name, split, pairing))
        value = 11 if "c_unet" in name else 10
        return {"split": split, "pairing": pairing, "num_items": 4, **score(value)}

    monkeypatch.setattr(suite, "train", fake_train)
    monkeypatch.setattr(suite, "evaluate", fake_evaluate)
    suite.execute()
    assert suite.state["status"] == "completed"
    training = [call for call in calls if call[0] == "train"]
    assert len(training) == 8  # four screens, two matched seed repeats, two continuations
    assert len({call[2] for call in training[:4]}) == 1
    assert training[4][2] == training[5][2] == training[0][2]
    assert training[6][2] == training[7][2]
    assert training[6][3]["lr"] == training[7][3]["lr"] == 3e-5
    first_test = next(
        index for index, call in enumerate(calls) if call[0] == "eval" and call[2] == "test"
    )
    assert not any(call[0] == "train" for call in calls[first_test:])


def test_no_promotion_stops_before_repeats_or_long_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = make_suite(tmp_path)
    import hashlib

    suite.manifest.write_text("[]")
    suite.state.update(dataset={"sha256": hashlib.sha256(b"[]").hexdigest()}, micro_batch=8,
                       seconds_per_step={arm["name"]: 0.05 for arm in suite.study["arms"]})
    gpu = json.dumps({"name": "RTX 4090", "vram_bytes": 24 * 1024**3})
    monkeypatch.setattr(
        "spectra_inn.experiments.runner.subprocess.check_output", lambda *a, **k: gpu
    )
    calls = []

    def fake_train(arm, name, *args, **kwargs):
        calls.append(name)
        return suite.root / name / "checkpoint.pt"

    monkeypatch.setattr(suite, "train", fake_train)
    monkeypatch.setattr(suite, "evaluate", lambda *args: deepcopy(score()))
    suite.execute()
    assert suite.state["status"] == "completed_screen_no_promotion"
    assert len(calls) == 4


def test_second_seed_failure_prevents_expensive_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    suite = make_suite(tmp_path)
    suite.manifest.write_text("[]")
    suite.state.update(dataset={"sha256": hashlib.sha256(b"[]").hexdigest()}, micro_batch=8,
                       seconds_per_step={arm["name"]: 0.05 for arm in suite.study["arms"]})
    gpu = json.dumps({"name": "RTX 4090", "vram_bytes": 24 * 1024**3})
    monkeypatch.setattr(
        "spectra_inn.experiments.runner.subprocess.check_output", lambda *a, **k: gpu
    )
    calls = []

    def fake_train(arm, name, *args, **kwargs):
        calls.append(name)
        return suite.root / name / "checkpoint.pt"

    def fake_evaluate(name, *args):
        return score(11 if name == "screen_c_unet" else 10)

    monkeypatch.setattr(suite, "train", fake_train)
    monkeypatch.setattr(suite, "evaluate", fake_evaluate)
    suite.execute()
    assert suite.state["status"] == "completed_screen_not_replicated"
    assert len(calls) == 6
    assert not any(name.startswith("refine_") for name in calls)
