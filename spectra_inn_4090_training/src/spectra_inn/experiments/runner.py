"""Sequential single-GPU study with persistent state and a suite-wide deadline.

No downloads, package installation, shell execution, or training is performed by --plan.
Run the actual suite in a persistent user terminal, not an assistant background job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from spectra_inn.experiments.planning import choose_challenger, common_steps, validate_study


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def suite_lock(path: Path):
    """OS lock is released on server/process death; no unsafe stale-PID cleanup."""
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stop_process(process: subprocess.Popen, grace: float) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.wait(timeout=max(0.1, grace))
    except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=15)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # query only, never terminate
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: conservatively assume alive
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def source_signature(project: Path, study_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    paths = [*project.glob("src/**/*.py"), study_path, base_path,
             project / "scripts/run_4090_suite.py"]
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(project)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_manifest(
    root: Path, splits: list[str], destination: Path, *, strict_counts: bool = True,
) -> dict[str, Any]:
    """Reindex on this machine: never carry stale absolute paths inside the future ZIP."""
    from spectra_inn.data.evaluation import speaker_id
    from spectra_inn.data.manifest import build_manifest

    sources = [(name, "train") for name in splits] + [("dev-clean", "val"), ("test-clean", "test")]
    expected_counts = {
        "train-clean-100": 28539, "train-clean-360": 104014,
        "dev-clean": 2703, "test-clean": 2620,
    }
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("Training splits must be nonempty and unique")
    temporary = destination.with_suffix(".tmp")
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted indexing file before retrying: {temporary}")
    for index, (name, split) in enumerate(sources):
        count, failures = build_manifest(
            root=root / name, output=temporary, train_ratio=0.90, val_ratio=0.05,
            source=f"librispeech_{name.replace('-', '_')}", forced_split=split, append=index > 0,
        )
        if count < 2 or failures:
            raise ValueError(f"Indexing failed for {name}: {count} records, errors: {failures[:5]}")
        if strict_counts and count != expected_counts[name]:
            raise ValueError(
                f"Incomplete or duplicated official split {name}: expected {expected_counts[name]} "
                f"audio records, found {count}. Finish copying/downloading before training."
            )
    speakers: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    hours: dict[str, float] = defaultdict(float)
    with temporary.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_rate"] != 16000 or row["channels"] != 1:
                raise ValueError(f"Expected 16kHz mono LibriSpeech, found {row['path']}")
            speakers[row["split"]].add(speaker_id(row))
            counts[row["split"]] += 1
            hours[row["split"]] += row["duration"] / 3600
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        if speakers[first] & speakers[second]:
            raise ValueError(f"Speaker IDs leak between {first} and {second}")
    temporary.replace(destination)
    return {"sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "counts": dict(counts), "hours": dict(hours),
            "speakers": {key: len(value) for key, value in speakers.items()},
            "audio_root": str(root), "train_splits": splits}


class Suite:
    def __init__(self, args, project: Path, study: dict, base: dict, signature: str) -> None:
        self.args, self.project, self.study, self.base = args, project, study, base
        self.root = args.output.expanduser().resolve()
        self.state_path = self.root / "suite_state.json"
        if args.resume_suite:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["source_sha256"] != signature:
                raise ValueError("Code/study changed; use a NEW suite output directory")
            if self.state["data_root"] != str(args.data_root.expanduser().resolve()):
                raise ValueError("Data root changed; do not mix datasets in one suite")
            if self.state["train_splits"] != args.train_splits:
                raise ValueError("Training splits changed; use a NEW output directory")
            if self.state["budget_hours"] != args.budget_hours:
                raise ValueError("Resume cannot reset or extend the original wall-clock budget")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            now = time.time()
            self.state = {
                "created_utc_epoch": now, "deadline_utc_epoch": now + args.budget_hours * 3600,
                "budget_hours": args.budget_hours, "source_sha256": signature,
                "data_root": str(args.data_root.expanduser().resolve()),
                "train_splits": args.train_splits, "host": socket.gethostname(),
                "workers": args.workers, "requested_micro_batch": args.micro_batch,
                "jobs": {}, "status": "created",
            }
            self.save()
        self.monotonic_deadline = time.monotonic() + self.state["deadline_utc_epoch"] - time.time()
        self.manifest = self.root / "manifest.jsonl"
        if self.state.get("workers") != args.workers or (
            self.state.get("requested_micro_batch") != args.micro_batch
        ):
            raise ValueError("Resume must use the same workers and requested micro-batch")

    def remaining(self) -> float:
        return min(self.state["deadline_utc_epoch"] - time.time(),
                   self.monotonic_deadline - time.monotonic())

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def report(self) -> None:
        rows = ["# RTX 4090 suite report", "", f"Status: **{self.state['status']}**", "",
                "Selection uses validation only; historical results are not new controls.",
                "Equal-step counts refer to update attempts; AMP skips are reported separately.",
                ""]
        for key in ("screen_steps", "winner", "selection_failures", "repeat_gate_failures",
                    "stop_reason"):
            if key in self.state:
                rows += [f"## {key}", "", "```json",
                         json.dumps(self.state[key], indent=2), "```", ""]
        rows += ["## Evaluation results", "",
                 "| Job | Split / pairing | Pairs | Cover SNR | Secret SNR | SI-SDR | Clipping |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for job, info in self.state["jobs"].items():
            result = info.get("result")
            if result and "metrics" in result:
                m = result["metrics"]
                rows.append(
                    f"| {job} | {result['split']} / {result.get('pairing', 'legacy')} "
                    f"| {result['num_items']} | {m['cover_snr_db']:.3f} "
                    f"| {m['secret_snr_db_predicted']:.3f} "
                    f"| {m['secret_si_sdr_db_predicted']:.3f} "
                    f"| {m['stego_clip_fraction']:.6g} |"
                )
        rows += ["", "## Jobs", "", "| Job | State | Latest log |", "|---|---|---|"]
        for job, info in self.state["jobs"].items():
            attempts = info.get("attempts", [])
            log = attempts[-1]["log"] if attempts else ""
            rows.append(f"| {job} | {info['status']} | `{log}` |")
        rows += ["", "Configs, commands, errors, runtimes and hashes are in suite_state.json.",
                 "Do not interpret an incomplete/failed run as a completed benchmark.", ""]
        (self.root / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")

    def job(self, name: str, argv: list[str], seconds: float, result_path: Path) -> dict:
        info = self.state["jobs"].setdefault(name, {"status": "pending", "attempts": []})
        if info["status"] == "completed":
            if not result_path.is_file():
                raise FileNotFoundError(f"Completed artifact is missing: {result_path}")
            return info["result"]
        if info["attempts"] and pid_alive(info["attempts"][-1].get("pid", -1)):
            raise RuntimeError(
                f"Recorded child for {name} is still alive; do not launch a duplicate GPU job"
            )
        limit = min(seconds, self.remaining() - 45)
        if limit < 10:
            raise TimeoutError("Suite deadline leaves insufficient time to start another job")
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{name}.attempt_{len(info['attempts']) + 1:02d}.log"
        attempt = {"argv": argv, "log": str(log), "started_utc_epoch": time.time()}
        info["attempts"].append(attempt)
        info["status"] = "running"
        self.save()
        self.report()
        print(f"[{name}] starting; log: {log}", flush=True)
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                   else {"start_new_session": True})
        with log.open("xb") as output:
            process = subprocess.Popen(argv, cwd=self.project, stdout=output,
                                       stderr=subprocess.STDOUT, **options)
            attempt["pid"] = process.pid
            self.save()
            try:
                code = process.wait(timeout=max(1, limit - 30))
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                stop_process(process, min(30, max(1, self.remaining() - 5)))
                info["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout"
                attempt["returncode"] = process.returncode
                self.save()
                raise
        attempt["returncode"] = code
        attempt["elapsed_seconds"] = time.time() - attempt["started_utc_epoch"]
        if code != 0 or not result_path.exists():
            info["status"] = "failed"
            info["error_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
            self.save()
            raise RuntimeError(f"Job {name} failed (exit {code}); see complete traceback in {log}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "target_reached" in result and not result["target_reached"]:
            info["status"] = "incomplete"
            info["partial_result"] = result
            self.save()
            raise RuntimeError(f"{name} hit its safety budget before the common step target")
        attempts = result.get("steps_attempted_session", 0)
        if attempts >= 1000 and result.get("amp_skipped_updates_session", 0) > 0.05 * attempts:
            info["status"] = "failed"
            info["partial_result"] = result
            self.save()
            raise RuntimeError(f"{name}: more than5percentAMPupdates skipped; investigate numerics")
        if "metrics" in result and not all(math.isfinite(v) for v in result["metrics"].values()):
            info["status"] = "failed"
            self.save()
            raise RuntimeError(f"Non-finite evaluation metrics in {name}")
        info.update(status="completed", result=result)
        self.save()
        self.report()
        return result

    def render(self, arm: dict, name: str, seed: int, batch: int, steps: int) -> tuple[Path, Path]:
        from spectra_inn.config import validate_config

        config = deepcopy(self.base)
        config.pop("_config_path", None)
        output = self.root / "training" / name
        config["experiment"].update(name=name, seed=seed, output_dir=str(output))
        config["data"].update(
            manifest=str(self.manifest), num_workers=self.args.workers,
            validation_pairing="cross_speaker", evaluation_pair_seed=int(self.study["pair_seed"]),
            validation_subset_size=int(self.study["monitor_pairs"]),
            loader_seed=seed,
        )
        config["data"].pop("overfit_pairs", None)
        config["model"]["predictor"] = deepcopy(arm["predictor"])
        config["model"]["latent_mode"] = "predicted"
        config["loss"].update(waveform_mode="normalized_mse", latent_prior=0.0,
                               latent_prediction=float(arm["latent_prediction"]))
        config["training"].update(
            batch_size=batch, accumulation_steps=16 // batch, max_steps=steps, resume=None,
            amp_init_scale=1024.0, checkpoint_every=int(self.study["checkpoint_every"]),
            validate_every=int(self.study["checkpoint_every"]),
            validation_batches=math.ceil(int(self.study["monitor_pairs"]) / batch),
        )
        if name.startswith("profile_"):
            config["training"].update(validation_batches=4, validate_every=100000)
        validate_config(config)
        destination = self.root / "configs" / f"{name}.yaml"
        destination.parent.mkdir(exist_ok=True)
        text = yaml.safe_dump(config, sort_keys=False)
        if destination.exists() and destination.read_text(encoding="utf-8") != text:
            raise ValueError(f"Generated configuration changed: {destination}")
        destination.write_text(text, encoding="utf-8")
        return destination, output

    def train(
        self, arm: dict, name: str, steps: int, seconds: float,
        resume: Path | None = None, seed: int | None = None, lr: float | None = None,
        batch: int | None = None,
    ) -> Path:
        batch = batch or self.state["micro_batch"]
        config, output = self.render(arm, name, seed or int(self.study["seed"]), batch, steps)
        result_path = output / "training_summary.json"
        expected = output / "checkpoints" / f"step_{steps:08d}.pt"
        existing = self.state["jobs"].get(name, {})
        if existing.get("status") == "completed":
            self.job(name, [], seconds, result_path)
            if not expected.is_file():
                raise FileNotFoundError(expected)
            return expected
        # Recover a child that finished just before the suite process/server disappeared.
        if result_path.is_file() and expected.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("target_reached") and result["step"] == steps:
                self.state["jobs"].setdefault(name, {"attempts": []}).update(
                    status="completed", result=result
                )
                self.save()
                return expected
        if expected.is_file() and not result_path.is_file():
            if name.startswith("profile_"):
                raise RuntimeError("Profile has no timing summary; inspect before retrying")
            result = {"step": steps, "target_steps": steps, "target_reached": True,
                      "recovered_checkpoint_only": True}
            atomic_json(result_path, result)
            self.state["jobs"].setdefault(name, {"attempts": []}).update(
                status="completed", result=result
            )
            self.save()
            return expected
        partials = sorted((output / "checkpoints").glob("step_*.pt"))
        if partials:
            resume = partials[-1]
        hours = max(1 / 3600, (min(seconds, self.remaining() - 45) - 120) / 3600)
        argv = [sys.executable, "-u", "-m", "spectra_inn.train", "--config", str(config),
                "--device", "cuda", "--max-steps", str(steps), "--max-hours", str(hours)]
        if resume is not None:
            argv += ["--resume", str(resume)]
        if lr is not None:
            argv += ["--learning-rate", str(lr)]
        self.job(name, argv, seconds, result_path)
        if not expected.is_file():
            raise FileNotFoundError(f"Target checkpoint not produced: {expected}")
        return expected

    def evaluate(self, name: str, checkpoint: Path, split: str, pairing: str) -> dict:
        job_name = f"eval_{name}_{split}_{pairing}"
        output = self.root / "evaluation" / job_name
        argv = [sys.executable, "-u", "-m", "spectra_inn.evaluate", "--checkpoint", str(checkpoint),
                "--manifest", str(self.manifest), "--split", split, "--device", "cuda",
                "--batch-size", str(self.state["micro_batch"]), "--max-batches", "100000",
                "--examples", "5" if split == "test" else "0", "--pairing", pairing,
                "--pair-seed", str(self.study["pair_seed"]), "--output-dir", str(output)]
        result = self.job(job_name, argv, min(1800, max(10, self.remaining() - 60)),
                          output / "metrics.json")
        if result["num_items"] != self.state["dataset"]["counts"][split]:
            raise ValueError(f"Incomplete {split} evaluation in {job_name}")
        return result

    def execute(self) -> None:
        if self.remaining() < 120:
            raise TimeoutError("Original suite deadline has expired; it is not reset by resume")
        command = [sys.executable, "-c", (
            "import json,torch; assert torch.cuda.is_available(); "
            "p=torch.cuda.get_device_properties(0); "
            "print(json.dumps({'name':p.name,'vram_bytes':p.total_memory,"
            "'torch':torch.__version__,'cuda':torch.version.cuda}))"
        )]
        gpu = json.loads(subprocess.check_output(command, cwd=self.project, timeout=30, text=True))
        if not self.args.allow_other_gpu and (
            "4090" not in gpu["name"] or gpu["vram_bytes"] < 22 * 1024**3
        ):
            raise ValueError("This suite requires a 24GB RTX4090; --allow-other-gpu is explicit")
        if "gpu" in self.state and self.state["gpu"] != gpu:
            raise ValueError("GPU/PyTorch environment changed; reprofile in a NEW suite")
        self.state["gpu"] = gpu
        self.save()
        if shutil.disk_usage(self.root).free < self.args.min_free_gib * 1024**3:
            raise ValueError("Insufficient disk headroom; no files are automatically deleted")
        if "dataset" not in self.state:
            summary = self.root / "manifest.summary.json"
            command = [sys.executable, "-u", "-m", "spectra_inn.experiments.runner",
                       "--suite", str(self.args.suite), "--data-root", self.state["data_root"],
                       "--train-splits", *self.args.train_splits,
                       "--index-manifest-output", str(self.manifest)]
            self.state["dataset"] = self.job(
                "index_manifest", command, min(1800, self.remaining() - 60), summary
            )
            self.save()
        elif hashlib.sha256(self.manifest.read_bytes()).hexdigest() != (
            self.state["dataset"]["sha256"]
        ):
            raise ValueError("Suite manifest changed; refusing to mix data")
        arms = self.study["arms"]
        control = arms[0]
        costs = self.state.get("seconds_per_step")
        if costs is None:
            for batch in [value for value in (8, 4, 2, 1) if value <= self.args.micro_batch]:
                measured = {}
                try:
                    for arm in arms:
                        name = f"profile_b{batch}_{arm['name']}"
                        self.train(arm, name, int(self.study["profile_steps"]), 900, batch=batch)
                        info = self.state["jobs"][name]
                        result = info["result"]
                        if result["steps_attempted_session"] < 50:
                            raise RuntimeError("Too few measured profile steps after recovery")
                        measured[arm["name"]] = (
                            result["elapsed_seconds"] / result["steps_attempted_session"]
                        )
                        if result["amp_skipped_updates_session"] > (
                            0.2 * result["steps_attempted_session"]
                        ):
                            raise RuntimeError(f"Excessive AMP skipped updates in {name}")
                    self.state.update(micro_batch=batch, seconds_per_step=measured)
                    self.save()
                    costs = measured
                    break
                except RuntimeError:
                    last = self.state["jobs"].get(name, {})
                    tail = last.get("error_tail", "").lower()
                    if batch == 1 or not ("out of memory" in tail or "outofmemoryerror" in tail):
                        raise
                    print(f"OOM with batch {batch}; reprofile ALL arms at a smaller micro-batch.")
            if costs is None:
                raise RuntimeError("No viable micro-batch")
        reserve = float(self.study["evaluation_reserve_hours"]) * 3600
        safety = float(self.study["throughput_safety"])
        interval = int(self.study["checkpoint_every"])
        if "screen_steps" not in self.state:
            # Includes BOTH second-seed replicas, not only the selected candidate.
            repeat_cost = costs[control["name"]] + max(costs[a["name"]] for a in arms[1:])
            budget = min(self.state["budget_hours"] * 0.45 * 3600, self.remaining() - reserve - 600)
            steps = common_steps(budget, [*costs.values(), repeat_cost], safety, interval,
                                 int(self.study["max_screen_steps"]))
            if steps < int(self.study["min_screen_steps"]):
                raise RuntimeError("Measured throughput cannot fit the minimum equal-step screen; "
                                   "use a new 30h suite or investigate CPU/SSD throughput")
            self.state["screen_steps"] = steps
            self.save()
        steps = self.state["screen_steps"]
        checkpoints = {}
        results = {}
        for arm in arms:
            name = arm["name"]
            checkpoints[name] = self.train(
                arm, f"screen_{name}", steps, steps * costs[name] / safety + 180
            )
            results[name] = {
                protocol: self.evaluate(f"screen_{name}", checkpoints[name], "val", protocol)
                for protocol in ("legacy", "cross_speaker")
            }
        winner, failures = choose_challenger(results, control["name"], self.study["selection"])
        self.state.update(winner=winner, selection_failures=failures)
        self.save()
        if winner is None:
            self.state.update(status="completed_screen_no_promotion", stop_reason=(
                "No challenger passed predeclared VALIDATION gates. Long continuations skipped; "
                "negative screening results are not failed training."
            ))
            return
        selected = next(arm for arm in arms if arm["name"] == winner)
        finalists = [control, selected]
        # Reserve matched-seed work before allocating the remaining refinement budget.
        repeats = {}
        for arm in finalists:
            name = arm["name"]
            replica = self.train(
                arm, f"repeat_{name}", steps, steps * costs[name] / safety + 180,
                seed=int(self.study["repeat_seed"]),
            )
            score = self.evaluate(f"repeat_{name}", replica, "val", "cross_speaker")
            repeats[name] = {"legacy": score, "cross_speaker": score}
        replicated, repeat_failures = choose_challenger(
            repeats, control["name"], self.study["selection"]
        )
        self.state["repeat_gate_failures"] = repeat_failures
        self.save()
        if replicated is None:
            self.state.update(status="completed_screen_not_replicated", stop_reason=(
                "The shortlisted challenger failed the SAME gates on cross-speaker validation "
                "at the matched second seed. Long continuations skipped."
            ))
            return
        if "refine_target" not in self.state:
            available = self.remaining() - reserve - 600
            additional = common_steps(
                available, [costs[a["name"]] for a in finalists], safety, interval,
                max(interval, int(self.study["max_refine_steps"]) - steps),
            )
            if additional < interval:
                raise RuntimeError("No safe budget remains for a matched continuation")
            self.state["refine_target"] = steps + additional
            self.save()
        target = self.state["refine_target"]
        final_checkpoints = {}
        for arm in finalists:
            name = arm["name"]
            final_checkpoints[name] = self.train(
                arm, f"refine_{name}", target,
                (target - steps) * costs[name] / safety + 180,
                resume=checkpoints[name], lr=float(self.study["fine_tune_learning_rate"]),
            )
        # Test is not read by choose_challenger and is scored only after finalists are frozen.
        for arm in finalists:
            for split in ("val", "test"):
                for protocol in ("legacy", "cross_speaker"):
                    self.evaluate(f"refine_{arm['name']}", final_checkpoints[arm["name"]],
                                  split, protocol)
        self.state["status"] = "completed"


def main() -> None:
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=project / "configs/4090_suite.yaml")
    parser.add_argument(
        "--data-root", type=Path, default=project / "datasets/librispeech/LibriSpeech"
    )
    parser.add_argument("--train-splits", nargs="+", choices=("train-clean-100", "train-clean-360"),
                        default=["train-clean-100", "train-clean-360"])
    parser.add_argument("--output", type=Path, default=project / "runs/4090_suite_v1")
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--resume-suite", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--index-manifest-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.index_manifest_output is not None:
        destination = args.index_manifest_output.expanduser().resolve()
        result = prepare_manifest(
            args.data_root.expanduser().resolve(), args.train_splits, destination
        )
        atomic_json(destination.with_suffix(".summary.json"), result)
        return
    args.suite = args.suite.expanduser().resolve()
    study = yaml.safe_load(args.suite.read_text(encoding="utf-8"))
    validate_study(study)
    if args.budget_hours is None:
        args.budget_hours = study["budget_hours"]
    if (not 24 <= args.budget_hours <= 30 or args.workers < 0
            or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0):
        parser.error("Budget must be 24..30 hours; workers and disk headroom must be nonnegative")
    base_path = args.suite.parent / study["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if args.plan:
        print(yaml.safe_dump({"budget_hours": args.budget_hours, "arms": study["arms"],
                             "common_effective_batch": 16,
                             "minimum_screen_steps": study["min_screen_steps"],
                             "maximum_screen_steps": study["max_screen_steps"],
                             "evaluation_reserve_hours": study["evaluation_reserve_hours"],
                             "execution": "profile -> matched screens -> validation gates -> "
                             "matched seed repeats -> matched LR continuations -> final val/test",
                             "note": "No fixed4090speedup assumed; no experiment files created."},
                            sort_keys=False))
        return
    suite = Suite(args, project, study, base, source_signature(project, args.suite, base_path))
    with suite_lock(suite.root / ".suite.lock"):
        if suite.state["status"].startswith("completed"):
            suite.report()
            print(f"Suite already finished: {suite.root / 'REPORT.md'}")
            return
        try:
            suite.state["status"] = "running"
            suite.save()
            suite.execute()
        except (Exception, KeyboardInterrupt) as exc:
            suite.state.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                stop_reason=f"{type(exc).__name__}: {exc}",
            )
            suite.save()
            suite.report()
            raise
        finally:
            suite.save()
            suite.report()
    print(f"Suite finished: {suite.state['status']}. Report: {suite.root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
