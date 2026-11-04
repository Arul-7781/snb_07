from __future__ import annotations

import math
import re
from typing import Any


def validate_study(study: dict[str, Any]) -> None:
    interval = int(study["checkpoint_every"])
    if interval < 1 or not 0 < float(study["throughput_safety"]) <= 1:
        raise ValueError("Invalid checkpoint interval or throughput safety")
    if not 0 < int(study["profile_steps"]) < int(study["min_screen_steps"]):
        raise ValueError("Profile must be shorter than the minimum screen")
    if not interval <= int(study["min_screen_steps"]) <= int(study["max_screen_steps"]):
        raise ValueError("Invalid common screen step limits")
    if int(study["max_refine_steps"]) < int(study["max_screen_steps"]) + interval:
        raise ValueError("Refinement cap must leave room after screening")
    if study["seed"] == study["repeat_seed"]:
        raise ValueError("Replication seed must differ from the screening seed")
    reserve = float(study["evaluation_reserve_hours"])
    if not math.isfinite(reserve) or not 0.5 <= reserve <= 6:
        raise ValueError("Evaluation reserve must be finite and between0.5and6hours")
    names = [arm["name"] for arm in study["arms"]]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Study needs a control and uniquely named challengers")
    if any(re.fullmatch(r"[a-z0-9_]+", name) is None for name in names):
        raise ValueError("Use only lowercase letters, digits and underscores in arm names")


def common_steps(seconds: float, costs: list[float], safety: float, interval: int, cap: int) -> int:
    """Equal update-attempt counts, accounting for the SUM of sequential runtime costs."""
    if (not costs or any(not math.isfinite(c) or c <= 0 for c in costs)
            or not math.isfinite(seconds) or seconds <= 0 or not 0 < safety <= 1
            or interval < 1 or cap < interval):
        raise ValueError("Invalid throughput/budget parameters")
    count = int(seconds * safety / sum(costs))
    return min(cap // interval, count // interval) * interval


def choose_challenger(
    results: dict[str, dict[str, Any]], control: str, rules: dict[str, float]
) -> tuple[str | None, dict[str, list[str]]]:
    """Gate on validation only. Never force promotion of a failed challenger."""
    baseline = results[control]
    for protocol in ("legacy", "cross_speaker"):
        metrics = baseline[protocol]["metrics"]
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Control has non-finite validation metrics; selection is invalid")
        if (metrics["secret_snr_db_true_residual"] < rules["min_oracle_snr_db"]
                or metrics["exact_cycle_max_global"] > rules["max_cycle_error"]):
            raise ValueError("Control failed numerical-health gates; comparison is invalid")
    failures: dict[str, list[str]] = {}
    eligible = []
    for name, protocols in results.items():
        if name == control:
            continue
        reasons = []
        for protocol in ("legacy", "cross_speaker"):
            candidate = protocols[protocol]["metrics"]
            reference = baseline[protocol]["metrics"]
            if not all(math.isfinite(value) for value in candidate.values()):
                reasons.append(f"{protocol}: non-finite metric")
                continue
            if candidate["cover_snr_db"] < reference["cover_snr_db"] - rules["max_cover_loss_db"]:
                reasons.append(f"{protocol}: cover quality regressed")
            if candidate["stego_clip_fraction"] > rules["max_clip_fraction"]:
                reasons.append(f"{protocol}: clipping gate failed")
            drop = candidate["secret_snr_db_predicted"] - candidate["secret_snr_db_predicted_pcm16"]
            if drop > rules["max_pcm16_drop_db"]:
                reasons.append(f"{protocol}: PCM16 gate failed")
            if candidate["secret_snr_db_true_residual"] < rules["min_oracle_snr_db"]:
                reasons.append(f"{protocol}: numerical-inversion gate failed")
            if candidate["exact_cycle_max_global"] > rules["max_cycle_error"]:
                reasons.append(f"{protocol}: worst-case cycle-error gate failed")
        c = protocols["cross_speaker"]["metrics"]
        b = baseline["cross_speaker"]["metrics"]
        gain = c["secret_snr_db_predicted"] - b["secret_snr_db_predicted"]
        if gain < rules["min_secret_gain_db"]:
            reasons.append("cross_speaker: no sufficient secret-SNR gain")
        if c["secret_si_sdr_db_predicted"] < (
            b["secret_si_sdr_db_predicted"] - rules["max_si_sdr_loss_db"]
        ):
            reasons.append("cross_speaker: SI-SDR gate failed")
        failures[name] = reasons
        if not reasons:
            eligible.append((c["secret_snr_db_predicted"], name))
    return (max(eligible)[1] if eligible else None), failures
