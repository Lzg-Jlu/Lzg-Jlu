"""Recuperation ↔ mechanical brake handover approximate algorithm (VIORE-like).

Assesses whether blended braking feels seamless:
  1) regen torque utilization during deceleration demand
  2) handover bump (ax jerk around hydraulic intervention)
  3) mode differentiation (optional second mode ax mean difference)
  4) deceleration reproducibility (optional multi-run std)

This is an open methodology sketch for BEV calibration benchmarking.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.filters import derivative, moving_average
from ..core.rating import DriveRating, combine_local_ratings
from ..core.transfer import piecewise_linear_score


def detect_decel_event(
    ax: np.ndarray,
    brake_req: np.ndarray,
    dt: float,
    ax_thr: float = -0.5,
    hold_s: float = 0.4,
) -> Optional[tuple[int, int]]:
    """Return [start, end) indices of a sustained deceleration request."""
    ax = np.asarray(ax, dtype=float)
    brake_req = np.asarray(brake_req, dtype=float)
    need = max(1, int(hold_s / dt))
    active = (ax < ax_thr) | (brake_req > 0.05)
    run = 0
    start = None
    for i, flag in enumerate(active):
        if flag:
            if start is None:
                start = i
            run += 1
        else:
            if start is not None and run >= need:
                return start, i
            start, run = None, 0
    if start is not None and run >= need:
        return start, len(active)
    return None


def _regen_utilization(
    motor_brake_nm: np.ndarray,
    hyd_brake: np.ndarray,
    i0: int,
    i1: int,
) -> float:
    """Fraction of braking effort provided by e-machine (proxy)."""
    m = np.abs(motor_brake_nm[i0:i1])
    h = np.maximum(hyd_brake[i0:i1], 0.0)
    # normalize hydraulic pressure/torque proxy into comparable magnitude
    # user should scale hyd_brake to Nm-equivalent offline if possible
    total = m + h
    if np.sum(total) < 1e-6:
        return 0.0
    return float(np.sum(m) / np.sum(total))


def _handover_bump(
    ax: np.ndarray,
    hyd_brake: np.ndarray,
    i0: int,
    i1: int,
    dt: float,
) -> float:
    """Peak |jerk| near first hydraulic intervention."""
    seg_h = hyd_brake[i0:i1]
    if len(seg_h) < 3:
        return 0.0
    # first sample where hydraulic clearly rises
    rel = np.where(seg_h > (np.min(seg_h) + 0.1 * (np.max(seg_h) - np.min(seg_h) + 1e-9)))[0]
    if len(rel) == 0:
        return 0.0
    k = i0 + int(rel[0])
    a = max(i0, k - int(0.25 / dt))
    b = min(i1, k + int(0.25 / dt))
    jerk = derivative(ax[a:b], dt)
    return float(np.max(np.abs(jerk))) if len(jerk) else 0.0


def _mode_difference(ax_mode_a: np.ndarray, ax_mode_b: np.ndarray) -> float:
    """Absolute difference of mean deceleration between regen modes."""
    return float(abs(np.mean(ax_mode_a) - np.mean(ax_mode_b)))


def _reproducibility(ax_runs: list[np.ndarray]) -> float:
    """Std of mean ax across repeated runs (lower is better)."""
    means = [float(np.mean(r)) for r in ax_runs if len(r)]
    if len(means) < 2:
        return 0.0
    return float(np.std(means))


def evaluate_recup_handover(
    t: np.ndarray,
    ax: np.ndarray,
    motor_brake_nm: np.ndarray,
    hyd_brake: np.ndarray,
    brake_req: np.ndarray | None = None,
    ax_mode_b: np.ndarray | None = None,
    ax_repeat_runs: list[np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
) -> DriveRating:
    t = np.asarray(t, dtype=float)
    ax = np.asarray(ax, dtype=float)
    motor_brake_nm = np.asarray(motor_brake_nm, dtype=float)
    hyd_brake = np.asarray(hyd_brake, dtype=float)
    dt = float(np.median(np.diff(t)))
    if brake_req is None:
        brake_req = np.clip(-ax, 0.0, None)
    else:
        brake_req = np.asarray(brake_req, dtype=float)

    ev = detect_decel_event(ax, brake_req, dt)
    notes: list[str] = []
    if ev is None:
        notes.append("no sustained decel event; using full window")
        i0, i1 = 0, len(ax)
    else:
        i0, i1 = ev

    util = _regen_utilization(motor_brake_nm, hyd_brake, i0, i1)
    bump = _handover_bump(ax, hyd_brake, i0, i1, dt)

    # higher utilization better (to a point)
    r_util = piecewise_linear_score(
        util,
        [(0.0, 1.0), (0.3, 4.0), (0.6, 7.0), (0.8, 9.0), (0.95, 10.0)],
    )
    # lower bump better
    r_bump = piecewise_linear_score(
        bump,
        [(1.0, 10.0), (3.0, 8.0), (6.0, 5.0), (10.0, 2.0), (20.0, 1.0)],
    )

    locals_: dict[str, float] = {
        "regen_utilization": r_util,
        "handover_bump": r_bump,
    }
    metrics = {
        "regen_utilization": util,
        "handover_jerk_peak": bump,
        "event_start": float(i0),
        "event_end": float(i1),
    }

    w = {
        "regen_utilization": 0.35,
        "handover_bump": 0.45,
        "mode_difference": 0.10,
        "reproducibility": 0.10,
    }
    if weights:
        w.update(weights)

    if ax_mode_b is not None:
        md = _mode_difference(ax[i0:i1], np.asarray(ax_mode_b, dtype=float))
        # want noticeable but not huge difference between modes
        r_mode = piecewise_linear_score(
            abs(md - 0.6),
            [(0.0, 10.0), (0.2, 8.0), (0.5, 5.0), (1.0, 2.0), (1.5, 1.0)],
        )
        locals_["mode_difference"] = r_mode
        metrics["mode_ax_diff"] = md
    else:
        notes.append("mode_difference skipped (ax_mode_b not provided)")
        w["mode_difference"] = 0.0

    if ax_repeat_runs:
        rep = _reproducibility(ax_repeat_runs)
        r_rep = piecewise_linear_score(
            rep,
            [(0.02, 10.0), (0.05, 8.0), (0.10, 5.0), (0.20, 2.0), (0.40, 1.0)],
        )
        locals_["reproducibility"] = r_rep
        metrics["ax_mean_std"] = rep
    else:
        notes.append("reproducibility skipped (no repeat runs)")
        w["reproducibility"] = 0.0

    # renormalize remaining positive weights inside combine
    overall = combine_local_ratings(locals_, w)
    return DriveRating(overall=overall, locals=locals_, metrics=metrics, notes=notes)
