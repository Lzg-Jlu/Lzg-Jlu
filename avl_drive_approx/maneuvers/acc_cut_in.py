"""ACC cut-in approximate algorithm (ADAS perceived safety / comfort).

Pipeline:
  1) detect cut-in from sudden drop in lead distance
  2) extract response delay, peak |ax|/|jerk|, min time-headway, overshoot
  3) map metrics → local DR → weighted overall

Scenario classification here is rule-based (public methodology style).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.filters import derivative, moving_average
from ..core.rating import DriveRating, combine_local_ratings
from ..core.transfer import piecewise_linear_score


def detect_cut_in(
    distance_m: np.ndarray,
    dt: float,
    drop_m: float = 8.0,
    window_s: float = 0.6,
) -> Optional[int]:
    """Cut-in onset: lead distance drops sharply within a short window."""
    d = moving_average(np.asarray(distance_m, dtype=float), max(1, int(0.05 / dt)))
    w = max(2, int(window_s / dt))
    for i in range(w, len(d)):
        if d[i - w] - d[i] >= drop_m:
            return i - w
    return None


def _response_delay(
    distance_m: np.ndarray,
    ax: np.ndarray,
    i_cut: int,
    dt: float,
    ax_thr: float = -0.4,
) -> float:
    """Time from cut-in cue to first meaningful ego deceleration."""
    for i in range(i_cut, len(ax)):
        if ax[i] <= ax_thr:
            return (i - i_cut) * dt
    return (len(ax) - i_cut) * dt


def _min_time_headway(distance_m: np.ndarray, speed_mps: np.ndarray, i0: int, i1: int) -> float:
    seg_d = np.asarray(distance_m[i0:i1], dtype=float)
    seg_v = np.maximum(np.asarray(speed_mps[i0:i1], dtype=float), 0.1)
    thw = seg_d / seg_v
    return float(np.min(thw))


def evaluate_acc_cut_in(
    t: np.ndarray,
    distance_m: np.ndarray,
    speed_mps: np.ndarray,
    ax: np.ndarray,
    weights: dict[str, float] | None = None,
) -> DriveRating:
    t = np.asarray(t, dtype=float)
    distance_m = np.asarray(distance_m, dtype=float)
    speed_mps = np.asarray(speed_mps, dtype=float)
    ax = np.asarray(ax, dtype=float)
    dt = float(np.median(np.diff(t)))

    notes: list[str] = []
    i_cut = detect_cut_in(distance_m, dt)
    if i_cut is None:
        notes.append("no cut-in trigger; evaluating from start")
        i_cut = 0

    i1 = min(len(ax), i_cut + int(5.0 / dt))
    delay = _response_delay(distance_m, ax, i_cut, dt)
    jerk = derivative(ax, dt)
    peak_ax = float(np.min(ax[i_cut:i1]))  # most negative
    peak_jerk = float(np.min(jerk[i_cut:i1]))  # most negative
    min_thw = _min_time_headway(distance_m, speed_mps, i_cut, i1)

    # speed undershoot vs pre-event speed (comfort)
    v0 = float(np.mean(speed_mps[max(0, i_cut - int(0.5 / dt)) : i_cut + 1]))
    v_min = float(np.min(speed_mps[i_cut:i1]))
    undershoot = max(0.0, v0 - v_min)

    r_delay = piecewise_linear_score(
        delay,
        [(0.2, 10.0), (0.5, 8.0), (1.0, 5.0), (1.8, 2.0), (3.0, 1.0)],
    )
    # harsh braking lowers score
    r_ax = piecewise_linear_score(
        abs(peak_ax),
        [(0.5, 10.0), (1.5, 8.0), (2.5, 5.0), (4.0, 2.0), (6.0, 1.0)],
    )
    r_jerk = piecewise_linear_score(
        abs(peak_jerk),
        [(1.0, 10.0), (3.0, 8.0), (6.0, 5.0), (10.0, 2.0), (15.0, 1.0)],
    )
    # very small THW is unsafe feeling
    r_thw = piecewise_linear_score(
        min_thw,
        [(0.3, 1.0), (0.6, 3.0), (1.0, 6.0), (1.5, 8.5), (2.5, 10.0)],
    )
    r_under = piecewise_linear_score(
        undershoot,
        [(0.5, 10.0), (1.5, 8.0), (3.0, 5.0), (5.0, 2.0), (8.0, 1.0)],
    )

    locals_ = {
        "response_delay": r_delay,
        "brake_harshness": r_ax,
        "jerk": r_jerk,
        "min_time_headway": r_thw,
        "speed_undershoot": r_under,
    }
    w = weights or {
        "response_delay": 0.20,
        "brake_harshness": 0.20,
        "jerk": 0.25,
        "min_time_headway": 0.25,
        "speed_undershoot": 0.10,
    }
    overall = combine_local_ratings(locals_, w)
    return DriveRating(
        overall=overall,
        locals=locals_,
        metrics={
            "delay_s": delay,
            "peak_ax_mps2": peak_ax,
            "peak_jerk_mps3": peak_jerk,
            "min_thw_s": min_thw,
            "speed_undershoot_mps": undershoot,
            "cut_in_index": float(i_cut),
        },
        notes=notes,
    )
