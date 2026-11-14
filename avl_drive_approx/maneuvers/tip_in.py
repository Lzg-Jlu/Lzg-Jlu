"""Tip-in approximate algorithm (longitudinal drivability).

Public-method approximation of the classic Tip-in facet split:
  1) response delay
  2) acceleration rise rate
  3) post-stabilization hesitation / drop
  4) shuffle / oscillation energy

References (methodology only): SAE papers on AVL-DRIVE; Ricardo US7149607
facet decomposition (transfer-function idea, not AVL source).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.filters import derivative, moving_average
from ..core.rating import DriveRating, combine_local_ratings
from ..core.transfer import piecewise_linear_score


def detect_tip_in(
    pedal: np.ndarray,
    ax: np.ndarray,
    dt: float,
    pedal_step_min: float = 15.0,
    pedal_rate_min: float = 80.0,
) -> Optional[int]:
    """Return index of tip-in onset, or None.

    Trigger: fast pedal opening with sustained step amplitude.
    """
    pedal = np.asarray(pedal, dtype=float)
    dpedal = derivative(pedal, dt)
    for i in range(1, len(pedal) - 5):
        if dpedal[i] < pedal_rate_min:
            continue
        # require a clear step over the next 0.3 s
        j = min(len(pedal) - 1, i + int(0.3 / dt))
        if pedal[j] - pedal[i - 1] >= pedal_step_min:
            return i
    return None


def _response_delay(pedal: np.ndarray, ax: np.ndarray, i0: int, dt: float) -> float:
    """Time from pedal onset to ax rising above a small threshold."""
    ax_f = moving_average(ax, max(3, int(0.05 / dt)))
    base = np.mean(ax_f[max(0, i0 - int(0.2 / dt)) : i0])
    thr = base + 0.15  # m/s^2
    for i in range(i0, len(ax_f)):
        if ax_f[i] >= thr:
            return (i - i0) * dt
    return (len(ax_f) - i0) * dt


def _rise_rate(ax: np.ndarray, i0: int, dt: float, win_s: float = 0.4) -> float:
    """Average ax rise over win_s after first reaction."""
    n = max(2, int(win_s / dt))
    i1 = min(len(ax) - 1, i0 + n)
    return float((ax[i1] - ax[i0]) / max((i1 - i0) * dt, 1e-6))


def _hesitation_drop(ax: np.ndarray, i0: int, dt: float) -> float:
    """Max drop after pedal is roughly stable (hesitation / hole)."""
    # look 0.3–1.2 s after onset
    a = i0 + int(0.3 / dt)
    b = min(len(ax) - 1, i0 + int(1.2 / dt))
    if b <= a + 2:
        return 0.0
    seg = ax[a:b]
    peak = np.max(seg)
    trough = np.min(seg)
    # only count drop after a local peak
    return float(max(0.0, peak - trough))


def _oscillation_metric(ax: np.ndarray, i0: int, dt: float) -> float:
    """Total absolute jerk energy as shuffle/thump proxy."""
    a = i0
    b = min(len(ax), i0 + int(1.5 / dt))
    jerk = derivative(ax[a:b], dt)
    # integral of |jerk| (m/s^2)
    # NumPy 2.0+: trapezoid; older: trapz
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.abs(jerk), dx=dt))


def evaluate_tip_in(
    t: np.ndarray,
    pedal: np.ndarray,
    ax: np.ndarray,
    weights: dict[str, float] | None = None,
) -> DriveRating:
    """Evaluate one Tip-in event (or auto-detect inside the window)."""
    t = np.asarray(t, dtype=float)
    pedal = np.asarray(pedal, dtype=float)
    ax = np.asarray(ax, dtype=float)
    dt = float(np.median(np.diff(t)))

    i0 = detect_tip_in(pedal, ax, dt)
    notes: list[str] = []
    if i0 is None:
        notes.append("no tip-in trigger found; scoring full window from t0")
        i0 = 0

    delay = _response_delay(pedal, ax, i0, dt)
    rise = _rise_rate(ax, i0, dt)
    drop = _hesitation_drop(ax, i0, dt)
    osc = _oscillation_metric(ax, i0, dt)

    # Transfer functions: tune knots to your brand DNA / class.
    # delay: lower better
    r_delay = piecewise_linear_score(
        delay,
        [(0.05, 10.0), (0.15, 8.0), (0.30, 5.0), (0.60, 2.0), (1.00, 1.0)],
    )
    # rise rate: moderate is good; too low = dull, too high = harsh (demo)
    r_rise = piecewise_linear_score(
        abs(rise - 2.5),  # distance from preferred ~2.5 m/s^3
        [(0.0, 10.0), (0.8, 8.0), (1.5, 5.0), (3.0, 2.0), (5.0, 1.0)],
    )
    r_hes = piecewise_linear_score(
        drop,
        [(0.00, 10.0), (0.20, 8.0), (0.50, 5.0), (1.00, 2.0), (1.50, 1.0)],
    )
    r_osc = piecewise_linear_score(
        osc,
        [(0.5, 10.0), (1.5, 8.0), (3.0, 5.0), (6.0, 2.0), (10.0, 1.0)],
    )

    locals_ = {
        "response_delay": r_delay,
        "rise_rate": r_rise,
        "hesitation": r_hes,
        "oscillation": r_osc,
    }
    w = weights or {
        "response_delay": 0.30,
        "rise_rate": 0.20,
        "hesitation": 0.20,
        "oscillation": 0.30,
    }
    overall = combine_local_ratings(locals_, w)
    return DriveRating(
        overall=overall,
        locals=locals_,
        metrics={
            "delay_s": delay,
            "rise_mps3": rise,
            "hesitation_drop_mps2": drop,
            "osc_jerk_int": osc,
            "trigger_index": float(i0),
        },
        notes=notes,
    )
