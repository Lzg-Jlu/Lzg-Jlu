"""Vertical ride-comfort approximate algorithm.

Demo-quality pipeline:
  1) optional bump / continuous-road segmentation by az energy
  2) time-domain peak-to-peak and RMS
  3) band RMS as bounce (~1–3 Hz) and wheel-hop (~8–15 Hz) proxies
  4) transfer functions → local DR → overall

For production use, replace FFT band RMS with proper frequency weightings
(e.g. ISO 2631-style filters) and seat-rail transfer calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.filters import bandpass_rms, moving_average
from ..core.rating import DriveRating, combine_local_ratings
from ..core.transfer import piecewise_linear_score


def detect_impact_window(
    az: np.ndarray,
    dt: float,
    thr_mps2: float = 1.5,
    pre_s: float = 0.2,
    post_s: float = 1.0,
) -> Optional[tuple[int, int]]:
    """Find first high-|az| impact and return analysis window."""
    az = np.asarray(az, dtype=float)
    idx = np.where(np.abs(az) >= thr_mps2)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    a = max(0, i - int(pre_s / dt))
    b = min(len(az), i + int(post_s / dt))
    return a, b


def evaluate_ride_vertical(
    t: np.ndarray,
    az: np.ndarray,
    ax: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> DriveRating:
    t = np.asarray(t, dtype=float)
    az = np.asarray(az, dtype=float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    notes: list[str] = []
    win = detect_impact_window(az, dt)
    if win is None:
        notes.append("no discrete impact; scoring full window as continuous road")
        i0, i1 = 0, len(az)
    else:
        i0, i1 = win

    seg = az[i0:i1]
    seg_f = moving_average(seg, max(1, int(0.02 / dt)))
    p2p = float(np.max(seg_f) - np.min(seg_f))
    rms = float(np.sqrt(np.mean(seg_f**2)))
    bounce = bandpass_rms(seg, fs, 1.0, 3.0)
    hop = bandpass_rms(seg, fs, 8.0, 15.0)

    r_p2p = piecewise_linear_score(
        p2p,
        [(0.5, 10.0), (1.5, 8.0), (3.0, 5.0), (5.0, 2.0), (8.0, 1.0)],
    )
    r_rms = piecewise_linear_score(
        rms,
        [(0.2, 10.0), (0.5, 8.0), (1.0, 5.0), (1.8, 2.0), (3.0, 1.0)],
    )
    r_bounce = piecewise_linear_score(
        bounce,
        [(0.1, 10.0), (0.3, 8.0), (0.6, 5.0), (1.0, 2.0), (1.5, 1.0)],
    )
    r_hop = piecewise_linear_score(
        hop,
        [(0.05, 10.0), (0.15, 8.0), (0.30, 5.0), (0.60, 2.0), (1.00, 1.0)],
    )

    locals_ = {
        "peak_to_peak": r_p2p,
        "rms": r_rms,
        "bounce_band": r_bounce,
        "wheel_hop_band": r_hop,
    }
    metrics = {
        "p2p_mps2": p2p,
        "rms_mps2": rms,
        "bounce_rms": bounce,
        "hop_rms": hop,
        "window_start": float(i0),
        "window_end": float(i1),
    }

    w = {
        "peak_to_peak": 0.30,
        "rms": 0.25,
        "bounce_band": 0.25,
        "wheel_hop_band": 0.20,
        "longitudinal_coupling": 0.0,
    }
    if weights:
        w.update(weights)

    if ax is not None:
        ax = np.asarray(ax, dtype=float)
        # coupling penalty: simultaneous large ax and az feels worse
        ax_seg = ax[i0:i1]
        couple = float(np.mean(np.abs(ax_seg) * np.abs(seg)))
        r_c = piecewise_linear_score(
            couple,
            [(0.05, 10.0), (0.20, 8.0), (0.50, 5.0), (1.00, 2.0), (2.00, 1.0)],
        )
        locals_["longitudinal_coupling"] = r_c
        metrics["ax_az_coupling"] = couple
        if w.get("longitudinal_coupling", 0.0) <= 0:
            w["longitudinal_coupling"] = 0.15
    else:
        notes.append("longitudinal_coupling skipped (ax not provided)")

    overall = combine_local_ratings(locals_, w)
    return DriveRating(overall=overall, locals=locals_, metrics=metrics, notes=notes)
