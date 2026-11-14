#!/usr/bin/env python3
"""Synthetic demos for the four approximate algorithms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avl_drive_approx import (  # noqa: E402
    evaluate_acc_cut_in,
    evaluate_recup_handover,
    evaluate_ride_vertical,
    evaluate_tip_in,
)


def demo_tip_in() -> dict:
    fs = 100.0
    t = np.arange(0, 3.0, 1 / fs)
    pedal = np.zeros_like(t)
    pedal[t >= 0.5] = 40.0
    # delayed, slightly oscillatory ax response
    ax = np.zeros_like(t)
    mask = t >= 0.72
    tau = t[mask] - 0.72
    ax[mask] = 2.2 * (1 - np.exp(-tau / 0.18)) + 0.35 * np.exp(-tau / 0.4) * np.sin(
        2 * np.pi * 2.2 * tau
    )
    return evaluate_tip_in(t, pedal, ax).as_dict()


def demo_recup() -> dict:
    fs = 100.0
    t = np.arange(0, 4.0, 1 / fs)
    ax = np.where(t >= 0.8, -1.2, 0.0)
    # regen first, then hydraulic rises → small bump
    motor = np.where((t >= 0.8) & (t < 2.2), 800.0, np.where(t >= 2.2, 200.0, 0.0))
    hyd = np.where(t >= 2.2, 500.0, 0.0)
    # inject a jerk bump at handover
    i = int(2.2 * fs)
    ax[i : i + 5] -= np.linspace(0, 0.8, 5)
    ax[i + 5 : i + 10] += np.linspace(0.8, 0, 5)
    mode_b = ax - 0.55
    return evaluate_recup_handover(
        t, ax, motor, hyd, ax_mode_b=mode_b, ax_repeat_runs=[ax, ax + 0.03, ax - 0.02]
    ).as_dict()


def demo_acc_cut_in() -> dict:
    fs = 50.0
    t = np.arange(0, 8.0, 1 / fs)
    speed = np.full_like(t, 25.0)  # m/s
    distance = np.full_like(t, 40.0)
    # cut-in at 2.0 s: distance jumps down
    distance[t >= 2.0] = 18.0
    distance[t >= 2.0] += np.minimum(12.0, 0.8 * (t[t >= 2.0] - 2.0) * 25.0 / 10.0)
    ax = np.zeros_like(t)
    # delayed braking response
    mask = t >= 2.7
    ax[mask] = -2.0 * (1 - np.exp(-(t[mask] - 2.7) / 0.35))
    speed = speed + np.cumsum(ax) / fs
    return evaluate_acc_cut_in(t, distance, speed, ax).as_dict()


def demo_ride() -> dict:
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)
    az = 0.05 * np.random.default_rng(0).normal(size=t.shape)
    # bump around 0.6 s
    bump = np.exp(-((t - 0.6) ** 2) / (2 * 0.02**2))
    az += 3.0 * bump - 1.2 * np.exp(-((t - 0.68) ** 2) / (2 * 0.03**2))
    ax = 0.2 * bump
    return evaluate_ride_vertical(t, az, ax=ax).as_dict()


def main() -> None:
    out = {
        "tip_in": demo_tip_in(),
        "recup_handover": demo_recup(),
        "acc_cut_in": demo_acc_cut_in(),
        "ride_vertical": demo_ride(),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
