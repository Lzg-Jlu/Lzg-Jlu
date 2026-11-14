from __future__ import annotations

import numpy as np

from avl_drive_approx import (
    evaluate_acc_cut_in,
    evaluate_recup_handover,
    evaluate_ride_vertical,
    evaluate_tip_in,
)
from avl_drive_approx.core.transfer import piecewise_linear_score
from avl_drive_approx.maneuvers.tip_in import detect_tip_in


def test_piecewise_linear_score_endpoints():
    s = piecewise_linear_score(0.0, [(0.0, 10.0), (1.0, 1.0)])
    assert s == 10.0
    s2 = piecewise_linear_score(1.0, [(0.0, 10.0), (1.0, 1.0)])
    assert s2 == 1.0


def test_tip_in_detect_and_score():
    fs = 100.0
    t = np.arange(0, 2.5, 1 / fs)
    pedal = np.zeros_like(t)
    pedal[t >= 0.4] = 35.0
    ax = np.zeros_like(t)
    ax[t >= 0.55] = 1.8 * (1 - np.exp(-(t[t >= 0.55] - 0.55) / 0.2))
    dt = 1 / fs
    i0 = detect_tip_in(pedal, ax, dt)
    assert i0 is not None
    rating = evaluate_tip_in(t, pedal, ax)
    assert 1.0 <= rating.overall <= 10.0
    assert "response_delay" in rating.locals


def test_recup_handover_scores():
    fs = 100.0
    t = np.arange(0, 3.0, 1 / fs)
    ax = np.where(t >= 0.5, -1.0, 0.0)
    motor = np.where(t >= 0.5, 700.0, 0.0)
    hyd = np.where(t >= 1.5, 400.0, 0.0)
    rating = evaluate_recup_handover(t, ax, motor, hyd)
    assert 1.0 <= rating.overall <= 10.0
    assert rating.metrics["regen_utilization"] > 0.0


def test_acc_cut_in_detect():
    fs = 50.0
    t = np.arange(0, 6.0, 1 / fs)
    speed = np.full_like(t, 20.0)
    distance = np.full_like(t, 35.0)
    distance[t >= 1.5] = 15.0
    ax = np.where(t >= 2.0, -1.5, 0.0)
    rating = evaluate_acc_cut_in(t, distance, speed, ax)
    assert 1.0 <= rating.overall <= 10.0
    assert "min_time_headway" in rating.locals


def test_ride_vertical():
    fs = 200.0
    t = np.arange(0, 1.5, 1 / fs)
    az = 0.02 * np.ones_like(t)
    az[(t > 0.4) & (t < 0.5)] = 2.5
    rating = evaluate_ride_vertical(t, az)
    assert 1.0 <= rating.overall <= 10.0
    assert rating.metrics["p2p_mps2"] > 0.0
