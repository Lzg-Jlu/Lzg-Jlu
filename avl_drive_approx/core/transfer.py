from __future__ import annotations

from typing import Sequence

import numpy as np


def soft_clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def piecewise_linear_score(
    metric: float,
    knots: Sequence[tuple[float, float]],
) -> float:
    """Map a physical metric to a local score via monotone piecewise-linear TF.

    knots: sequence of (metric_value, score) sorted by metric_value ascending.
    Typical pattern for "lower metric is better":
        [(0.0, 10.0), (0.2, 8.0), (0.5, 5.0), (1.0, 2.0)]
    """
    if len(knots) < 2:
        raise ValueError("need at least two knots")
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    if any(xs[i] > xs[i + 1] for i in range(len(xs) - 1)):
        raise ValueError("metric knots must be non-decreasing")

    if metric <= xs[0]:
        return float(ys[0])
    if metric >= xs[-1]:
        return float(ys[-1])

    return float(np.interp(metric, xs, ys))
