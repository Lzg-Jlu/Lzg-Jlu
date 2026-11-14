from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .transfer import soft_clamp


@dataclass
class DriveRating:
    """Container for overall DR and transparent local facet scores."""

    overall: float
    locals: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "locals": dict(self.locals),
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


def combine_local_ratings(
    local_scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted average of local ratings, clamped to [1, 10]."""
    if not local_scores:
        raise ValueError("local_scores is empty")

    if weights is None:
        weights = {k: 1.0 for k in local_scores}

    num = 0.0
    den = 0.0
    for key, score in local_scores.items():
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        num += w * float(score)
        den += w
    if den <= 0:
        raise ValueError("sum of weights must be > 0")
    return soft_clamp(num / den, 1.0, 10.0)
