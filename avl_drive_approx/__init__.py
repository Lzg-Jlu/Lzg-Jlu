"""AVL-DRIVE style open approximate algorithms for methodology benchmarking.

These implementations intentionally approximate publicly described methodology
(trigger → physical features → transfer function / fuzzy-like scoring → DR 1–10).
They are NOT a reverse-engineering of proprietary AVL-DRIVE code or weights.
"""

from .maneuvers.tip_in import evaluate_tip_in
from .maneuvers.recup_handover import evaluate_recup_handover
from .maneuvers.acc_cut_in import evaluate_acc_cut_in
from .maneuvers.ride_vertical import evaluate_ride_vertical

__all__ = [
    "evaluate_tip_in",
    "evaluate_recup_handover",
    "evaluate_acc_cut_in",
    "evaluate_ride_vertical",
]

__version__ = "0.1.0"
