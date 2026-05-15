"""Training schedules - beta annealing for the soft tropical relaxation.

Per blueprint §4.5: start with small beta (soft max ~ smooth interpolation),
anneal to large beta (hard max). Linear in log-space matches the geometric
spacing typically used for temperature schedules.
"""

from __future__ import annotations


def beta_schedule_log(step: int, total: int, beta0: float = 1.0, beta1: float = 100.0) -> float:
    """Geometric (log-linear) interpolation between beta0 and beta1."""
    import math

    if total <= 0:
        return beta1
    t = min(max(step / total, 0.0), 1.0)
    return math.exp((1 - t) * math.log(beta0) + t * math.log(beta1))
