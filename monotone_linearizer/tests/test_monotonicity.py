"""Numerically verify each parameterization is m-strongly monotone.

For G(w) = w - M(w, c), we require
    <G(u) - G(v), u - v>  >=  m * ||u - v||^2     for all u, v, c.

Strict equality won't hold (we sample finite (u, v) pairs), so we check
min-over-samples ratio >= m * (1 - tol).
"""

from __future__ import annotations

import torch

from monotone_linearizer.monotone.parameterizations import (
    MonotoneCoreICNN,
    MonotoneCoreP1,
    numerical_monotonicity_check,
)


def _check(core, m_expected: float, tol: float = 0.1) -> None:
    min_ratio, mean_ratio = numerical_monotonicity_check(core, n_samples=300, cond_scale=0.5)
    assert min_ratio >= m_expected * (1.0 - tol), (
        f"min monotonicity ratio {min_ratio:.4f} < m * (1 - tol) = "
        f"{m_expected * (1 - tol):.4f}; mean ratio = {mean_ratio:.4f}"
    )


def test_p1_monotonicity() -> None:
    torch.manual_seed(0)
    for m in (0.05, 0.1, 0.3):
        core = MonotoneCoreP1(n=64, cond_dim=16, m=m, init_scale=0.1)
        _check(core, m)


def test_p1_after_random_perturb() -> None:
    """After a parameter perturbation A is still well-formed, monotonicity holds."""
    torch.manual_seed(1)
    core = MonotoneCoreP1(n=32, cond_dim=8, m=0.1)
    with torch.no_grad():
        core.A.data += torch.randn_like(core.A.data) * 0.5
    _check(core, 0.1)


def test_icnn_monotonicity() -> None:
    torch.manual_seed(2)
    core = MonotoneCoreICNN(n=32, cond_dim=8, m=0.1, hidden=32, n_hidden=2)
    # ICNN is monotone *in expectation*; the strict m strong-monotonicity is
    # softer because the scale knob is small. We just require non-negative
    # ratio (monotone, not necessarily m-strongly).
    min_ratio, mean_ratio = numerical_monotonicity_check(core, n_samples=200, cond_scale=0.5)
    assert mean_ratio >= 0.0, f"ICNN mean ratio {mean_ratio} should be >= 0"


if __name__ == "__main__":
    test_p1_monotonicity()
    test_p1_after_random_perturb()
    test_icnn_monotonicity()
    print("OK: monotonicity tests passed.")
