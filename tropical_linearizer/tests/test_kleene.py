"""Kleene star sanity tests.

The headline check: our differentiable Kleene star (max-plus on -W) reproduces
NetworkX-style all-pairs shortest paths (min-plus on W) up to sign.
"""

from __future__ import annotations

import numpy as np
import torch

from tropical_linearizer.tropical.kleene import (
    kleene_floyd_warshall,
    kleene_power_iter,
    kleene_squaring,
)
from tropical_linearizer.tropical.ops import NEG_INF, trop_matmul
from tropical_linearizer.utils.networkx_ref import all_pairs_shortest_path


def _random_weight_matrix(n: int, *, seed: int = 0, no_edge_prob: float = 0.3) -> np.ndarray:
    """Random non-negative edge weights; no_edge_prob fraction set to +inf."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 2.0, size=(n, n))
    mask = rng.random((n, n)) < no_edge_prob
    W[mask] = np.inf
    np.fill_diagonal(W, 0.0)
    return W


def test_kleene_squaring_idempotent() -> None:
    """A^* is tropically idempotent: A^* ⊗ A^* = A^*."""
    torch.manual_seed(0)
    n = 6
    P = -torch.abs(torch.randn(n, n))  # all non-positive => no positive cycles
    A_star = kleene_squaring(P)
    A_star_sq = trop_matmul(A_star, A_star)
    assert torch.allclose(A_star_sq, A_star, atol=1e-4), torch.max(torch.abs(A_star_sq - A_star))


def test_kleene_squaring_vs_floyd() -> None:
    torch.manual_seed(1)
    n = 8
    P = -torch.abs(torch.randn(n, n))
    sq = kleene_squaring(P)
    fw = kleene_floyd_warshall(P)
    assert torch.allclose(sq, fw, atol=1e-4), torch.max(torch.abs(sq - fw))


def test_kleene_vs_networkx_shortest_paths() -> None:
    """Convert min-plus shortest paths to max-plus and compare."""
    n = 6
    W = _random_weight_matrix(n, seed=42)
    D_min = all_pairs_shortest_path(W)  # min-plus reference

    # Translate to max-plus: tropical weight = -W (so max-plus longest path on
    # negated weights equals -shortest path). +inf -> -inf, become NEG_INF.
    W_neg = -W
    W_neg[np.isneginf(W_neg)] = NEG_INF
    P = torch.from_numpy(W_neg).float()

    A_star = kleene_floyd_warshall(P).numpy()
    D_max = -A_star  # back to min-plus

    # Compare only finite entries (where there's actually a path).
    finite = np.isfinite(D_min)
    diff = np.max(np.abs(D_min[finite] - D_max[finite]))
    assert diff < 1e-3, f"max-plus Kleene disagrees with min-plus SP: {diff}"


def test_kleene_power_iter_converges() -> None:
    torch.manual_seed(2)
    n = 5
    P = -torch.abs(torch.randn(n, n))
    A_star_pi, iters = kleene_power_iter(P, max_iter=50)
    A_star_sq = kleene_squaring(P)
    assert torch.allclose(A_star_pi, A_star_sq, atol=1e-4)
    assert iters < 50


def test_kleene_grad_flows() -> None:
    """Subgradients flow through the max in Kleene squaring."""
    torch.manual_seed(3)
    n = 4
    P = torch.randn(n, n, requires_grad=True)
    A_star = kleene_squaring(-torch.abs(P))  # negate for stability
    A_star.sum().backward()
    assert P.grad is not None
    assert torch.isfinite(P.grad).all()


if __name__ == "__main__":
    test_kleene_squaring_idempotent()
    test_kleene_squaring_vs_floyd()
    test_kleene_vs_networkx_shortest_paths()
    test_kleene_power_iter_converges()
    test_kleene_grad_flows()
    print("OK: Kleene tests passed.")
