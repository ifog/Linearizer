"""NetworkX reference for shortest paths - used in tests to validate the Kleene
star against an independent implementation.

Convention: we work in max-plus internally. For shortest-path comparisons,
we negate weights so that max-plus on -W matches min-plus on W
(equivalently: shortest paths under W = longest paths under -W).
"""

from __future__ import annotations

import numpy as np


def all_pairs_shortest_path(W: np.ndarray) -> np.ndarray:
    """Standard Floyd-Warshall (min-plus) all-pairs shortest path on W.

    W[i,j] = edge weight from i to j; +inf for no edge. Returns D[i,j]
    = shortest-path length from i to j, with D[i,i] = 0 by convention.
    """
    n = W.shape[0]
    D = W.astype(np.float64).copy()
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, k : k + 1] + D[k : k + 1, :])
    return D


def kleene_from_weights(W: np.ndarray) -> np.ndarray:
    """Reference Kleene star for a max-plus weight matrix W (numpy)."""
    n = W.shape[0]
    NEG = -1e18
    A = np.full((n, n), NEG, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, W)  # I ⊕ W
    for k in range(n):
        col = A[:, k : k + 1]
        row = A[k : k + 1, :]
        A = np.maximum(A, col + row)
    return A
