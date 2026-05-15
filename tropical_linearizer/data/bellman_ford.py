"""Synthetic Bellman-Ford / SSSP dataset.

We generate random Erdős–Rényi directed graphs with non-negative edge weights
and compute all-pairs shortest paths via numpy Floyd-Warshall as ground truth.
This avoids any external dependency (no `clrs` install needed) and gives us
full control over n, density, weight range, and OOD-test sizes.

Conventions:
- All tensors are 3D where the leading dim is the batch.
- W (batch, n, n) holds edge weights in *min-plus* form: W[b, i, j] is the
  weight of the directed edge i -> j; `np.inf` (mapped to a large positive
  number for tensors) means "no edge". Self-loops are zero.
- D (batch, n, n) holds the ground-truth all-pairs shortest-path distances
  (D[b, i, j] = shortest path from i to j; np.inf if unreachable).
- For tensor-friendliness we pass `MISSING_EDGE = 1e9` instead of `np.inf`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

MISSING_EDGE = 1.0e9  # large positive number used in min-plus tensors.


@dataclass
class BellmanFordBatch:
    """A batch of shortest-path problems.

    All tensors are float32, on CPU by default. Shape conventions:

    - W:   (B, N, N) min-plus edge weights (self-loops = 0, missing = MISSING_EDGE)
    - D:   (B, N, N) min-plus ground-truth all-pairs shortest-path distances
    - mask:(B, N, N) 1.0 where D < MISSING_EDGE / 10, else 0.0 (reachability)
    """
    W: torch.Tensor
    D: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> "BellmanFordBatch":
        return BellmanFordBatch(W=self.W.to(device), D=self.D.to(device), mask=self.mask.to(device))


def _floyd_warshall_np(W: np.ndarray) -> np.ndarray:
    """Min-plus all-pairs shortest paths. Treats large values as +inf."""
    n = W.shape[-1]
    D = W.astype(np.float64).copy()
    # Replace MISSING_EDGE with np.inf for arithmetic correctness, then map back.
    D[D > MISSING_EDGE / 10] = np.inf
    np.fill_diagonal_view = None  # noqa
    # Per-batch fill_diagonal:
    if D.ndim == 2:
        np.fill_diagonal(D, 0.0)
    elif D.ndim == 3:
        for b in range(D.shape[0]):
            np.fill_diagonal(D[b], 0.0)
    else:
        raise ValueError(f"unexpected W ndim: {D.ndim}")
    for k in range(n):
        # D_{ij} = min(D_{ij}, D_{ik} + D_{kj}); broadcast over batch.
        col = D[..., :, k : k + 1]
        row = D[..., k : k + 1, :]
        D = np.minimum(D, col + row)
    # Map +inf back to MISSING_EDGE for tensor friendliness.
    D[np.isposinf(D)] = MISSING_EDGE
    return D.astype(np.float32)


def make_batch(
    batch_size: int,
    n: int,
    *,
    edge_prob: float = 0.5,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
    seed: Optional[int] = None,
    ensure_connected_source: bool = True,
) -> BellmanFordBatch:
    """Generate a batch of random ER graphs with non-negative weights and SSSP labels.

    edge_prob controls the (independent) probability that each directed edge
    (i, j) with i != j exists. Self-loops always present with weight 0.
    weight_low/high define the uniform edge-weight distribution.
    """
    rng = np.random.default_rng(seed)
    W = np.full((batch_size, n, n), MISSING_EDGE, dtype=np.float32)
    weights = rng.uniform(weight_low, weight_high, size=(batch_size, n, n)).astype(np.float32)
    edge_mask = rng.random((batch_size, n, n)) < edge_prob
    # No self-loops in the random part; we'll add zero self-loops below.
    for b in range(batch_size):
        np.fill_diagonal(edge_mask[b], False)
    W[edge_mask] = weights[edge_mask]
    for b in range(batch_size):
        np.fill_diagonal(W[b], 0.0)

    D = _floyd_warshall_np(W)
    # Reachability mask: 1 where D is finite (i.e. < MISSING_EDGE / 10).
    mask = (D < MISSING_EDGE / 10).astype(np.float32)

    if ensure_connected_source:
        # Re-roll any graph where vertex 0 cannot reach all other vertices,
        # since training is much smoother when the source has finite distance
        # to everyone. Bounded retries to avoid infinite loops on tiny edge_prob.
        for b in range(batch_size):
            for _ in range(8):
                if mask[b, 0].sum() == n:
                    break
                w_b = np.full((n, n), MISSING_EDGE, dtype=np.float32)
                em = rng.random((n, n)) < edge_prob
                np.fill_diagonal(em, False)
                ww = rng.uniform(weight_low, weight_high, size=(n, n)).astype(np.float32)
                w_b[em] = ww[em]
                np.fill_diagonal(w_b, 0.0)
                d_b = _floyd_warshall_np(w_b[None, ...])[0]
                m_b = (d_b < MISSING_EDGE / 10).astype(np.float32)
                W[b] = w_b
                D[b] = d_b
                mask[b] = m_b

    return BellmanFordBatch(
        W=torch.from_numpy(W),
        D=torch.from_numpy(D),
        mask=torch.from_numpy(mask),
    )


def to_max_plus_weights(W_minplus: torch.Tensor) -> torch.Tensor:
    """Convert a min-plus weight tensor to max-plus (-W with MISSING_EDGE -> NEG_INF).

    Used to feed a min-plus shortest-path problem to the max-plus Kleene
    machinery. Self-loops (W=0) become 0; missing edges (W=MISSING_EDGE)
    become a large negative number (the NEG_INF constant from tropical.ops).
    """
    from ..tropical.ops import NEG_INF

    P = -W_minplus.clone()
    # MISSING_EDGE -> -MISSING_EDGE, replace with NEG_INF.
    P[P < -MISSING_EDGE / 10] = NEG_INF
    return P
