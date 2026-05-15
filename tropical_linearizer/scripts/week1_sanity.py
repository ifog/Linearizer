"""Week-1 sanity script (blueprint §17): tropical Linearizer with identity g
should match Floyd-Warshall shortest path on a small graph.

We construct a small min-plus shortest-path problem, embed it as max-plus
(negate weights), build a tropical Linearizer with g = identity and the
tropical core being a Kleene star with hand-set weights, then verify the
one-shot output matches NetworkX's Floyd-Warshall.

Usage:
    PYTHONPATH=/home/nvidia/Linearizer \
        /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/week1_sanity.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tropical_linearizer.tropical.kleene import kleene_floyd_warshall
from tropical_linearizer.tropical.ops import NEG_INF, trop_matvec
from tropical_linearizer.utils.networkx_ref import all_pairs_shortest_path


class Identity(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, z):
        return z


def main() -> None:
    torch.manual_seed(0)
    n = 6

    # Build a random non-negative-weight directed graph in min-plus form.
    rng = np.random.default_rng(0)
    W = rng.uniform(0.1, 2.0, size=(n, n))
    mask = rng.random((n, n)) < 0.3
    W[mask] = np.inf
    np.fill_diagonal(W, 0.0)

    print("Edge-weight matrix W (min-plus, inf = no edge):")
    print(np.round(W, 2))

    # Reference: all-pairs shortest paths via Floyd-Warshall.
    D = all_pairs_shortest_path(W)
    print("\nFloyd-Warshall all-pairs shortest paths D:")
    print(np.round(D, 2))

    # Translate to max-plus: P_{ij} = -W_{ij}.  inf entries -> NEG_INF.
    W_neg = -W
    W_neg[np.isneginf(W_neg)] = NEG_INF
    P = torch.from_numpy(W_neg).float()

    # Compute A^* via our differentiable Kleene star.
    A_star = kleene_floyd_warshall(P)
    D_max = -A_star.numpy()  # back to min-plus

    print("\nMax-plus Kleene star  =>  shortest paths (after negation):")
    print(np.round(D_max, 2))

    # Compare in finite entries.
    finite = np.isfinite(D)
    diff = np.max(np.abs(D[finite] - D_max[finite]))
    print(f"\nMax abs diff vs reference (finite entries): {diff:.2e}")
    assert diff < 1e-3, "Sanity check FAILED"

    # Now exercise the tropical-Linearizer composition: apply A*^T to a source
    # indicator vector. The matvec picks column src of A*^T, i.e. row src of
    # A*, which is the SSSP-from-src row.
    src = 0
    x = torch.full((n,), NEG_INF, dtype=torch.float32)
    x[src] = 0.0  # max-plus indicator for source `src`

    y = trop_matvec(A_star.T, x)         # row `src` of A* in max-plus
    sssp = -y.numpy()                    # min-plus shortest paths from src
    ref_row = D[src]
    finite = np.isfinite(ref_row)
    diff2 = float(np.max(np.abs(ref_row[finite] - sssp[finite])))
    print(f"\nSSSP from node {src}:")
    print("  ours:", np.round(sssp, 2))
    print("  ref :", np.round(ref_row, 2))
    print(f"  max abs diff = {diff2:.2e}")
    assert diff2 < 1e-3
    print("\nWeek-1 sanity check PASSED.")


if __name__ == "__main__":
    main()
