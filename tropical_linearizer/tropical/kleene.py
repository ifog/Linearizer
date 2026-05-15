"""Kleene-star computations for max-plus matrices.

For a matrix A in R_max^{n x n}, the Kleene star is
    A^{*,N} = I ⊕ A ⊕ A^{⊗2} ⊕ ... ⊕ A^{⊗N}.
If A has no positive-weight cycles (in min-plus framing: no negative cycles),
A^{*,N} stabilises at N = n-1, and we write A^* := A^{*,n-1}.
Algorithmically: A^* = (I ⊕ A)^{⊗(n-1)}, and (I ⊕ A)^{⊗k} = I ⊕ A^{⊗1} ⊕ ... ⊕ A^{⊗k}.

Three implementations:
    1. kleene_squaring: iterated squaring, O(log n) tropical matmuls.
    2. kleene_floyd_warshall: standard Floyd-Warshall, O(n^3) but lower memory.
    3. kleene_power_iter: iterate until convergence (for analysis only).
"""

from __future__ import annotations

import torch

from .ops import (
    NEG_INF,
    log2_ceil,
    soft_trop_matmul,
    st_trop_matmul,
    trop_add,
    trop_identity,
    trop_matmul,
)


def _expand_identity_like(A: torch.Tensor) -> torch.Tensor:
    """Build a tropical identity matching A's batch + dtype + device."""
    n = A.shape[-1]
    assert A.shape[-2] == n, f"Kleene requires a square matrix, got {tuple(A.shape)}"
    I = trop_identity(n, dtype=A.dtype, device=A.device)
    # Expand to A's batch shape.
    if A.dim() > 2:
        I = I.expand(*A.shape[:-2], n, n).clone()
    return I


def _kleene_matmul(A: torch.Tensor, B: torch.Tensor, beta: float | None, mode: str) -> torch.Tensor:
    if mode == "hard":
        return trop_matmul(A, B)
    if mode == "soft":
        assert beta is not None
        return soft_trop_matmul(A, B, beta=beta)
    if mode == "st":
        assert beta is not None
        return st_trop_matmul(A, B, beta=beta)
    raise ValueError(f"Unknown mode: {mode!r}")


def kleene_squaring(
    A: torch.Tensor,
    n_iter: int | None = None,
    *,
    mode: str = "hard",
    beta: float | None = None,
) -> torch.Tensor:
    """Compute A^* via iterated squaring of (I ⊕ A).

    (I ⊕ A)^{⊗(2^k)} = I ⊕ A ⊕ ... ⊕ A^{⊗(2^k)}, so after k = ceil(log2(n))
    squarings we have A^{*,2^k} which equals A^* whenever 2^k >= n-1 and A
    has no positive cycles. If A has positive cycles, this still computes
    A^{*,N} for N = 2^{n_iter}, which is well-defined.

    mode = "hard" uses exact max, "soft" uses logsumexp/beta, "st" uses
    straight-through (hard forward, soft backward).
    """
    n = A.shape[-1]
    n_iter = n_iter if n_iter is not None else log2_ceil(n)
    I = _expand_identity_like(A)
    M = trop_add(I, A)  # I ⊕ A
    for _ in range(n_iter):
        M = _kleene_matmul(M, M, beta=beta, mode=mode)
    return M


def kleene_floyd_warshall(A: torch.Tensor) -> torch.Tensor:
    """Compute A^* via Floyd-Warshall on (I ⊕ A).

    For each intermediate node k, update M_{ij} = max(M_{ij}, M_{ik} + M_{kj}).
    O(n^3) work, O(n^2) memory. Differentiable through torch.maximum.

    Note: hard-max only (no soft variant) - intended for inference / large n.
    """
    n = A.shape[-1]
    I = _expand_identity_like(A)
    M = trop_add(I, A)
    for k in range(n):
        col_k = M[..., :, k : k + 1]    # (..., n, 1)
        row_k = M[..., k : k + 1, :]    # (..., 1, n)
        M = torch.maximum(M, col_k + row_k)
    return M


def kleene_power_iter(
    A: torch.Tensor,
    max_iter: int = 256,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    """Iterate M_{k+1} = (I ⊕ A) ⊗ M_k = I ⊕ A ⊕ A^{⊗2} ⊕ ... until convergence.

    Returns (A^*, num_iters). Hard-max only.
    """
    n = A.shape[-1]
    I = _expand_identity_like(A)
    IA = trop_add(I, A)
    M = I.clone()
    for it in range(1, max_iter + 1):
        M_new = trop_matmul(IA, M)
        if torch.max(torch.abs(M_new - M)).item() < tol:
            return M_new, it
        M = M_new
    return M, max_iter
