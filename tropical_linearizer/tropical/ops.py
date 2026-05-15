"""Core tropical (max-plus) operations.

We use the max-plus semiring (R ∪ {-inf}, max, +) with:
    a ⊕ b := max(a, b)
    a ⊗ b := a + b
    additive identity = -inf, multiplicative identity = 0

The shortest-path version (min-plus) is recovered by negating inputs and outputs.

Numerical convention: additive identity (-inf) is represented by a large negative
constant NEG_INF to avoid NaN propagation through autograd. Anything that started
out at -inf stays at -inf because max(NEG_INF + finite, anything_real) = anything_real
provided finite values fit in (-NEG_INF, +inf).
"""

from __future__ import annotations

import math

import torch

# Additive identity ("-infinity") for the max-plus semiring.
# Chosen small enough that NEG_INF + (typical finite value) is still tiny,
# but large enough to keep gradients finite. Do NOT use -inf directly:
# -inf + anything = -inf and -inf - (-inf) = nan, both of which corrupt autograd.
NEG_INF: float = -1.0e9


def trop_identity(n: int, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """Tropical identity matrix I: 0 on the diagonal, -inf off-diagonal."""
    I = torch.full((n, n), NEG_INF, dtype=dtype, device=device)
    I.fill_diagonal_(0.0)
    return I


def trop_zero(*shape: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Tropical zero tensor (additive identity), filled with NEG_INF."""
    return torch.full(shape, NEG_INF, dtype=dtype, device=device)


def trop_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tropical addition: elementwise max with broadcasting."""
    return torch.maximum(A, B)


def trop_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Tropical matrix-vector product.

    A: (..., m, n)
    x: (..., n)
    returns: (..., m) with (A ⊗ x)_i = max_j (A_{ij} + x_j)

    Broadcasts over batch dimensions.
    """
    # A[..., m, n] + x[..., 1, n] -> (..., m, n), then max over the last dim.
    return (A + x.unsqueeze(-2)).amax(dim=-1)


def trop_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tropical matrix-matrix product.

    A: (..., m, k)
    B: (..., k, n)
    returns: (..., m, n) with (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj}).
    """
    # A[..., m, k, 1] + B[..., 1, k, n] -> (..., m, k, n), max over k.
    return (A.unsqueeze(-1) + B.unsqueeze(-3)).amax(dim=-2)


def soft_trop_matvec(A: torch.Tensor, x: torch.Tensor, beta: float) -> torch.Tensor:
    """Soft tropical matvec via logsumexp / beta. Approaches trop_matvec as beta -> inf."""
    z = A + x.unsqueeze(-2)
    return torch.logsumexp(beta * z, dim=-1) / beta


def soft_trop_matmul(A: torch.Tensor, B: torch.Tensor, beta: float) -> torch.Tensor:
    """Soft tropical matmul via logsumexp / beta."""
    z = A.unsqueeze(-1) + B.unsqueeze(-3)
    return torch.logsumexp(beta * z, dim=-2) / beta


def st_max(z: torch.Tensor, beta: float, dim: int = -1) -> torch.Tensor:
    """Straight-through max: hard forward, soft (logsumexp/beta) backward."""
    hard = z.amax(dim=dim)
    soft = torch.logsumexp(beta * z, dim=dim) / beta
    return hard.detach() + (soft - soft.detach())


def st_trop_matvec(A: torch.Tensor, x: torch.Tensor, beta: float) -> torch.Tensor:
    """Straight-through tropical matvec."""
    z = A + x.unsqueeze(-2)
    return st_max(z, beta=beta, dim=-1)


def st_trop_matmul(A: torch.Tensor, B: torch.Tensor, beta: float) -> torch.Tensor:
    """Straight-through tropical matmul."""
    z = A.unsqueeze(-1) + B.unsqueeze(-3)
    return st_max(z, beta=beta, dim=-2)


def log2_ceil(n: int) -> int:
    """Smallest k such that 2**k >= n. Useful for Kleene squaring iteration count."""
    return max(1, int(math.ceil(math.log2(max(n, 2)))))
