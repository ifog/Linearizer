"""Tropical core operator A.

The tropical Linearizer is f(x) = g_y^{-1}(A ⊗ g_x(x)). A lives in
R_max^{m x n}. We provide three parametrisations (blueprint §4.2):

  Regime A (DenseTropicalCore):   dense learnable matrix, no constraints.
  Regime B (HyperTropicalCore):   A is produced by a hypernetwork.
  Regime C (KleeneTropicalCore):  A := P^* (Kleene star), tropically
                                  idempotent by construction.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn

from .kleene import kleene_floyd_warshall, kleene_squaring
from .ops import soft_trop_matvec, st_trop_matvec, trop_matvec


class TropicalCore(nn.Module):
    """Abstract base: maps a tropical latent g(x) ∈ R^n → A ⊗ g(x) ∈ R^m."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def get_A(self, **kwargs) -> torch.Tensor:
        """Return the current tropical matrix A of shape (..., out_dim, in_dim).

        May be batch-dependent for Regime B (hypernetwork-produced A).
        """

    def forward(
        self,
        z: torch.Tensor,
        *,
        mode: str = "hard",
        beta: float | None = None,
        A: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply A ⊗ z. `A` may be passed in to avoid recomputation."""
        if A is None:
            A = self.get_A(z=z, **kwargs)
        if mode == "hard":
            return trop_matvec(A, z)
        if mode == "soft":
            assert beta is not None, "Soft mode requires beta"
            return soft_trop_matvec(A, z, beta=beta)
        if mode == "st":
            assert beta is not None, "Straight-through mode requires beta"
            return st_trop_matvec(A, z, beta=beta)
        raise ValueError(f"Unknown mode: {mode!r}")


class DenseTropicalCore(TropicalCore):
    """Regime A: dense unconstrained tropical matrix learned directly."""

    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 0.1):
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.A_raw = nn.Parameter(init_scale * torch.randn(out_dim, in_dim))

    def get_A(self, **kwargs) -> torch.Tensor:
        return self.A_raw


class HyperTropicalCore(TropicalCore):
    """Regime B: A is produced by a hypernetwork conditioned on context.

    Use when the operator must depend on input (e.g. graph structure / edge
    weights). Pass `context` through forward(..., context=...).
    """

    def __init__(self, in_dim: int, out_dim: int, hyper: nn.Module):
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.hyper = hyper

    def get_A(self, context: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if context is None:
            raise ValueError("HyperTropicalCore.get_A requires `context`")
        out = self.hyper(context)
        # Expect hyper to output a flat (..., out_dim * in_dim) tensor.
        return out.reshape(*out.shape[:-1], self.out_dim, self.in_dim)


class KleeneTropicalCore(TropicalCore):
    """Regime C: A := P^* (Kleene star). Tropically idempotent by construction.

    A Kleene star A^* is defined and idempotent (A^* ⊗ A^* = A^*) exactly when
    P has no positive-weight cycles. Enforcing "no positive cycle" exactly is
    NP-hard, but the simplest *sufficient* condition is P_{ij} ≤ 0 for all i,j
    (no positive edges -> no positive cycles). We use this by default via
    -softplus on the raw parameter. The trade-off: this rules out positive
    edges entirely, which is fine for shortest-path-style applications
    (min-plus equivalent: all weights are non-negative).

    For settings that need positive entries (e.g. interpretability with
    tropical polytope vertices), set `constraint="diag_only"` and accept that
    A^* may not be exactly idempotent unless P happens to have no positive
    cycles after training.
    """

    def __init__(
        self,
        n: int,
        *,
        method: str = "squaring",  # "squaring" or "floyd"
        kleene_iter: int | None = None,
        constraint: str = "all_non_positive",  # "all_non_positive" | "diag_only"
        init_scale: float = 0.1,
    ):
        super().__init__(in_dim=n, out_dim=n)
        self.n = n
        self.method = method
        self.kleene_iter = kleene_iter
        self.constraint = constraint
        self.P_raw = nn.Parameter(init_scale * torch.randn(n, n))

    def _P(self) -> torch.Tensor:
        if self.constraint == "all_non_positive":
            # P_{ij} = -softplus(raw_{ij}) ∈ (-inf, 0]. Sufficient for the
            # no-positive-cycle condition required for A^* to be idempotent.
            return -torch.nn.functional.softplus(self.P_raw)
        if self.constraint == "diag_only":
            P = self.P_raw
            diag = -torch.nn.functional.softplus(torch.diagonal(P, 0))
            P = P - torch.diag(torch.diagonal(P, 0)) + torch.diag(diag)
            return P
        raise ValueError(f"Unknown constraint: {self.constraint!r}")

    def get_A(self, *, mode: str = "hard", beta: float | None = None, **kwargs) -> torch.Tensor:
        P = self._P()
        if self.method == "squaring":
            # When all entries are non-positive, n-1 doublings is overkill but
            # safe; we use the standard ceil(log2(n)) default in kleene_squaring.
            return kleene_squaring(P, n_iter=self.kleene_iter, mode=mode, beta=beta)
        if self.method == "floyd":
            # Floyd-Warshall is hard-max only.
            return kleene_floyd_warshall(P)
        raise ValueError(f"Unknown method: {self.method!r}")
