"""Invertible flow over vectors.

The image-based InvUnet in ../../one_step/modules/invertable_network.py is
overkill for our first tropical-Linearizer experiments (CLRS, T-IGN on small
data), and the tropical core wants a vector latent anyway. This module
implements a RealNVP / Glow-style flow that operates directly on vectors.

Architecture per blueprint §4.1:
    g = [VectorCouplingBlock] * K
    each block: ActNorm -> Inv1x1 (full-rank linear mix) -> Affine coupling

forward() and inverse() are exact (no Jacobian determinant needed - we
transport algebra, not density).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActNorm1d(nn.Module):
    """Per-feature affine normalisation with data-dependent initialisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))
        self.eps = eps

    @torch.no_grad()
    def _init(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=0)
        std = x.std(dim=0) + self.eps
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized) and self.training:
            self._init(x)
        return (x + self.bias) * torch.exp(self.log_scale)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.log_scale) - self.bias


class InvLinear(nn.Module):
    """Invertible full-rank linear mix via PLU parameterization (Glow's 1x1 conv).

    W = P @ L @ U where
        P       fixed random permutation matrix (registered as a buffer)
        L       lower triangular with 1s on the diagonal
        U       upper triangular; |U_ii| = exp(log_diag_i) bounded away from 0
                via softplus on the raw parameter so the matrix is invertible
                throughout training.

    This is the standard Glow-style guarantee (Kingma-Dhariwal 2018, §3.2):
    parameterising W via PLU prevents the matrix from drifting to singular,
    which can otherwise happen with the naive `W = nn.Parameter(...)` form
    used in early versions of this module.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Random orthogonal initial matrix -> compute its PLU once and freeze
        # P, then start L and U at that LU.
        W = torch.linalg.qr(torch.randn(dim, dim)).Q
        P, L, U = torch.linalg.lu(W)
        # P is a permutation matrix; cache it as a buffer.
        self.register_buffer("P", P)
        # L has 1s on the diagonal; we parameterise only its strictly-lower part.
        self.L_lower = nn.Parameter(L.clone())          # full matrix, masked in forward
        self.register_buffer("eye", torch.eye(dim))
        self.register_buffer("lower_mask", torch.tril(torch.ones(dim, dim), diagonal=-1))
        # U: split into diagonal (sign-fixed, magnitude via softplus on raw log) and strictly-upper.
        diag_U = torch.diagonal(U, 0)
        sign_U = torch.sign(diag_U)
        sign_U[sign_U == 0] = 1.0
        # parameterise |U_ii| = softplus(raw) so it's > 0
        # softplus(raw) = log(1 + exp(raw)); we invert: raw = log(exp(|d|) - 1).
        abs_d = diag_U.abs().clamp_min(1e-3)
        raw_diag = torch.log(torch.expm1(abs_d).clamp_min(1e-6))
        self.U_log_diag = nn.Parameter(raw_diag)
        self.register_buffer("U_sign", sign_U)
        self.U_upper = nn.Parameter(U.clone())          # full matrix, masked in forward
        self.register_buffer("upper_mask", torch.triu(torch.ones(dim, dim), diagonal=1))

    def _build_L(self) -> torch.Tensor:
        return self.lower_mask * self.L_lower + self.eye

    def _build_U(self) -> torch.Tensor:
        # Floor the magnitude at 1e-2 so the triangular solve never explodes.
        diag = self.U_sign * (torch.nn.functional.softplus(self.U_log_diag) + 1e-2)
        return self.upper_mask * self.U_upper + torch.diag(diag)

    def _W(self) -> torch.Tensor:
        return self.P @ self._build_L() @ self._build_U()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._W().T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # W^-1 y = U^-1 L^-1 P^T y.  Both L and U are triangular -> use solver.
        # We treat (y) as a batch of right-hand-sides for a triangular solve.
        L = self._build_L()
        U = self._build_U()
        # P^T @ y^T  (shape (dim, B))
        rhs = (y @ self.P).T
        x = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
        x = torch.linalg.solve_triangular(U, x, upper=True)
        return x.T


class _CondMLP(nn.Module):
    """Conditioner MLP producing (shift, log_scale) for affine coupling."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_hidden: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 2 * out_dim)]
        # Zero-init the final layer so the coupling starts as identity.
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        shift, log_scale = h.chunk(2, dim=-1)
        return shift, log_scale


class AffineCoupling1d(nn.Module):
    """Affine coupling: split vector into (x1, x2), keep x2 untouched, map
    x1 -> (x1 + shift(x2)) * exp(clamp(log_scale(x2))).
    Exactly invertible; no Jacobian computation needed for our purposes.
    """

    def __init__(self, dim: int, hidden: int, n_hidden: int = 2, clamp: float = 5.0):
        super().__init__()
        d1 = dim // 2
        d2 = dim - d1
        self.d1 = d1
        self.d2 = d2
        self.clamp = clamp
        self.cond = _CondMLP(in_dim=d2, out_dim=d1, hidden=hidden, n_hidden=n_hidden)

    def _params(self, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_s = self.cond(x2)
        log_s = torch.clamp(log_s, -self.clamp, self.clamp)
        return shift, log_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : self.d1], x[..., self.d1 :]
        shift, log_s = self._params(x2)
        y1 = (x1 + shift) * torch.exp(log_s)
        return torch.cat([y1, x2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, x2 = y[..., : self.d1], y[..., self.d1 :]
        shift, log_s = self._params(x2)
        x1 = y1 * torch.exp(-log_s) - shift
        return torch.cat([x1, x2], dim=-1)


class _Permutation(nn.Module):
    """Fixed permutation (so successive couplings touch different coords)."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.perm)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y.index_select(-1, self.inv_perm)


class VectorCouplingBlock(nn.Module):
    """One block of the vector flow: ActNorm -> InvLinear -> Coupling -> Perm.

    The trailing permutation ensures the next block's coupling sees a
    different (x1, x2) split.
    """

    def __init__(self, dim: int, hidden: int, n_hidden: int = 2, clamp: float = 5.0, seed: int = 0):
        super().__init__()
        self.actnorm = ActNorm1d(dim)
        self.mix = InvLinear(dim)
        self.coupling = AffineCoupling1d(dim, hidden=hidden, n_hidden=n_hidden, clamp=clamp)
        self.perm = _Permutation(dim, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.actnorm(x)
        x = self.mix(x)
        x = self.coupling(x)
        x = self.perm(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = self.perm.inverse(y)
        y = self.coupling.inverse(y)
        y = self.mix.inverse(y)
        y = self.actnorm.inverse(y)
        return y


class VectorFlow(nn.Module):
    """K-block invertible flow on R^dim.

    forward(x): x -> z = g(x)
    inverse(z): z -> x = g^{-1}(z)

    Used as g_x and g_y in the tropical Linearizer.
    """

    def __init__(self, dim: int, n_blocks: int = 6, hidden: int = 128, n_hidden: int = 2, clamp: float = 5.0):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                VectorCouplingBlock(dim=dim, hidden=hidden, n_hidden=n_hidden, clamp=clamp, seed=i)
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for blk in reversed(self.blocks):
            z = blk.inverse(z)
        return z
