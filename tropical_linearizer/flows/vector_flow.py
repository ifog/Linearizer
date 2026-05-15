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
    """Invertible full-rank linear mix (Glow's 1x1 conv, but for vectors)."""

    def __init__(self, dim: int):
        super().__init__()
        # Orthogonal init guarantees the linear map starts invertible with
        # condition number ~1; weights drift during training but stay
        # well-conditioned in practice.
        W = torch.linalg.qr(torch.randn(dim, dim)).Q
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        W_inv = torch.linalg.inv(self.W)
        return y @ W_inv.T


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
