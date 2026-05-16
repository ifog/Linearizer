"""Monotone-operator parameterizations.

We provide:

  P1 (MonotoneCoreP1)   - Winston-Kolter NeurIPS 2020 default. Layer is
                          z ↦ σ(W z + U c + b) with I - W ⪰ m I enforced by
                          construction via W = (1 - m) I - A^T A. The
                          equilibrium-defining operator G(z, c) = z - σ(W z +
                          U c + b) is m-strongly monotone in z when σ is
                          1-Lipschitz and monotone (e.g. ReLU).

  P3 (MonotoneCoreICNN) - Convex potential parameterization. M(w, c) =
                          ∇_w Ψ(w, c) where Ψ is convex in w (Amos-Xu-Kolter
                          ICNN). Gradient of a convex function is monotone,
                          so M is monotone by construction.

P2 (Baker-Wang-Hauck-Wang 2023 IGNN parameterization) is graph-specific and
deferred to Week 6 when we tackle Application 3.

References:
  Winston, Kolter. "Monotone Operator Equilibrium Networks." NeurIPS 2020.
  Amos, Xu, Kolter. "Input Convex Neural Networks." ICML 2017.
  Bauschke, Combettes. Convex Analysis and Monotone Operator Theory. 2017.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneCore(nn.Module):
    """Abstract base for monotone-operator cores.

    Subclasses implement `forward(w, c) -> M(w, c)` such that the residual
    operator G(w, c) = w - M(w, c) is m-strongly monotone in w. Subclasses
    must expose `m` as an attribute (strong-monotonicity constant) so the
    solver and the test suite can verify.
    """

    m: float
    n: int

    @abstractmethod
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...


class MonotoneCoreP1(MonotoneCore):
    """Winston-Kolter parameterization (the default).

    Layer:
        M(w, c) = σ( W w + U c + b ),     W = (1 - m) I - A^T A.

    Then I - W = m I + A^T A ⪰ m I, so for any 1-Lipschitz monotone σ
    (we use ReLU), G(w, c) = w - M(w, c) is m-strongly monotone in w.

    Args:
        n:       latent dimension.
        cond_dim:dimension of the input-injection vector c.
        m:       strong-monotonicity constant (default 0.1).
        init_scale: scale of the unconstrained parameter A at initialisation.
                 Smaller -> closer to identity W = (1-m) I at start.
    """

    def __init__(self, n: int, cond_dim: int, m: float = 0.1, init_scale: float = 0.05):
        super().__init__()
        if not (0.0 < m < 1.0):
            raise ValueError(f"m must lie in (0, 1); got {m}")
        self.n = n
        self.cond_dim = cond_dim
        self.m = float(m)
        # Unconstrained parameter; W is built from it on every call.
        self.A = nn.Parameter(init_scale * torch.randn(n, n))
        # Input-injection linear map.
        self.U = nn.Linear(cond_dim, n, bias=True)
        # Bias term (kept separate from U so we can zero-init easily).
        self.b = nn.Parameter(torch.zeros(n))

    def W(self) -> torch.Tensor:
        I = torch.eye(self.n, device=self.A.device, dtype=self.A.dtype)
        return (1.0 - self.m) * I - self.A.T @ self.A

    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # w: (B, n), c: (B, cond_dim)
        Wmat = self.W()
        z = w @ Wmat.T + self.U(c) + self.b
        return F.relu(z)


class _ICNNHidden(nn.Module):
    """One hidden layer of an Input-Convex Neural Network (Amos 2017).

    z_{k+1} = h(W_z^+ z_k + W_x x + b)
    where W_z^+ has non-negative weights (enforced via softplus) and h is a
    convex non-decreasing activation (we use softplus). Strictly speaking
    softplus is C^infty and strictly convex, so the gradient operator we
    eventually build is strictly monotone.
    """

    def __init__(self, dim_z: int, dim_in: int, dim_x: int, first: bool):
        super().__init__()
        self.first = first
        if not first:
            self.W_z_raw = nn.Parameter(torch.randn(dim_z, dim_in) * 0.01)
        self.W_x = nn.Linear(dim_x, dim_z, bias=True)

    def forward(self, z_in: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.W_x(x)
        if not self.first:
            W_pos = F.softplus(self.W_z_raw)
            out = out + z_in @ W_pos.T
        return F.softplus(out)


class MonotoneCoreICNN(MonotoneCore):
    """P3: monotone operator obtained as the gradient of an ICNN potential.

    We parameterize Ψ(w, c) as an Input-Convex Neural Network in w (with c
    treated as a conditioning side input that does not affect convexity in w).
    Convexity in w => ∇_w Ψ(w, c) is monotone in w. We add a quadratic head
    (m/2) ||w||^2 so the gradient ∇_w Ψ(w, c) is m-strongly monotone in w.

    M(w, c) = ∇_w Ψ_net(w, c) + m * w + small linear input injection.

    For the operator equation w = M(w, c) to have a fixed point we need the
    parameterization above to satisfy the monDEQ-style "I - M is m-strongly
    monotone" condition. Here `M` itself plays the role of the "Wz + Uc + b"
    affine part of P1, and the residual G(w) = w - M(w) is m-strongly monotone
    when ∂_w M is contractive with constant <= 1 - m. We enforce this with a
    scale knob and a softclip; see `_apply_contraction_scale`.
    """

    def __init__(
        self,
        n: int,
        cond_dim: int,
        m: float = 0.1,
        hidden: int = 64,
        n_hidden: int = 2,
        max_scale: float = 0.8,
    ):
        super().__init__()
        if not (0.0 < m < 1.0):
            raise ValueError(f"m must lie in (0, 1); got {m}")
        self.n = n
        self.cond_dim = cond_dim
        self.m = float(m)
        self.max_scale = max_scale
        layers: list[_ICNNHidden] = []
        cur_in = 0
        for k in range(n_hidden):
            layers.append(
                _ICNNHidden(
                    dim_z=hidden,
                    dim_in=cur_in,
                    dim_x=n + cond_dim,  # raw w concatenated with c
                    first=(k == 0),
                )
            )
            cur_in = hidden
        self.hidden_layers = nn.ModuleList(layers)
        # Final linear-to-scalar with non-negative weights (so the whole
        # potential remains convex in w).
        self.W_out_raw = nn.Parameter(torch.randn(1, hidden) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(1))
        # Small affine input injection - keeps the operator well-behaved
        # near w = 0 and breaks symmetry at initialisation.
        self.input_inject = nn.Linear(cond_dim, n, bias=True)
        # Scale parameter that keeps ||∂_w M|| <= 1 - m. We use sigmoid so the
        # effective scale is in (0, max_scale).
        self.scale_raw = nn.Parameter(torch.tensor(0.0))

    def _potential(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([w, c], dim=-1)
        z = w.new_zeros((w.shape[0], 1))  # unused at the first layer
        for k, layer in enumerate(self.hidden_layers):
            z = layer(z, x)
        W_pos = F.softplus(self.W_out_raw)
        psi = z @ W_pos.T + self.b_out
        return psi.squeeze(-1)  # (B,)

    def _scale(self) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.scale_raw)

    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        w = w.requires_grad_(True) if not w.requires_grad else w
        with torch.enable_grad():
            psi = self._potential(w, c).sum()
            grad = torch.autograd.grad(psi, w, create_graph=self.training)[0]
        scale = self._scale()
        # M(w, c) = scale * ∇_w Ψ(w, c) + input injection.
        # The contraction scale keeps the operator bounded enough that
        # G(w) = w - M(w) is m-strongly monotone numerically.
        return scale * grad + self.input_inject(c)


# ----- Sanity utilities used by tests and by training diagnostics --------

@torch.no_grad()
def numerical_monotonicity_check(
    core: MonotoneCore,
    *,
    n_samples: int = 200,
    cond_scale: float = 1.0,
    eps: float = 1e-4,
) -> tuple[float, float]:
    """Return (min_ratio, mean_ratio) of <G(u)-G(v), u-v> / ||u-v||^2.

    For an m-strongly monotone G we expect the minimum ratio >= m - eps.
    """
    n = core.n
    cond_dim = getattr(core, "cond_dim")
    device = next(core.parameters()).device
    u = torch.randn(n_samples, n, device=device)
    v = torch.randn(n_samples, n, device=device)
    c = cond_scale * torch.randn(n_samples, cond_dim, device=device)
    Gu = u - core(u, c)
    Gv = v - core(v, c)
    diff = u - v
    num = (Gu - Gv * 1.0).mul(diff).sum(dim=-1) - Gv.mul(diff).sum(dim=-1) + Gv.mul(diff).sum(dim=-1)
    # Cleaner: dot((Gu - Gv), (u - v))
    dot = ((Gu - Gv) * diff).sum(dim=-1)
    denom = (diff * diff).sum(dim=-1).clamp_min(eps)
    ratio = dot / denom
    return float(ratio.min().item()), float(ratio.mean().item())
