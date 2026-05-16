"""Top-level Monotone Linearizer model.

Forward computation (blueprint §4.1):
    1.  c = φ(x)                              encoder
    2.  w* = solve   M(w*, c)                 monotone fixed-point (latent)
    3.  z* = g^{-1}(w*)                       flow inverse  (state space)
    4.  y  = head(z*)                         task head

φ, g, M, head are all swappable. The interesting structural piece is the
trio (M, g, head). With g = identity we recover monDEQ exactly; with g a
learned invertible flow we have the Monotone Linearizer.

Composition closure (Lemma 3) is not exposed via this top-level model since
the experiments we plan use one equilibrium layer; stacking is a Week-7
investigation.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..flows import VectorFlow
from ..monotone.implicit_diff import monotone_fixed_point
from ..monotone.parameterizations import MonotoneCore


class IdentityFlow(nn.Module):
    """g = identity. Used for ablation A1 (recover vanilla monDEQ)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z


class MonotoneLinearizer(nn.Module):
    """head ∘ g^{-1} ∘ FixedPoint(M, φ(x))."""

    def __init__(
        self,
        encoder: nn.Module,
        core: MonotoneCore,
        flow: nn.Module,
        head: nn.Module,
        *,
        solver: str = "forward_backward",
        solver_max_iter: int = 50,
        solver_tol: float = 1e-4,
        solver_step: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.flow = flow
        self.head = head
        self.solver = solver
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.solver_step = solver_step

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_info: bool = False,
        w_init: torch.Tensor | None = None,
    ):
        c = self.encoder(x)
        w_star, info = monotone_fixed_point(
            self.core,
            c,
            w_init=w_init,
            method=self.solver,
            max_iter=self.solver_max_iter,
            tol=self.solver_tol,
            step=self.solver_step,
        )
        z_star = self.flow.inverse(w_star)
        y = self.head(z_star)
        if return_info:
            return y, info
        return y


# ----- Convenience builders ----------------------------------------------------

def build_default(
    *,
    in_dim: int,
    latent_dim: int,
    out_dim: int,
    m: float = 0.1,
    encoder_hidden: int = 128,
    flow_blocks: int = 6,
    flow_hidden: int = 128,
    use_flow: bool = True,
) -> MonotoneLinearizer:
    """Quick build for tabular-style experiments and tests."""
    from ..monotone.parameterizations import MonotoneCoreP1

    encoder = nn.Sequential(
        nn.Linear(in_dim, encoder_hidden),
        nn.GELU(),
        nn.Linear(encoder_hidden, latent_dim),
    )
    core = MonotoneCoreP1(n=latent_dim, cond_dim=latent_dim, m=m)
    flow: nn.Module
    if use_flow:
        flow = VectorFlow(dim=latent_dim, n_blocks=flow_blocks, hidden=flow_hidden)
    else:
        flow = IdentityFlow()
    head = nn.Linear(latent_dim, out_dim)
    return MonotoneLinearizer(encoder=encoder, core=core, flow=flow, head=head)
