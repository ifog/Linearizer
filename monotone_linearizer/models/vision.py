"""Vision instantiation of the Monotone Linearizer (blueprint §5.1).

Architecture:
    input  -> CNN encoder φ  -> latent c ∈ R^d
                                       \\
                                        +--->  M-DEQ fixed point in R^n
                                       /
    optional g (VectorFlow) wraps the latent state w*: z* = g^{-1}(w*).
    Linear classifier head on z*.

Set use_flow=False to recover monDEQ exactly (the central A1 ablation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flows import VectorFlow
from ..monotone.implicit_diff import monotone_fixed_point
from ..monotone.parameterizations import MonotoneCoreP1
from .monotone_linearizer import IdentityFlow


class SmallCNNEncoder(nn.Module):
    """A small ConvNet encoder.

    For CIFAR (3x32x32): 3 conv blocks with channels (base, 2*base, 4*base).
    For MNIST/FMNIST (1x28x28): 2 conv blocks.

    Final stage: adaptive-avg-pool -> linear projection to `out_dim`.
    """

    def __init__(self, in_channels: int, img_size: int, out_dim: int, base: int = 32):
        super().__init__()
        if img_size >= 32:  # CIFAR-style
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.GELU(),
                nn.Conv2d(base, 2 * base, 3, padding=1, stride=2), nn.BatchNorm2d(2 * base), nn.GELU(),
                nn.Conv2d(2 * base, 4 * base, 3, padding=1, stride=2), nn.BatchNorm2d(4 * base), nn.GELU(),
            )
            feat_ch = 4 * base
        else:  # 28x28 (M/FMNIST)
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.GELU(),
                nn.Conv2d(base, 2 * base, 3, padding=1, stride=2), nn.BatchNorm2d(2 * base), nn.GELU(),
            )
            feat_ch = 2 * base
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class VisionMonotoneLinearizer(nn.Module):
    """Full classifier wrapping the monotone DEQ + optional flow."""

    def __init__(
        self,
        *,
        in_channels: int,
        img_size: int,
        n_classes: int,
        latent_dim: int = 128,
        encoder_base: int = 32,
        m: float = 0.1,
        use_flow: bool = True,
        flow_blocks: int = 6,
        flow_hidden: int = 128,
        solver: str = "forward_backward",
        solver_max_iter: int = 50,
        solver_tol: float = 1e-4,
        solver_step: float = 1.0,
        head_norm: bool = True,
    ):
        super().__init__()
        self.use_flow = use_flow
        self.encoder = SmallCNNEncoder(in_channels, img_size, latent_dim, base=encoder_base)
        self.core = MonotoneCoreP1(n=latent_dim, cond_dim=latent_dim, m=m, init_scale=0.05)
        self.flow: nn.Module = VectorFlow(dim=latent_dim, n_blocks=flow_blocks, hidden=flow_hidden) if use_flow else IdentityFlow()
        head_layers: list[nn.Module] = []
        if head_norm:
            head_layers.append(nn.LayerNorm(latent_dim))
        head_layers.append(nn.Linear(latent_dim, n_classes))
        self.head = nn.Sequential(*head_layers)
        self.solver = solver
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.solver_step = solver_step

    def forward(self, x: torch.Tensor, *, return_info: bool = False):
        c = self.encoder(x)
        w_star, info = monotone_fixed_point(
            self.core, c,
            method=self.solver, max_iter=self.solver_max_iter,
            tol=self.solver_tol, step=self.solver_step,
        )
        z_star = self.flow.inverse(w_star)
        logits = self.head(z_star)
        if return_info:
            return logits, info
        return logits


def warmup_actnorm(model: VisionMonotoneLinearizer, loader, n_batches: int = 1, device: str = "cuda") -> None:
    """Push n_batches of *encoded* samples through the flow forward direction
    once so ActNorm's data-dependent init triggers. The model must be in train mode.
    """
    if not model.use_flow:
        return
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            c = model.encoder(x)
            # Run a few solver steps to get a reasonable w*; then push it
            # through the FLOW forward direction (which is where ActNorm sees
            # data) - this is exactly what flow.forward needs to data-init.
            from ..monotone.solvers import fixed_point_solve
            w_star, _ = fixed_point_solve(model.core, c, method=model.solver,
                                          max_iter=model.solver_max_iter,
                                          tol=model.solver_tol,
                                          step=model.solver_step)
            _ = model.flow(w_star)
            if i + 1 >= n_batches:
                break
