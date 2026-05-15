"""TropicalLinearizer: f(x) = g_y^{-1}(A ⊗ g_x(x)).

Mirrors the structure of the linear Linearizer in /linearizer.py but uses
tropical (max-plus) multiplication in the latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..flows.vector_flow import VectorFlow
from ..tropical.core import TropicalCore
from ..tropical.kleene import kleene_squaring
from ..tropical.ops import trop_matvec


class TropicalLinearizer(nn.Module):
    """f(x) = g_y^{-1}(A ⊗ g_x(x)).

    The two flows g_x, g_y default to a shared instance (g_x = g_y = g), which
    is the setting required for self-composition (Lemma 4: Kleene collapse).
    Pass distinct flows only when mapping between different spaces.

    Args:
        flow_x: invertible flow producing the input latent.
        core: tropical core operator A.
        flow_y: invertible flow on the output side. If None, shares flow_x.
    """

    def __init__(
        self,
        flow_x: nn.Module,
        core: TropicalCore,
        flow_y: nn.Module | None = None,
    ):
        super().__init__()
        self.flow_x = flow_x
        self.flow_y = flow_y if flow_y is not None else flow_x
        self.core = core
        self.shared_g = flow_y is None

    # ---- The pieces. Subclass and override if the flow API differs. ----

    def gx(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow_x(x)

    def gy(self, y: torch.Tensor) -> torch.Tensor:
        return self.flow_y(y)

    def gx_inverse(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow_x.inverse(z)

    def gy_inverse(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow_y.inverse(z)

    def apply_core(
        self,
        z: torch.Tensor,
        *,
        mode: str = "hard",
        beta: float | None = None,
        A: torch.Tensor | None = None,
        **core_kwargs,
    ) -> torch.Tensor:
        return self.core(z, mode=mode, beta=beta, A=A, **core_kwargs)

    # ---- The two main forward modes. ----

    def forward(
        self,
        x: torch.Tensor,
        *,
        mode: str = "hard",
        beta: float | None = None,
        A: torch.Tensor | None = None,
        **core_kwargs,
    ) -> torch.Tensor:
        z = self.gx(x)
        z2 = self.apply_core(z, mode=mode, beta=beta, A=A, **core_kwargs)
        return self.gy_inverse(z2)

    def kleene_forward(
        self,
        x: torch.Tensor,
        *,
        n_iter: int | None = None,
        mode: str = "hard",
        beta: float | None = None,
        A: torch.Tensor | None = None,
        **core_kwargs,
    ) -> torch.Tensor:
        """One-shot Kleene-star application: F_inf(x) = g^{-1}(A^* ⊗ g(x)).

        Requires shared g (g_x = g_y), since otherwise self-composition isn't
        defined. n_iter caps the squaring count; default = ceil(log2(n)).
        """
        if not self.shared_g:
            raise RuntimeError(
                "kleene_forward requires shared g (flow_y was provided separately)"
            )
        if A is None:
            A = self.core.get_A(mode=mode, beta=beta, **core_kwargs)
        if A.shape[-1] != A.shape[-2]:
            raise RuntimeError(
                "kleene_forward requires a square core; got "
                f"A of shape {tuple(A.shape)}"
            )
        A_star = kleene_squaring(A, n_iter=n_iter, mode=mode, beta=beta)
        z = self.gx(x)
        z_star = trop_matvec(A_star, z) if mode == "hard" else self.apply_core(z, mode=mode, beta=beta, A=A_star)
        return self.gx_inverse(z_star)

    def iterate(
        self,
        x: torch.Tensor,
        n_steps: int,
        *,
        accumulate: bool = True,
        mode: str = "hard",
        beta: float | None = None,
        A: torch.Tensor | None = None,
        **core_kwargs,
    ) -> torch.Tensor:
        """Run x -> f(x) -> f(f(x)) -> ... for n_steps.

        If accumulate=True, returns the running tropical accumulation in latent
        space (this is F_N from Lemma 4: tropical sum of f^{circ k}(x)).
        If accumulate=False, returns the trajectory's final point.
        """
        if not self.shared_g:
            raise RuntimeError("iterate requires shared g")
        if A is None:
            A = self.core.get_A(mode=mode, beta=beta, **core_kwargs)
        z = self.gx(x)
        if accumulate:
            agg = z.clone()
        Az = z
        for _ in range(n_steps):
            Az = self.apply_core(Az, mode=mode, beta=beta, A=A)
            if accumulate:
                agg = torch.maximum(agg, Az)
        out_z = agg if accumulate else Az
        return self.gx_inverse(out_z)


def build_default(
    dim: int,
    *,
    n_blocks: int = 6,
    hidden: int = 128,
    core: TropicalCore | None = None,
    shared_g: bool = True,
) -> TropicalLinearizer:
    """Convenience: vector flow + square Kleene core, dim x dim."""
    from ..tropical.core import KleeneTropicalCore

    gx = VectorFlow(dim=dim, n_blocks=n_blocks, hidden=hidden)
    gy = None if shared_g else VectorFlow(dim=dim, n_blocks=n_blocks, hidden=hidden)
    if core is None:
        core = KleeneTropicalCore(n=dim)
    return TropicalLinearizer(flow_x=gx, core=core, flow_y=gy)
