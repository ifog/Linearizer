"""Kleene-collapse test - the central architectural claim (blueprint Lemma 4).

For a tropical Linearizer with shared g and core A (no positive cycles),
the accumulated trajectory F_N(x) = ⊕_{k=0}^N f^{∘k}(x) equals
g^{-1}(A^{*,N} ⊗ g(x)). When N >= n-1, this collapses to F_inf(x), and one
application of A^* matches T iterations for any T >= n-1.

This test is the empirical analog of the headline figure (blueprint Fig. 2).
"""

from __future__ import annotations

import torch

from tropical_linearizer.flows.vector_flow import VectorFlow
from tropical_linearizer.models.tropical_linearizer import TropicalLinearizer
from tropical_linearizer.tropical.core import KleeneTropicalCore
from tropical_linearizer.tropical.kleene import kleene_squaring
from tropical_linearizer.tropical.ops import trop_matvec


def test_kleene_collapse_random_init() -> None:
    """Iterating f T>=n-1 times and one-shot Kleene give the same answer."""
    torch.manual_seed(0)
    n = 8
    flow = VectorFlow(dim=n, n_blocks=2, hidden=32)
    core = KleeneTropicalCore(n=n)
    model = TropicalLinearizer(flow_x=flow, core=core)

    # Initialise ActNorm.
    model.train()
    x_init = torch.randn(16, n)
    _ = model.gx(x_init)
    model.eval()

    x = torch.randn(4, n)
    # Use a fixed underlying P matrix (not the trained core) for a crisp
    # check on the algebraic identity.
    with torch.no_grad():
        P = -torch.abs(torch.randn(n, n))   # non-positive -> stabilises
        # Apply T iterations using P (bypass the core's Kleene-by-construction).
        z = model.gx(x)
        Az = z.clone()
        agg = z.clone()
        for _ in range(2 * n):
            Az = trop_matvec(P, Az)
            agg = torch.maximum(agg, Az)
        out_iter = model.gx_inverse(agg)

        # One-shot via Kleene.
        P_star = kleene_squaring(P, n_iter=None)  # 2^{ceil(log2(n))} >> n-1
        z_star = trop_matvec(P_star, z)
        out_one_shot = model.gx_inverse(z_star)

    diff = torch.max(torch.abs(out_iter - out_one_shot)).item()
    assert diff < 1e-4, f"Kleene collapse failed: max abs diff = {diff}"


def test_kleene_forward_method() -> None:
    """TropicalLinearizer.kleene_forward gives the same result as manually
    accumulating iterations."""
    torch.manual_seed(1)
    n = 6
    flow = VectorFlow(dim=n, n_blocks=2, hidden=32)
    # Use a dense (non-Kleene) core so kleene_forward has work to do.
    from tropical_linearizer.tropical.core import DenseTropicalCore

    core = DenseTropicalCore(in_dim=n, out_dim=n, init_scale=0.05)
    # Force the core to start non-positive for stability.
    with torch.no_grad():
        core.A_raw.data = -torch.abs(core.A_raw.data)
    model = TropicalLinearizer(flow_x=flow, core=core)

    # ActNorm warm-up.
    model.train()
    _ = model.gx(torch.randn(16, n))
    model.eval()

    x = torch.randn(3, n)
    one_shot = model.kleene_forward(x)
    iter_out = model.iterate(x, n_steps=2 * n, accumulate=True)
    diff = torch.max(torch.abs(one_shot - iter_out)).item()
    assert diff < 1e-3, f"kleene_forward vs iterate diff = {diff}"


if __name__ == "__main__":
    test_kleene_collapse_random_init()
    test_kleene_forward_method()
    print("OK: collapse tests passed.")
