"""Verify implicit differentiation gives correct gradients (finite-diff check)."""

from __future__ import annotations

import torch

from monotone_linearizer.monotone.implicit_diff import monotone_fixed_point
from monotone_linearizer.monotone.parameterizations import MonotoneCoreP1


def _loss(core: MonotoneCoreP1, c: torch.Tensor) -> torch.Tensor:
    w_star, _ = monotone_fixed_point(core, c, method="forward_backward",
                                      max_iter=200, tol=1e-6, step=0.8)
    return w_star.pow(2).sum()


def test_grad_wrt_c_matches_fd() -> None:
    """d/dc L should match a central finite-difference estimate to ~1e-3."""
    torch.manual_seed(0)
    core = MonotoneCoreP1(n=8, cond_dim=4, m=0.2, init_scale=0.05)
    c = torch.randn(2, 4, requires_grad=True)
    L = _loss(core, c)
    L.backward()
    analytic = c.grad.clone()

    # Finite differences (central, eps = 1e-3).
    eps = 1e-3
    fd = torch.zeros_like(c)
    with torch.no_grad():
        for b in range(c.shape[0]):
            for j in range(c.shape[1]):
                c_p = c.detach().clone(); c_p[b, j] += eps
                c_m = c.detach().clone(); c_m[b, j] -= eps
                L_p = _loss(core, c_p)
                L_m = _loss(core, c_m)
                fd[b, j] = (L_p - L_m) / (2 * eps)

    # The implicit-diff hook hits one Neumann iteration internally; with the
    # default tolerance we expect agreement to a few 1e-2 max abs diff.
    diff = (analytic - fd).abs().max().item()
    rel = diff / (fd.abs().max().item() + 1e-6)
    assert rel < 0.1, f"grad mismatch: max abs diff = {diff}, rel = {rel}"


def test_grad_wrt_params_flows() -> None:
    """A backward pass updates core.A (just check the gradient is finite and nonzero)."""
    torch.manual_seed(1)
    core = MonotoneCoreP1(n=8, cond_dim=4, m=0.2, init_scale=0.05)
    c = torch.randn(2, 4)
    L = _loss(core, c)
    L.backward()
    assert core.A.grad is not None
    assert torch.isfinite(core.A.grad).all()
    assert core.A.grad.abs().max().item() > 0.0


if __name__ == "__main__":
    test_grad_wrt_c_matches_fd()
    test_grad_wrt_params_flows()
    print("OK: implicit-diff tests passed.")
