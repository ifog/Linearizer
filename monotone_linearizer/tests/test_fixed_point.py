"""Verify all three solvers converge to the same fixed point with low residual."""

from __future__ import annotations

import torch

from monotone_linearizer.monotone.parameterizations import MonotoneCoreP1
from monotone_linearizer.monotone.solvers import (
    forward_backward,
    naive_fixed_point,
    peaceman_rachford,
)


def test_naive_converges() -> None:
    torch.manual_seed(0)
    core = MonotoneCoreP1(n=32, cond_dim=8, m=0.1)
    c = torch.randn(4, 8)
    w_star, info = naive_fixed_point(core, c, max_iter=200, tol=1e-5)
    final_err = (w_star - core(w_star, c)).norm(dim=-1).max().item()
    assert final_err < 1e-4, f"naive solver did not converge: err = {final_err}"


def test_forward_backward_converges() -> None:
    torch.manual_seed(1)
    core = MonotoneCoreP1(n=32, cond_dim=8, m=0.1)
    c = torch.randn(4, 8)
    w_star, info = forward_backward(core, c, max_iter=200, tol=1e-5, step=0.8)
    final_err = (w_star - core(w_star, c)).norm(dim=-1).max().item()
    assert final_err < 1e-4


def test_pr_converges() -> None:
    torch.manual_seed(2)
    core = MonotoneCoreP1(n=32, cond_dim=8, m=0.1)
    c = torch.randn(4, 8)
    w_star, info = peaceman_rachford(core, c, max_iter=200, tol=1e-5)
    final_err = (w_star - core(w_star, c)).norm(dim=-1).max().item()
    assert final_err < 1e-4


def test_solvers_agree() -> None:
    """All three solvers should converge to the same w*."""
    torch.manual_seed(3)
    core = MonotoneCoreP1(n=32, cond_dim=8, m=0.1)
    c = torch.randn(4, 8)
    w_n, _ = naive_fixed_point(core, c, max_iter=500, tol=1e-7)
    w_fb, _ = forward_backward(core, c, max_iter=500, tol=1e-7, step=0.8)
    w_pr, _ = peaceman_rachford(core, c, max_iter=500, tol=1e-7)
    assert torch.allclose(w_n, w_fb, atol=1e-3), (w_n - w_fb).abs().max().item()
    # PR is the most approximate (inexact resolvent); loosen tolerance.
    assert torch.allclose(w_n, w_pr, atol=5e-2), (w_n - w_pr).abs().max().item()


if __name__ == "__main__":
    test_naive_converges()
    test_forward_backward_converges()
    test_pr_converges()
    test_solvers_agree()
    print("OK: fixed-point tests passed.")
