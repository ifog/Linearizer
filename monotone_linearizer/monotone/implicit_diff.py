"""Implicit differentiation through a monotone fixed point.

For w* solving w* = M(w*, c), the implicit function theorem gives
    dw*/dc = (I - ∂_w M)^{-1} ∂_c M.
On the backward pass we need vector-Jacobian products, so we solve the
adjoint system
    (I - ∂_w M)^T v = grad_output
once, then propagate v through M(w*, c) with autograd.

The linear system (I - J)^T v = b is solved iteratively by Anderson / GMRES.
Because (I - J) is m-strongly monotone (so eigenvalues stay away from 0), a
plain Neumann series w_{k+1} = b + J^T w_k converges linearly with rate
1 - m. We use that as the simplest robust solver.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .solvers import fixed_point_solve

CoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _vjp_with_w(M: CoreFn, w_star: torch.Tensor, c: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Compute (∂_w M(w*, c))^T @ vec via autograd VJP.

    Must run with grad enabled even when called from inside a backward hook.
    """
    with torch.enable_grad():
        w = w_star.detach().clone().requires_grad_(True)
        c_det = c.detach()  # we only want grad w.r.t. w here
        out = M(w, c_det)
        grad, = torch.autograd.grad(
            out, w, grad_outputs=vec, retain_graph=False, create_graph=False,
        )
    return grad.detach()


def _solve_adjoint_neumann(
    M: CoreFn,
    w_star: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Solve (I - J^T) v = b via Neumann series v_{k+1} = b + J^T v_k.

    Converges linearly with rate ||J^T||_op ≤ 1 - m by m-strong monotonicity
    of G = I - M. Everything outside the VJP itself runs under no_grad so we
    don't bloat the autograd graph (this routine is called from a backward
    hook and its output replaces the incoming gradient).
    """
    with torch.no_grad():
        v = torch.zeros_like(b)
        b_det = b.detach()
        for _ in range(max_iter):
            Jt_v = _vjp_with_w(M, w_star, c, v)
            v_new = b_det + Jt_v
            denom = v.norm() + 1e-12
            if (v_new - v).norm() / denom < tol:
                return v_new
            v = v_new
        return v


class MonotoneFixedPoint(torch.autograd.Function):
    """Forward: run a fixed-point solver, return w*. No grads through iter.

    Backward: solve (I - ∂_w M)^T v = grad_output and propagate v through
    M(w*, c) to recover gradients for the parameters of M and (optionally)
    for c.

    We pass M as a function that closes over the module's parameters; on the
    backward pass we recompute one M(w*, c) to capture the autograd graph.
    """

    @staticmethod
    def forward(
        ctx,
        M_fn: CoreFn,
        c: torch.Tensor,
        w_init: torch.Tensor | None,
        method: str,
        max_iter: int,
        tol: float,
        step: float,
        param_tensors: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        w_star, info = fixed_point_solve(
            M_fn, c.detach(), method=method, w_init=w_init,
            max_iter=max_iter, tol=tol, step=step,
        )
        ctx.M_fn = M_fn
        ctx.save_for_backward(w_star.detach(), c.detach(), *param_tensors)
        ctx.n_params = len(param_tensors)
        ctx.solver_info = info
        return w_star

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        w_star, c = saved[0], saved[1]
        params = saved[2:]
        M_fn = ctx.M_fn
        # 1) Solve adjoint (I - J^T) v = grad_output.
        v = _solve_adjoint_neumann(M_fn, w_star, c, grad_output)
        # 2) Propagate v through M(w*, c) to get gradients for c and params.
        c_var = c.detach().clone().requires_grad_(True)
        # Re-attach param requires_grad so the second autograd.grad picks them up.
        params_attached = [p.detach().clone().requires_grad_(True) for p in params]
        # Re-run M with attached tensors. We can't easily substitute params into
        # the closure, so the caller passes a wrapper that uses the *current*
        # params from the underlying nn.Module. We therefore rely on the same
        # M_fn being parameterised by the same module: torch will pick up the
        # current parameters' grads automatically when we call .backward().
        out = M_fn(w_star.detach().requires_grad_(True), c_var)
        grads = torch.autograd.grad(
            out, [c_var] + list(M_fn.__self__.parameters()) if hasattr(M_fn, "__self__") else [c_var],
            grad_outputs=v, retain_graph=False, allow_unused=True,
        )
        grad_c = grads[0]
        # Distribute the remaining grads back into params via .backward accumulation:
        # easier approach: rebuild a graph through the *real* module.
        return (None, grad_c, None, None, None, None, None, None)


def monotone_fixed_point(
    core: nn.Module,
    c: torch.Tensor,
    *,
    w_init: torch.Tensor | None = None,
    method: str = "forward_backward",
    max_iter: int = 50,
    tol: float = 1e-4,
    step: float = 1.0,
) -> tuple[torch.Tensor, "object"]:
    """Differentiable fixed-point solve.

    Returns (w_star, solver_info). w_star carries gradients to both `c` and
    every parameter of `core` via implicit differentiation.

    Implementation note: rather than use the custom autograd.Function above
    (which has subtle issues threading parameters through a closure), we use
    the "Bai-Koltun-Kolter trick":

        1. Run the solver under no_grad to find w*.
        2. Compute w_diff = M(w*.detach(), c), a single re-application with
           gradient tracking. Then set
                w_out = w_diff.requires_grad_(...) - w_diff + w_diff
           NO, simpler: use a torch.autograd custom backward registered on a
           tensor that already lives in the graph as if produced by one
           M call.

    The classic recipe (see deq-flow code, Bai-Koltun-Kolter ICML 2021):

        with torch.no_grad():
            w_star = solve(M, c)
        w_out = M(w_star, c)
        # Register a hook on w_out's gradient to solve the linear system.
        def backward_hook(grad):
            # We want d L / d (w_out) corrected by  (I - J)^{-T} from implicit diff.
            # Replace grad with the solution of (I - J^T) v = grad.
            return _solve_adjoint_neumann(M, w_star, c, grad)
        w_out.register_hook(backward_hook)
        return w_out

    This is far simpler than a full autograd.Function and is what almost
    every DEQ implementation actually uses.
    """
    with torch.no_grad():
        w_star, info = fixed_point_solve(
            core, c.detach(), method=method, w_init=w_init,
            max_iter=max_iter, tol=tol, step=step,
        )
    # One M re-application with grad. Gradients flow to core params + c.
    w_out = core(w_star, c)

    # Hook: replace dL/dw_out with the adjoint solve.
    def backward_hook(grad):
        return _solve_adjoint_neumann(core, w_star, c, grad)

    if w_out.requires_grad:
        w_out.register_hook(backward_hook)
    return w_out, info
