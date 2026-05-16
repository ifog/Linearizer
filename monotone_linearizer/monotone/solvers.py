"""Fixed-point solvers for monotone DEQ layers.

We provide three:

  naive_fixed_point   - Plain Picard iteration w_{k+1} = M(w_k, c). Cheap and
                        usually convergent for P1 with m > 0 but no formal
                        guarantee.

  forward_backward    - Forward-backward splitting on
                        find w*:  w* = M(w*, c)
                        rewritten as
                            0 = (I - M)(w*),
                        with (I - M) split as (alpha * (I - M)) - 0. This is
                        essentially gradient descent on the residual norm,
                        with provable linear convergence under m-strong
                        monotonicity.

  peaceman_rachford   - PR splitting (Winston-Kolter Alg. 1 reformulation).
                        Stronger convergence guarantees, used as the default
                        for monDEQ in their paper.

All three return (w_star, info_dict). info_dict contains 'iters' and the
final equilibrium error ||w - M(w, c)||.

For training time we wrap the chosen solver in a torch.autograd.Function
(see implicit_diff.py) so that gradients flow via implicit differentiation,
not through the iteration unrolled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

CoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SolverInfo:
    iters: int = 0
    final_err: float = float("inf")
    err_history: list[float] = field(default_factory=list)


def _equilibrium_err(w: torch.Tensor, Mw: torch.Tensor) -> torch.Tensor:
    """Per-batch L2 of (w - M(w, c)). Returns scalar = max over batch."""
    return (w - Mw).norm(dim=-1).max()


@torch.no_grad()
def naive_fixed_point(
    M: CoreFn,
    c: torch.Tensor,
    w_init: torch.Tensor | None = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    record_history: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    """w_{k+1} = M(w_k, c). Stops when ||w - M(w, c)||_max <= tol."""
    w = torch.zeros(c.shape[0], _infer_n_from_M(M, c), device=c.device, dtype=c.dtype) if w_init is None else w_init
    info = SolverInfo()
    for k in range(1, max_iter + 1):
        w_new = M(w, c)
        err = _equilibrium_err(w_new, M(w_new, c)).item() if record_history else _equilibrium_err(w, w_new).item()
        if record_history:
            info.err_history.append(err)
        info.iters = k
        info.final_err = err
        if err <= tol:
            return w_new, info
        w = w_new
    return w, info


@torch.no_grad()
def forward_backward(
    M: CoreFn,
    c: torch.Tensor,
    w_init: torch.Tensor | None = None,
    *,
    step: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    record_history: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    """w_{k+1} = w_k - step * (w_k - M(w_k, c)).

    For step in (0, 2 / L) where L = Lipschitz of (I - M), this converges
    linearly when M is m-strongly monotone. step = 1 collapses to the naive
    Picard iteration; step < 1 damps it.
    """
    w = torch.zeros(c.shape[0], _infer_n_from_M(M, c), device=c.device, dtype=c.dtype) if w_init is None else w_init
    info = SolverInfo()
    for k in range(1, max_iter + 1):
        Mw = M(w, c)
        residual = w - Mw
        w_new = w - step * residual
        err = residual.norm(dim=-1).max().item()
        if record_history:
            info.err_history.append(err)
        info.iters = k
        info.final_err = err
        if err <= tol:
            return w_new, info
        w = w_new
    return w, info


@torch.no_grad()
def peaceman_rachford(
    core,
    c: torch.Tensor,
    w_init: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    record_history: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    """Peaceman-Rachford splitting (Winston-Kolter Alg. 1) for ReLU monDEQ.

    Specialised to P1 (cores that expose `W()` and `U`, with ReLU activation):
    the fixed point z = relu(W z + U c + b) is rewritten as the inclusion
        0 ∈ (I - W)(z) - U c - b + N_{+}(z),
    where N_+ is the normal cone of {z ≥ 0}. The proximal/resolvent of N_+
    is the projection onto the non-negative orthant (= relu). The Winston-
    Kolter PR iteration is then:
        z       = relu(u)
        u_half  = 2 z - u
        v       = solve  (I + alpha (I - W)) v = u_half + alpha (U c + b)
        u       = u + (v - z)
    Repeated until ||z - relu(W z + U c + b)|| <= tol.

    Requires a P1-style core (exposes W() and U). For other cores this falls
    back to forward_backward.
    """
    n = _infer_n_from_M(core, c)
    if not (hasattr(core, "W") and hasattr(core, "U") and hasattr(core, "b")):
        # Fall back: not a P1 core.
        return forward_backward(
            core, c, w_init=w_init, step=0.8,
            max_iter=max_iter, tol=tol, record_history=record_history,
        )
    device, dtype = c.device, c.dtype
    z = torch.zeros(c.shape[0], n, device=device, dtype=dtype) if w_init is None else w_init.clone()
    u = z.clone()
    info = SolverInfo()
    Wmat = core.W()
    I = torch.eye(n, device=device, dtype=dtype)
    A = I + alpha * (I - Wmat)        # SPD (since I - W ⪰ mI > 0)
    A_inv = torch.linalg.inv(A)        # n x n, cheap for the latent dims we use
    Uc_plus_b = core.U(c) + core.b     # (B, n)
    for k in range(1, max_iter + 1):
        z = torch.clamp(u, min=0.0)
        u_half = 2.0 * z - u
        v = (u_half + alpha * Uc_plus_b) @ A_inv.T
        u = u + (v - z)
        err = _equilibrium_err(z, core(z, c)).item()
        if record_history:
            info.err_history.append(err)
        info.iters = k
        info.final_err = err
        if err <= tol:
            return z, info
    return z, info


def fixed_point_solve(
    M: CoreFn,
    c: torch.Tensor,
    *,
    method: str = "forward_backward",
    w_init: torch.Tensor | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    step: float = 1.0,
    record_history: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    if method == "naive":
        return naive_fixed_point(M, c, w_init=w_init, max_iter=max_iter, tol=tol, record_history=record_history)
    if method == "forward_backward":
        return forward_backward(M, c, w_init=w_init, step=step, max_iter=max_iter, tol=tol, record_history=record_history)
    if method == "peaceman_rachford":
        return peaceman_rachford(M, c, w_init=w_init, max_iter=max_iter, tol=tol, record_history=record_history)
    raise ValueError(f"Unknown method: {method!r}")


def _infer_n_from_M(M: CoreFn, c: torch.Tensor) -> int:
    """Probe the core with a zero w to recover the latent dim n."""
    # Try to grab .n attribute if M is a module; otherwise probe.
    n_attr = getattr(M, "n", None)
    if isinstance(n_attr, int):
        return n_attr
    # Probe via a dummy call.
    with torch.no_grad():
        for try_n in (16, 32, 64, 128, 256, 512, 1024):
            try:
                w_try = torch.zeros(c.shape[0], try_n, device=c.device, dtype=c.dtype)
                out = M(w_try, c)
                if out.shape == w_try.shape:
                    return try_n
            except Exception:
                continue
    raise RuntimeError("Could not infer latent dim from M; expose M.n.")
