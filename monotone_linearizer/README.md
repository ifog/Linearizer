# Monotone Linearizer

Empirical follow-up to the Linearizer paper, extending the framework from a
*linear* core to a *monotone-operator* core, targeting the well-known
expressivity-stability tradeoff in implicit deep learning:

    f(z, x) = g^{-1}( M( g(z), φ(x) ) )

where M is m-strongly monotone (so the equilibrium z* = f(z*, x) exists,
is unique, and is found by provably-convergent splitting), g is a learned
invertible flow that lifts monDEQ's expressivity restriction, and φ is a
standard encoder. With g = identity we recover monDEQ exactly — Ablation A1
in the blueprint is the empirical proxy for the central "non-trivial g earns
its keep" claim.

Full design doc: `monotone_linearizer_blueprint.md` at the repo root.

## Status

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Monotone parameterizations (P1, P3), three solvers (naive, FB, PR), implicit differentiation, end-to-end model with g = flow or g = identity, 12 unit tests | done |
| 2 | CIFAR-10 pilot, equilibrium-error tracking, first stability comparison | pending |
| 3 | **Ablation A1 (CRITICAL gate):** g = Id vs. g = learned flow on CIFAR-100 | pending |
| 4 | CIFAR-100 full + Pareto headline figure + ablations A2–A6 | pending |
| 5 | WikiText-103 | pending |
| 6 | OGB / IGNN | pending |

## Layout

```
monotone_linearizer/
├── monotone/
│   ├── parameterizations.py   # P1 (Winston-Kolter), P3 (ICNN gradient)
│   ├── solvers.py             # naive, forward-backward, Peaceman-Rachford
│   └── implicit_diff.py       # Neumann-series adjoint solve
├── flows/                     # re-exports tropical_linearizer.flows.vector_flow
├── models/
│   └── monotone_linearizer.py  # head ∘ g^-1 ∘ FixedPoint(M, φ(x))
└── tests/                     # 12 unit tests covering all Week-1 components
    ├── test_monotonicity.py
    ├── test_fixed_point.py
    ├── test_invertibility.py
    ├── test_implicit_diff.py
    ├── test_end_to_end.py
    └── run_all.py
```

## Running

```bash
# Tests
PYTHONPATH=/home/nvidia/Linearizer \
    /home/nvidia/anaconda3/bin/python monotone_linearizer/tests/run_all.py
```

## Implementation notes (gotchas hit during Week 1)

- **PR splitting needs the activation's resolvent.** A generic "inexact
  resolvent via inner gradient steps" diverges. For P1 with ReLU, the
  resolvent is the projection onto the non-negative orthant, and we
  implement Winston-Kolter Alg. 1 directly. For other parameterizations
  PR currently falls back to forward-backward.
- **Implicit-diff hook needs `torch.enable_grad()`.** Backward hooks run
  with grad disabled by default; the adjoint Jacobian-VJP must explicitly
  re-enable autograd, otherwise the inner `M(w, c)` doesn't track gradients.

These match standard DEQ pitfalls; recording them so future readers do not
have to rediscover.
