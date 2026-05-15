# Tropical Linearizer

Empirical follow-up to the Linearizer paper, extending the framework from the
linear semiring (ℝ, +, ×) to the tropical (max-plus) semiring (ℝ_max, max, +).

Core claim: a learned invertible coordinate map *g* can transport a
conventionally non-tropical neural network into a space where it is exactly
tropical-linear:

    f(x) = g_y⁻¹(A ⊗ g_x(x)),   (A ⊗ z)_i = max_j (A_{ij} + z_j).

See `tropical_linearizer_blueprint.md` at the repository root for the full
design document (theory, applications, ablations, milestones).

## Status (Week 1)

Foundations are in place and unit-tested:

| Component                                | Status |
|------------------------------------------|--------|
| Tropical ops (matvec, matmul, soft/ST)   | done   |
| Kleene star (squaring + Floyd-Warshall)  | done   |
| Tropical core (Regimes A / B / C)        | done   |
| Vector invertible flow (RealNVP-style)   | done   |
| TropicalLinearizer model + Kleene forward| done   |
| Tests: ops, Kleene-vs-NetworkX, invert,  | done   |
| Kleene collapse, idempotency-by-constr.  | done   |
| End-to-end Week-1 sanity (Floyd vs ours) | done   |

## Layout

```
tropical_linearizer/
├── tropical/
│   ├── ops.py        # max-plus matvec/matmul, soft + ST variants
│   ├── kleene.py     # Kleene star: squaring, Floyd-Warshall, power-iter
│   └── core.py       # TropicalCore: Dense / Hyper / Kleene regimes
├── flows/
│   └── vector_flow.py  # ActNorm + InvLinear + AffineCoupling stack
├── models/
│   └── tropical_linearizer.py   # f(x) = g_y^{-1}(A ⊗ g_x(x))
├── train/schedules.py           # β-annealing schedule for soft tropical
├── utils/networkx_ref.py        # numpy Floyd-Warshall reference
├── scripts/week1_sanity.py      # end-to-end Week-1 sanity check
└── tests/                       # pytest-style unit tests
    ├── test_tropical_ops.py
    ├── test_kleene.py
    ├── test_invertibility.py
    ├── test_collapse.py
    ├── test_idempotency.py
    └── run_all.py
```

## Running

```bash
# Tests
PYTHONPATH=/home/nvidia/Linearizer \
    /home/nvidia/anaconda3/bin/python tropical_linearizer/tests/run_all.py

# Week-1 sanity (tropical Floyd-Warshall agrees with NetworkX reference)
PYTHONPATH=/home/nvidia/Linearizer \
    /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/week1_sanity.py
```

We use the base conda Python (3.8.3, torch 2.1.0). The `isaac` conda env
(Python 3.6) is too old for the modern type-hint syntax used here.

## Next steps (per blueprint §17)

- **Week 2** — CLRS-30 data pipeline; train tropical Linearizer on Bellman-Ford
  with n = 8; produce first version of the headline Kleene-collapse figure.
- **Week 3** — Scale to n = 16 train / n = 32, 64 OOD eval; ablations A1–A6;
  add ECHO benchmark.
- **Week 4** — Warcraft 12×12 with CNN flow.
- **Week 5** — T-IGN on MNIST + piecewise-constant data.
- **Week 6** — Polish, qualitative figures, paper writing.
