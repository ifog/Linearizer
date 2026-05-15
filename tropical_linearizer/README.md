# Tropical Linearizer

Empirical follow-up to the Linearizer paper, extending the framework from the
linear semiring (ℝ, +, ×) to the tropical (max-plus) semiring (ℝ_max, max, +).

Core claim: a learned invertible coordinate map *g* can transport a
conventionally non-tropical neural network into a space where it is exactly
tropical-linear:

    f(x) = g_y⁻¹(A ⊗ g_x(x)),   (A ⊗ z)_i = max_j (A_{ij} + z_j).

See `tropical_linearizer_blueprint.md` at the repository root for the full
design document (theory, applications, ablations, milestones).

## Status

| Week | Milestone                                | Status |
|------|------------------------------------------|--------|
| 1    | Tropical ops, Kleene star, invertible flow, model scaffold, 19 unit tests, end-to-end Floyd-Warshall sanity | done |
| 2    | Synthetic Bellman-Ford pipeline, edge-hypernetwork tropical model, trained on n=8, headline Fig. 2 (Kleene collapse) shown OOD at n=16, n=32 | done |
| 3+   | Ablations, ECHO benchmark, Warcraft, T-IGN | pending |

**Week 2 headline result** (`results/bf_n8/kleene_collapse.png`):
After training the tropical Bellman-Ford model on random ER graphs at n=8,
iterative T-step MAE drops to the one-shot Kleene-star MAE at exactly T=n-1
and plateaus there — confirming Lemma 4 empirically. OOD at n=16, n=32 the
same collapse pattern holds, only that the plateau MAE grows slightly with
graph size (0.002 → 0.004 in this run).

## Layout

```
tropical_linearizer/
├── tropical/
│   ├── ops.py        # max-plus matvec/matmul, soft + ST variants
│   ├── kleene.py     # Kleene star: squaring, Floyd-Warshall, power-iter
│   └── core.py       # TropicalCore: Dense / Hyper / Kleene regimes
├── flows/
│   └── vector_flow.py            # ActNorm + InvLinear + AffineCoupling stack
├── models/
│   ├── tropical_linearizer.py    # f(x) = g_y^{-1}(A ⊗ g_x(x))
│   └── bellman_ford_model.py     # Tropical SSSP / all-pairs Bellman-Ford
├── data/bellman_ford.py          # Synthetic ER graphs + numpy Floyd-Warshall labels
├── train/schedules.py            # β-annealing schedule for soft tropical
├── utils/networkx_ref.py         # numpy Floyd-Warshall reference for tests
├── scripts/
│   ├── week1_sanity.py           # end-to-end Week-1 sanity check
│   ├── train_bellman_ford.py     # Week-2 training script
│   └── plot_kleene_collapse.py   # Headline Fig. 2 generator
├── results/bf_n8/                # Week-2 outputs: model.pt, history.json, figure
└── tests/                        # 19 unit tests
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

# Week-2 training (≈30 s on CPU for 4000 steps at n=8)
PYTHONPATH=/home/nvidia/Linearizer \
    /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/train_bellman_ford.py \
        --steps 4000 --batch_size 32 --n 8 --hidden 32 --eval_every 500 \
        --eval_sizes 8,16,32 --out tropical_linearizer/results/bf_n8

# Week-2 headline figure (Kleene collapse)
PYTHONPATH=/home/nvidia/Linearizer \
    /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/plot_kleene_collapse.py \
        --model tropical_linearizer/results/bf_n8/model.pt \
        --out tropical_linearizer/results/bf_n8/kleene_collapse.png \
        --sizes 8,16,32 --max_T 64
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
