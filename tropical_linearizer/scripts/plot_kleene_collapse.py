"""Plot the headline Kleene-collapse figure (blueprint Fig. 2).

Loads a trained Bellman-Ford tropical model, evaluates per-step MAE for
T iterative steps in a sweep T ∈ {1, 2, 4, 8, ...}, and overlays one-shot
Kleene-star inference as a horizontal line.

Expected story: iter MAE drops as T grows and plateaus at T ≥ n-1, exactly
matching the one-shot Kleene line.

Usage:
    PYTHONPATH=/home/nvidia/Linearizer \\
        /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/plot_kleene_collapse.py \\
        --model tropical_linearizer/results/bf_n8/model.pt \\
        --out tropical_linearizer/results/bf_n8/kleene_collapse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tropical_linearizer.data import make_batch
from tropical_linearizer.models import BellmanFordTropical, predicted_distances
from tropical_linearizer.scripts.train_bellman_ford import (
    evaluate,
    gather_source_mask,
    gather_source_row,
    make_random_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sizes", type=str, default="8,16,32")
    parser.add_argument("--max_T", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    ckpt = torch.load(args.model, map_location="cpu")
    model_args = ckpt["args"]
    model = BellmanFordTropical(hidden=model_args["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # T sweep: dense early, log-spaced late.
    Ts: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, args.max_T]
    Ts = sorted(set([t for t in Ts if t <= args.max_T]))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))

    palette = plt.get_cmap("viridis")
    colors = [palette(0.15 + 0.7 * i / max(len(sizes) - 1, 1)) for i in range(len(sizes))]

    results = {}
    for size, color in zip(sizes, colors):
        torch.manual_seed(args.seed + size)
        np.random.seed(args.seed + size)
        evals = evaluate(
            model,
            n=size,
            batch_size=args.batch_size,
            n_steps_list=Ts,
            include_one_shot=True,
            seed_offset=10_000_000 + 1000 * size,
        )
        iter_mae = [evals["iter"][t] for t in Ts]
        one_shot_mae = evals["one_shot_mae"]
        results[size] = {"Ts": Ts, "iter_mae": iter_mae, "one_shot_mae": one_shot_mae}

        ax.plot(Ts, iter_mae, marker="o", color=color, label=f"iterative,  n={size}")
        ax.axhline(one_shot_mae, color=color, linestyle="--", alpha=0.8,
                   label=f"one-shot Kleene, n={size}")
        # Mark the theoretical plateau at T = n - 1.
        ax.axvline(size - 1, color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("T (iterative tropical-MPNN steps)")
    ax.set_ylabel("MAE on shortest-path distances")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Kleene collapse: iterative MAE plateaus at one-shot A* level")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved figure -> {out_path}")

    # Dump JSON alongside the figure.
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved data   -> {json_path}")

    # Print a summary table.
    print("\nSummary (iterative MAE at the largest T, vs one-shot Kleene):")
    print(f"{'n':>5} | {'iter@'+str(args.max_T):>12} | {'one-shot':>10} | {'diff':>10}")
    for size, r in results.items():
        last_mae = r["iter_mae"][-1]
        one_shot = r["one_shot_mae"]
        diff = abs(last_mae - one_shot)
        print(f"{size:>5} | {last_mae:>12.5f} | {one_shot:>10.5f} | {diff:>10.5f}")


if __name__ == "__main__":
    main()
