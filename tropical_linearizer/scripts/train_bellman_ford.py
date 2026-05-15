"""Train the tropical Bellman-Ford model on synthetic Erdős-Rényi graphs.

We deliberately re-randomise the hyperparameter skip (A = -alpha*w + beta) so
the model does NOT start at the trivial A = -W solution. That makes the
training story honest: the hypernetwork has to learn the right tropical
operator from edge features.

Loss: masked MSE in max-plus space (predicted h vs -D ground truth), only on
reachable (i, src) pairs. We train under a soft-max relaxation with
β annealed from 1 -> 50 over training (blueprint §4.5).

Usage:
    PYTHONPATH=/home/nvidia/Linearizer \\
        /home/nvidia/anaconda3/bin/python tropical_linearizer/scripts/train_bellman_ford.py \\
        --steps 4000 --n 8 --eval_n_steps 16 --out results/bf_n8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tropical_linearizer.data import MISSING_EDGE, make_batch
from tropical_linearizer.models import BellmanFordTropical, predicted_distances
from tropical_linearizer.train.schedules import beta_schedule_log
from tropical_linearizer.tropical.ops import NEG_INF


def make_random_source(B: int, n: int) -> torch.Tensor:
    """Random source per graph (uniformly over [0, n))."""
    return torch.randint(0, n, (B,))


def gather_source_row(D: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """D: (B, N, N) -> (B, N), the source-row distances."""
    B, N, _ = D.shape
    idx = src.view(B, 1, 1).expand(B, 1, N)
    return D.gather(1, idx).squeeze(1)


def gather_source_mask(mask: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Like gather_source_row but for the reachability mask."""
    B, N, _ = mask.shape
    idx = src.view(B, 1, 1).expand(B, 1, N)
    return mask.gather(1, idx).squeeze(1)


def compute_loss(
    y_pred: torch.Tensor,
    d_target: torch.Tensor,
    reach_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE between predicted max-plus y and ground-truth -d.

    Operates on reachable entries only (reach_mask = 1.0 where reachable).
    """
    y_target = -d_target
    err = (y_pred - y_target) ** 2
    return (err * reach_mask).sum() / reach_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: BellmanFordTropical,
    *,
    n: int,
    batch_size: int = 64,
    n_steps_list: list[int] | None = None,
    include_one_shot: bool = True,
    seed_offset: int = 1_000_000,
) -> dict:
    """Evaluate on a freshly-sampled batch at size n.

    Returns a dict with:
        "iter": {T: mae}  ground-truth MAE for each T in n_steps_list (hard max)
        "one_shot_mae": MAE for one-shot Kleene-star prediction
    """
    n_steps_list = n_steps_list or [1, 2, 4, 8, 16, 32]
    batch = make_batch(batch_size=batch_size, n=n, seed=seed_offset + n)
    src = make_random_source(batch_size, n)
    d_target = gather_source_row(batch.D, src)
    reach = gather_source_mask(batch.mask, src)

    out = {"iter": {}, "one_shot_mae": None, "n": n, "batch_size": batch_size}
    for T in n_steps_list:
        y = model(batch.W, src, n_steps=T, one_shot=False, mode="hard")
        d_pred = predicted_distances(y)
        err = (d_pred - d_target).abs()
        mae = (err * reach).sum().item() / reach.sum().clamp_min(1.0).item()
        out["iter"][int(T)] = float(mae)

    if include_one_shot:
        y = model(batch.W, src, one_shot=True, mode="hard")
        d_pred = predicted_distances(y)
        err = (d_pred - d_target).abs()
        mae = (err * reach).sum().item() / reach.sum().clamp_min(1.0).item()
        out["one_shot_mae"] = float(mae)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n", type=int, default=8, help="Train graph size")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=50.0)
    parser.add_argument("--init_perturb", type=float, default=1.0,
                        help="Std of Gaussian noise added to alpha/beta at init "
                             "so we don't start at the trivial A=-W solution.")
    parser.add_argument("--eval_n_steps", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--eval_sizes", type=str, default="8")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/bf_n8")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BellmanFordTropical(hidden=args.hidden)
    # Perturb the skip params so we don't start at the trivial A=-W solution.
    with torch.no_grad():
        model.alpha.data += args.init_perturb * torch.randn_like(model.alpha)
        model.beta.data += args.init_perturb * torch.randn_like(model.beta)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    eval_n_steps_list = [int(x) for x in args.eval_n_steps.split(",") if x.strip()]
    eval_sizes = [int(x) for x in args.eval_sizes.split(",") if x.strip()]

    history = {"steps": [], "loss": [], "beta": [], "alpha": [], "alpha_param": [], "eval": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        beta = beta_schedule_log(step, args.steps, beta0=args.beta0, beta1=args.beta1)
        batch = make_batch(batch_size=args.batch_size, n=args.n, seed=args.seed * 100003 + step)
        src = make_random_source(args.batch_size, args.n)
        d_target = gather_source_row(batch.D, src)
        reach = gather_source_mask(batch.mask, src)

        # Train with as many iterative steps as needed to cover an n-node graph.
        # We use 2*n to be safe under no positive cycles.
        T_train = 2 * args.n
        y = model(batch.W, src, n_steps=T_train, one_shot=False, mode="soft", beta=beta)
        loss = compute_loss(y, d_target, reach)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            evals = {n: evaluate(model, n=n, n_steps_list=eval_n_steps_list) for n in eval_sizes}
            history["steps"].append(step)
            history["loss"].append(float(loss.item()))
            history["beta"].append(float(beta))
            history["alpha"].append(float(model.alpha.detach().item()))
            history["alpha_param"].append(float(model.beta.detach().item()))
            history["eval"].append(evals)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.steps} | loss {loss.item():.5f} | "
                f"beta {beta:.2f} | alpha {model.alpha.item():.3f} | "
                f"intercept {model.beta.item():.3f} | "
                f"eval n={eval_sizes[0]}: iter@{eval_n_steps_list[-1]}="
                f"{evals[eval_sizes[0]]['iter'][eval_n_steps_list[-1]]:.4f} "
                f"one-shot={evals[eval_sizes[0]]['one_shot_mae']:.4f} | "
                f"{elapsed:.0f}s",
                flush=True,
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
        },
        out_dir / "model.pt",
    )
    print(f"\nSaved history -> {out_dir / 'history.json'}")
    print(f"Saved model   -> {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
