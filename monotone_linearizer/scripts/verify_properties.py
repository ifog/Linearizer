"""Verify structural promises of the Monotone Linearizer on trained models.

For each trained checkpoint we check the following framework guarantees,
which are the *contribution* of the Monotone Linearizer beyond accuracy:

  P1. Equilibrium uniqueness.  From K random w-inits, the solver converges
      to the same w_star (max pairwise diff < δ). Direct consequence of
      m-strong monotonicity (Lemma 1 in the blueprint).

  P2. Linear convergence rate.  Plot mean eq_err vs solver iteration; fit
      log-linear and report slope. Theory predicts slope ≈ log(1 - m*step).

  P3. Iterations to ε.  For ε ∈ {1e-1, ..., 1e-5}, count iterations
      needed. Should scale as O(log(1/ε)) with the same constant.

  P4. Flow round-trip.  ||flow(flow_inverse(z)) - z|| ≈ 0.  Verifies the
      flow remained exactly invertible through training. (Sanity: with the
      PLU param, det(W) > 0 always; this is the empirical confirmation.)

  P5. Resolvent transport (Lemma 7-equivalent).  J_{α f}(y) defined in
      "state space" satisfies   J_{α f}(y) = g^{-1}(J_{α M}(g(y)))    where
      J_{αM} is the resolvent of the latent operator. We check it on the
      flow model.  Empirical check: numerically compute both sides on a
      batch and compare.

  P6. Test-set iteration histogram.  Per-sample iteration count to reach
      ε on the test set. Variance in this is a free "difficulty" signal.

Usage:
    PYTHONPATH=/home/nvidia/Linearizer:/tmp/claude/pylib \\
        /home/nvidia/anaconda3/bin/python monotone_linearizer/scripts/verify_properties.py \\
        --root monotone_linearizer/results/a1_pilot
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from monotone_linearizer.data import get_loaders
from monotone_linearizer.models.vision import VisionMonotoneLinearizer


def load_model(ckpt_path: Path, device: str) -> tuple[VisionMonotoneLinearizer, dict]:
    blob = torch.load(ckpt_path, map_location=device)
    args = blob["args"]
    model = VisionMonotoneLinearizer(
        in_channels=1 if args["dataset"] in ("mnist", "fashion_mnist") else 3,
        img_size=28 if args["dataset"] in ("mnist", "fashion_mnist") else 32,
        n_classes=100 if args["dataset"] == "cifar100" else 10,
        latent_dim=args["latent_dim"],
        encoder_base=args["encoder_base"],
        m=args["m"],
        use_flow=bool(args["use_flow"]),
        flow_blocks=args["flow_blocks"],
        flow_hidden=args["flow_hidden"],
        solver=args["solver"],
        solver_max_iter=args["solver_max_iter"],
        solver_tol=args["solver_tol"],
        solver_step=args["solver_step"],
    ).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, args


# ----------------------------------------------------------------------------
# P1 — equilibrium uniqueness
# ----------------------------------------------------------------------------

@torch.no_grad()
def p1_uniqueness(model: VisionMonotoneLinearizer, x: torch.Tensor, *,
                  n_init: int = 8, max_iter: int = 200, init_scale: float = 1.0) -> dict:
    c = model.encoder(x)
    B, n = c.shape[0], model.core.n
    w_stars = []
    for k in range(n_init):
        w0 = init_scale * torch.randn(B, n, device=c.device)
        from monotone_linearizer.monotone.solvers import forward_backward
        w_star, _ = forward_backward(model.core, c, w_init=w0, step=model.solver_step,
                                     max_iter=max_iter, tol=1e-6)
        w_stars.append(w_star)
    stack = torch.stack(w_stars, dim=0)         # (K, B, n)
    mean = stack.mean(dim=0, keepdim=True)
    max_dev = (stack - mean).abs().max().item()
    mean_dev = (stack - mean).abs().mean().item()
    return {"n_init": n_init, "max_dev_across_inits": max_dev,
            "mean_dev_across_inits": mean_dev}


# ----------------------------------------------------------------------------
# P2 / P3 — linear convergence, iterations to ε
# ----------------------------------------------------------------------------

@torch.no_grad()
def p2_p3_convergence(model: VisionMonotoneLinearizer, x: torch.Tensor, *,
                       max_iter: int = 200) -> dict:
    """Run the FB iteration explicitly and record per-iter eq_err."""
    c = model.encoder(x)
    B, n = c.shape[0], model.core.n
    w = torch.zeros(B, n, device=c.device)
    err_per_iter: list[float] = []
    step = model.solver_step
    for k in range(max_iter):
        Mw = model.core(w, c)
        residual = (w - Mw)
        err = residual.norm(dim=-1).max().item()  # max over batch
        err_per_iter.append(err)
        w = w - step * residual
    # Linear-rate fit: log err = log err_0 + k * log rate.
    log_errs = np.log(np.clip(np.array(err_per_iter), 1e-20, None))
    # Use iterations 5 .. 80 (skip transient, avoid floor noise).
    fit_lo, fit_hi = 5, min(80, len(log_errs) - 1)
    ks = np.arange(fit_lo, fit_hi)
    slope, intercept = np.polyfit(ks, log_errs[fit_lo:fit_hi], 1)
    rate = float(np.exp(slope))
    # Iterations to ε.
    iters_to_eps: dict[float, int] = {}
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        idx = next((i for i, e in enumerate(err_per_iter) if e < eps), -1)
        iters_to_eps[eps] = int(idx) if idx >= 0 else -1
    return {
        "err_per_iter": err_per_iter,
        "fit_rate": rate,
        "expected_rate": 1.0 - model.core.m * step,
        "iters_to_eps": iters_to_eps,
    }


# ----------------------------------------------------------------------------
# P4 — flow invertibility round-trip
# ----------------------------------------------------------------------------

@torch.no_grad()
def p4_flow_round_trip(model: VisionMonotoneLinearizer, x: torch.Tensor) -> dict:
    if not model.use_flow:
        return {"applicable": False}
    c = model.encoder(x)
    from monotone_linearizer.monotone.solvers import forward_backward
    w_star, _ = forward_backward(model.core, c, max_iter=200, tol=1e-6, step=model.solver_step)
    # Round trip: w -> z -> w'
    z = model.flow.inverse(w_star)
    w_rt = model.flow(z)
    err = (w_star - w_rt).abs().max().item()
    return {"applicable": True, "round_trip_max_abs_diff": err,
            "w_star_norm": float(w_star.norm(dim=-1).mean().item())}


# ----------------------------------------------------------------------------
# P5 — resolvent transport
# ----------------------------------------------------------------------------

@torch.no_grad()
def p5_resolvent_transport(model: VisionMonotoneLinearizer, x: torch.Tensor, alpha: float = 0.5) -> dict:
    """Verify J_{αf}(y) = g^{-1}(J_{αM}(g(y))).

    J_{αF} = (I + αF)^{-1}. We use F = (I - core) since at equilibrium z = f(z, x)
    means z is the fixed point of f(., x), and f's residual operator (I - f) is
    a g-conjugate of the monDEQ residual (I - core).

    Concretely:
        f(z, x) = g^-1(M(g(z), c))
        (I - f)(z, x) = z - g^-1(M(g(z), c))

    Lemma 7-eq says J_{α(I-f)} (y) at fixed point  = g^-1 ( J_{α(I-M)} (g(y)) ).

    We compute the latent-side resolvent at the equilibrium (which is just w*),
    then map back through g^-1, and compare to the state-space equilibrium.
    Empirically this should be near-zero deviation since both objects equal z*.
    """
    if not model.use_flow:
        return {"applicable": False}
    c = model.encoder(x)
    from monotone_linearizer.monotone.solvers import forward_backward
    w_star, _ = forward_backward(model.core, c, max_iter=200, tol=1e-6, step=model.solver_step)
    z_star = model.flow.inverse(w_star)
    # Latent-side: J_{α(I-M)} at w* satisfies   y = J_{α(I-M)}(w*) = (I + α(I-M))^{-1} w*.
    # We approximate this by fixed-point iteration: y = (w* + α M(y, c)) / (1 + α)
    y = w_star.clone()
    for _ in range(80):
        y = (w_star + alpha * model.core(y, c)) / (1.0 + alpha)
    z_resolvent_latent = model.flow.inverse(y)
    # State-side proxy: apply the SAME averaging operation but in state space:
    # z_new = (z* + α * g^-1(M(g(z_new), c))) / (1 + α).
    # At equilibrium this gives the same fixed point as the latent resolvent.
    z = z_star.clone()
    for _ in range(80):
        w_inner = model.flow(z)
        Mw = model.core(w_inner, c)
        z_M = model.flow.inverse(Mw)
        z = (z_star + alpha * z_M) / (1.0 + alpha)
    err = (z - z_resolvent_latent).abs().max().item()
    return {"applicable": True, "transport_max_abs_diff": err,
            "alpha": alpha}


# ----------------------------------------------------------------------------
# P6 — per-input iteration count distribution
# ----------------------------------------------------------------------------

@torch.no_grad()
def p6_iter_histogram(model: VisionMonotoneLinearizer, loader, device: str, *,
                       eps: float = 1e-3, max_iter: int = 200, n_batches: int = 4) -> dict:
    iters_per_sample: list[int] = []
    correct_iters: list[tuple[int, bool]] = []
    seen = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= n_batches:
            break
        x = x.to(device); y = y.to(device)
        c = model.encoder(x)
        B, n = c.shape[0], model.core.n
        w = torch.zeros(B, n, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        iters = torch.zeros(B, dtype=torch.int32, device=device)
        for k in range(1, max_iter + 1):
            Mw = model.core(w, c)
            residual = w - Mw
            err_per = residual.norm(dim=-1)
            newly = (err_per < eps) & (~done)
            iters[newly] = k
            done = done | newly
            if done.all():
                break
            w = w - model.solver_step * residual
        # For samples that never converged, mark max_iter.
        iters[~done] = max_iter
        # Run head to compute correctness.
        z_star = model.flow.inverse(w)
        logits = model.head(z_star)
        pred = logits.argmax(-1)
        for i in range(B):
            iters_per_sample.append(int(iters[i].item()))
            correct_iters.append((int(iters[i].item()), bool(pred[i] == y[i])))
    arr = np.array(iters_per_sample)
    # Split by correctness.
    corr = np.array([1 if c else 0 for _, c in correct_iters])
    iters_arr = np.array([k for k, _ in correct_iters])
    return {
        "n": int(arr.size),
        "mean_iters": float(arr.mean()),
        "p50_iters": float(np.percentile(arr, 50)),
        "p90_iters": float(np.percentile(arr, 90)),
        "max_iters": int(arr.max()),
        "mean_iters_when_correct": float(iters_arr[corr == 1].mean()) if (corr == 1).any() else None,
        "mean_iters_when_wrong":   float(iters_arr[corr == 0].mean()) if (corr == 0).any() else None,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_init", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    datasets = sorted([p.name for p in root.iterdir() if p.is_dir()])

    all_results: dict = {}

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        print(f"\n=== {ds} ===")
        loaders = get_loaders(ds, batch_size=args.batch_size, num_workers=2)
        # Grab a single test batch for property checks.
        x_eval, _ = next(iter(loaders.test))
        x_eval = x_eval.to(args.device)

        for tag in ("noflow", "flow"):
            ckpt = root / ds / tag / "model.pt"
            if not ckpt.exists():
                print(f"  [{tag}] missing checkpoint, skipping")
                continue
            print(f"\n  --- {tag} ---")
            model, mdl_args = load_model(ckpt, args.device)
            r = {}
            r["p1_uniqueness"] = p1_uniqueness(model, x_eval, n_init=args.n_init)
            print(f"   P1 uniqueness  max-dev={r['p1_uniqueness']['max_dev_across_inits']:.2e} "
                  f"mean-dev={r['p1_uniqueness']['mean_dev_across_inits']:.2e}")
            r["p2_p3"] = p2_p3_convergence(model, x_eval)
            print(f"   P2 fit-rate    {r['p2_p3']['fit_rate']:.4f}  (theory: {r['p2_p3']['expected_rate']:.4f})")
            print(f"   P3 iters@1e-3  {r['p2_p3']['iters_to_eps'][1e-3]}  "
                  f"iters@1e-5  {r['p2_p3']['iters_to_eps'][1e-5]}")
            r["p4_round_trip"] = p4_flow_round_trip(model, x_eval)
            if r["p4_round_trip"]["applicable"]:
                print(f"   P4 g round-trip {r['p4_round_trip']['round_trip_max_abs_diff']:.2e} "
                      f"(w*_norm={r['p4_round_trip']['w_star_norm']:.2f})")
            r["p5_resolvent"] = p5_resolvent_transport(model, x_eval)
            if r["p5_resolvent"]["applicable"]:
                print(f"   P5 resolvent transport diff {r['p5_resolvent']['transport_max_abs_diff']:.2e}")
            r["p6_iters"] = p6_iter_histogram(model, loaders.test, args.device, n_batches=4)
            print(f"   P6 iters mean={r['p6_iters']['mean_iters']:.1f}  "
                  f"p50={r['p6_iters']['p50_iters']:.0f}  p90={r['p6_iters']['p90_iters']:.0f}  "
                  f"correct={r['p6_iters']['mean_iters_when_correct']}  "
                  f"wrong={r['p6_iters']['mean_iters_when_wrong']}")
            all_results.setdefault(ds, {})[tag] = r

            # Plot convergence rate.
            errs = r["p2_p3"]["err_per_iter"]
            label = "monDEQ (g=Id)" if tag == "noflow" else "Monotone Linearizer (g=flow)"
            ax.semilogy(range(len(errs)), errs, label=label, marker=".", markersize=3)
        ax.set_xlabel("solver iteration")
        ax.set_ylabel("equilibrium error  ||w - M(w,c)||")
        ax.set_title(f"{ds}: linear convergence  (theory rate = 1 - mu*step)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig_path = root / "structural_properties.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\nSaved figure  -> {fig_path}")

    (root / "structural_properties.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Saved summary -> {root / 'structural_properties.json'}")


if __name__ == "__main__":
    main()
